"""
lits-eval: Evaluate tree search results from checkpoint files.

Usage:
    lits-eval --result_dir results/math500_rest/run_0.2.10 --dataset_name math500 \
        --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
    lits-eval --result_dir results/blocksworld_rap/run_0.2.10 --dataset_name blocksworld \
        --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
    lits-eval --help

Auto-loads import_modules and dataset_kwargs from config.json in result_dir.
See docs/cli/search.md for full CLI documentation.
"""

import sys
import os
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Callable

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from lits.agents.tree.node import SearchNode, MCTSNode
from lits.benchmarks.registry import load_dataset, has_resource, has_evaluator, get_evaluator
from lits.components.registry import ComponentRegistry
from lits.registry import import_custom_modules, load_config_from_result_dir
from lits.lm import get_lm
from lits.eval.aggregation import aggregate_results
from lits.eval.inference_report import generate_report
from lits.eval.search import (
    check_answer,
    evaluate_search_result,
)
from lits.log import setup_logging
from lits.structures import ToolUseState, ToolUseStep  # Import to register types
from lits.structures.env_grounded import EnvState, EnvStep  # Import to register types

try:
    import botocore.exceptions
except ImportError:
    botocore = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def load_terminal_nodes_from_file(filepath: Path) -> Dict[str, Any]:
    """
    Load terminal nodes from a checkpoint file.
    
    Args:
        filepath: Path to terminal_nodes_{query_idx}.json file
    
    Returns:
        Dictionary containing terminal_nodes, query, and query_idx
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle empty terminal_nodes list
    if not data['terminal_nodes']:
        return {
            'terminal_nodes': [],
            'query': data['query'],
            'query_idx': data['query_idx']
        }
    
    # Deserialize nodes using from_dict
    # Determine node type from the data
    if 'cum_rewards' in data['terminal_nodes'][0]:
        node_class = MCTSNode
    else:
        node_class = SearchNode
    
    terminal_nodes = [node_class.from_dict(node_dict) for node_dict in data['terminal_nodes']]
    
    return {
        'terminal_nodes': terminal_nodes,
        'query': data['query'],
        'query_idx': data['query_idx']
    }


def get_goal_check_from_registry(dataset_name: str) -> Callable:
    """Get goal_check function from registry for env_grounded tasks.
    
    Args:
        dataset_name: Name of the registered benchmark
        
    Returns:
        The goal_check static method from the registered Transition class
        
    Raises:
        KeyError: If the benchmark is not registered
    """
    try:
        TransitionCls = ComponentRegistry.get_transition(dataset_name)
    except KeyError as e:
        available = ComponentRegistry.list_by_task_type("env_grounded")
        raise KeyError(
            f"Benchmark '{dataset_name}' not found in registry. "
            f"Available env_grounded benchmarks: {available}. "
            f"Did you forget to import the module containing @register_transition('{dataset_name}')?"
        ) from e
    
    if not hasattr(TransitionCls, 'goal_check'):
        raise AttributeError(
            f"Transition class '{TransitionCls.__name__}' for benchmark '{dataset_name}' "
            f"does not have a 'goal_check' static method."
        )
    
    return TransitionCls.goal_check


def is_env_grounded_task(dataset_name: str) -> bool:
    """Check if dataset is an env_grounded task using registry."""
    env_grounded_benchmarks = ComponentRegistry.list_by_task_type("env_grounded")
    return dataset_name.lower() in [b.lower() for b in env_grounded_benchmarks]


def _discover_checkpoint_files(checkpoints_dir: Path) -> tuple:
    """Scan a checkpoints directory and classify its contents.

    Returns:
        ``(files, is_pass_at_n, n_attempts)`` where *files* is a sorted
        list of ``Path`` objects (only recognized checkpoint patterns),
        *is_pass_at_n* is ``True`` when ``{idx}_a{attempt}.json`` files
        are present, and *n_attempts* is the max attempt count (1 for
        single-attempt runs).
    """
    all_json = sorted(checkpoints_dir.glob("*.json"), key=lambda f: f.stem)
    files = []
    max_attempt = -1
    for f in all_json:
        try:
            _idx, attempt = _parse_checkpoint_filename(f)
        except ValueError:
            continue  # e.g. "pass_at_n_summary.json", "3_a2_incomplete.json"
        files.append(f)
        if attempt is not None and attempt > max_attempt:
            max_attempt = attempt

    is_pass_at_n = max_attempt >= 0
    n_attempts = (max_attempt + 1) if is_pass_at_n else 1  # attempts are 0-indexed
    return files, is_pass_at_n, n_attempts


def _parse_checkpoint_filename(filepath: Path) -> tuple:
    """Extract ``(query_idx, attempt)`` from a checkpoint filename.

    Returns:
        ``(query_idx: int, attempt: int | None)``.

    Raises:
        ``ValueError`` if the filename doesn't match any known pattern.

    Supported patterns::

        "3_a2.json"             → (3, 2)   pass@N checkpoint
        "3.json"                → (3, None) single-attempt chain checkpoint
        "terminal_nodes_3.json" → (3, None) tree search result
    """
    import re
    filename = filepath.stem
    # "3_a2" → pass@N checkpoint, example idx = 3, attempt = 2
    m = re.match(r'^(\d+)_a(\d+)$', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    # "3" → single-attempt chain checkpoint, example idx = 3
    if re.match(r'^\d+$', filename):
        return int(filename), None
    # "terminal_nodes_3" → tree search result, example idx = 3
    m = re.match(r'^terminal_nodes_(\d+)$', filename)
    if m:
        return int(m.group(1)), None
    raise ValueError(f"Unrecognized checkpoint filename: {filepath.name}")


def _should_include_file(filepath: Path, offset: int, end_idx: int = None) -> bool:
    """Check if a checkpoint file's example index falls within ``[offset, end_idx)``.

    Only includes files matching known checkpoint patterns (see
    ``_parse_checkpoint_filename``). Files with extra suffixes
    (e.g. ``1_a2_incomplete.json``) are excluded.
    """
    try:
        idx, _attempt = _parse_checkpoint_filename(filepath)
    except ValueError:
        return False
    if idx < offset:
        return False
    if end_idx is not None and idx >= end_idx:
        return False
    return True


def _create_llm_evaluator(base_model, llm_eval_mode: str, eval_logger):
    """Create an LLM-based evaluator if mode is not ``"none"``.

    Returns:
        A ``GeneralEvaluator`` instance, or *None*.
    """
    if llm_eval_mode == "none":
        return None
    from lits.eval.general_eval import GeneralEvaluator
    if llm_eval_mode == "f1":
        evaluator = GeneralEvaluator(
            base_model=base_model,
            eval_perspectives=[{
                "eval_id": "score",
                "output_type": "float",
                "description": (
                    "Compute the F1 score between the predicted and ground truth answer sets. "
                    "F1 = 2 * precision * recall / (precision + recall), where "
                    "precision = (correct predictions) / (total predictions), "
                    "recall = (correct predictions) / (total ground truth elements). "
                    "Output 0.0 if no overlap, 1.0 if exact match. "
                    "Penalize both missing elements (low recall) and extra wrong elements (low precision)."
                ),
            }],
        )
        eval_logger.info("LLM-based F1 score evaluator enabled")
    else:
        evaluator = GeneralEvaluator(
            base_model=base_model,
            eval_perspectives=[{
                "eval_id": "correct",
                "description": (
                    "Does the predicted answer contain the correct value? "
                    "Ignore formatting differences, extra explanation, or markdown. "
                    "Focus only on whether the core answer value matches."
                ),
                "options": ["yes", "no"],
            }],
        )
        eval_logger.info("LLM-based binary evaluator enabled")
    return evaluator


def _save_eval_results(results: dict, result_dir: "Path", eval_logger):
    """Write ``eval_results.json`` to *result_dir*.

    Both the single-attempt and pass@N evaluation paths call this so
    there is exactly one save location for every eval run.
    """
    path = result_dir / "eval_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    eval_logger.info(f"Evaluation results saved to {path}")


def evaluate_from_checkpoints(
    result_dir: str,
    dataset_name: str,
    eval_model_name: str,
    offset: int = 0,
    limit: int = None,
    dataset_kwargs: dict = None,
    input_price_per_m: float = None,
    output_price_per_m: float = None,
    verbose: bool = False,
    llm_eval_mode: str = "binary",
    terminal_selection: str = "q-mean",
    enable_think_eval: bool = False,
):
    """
    Evaluate tree search results from checkpoint files.

    Loaded tree terminals are evaluated by lits.eval.search.evaluate_search_result.
    File loading, evaluator setup, chain evaluation, and reporting remain here.
    
    Args:
        result_dir: Directory containing terminal_nodes_*.json files
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'math500', 'blocksworld')
        eval_model_name: Tool-capable model required for language-tree answer voting;
            also reused for optional correctness judging. No voting occurs for
            tool-use/environment tasks or chain/pass-at-N checkpoints.
        offset: Dataset offset used during search
        limit: Dataset limit used during search
        dataset_kwargs: Dataset-specific arguments (loaded from config)
        verbose: If True, print detailed output to console
        llm_eval_mode: Optional correctness judge mode. "none" disables judging,
            not mandatory language-tree voting.
        enable_think_eval: Model-construction thinking setting for the shared voting
            and judging model. Defaults to False; backend support determines its effect.
        terminal_selection: How to pick which terminal node's answer to evaluate
            for tool_use tasks. One of:
              - "first":   terminal_nodes[0] (legacy save-order behaviour;
                           reproduces pre-fix evals)
              - "q-max":   argmax over max(cum_rewards) — best single rollout Q
              - "q-mean":  argmax over mean(cum_rewards)  (default)
                           matches MCTSNode.DEFAULT_Q_FUNC = np.mean,
                           i.e. the same Q-value MCTS uses for UCT
            Missing/empty returns fall back per node to fast_reward, including
            environment maximum-return ranking.
            See `paper/lits_memory/results/summary_wikisql.md` for empirical
            comparison of selection strategies on WikiSQL MCTS.
    """
    if not isinstance(enable_think_eval, bool):
        raise ValueError("enable_think_eval must be a boolean")
    result_dir = Path(result_dir)
    
    # Setup logging - only log to file, not console (unless verbose)
    eval_logger = setup_logging(
        run_id="eval",
        result_dir=result_dir,
        add_console_handler=verbose,
        verbose=True
    )
    
    from lits.cli import log_command
    log_command(eval_logger)
    
    # Determine task type using registry
    is_env_grounded = is_env_grounded_task(dataset_name)
    is_tool_use = has_resource(dataset_name)
    
    # Load dataset for ground truths (not needed for env_grounded tasks)
    goal_check = None
    if is_env_grounded:
        task_type = "env_grounded"
        # For env_grounded tasks, get goal_check from registry
        goal_check = get_goal_check_from_registry(dataset_name)
        full_dataset = None
        eval_logger.info(f"Evaluating {dataset_name} results (goal checking via registry)")
    elif is_tool_use:
        task_type = "tool_use"
        # Tool-use datasets are registered via @register_dataset, same as other task types
        full_dataset = load_dataset(dataset_name, **(dataset_kwargs or {}))
        # NOTE: Do NOT slice the dataset here. The query_idx in terminal node files
        # refers to the original dataset index, not the sliced index.
    else:
        task_type = "language_grounded"
        dataset_kwargs = dataset_kwargs or {}
        full_dataset = load_dataset(dataset_name, **dataset_kwargs)
        # NOTE: Do NOT slice the dataset here. The query_idx in terminal node files
        # refers to the index within the (possibly filtered) dataset used during search.
        # We need the same filtered dataset to look up ground truths by query_idx.
    
    # Find result files: terminal_nodes/ (tree search) or checkpoints/ (chain)
    terminal_nodes_dir = result_dir / "terminal_nodes"
    checkpoints_dir = result_dir / "checkpoints"
    use_chain_checkpoints = False
    is_pass_at_n = False

    if terminal_nodes_dir.exists():
        terminal_node_files = sorted(terminal_nodes_dir.glob("terminal_nodes_*.json"),
                                     key=lambda f: int(f.stem.split('_')[-1]))
        if not terminal_node_files:
            eval_logger.error(f"No terminal node files found in {terminal_nodes_dir}")
            return
        eval_logger.info(f"Found {len(terminal_node_files)} terminal node files")
    elif checkpoints_dir.exists():
        use_chain_checkpoints = True
        terminal_node_files, is_pass_at_n, n_attempts = _discover_checkpoint_files(checkpoints_dir)
        if not terminal_node_files:
            eval_logger.error(f"No checkpoint files found in {checkpoints_dir}")
            return
        if is_pass_at_n:
            eval_logger.info(f"Found {len(terminal_node_files)} pass@N checkpoint files (up to {n_attempts} attempts)")
        else:
            eval_logger.info(f"Found {len(terminal_node_files)} chain checkpoint files")
    else:
        eval_logger.error(f"Neither terminal_nodes/ nor checkpoints/ found in {result_dir}")
        return
    
    # Filter terminal node files by offset and limit based on query_idx
    # The query_idx in terminal node files corresponds to the original dataset index
    end_idx = offset + limit if limit is not None else None
    
    filtered_files = [f for f in terminal_node_files if _should_include_file(f, offset, end_idx)]
    eval_logger.info(f"Processing {len(filtered_files)} files (offset={offset}, limit={limit})")

    # --- Setup evaluators (shared by both single-attempt and pass@N) ---
    custom_evaluator = get_evaluator(dataset_name) if has_evaluator(dataset_name) else None
    if custom_evaluator:
        eval_logger.info(f"Using registered evaluator for '{dataset_name}'")

    # Tree language tasks vote; chain checkpoints already store the chosen answer.
    needs_voting = task_type == "language_grounded" and not use_chain_checkpoints
    needs_judging = not is_env_grounded and llm_eval_mode != "none"
    base_model = (
        get_lm(eval_model_name, enable_thinking=enable_think_eval)
        if needs_voting or needs_judging else None
    )
    if base_model is not None:
        eval_logger.info("Evaluation model: %s; enable_think_eval=%s",
                         eval_model_name, enable_think_eval)
    voting_model = base_model if needs_voting else None
    llm_evaluator = (
        _create_llm_evaluator(base_model, llm_eval_mode, eval_logger) if needs_judging else None
    )

    # --- Evaluate every checkpoint file (single-attempt and pass@N alike) ---
    # Each file produces one (query_idx, correct, score) result.
    # For pass@N files ({idx}_a{attempt}.json), query_idx is parsed from the
    # filename prefix; for single files ({idx}.json), it's the whole stem.
    # Binary correctness stays separate from environment progress and evaluator scores.
    EvalPoint = dict  # idx, attempt, correct, score, environment_progress, evaluation_score
    eval_points: List[EvalPoint] = []

    eval_logger.info("=" * 40)
    eval_logger.info("Detailed comparison:")

    for filepath in tqdm(filtered_files, desc="Evaluating", unit="file"):
        try:
            # --- Parse query_idx and attempt from filename ---
            query_idx, attempt = _parse_checkpoint_filename(filepath)
            label = f"[{query_idx}_a{attempt}]" if attempt is not None else f"[{query_idx}]"

            # --- Extract and score the prediction ---
            environment_progress = None  # env_grounded word-level score, when available
            if use_chain_checkpoints:
                from lits.structures.tool_use import ToolUseState
                _query, state = ToolUseState.load(str(filepath))
                ground_truth = full_dataset[query_idx]['answer']
                answer_pred = state.get_final_answer() or ""
                if is_env_grounded:
                    correct = (answer_pred == ground_truth)
                    score = None
                else:
                    correct, score = check_answer(
                        answer_pred, ground_truth,
                        context=_query,
                        custom_evaluator=custom_evaluator,
                        llm_evaluator=llm_evaluator,
                        llm_eval_mode=llm_eval_mode,
                        eval_logger=eval_logger,
                        label=label,
                    )
            else:
                data = load_terminal_nodes_from_file(filepath)
                terminal_nodes = data['terminal_nodes']
                _query = data['query']
                ground_truth = "correct" if is_env_grounded else full_dataset[query_idx]['answer']
                if is_tool_use and not is_env_grounded and "mapeval" in dataset_name:
                    ground_truth = "Option " + str(ground_truth)

                # Tool-use terminal selection in the shared evaluator is
                # configurable via --terminal_selection (default q-mean).
                # See `select_tool_use_terminal` docstring and the
                # "Eval pipeline bug" section in
                # paper/lits_memory/results/summary_wikisql.md for context.
                outcome = evaluate_search_result(
                    terminal_nodes, _query, ground_truth,
                    task_type=task_type,
                    voting_model=voting_model,
                    terminal_selection=terminal_selection,
                    custom_evaluator=custom_evaluator,
                    llm_evaluator=llm_evaluator,
                    llm_eval_mode=llm_eval_mode,
                    goal_check=goal_check,
                    eval_logger=eval_logger,
                    label=label,
                )
                answer_pred = outcome["prediction"]
                correct = outcome["correct"]
                score = outcome["score"]
                if is_env_grounded:
                    environment_progress = outcome["environment_progress"]

            eval_logger.info(f"  {label} Pred='{str(answer_pred)[:80]}', Truth='{str(ground_truth)[:80]}', Correct={correct}")
            eval_points.append({
                "idx": query_idx, "attempt": attempt, "correct": correct,
                "score": 1.0 if correct else 0.0,
                "environment_progress": environment_progress,
                "evaluation_score": score,  # continuous score from float-returning evaluators
            })

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            # Auth errors are global — fail fast instead of silently skipping all files
            if botocore is not None and isinstance(e, (
                botocore.exceptions.SSOError,
                botocore.exceptions.UnauthorizedSSOTokenError,
                botocore.exceptions.TokenRetrievalError,
            )):
                raise
            eval_logger.error(f"Error processing {filepath}: {e}")
            eval_logger.error(traceback.format_exc())

    eval_logger.info("=" * 40)

    # --- Aggregate results ---
    summary = aggregate_results(
        eval_points, pass_at_n=is_pass_at_n,
        mean_fields=("environment_progress", "evaluation_score"),
    )
    if is_pass_at_n:
        # Group by example_idx: pass@1 uses the first available attempt; pass@N uses any.
        examples = summary["examples"]
        total = summary["total"]
        eval_results = {
            "n_attempts": n_attempts,
            "temperature": None,
            "pass_at_1": summary["pass_at_1"],
            "pass_at_n": summary["pass_at_n"],
            "examples": examples,
            "dataset_name": dataset_name,
            "result_dir": str(result_dir),
        }

        eval_logger.info("=" * 60)
        eval_logger.info(f"Pass@N Evaluation Results: {result_dir.name}")
        eval_logger.info(f"Dataset: {dataset_name}, Examples: {total}, Attempts: {n_attempts}")
        eval_logger.info(f"pass@1: {eval_results['pass_at_1']:.1%} ({summary["pass_at_1_count"]}/{total})")
        eval_logger.info(f"pass@{n_attempts}: {eval_results['pass_at_n']:.1%} ({summary["pass_at_n_count"]}/{total})")
        eval_logger.info("=" * 60)

        print()
        print("=" * 60)
        print(f"  Pass@N Evaluation: {result_dir.name}")
        print("=" * 60)
        print(f"  Dataset:    {dataset_name}")
        print(f"  Examples:   {total}")
        print(f"  Attempts:   {n_attempts}")
        print(f"  pass@1:     {eval_results['pass_at_1']:.1%} ({n_pass_1}/{total})")
        print(f"  pass@{n_attempts}:     {eval_results['pass_at_n']:.1%} ({n_pass_n}/{total})")
        print("=" * 60)

    else:
        # Single-attempt: flat accuracy
        correct_count = summary["correct_count"]
        total_count = summary["total"]
        accuracy = summary["accuracy"]

        soft_accuracy = summary["means"]["environment_progress"] if is_env_grounded else None
        mean_score = summary["means"]["evaluation_score"]

        eval_results = {
            "n_attempts": 1,
            "temperature": None,
            "accuracy": accuracy,
            "examples": summary["examples"],
            "dataset_name": dataset_name,
            "result_dir": str(result_dir),
        }
        if soft_accuracy is not None:
            eval_results["soft_accuracy"] = soft_accuracy
        if mean_score is not None:
            eval_results["mean_score"] = mean_score

        eval_logger.info("=" * 80)
        eval_logger.info(f"Evaluation Results for {result_dir.name}")
        eval_logger.info(f"Dataset: {dataset_name}, Examples: {total_count}, Correct: {correct_count}")
        eval_logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if mean_score is not None:
            eval_logger.info(f"Mean Score: {mean_score:.4f} ({mean_score*100:.2f}%)")
        if soft_accuracy is not None:
            eval_logger.info(f"Soft Accuracy: {soft_accuracy:.4f} ({soft_accuracy*100:.2f}%)")
        eval_logger.info("=" * 80)

        # Token usage report
        report_kwargs = {}
        if input_price_per_m is not None:
            report_kwargs["input_price_per_m"] = input_price_per_m
        if output_price_per_m is not None:
            report_kwargs["output_price_per_m"] = output_price_per_m
        report = generate_report(str(result_dir), **report_kwargs)
        eval_logger.info(report)

        print()
        print("=" * 60)
        print(f"  Evaluation Results: {result_dir.name}")
        print("=" * 60)
        print(f"  Dataset:    {dataset_name}")
        print(f"  Examples:   {total_count}")
        print(f"  Correct:    {correct_count}")
        print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        if mean_score is not None:
            print(f"  Mean Score: {mean_score:.4f} ({mean_score*100:.2f}%)")
        if soft_accuracy is not None:
            print(f"  Soft Acc:   {soft_accuracy:.4f} ({soft_accuracy*100:.2f}%)")
        print("=" * 60)
        print()
        print(report)

    _save_eval_results(eval_results, result_dir, eval_logger)
    return eval_results


def main() -> int:
    """Entry point for lits-eval command.
    
    Evaluates tree search results from checkpoint files. Auto-loads
    import_modules and dataset_kwargs from config.json in result_dir.
    --enable_think_eval/--no-enable_think_eval override saved
    search_args.enable_think_eval; absent settings default to False.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load .env — find_dotenv() searches upward from cwd
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Evaluate tree search results from checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate results (auto-loads dataset_name, eval_model_name, import_modules from config)
  lits-eval --result_dir demo_results
  
  # Explicit overrides
  lits-eval --result_dir claude35v1_results/blocksworld_rap/run_0.2.10 \\
      --dataset_name blocksworld --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
  
  # Evaluate custom benchmark (import module to register Transition)
  lits-eval --result_dir results/robot_arm_rap/run_0.2.10 \\
      --dataset_name robot_arm --eval_model_name bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 \\
      --include my_project.robot_arm
"""
    )
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing terminal_nodes/ subdirectory")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (e.g., gsm8k, math500, blocksworld). Auto-loaded from config.json if not specified.")
    parser.add_argument("--eval_model_name", type=str, default=None, help="Model for language-tree answer voting and optional correctness judging. Must support native tools for voting. Auto-loaded from config.json if not specified.")
    parser.add_argument("--enable_think_eval", action=argparse.BooleanOptionalAction,
                        default=None, help="Thinking for the voting/judging model. Overrides "
                        "saved search_args.enable_think_eval; otherwise defaults to False.")
    parser.add_argument("--offset", type=int, default=0, help="Dataset offset used during search")
    parser.add_argument("--limit", type=int, default=None, help="Dataset limit used during search")
    parser.add_argument(
        "--include",
        dest="import_modules",
        type=str,
        nargs="+",
        metavar="MODULE",
        help="Python module(s)/package(s) to include for custom component registration. Auto-loaded from config if not specified."
    )
    parser.add_argument("--input-price", type=float, default=None, help="Price per 1M input tokens (for cost estimation)")
    parser.add_argument("--output-price", type=float, default=None, help="Price per 1M output tokens (for cost estimation)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output to console (default: progress bar + summary only)")
    parser.add_argument("--llm-eval", choices=["binary", "f1", "none"], default="binary",
                        help="Correctness judge mode: 'binary' (default, yes/no), 'f1' (float F1 score), 'none' (disable judging, not language-tree voting)")
    parser.add_argument("--terminal_selection", choices=["first", "q-max", "q-mean"],
                        default="q-mean",
                        help=("How to pick which terminal node's answer to evaluate "
                              "for tool_use tasks. 'first' = legacy save-order behaviour "
                              "(terminal_nodes[0]); 'q-max' = argmax over max(cum_rewards); "
                              "'q-mean' (default) = argmax over mean(cum_rewards), matches "
                              "MCTSNode.DEFAULT_Q_FUNC. Missing returns use fast_reward. See "
                              "paper/lits_memory/results/summary_wikisql.md for the "
                              "empirical comparison on WikiSQL MCTS."))
    
    args = parser.parse_args()
    
    # Validate result_dir exists
    if not Path(args.result_dir).exists():
        print(f"Error: Directory not found: {args.result_dir}", file=sys.stderr)
        return 1
    
    # Load config from result_dir if available (for auto-loading import_modules)
    config = load_config_from_result_dir(args.result_dir, config_filename="config.json")
    
    # Determine import_modules: CLI args override config (top-level field)
    import_modules = args.import_modules
    if not import_modules and config.get("import_modules"):
        import_modules = config["import_modules"]
        print(f"Auto-loaded import_modules from config: {import_modules}")
    
    # Load dataset_kwargs from config (top-level field)
    dataset_kwargs = config.get("dataset_kwargs", {})
    if dataset_kwargs:
        print(f"Auto-loaded dataset_kwargs from config: {dataset_kwargs}")
    
    # Resolve dataset_name: CLI > config > error
    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = config.get("dataset") or config.get("benchmark")
        if dataset_name:
            print(f"Auto-loaded dataset_name from config: {dataset_name}")
        else:
            print("Error: --dataset_name not specified and not found in config.json", file=sys.stderr)
            return 1

    # Resolve eval_model_name: CLI > config (eval_model_name or policy_model_name) > error
    eval_model_name = args.eval_model_name
    if not eval_model_name:
        eval_model_name = config.get("eval_model_name") or config.get("policy_model_name")
        if eval_model_name:
            print(f"Auto-loaded eval_model_name from config: {eval_model_name}")
        else:
            print("Error: --eval_model_name not specified and not found in config.json", file=sys.stderr)
            return 1
    
    enable_think_eval = args.enable_think_eval
    if enable_think_eval is None:
        enable_think_eval = config.get("search_args", {}).get("enable_think_eval", False)

    # Import custom modules to trigger registration
    try:
        import_custom_modules(import_modules)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    try:
        evaluate_from_checkpoints(
            result_dir=args.result_dir,
            dataset_name=dataset_name,
            eval_model_name=eval_model_name,
            offset=args.offset,
            limit=args.limit,
            dataset_kwargs=dataset_kwargs,
            input_price_per_m=args.input_price,
            output_price_per_m=args.output_price,
            verbose=args.verbose,
            llm_eval_mode=args.llm_eval,
            terminal_selection=args.terminal_selection,
            enable_think_eval=enable_think_eval,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    
    eval_log = Path(args.result_dir) / "eval.log"
    if eval_log.exists():
        print(f"Evaluation log saved to: {eval_log}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
