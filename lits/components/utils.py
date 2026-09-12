import re
import logging
from typing import Optional
from pydantic import BaseModel
from ..structures import StateT

logger = logging.getLogger(__name__)

def create_role(llm_role: str, query_idx: Optional[int] = None, from_phase: str = "") -> str:
    """
    Construct a role string for inference logging.
    
    The role string is used to track and categorize LLM calls in logs. It combines
    the component type (llm_role), example index (query_idx), and algorithm phase
    (from_phase) into a single identifier.
    
    Args:
        llm_role: Component identifier (e.g., "policy", "rm", "dynamics")
        query_idx: Index of the current example for tracking
        from_phase: Algorithm phase (e.g., "expand", "simulate", "continuation")
    
    Returns:
        Role string in format: "{llm_role}_{query_idx}_{from_phase}"
        Components are omitted if None or empty.
    
    Examples:
        create_role("policy", 3, "expand") -> "policy_3_expand"
        create_role("rm", 5, "") -> "rm_5"
        create_role("dynamics", None, "simulate") -> "dynamics_simulate"
    """
    # TODO: Remove VALID_LLM_ROLES validation once all subclasses use _call_model() helpers
    # which manage roles via _get_llm_role() in base classes (Policy, RewardModel, LlmTransition)
    VALID_LLM_ROLES = [
        "prm_env", "prm_tool", "prm_language",
        "evaluator_logits_ORM", "evaluator_logits",
        "evaluator_tooluse", "evaluator_correctness", "evaluator_usefulness",
        "dynamics", "dynamics_verify", "dynamics_critic",
        "policy", "policy_env_grounded",
        "bn_entropy_agg", "bn_entropy_remove", "bn_eval", "bn_entropy",
        "rm",
        "augmentor",
        "memory",
        None, ""
    ]
    VALID_PHASES = [
        'expand', 'continuation', 'simulate', 'sort',
        'expand_prm', 'continuation_prm', 'simulate_prm', 'sort_prm',
        '', None
    ]
    assert llm_role in VALID_LLM_ROLES, f"Invalid llm_role: {llm_role}"
    assert from_phase in VALID_PHASES, f"Invalid from_phase: {from_phase}"
    
    role = llm_role if llm_role else ""
    if query_idx is not None and query_idx != '':
        role += f"_{query_idx}"
    if from_phase:
        role += f"_{from_phase}"
    return role
    
def extract_existing_steps(state: StateT) -> list[str]:
    existing_steps = []
    for idx, thought in enumerate(state):
        assert isinstance(thought.get_action(), str)
        existing_steps.append(thought.get_action())
    return existing_steps

# ----------------- Verbalization ---------------- 
# --- Tool verbalization ---
def verb_tool(tool, include_schema: bool = True) -> str:
    """Generate a verbal description of a tool, including its schema if requested.

    Args:
        tool (BaseTool): The tool to describe.
        include_schema (bool, optional): Whether to include the tool's argument schema in the description. Defaults to False.

    Returns:
        str: A string describing the tool and its arguments (if included). 
        
        Example schema (`props`):
            {'placeName': {'description': 'Name and address of the place', 'title': 'Placename', 'type': 'string'}}
    
        Example output with schema:
            PlaceSearch: Get place ID for a given location name and address.
            Arguments:
            - placeName (string): Name and address of the place
    """
    base_info = f"{tool.name}: {tool.description}"

    if include_schema and hasattr(tool, "args_schema"):
        schema_model = tool.args_schema
        # Some tools populate args_schema with a BaseModel subclass while others
        # provide None. Guard against non-class values.
        if (
            isinstance(schema_model, type)
            and issubclass(schema_model, BaseModel)
            and schema_model is not BaseModel  # 避免直接是 BaseModel
        ):
            schema = schema_model.model_json_schema()
        elif isinstance(schema_model, BaseModel):
            schema = schema_model.model_json_schema()
        else:
            schema = None
        if schema:
            # Get property descriptions from JSON schema
            props = schema.get("properties", {})
            arg_lines = [
                f"  - {name} ({prop.get('type', 'object')}): {prop.get('description', 'No description provided')}"
                for name, prop in props.items()
            ]
            schema_text = "\n".join(arg_lines)
            return f"{base_info}\nArguments:\n{schema_text}"
    return base_info

def verb_tools(tools, include_schema: bool = True, join_str: str = "\n\n") -> str:
    """Generate verbal descriptions for a list of tools.
    """
    return join_str.join([verb_tool(tool, include_schema) for tool in tools])

# --- State verbalization ---
def verbalize_concat_state(question, state):
    """ The format of the prompt is:
        Problem: ...
        Existing Steps:
        Step 1: ...
        Step 2: ...
        ...
        Step n: ...
    """
    # Debug: Check if question is a string
    if not isinstance(question, str):
        logger.error(f"ERROR: question is not a string! Type: {type(question)}, Value: {question}")
        if isinstance(question, list):
            logger.error(f"question is a list with {len(question)} elements")
            logger.error(f"First element type: {type(question[0]) if question else 'empty list'}")
        raise TypeError(f"question must be a string, got {type(question)}: {question}")
    
    question = question + '?' if not question.endswith('?') else question
    verbalized_state = "Problem: " + question 

    existing_steps = extract_existing_steps(state)
    if len(existing_steps) > 0:
        verbalized_state += "\nExisting Steps:\n"
    else:
        verbalized_state += "\nExisting Steps: None\n"

    for idx, action in enumerate(existing_steps):
        verbalized_state += "Step " + str(idx + 1) + ": " + action + "\n"
    return verbalized_state
    
# ----------------- Answer Retrieval ----------------
def strip_num(num):
    """Aim to get a number parsable by Python's float()"""
    return num.strip("*").strip("`").strip("'").strip('"').strip()

def retrieve_answer_from_last_step(step) -> Optional[str]:
    """Extract numerical answer from a step.
    
    Parses the step output to find answers in the format "The answer is X".
    Works with any step type that has a get_answer() method.
    
    Args:
        step: Step object with get_answer() method, or string containing the answer
    
    Returns:
        Extracted numerical answer string, or empty string if not found
    """
    # extract the answer from the last step
    if isinstance(step, str):
        output = step
    elif hasattr(step, 'get_answer'):
        output = step.get_answer()
    else:
        raise TypeError(f"step should have get_answer() method or be str, got {type(step)}: {step}")
    
    pattern = r'.*[Tt]he answer is .*?([$.0-9,\-]+)(?:\..*)?'
    match = re.match(pattern, output, re.DOTALL)
    # `re.DOTALL` to make .* match across newline characters
    # .*?  
        # . = any character except newline
        # * = zero or more times
        # ? = non-greedy (matches as few characters as possible
    # [...] = character class (match any single character from this set)
        # \- = escaped hyphen (literal minus sign)
    # (?:...) = non-capturing group (groups for precedence but doesn't create a capture group)
        # \. = literal period (escaped because . normally means "any character")
        # .* = any characters, zero or more times
        # ? = makes the entire group optional (zero or one occurrence)
    if match is None:
        answer = ""
    else:
        answer = match[1].replace(',', '').replace('$', '').replace(' ', '').rstrip('.') # 
        if '=' in answer:
            answer = answer[answer.rindex('=') + 1:]
    return answer
