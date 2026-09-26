from ..base import Policy
from ..registry import register_policy
from json import JSONDecodeError
import logging
import math
import re
from ...lm.base import HfChatModel, ToolCallOutput
from ...structures import State, Action, ThoughtStep
from ..utils import verbalize_concat_state, extract_existing_steps
from ...log import log_event
from ...utils.formatted_text_parser import parse_json_string

logger = logging.getLogger(__name__)


def count_tokens(text: str, model) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The language model (used to access tokenizer if available)
    
    Returns:
        Number of tokens in the text
    
    Note:
        - For HF models: Uses the model's tokenizer
        - For other models (OpenAI, Bedrock, etc.): Uses tiktoken
        - Fallback: Character-based estimate (~4 chars per token)
    """
    # Try using model's tokenizer (HF models)
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        tokens = model.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return tokens.shape[1]
    
    # Use tiktoken for non-HF models
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/Claude compatible encoding
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to count tokens with tiktoken: {e}. Using character-based estimate.")
        return len(text) // 4  # Rough estimate: ~4 chars per token


@register_policy("concat", task_type="language_grounded")
class ConcatPolicy(Policy):
    """Policy that generates reasoning actions by concatenating new steps to the existing trace.
    
    This policy generates step-by-step reasoning by:
    1. Building a prompt with existing steps
    2. Generating new actions from the LLM
    3. Validating outputs (similarity, length, repetition)
    4. Retrying if validation fails
    
    Config Args (via --search-arg):
        n_actions: Number of actions to generate per expansion (default: 3)
        max_steps: Maximum reasoning depth (default: 10)
        max_length: Maximum sequence length for LLM (default: 32768)
        force_terminating_on_depth_limit: Force termination at max_steps (default: False)
        check_action_sim: Enable embedding-based action similarity checking (default: False)

    Config Args (via component_args):
        output_format: ``free_form`` or ``structured`` (default: ``free_form``)
    """
    
    # Interface category for this policy type
    TASK_TYPE: str = "language_grounded"
    
    # Constants
    SIMILARITY_THRESHOLD = 0.98
    MAX_TOKEN_LENGTH = 1000
    MAX_RETRY_REPEAT = 2
    STEP_PREFIX_PATTERN = r"Step \d+:"
    ANSWER_PREFIX_PATTERN = r"^(?:The answer is\s*)+"
    STRUCTURED_FIELDS = {"step", "answer"}
    STRUCTURED_PLACEHOLDERS = {
        "...",
        "<answer>",
        "<step>",
        "final answer",
        "one next reasoning step",
    }
    
    @classmethod
    def from_config(cls, base_model, search_args: dict, component_args: dict, **kwargs):
        """Create ConcatPolicy from configuration dicts.
        
        Args:
            base_model: LLM for action generation
            search_args: Search algorithm parameters:
                - n_actions: Number of actions to generate (default: 3)
                - max_steps: Maximum depth (default: 10)
                - force_terminating_on_depth_limit: Force termination at max_steps (default: False)
                - max_length: Maximum sequence length (default: 32768)
                - max_new_tokens: Maximum new tokens per generation (default: None, no limit)
                - check_action_sim: Enable action similarity checking (default: False)
            component_args: Component parameters:
                - output_format: ``free_form`` or ``structured`` (default: ``free_form``)
            **kwargs: Additional arguments (task_name, task_prompt_spec)
        
        Returns:
            ConcatPolicy instance
        """
        return cls(
            base_model=base_model,
            task_prompt_spec=kwargs.get('task_prompt_spec'),
            task_name=kwargs.get('task_name'),
            n_actions=search_args.get('n_actions', 3),
            temperature=0.7,
            force_terminating_on_depth_limit=search_args.get('force_terminating_on_depth_limit', False),
            max_steps=search_args.get('max_steps', 10),
            max_length=component_args.get('max_length', 32768),
            max_new_tokens=component_args.get('max_new_tokens'),
            check_action_sim=search_args.get('check_action_sim', False),
            output_format=component_args.get('output_format', 'free_form'),
        )
    
    def __init__(self, **kwargs):
        """Initialize ConcatPolicy with action similarity checking option."""
        self.check_action_sim = kwargs.pop('check_action_sim', False)
        self.output_format = kwargs.pop('output_format', 'free_form')
        if self.output_format not in {'free_form', 'structured'}:
            raise ValueError(
                "ConcatPolicy output_format must be 'free_form' or 'structured'"
            )

        if self.output_format == 'structured' and kwargs.get('task_prompt_spec') is None:
            from ...prompts.registry import PromptRegistry

            kwargs['task_prompt_spec'] = PromptRegistry.get(
                'policy',
                'concat_structured',
                task_name=kwargs.get('task_name'),
                task_type=self.TASK_TYPE,
            )
        
        logger.debug(f"ConcatPolicy.__init__: kwargs keys = {kwargs.keys()}")
        logger.debug(f"ConcatPolicy.__init__: task_prompt_spec = {kwargs.get('task_prompt_spec', 'NOT PROVIDED')}")
        
        super().__init__(**kwargs)
        
        logger.debug(f"ConcatPolicy.__init__: After super().__init__, task_prompt_spec type = {type(self.task_prompt_spec)}")
        self._validate_task_prompt_spec()
        
    def _build_system_prompt(self) -> str:
        return self.task_prompt_spec
        
    def _create_error_steps(self, n_actions: int, error_msg: str) -> list[ThoughtStep]:
        """Create ThoughtStep error steps for ConcatPolicy."""
        return [ThoughtStep(action="", error=error_msg) for _ in range(n_actions)]
    
    def _validate_task_prompt_spec(self):
        """Validate that task_prompt_spec is a string, not a tuple."""
        if isinstance(self.task_prompt_spec, tuple):
            logger.error(f"ERROR: task_prompt_spec is a tuple with {len(self.task_prompt_spec)} elements!")
            logger.error(f"Tuple contents: {self.task_prompt_spec}")
            raise TypeError(f"task_prompt_spec should be a string, not a tuple: {self.task_prompt_spec}")

    def _build_messages(self, query: str, state: State, at_depth_limit: bool = False) -> str:
        """
        Generate the user message for the LLM.
        
        Args:
            query: The question/problem to solve
            state: Current reasoning state with existing steps
            at_depth_limit: Whether we're at the maximum depth
        
        Returns:
            Formatted user message string
        """
        logger.debug(f"_build_messages called with query type: {type(query)}, query value: {query}")
        
        user_message = verbalize_concat_state(query, state)
        
        user_message += f"Step {len(state) + 1}: "
        
        if at_depth_limit:
            user_message += "This is the last step, and the answer to the question has to be reached. "
        
        return user_message

    def _check_similarity(self, embedding, existing_embeddings):
        """
        Check if an embedding is too similar to existing embeddings.
        
        Args:
            embedding: New embedding to check
            existing_embeddings: List of existing embeddings
        
        Returns:
            Tuple of (is_similar, similar_index, similarity_score)
        """
        for idx, existing_embedding in enumerate(existing_embeddings):
            similarity = (embedding * existing_embedding).sum(dim=-1)
            
            if similarity > self.SIMILARITY_THRESHOLD:
                return True, idx, similarity
        
        return False, -1, 0.0

    def _generate_single_output(self, prompt: str, temperature: float):
        """
        Generate a single output from the LLM.
        
        Uses the base class _call_model() helper which auto-constructs the role
        from stored context (_query_idx, _from_phase) set by get_actions().
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature
        
        Returns:
            Tuple of (output_text, embedding) if check_action_sim is True, else (output_text, None)
        """
        response = self._call_model(
            prompt,
            temperature=temperature,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            new_sent_stop=False,
            enable_thinking=False,
            return_embedding=self.check_action_sim
        )
        
        if self.check_action_sim:
            output_text, embedding = response
            return output_text.text.strip(), embedding
        else:
            return response.text.strip(), None

    def _clean_output_text(self, output_text: str) -> str:
        """
        Clean the output text by removing common prefixes.
        
        Args:
            output_text: Raw output from LLM
        
        Returns:
            Cleaned output text
        """
        # Remove "Next step: " prefix
        if output_text.startswith("Next step: "):
            output_text = output_text[11:]
        
        # Remove "Step #:" prefix
        match = re.match(self.STEP_PREFIX_PATTERN, output_text)
        if match:
            output_text = output_text[len(match[0]):].strip()
        
        return output_text

    def _is_valid_output(self, output_text: str, query: str, existing_steps: list[str]) -> bool:
        """
        Check if output is valid (not a repeat of existing steps or query).
        
        Args:
            output_text: Generated output to validate
            query: Original query
            existing_steps: List of existing step texts
        
        Returns:
            True if output is valid, False otherwise
        """
        return output_text not in existing_steps and query not in output_text

    @classmethod
    def _parse_structured_output(cls, output_text: str) -> dict[str, str]:
        """Validate one exclusive step/answer string without repairing mathematical text.

        Reject control characters from accidentally decoded LaTeX escapes, such
        as a single-backslash frac becoming a form-feed. Newlines remain allowed.
        """
        payload, feedback = parse_json_string(output_text)
        if feedback or not isinstance(payload, dict):
            raise ValueError("Structured output must be a JSON object")
        if len(payload) != 1:
            raise ValueError("Structured output must contain exactly one field")

        field, content = next(iter(payload.items()))
        if field not in cls.STRUCTURED_FIELDS:
            raise ValueError("Structured output field must be 'step' or 'answer'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Structured output content must be a non-empty string")
        if any(ord(char) < 32 and char != "\n" for char in content):
            raise ValueError("Structured output contains control characters; use plain-text math or escaped backslashes")

        content = content.strip()
        if content.lower() in cls.STRUCTURED_PLACEHOLDERS:
            raise ValueError("Structured output content must not be a placeholder")
        return {field: content}

    @classmethod
    def _structured_payload_to_step(cls, payload: dict[str, str]) -> ThoughtStep:
        """Convert a validated structured payload into a canonical thought step."""
        if "answer" in payload:
            answer = re.sub(
                cls.ANSWER_PREFIX_PATTERN,
                "",
                payload["answer"],
                flags=re.IGNORECASE,
            ).strip()
            if not answer:
                raise ValueError("Structured answer must contain content after its prefix")
            return ThoughtStep(
                action=f"The answer is {answer}",
                terminate=True,
            )
        return ThoughtStep(action=payload["step"], terminate=False)

    def _log_validation_failure(self, reason: str, output_text: str, temperature: float, prompt: str):
        """Log details when output validation fails."""
        log_event(logger, "VALIDATION", f"{reason}", level="debug")
        logger.debug(f"  Output (temperature={temperature}): {output_text}")
        logger.debug(f"  System prompt: {self.task_prompt_spec[:200]}...")
        logger.debug(f"  User prompt: {prompt[:200]}...")

    def _generate_action_with_retry(
        self,
        prompt: str,
        query: str,
        temperature: float,
        existing_steps: list[str],
        existing_embeddings: list
    ) -> tuple[ThoughtStep, any]:
        """
        Generate a single action with retry logic for validation failures.

        Invalid structured output gets one retry with explicit format feedback;
        a second invalid response propagates rather than inventing a reasoning step.
        
        Uses stored context (_query_idx, _from_phase) from get_actions() for LLM calls.
        
        Args:
            prompt: The prompt to send to LLM
            query: Original query
            temperature: Sampling temperature
            existing_steps: List of existing step texts
            existing_embeddings: List of existing embeddings (if similarity checking enabled)
        
        Returns:
            Tuple of (thought_step, embedding)
        """
        n_retry_repeat = 0
        n_retry_structured = 0
        
        while True:
            # Generate output (uses _call_model which auto-constructs role)
            output_text, embedding = self._generate_single_output(prompt, temperature)
            structured_payload = None

            if self.output_format == 'structured':
                try:
                    structured_payload = self._parse_structured_output(output_text)
                except ValueError as error:
                    self._log_validation_failure(
                        str(error), output_text, temperature, prompt
                    )
                    n_retry_structured += 1
                    if n_retry_structured > 1:
                        raise
                    from ...prompts.policy.concat import structured_reasoning_json_instruction
                    prompt += (
                        f"\nYour previous response was invalid: {error}. "
                        'Return exactly one JSON object with one nonempty string field, '
                        '"step" or "answer". ' + structured_reasoning_json_instruction
                    )
                    continue

            candidate_text = (
                next(iter(structured_payload.values()))
                if structured_payload is not None
                else output_text
            )
            
            # Check similarity if enabled
            if self.check_action_sim:
                assert len(existing_embeddings) == len(existing_steps), \
                    "embeddings and outputs should have the same length"
                
                is_similar, similar_idx, similarity = self._check_similarity(embedding, existing_embeddings)
                if is_similar:
                    log_event(logger, "SIMILARITY", f"Found similar embedding (sim={similarity:.3f})", level="debug")
                    logger.debug(f"  Existing text: {existing_steps[similar_idx]}")
                    logger.debug(f"  New text: {output_text}")
                    continue
            
            # Check token length
            num_tokens = count_tokens(candidate_text, self.base_model)
            if num_tokens > self.MAX_TOKEN_LENGTH:
                self._log_validation_failure("Output is larger than 1000 tokens", output_text, temperature, prompt)
                continue
            
            if structured_payload is None:
                candidate_text = self._clean_output_text(candidate_text)
            
            # Check for repetition
            if self._is_valid_output(candidate_text, query, existing_steps):
                if structured_payload is not None:
                    return self._structured_payload_to_step(structured_payload), embedding
                return ThoughtStep(action=candidate_text), embedding
            else:
                self._log_validation_failure("Output is in existing steps", output_text, temperature, prompt)
                n_retry_repeat += 1
                if n_retry_repeat > self.MAX_RETRY_REPEAT:
                    return ThoughtStep(action="ALWAY REPEAT. TERMINATE"), embedding

    def _get_actions(
        self,
        query: str,
        state: State,
        n_actions: int,
        temperature: float,
        at_depth_limit: bool,
        query_idx: int,
        from_phase: str = "",
        existing_siblings: list = None,
        **kwargs
    ) -> list[ThoughtStep]:
        """
        Generate multiple reasoning actions for the current state.
        
        Args:
            query: The question/problem to solve
            state: Current reasoning state
            n_actions: Number of actions to generate
            temperature: Sampling temperature
            at_depth_limit: Whether at maximum depth
            query_idx: Query index for logging
            from_phase: Phase identifier for logging
            existing_siblings: Step objects already chosen by other candidates.
                When provided, appends a diversity prompt to avoid repeats.
        
        Returns:
            List of ThoughtStep objects containing generated actions
        """
        assert isinstance(query_idx, int), f"query_idx should be an integer, got {query}"
        
        # Build prompt
        user_message = self._build_messages(query, state, at_depth_limit=at_depth_limit)

        # Sibling-aware diversity prompt
        if existing_siblings:
            siblings_str = "\n".join(f"- {s.verb_step()}" for s in existing_siblings)
            user_message += (
                "\n\nThe following actions have already been chosen by other candidates. "
                "Choose a DIFFERENT action:\n" + siblings_str
            )

        if isinstance(self.base_model, HfChatModel):
            prompt = user_message
        else:
            prompt = self.task_prompt_spec + user_message
        
        # Generate actions
        outputs = []
        embeddings = []
        existing_steps = extract_existing_steps(state)
        
        for _ in range(n_actions):
            thought_step, embedding = self._generate_action_with_retry(
                prompt=prompt,
                query=query,
                temperature=temperature,
                existing_steps=existing_steps,
                existing_embeddings=embeddings
            )
            outputs.append(thought_step)
            
            if self.check_action_sim and embedding is not None:
                embeddings.append(embedding)
        
        return outputs


@register_policy("joint_action", task_type="language_grounded")
class JointActionPolicy(ConcatPolicy):
    """Generate and score alternative thought steps in one native tool call.

    Requires a model supporting tools and returning ToolCallOutput. One invalid
    response is retried; a second fails without falling back to free-text parsing.
    Completion formatting and depth-limit handling follow ConcatPolicy.
    Select with ``--policy joint_action``; inherited from_config() supplies the
    search and generation settings, while this class supplies the tool prompt.
    """

    def __init__(self, **kwargs):
        """Initialize with the joint-generation prompt and policy model options."""
        from ...prompts.policy.concat import task_prompt_spec_joint

        if kwargs.get("task_prompt_spec") is None:
            kwargs["task_prompt_spec"] = task_prompt_spec_joint
        super().__init__(**kwargs)

    @staticmethod
    def _action_tool(n_actions: int) -> dict:
        """Return a native tool schema for exactly n_actions scored alternatives."""
        variants = [
            {
                "type": "object",
                "properties": {
                    field: {"type": "string", "minLength": 1, "description": description},
                    "reward": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [field, "reward"],
                "additionalProperties": False,
            }
            for field, description in (
                ("step", "An intermediate deduction with its result when further reasoning "
                 "is needed. Never put the final answer here."),
                ("answer", "The established final answer to the original question. "
                 "Use immediately once solved, without repeating the derivation."),
            )
        ]
        return {
            "name": "submit_reasoning_steps",
            "description": (
                "Submit scored alternative next reasoning steps, each containing the actual "
                "deduction or calculation and its result, or an established final answer."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {"anyOf": variants},
                        "minItems": n_actions,
                        "maxItems": n_actions,
                    },
                },
                "required": ["candidates"],
                "additionalProperties": False,
            },
        }

    @classmethod
    def _read_action_call(cls, response: ToolCallOutput, n_actions: int) -> list[ThoughtStep]:
        """Validate one submit_reasoning_steps call and return scored, canonical steps.

        Reject malformed schemas, counts, scores, and placeholders with ValueError.
        Boolean scores are not numeric scores. Collapse identical canonical text
        with the same termination flag, preserving the first occurrence and score.
        The result can therefore contain fewer than n_actions distinct steps.
        """
        if not isinstance(response, ToolCallOutput) or len(response.tool_calls) != 1:
            raise ValueError("Expected exactly one submit_reasoning_steps tool call")
        call = response.tool_calls[0]
        payload = call.input_args
        if call.name != "submit_reasoning_steps" or not isinstance(payload, dict):
            raise ValueError("Expected submit_reasoning_steps with object arguments")
        if set(payload) != {"candidates"}:
            raise ValueError("Tool arguments must contain only candidates")
        items = payload["candidates"]
        if not isinstance(items, list) or len(items) != n_actions:
            raise ValueError(f"Expected exactly {n_actions} candidates")
        steps = []
        seen = set()
        for item in items:
            if not isinstance(item, dict) or set(item) not in (
                {"step", "reward"}, {"answer", "reward"}
            ):
                raise ValueError("Each candidate requires reward and exactly one step/answer")
            field = "step" if "step" in item else "answer"
            content = item[field]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Candidate content must be a nonempty string")
            content = content.strip()
            if content.lower() in cls.STRUCTURED_PLACEHOLDERS:
                raise ValueError("Candidate content must not be a placeholder")
            score = item["reward"]
            if (type(score) not in (int, float) or not 0 <= score <= 1
                    or not math.isfinite(score)):
                raise ValueError("Reward must be a finite number in [0, 1]")
            step = cls._structured_payload_to_step({field: content})
            key = (step.action, step.terminate)
            if key in seen:
                continue
            seen.add(key)
            step.policy_score = float(score)
            steps.append(step)
        return steps

    def _get_actions(
        self, query: str, state: State, n_actions: int, temperature: float,
        at_depth_limit: bool, query_idx: int, from_phase: str = "",
        existing_siblings: list = None, **kwargs,
    ) -> list[ThoughtStep]:
        """Request N alternatives (including rollout N=1); retry invalid output once.

        query/state provide the shared prefix, existing_siblings supply exclusions,
        and at_depth_limit requests completion. query_idx/from_phase are logging
        context supplied by get_actions(). Malformed tool-argument JSON and invalid
        candidates share one retry. Other model/API exceptions propagate unchanged.
        """
        if type(n_actions) is not int or n_actions < 1:
            raise ValueError("n_actions must be a positive integer")
        prompt = self._build_messages(query, state, at_depth_limit=at_depth_limit)
        prompt += (
            f"\nSubmit exactly {n_actions} candidates in one tool call. "
            "Use answer for a final answer whenever the question is solved; "
            "use step only when further reasoning is needed."
        )
        if existing_siblings:
            prompt += "\nAlready chosen siblings; provide different reasoning steps:\n" + "\n".join(
                step.verb_step() for step in existing_siblings
            )
        prompt = self.task_prompt_spec + "\n\n" + prompt
        for attempt in range(2):
            try:
                response = self._call_model(
                    prompt, temperature=temperature, tools=[self._action_tool(n_actions)],
                    parallel_tool_calls=False,
                )
            except JSONDecodeError as error:
                if attempt == 1:
                    raise
                error_message = f"Invalid tool-argument JSON: {error}"
            else:
                try:
                    return self._read_action_call(response, n_actions)
                except ValueError as error:
                    if attempt == 1:
                        raise
                    error_message = str(error)
            logger.warning("Invalid joint action tool call: %s; retrying once", error_message)
            prompt += (
                f"\nThe previous response was invalid: {error_message}. "
                f"Submit exactly {n_actions} concise candidates in one complete "
                "submit_reasoning_steps tool call. Use answer when solved; otherwise use step. "
                "Do not repeat text or emit multiple tool calls."
            )
