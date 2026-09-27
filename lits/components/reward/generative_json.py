"""Native-tool scoring with GenerativePRM's correctness gate and usefulness mean."""

import json
import logging
from pathlib import Path

from .generative import GenerativePRM
from ..registry import register_reward_model
from ...lm.base import ToolCallOutput
from ...prompts.reward.generative import task_prompt_spec_tool
from ...structures.base import Step

logger = logging.getLogger(__name__)


@register_reward_model("generative_json", task_type="language_grounded")
class GenerativeJSONPRM(GenerativePRM):
    """Score through native tool arguments, without parsing free-form numeric text.

    Select reward="generative_json" through the component factory. Inherited
    configuration controls n_for_correctness, n_for_usefulness and their thinking
    flags. Any validated correctness zero stops scoring and returns zero;
    otherwise the reward is the mean usefulness score, exactly as in GenerativePRM.

    Thinking requests add a reasoning string to the tool schema. Each invalid
    response gets one corrective retry; repeated invalid output raises ValueError,
    never a fabricated zero or logits fallback. Network errors propagate.
    strict=True and tool_choice="required" request server enforcement; local
    validation remains mandatory. No tool is executed. Optional save_dir retains
    validated sample scores and tool arguments in the existing JSONL layout.
    """

    def __init__(self, **kwargs):
        """Use shared scoring criteria with tool-specific output instructions."""
        if kwargs.get("task_prompt_spec") is None:
            kwargs["task_prompt_spec"] = task_prompt_spec_tool
        super().__init__(**kwargs)

    @staticmethod
    def _score_tool(score_type: str, thinking: bool) -> dict:
        """Describe a binary correctness or bounded usefulness score."""
        if score_type not in ("correctness", "usefulness"):
            raise ValueError(f"Unknown reward score type: {score_type}")
        score = {"type": "number", "minimum": 0, "maximum": 1}
        if score_type == "correctness":
            score["enum"] = [0, 1]
        properties = {"score": score}
        if thinking:
            properties = {"reasoning": {"type": "string"}, **properties}
        return {
            "name": "submit_reward", "description": f"Submit the {score_type} score.",
            "strict": True,
            "input_schema": {"type": "object", "properties": properties,
                             "required": list(properties), "additionalProperties": False},
        }

    @classmethod
    def _read_score_call(cls, response, score_type: str, thinking: bool) -> dict:
        """Validate one named call, exact fields and numeric domain; return arguments."""
        if not isinstance(response, ToolCallOutput) or len(response.tool_calls) != 1:
            raise ValueError("Expected exactly one submit_reward tool call")
        call = response.tool_calls[0]
        payload = call.input_args
        fields = {"score", "reasoning"} if thinking else {"score"}
        if call.name != "submit_reward" or not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError(f"submit_reward requires exactly {sorted(fields)}")
        if thinking and not isinstance(payload["reasoning"], str):
            raise ValueError("reasoning must be a string")
        if type(payload["score"]) not in (int, float):
            raise ValueError("score must be numeric, not a string or boolean")
        cls._parse_score(str(payload["score"]), score_type)
        return payload

    def _sample_score(self, message: str, score_type: str, thinking: bool) -> dict:
        """Request one validated sample, with one retry for malformed tool output."""
        tool = self._score_tool(score_type, thinking)
        for attempt in range(3):
            try:
                response = self._call_model_with_role(
                    message, f"evaluator_{score_type}", max_new_tokens=500,
                    enable_thinking=thinking, tools=[tool], tool_choice="required",
                    parallel_tool_calls=False,
                )
                return self._read_score_call(response, score_type, thinking)
            except ValueError as error:
                if attempt:
                    raise ValueError(f"Invalid {score_type} tool score after two attempts: {error}") from error
                logger.warning("Invalid %s tool score; retrying: %s", score_type, error)
                message += (f"\nPrevious response invalid: {error}. Call submit_reward once, "
                            "using exactly the supplied schema and a numeric score; no ordinary prose.")

    def _fast_reward(self, state, action_or_step, query, query_idx, from_phase="") -> float:
        """Apply the original sample counts, correctness short circuit and usefulness mean."""
        action = action_or_step.get_action() if isinstance(action_or_step, Step) else action_or_step
        message = self._generate_usr_msg(query, state, action)
        for score_type in ("correctness", "usefulness"):
            self.base_model.sys_prompt = getattr(self, f"{score_type}_instruction")
            scores = []
            for _ in range(getattr(self, f"n_for_{score_type}")):
                payload = self._sample_score(message, score_type, getattr(self, f"think_for_{score_type}"))
                score = float(payload["score"])
                if self.save_dir is not None:
                    record = {"query_idx": query_idx, "query": query,
                              "steps": [step.action for step in state] + [action],
                              "from_phase": from_phase, "score": score,
                              "reasoning": payload.get("reasoning", ""), "text": json.dumps(payload)}
                    with Path(getattr(self, f"file_path_{score_type}")).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                if score_type == "correctness" and score == 0:
                    return 0.0
                scores.append(score)
        return sum(scores) / len(scores)
