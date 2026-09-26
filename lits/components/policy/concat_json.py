"""Native-tool reasoning generation with independent reward evaluation."""

import json

from .concat import ConcatPolicy
from ..registry import register_policy
from ...lm.base import ToolCallOutput


@register_policy("concat_json", task_type="language_grounded")
class ConcatJSONPolicy(ConcatPolicy):
    """Generate one unscored reasoning step or answer per native tool request.

    Select with policy="concat_json" and an independent reward model, such as
    reward="generative". Inherited sampling makes N requests for N candidates;
    depth-limit handling, repeat checks, and canonical ThoughtStep conversion
    follow ConcatPolicy. Tool/schema failures receive one corrective retry.

    strict=True requests server-side schema enforcement, not a client guarantee.
    Required fields are text content and a boolean final-answer signal.
    Local validation still rejects missing calls, extra fields, invalid
    content, and incorrectly escaped mathematical text. No tool is executed.
    """

    def __init__(self, **kwargs):
        """Use tool output while retaining caller-supplied reasoning instructions."""
        kwargs["output_format"] = "structured"
        if kwargs.get("task_prompt_spec") is None:
            kwargs["task_prompt_spec"] = (
                "Return one candidate via submit_reasoning_step exactly once. "
                "Continue the supplied partial solution; do not repeat existing steps or siblings. "
                "Put an intermediate deduction in content with is_final=false when further "
                "reasoning is needed, or the established final answer to the original "
                "question with is_final=true. Once the question is solved, return the "
                "answer immediately; do not repeat the derivation or mark it as intermediate. "
                "Perform the reasoning; do not merely describe what to do next. "
                "Do not assign a score. "
                "Use plain-text math or correctly JSON-escaped backslashes."
            )
        super().__init__(**kwargs)

    @staticmethod
    def _action_tool() -> dict:
        """Describe one reasoning result and its explicit completion signal."""
        return {
            "name": "submit_reasoning_step",
            "description": "Submit one intermediate reasoning result or final answer.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "An actual deduction or established final answer.",
                    },
                    "is_final": {
                        "type": "boolean",
                        "description": "True only when content answers the original question.",
                    },
                },
                "required": ["content", "is_final"],
                "additionalProperties": False,
            },
        }

    @classmethod
    def _read_action_call(cls, response: ToolCallOutput) -> dict[str, str]:
        """Validate one call and return ConcatPolicy's exclusive step/answer payload."""
        if not isinstance(response, ToolCallOutput) or len(response.tool_calls) != 1:
            raise ValueError("Expected exactly one submit_reasoning_step tool call")
        call = response.tool_calls[0]
        payload = call.input_args
        if call.name != "submit_reasoning_step" or not isinstance(payload, dict):
            raise ValueError("Expected submit_reasoning_step with object arguments")
        if set(payload) != {"content", "is_final"} or type(payload["is_final"]) is not bool:
            raise ValueError("Tool arguments require content and boolean is_final")
        content = {"answer" if payload["is_final"] else "step": payload["content"]}
        return cls._parse_structured_output(json.dumps(content))

    def _generate_single_output(self, prompt: str, temperature: float):
        """Request one native call; the inherited validation loop handles retries.

        Return serialized validated content and no embedding, matching the parent
        sampling interface. Serialization escapes text rather than repairing it.
        """
        response = self._call_model(
            prompt, temperature=temperature, tools=[self._action_tool()],
            parallel_tool_calls=False, enable_thinking=False,
        )
        return json.dumps(self._read_action_call(response)), None
