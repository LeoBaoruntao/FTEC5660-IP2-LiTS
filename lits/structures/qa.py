from typing import NamedTuple
from dataclasses import dataclass

from ..type_registry import register_type
from .base import ActionT, Step, State, TrajectoryState, StringAction


@register_type
@dataclass
class ThoughtStep(Step):
    """General reasoning step used by concatenation-style policies.

    ``policy_score`` is an optional correctness/progress estimate in [0, 1]
    supplied by the policy. It is checkpoint metadata, excluded from reasoning
    text and chat messages. Missing scores default to None for older checkpoints.
    """

    action: ActionT = ""
    policy_score: float | None = None

    def get_action(self) -> ActionT:
        return self.action

    def get_answer(self) -> str:
        return self.action

    def verb_step(self) -> str:
        """Return a string representation of the thought/action."""
        return f"Thought: {self.action}"

    def to_messages(self) -> list[dict]:
        """Convert the step into a chat message."""
        return [{"role": "assistant", "content": str(self.action)}]
    
    def to_dict(self) -> dict:
        """Serialize the action and, when present, policy score for checkpointing."""
        data = super().to_dict()
        data["action"] = self.action
        if self.policy_score is not None:
            data["policy_score"] = self.policy_score
        return data

    @classmethod
    def verbalize_state(cls, question: str, state: list["ThoughtStep"]) -> str:
        """Verbalize ThoughtStep state into a prompt string.
        
        Delegates to verbalize_concat_state for consistency with existing code.
        
        Args:
            question: The original question
            state: List of ThoughtStep objects
        
        Returns:
            Formatted string for LLM prompt
        """
        from lits.components.utils import verbalize_concat_state
        return verbalize_concat_state(question, state)


@register_type
@dataclass
class ContinuationStep(ThoughtStep):
    """A raw-completion chunk and its request metadata.

    action is the retained text appended to the reasoning prefix. raw_text
    preserves the full response before newline trimming; output_tokens counts
    all generated tokens, including discarded text and EOS. terminate marks
    natural completion, not a chunk-length cutoff. seed identifies this request.
    """

    raw_text: str = ""
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    seed: int = 0

    @classmethod
    def from_completion(cls, output, seed: int) -> "ContinuationStep":
        """Retain a length-limited chunk through its last noninitial newline.

        Without a newline, retain the whole chunk. Natural completions are not
        trimmed. No stop strings are used, so finish_reason='stop' means EOS.
        Unknown finish reasons fail rather than silently becoming continuations.
        """
        if output.finish_reason not in ("length", "stop"):
            raise ValueError(f"Unexpected completion finish reason: {output.finish_reason}")
        text = output.text
        if output.finish_reason == "length":
            last_newline = text.rfind("\n")
            if last_newline > 0:
                text = text[:last_newline + 1]
        return cls(action=text, raw_text=output.text,
                   terminate=output.finish_reason == "stop",
                   finish_reason=output.finish_reason, input_tokens=output.input_tokens,
                   output_tokens=output.output_tokens, seed=seed)

    def to_dict(self) -> dict:
        """Include chunk provenance in the existing typed-step serialization."""
        data = super().to_dict()
        for name in ("raw_text", "finish_reason", "input_tokens", "output_tokens", "seed"):
            data[name] = getattr(self, name)
        return data
