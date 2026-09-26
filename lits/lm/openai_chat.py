import time, json, logging, warnings
from openai import OpenAI
from typing import List, Optional
import numpy as np
import os
from lits import observability as tracing
from .base import LanguageModel, Output, InferenceLogger, ToolCall, ToolCallOutput

logger = logging.getLogger(__name__)

class OpenAIChatModel(LanguageModel):
    """Wrapper for OpenAI-compatible chat models (e.g., gpt-4o, gpt-4-turbo)."""

    def __init__(
        self,
        model_name: str,
        sys_prompt: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        inference_logger: InferenceLogger = None,
        max_length: int = None,
        max_new_tokens: int = None,
        verbose: bool = False,
        enable_thinking: bool = False,
        **kwargs
    ):
        """Create a client; tracing must be configured first to capture its requests."""
        # Log unsupported kwargs at debug level (expected in multi-backend frameworks
        # where HF-specific params like device, top_k leak through the unified interface)
        unsupported_kwargs = set(kwargs.keys())
        if unsupported_kwargs:
            logger.debug(f"Ignoring kwargs not applicable to OpenAI chat models: {unsupported_kwargs}")
        super().__init__(
            model_name=model_name,
            model=None,
            tokenizer=None,
            inference_logger=inference_logger,
            enable_thinking=enable_thinking,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            verbose=verbose
        )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_API_BASE")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        tracing.instrument_openai(self.client)
        self.model_name = model_name
        self.sys_prompt = sys_prompt

    def _format_messages(self, prompt):
        """Convert prompt to OpenAI chat message format, including optional system prompt."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            if all(isinstance(p, dict) and "role" in p and "content" in p for p in prompt):
                messages = prompt
            elif all(isinstance(p, (list, tuple)) and len(p) == 2 for p in prompt):
                messages = [{"role": r, "content": c} for r, c in prompt]
            else:
                raise ValueError("Prompt must be a string or list of {'role','content'} or (role, content).")
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

        # prepend system prompt if not already there
        if self.sys_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.sys_prompt}] + messages
        return messages

    def __call__(
        self,
        prompt,
        role: str = "default",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        stop: Optional[str | List[str]] = None,
        return_embedding: bool = False,
        tools: Optional[List[dict]] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[str | dict] = None,
        new_line_stop: bool = False,
        **kwargs
    ):
        """Generate text or return native function arguments as ToolCallOutput.

        tools uses LiTS schemas: name, description, input_schema, and optional
        strict. Strict schema enforcement depends on the server/model. Without
        tools, the request and Output return type remain unchanged. Tool calls
        are returned to the caller, not executed here; both paths log usage.
        parallel_tool_calls=False requests at most one tool call per response.
        None preserves the server default; applied only when tools are supplied.
        tool_choice forwards auto/required/none or a named-function selection
        when tools are supplied. None preserves the server default.

        new_line_stop adds a newline to the server-side stop sequences. Existing
        stop strings/lists are preserved without mutating the caller's list;
        when False, stop is forwarded unchanged. The server must support stop.

        Enabled tracing nests the SDK request under llm.call with the LiTS role.
        request_extra_body retains custom options merged after SDK instrumentation.
        """
        if return_embedding:
            raise NotImplementedError("Embedding retrieval not implemented for OpenAI chat models.")
        # Log unsupported kwargs at debug level (expected in multi-backend frameworks
        # where HF-specific params like top_k, new_sent_stop, enable_thinking leak through)
        unsupported_kwargs = set(kwargs.keys()) - {"frequency_penalty", "presence_penalty", "n"}
        if unsupported_kwargs:
            logger.debug(f"Ignoring kwargs not applicable to OpenAI chat models: {unsupported_kwargs}")
        if new_line_stop:
            stop = [stop] if isinstance(stop, str) else list(stop or [])
            if "\n" not in stop:
                stop.append("\n")
        messages = self._format_messages(prompt)
        start = time.time()
        
        # Build extra_body for vLLM/Qwen3 thinking mode control
        # Only add for non-OpenAI endpoints (vLLM, etc.) to avoid API errors
        extra_body = None
        is_custom_endpoint = self.client.base_url and "api.openai.com" not in str(self.client.base_url)
        if is_custom_endpoint and not self.enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        tool_options = {}
        if tools:
            tool_options["tools"] = [
                {"type": "function", "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                    **({"strict": tool["strict"]} if "strict" in tool else {}),
                }}
                for tool in tools
            ]
            if parallel_tool_calls is not None:
                tool_options["parallel_tool_calls"] = parallel_tool_calls
            if tool_choice is not None:
                tool_options["tool_choice"] = tool_choice
        
        with tracing.span("llm.call", lits_role=role, request_extra_body=extra_body):
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens or self.max_new_tokens or 512,
                frequency_penalty=kwargs.get("frequency_penalty"),
                presence_penalty=kwargs.get("presence_penalty"),
                n=kwargs.get("n", 1),
                stop=stop,
                extra_body=extra_body,
                **tool_options,
            )
        end = time.time()

        message = resp.choices[0].message
        text = (message.content or "").strip()
        input_toks = getattr(resp.usage, "prompt_tokens", 0)
        output_toks = getattr(resp.usage, "completion_tokens", 0)

        if self.inference_logger and role is not None:
            self.inference_logger.update_usage(
                input_tokens=input_toks,
                output_tokens=output_toks,
                batch=False,
                batch_size=1,
                role=role,
                running_time=end - start
            )

        if self.verbose and self.LOG_MODEL_OUTPUT:
            print(f"[{self.model_name}] → {text[:300]}")

        if message.tool_calls:
            calls = []
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool-call arguments must be a JSON object")
                calls.append(ToolCall(call.id, call.function.name, arguments))
            return ToolCallOutput(
                text, calls, stop_reason="tool_use", raw_message=message.model_dump(),
            )
        return Output(text)

    def batch_generate(self, prompts: List[str], **kwargs):
        """Sequential batch inference."""
        outputs = []
        for p in prompts:
            outputs.append(self(p, **kwargs).text)
        return outputs

    def sc_generate(self, example_txt: str, n_sc: int, bs: int = 8, **kwargs):
        """Self-consistency generation."""
        outputs = []
        for _ in range(n_sc):
            outputs.append(self(example_txt, **kwargs).text.strip())
        return outputs

    def get_next_token_logits(self, *_, **__):
        warnings.warn("OpenAI API does not expose logits; returning None.")
        return None

    @classmethod
    def load_from_openai(cls, model_name: str, **kwargs):
        return cls(model_name, **kwargs)
