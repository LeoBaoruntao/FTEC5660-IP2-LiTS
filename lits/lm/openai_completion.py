"""Raw OpenAI-compatible completion without chat formatting or text stripping."""

import time

from lits import observability as tracing
from .base import Output
from .openai_chat import OpenAIChatModel
from .transport import request_with_network_retry


class CompletionOutput(Output):
    """One completion with unmodified text, finish reason, logprobs and usage.

    Unlike chat Output, text includes any generated thinking tags/content.
    No model-specific reasoning parser is applied. logprobs retains the server's
    dictionary format, including llama.cpp's content entries and EOS token IDs.
    Token counts describe the actual request, before any caller-side trimming.
    """

    def __init__(self, response):
        """Read the single SDK completion without discarding generation metadata."""
        choice = response.choices[0]
        super().__init__(choice.text)
        self.finish_reason = choice.finish_reason
        self.logprobs = choice.logprobs.model_dump(exclude_unset=True) if choice.logprobs else None
        self.input_tokens = response.usage.prompt_tokens
        self.output_tokens = response.usage.completion_tokens


class OpenAICompletionModel(OpenAIChatModel):
    """Reuse client initialization and logging, but send already-rendered prompts.

    The caller owns chat-template rendering and thinking-mode selection; inherited
    sys_prompt/enable_thinking settings do not alter raw prompts. One call returns
    one candidate; call again with a distinct seed for another candidate.

    Example::

        model = OpenAICompletionModel("local-instruct", base_url=server_url)
        output = model(rendered_prompt + prefix, seed=7, max_new_tokens=512,
                       top_k=20, min_p=0.0, logprobs=1)
        next_prefix = prefix + output.text
    """

    def __call__(self, prompt: str, role: str = "default", temperature: float = 1.0,
                 top_p: float = 1.0, max_new_tokens: int | None = None,
                 stop: str | list[str] | None = None, seed: int | None = None,
                 top_k: int | None = None, min_p: float | None = None,
                 logprobs: int | None = None) -> CompletionOutput:
        """Complete a string verbatim; reject unsupported arguments instead of ignoring them.

        seed/logprobs use SDK fields; top_k/min_p use server-specific extra_body.
        max_new_tokens defaults to the configured limit or 512. stop is forwarded
        unchanged. Failed requests propagate; successful usage is recorded once
        when an inference logger and non-None role are supplied.
        """
        if not isinstance(prompt, str):
            raise TypeError("Raw completion requires a rendered prompt string, not chat messages")
        extra_body = {key: value for key, value in {"top_k": top_k, "min_p": min_p}.items()
                      if value is not None}
        limit = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        started = time.monotonic()
        with tracing.span("llm.call", lits_role=role, request_extra_body=extra_body):
            response = request_with_network_retry(self.client.completions.create,
                model=self.model_name, prompt=prompt, n=1, temperature=temperature,
                top_p=top_p, max_tokens=512 if limit is None else limit,
                stop=stop, seed=seed, logprobs=logprobs, extra_body=extra_body,
            )
        output = CompletionOutput(response)
        if self.inference_logger is not None and role is not None:
            self.inference_logger.update_usage(
                input_tokens=output.input_tokens, output_tokens=output.output_tokens,
                batch=False, batch_size=1, role=role,
                running_time=time.monotonic() - started,
            )
        return output
