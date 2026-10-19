"""
llm.py — single place for the Groq client, model names and retry policy.

Every module used to create its own Groq client with its own timeout handling.
Free-tier Groq returns HTTP 429 under bursts (very common during benchmarks), which
silently turned into "evaluation unavailable" scores. chat() retries with backoff.

Environment variables (all optional):
  GROQ_API_KEY   required
  GEN_MODEL      answer generation model   (default openai/gpt-oss-20b)
  JUDGE_MODEL    evaluator / judge model   (default openai/gpt-oss-120b)
                 For research results use a DIFFERENT, stronger model here to avoid
                 self-preference bias (the model grading its own answers).
"""
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# llama-3.1-8b-instant was shut down by Groq on 2026-08-16 (replacement: openai/gpt-oss-20b).
GEN_MODEL   = os.getenv("GEN_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _is_retryable(err):
    s = str(err).lower()
    return any(k in s for k in ("429", "rate limit", "rate_limit", "timeout",
                                "timed out", "503", "502", "overloaded", "connection"))


def _reasoning_param(model):
    """Reasoning-effort value for model families that think before answering (None = not a reasoning model)."""
    m = model or ""
    if "gpt-oss" in m:
        return "low"
    if "qwen3" in m or "qwen/qwen" in m:
        return "none"           # Qwen3 on Groq: "none" switches thinking off
    return None


def chat(messages, model=None, temperature=0.0, timeout=30,
         max_tokens=None, retries=3, stream=False, floor=True):
    """
    Thin wrapper over client.chat.completions.create with exponential backoff.
    Returns the raw response (or the stream iterator when stream=True).

    Reasoning models spend part of max_tokens on hidden thinking, so a small limit can return EMPTY
    content. For gpt-oss the limit is raised (floor=True) and effort is set to low. If the server rejects
    the reasoning parameter we retry without it. Callers that must keep requests small (vision, which
    counts against a tokens-per-minute budget) pass floor=False.
    """
    model = model or GEN_MODEL
    kwargs = dict(model=model, messages=messages, temperature=temperature, timeout=timeout)
    effort = _reasoning_param(model)
    if max_tokens:
        kwargs["max_tokens"] = max(max_tokens, 1500) if (effort == "low" and floor) else max_tokens
    if stream:
        kwargs["stream"] = True
    if effort:
        kwargs["extra_body"] = {"reasoning_effort": effort}

    last = None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:                     # noqa: BLE001
            last = e
            if "extra_body" in kwargs and "reasoning" in str(e).lower():
                kwargs.pop("extra_body"); continue          # server rejected the parameter
            if attempt < retries and _is_retryable(e):
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise
    raise last


def chat_text(messages, **kw):
    """chat() that returns just the stripped message text."""
    resp = chat(messages, **kw)
    return (resp.choices[0].message.content or "").strip()
