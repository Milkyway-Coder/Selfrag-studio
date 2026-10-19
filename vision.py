"""
vision.py — image understanding through a Groq vision model.

Groq's vision-capable model list has changed more than once (Llama 4 Scout/Maverick, then Qwen 3.x),
so nothing is hard-wired: set VISION_MODEL in .env, otherwise the candidates below are tried in order
and the first one that exists for your account is remembered.
"""
import base64
import io
import os
import re
import time
from llm import chat_text

CANDIDATES = [m for m in (
    os.getenv("VISION_MODEL"),
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
) if m]

_working = None


class VisionUnavailable(RuntimeError):
    pass


FIGURE_PROMPT = (
    "Describe this figure so it can be found by a search index. State what kind it is (chart, diagram, "
    "flowchart, table, photo, screenshot, equation), what it shows, axis titles, legend entries, key values "
    "and trends, and any text written inside it. Write formulas as LaTeX. Be factual, under 150 words."
)

PAGE_PROMPT = (
    "This is one page of a document. Transcribe ONLY what plain text extraction would miss:\n"
    "1. mathematical formulas and equations, as LaTeX (inline $...$, display $$...$$)\n"
    "2. tables, as Markdown tables\n"
    "3. figures, charts and diagrams: for each one write a line that starts exactly with 'Figure:' followed by a short "
    "title, then describe it on the next lines (what it is, labels, axis titles, key values, relationships)\n"
    "If the page is scanned, transcribe all of its text. If the page has none of the above and is only "
    "ordinary paragraphs, reply exactly: NONE"
)


def _data_url(img_bytes, max_side=1400):
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _model_missing(err):
    s = str(err).lower()
    return any(k in s for k in ("does not exist", "model_not_found", "decommission", "do not have access",
                                "no access", "not supported"))


def _too_large(s):
    return "request too large" in s or "reduce your message size" in s or "context length" in s or "too large" in s


def _rate_limited(s):
    return "429" in s or "rate limit" in s or "rate_limit" in s


def _retry_after(msg):
    """Seconds Groq asks us to wait: 'try again in 7.5s', '1m12.3s' or '450ms'."""
    m = re.search(r"try again in\s*([\d.]+)ms", msg, re.I)
    if m:
        return float(m.group(1)) / 1000
    m = re.search(r"try again in\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?", msg, re.I)
    if m and (m.group(1) or m.group(2)):
        return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return 10.0


def describe(img_bytes, prompt=FIGURE_PROMPT, max_tokens=800, on_wait=None, max_wait=150):
    """
    Returns the model's text, or raises VisionUnavailable if no candidate model works.

    Groq limits tokens per minute, and every image costs about 2,048 tokens on top of the prompt and the
    reply allowance. So: a request that is "too large" is retried smaller (lower resolution, shorter
    reply) and then on the next model; a rate limit waits the number of seconds Groq asks for.
    """
    global _working
    order = ([_working] if _working else []) + [c for c in CANDIDATES if c != _working]
    plans = [(1400, max_tokens), (1000, max(300, int(max_tokens * 0.7))), (800, max(250, int(max_tokens * 0.5)))]
    last, waited = None, 0.0
    for model in order:
        skip_model = False
        for side, mt in plans:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(img_bytes, side)}}]}]
            while True:
                try:
                    text = chat_text(messages, model=model, temperature=0.0, timeout=90, max_tokens=mt,
                                     retries=0, floor=False)
                    _working = model
                    return text
                except Exception as e:                       # noqa: BLE001
                    last, s = e, str(e).lower()
                    if _too_large(s):
                        break                                # try the next, smaller plan
                    if _rate_limited(s) and waited < max_wait and _retry_after(s) <= 90:       # a daily limit ("try again in 2h") is not worth waiting for
                        w = min(_retry_after(s) + 0.5, 60)
                        waited += w
                        if on_wait:
                            on_wait(w)
                        time.sleep(w)
                        continue
                    if _model_missing(e):
                        skip_model = True
                        break
                    raise
            if skip_model:
                break
    raise VisionUnavailable(
        f"No vision model could handle the request (tried {', '.join(order)}). It may be too large for your Groq "
        f"plan's tokens-per-minute limit, or the model is unavailable. Set VISION_MODEL in .env to a model "
        f"listed at console.groq.com/docs/vision. Last error: {str(last)[:200]}")
