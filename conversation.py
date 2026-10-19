"""
conversation.py — memory for multi-turn chat.

Three layers:
  1. recent turns     the last N user/assistant turns are sent verbatim (N is set in the UI, 1 to 12)
  2. rolling summary  older turns are summarised in one short LLM call (cached) and sent as context, so
                      references to something said 20 turns ago still resolve
  3. follow-up rewrite  "what about the second one?" is rewritten into a standalone question for
                      RETRIEVAL, because a search index cannot resolve pronouns
"""
import hashlib
import re
from llm import chat_text, GEN_MODEL

MAX_HISTORY_MESSAGES = 60
_cache = {}

_FOLLOW_UP = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|the (above|previous|last|first|second|third|"
    r"former|latter)|above|earlier|previously|another|more|also|too|same|again|and what about|what about|how about|"
    r"why\??$|explain (it|that|more|further)|elaborate|continue|go on)\b", re.I)


def clean(history, limit=MAX_HISTORY_MESSAGES):
    ok = [{"role": t["role"], "content": str(t["content"])[:1500]}
          for t in (history or []) if isinstance(t, dict) and t.get("role") in ("user", "assistant")
          and str(t.get("content", "")).strip()]
    return ok[-limit:]


def split(history, turns):
    """(older, recent): recent is the last `turns` exchanges."""
    h = clean(history)
    cut = max(0, len(h) - turns * 2)
    return h[:cut], h[cut:]


def summarize(older):
    if len(older) < 2:
        return None
    key = hashlib.md5("|".join(m["content"] for m in older).encode("utf-8", "ignore")).hexdigest()
    if key in _cache:
        return _cache[key]
    transcript = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}" for m in older)
    try:
        s = chat_text([{"role": "user", "content":
                        "Summarise this earlier part of a conversation in at most 120 words. Keep names, "
                        "numbers, files, figure and table references and what the user is trying to do.\n\n" + transcript}],
                      model=GEN_MODEL, temperature=0.0, timeout=25, max_tokens=300, retries=1)
    except Exception as e:                               # noqa: BLE001
        print(f"summarize error: {e}")
        return None
    if len(_cache) > 64:
        _cache.clear()
    _cache[key] = s or None
    return _cache[key]


def looks_like_follow_up(query, history):
    if not history:
        return False
    return len(query.split()) <= 8 or bool(_FOLLOW_UP.search(query))


def condense(query, recent):
    """Standalone version of a follow-up question, or None if unchanged / on failure."""
    if not recent:
        return None
    transcript = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}" for m in recent[-4:])
    try:
        out = chat_text([{"role": "user", "content":
                          "Rewrite the user's last question so it can be understood without the conversation. "
                          "Replace pronouns and vague references with what they refer to. Keep it short. If it is "
                          "already standalone, return it unchanged. Return ONLY the question.\n\n"
                          f"Conversation:\n{transcript}\n\nLast question: {query}\n\nStandalone question:"}],
                        model=GEN_MODEL, temperature=0.0, timeout=15, max_tokens=120, retries=1).strip().strip('"\'')
    except Exception as e:                               # noqa: BLE001
        print(f"condense error: {e}")
        return None
    return out if out and out.lower() != query.lower() and len(out) < 400 else None
