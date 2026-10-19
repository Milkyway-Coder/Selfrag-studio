"""
chitchat.py — greeting / small-talk detection.

The original check used startswith() on a prefix list, so a real question such as
"history of the Roman empire" (starts with "hi") or "okta SSO setup" (starts with "ok")
was swallowed as small talk and never reached the RAG pipeline. This version matches
whole words only and treats a message as small talk only when EVERY word is small talk.
"""
import re

_REPLIES = [
    ({"hello", "hi", "hey", "hii", "hiii", "helo", "hola", "namaste", "howdy", "greetings", "yo", "sup", "heyy", "morning", "evening", "afternoon", "night"},
     "Hello. Ask me anything about your indexed documents."),
    ({"thanks", "thank", "thx", "tq", "ty", "tysm", "tnx", "thnx"},
     "You're welcome. What else would you like to know?"),
    ({"bye", "goodbye", "cya", "later"},
     "Goodbye. Your chats are saved, so come back any time."),
    ({"ok", "okay", "k", "kk", "alright", "cool", "nice", "great", "awesome", "perfect", "good", "fine", "sure",
      "noted", "understood", "lol", "haha", "hmm"},
     "Got it. Ask a question whenever you're ready."),
]
_FIXED = {
    "how are you": "Running well and ready to help. What would you like to know?",
    "who are you": "I'm Self-RAG: a document question-answering system that retrieves evidence, "
                   "evaluates its own answers and flags unsupported statements.",
    "what can you do": "I index PDFs, text, URLs and crawled sites, then answer questions with cited evidence, "
                       "self-evaluation scores and hallucination checks.",
    "help": "Add a document in the Library, then ask a question here.",
}
_FILLER = {"you", "so", "much", "a", "lot", "ton", "very", "many", "there", "again", "then", "np", "no", "problem"}
_WORD = re.compile(r"[a-z']+|[^\sa-z0-9]")


def reply_for(query):
    """Returns a canned reply when the whole message is small talk, else None."""
    q = re.sub(r"[!.,?]+$", "", query.lower().strip())
    if q in _FIXED:
        return _FIXED[q]
    words = [w for w in _WORD.findall(q) if re.match(r"[a-z']+$", w)]
    if not words or len(words) > 5 or len(q.split()) != len(words):
        return None
    union = set().union(*(v for v, _ in _REPLIES))
    if not all(w in union or w in _FILLER for w in words):
        return None
    # priority: thanks > bye > hello > acknowledgement
    for vocab, reply in (_REPLIES[1], _REPLIES[2], _REPLIES[0], _REPLIES[3]):
        if any(w in vocab for w in words):
            return reply
    return None
