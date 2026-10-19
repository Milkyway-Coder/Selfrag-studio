"""Offline stand-in for the Groq client with deterministic, prompt-sniffing replies."""
import json, re
from types import SimpleNamespace as NS

def _flat(c):
    if isinstance(c, str): return c
    return " ".join(p.get("text", "") if p.get("type") == "text" else "<image>" for p in c)

def _reply(messages):
    text = "\n".join(_flat(m["content"]) for m in messages)
    last = _flat(messages[-1]["content"])
    if "<image>" in text:
        if "Transcribe ONLY" in text: return "[formula] $$E = mc^2$$ and a bar chart of loss versus epoch."
        return "A line chart showing training loss decreasing over epochs."
    if "You plan a computation" in text:
        label = re.search(r'- "(.*?)" \[', text).group(1)
        return json.dumps({"source": label, "group_by": ["dept"], "aggregations": [{"column": "salary", "fn": "mean"}], "sort_by": "mean(salary)", "ascending": False})
    if "multiple-choice questions" in text:
        return json.dumps([{"question": f"Question {i}: who created Flask?", "options": ["Armin Ronacher", "Guido van Rossum", "Linus Torvalds", "Brendan Eich"], "answer": 0,
                            "explanation": "The passages say Flask was created by Armin Ronacher."} for i in range(1, 6)] + [{"question": "bad", "options": ["a", "a", "b", "c"], "answer": 9}])
    if "sampled from across one document" in text:
        return json.dumps({"summary": "A short document that explains what Flask is, who created it and which libraries it builds on.",
                           "tags": ["Flask", "python web", "framework"],
                           "entities": {"people": ["Armin Ronacher"], "organizations": [], "locations": [], "dates": [],
                                        "technologies": ["Flask", "Werkzeug", "Jinja2"], "key_terms": ["microframework"]}})
    if "Summarise this earlier part" in text: return "The user asked who created Flask."
    if "Rewrite the user's last question" in text: return "Who created Flask and what is it based on?"
    if "Score each dimension" in text:
        return json.dumps({"retrieval_relevance": 4, "answer_grounding": 4, "answer_quality": 4, "reasoning": "stub"})
    if "strict fact-checker" in text or "fact-checker for an AI" in text:
        sents = re.findall(r"^\d+\.\s+(.*)$", text, re.M)
        return json.dumps([{"id": i + 1, "supported": "moon" not in s.lower()} for i, s in enumerate(sents)])
    if "answer evaluator" in text:
        return json.dumps({"correct": True, "reason": "stub"})
    if "Generate exactly 2 alternative" in text:
        return "alternative phrasing one\nalternative phrasing two"
    if "Rewrite the query" in text or "Rewrite the query as" in text:
        return "keyword focused rewrite"
    if "Revise the ANSWER" in text:
        return "Flask was created by Armin Ronacher."
    if "exactly 3 sentences" in text:
        return "Sentence one. Sentence two. Sentence three."
    if "Extract named entities" in text:
        return json.dumps({"people": ["Armin Ronacher"], "organizations": [], "locations": [], "dates": [], "technologies": ["Flask"], "key_terms": []})
    if "benchmark test set" in text:
        return json.dumps([{"question": "Who created Flask?", "expected": "Armin Ronacher"}])
    m = re.search(r"\[Passage 1\]\n(.*?)(\n\n---|\n---|$)", text, re.S)
    base = (m.group(1).strip().split(". ")[0] if m else "Flask was created by Armin Ronacher")
    return base + ". The moon is made of cheese."

class _Completions:
    def create(self, model=None, messages=None, stream=False, **k):
        out = _reply(messages)
        if stream:
            def gen():
                for tok in re.findall(r"\S+\s*", out):
                    yield NS(choices=[NS(delta=NS(content=tok))])
            return gen()
        return NS(choices=[NS(message=NS(content=out))])

class Groq:
    def __init__(self, *a, **k):
        self.chat = NS(completions=_Completions())
