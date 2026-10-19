"""
quiz.py — turn a document into a multiple-choice quiz for revision.

Questions are written from passages sampled across the whole document, so a long PDF is covered end to end.
Every question must be answerable from the text, have four options and one correct answer, and carry a
one-sentence explanation. Anything malformed is dropped rather than shown.
"""
import json
import random
import re

from enrich import sample_passages
from llm import chat_text, GEN_MODEL


def _parse(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    a, b = raw.find("["), raw.rfind("]") + 1
    if a < 0 or b <= a:
        raise ValueError("The model did not return a list of questions.")
    return json.loads(raw[a:b])


def make_quiz(doc_id, count=5):
    count = max(3, min(int(count), 10))
    passages = sample_passages(doc_id, n=10, chars=800)
    if not passages:
        raise ValueError("The document has no indexed text.")
    context = "\n\n---\n\n".join(passages)
    prompt = f"""Write {count} multiple-choice questions that test understanding of the passages below.

Reply with ONLY a JSON array. Each item:
{{"question": "...", "options": ["A text", "B text", "C text", "D text"], "answer": <0-3 index of the correct option>, "explanation": "<one sentence saying why, using the passages>"}}

Rules: every question must be answerable from the passages alone; exactly four options with one clearly correct answer;
plausible wrong options; vary which position is correct; do not write "all of the above"; no letters inside the option text.

PASSAGES:
{context}"""
    items = _parse(chat_text([{"role": "user", "content": prompt}], model=GEN_MODEL, temperature=0.3, timeout=90, max_tokens=2200))
    quiz = []
    for it in items:
        try:
            opts = [re.sub(r"^\s*[A-Da-d][.)]\s+", "", str(o)).strip() for o in it["options"]]
            ans = int(it["answer"])
            q = str(it["question"]).strip()
            if len(opts) != 4 or len(set(opts)) != 4 or not 0 <= ans <= 3 or not q or not all(opts):
                continue
            correct = opts[ans]
            random.shuffle(opts)                          # models over-use "B"; the answer index follows the text
            quiz.append({"question": q, "options": opts, "answer": opts.index(correct),
                         "explanation": str(it.get("explanation", "")).strip()})
        except (KeyError, ValueError, TypeError):
            continue
    if not quiz:
        raise ValueError("The model's questions were not usable. Try again.")
    return quiz[:count]
