"""
benchmark.py — Automated accuracy testing for Self-RAG.

Accepts a list of {"question": str, "expected": str} pairs,
runs each through the full pipeline, and uses the LLM to judge
whether the generated answer matches the expected answer semantically.

Returns a structured report with per-question pass/fail and an overall
accuracy percentage.
"""

import json
from llm import chat_text, JUDGE_MODEL
from retriever import retrieve
from generator import generate
from evaluator import evaluate



def judge_answer(question, expected, actual):
    """
    Uses the LLM as a judge to decide if `actual` semantically matches
    `expected`. Returns True/False and a one-line reason.
    """
    prompt = f"""You are an answer evaluator. Decide if the ACTUAL answer correctly
answers the question given the EXPECTED answer as ground truth.

Be lenient about wording — if the actual answer conveys the same information
as the expected answer, even if phrased differently, mark it as CORRECT.
Mark it INCORRECT only if the actual answer is wrong, missing the key fact,
or says it doesn't know.

Question : {question}
Expected : {expected}
Actual   : {actual}

Reply with ONLY this JSON, nothing else:
{{"correct": true/false, "reason": "<one sentence>"}}"""

    try:
        raw = chat_text([{"role": "user", "content": prompt}],
                        model=JUDGE_MODEL, temperature=0.0, timeout=30)
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        result = json.loads(raw[start:end])
        return bool(result.get("correct", False)), result.get("reason", "")
    except Exception as e:
        print(f"Judge error: {e}")
        return False, f"Judge failed: {e}"


def run_benchmark(questions, doc_id=None, top_k=5):
    """
    Runs a full benchmark over a list of question dicts.

    questions : list of {"question": str, "expected": str}
    doc_id    : restrict retrieval to this doc (None = all docs)
    top_k     : number of chunks to retrieve per question

    Returns:
    {
        "total"        : int,
        "correct"      : int,
        "accuracy_pct" : float,
        "results"      : [
            {
                "question"     : str,
                "expected"     : str,
                "actual"       : str,
                "correct"      : bool,
                "reason"       : str,
                "avg_score"    : float,
                "retrieval"    : int,
                "grounding"    : int,
                "quality"      : int,
            },
            ...
        ]
    }
    """
    results    = []
    correct_ct = 0

    print(f"\n{'='*50}")
    print(f"Running benchmark: {len(questions)} questions")
    print(f"Doc filter: {doc_id or 'ALL'}")
    print(f"{'='*50}\n")

    for i, item in enumerate(questions):
        question = item.get("question", "").strip()
        expected = item.get("expected", "").strip()

        if not question or not expected:
            continue

        print(f"[{i+1}/{len(questions)}] {question[:70]}")

        try:
            chunks = retrieve(question, top_k=top_k, doc_id=doc_id)
            if not chunks:
                results.append({
                    "question" : question,
                    "expected" : expected,
                    "actual"   : "No relevant content found.",
                    "correct"  : False,
                    "reason"   : "Retrieval returned no chunks.",
                    "avg_score": 0.0,
                    "retrieval": 0,
                    "grounding": 0,
                    "quality"  : 0,
                })
                print(f"  ✗ No chunks retrieved")
                continue

            actual = generate(question, chunks)

            try:
                scores = evaluate(question, chunks, actual)
                avg_score = scores["average_score"]
                retrieval = scores["retrieval_relevance"]
                grounding = scores["answer_grounding"]
                quality   = scores["answer_quality"]
            except Exception:
                avg_score = 0.0
                retrieval = grounding = quality = 0

            correct, reason = judge_answer(question, expected, actual)
            if correct:
                correct_ct += 1

            results.append({
                "question" : question,
                "expected" : expected,
                "actual"   : actual,
                "correct"  : correct,
                "reason"   : reason,
                "avg_score": avg_score,
                "retrieval": retrieval,
                "grounding": grounding,
                "quality"  : quality,
            })

            status = "✓" if correct else "✗"
            print(f"  {status} score={avg_score} | {reason[:80]}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "question" : question,
                "expected" : expected,
                "actual"   : f"Error: {e}",
                "correct"  : False,
                "reason"   : str(e),
                "avg_score": 0.0,
                "retrieval": 0,
                "grounding": 0,
                "quality"  : 0,
            })

    total        = len(results)
    accuracy_pct = round((correct_ct / total * 100), 1) if total else 0.0

    print(f"\n{'='*50}")
    print(f"Benchmark complete: {correct_ct}/{total} correct ({accuracy_pct}%)")
    print(f"{'='*50}\n")

    return {
        "total"        : total,
        "correct"      : correct_ct,
        "accuracy_pct" : accuracy_pct,
        "results"      : results,
    }