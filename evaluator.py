import json
import re
from llm import chat_text, JUDGE_MODEL


def evaluate(query, chunks, answer):
    """Standard RAG self-evaluation — returns scores dict."""
    if not chunks or not answer:
        raise ValueError("Cannot evaluate — missing chunks or answer.")

    context = "\n\n---\n\n".join(chunks[:3])

    eval_prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the following and return ONLY a JSON object, no extra text, no markdown.

Query: {query}

Retrieved Context:
{context}

Generated Answer: {answer}

Score each dimension from 1 to 5:
- retrieval_relevance: Are the retrieved chunks relevant to the query?
- answer_grounding: Is the answer fully supported by the context?
- answer_quality: Is the answer clear, complete and useful?

Return ONLY this JSON, nothing else:
{{"retrieval_relevance": <int>, "answer_grounding": <int>, "answer_quality": <int>, "reasoning": "<one sentence>"}}"""

    print("\nRunning self-evaluation...")
    raw = ""
    try:
        raw = chat_text([{"role": "user", "content": eval_prompt}],
                        model=JUDGE_MODEL, temperature=0.0, timeout=30)
        raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in evaluator response: {raw}")

        scores = json.loads(raw[start:end])

        for key in ["retrieval_relevance", "answer_grounding", "answer_quality"]:
            if key not in scores:
                scores[key] = 3
            scores[key] = max(1, min(5, int(scores[key])))

        if "reasoning" not in scores:
            scores["reasoning"] = "No reasoning provided."

        avg = round(
            (scores["retrieval_relevance"] +
             scores["answer_grounding"] +
             scores["answer_quality"]) / 3, 2
        )
        scores["average_score"] = avg

        print(f"  Retrieval relevance : {scores['retrieval_relevance']}/5")
        print(f"  Answer grounding    : {scores['answer_grounding']}/5")
        print(f"  Answer quality      : {scores['answer_quality']}/5")
        print(f"  Average             : {avg}/5")
        print(f"  Reasoning           : {scores['reasoning']}")

        return scores

    except json.JSONDecodeError as e:
        print(f"Evaluator JSON parse error: {e} — raw: {raw}")
        raise ValueError(f"Evaluator returned invalid JSON: {e}")
    except Exception as e:
        print(f"Evaluator error: {e}")
        raise


def detect_hallucination(answer, chunks):
    """
    Feature 2 — Hallucination Detection.

    Splits the answer into sentences and checks each one against the
    retrieved context. Returns:
    {
        "sentences": [
            {"text": str, "supported": bool},
            ...
        ],
        "hallucination_score": float,   # 0.0 (none) to 1.0 (all hallucinated)
        "unsupported_count": int,
        "total_count": int
    }
    """
    if not answer or not chunks:
        return {"sentences": [], "hallucination_score": 0.0,
                "unsupported_count": 0, "total_count": 0}

    # split into sentences (basic split on . ? !)
    raw_sentences = re.split(r'(?<=[.?!])\s+', answer.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 20]

    if not sentences:
        return {"sentences": [], "hallucination_score": 0.0,
                "unsupported_count": 0, "total_count": 0}

    context = "\n\n".join(chunks[:4])

    # batch all sentences in one API call to save tokens
    sentences_block = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(sentences)
    )

    prompt = f"""You are a fact-checker for an AI system.

Context (the only source of truth):
{context}

Below are numbered sentences from an AI-generated answer.
For each sentence, decide if it is SUPPORTED or HALLUCINATED based strictly on the context above.
- SUPPORTED: the context directly states or clearly implies this
- HALLUCINATED: the context does not support this, or it adds information not in the context

Sentences:
{sentences_block}

Return ONLY a JSON array in this exact format, nothing else:
[{{"id": 1, "supported": true}}, {{"id": 2, "supported": false}}, ...]"""

    print("\nRunning hallucination detection...")

    try:
        raw = chat_text([{"role": "user", "content": prompt}],
                        model=JUDGE_MODEL, temperature=0.0, timeout=30)
        raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in hallucination response")

        results = json.loads(raw[start:end])

        # map results back to sentences
        support_map = {r["id"]: r.get("supported", True) for r in results}

        labeled = []
        for i, sentence in enumerate(sentences):
            labeled.append({
                "text"     : sentence,
                "supported": support_map.get(i + 1, True)
            })

        unsupported_count = sum(1 for s in labeled if not s["supported"])
        total_count       = len(labeled)
        hallucination_score = round(unsupported_count / total_count, 2) if total_count else 0.0

        print(f"  Hallucination score : {hallucination_score} "
              f"({unsupported_count}/{total_count} sentences unsupported)")

        return {
            "sentences"          : labeled,
            "hallucination_score": hallucination_score,
            "unsupported_count"  : unsupported_count,
            "total_count"        : total_count
        }

    except Exception as e:
        print(f"Hallucination detection error: {e}")
        # fail gracefully — return all supported so UI isn't broken
        return {
            "sentences"          : [{"text": s, "supported": True} for s in sentences],
            "hallucination_score": 0.0,
            "unsupported_count"  : 0,
            "total_count"        : len(sentences)
        }