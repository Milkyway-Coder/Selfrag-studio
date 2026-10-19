"""
research.py — the research contribution layer of Self-RAG.

Four ideas, each independently switchable so they can be ablated:

 1. Chunk-level hybrid retrieval with Reciprocal Rank Fusion (RRF)
    The original pipeline ranks DOCUMENTS with TF-IDF and only then searches vectors.
    Here lexical (BM25) and dense rankings are fused at CHUNK level, so an exact-term
    match can pull in a chunk the dense model would have missed (and vice versa).

 2. Cascaded sentence verification
    Every answer sentence is compared with the retrieved evidence sentences by embedding
    similarity. Clear matches / clear misses are decided without any LLM call; only the
    ambiguous middle band goes to an LLM judge. Cuts verification cost and gives a
    measurable "LLM calls saved" metric.

 3. Sentence-level corrective regeneration ("verify & repair")
    The original Self-RAG loop retries the WHOLE query when the average score is low.
    Here, unsupported sentences are rewritten or pruned while supported ones are kept,
    and the repaired answer is re-verified before it is accepted.

 4. Calibrated abstention
    The LLM self-evaluation score is treated as a confidence estimate. We measure its
    calibration (ECE, Brier, AUROC), learn a logistic recalibration with cross-validation,
    and report risk–coverage curves (selective prediction).

run_ablation() runs a ladder of configurations and returns per-question records plus
aggregates with confidence intervals, paired-bootstrap p-values and a LaTeX table.
"""

import re
import math
import time
import random
import threading
import uuid
from collections import Counter

import numpy as np

from indexer import model, small_col, large_col
from retriever import _rerank, _expand_query
from page_index import tokenize
from generator import generate, reformulate_query
from evaluator import evaluate
from llm import chat_text, GEN_MODEL, JUDGE_MODEL

_SENT_RE = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"\'(\[])')


def split_sentences(text, min_len=20):
    """Sentence splitter shared by verification and repair."""
    text = re.sub(r'\s+', ' ', (text or '').strip())
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chunk-level hybrid retrieval (BM25 + dense, fused with RRF)
# ══════════════════════════════════════════════════════════════════════════════

class _BM25:
    """Minimal Okapi BM25 so the project needs no extra dependency."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = (sum(len(d) for d in docs_tokens) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(self.N)
        for i, (tf, doc) in enumerate(zip(self.tf, self.docs)):
            dl = len(doc) or 1
            s = 0.0
            for t in query_tokens:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out[i] = s
        return out


_bm25_cache = {}
_bm25_lock = threading.Lock()


def _where(doc_ids):
    if not doc_ids:
        return None
    ids = list(doc_ids)
    return {"doc_id": ids[0]} if len(ids) == 1 else {"doc_id": {"$in": ids}}


def _load_large(doc_ids):
    res = large_col.get(where=_where(doc_ids), include=["documents", "metadatas"])
    return res["ids"], res["documents"], res["metadatas"]


def _bm25_for(doc_ids):
    ids, docs, metas = _load_large(doc_ids)
    if not ids:
        return None
    key = hash(tuple(ids))
    with _bm25_lock:
        hit = _bm25_cache.get(key)
        if hit:
            return hit
        index = (_BM25([tokenize(d) for d in docs]), ids, docs, metas)
        if len(_bm25_cache) > 16:
            _bm25_cache.clear()
        _bm25_cache[key] = index
        return index


def bm25_rank(query, doc_ids=None, k=20):
    """Ranked list of large-chunk ids by BM25."""
    qt = tokenize(query)
    idx = _bm25_for(doc_ids)
    if not idx or not qt:
        return []
    bm, ids, _, _ = idx
    sc = bm.scores(qt)
    order = np.argsort(-sc)[:k]
    return [ids[i] for i in order if sc[i] > 0]


def dense_large_rank(query, doc_ids=None, k=20):
    """Ranked list of large-chunk ids by cosine distance on the large collection."""
    try:
        qe = model.encode([query], show_progress_bar=False).tolist()
        n = max(1, min(k, large_col.count()))
        res = large_col.query(query_embeddings=qe, n_results=n, where=_where(doc_ids))
        return res["ids"][0]
    except Exception as e:
        print(f"dense_large_rank: {e}")
        return []


def dense_parent_rank(query, doc_ids=None, k=20):
    """Parent-child: rank SMALL chunks, return their distinct PARENT ids in order."""
    try:
        qe = model.encode([query], show_progress_bar=False).tolist()
        n = max(1, min(k * 2, small_col.count()))
        res = small_col.query(query_embeddings=qe, n_results=n, where=_where(doc_ids))
        seen, out = set(), []
        for m in res["metadatas"][0]:
            pid = m.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out[:k]
    except Exception as e:
        print(f"dense_parent_rank: {e}")
        return []


def rrf_fuse(rankings, k=60, weights=None):
    """
    Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d)).
    rankings: list of ranked id lists. Returns [(id, score)] best first.
    """
    weights = weights or [1.0] * len(rankings)
    score = {}
    for w, ranking in zip(weights, rankings):
        for rank, did in enumerate(ranking, start=1):
            score[did] = score.get(did, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def retrieve_cfg(query, cfg, doc_ids=None, top_k=5):
    """
    Retrieval under a named configuration (used by the ablation ladder and by the app).
    cfg keys: dense (large dense) | parent (small->parent dense) | bm25 | rerank | expand
    Returns (chunks:list[str], llm_calls:int)
    """
    calls = 0
    queries = [query]
    if cfg.get("expand"):
        queries = _expand_query(query)
        calls += 1 if len(queries) > 1 or len(query.split()) >= 4 else 0

    rankings = []
    for q in queries:
        if cfg.get("dense", True) and not cfg.get("parent"):
            rankings.append(dense_large_rank(q, doc_ids, k=top_k * 3))
        if cfg.get("parent"):
            rankings.append(dense_parent_rank(q, doc_ids, k=top_k * 3))
        if cfg.get("bm25"):
            rankings.append(bm25_rank(q, doc_ids, k=top_k * 3))
    rankings = [r for r in rankings if r]
    if not rankings:
        return [], calls

    fused = rrf_fuse(rankings)
    pool_n = top_k * 3 if cfg.get("rerank") else top_k
    top_ids = [i for i, _ in fused[:pool_n]]
    got = large_col.get(ids=top_ids, include=["documents"])
    by_id = dict(zip(got["ids"], got["documents"]))
    texts = [by_id[i] for i in top_ids if i in by_id and by_id[i] and by_id[i].strip()]

    if cfg.get("rerank"):
        texts = _rerank(query, texts, top_n=top_k)
    return texts[:top_k], calls


def hybrid_retrieve(query, top_k=5, doc_ids=None, rerank=True, expand=False):
    """Convenience wrapper used by the app: parent-child dense + BM25 fused with RRF."""
    chunks, _ = retrieve_cfg(
        query, {"parent": True, "bm25": True, "rerank": rerank, "expand": expand},
        doc_ids=doc_ids, top_k=top_k)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 2. Cascaded sentence verification
# ══════════════════════════════════════════════════════════════════════════════

_NOT_FOUND_RE = re.compile(r"not available in the indexed documents", re.I)


def _llm_judge(sentences, context, model_name=None):
    """One batched LLM call -> {1-based index: bool supported}. Unparseable -> {}."""
    import json
    block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = f"""You are a strict fact-checker.

Context (the only source of truth):
{context}

Numbered sentences from an AI answer:
{block}

For each sentence decide SUPPORTED (the context states or clearly implies it) or
HALLUCINATED (not supported, contradicted, or adds information not in the context).

Return ONLY a JSON array: [{{"id": 1, "supported": true}}, ...]"""
    try:
        raw = chat_text([{"role": "user", "content": prompt}],
                        model=model_name or JUDGE_MODEL, temperature=0.0, timeout=30)
        raw = raw.replace("```json", "").replace("```", "").strip()
        a, b = raw.find("["), raw.rfind("]") + 1
        arr = json.loads(raw[a:b])
        return {int(r["id"]): bool(r.get("supported", True)) for r in arr}
    except Exception as e:
        print(f"_llm_judge error: {e}")
        return {}


def verify_answer(answer, chunks, hi=0.65, lo=0.30, mode="cascade", judge_model=None):
    """
    mode="cascade"  : embedding similarity decides clear cases, LLM only for hi>sim>lo
    mode="llm_only" : every sentence goes to the LLM judge (strict, used as the
                      INDEPENDENT measurement verifier in ablations)
    mode="embed"    : embeddings only (no LLM at all)

    Returns {sentences:[{text,supported,similarity,method,evidence}],
             hallucination_score, unsupported_count, total_count,
             llm_calls, embedding_decided}
    """
    sents = [s for s in split_sentences(answer) if not _NOT_FOUND_RE.search(s)]
    empty = {"sentences": [], "hallucination_score": 0.0, "unsupported_count": 0,
             "total_count": 0, "llm_calls": 0, "embedding_decided": 0}
    if not sents or not chunks:
        return empty

    evidence = []
    for ch in chunks:
        evidence.extend(split_sentences(ch, min_len=15))
    evidence = evidence[:300]
    if not evidence:
        return empty

    embs = model.encode(sents + evidence, normalize_embeddings=True, show_progress_bar=False)
    s_emb, e_emb = embs[:len(sents)], embs[len(sents):]
    sim = s_emb @ e_emb.T
    best_j = sim.argmax(axis=1)
    best = sim.max(axis=1)

    rows = []
    for i, s in enumerate(sents):
        rows.append({"text": s, "supported": None, "similarity": round(float(best[i]), 3),
                     "method": None, "evidence": evidence[int(best_j[i])][:300]})

    pending = []
    for i, r in enumerate(rows):
        if mode == "llm_only":
            pending.append(i)
        elif mode == "embed":
            r["supported"], r["method"] = bool(r["similarity"] >= (hi + lo) / 2), "embedding"
        elif r["similarity"] >= hi:
            r["supported"], r["method"] = True, "embedding"
        elif r["similarity"] <= lo:
            r["supported"], r["method"] = False, "embedding"
        else:
            pending.append(i)

    llm_calls = 0
    if pending:
        context = "\n\n".join(chunks[:4])
        verdict = _llm_judge([rows[i]["text"] for i in pending], context, judge_model)
        llm_calls = 1
        for k, i in enumerate(pending, start=1):
            if k in verdict:
                rows[i]["supported"], rows[i]["method"] = verdict[k], "llm"
            else:   # judge failed: fall back to the midpoint of the embedding band
                rows[i]["supported"] = bool(rows[i]["similarity"] >= (hi + lo) / 2)
                rows[i]["method"] = "embedding-fallback"

    bad = sum(1 for r in rows if not r["supported"])
    return {"sentences": rows,
            "hallucination_score": round(bad / len(rows), 2),
            "unsupported_count": bad, "total_count": len(rows),
            "llm_calls": llm_calls,
            "embedding_decided": sum(1 for r in rows if r["method"] == "embedding")}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sentence-level corrective regeneration
# ══════════════════════════════════════════════════════════════════════════════

def repair_answer(query, answer, chunks, verification):
    """
    Rewrites ONLY what failed verification.
    Candidates: original, LLM-revised, pruned (unsupported sentences removed).
    The candidate with the lowest re-verified hallucination score wins; ties prefer the
    one that keeps more content.

    Returns {answer, verification, method: 'none'|'revised'|'pruned', changed, llm_calls,
             before, after}
    """
    bad = [s["text"] for s in verification.get("sentences", []) if not s["supported"]]
    before = verification.get("hallucination_score", 0.0)
    if not bad:
        return {"answer": answer, "verification": verification, "method": "none",
                "changed": False, "llm_calls": 0, "before": before, "after": before}

    calls = 0
    cands = []

    # candidate A: LLM revision constrained to the evidence
    context = "\n\n---\n\n".join(chunks[:4])
    flagged = "\n".join(f"- {b}" for b in bad)
    prompt = f"""Revise the ANSWER so that every sentence is supported by the CONTEXT.

Rules:
- Keep sentences that are supported, unchanged.
- Fix or delete the FLAGGED sentences. Never add facts that are not in the context.
- If nothing in the context answers the question, reply exactly:
  This information is not available in the indexed documents.
- Return only the revised answer.

QUESTION: {query}

CONTEXT:
{context}

ANSWER:
{answer}

FLAGGED (unsupported):
{flagged}

REVISED ANSWER:"""
    try:
        revised = chat_text([{"role": "user", "content": prompt}],
                            model=GEN_MODEL, temperature=0.0, timeout=45)
        calls += 1
        if revised:
            v = verify_answer(revised, chunks)
            calls += v["llm_calls"]
            cands.append(("revised", revised, v))
    except Exception as e:
        print(f"repair revise error: {e}")

    # candidate B: prune unsupported sentences from the original
    bad_set = set(bad)
    kept = [s["text"] for s in verification["sentences"] if s["text"] not in bad_set]
    if kept:
        pruned = " ".join(kept)
        cands.append(("pruned", pruned, verify_answer(pruned, chunks, mode="embed")))

    if not cands:
        return {"answer": answer, "verification": verification, "method": "none",
                "changed": False, "llm_calls": calls, "before": before, "after": before}

    cands.sort(key=lambda c: (c[2]["hallucination_score"], -len(c[1])))
    method, best_text, best_v = cands[0]
    if best_v["hallucination_score"] >= before and method != "pruned":
        return {"answer": answer, "verification": verification, "method": "none",
                "changed": False, "llm_calls": calls, "before": before, "after": before}
    return {"answer": best_text, "verification": best_v, "method": method,
            "changed": True, "llm_calls": calls, "before": before,
            "after": best_v["hallucination_score"]}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Calibration & selective prediction
# ══════════════════════════════════════════════════════════════════════════════

def ece(conf, correct, bins=10):
    """Expected Calibration Error with equal-width bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2)) if len(conf) else 0.0


def auroc(conf, correct):
    """Rank-based AUROC of confidence as a predictor of correctness (ties averaged)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, int)
    pos, neg = conf[correct == 1], conf[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def reliability_bins(conf, correct, bins=5):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        out.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(m.sum()),
                    "confidence": round(float(conf[m].mean()), 3) if m.any() else None,
                    "accuracy": round(float(correct[m].mean()), 3) if m.any() else None})
    return out


def risk_coverage(conf, correct):
    """
    Selective prediction: answer the most confident x% of questions, abstain on the rest.
    Returns (points[{coverage, risk}], AURC) — lower AURC is better.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return [], None
    order = np.argsort(-conf, kind="stable")
    err = 1 - correct[order]
    cum = np.cumsum(err) / np.arange(1, n + 1)
    pts = [{"coverage": round((i + 1) / n, 3), "risk": round(float(cum[i]), 3)} for i in range(n)]
    return pts, float(cum.mean())


def fit_logistic(X, y, l2=1.0, iters=300, lr=0.5):
    """L2-regularised logistic regression (full-batch gradient descent, numpy only)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        gw = Z.T @ (p - y) / len(y) + l2 * w / len(y)
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def predict_logistic(m, X):
    Z = (np.asarray(X, float) - m["mu"]) / m["sd"]
    return 1 / (1 + np.exp(-(Z @ m["w"] + m["b"])))


def calibration_report(records, folds=5, seed=7):
    """
    records: [{conf_features:[retrieval, grounding, quality, avg]/5 each, correct:0/1}]
    Compares raw self-evaluation confidence against a cross-validated logistic
    recalibration (out-of-fold predictions, so no leakage).
    """
    recs = [r for r in records if r.get("correct") is not None and r.get("features")]
    n = len(recs)
    if n < 10:
        return {"error": f"Need at least 10 scored questions for calibration (have {n})."}
    X = np.array([r["features"] for r in recs], float)
    y = np.array([int(r["correct"]) for r in recs])
    raw = X[:, -1]                                   # average_score / 5

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    k = min(folds, n)
    cal = np.zeros(n)
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        if len(set(y[tr])) < 2:                      # one class in training fold
            cal[te] = y[tr].mean() if len(tr) else 0.5
            continue
        cal[te] = predict_logistic(fit_logistic(X[tr], y[tr]), X[te])

    rc_raw, aurc_raw = risk_coverage(raw, y)
    rc_cal, aurc_cal = risk_coverage(cal, y)
    return {
        "n": n, "accuracy": round(float(y.mean()), 3),
        "raw": {"ece": round(ece(raw, y), 4), "brier": round(brier(raw, y), 4),
                "auroc": auroc(raw, y), "aurc": aurc_raw,
                "bins": reliability_bins(raw, y), "risk_coverage": rc_raw},
        "calibrated": {"ece": round(ece(cal, y), 4), "brier": round(brier(cal, y), 4),
                       "auroc": auroc(cal, y), "aurc": aurc_cal,
                       "bins": reliability_bins(cal, y), "risk_coverage": rc_cal},
        "oracle_aurc": float(np.mean(np.cumsum(np.sort(1 - y)) / np.arange(1, n + 1))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Ablation study runner
# ══════════════════════════════════════════════════════════════════════════════

# Each configuration ADDS one component to the previous one (a ladder), so the delta
# between rows isolates that component's contribution.
ABLATION_LADDER = [
    ("A0", "Dense (large chunks)",          {"dense": True}),
    ("A1", "+ Parent-child",                {"parent": True}),
    ("A2", "+ BM25 / RRF hybrid",           {"parent": True, "bm25": True}),
    ("A3", "+ Cross-encoder rerank",        {"parent": True, "bm25": True, "rerank": True}),
    ("A4", "+ Self-reflection loop",        {"parent": True, "bm25": True, "rerank": True, "reflect": True}),
    ("A5", "+ Verify & repair (full)",      {"parent": True, "bm25": True, "rerank": True, "reflect": True, "repair": True}),
]


def run_pipeline(query, cfg, doc_ids=None, threshold=3.0, max_retries=1, top_k=5):
    """One question through one configuration. Counts every LLM call the SYSTEM makes."""
    t0 = time.time()
    calls = 0
    cur, attempt, reformulated = query, 0, None
    answer, chunks, scores = "", [], None

    while True:
        attempt += 1
        chunks, c = retrieve_cfg(cur, cfg, doc_ids, top_k=top_k)
        calls += c
        if not chunks:
            break
        answer = generate(cur, chunks)
        calls += 1
        scores = evaluate(cur, chunks, answer)
        calls += 1
        if not cfg.get("reflect") or scores["average_score"] >= threshold or attempt > max_retries:
            break
        new_q = reformulate_query(cur, f"score {scores['average_score']}/5")
        calls += 1
        if not new_q:
            break
        reformulated, cur = new_q, new_q

    repair_info = None
    if cfg.get("repair") and answer and chunks:
        v = verify_answer(answer, chunks)               # system-side cascade verifier
        calls += v["llm_calls"]
        if v["unsupported_count"]:
            repair_info = repair_answer(query, answer, chunks, v)
            calls += repair_info["llm_calls"]
            answer = repair_info["answer"]

    return {"answer": answer, "chunks": chunks, "scores": scores, "attempts": attempt,
            "reformulated": reformulated, "llm_calls": calls,
            "repair": repair_info, "latency": round(time.time() - t0, 2)}


def token_f1(pred, gold):
    """SQuAD-style token F1 — an LLM-free correctness signal to sit beside the judge."""
    p, g = tokenize(pred or ""), tokenize(gold or "")
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def paired_bootstrap_p(a, b, n_boot=2000, seed=11):
    """Two-sided paired bootstrap p-value for mean(a) - mean(b) == 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 5:
        return None
    d = a - b
    obs = d.mean()
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.RandomState(seed)
    centered = d - obs
    boots = np.array([rng.choice(centered, len(d), replace=True).mean() for _ in range(n_boot)])
    return float((np.abs(boots) >= abs(obs)).mean())


def aggregate(records_by_cfg, ladder):
    """records_by_cfg: {cfg_id: [record]} -> list of summary rows (ladder order)."""
    rows, prev = [], None
    for cid, name, _ in ladder:
        recs = records_by_cfg.get(cid, [])
        if not recs:
            continue
        ok = [r for r in recs if not r.get("error")]
        n = len(ok)
        correct = [int(r["correct"]) for r in ok]
        conf = [r["features"][-1] for r in ok if r.get("features")]
        corr_for_conf = [int(r["correct"]) for r in ok if r.get("features")]
        _, aurc = risk_coverage(conf, corr_for_conf) if conf else (None, None)
        lo, hi = wilson(sum(correct), n)
        row = {
            "id": cid, "name": name, "n": n, "errors": len(recs) - n,
            "accuracy": round(100 * sum(correct) / n, 1) if n else 0,
            "ci": [round(100 * lo, 1), round(100 * hi, 1)],
            "token_f1": round(float(np.mean([r["f1"] for r in ok])), 3) if n else 0,
            "faithfulness": round(100 * float(np.mean([1 - r["halluc"] for r in ok])), 1) if n else 0,
            "ece": round(ece(conf, corr_for_conf), 3) if conf else None,
            "aurc": round(aurc, 3) if aurc is not None else None,
            "llm_calls": round(float(np.mean([r["llm_calls"] for r in ok])), 1) if n else 0,
            "latency": round(float(np.mean([r["latency"] for r in ok])), 2) if n else 0,
            "p_vs_prev": None,
        }
        if prev is not None:
            pr = {r["q"]: r for r in prev if not r.get("error")}
            pairs = [(int(r["correct"]), int(pr[r["q"]]["correct"])) for r in ok if r["q"] in pr]
            if pairs:
                row["p_vs_prev"] = paired_bootstrap_p([p[0] for p in pairs], [p[1] for p in pairs])
        prev = recs
        rows.append(row)
    return rows


def to_latex(rows, caption="Ablation of Self-RAG components.", label="tab:ablation"):
    def esc(s):
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"ID & Configuration & Acc.\ (\%) [95\% CI] & F1 & Faith.\ (\%) & ECE$\downarrow$ & AURC$\downarrow$ & LLM calls \\",
             r"\midrule"]
    for r in rows:
        ece_s = "--" if r["ece"] is None else f"{r['ece']:.3f}"
        aurc_s = "--" if r["aurc"] is None else f"{r['aurc']:.3f}"
        sig = ""
        if r["p_vs_prev"] is not None and r["p_vs_prev"] < 0.05:
            sig = r"$^{*}$"
        lines.append(
            f"{r['id']} & {esc(r['name'])} & {r['accuracy']:.1f}{sig} [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] "
            f"& {r['token_f1']:.3f} & {r['faithfulness']:.1f} & {ece_s} & {aurc_s} & {r['llm_calls']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\vspace{2pt}\par\footnotesize $^{*}$ paired bootstrap $p<0.05$ vs.\ the previous row.",
              r"\end{table}"]
    return "\n".join(lines)


def to_csv(rows):
    cols = ["id", "name", "n", "errors", "accuracy", "ci", "token_f1", "faithfulness",
            "ece", "aurc", "llm_calls", "latency", "p_vs_prev"]
    out = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if c == "ci":
                v = f"{v[0]}-{v[1]}"
            v = "" if v is None else str(v)
            vals.append('"' + v.replace('"', '""') + '"')
        out.append(",".join(vals))
    return "\n".join(out)


def run_ablation(questions, doc_ids=None, config_ids=None, threshold=3.0,
                 on_progress=None, should_stop=None, judge_fn=None):
    """
    questions : [{"question","expected"}]
    judge_fn  : (question, expected, actual) -> (bool, reason). Defaults to benchmark.judge_answer.
    Returns {"rows": aggregate rows, "records": {cfg_id: [per-question record]}}

    Fairness: every configuration is MEASURED with the same independent verifier
    (verify_answer mode="llm_only") and the same judge, regardless of which verifier
    the configuration itself uses internally.
    """
    if judge_fn is None:
        from benchmark import judge_answer as judge_fn
    ladder = [c for c in ABLATION_LADDER if not config_ids or c[0] in config_ids]
    total = len(ladder) * len(questions)
    done = 0
    records = {}

    for cid, name, cfg in ladder:
        records[cid] = []
        for item in questions:
            if should_stop and should_stop():
                return {"rows": aggregate(records, ladder), "records": records, "stopped": True}
            q, gold = item["question"], item["expected"]
            rec = {"q": q, "expected": gold}
            try:
                out = _retry(lambda: run_pipeline(q, cfg, doc_ids, threshold))
                ans = out["answer"] or "No relevant content found."
                correct, reason = _retry(lambda: judge_fn(q, gold, ans))
                ver = _retry(lambda: verify_answer(ans, out["chunks"], mode="llm_only")) if out["chunks"] else \
                    {"hallucination_score": 1.0}
                s = out["scores"] or {}
                rec.update({
                    "answer": ans, "correct": bool(correct), "reason": reason,
                    "f1": token_f1(ans, gold),
                    "halluc": ver["hallucination_score"],
                    "llm_calls": out["llm_calls"], "latency": out["latency"],
                    "features": [s.get("retrieval_relevance", 0) / 5, s.get("answer_grounding", 0) / 5,
                                 s.get("answer_quality", 0) / 5, s.get("average_score", 0) / 5] if s else None,
                    "attempts": out["attempts"],
                    "repaired": bool(out["repair"] and out["repair"]["changed"]),
                })
            except Exception as e:                     # noqa: BLE001
                rec["error"] = str(e)
            records[cid].append(rec)
            done += 1
            if on_progress:
                on_progress(done, total, f"{cid} · {q[:50]}")

    return {"rows": aggregate(records, ladder), "records": records}


def _retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                         # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ══════════════════════════════════════════════════════════════════════════════
# Background jobs (ablations take minutes — never block a request thread)
# ══════════════════════════════════════════════════════════════════════════════

JOBS = {}
_jobs_lock = threading.Lock()


def start_ablation_job(questions, doc_ids=None, config_ids=None, threshold=3.0):
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "status": "running", "done": 0, "total": 0, "message": "starting",
           "result": None, "calibration": None, "error": None, "stop": False,
           "started": time.time()}
    with _jobs_lock:
        JOBS[jid] = job

    def progress(done, total, msg):
        job.update(done=done, total=total, message=msg)

    def work():
        try:
            res = run_ablation(questions, doc_ids, config_ids, threshold,
                               on_progress=progress, should_stop=lambda: job["stop"])
            # calibration of the full system (last configuration run)
            last = [cid for cid, _, _ in ABLATION_LADDER if cid in res["records"]][-1:]
            if last:
                job["calibration"] = calibration_report(
                    [r for r in res["records"][last[0]] if not r.get("error")])
            res["latex"] = to_latex(res["rows"])
            res["csv"] = to_csv(res["rows"])
            job["result"] = res
            job["status"] = "stopped" if res.get("stopped") else "done"
        except Exception as e:                         # noqa: BLE001
            job["status"], job["error"] = "error", str(e)

    threading.Thread(target=work, daemon=True).start()
    return jid


def get_job(jid):
    return JOBS.get(jid)
