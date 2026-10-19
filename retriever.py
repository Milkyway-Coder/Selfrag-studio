"""
retriever.py — Two-stage retrieval with three quality improvements:

1. Parent-child retrieval:
   Search small chunks (precise) → fetch large parent chunks (rich context)

2. Cross-encoder reranking:
   After retrieval, score every (query, chunk) pair with a cross-encoder.
   Removes irrelevant chunks that slipped through TF-IDF + vector search.

3. Query expansion:
   Generate 2 alternative phrasings of the query.
   Retrieve for all phrasings and merge results.
   Fixes vocabulary mismatch between question words and document words.
"""

from indexer import model, small_col, large_col, collection
from page_index import search_index, get_conn, tokenize
import os
import threading

# ── Cross-encoder reranker ────────────────────────────────────────────────────
# Loaded in a background thread after Flask starts — avoids the httpx client
# "closed" error that happens when loading at module import time.
_reranker       = None
_reranker_ready = False
_reranker_lock  = threading.Lock()


def _load_reranker_bg():
    """Background thread: downloads and loads the cross-encoder once."""
    global _reranker, _reranker_ready
    try:
        from sentence_transformers import CrossEncoder
        print("Loading cross-encoder reranker (background)...")
        r = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        with _reranker_lock:
            _reranker       = r
            _reranker_ready = True
        print("Cross-encoder reranker ready.")
    except Exception as e:
        print(f"Cross-encoder not available: {e}. Reranking disabled.")
        with _reranker_lock:
            _reranker       = False   # False = failed, don't retry
            _reranker_ready = True


def preload_reranker():
    """Call once at Flask startup — kicks off the background load."""
    t = threading.Thread(target=_load_reranker_bg, daemon=True)
    t.start()


def _get_reranker():
    """Returns the reranker if ready, None otherwise."""
    with _reranker_lock:
        if _reranker is None:
            return None          # still loading — skip reranking for now
        if _reranker is False:
            return None          # failed — permanently disabled
        return _reranker


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve(query, top_k=5, doc_id=None):
    """
    Full retrieval pipeline:
    1. Query expansion  → multiple phrasings of the query
    2. TF-IDF ranking   → find top documents for each phrasing
    3. Vector search    → find small child chunks across top documents
    4. Parent fetch     → swap child chunks for their large parent chunks
    5. Reranking        → cross-encoder scores (query, chunk) and keeps top_k
    """
    print(f"\nRetrieving: '{query}'" + (f" [doc: {doc_id}]" if doc_id else ""))

    # ── Step 1: query expansion ───────────────────────────────────────────────
    all_queries = _expand_query(query)
    print(f"Query expansion: {len(all_queries)} phrasings")

    # ── Step 2: TF-IDF ranking across all query phrasings ────────────────────
    tfidf_doc_scores = {}
    for q in all_queries:
        results = search_index(q, top_k=top_k, doc_id_filter=doc_id)
        for did, score in results:
            tfidf_doc_scores[did] = max(tfidf_doc_scores.get(did, 0), score)

    if not tfidf_doc_scores:
        print("TF-IDF found nothing. Falling back to direct vector search.")
        return _parent_vector_search(query, top_k * 2, doc_id)

    top_doc_ids = sorted(tfidf_doc_scores, key=tfidf_doc_scores.get, reverse=True)[:3]
    print(f"TF-IDF top docs: {top_doc_ids}")

    # ── Step 3: vector search on small chunks, then Step 4: fetch parents ─────
    candidate_chunks = []
    for q in all_queries:
        for did in top_doc_ids:
            chunks = _small_to_large(q, top_k=4, doc_id=did)
            candidate_chunks.extend(chunks)

    # deduplicate by content
    seen, unique_chunks = set(), []
    for ch in candidate_chunks:
        key = ch[:80]
        if key not in seen:
            seen.add(key); unique_chunks.append(ch)

    if not unique_chunks:
        print("No chunks after parent fetch. Falling back to snippet summaries.")
        unique_chunks = []
        for d in top_doc_ids:
            unique_chunks.extend(_direct_large_search(query, 2, d))

    # ── Step 5: cross-encoder reranking ──────────────────────────────────────
    final_chunks = _rerank(query, unique_chunks, top_n=top_k)

    print(f"Final chunks after reranking: {len(final_chunks)}")
    return final_chunks


def retrieve_with_scores(query, top_k=5, doc_id=None):
    """Same as retrieve() but also returns TF-IDF ranked doc list."""
    all_queries = _expand_query(query)
    tfidf_doc_scores = {}
    for q in all_queries:
        for did, score in search_index(q, top_k=top_k, doc_id_filter=doc_id):
            tfidf_doc_scores[did] = max(tfidf_doc_scores.get(did, 0), score)

    tfidf_results = sorted(tfidf_doc_scores.items(), key=lambda x: x[1], reverse=True)

    if not tfidf_results:
        return _parent_vector_search(query, top_k * 2, doc_id), []

    top_doc_ids = [d for d, _ in tfidf_results[:3]]
    candidate_chunks = []
    for did in top_doc_ids:
        candidate_chunks.extend(_small_to_large(query, top_k=4, doc_id=did))

    seen, unique = set(), []
    for ch in candidate_chunks:
        if ch[:80] not in seen:
            seen.add(ch[:80]); unique.append(ch)

    return _rerank(query, unique, top_n=top_k), tfidf_results


def retrieve_selective(query, doc_ids, top_k_per_doc=3):
    """Retrieve from a user-specified subset of documents."""
    if not doc_ids:
        return []
    print(f"\nSelective retrieval: {len(doc_ids)} docs | '{query}'")
    all_queries = _expand_query(query)

    tfidf_doc_scores = {}
    for q in all_queries:
        for did in doc_ids:
            for rdid, score in search_index(q, top_k=top_k_per_doc, doc_id_filter=did):
                tfidf_doc_scores[rdid] = max(tfidf_doc_scores.get(rdid, 0), score)

    if not tfidf_doc_scores:
        return _vector_search_multiple(query, doc_ids, top_k=top_k_per_doc * len(doc_ids))

    top_doc_ids = sorted(tfidf_doc_scores, key=tfidf_doc_scores.get, reverse=True)[:3]
    candidate_chunks = []
    for did in top_doc_ids:
        candidate_chunks.extend(_small_to_large(query, top_k=top_k_per_doc, doc_id=did))

    seen, unique = set(), []
    for ch in candidate_chunks:
        if ch[:80] not in seen:
            seen.add(ch[:80]); unique.append(ch)

    return _rerank(query, unique, top_n=top_k_per_doc * 2)


def retrieve_bm25(query, top_k=5, doc_id=None):
    """BM25-based retrieval for comparison tab."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return retrieve_with_scores(query, top_k, doc_id)

    query_tokens = tokenize(query)
    if not query_tokens:
        return [], []

    conn = get_conn(); c = conn.cursor()
    rows = c.execute("SELECT doc_id FROM documents" +
                     (" WHERE doc_id=?" if doc_id else ""),
                     (doc_id,) if doc_id else ()).fetchall()
    if not rows:
        conn.close(); return [], []

    all_doc_ids = [r[0] for r in rows]
    import json as _j
    corpus = []
    for did in all_doc_ids:
        terms = c.execute("SELECT term, positions FROM index_term WHERE doc_id=?",
                          (did,)).fetchall()
        toks  = []
        for term, pos in terms:
            try: toks.extend([term] * len(_j.loads(pos)))
            except: toks.append(term)
        corpus.append(toks or [""])
    conn.close()

    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(all_doc_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    ranked = [(d, float(s)) for d, s in ranked if s > 0]
    if not ranked:
        return [], []

    all_chunks = []
    for did, _ in ranked[:3]:
        all_chunks.extend(_small_to_large(query, top_k=3, doc_id=did))

    seen, unique = set(), []
    for ch in all_chunks:
        if ch[:80] not in seen:
            seen.add(ch[:80]); unique.append(ch)

    return _rerank(query, unique, top_n=top_k), ranked


# ── Internal helpers ──────────────────────────────────────────────────────────

def _expand_query(query):
    """
    Query expansion — only runs for queries with 4+ words.
    Short queries (< 4 words) are specific enough and don't benefit from expansion.
    Timeout kept at 3s so a slow Groq response doesn't block retrieval.
    """
    words = query.split()
    if len(words) < 4:
        return [query]   # skip expansion for short queries

    try:
        from llm import chat_text, GEN_MODEL
        text = chat_text([{"role": "user", "content":
            f"""Generate exactly 2 alternative search queries for finding this information in a document.
Return only the 2 queries, one per line, no numbering, no explanation.

Original query: {query}

Alternative queries:"""}],
            model=GEN_MODEL, temperature=0.4, max_tokens=60, timeout=4, retries=1)
        lines = [l.strip() for l in text.split("\n")
                 if l.strip() and l.strip().lower() != query.lower()]
        return [query] + lines[:2]
    except Exception:
        return [query]


def _small_to_large(query, top_k=4, doc_id=None):
    """
    Parent-child retrieval:
    1. Query small_col for precise child matches
    2. Collect parent_ids from metadata
    3. Fetch those parent chunks from large_col
    4. Return large chunks (richer context for LLM)
    """
    try:
        qe    = model.encode([query]).tolist()
        where = {"$and": [{"doc_id": {"$eq": doc_id}},
                          {"chunk_type": {"$eq": "small"}}]} if doc_id else \
                {"chunk_type": {"$eq": "small"}}

        # retrieve more small chunks to get diverse parents
        results  = small_col.query(query_embeddings=qe, n_results=top_k * 2, where=where)
        metas    = results["metadatas"][0]

        # collect unique parent IDs
        parent_ids = []
        seen_pids  = set()
        for m in metas:
            pid = m.get("parent_id")
            if pid and pid not in seen_pids:
                seen_pids.add(pid); parent_ids.append(pid)

        if not parent_ids:
            return _direct_large_search(query, top_k, doc_id)

        # fetch parent large chunks
        parent_res = large_col.get(ids=parent_ids)
        chunks     = parent_res.get("documents", [])
        return [c for c in chunks if c and c.strip()]

    except Exception as e:
        print(f"Parent-child retrieval error: {e}")
        return _direct_large_search(query, top_k, doc_id)


def _direct_large_search(query, top_k=5, doc_id=None):
    """Direct vector search on large chunks — fallback when small_col unavailable."""
    try:
        qe    = model.encode([query]).tolist()
        where = {"doc_id": doc_id} if doc_id else None
        res   = large_col.query(query_embeddings=qe, n_results=top_k, where=where)
        return [c for c in res["documents"][0] if c and c.strip()]
    except Exception:
        return _legacy_vector_search(query, top_k, doc_id)


def _legacy_vector_search(query, top_k=5, doc_id=None):
    """Fallback to old 'pages' collection for backward compatibility."""
    try:
        qe    = model.encode([query]).tolist()
        where = {"doc_id": doc_id} if doc_id else None
        res   = collection.query(query_embeddings=qe, n_results=top_k, where=where)
        return res["documents"][0]
    except Exception as e:
        print(f"Legacy vector search error: {e}"); return []


def _parent_vector_search(query, top_k=8, doc_id=None):
    """Pure large-chunk vector search — used as complete fallback."""
    chunks = _direct_large_search(query, top_k, doc_id)
    return _rerank(query, chunks, top_n=min(5, len(chunks)))


def _vector_search_multiple(query, doc_ids, top_k=9):
    """Vector search across multiple specified documents."""
    try:
        qe  = model.encode([query]).tolist()
        res = large_col.query(
            query_embeddings = qe,
            n_results        = top_k,
            where            = {"doc_id": {"$in": list(doc_ids)}}
        )
        return res["documents"][0] if res["documents"] else []
    except Exception:
        all_chunks = []
        for did in doc_ids:
            all_chunks.extend(_direct_large_search(query, top_k // len(doc_ids) + 1, did))
        return all_chunks


def _rerank(query, chunks, top_n=5):
    """
    Cross-encoder reranking.
    Scores every (query, chunk) pair and returns top_n by score.
    If the reranker is not available, returns chunks unchanged.
    """
    if not chunks:
        return chunks

    reranker = _get_reranker()
    if reranker is None:
        return chunks[:top_n]

    try:
        pairs  = [(query, ch) for ch in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        result = [ch for ch, _ in ranked[:top_n]]
        print(f"Reranker kept {len(result)}/{len(chunks)} chunks "
              f"(top score: {ranked[0][1]:.3f})")
        return result
    except Exception as e:
        print(f"Reranking error: {e} — using unreranked chunks")
        return chunks[:top_n]