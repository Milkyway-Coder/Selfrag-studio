import math
import re
import json
import sqlite3
from collections import defaultdict


# ─── Tokenizer ────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "are", "were",
    "be", "been", "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "i", "we", "you", "he", "she", "they", "what", "which", "who",
    "not", "no", "so", "if", "as", "up", "out", "about", "into", "than",
    "then", "when", "there", "their", "they", "also", "just", "more", "over"
}

def tokenize(text):
    """
    Converts raw text into a clean list of tokens.
    Steps: lowercase → remove punctuation → split → remove stopwords
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ─── Database setup ───────────────────────────────────────────────────────────

DB_PATH = "page_index.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    return conn

def init_page_index_db():
    """
    Creates 3 tables:
    - documents  : stores each indexed doc (id, name, word count)
    - index_term : the inverted index (term → doc_id, positions, tf score)
    - idf_scores : stores IDF score per term (updated after every new doc)
    """
    conn = get_conn()
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id     TEXT PRIMARY KEY,
            word_count INTEGER,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS index_term (
            term       TEXT,
            doc_id     TEXT,
            tf         REAL,
            positions  TEXT,
            PRIMARY KEY (term, doc_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS idf_scores (
            term TEXT PRIMARY KEY,
            idf  REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS doc_meta (
            doc_id  TEXT PRIMARY KEY,
            kind    TEXT,
            stats   TEXT,
            source  TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS doc_enrich (
            doc_id   TEXT PRIMARY KEY,
            summary  TEXT,
            entities TEXT,
            tags     TEXT,
            created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("Page index DB ready.")


# ─── Indexing ─────────────────────────────────────────────────────────────────

def index_page(doc_id, text):
    """
    Full indexing pipeline for one document:
    1. Tokenize text
    2. Compute TF per term
    3. Store in inverted index with positions
    4. Recompute IDF for all affected terms
    """
    tokens = tokenize(text)
    if not tokens:
        print(f"No tokens found for: {doc_id}")
        return

    total_tokens = len(tokens)

    # build term → positions mapping
    positions = defaultdict(list)
    for pos, token in enumerate(tokens):
        positions[token].append(pos)

    # compute TF = (occurrences of term) / (total tokens in doc)
    tf_scores = {
        term: len(pos_list) / total_tokens
        for term, pos_list in positions.items()
    }

    conn = get_conn()
    c    = conn.cursor()

    # remove old entries for this doc if re-indexing
    c.execute("DELETE FROM documents  WHERE doc_id = ?", (doc_id,))
    c.execute("DELETE FROM index_term WHERE doc_id = ?", (doc_id,))

    # insert document record
    c.execute(
        "INSERT INTO documents (doc_id, word_count) VALUES (?, ?)",
        (doc_id, total_tokens)
    )

    # insert inverted index entries
    for term, pos_list in positions.items():
        c.execute(
            "INSERT INTO index_term (term, doc_id, tf, positions) VALUES (?, ?, ?, ?)",
            (term, doc_id, tf_scores[term], json.dumps(pos_list))
        )

    conn.commit()

    # recompute IDF for every term in this document
    _recompute_idf(c, conn)

    conn.close()
    print(f"Page indexed: '{doc_id}' — {total_tokens} tokens, {len(positions)} unique terms")


def _recompute_idf(cursor, conn, terms=None):
    """
    Smoothed IDF = log((N + 1) / (df + 1)) + 1, recomputed for the WHOLE vocabulary.
    N changes whenever a document is added or removed, so every term's IDF changes,
    not only the terms of the document that was just touched.
    """
    total_docs = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    rows = cursor.execute(
        "SELECT term, COUNT(DISTINCT doc_id) FROM index_term GROUP BY term"
    ).fetchall()
    cursor.execute("DELETE FROM idf_scores")
    cursor.executemany(
        "INSERT INTO idf_scores (term, idf) VALUES (?, ?)",
        [(t, math.log((total_docs + 1) / (df + 1)) + 1) for t, df in rows]
    )
    conn.commit()


def delete_page(doc_id):
    """Removes a document from the inverted index and refreshes IDF for the rest."""
    conn = get_conn()
    c    = conn.cursor()
    c.execute("DELETE FROM documents  WHERE doc_id = ?", (doc_id,))
    c.execute("DELETE FROM index_term WHERE doc_id = ?", (doc_id,))
    c.execute("DELETE FROM doc_meta   WHERE doc_id = ?", (doc_id,))
    c.execute("DELETE FROM doc_enrich WHERE doc_id = ?", (doc_id,))
    conn.commit()
    _recompute_idf(c, conn)
    conn.close()


def set_doc_meta(doc_id, kind, stats=None, source=""):
    """Remembers what kind of document this is (pdf, docx, sheet, code...) and its content counts."""
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO doc_meta (doc_id, kind, stats, source) VALUES (?, ?, ?, ?)",
                 (doc_id, kind, json.dumps(stats or {}), source or ""))
    conn.commit()
    conn.close()


def set_enrichment(doc_id, summary, entities, tags):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO doc_enrich (doc_id, summary, entities, tags) VALUES (?, ?, ?, ?)",
                 (doc_id, summary or "", json.dumps(entities or {}), json.dumps(tags or [])))
    conn.commit()
    conn.close()


def get_enrichment(doc_id):
    conn = get_conn()
    row = conn.execute("SELECT summary, entities, tags FROM doc_enrich WHERE doc_id = ?", (doc_id,)).fetchone()
    conn.close()
    if not row:
        return None
    try:
        return {"summary": row[0], "entities": json.loads(row[1] or "{}"), "tags": json.loads(row[2] or "[]")}
    except Exception:
        return {"summary": row[0], "entities": {}, "tags": []}


def get_all_doc_meta():
    conn = get_conn()
    rows = conn.execute("SELECT doc_id, kind, stats, source FROM doc_meta").fetchall()
    conn.close()
    out = {}
    for d, k, s, src in rows:
        try:
            out[d] = {"kind": k, "stats": json.loads(s or "{}"), "source": src}
        except Exception:
            out[d] = {"kind": k, "stats": {}, "source": src}
    return out


# ─── Retrieval ────────────────────────────────────────────────────────────────

def search_index(query, top_k=5, doc_id_filter=None):
    """
    TF-IDF search against the inverted index.

    Steps:
    1. Tokenize the query
    2. For each query token, look up which docs contain it
    3. Score each doc: sum of (TF * IDF) for all matching query terms
    4. Rank docs by score, return top_k
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    print(f"\nSearching index for: '{query}' → tokens: {query_tokens}")
    if doc_id_filter:
        print(f"Filtered to doc: {doc_id_filter}")

    conn   = get_conn()
    c      = conn.cursor()
    scores = defaultdict(float)

    for token in query_tokens:
        row = c.execute(
            "SELECT idf FROM idf_scores WHERE term = ?",
            (token,)
        ).fetchone()

        if not row:
            continue

        idf = row[0]

        if doc_id_filter:
            rows = c.execute(
                "SELECT doc_id, tf FROM index_term WHERE term = ? AND doc_id = ?",
                (token, doc_id_filter)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT doc_id, tf FROM index_term WHERE term = ?",
                (token,)
            ).fetchall()

        for doc_id, tf in rows:
            scores[doc_id] += tf * idf

    conn.close()

    if not scores:
        print("No matching documents found.")
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top results: {ranked}")
    return ranked  # list of (doc_id, score)


def get_doc_snippet(doc_id, query, snippet_len=300):
    """
    The inverted index stores positions, not text, so a real snippet has to come from the
    vector store. retriever.py handles that fallback; this stub only reports matched terms.
    """
    tokens = tokenize(query)
    return f"[From '{doc_id}'] matched terms: {', '.join(tokens)}"


def get_all_indexed_docs():
    """Returns list of all doc_ids in the page index."""
    conn  = get_conn()
    c     = conn.cursor()
    rows  = c.execute(
        "SELECT doc_id, word_count FROM documents ORDER BY indexed_at DESC"
    ).fetchall()
    conn.close()
    return [{"doc_id": r[0], "word_count": r[1]} for r in rows]


# ─── Change Detection ─────────────────────────────────────────────────────────

def get_term_snapshot(doc_id):
    """
    Returns a {term: {tf, idf, tfidf}} snapshot for a document.
    Call this BEFORE re-indexing to capture the old state.
    """
    conn = get_conn()
    c    = conn.cursor()
    rows = c.execute("""
        SELECT it.term, it.tf, COALESCE(idf.idf, 1.0) AS idf,
               it.tf * COALESCE(idf.idf, 1.0) AS tfidf
        FROM index_term it
        LEFT JOIN idf_scores idf ON it.term = idf.term
        WHERE it.doc_id = ?
    """, (doc_id,)).fetchall()
    conn.close()
    return {
        r[0]: {"tf": round(r[1], 5), "idf": round(r[2], 4), "tfidf": round(r[3], 5)}
        for r in rows
    }


def compute_term_diff(old_snap, new_snap, top_n=20):
    """
    Compares two term snapshots.
    Returns categorised diff: added / removed / changed terms + summary stats.
    """
    old_set = set(old_snap)
    new_set = set(new_snap)
    added   = new_set - old_set
    removed = old_set - new_set
    changed = {}

    for term in old_set & new_set:
        delta = new_snap[term]["tfidf"] - old_snap[term]["tfidf"]
        if abs(delta) > 0.0005:
            changed[term] = {
                "old"  : round(old_snap[term]["tfidf"], 5),
                "new"  : round(new_snap[term]["tfidf"], 5),
                "delta": round(delta, 5),
            }

    return {
        "added": sorted(
            [{"term": t, "tfidf": new_snap[t]["tfidf"]} for t in added],
            key=lambda x: x["tfidf"], reverse=True
        )[:top_n],
        "removed": sorted(
            [{"term": t, "tfidf": old_snap[t]["tfidf"]} for t in removed],
            key=lambda x: x["tfidf"], reverse=True
        )[:top_n],
        "changed": sorted(
            [{"term": t, **v} for t, v in changed.items()],
            key=lambda x: abs(x["delta"]), reverse=True
        )[:top_n],
        "stats": {
            "old_vocab"    : len(old_set),
            "new_vocab"    : len(new_set),
            "added_count"  : len(added),
            "removed_count": len(removed),
            "changed_count": len(changed),
        }
    }