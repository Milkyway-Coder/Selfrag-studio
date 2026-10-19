"""
indexer.py — Document chunking, embedding, and dual indexing.

Parent-child chunking strategy:
  - SMALL chunks (~100 words) → stored in ChromaDB collection "chunks_small"
    Used for precise vector retrieval (small = more specific matches)
  - LARGE chunks (~400 words) → stored in ChromaDB collection "chunks_large"
    Passed to the LLM as context (large = richer explanation)

Each small chunk stores the ID of its parent large chunk.
Retrieval queries the small collection, then fetches the large parent chunks
to give the LLM full context around each matched passage.
"""

from sentence_transformers import SentenceTransformer
import chromadb
import fitz
import re
import os
import hashlib
from collections import Counter
from page_index import index_page, init_page_index_db, delete_page, set_doc_meta

init_page_index_db()

model     = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
chroma    = chromadb.PersistentClient(path="./chroma_store")

# Two collections — small for retrieval precision, large for LLM context richness
small_col = chroma.get_or_create_collection("chunks_small")   # ~100-word child chunks
large_col = chroma.get_or_create_collection("chunks_large")   # ~400-word parent chunks

# Keep backward compatibility — the old "pages" collection still used if present
try:
    collection = chroma.get_or_create_collection("pages")
except Exception:
    collection = small_col


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_sentences(text):
    """Split text into sentences using punctuation rules."""
    text    = re.sub(r'\r\n', '\n', text)
    text    = re.sub(r'\n{3,}', '\n\n', text)
    sent_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\(])')
    paras   = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    sents   = []
    for para in paras:
        sents.extend([s.strip() for s in sent_re.split(para) if s.strip()])
        sents.append("")   # paragraph boundary
    return sents


def chunk_semantic(text, max_words, overlap_sents=2):
    """Generic semantic chunker at any size."""
    sents = _split_sentences(text)
    chunks, current, cur_words = [], [], 0
    for sent in sents:
        if sent == "":
            if cur_words >= max_words // 2 and current:
                blob = " ".join(s for s in current if s)
                if len(blob.split()) >= 8:
                    chunks.append(blob)
                current   = current[-overlap_sents:]
                cur_words = sum(len(s.split()) for s in current)
            continue
        sw = len(sent.split())
        if cur_words + sw > max_words and current:
            blob = " ".join(s for s in current if s)
            if len(blob.split()) >= 8:
                chunks.append(blob)
            current   = current[-overlap_sents:]
            cur_words = sum(len(s.split()) for s in current)
        current.append(sent)
        cur_words += sw
    if current:
        blob = " ".join(s for s in current if s)
        if len(blob.split()) >= 8:
            chunks.append(blob)
    return chunks


def chunk_text(text, size=300, overlap=50):
    """Fixed-size fallback chunker."""
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


# ── Deletion ──────────────────────────────────────────────────────────────────

def _purge_vectors(doc_id):
    """Removes every vector row for doc_id from all three collections."""
    removed = 0
    for col in (small_col, large_col, collection):
        try:
            ex = col.get(where={"doc_id": doc_id}, include=[])
            if ex["ids"]:
                col.delete(ids=ex["ids"])
                removed += len(ex["ids"])
        except Exception as e:
            print(f"Vector purge warning ({col.name}): {e}")
    return removed


def delete_document(doc_id):
    """
    Fully removes a document: small chunks, large chunks, legacy collection AND the
    SQLite inverted index (with IDF refresh). The original delete route only touched
    the legacy collection, so deleted documents kept showing up in get_indexed_docs().
    """
    removed = _purge_vectors(doc_id)
    delete_page(doc_id)
    try:
        import ingest
        ingest.clear_artifacts(doc_id)
    except Exception as e:
        print(f"Artifact cleanup warning: {e}")
    print(f"Deleted '{doc_id}' ({removed} vector rows).")
    return removed


# ── Indexing ──────────────────────────────────────────────────────────────────

def chunk_hash(text):
    """Stable key for looking a retrieved chunk's metadata back up (see meta_for_texts)."""
    return hashlib.md5(text.strip()[:200].encode("utf-8", "ignore")).hexdigest()


def _meta(doc_id, ctype, btype, page, page_end, extra, text, parent_id=None, parent_idx=None):
    m = {"doc_id": doc_id, "chunk_type": ctype, "block_type": btype,
         "page": int(page) if page else -1,
         "page_end": int(page_end) if page_end else (int(page) if page else -1),
         "file": "", "lang": "", "fig_id": "", "fig_label": "", "fig_url": "", "h": chunk_hash(text)}
    for k in ("file", "lang", "fig_id", "fig_label", "fig_url"):
        if extra and extra.get(k):
            m[k] = str(extra[k])
    if extra and extra.get("caption"):
        m["caption"] = str(extra["caption"])[:200]
    if extra and extra.get("sheet"):
        m["file"] = str(extra["sheet"])
    if extra and extra.get("start"):
        m["start"] = int(extra["start"]); m["end"] = int(extra.get("end") or extra["start"])
    if parent_id:
        m["parent_id"] = parent_id; m["parent_idx"] = parent_idx
    return m


def _split_table(md, max_words=380):
    """Splits a long Markdown table into pieces, repeating the header row in each."""
    if len(md.split()) <= max_words:
        return [md]
    lines = md.split("\n")
    idx = next((i for i, l in enumerate(lines) if l.startswith("|")), 0)
    head, body = lines[:idx + 2], lines[idx + 2:]
    out, cur, w = [], [], 0
    for l in body:
        lw = len(l.split())
        if cur and w + lw > max_words:
            out.append("\n".join(head + cur)); cur, w = [], 0
        cur.append(l); w += lw
    if cur or not out:
        out.append("\n".join(head + cur))
    return out


def _code_children(text, window=25, overlap=3):
    lines = text.split("\n")
    if len(lines) <= window + 5:
        return [text]
    header = lines[0] if lines and lines[0].startswith(("File:", "Notebook")) else ""
    body = lines[1:] if header else lines
    out, i = [], 0
    while i < len(body):
        part = body[i:i + window]
        out.append((header + "\n" if header else "") + "\n".join(part))
        if i + window >= len(body):
            break
        i += window - overlap
    return out


def _add_batched(col, ids, documents, embeddings, metadatas, batch=400):
    for s in range(0, len(ids), batch):
        col.add(ids=ids[s:s + batch], documents=documents[s:s + batch],
                embeddings=embeddings[s:s + batch], metadatas=metadatas[s:s + batch])


def index_blocks(doc_id, blocks, kind="text", stats=None, source=""):
    """
    Typed-block indexing. Prose is grouped into ~400-word parents (with ~100-word children for precise
    search). Tables, figures, formulas, spreadsheet rows and code keep their own chunks and their
    metadata (page, file, line range, figure id), so answers can cite them and show the figure.
    """
    larges, buf, buf_words, buf_pages = [], [], 0, []

    def flush():
        nonlocal buf, buf_words, buf_pages
        if not buf:
            return
        pg = [x for x in buf_pages if x]
        larges.append(("\n\n".join(buf), "text", min(pg) if pg else None, max(pg) if pg else None, {}))
        buf, buf_words, buf_pages = [], 0, []

    for b in blocks:
        t = (b.get("text") or "").strip()
        if not t:
            continue
        typ = b.get("type", "text")
        if typ == "text":
            w = len(t.split())
            if w > 400:
                flush()
                for part in (chunk_semantic(t, 400, 3) or chunk_text(t, 400, 60)):
                    larges.append((part, "text", b.get("page"), b.get("page"), {}))
                continue
            if buf and buf_words + w > 400:
                flush()
            buf.append(t); buf_words += w; buf_pages.append(b.get("page"))
        else:
            flush()
            for part in (_split_table(t) if typ in ("table", "sheet") else [t]):
                larges.append((part, typ, b.get("page"), b.get("page"), b.get("meta") or {}))
    flush()
    if not larges:
        print(f"No chunks for doc: {doc_id}")
        return 0

    smalls = []
    for li, (text, btype, pg, pe, extra) in enumerate(larges):
        if btype == "text":
            kids = chunk_semantic(text, 100, 1) or chunk_text(text, 100, 15)
        elif btype == "code" and not extra.get("outline"):
            kids = _code_children(text)
        else:
            kids = [text]
        for k in kids:
            smalls.append((k, li, btype, pg, pe, extra))

    print(f"Indexing '{doc_id}': {len(larges)} large chunks, {len(smalls)} small chunks...")
    _purge_vectors(doc_id)

    l_texts = [x[0] for x in larges]
    l_embs  = model.encode(l_texts, batch_size=32, show_progress_bar=False).tolist()
    l_ids   = [f"{doc_id}_large_{i}" for i in range(len(larges))]
    l_metas = [_meta(doc_id, "large", x[1], x[2], x[3], x[4], x[0]) for x in larges]
    _add_batched(large_col, l_ids, l_texts, l_embs, l_metas)

    s_texts = [x[0] for x in smalls]
    s_embs  = model.encode(s_texts, batch_size=32, show_progress_bar=False).tolist()
    s_ids   = [f"{doc_id}_small_{i}" for i in range(len(smalls))]
    s_metas = [_meta(doc_id, "small", x[2], x[3], x[4], x[5], x[0], f"{doc_id}_large_{x[1]}", x[1]) for x in smalls]
    _add_batched(small_col, s_ids, s_texts, s_embs, s_metas)

    try:                                    # legacy collection kept for backward compatibility
        _add_batched(collection, l_ids, l_texts, l_embs, l_metas)
    except Exception as e:
        print(f"Legacy collection warning: {e}")

    index_page(doc_id, "\n".join((b.get("text") or "") for b in blocks))
    counts = Counter(x[1] for x in larges)
    set_doc_meta(doc_id, kind, {**(stats or {}), "chunks": dict(counts)}, source)
    print(f"Done. '{doc_id}' indexed ({dict(counts)}).")
    return len(larges)


def index_document(doc_id, text):
    """Plain-text indexing (kept for the paste-text route and the tests)."""
    return index_blocks(doc_id, [{"type": "text", "text": text, "page": None, "meta": {}}], kind="text")


def meta_for_texts(texts, doc_ids=None):
    """Metadata (page, block type, file, figure) for retrieved chunk texts, aligned with the input."""
    if not texts:
        return []
    hs = [chunk_hash(t) for t in texts]
    try:
        res = large_col.get(where={"h": {"$in": list(set(hs))}}, include=["metadatas"])
    except Exception:
        return [None] * len(texts)
    by_h = {}
    for m in res.get("metadatas", []):
        if doc_ids and m.get("doc_id") not in doc_ids:
            by_h.setdefault(m["h"], by_h.get(m["h"]))
            continue
        by_h[m["h"]] = m
    return [by_h.get(h) for h in hs]


def list_figures(doc_id):
    from ingest import display_caption
    try:
        res = large_col.get(where={"$and": [{"doc_id": doc_id}, {"block_type": "figure"}]},
                            include=["metadatas", "documents"])
    except Exception:
        return []
    out = []
    for m, d in zip(res["metadatas"], res["documents"]):
        out.append({"doc_id": doc_id, "fig_id": m.get("fig_id"), "url": m.get("fig_url"), "label": m.get("fig_label"),
                    "page": m.get("page") if m.get("page", -1) > 0 else None,
                    "caption": display_caption(m.get("caption") or "") or d.split("\n")[0][:200]})
    return sorted(out, key=lambda f: f["fig_id"] or "")


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    doc        = fitz.open(pdf_path)
    full_text  = ""
    total_pages = len(doc)
    print(f"Extracting text from {total_pages} pages...")
    for i, page in enumerate(doc):
        full_text += f"\n[Page {i+1}]\n{page.get_text()}"
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total_pages} pages processed...")
    doc.close()
    word_count = len(full_text.split())
    print(f"Native extraction: {word_count} words.")

    # OCR fallback for scanned PDFs
    if word_count / max(total_pages, 1) < 20:
        print("Low text yield — attempting OCR...")
        ocr = _ocr_pdf(pdf_path)
        if ocr and len(ocr.split()) > word_count:
            return ocr
    return full_text


def _ocr_pdf(pdf_path):
    try:
        import pytesseract
        from PIL import Image
        import io
        doc = fitz.open(pdf_path); full = ""; n = len(doc)
        for i, page in enumerate(doc):
            mat = fitz.Matrix(300/72, 300/72)
            img = Image.open(io.BytesIO(page.get_pixmap(matrix=mat).tobytes("png")))
            full += f"\n[Page {i+1}]\n{pytesseract.image_to_string(img)}"
            if (i + 1) % 5 == 0: print(f"  OCR: {i+1}/{n} pages...")
        doc.close(); return full
    except ImportError:
        print("pytesseract not installed — skipping OCR.")
        return ""
    except Exception as e:
        print(f"OCR error: {e}"); return ""


def index_pdf(doc_id, pdf_path, vision_on=False, progress=None):
    """Structured PDF indexing: text, tables and figures (optionally a vision pass over every page)."""
    import ingest
    res = ingest.parse_file(pdf_path, doc_id, os.path.basename(pdf_path) or "file.pdf", vision_on, progress)
    index_blocks(doc_id, res["blocks"], res["kind"], res["stats"], os.path.basename(pdf_path))
    return res


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_indexed_docs():
    """Returns sorted list of all indexed doc_ids."""
    try:
        ids     = large_col.get(include=[]).get("ids", [])
        doc_ids = set()
        for cid in ids:
            # format: docid_large_N
            parts = cid.rsplit("_large_", 1)
            if len(parts) == 2:
                doc_ids.add(parts[0])
        return sorted(doc_ids)
    except Exception:
        return []


def get_partial_matches(query, doc_id=None, top_k=3):
    """Returns brief snippet previews for not-found responses."""
    try:
        qe    = model.encode([query]).tolist()
        where = {"doc_id": doc_id} if doc_id else None
        res   = large_col.query(query_embeddings=qe, n_results=top_k, where=where)
        return [c[:120].strip() for c in res["documents"][0] if c.strip()]
    except Exception:
        return []


def get_doc_chunk_counts(doc_id):
    """Returns (large_chunks, small_chunks) for a document."""
    try:
        l = len(large_col.get(where={"doc_id": doc_id}, include=[])["ids"])
        s = len(small_col.get(where={"doc_id": doc_id}, include=[])["ids"])
        return l, s
    except Exception:
        return 0, 0
