"""
multimodal.py — source citations and related figures for an answer.

sources : where each retrieved passage came from (document, page, table/figure/code, file and line range)
figures : figures worth showing next to the answer, chosen by four signals (strongest first)
          1. the question names a figure ("explain Figure 3")
          2. a figure chunk was itself retrieved as evidence
          3. the question has visual intent ("diagram", "chart", "architecture"...) and a figure matches it
          4. visual intent plus a figure on a page whose text was used as evidence
"""
import re
from indexer import model, large_col, meta_for_texts

VISUAL_RE = re.compile(
    r"\b(figure|fig\.?|diagram|chart|graph|plot|image|picture|illustrat\w*|architecture|flow ?chart|"
    r"schematic|screenshot|visuali[sz]\w*|curve|histogram|block diagram|show me)\b", re.I)
FIGNUM_RE = re.compile(r"\b(?:figure|fig\.?)\s*(\d+[A-Za-z]?)\b", re.I)

_TYPE_LABEL = {"table": "table", "figure": "figure description", "vision": "page transcription (formulas, charts)",
               "sheet": "spreadsheet statistics"}


def passage_label(m):
    if not m:
        return ""
    t = m.get("block_type", "text")
    parts = []
    if t == "code":
        parts.append("code")
        if m.get("file"):
            parts.append(m["file"] + (f" lines {m['start']}-{m['end']}" if m.get("start") else ""))
    else:
        if t in _TYPE_LABEL:
            parts.append(_TYPE_LABEL[t])
        if m.get("file") and t in ("table", "sheet"):
            parts.append(f"sheet {m['file']}")
    if m.get("page", -1) > 0:
        pe = m.get("page_end", -1)
        parts.append(f"page {m['page']}" + (f"-{pe}" if pe and pe > m["page"] else ""))
    return " | ".join(parts)


def _chip(m):
    """Short label for the source chips under an answer."""
    t = m.get("block_type", "text") if m else "text"
    pg = f"p.{m['page']}" if m and m.get("page", -1) > 0 else ""
    if t == "code":
        return " ".join(x for x in [m.get("file", ""), f"lines {m['start']}-{m['end']}" if m.get("start") else ""] if x) or "code"
    name = {"table": "Table", "figure": "Figure", "vision": "Page (vision)", "sheet": "Sheet"}.get(t, "")
    if t in ("table", "sheet") and m.get("file"):
        name = f"{name} {m['file']}"
    return " · ".join(x for x in [name, pg] if x) or "Passage"


def build_sources(chunks, metas):
    out = []
    for i, (c, m) in enumerate(zip(chunks, metas), 1):
        m = m or {}
        out.append({
            "n": i, "doc_id": m.get("doc_id", ""), "type": m.get("block_type", "text"),
            "page": m["page"] if m.get("page", -1) > 0 else None,
            "file": m.get("file") or None,
            "lines": f"{m['start']}-{m['end']}" if m.get("start") else None,
            "label": passage_label(m).replace(" | ", " · "),
            "chip": _chip(m),
            "snippet": re.sub(r"\s+", " ", c)[:240],
        })
    return out


def _doc_filter(doc_ids):
    if not doc_ids:
        return None
    ids = list(doc_ids)
    return {"doc_id": ids[0]} if len(ids) == 1 else {"doc_id": {"$in": ids}}


def _fig_where(doc_ids, *extra):
    conds = [{"block_type": "figure"}] + list(extra)
    f = _doc_filter(doc_ids)
    if f:
        conds.append(f)
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _fig(m, reason):
    from ingest import display_caption
    return {"doc_id": m.get("doc_id"), "fig_id": m.get("fig_id"), "url": m.get("fig_url"),
            "label": m.get("fig_label") or None, "page": m["page"] if m.get("page", -1) > 0 else None,
            "caption": display_caption(m.get("caption") or ""), "reason": reason}


ONLY_FIG_RE = re.compile(
    r"\b(only|just)\b.{0,50}\b(diagram|figure|fig|chart|graph|image|picture|map|illustration|schematic)\b"
    r"|^\s*(please\s+)?(show|display|give|send|get|fetch|find|share)\b.{0,20}\b(the\s+|a\s+)?(diagram|figure|image|picture|chart|graph|map|illustration)\b", re.I)


def wants_figure_only(query):
    """'Give me only the hydrological cycle diagram' wants the picture, not an essay about it."""
    return bool(ONLY_FIG_RE.search(query or ""))


def _words(text):
    from page_index import tokenize
    return set(tokenize(text or ""))


def related_figures(query, doc_ids, metas, max_figs=3, strict=False):
    """
    Figures worth showing for this question, best first.
    Candidates come from four signals (named in the question, used as evidence, visual intent, same page as
    evidence) and are RANKED by embedding similarity plus word overlap with the figure's description, then
    cut at a relative threshold so a diagram question does not drag in every other picture in the document.
    strict=True (a "show me only the diagram" request) keeps just the best match, plus a close runner-up.
    """
    cand, order = {}, []

    def add(m, doc_text, reason, sim=None, bonus=0.0):
        key = (m.get("doc_id"), m.get("fig_id"))
        if not m.get("fig_url"):
            return
        if key not in cand:
            cand[key] = {"m": m, "text": doc_text or "", "reason": reason, "sim": 0.0, "bonus": 0.0}; order.append(key)
        c = cand[key]
        if sim is not None:
            c["sim"] = max(c["sim"], sim)
        if bonus > c["bonus"]:
            c["bonus"], c["reason"] = bonus, reason

    try:
        labels = list(dict.fromkeys(FIGNUM_RE.findall(query)))
        for label in labels:
            res = large_col.get(where=_fig_where(doc_ids, {"fig_label": label}), include=["metadatas", "documents"])
            for m, d in zip(res["metadatas"], res["documents"]):
                add(m, d, f"named in your question (Figure {label})", bonus=10.0)
        if labels and order:                              # the question names a figure: show exactly that
            return [_fig(cand[k]["m"], cand[k]["reason"]) for k in order][:max_figs]

        evidence_pages = {(m["doc_id"], m["page"]) for m in (metas or []) if m and m.get("page", -1) > 0}
        visual = bool(VISUAL_RE.search(query)) or strict
        qe = model.encode([query], show_progress_bar=False).tolist()
        n_figs = large_col.get(where=_fig_where(doc_ids), include=[])["ids"]
        if not n_figs:
            return []
        res = large_col.query(query_embeddings=qe, n_results=min(8, len(n_figs)), where=_fig_where(doc_ids),
                              include=["metadatas", "documents", "distances"])
        for m, d, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
            sim = max(0.0, 1.0 - float(dist) ** 2 / 2.0)          # unit vectors: cosine = 1 - d^2/2
            on_page = (m.get("doc_id"), m.get("page")) in evidence_pages
            retrieved = any(x and x.get("fig_id") == m.get("fig_id") and x.get("doc_id") == m.get("doc_id") for x in (metas or []))
            if visual or retrieved or on_page:
                add(m, d, "used as evidence" if retrieved else ("matches your question" if visual else f"on page {m.get('page')}, which was used as evidence"),
                    sim=sim, bonus=0.15 if retrieved else (0.05 if on_page else 0.0))
    except Exception as e:                               # no figures indexed, or an older index without metadata
        print(f"related_figures: {e}")
        return []

    q_words = _words(query) - {"diagram", "figure", "image", "picture", "chart", "graph", "show", "explain", "give", "only", "help"}
    scored = []
    for key in order:
        c = cand[key]
        overlap = len(q_words & _words(c["text"])) / max(len(q_words), 1) if q_words else 0.0
        scored.append((c["sim"] + 0.5 * overlap + c["bonus"], c["sim"], overlap, key))
    scored.sort(reverse=True)
    if not scored:
        return []
    best = scored[0][0]
    keep = []
    for score, sim, overlap, key in scored:
        if sim < 0.22 and overlap == 0 and cand[key]["bonus"] < 0.1:
            continue                                      # unrelated picture that merely sits near the evidence
        if score >= best - (0.06 if strict else 0.12):
            keep.append(key)
    if strict:
        keep = keep[:2]
    return [_fig(cand[k]["m"], cand[k]["reason"]) for k in keep][:max_figs]
