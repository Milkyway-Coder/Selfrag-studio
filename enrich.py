"""
enrich.py — automatic summary, named entities and tags for every document.

One LLM call per document (not three) over a spread of passages taken from the whole file, so a long PDF is
summarised from its beginning, middle and end rather than from whatever a search query happened to retrieve.
Runs right after indexing; failures never block an upload (the Library offers a Regenerate button).
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from indexer import large_col
from page_index import set_enrichment, get_enrichment
from database import add_doc_tag
from llm import chat_text, GEN_MODEL

_pool = ThreadPoolExecutor(max_workers=1)          # one at a time: avoids bursts against the rate limit
ENTITY_KEYS = ["people", "organizations", "locations", "dates", "technologies", "key_terms"]


def sample_passages(doc_id, n=8, chars=900):
    res = large_col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    items = sorted(zip(res["ids"], res["documents"]), key=lambda x: int(x[0].rsplit("_", 1)[-1]) if x[0].rsplit("_", 1)[-1].isdigit() else 0)
    if not items:
        return []
    if len(items) > n:                                   # evenly spaced across the whole document
        step = (len(items) - 1) / (n - 1)
        items = [items[round(i * step)] for i in range(n)]
    return [t.strip()[:chars] for _, t in items if t and t.strip()]


def _parse(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    a, b = raw.find("{"), raw.rfind("}") + 1
    if a < 0 or b <= a:
        raise ValueError("no JSON in the model reply")
    return json.loads(raw[a:b])


def _clean_tags(tags):
    out = []
    for t in tags or []:
        t = re.sub(r"[^a-z0-9+#. -]", "", str(t).lower()).strip()
        if 1 < len(t) <= 28 and len(t.split()) <= 3 and t not in out:
            out.append(t)
    return out[:6]


def enrich_document(doc_id):
    passages = sample_passages(doc_id)
    if not passages:
        raise ValueError("The document has no indexed text.")
    context = "\n\n---\n\n".join(passages)
    prompt = f"""Read these passages sampled from across one document and reply with ONLY a JSON object:

{{"summary": "<90 to 130 words of plain prose describing what the document is about and its main points>",
  "tags": ["<3 to 6 short lowercase topic tags, one to three words each>"],
  "entities": {{"people": [], "organizations": [], "locations": [], "dates": [], "technologies": [], "key_terms": []}}}}

Rules: use only what the passages say; list at most 10 items per entity group; leave a group empty if nothing fits;
no markdown inside the values.

PASSAGES:
{context}"""
    data = _parse(chat_text([{"role": "user", "content": prompt}], model=GEN_MODEL, temperature=0.0, timeout=60, max_tokens=900))
    summary = re.sub(r"\s+", " ", str(data.get("summary", ""))).strip()
    tags = _clean_tags(data.get("tags"))
    ents = {k: [str(x).strip() for x in (data.get("entities", {}).get(k) or []) if str(x).strip()][:10] for k in ENTITY_KEYS}
    if not summary:
        raise ValueError("The model returned no summary.")
    set_enrichment(doc_id, summary, ents, tags)
    for t in tags:
        add_doc_tag(doc_id, t)
    return get_enrichment(doc_id)


def enrich_async(doc_id):
    """Queue enrichment in the background; errors are printed, never raised."""
    def run():
        try:
            enrich_document(doc_id)
        except Exception as e:                           # noqa: BLE001
            print(f"Enrichment skipped for '{doc_id}': {e}")
    _pool.submit(run)
