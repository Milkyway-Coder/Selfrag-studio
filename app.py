from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, send_from_directory
from dotenv import load_dotenv
from indexer import index_document, delete_document, get_doc_chunk_counts
import chitchat
import research
import conversation
import ingest
import tabular
import multimodal
import enrich
import quiz as quiz_mod
import html as _html
from retriever import retrieve, retrieve_selective
from generator import generate, generate_with_citations, reformulate_query, generate_stream
from evaluator import evaluate, detect_hallucination
from database import (init_db, init_sessions, init_chat_history, log_query,
                      save_chat_message, get_chat_history, clear_chat_history)
from page_index import get_all_indexed_docs, init_page_index_db
import os, re, traceback, json, time, threading, uuid, tempfile
from werkzeug.utils import secure_filename
from scraper import scrape_url

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)
init_db()
init_sessions()
init_chat_history()
init_page_index_db()

from database import migrate_schema, adopt_orphan_messages, get_all_chat_sessions, create_chat_session
migrate_schema()
adopt_orphan_messages()

# Pre-load cross-encoder in background thread (avoids httpx closed-client error)
from retriever import preload_reranker
preload_reranker()

active_doc_id     = None
_sessions_at_boot = get_all_chat_sessions()
active_session_id = _sessions_at_boot[0]["id"] if _sessions_at_boot else create_chat_session(name="Chat 1")
SCORE_THRESHOLD = 3.0
MAX_RETRIES     = 3


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):        return jsonify({"error": "Route not found"}), 404
@app.errorhandler(413)
def file_too_large(e):   return jsonify({"error": "File too large. Max 100MB."}), 413
@app.errorhandler(Exception)
def handle_exception(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

# The old multi-page layout is now one app with hash routes; keep the old URLs working.
@app.route("/dashboard")
def dashboard():    return redirect("/#/insights")

@app.route("/architecture")
def architecture(): return redirect("/#/architecture")

@app.route("/demo")
def demo():         return redirect("/#/library")


@app.route("/api/status")
def api_status():
    """Health + capability probe used by the header status pill."""
    from indexer import get_indexed_docs
    import retriever
    from llm import GEN_MODEL, JUDGE_MODEL
    with retriever._reranker_lock:
        rr = "ready" if retriever._reranker not in (None, False) else \
             ("unavailable" if retriever._reranker is False else "loading")
    return jsonify({"docs": len(get_indexed_docs()), "active_doc_id": active_doc_id,
                    "active_session_id": active_session_id, "reranker": rr,
                    "gen_model": GEN_MODEL, "judge_model": JUDGE_MODEL,
                    "vision_model": os.getenv("VISION_MODEL") or "auto",
                    "groq_key": bool(os.getenv("GROQ_API_KEY"))})

@app.route("/active-doc")
def get_active_doc():
    return jsonify({"active_doc_id": active_doc_id})


# ── Indexing ──────────────────────────────────────────────────────────────────

@app.route("/index-doc", methods=["POST"])
def index_doc():
    global active_doc_id
    data   = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "").strip()
    text   = data.get("text",   "").strip()
    if not doc_id: return jsonify({"error": "doc_id is required"}), 400
    if not text:   return jsonify({"error": "text is required"}), 400
    if len(text.split()) < 10:
        return jsonify({"error": "Text too short (minimum 10 words)"}), 400
    try:
        index_document(doc_id, text)
        active_doc_id = doc_id
        enrich.enrich_async(doc_id)
        return jsonify({"status": "success", "doc_id": doc_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Indexing failed: {e}"}), 500


@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    global active_doc_id
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file   = request.files["file"]
    doc_id = request.form.get("doc_id", "").strip()
    if not file or file.filename == "": return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400
    if not doc_id:
        doc_id = file.filename.replace(".pdf","").replace(" ","-").lower()
    doc_id = "".join(c if c.isalnum() or c in "-_" else "-" for c in doc_id)
    try:
        path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(path)
        from indexer import index_pdf
        index_pdf(doc_id, path)
        enrich.enrich_async(doc_id)
        active_doc_id = doc_id
        return jsonify({"status": "success", "doc_id": doc_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"PDF processing failed: {e}"}), 500


@app.route("/scrape-url", methods=["POST"])
def scrape_url_route():
    global active_doc_id
    data   = request.get_json(silent=True) or {}
    url    = data.get("url",    "").strip()
    doc_id = data.get("doc_id", "").strip()
    if not url: return jsonify({"error": "URL is required"}), 400
    if not url.startswith(("http://","https://")):
        return jsonify({"error": "URL must start with http:// or https://"}), 400
    if not doc_id:
        doc_id = url.replace("https://","").replace("http://","").split("/")[0]
    doc_id = "".join(c if c.isalnum() or c in "-_." else "-" for c in doc_id)
    try:
        text = scrape_url(url)
        if len(text.split()) < 30:
            return jsonify({"error": "Page has too little content."}), 400
        index_document(doc_id, text)
        enrich.enrich_async(doc_id)
        active_doc_id = doc_id
        return jsonify({"status":"success","doc_id":doc_id,"word_count":len(text.split())})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/crawl", methods=["POST"])
def crawl():
    global active_doc_id
    data      = request.get_json(silent=True) or {}
    url       = data.get("url","").strip()
    max_pages = data.get("max_pages", 10)
    if not url: return jsonify({"error": "URL is required"}), 400
    if not url.startswith(("http://","https://")):
        return jsonify({"error": "URL must start with http:// or https://"}), 400
    try:
        max_pages = int(max_pages)
        if not 1 <= max_pages <= 50:
            return jsonify({"error": "max_pages must be 1–50"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "max_pages must be a number"}), 400
    try:
        from crawler import crawl_and_index
        summary = crawl_and_index(url, max_pages=max_pages)
        for pg in summary.get("pages", []):
            enrich.enrich_async(pg["doc_id"])
        if summary["indexed_count"] == 0:
            return jsonify({"error": "No pages could be indexed."}), 400
        if summary["pages"]:
            active_doc_id = summary["pages"][0]["doc_id"]
        return jsonify({
            "status":"success", "indexed_count":summary["indexed_count"],
            "failed_count":summary["failed_count"], "skipped_count":summary["skipped_count"],
            "pages":summary["pages"], "active_doc":active_doc_id
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Crawl failed: {e}"}), 500


# ── Document management ───────────────────────────────────────────────────────

@app.route("/doc/<path:doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    """Delete a document from every vector collection AND the SQLite inverted index."""
    global active_doc_id
    from indexer import get_indexed_docs
    from database import DBSession, DocTag
    try:
        removed = delete_document(doc_id)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Delete failed: {e}"}), 500
    try:
        db = DBSession(); db.query(DocTag).filter_by(doc_id=doc_id).delete(); db.commit(); db.close()
    except Exception as e:
        print(f"Tag cleanup warning: {e}")
    if active_doc_id == doc_id:
        remaining     = get_indexed_docs()
        active_doc_id = remaining[0] if remaining else None
    return jsonify({"status": "success", "doc_id": doc_id, "removed": removed, "new_active": active_doc_id})


@app.route("/doc-info/<path:doc_id>", methods=["GET"])
def doc_info(doc_id):
    """Health stats for a document: chunk counts, unique terms, word count."""
    from page_index import get_conn
    large, small = get_doc_chunk_counts(doc_id)
    word_count = term_count = 0
    try:
        conn = get_conn(); c = conn.cursor()
        row  = c.execute("SELECT word_count FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        term_count = c.execute("SELECT COUNT(*) FROM index_term WHERE doc_id=?", (doc_id,)).fetchone()[0]
        conn.close()
        word_count = row[0] if row else 0
    except Exception:
        pass
    return jsonify({"doc_id": doc_id, "chunk_count": large, "small_chunk_count": small,
                    "word_count": word_count, "term_count": term_count,
                    "diversity": round(term_count / max(word_count, 1), 3)})


# ── Multi-format upload (runs as a background job: vision passes can take minutes) ─────────────

INGEST_JOBS = {}


def _doc_id_from(name):
    stem = os.path.splitext(name)[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-").lower() or "document"


@app.route("/upload-file", methods=["POST"])
def upload_file():
    files = request.files.getlist("files") or request.files.getlist("file")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "No files provided"}), 400
    deep   = request.form.get("deep", "").lower() in ("1", "true", "on", "yes")
    custom = re.sub(r"[^A-Za-z0-9._-]+", "-", request.form.get("doc_id", "").strip()).strip("-").lower() \
        if len(files) == 1 else ""
    accepted, rejected = [], []
    for f in files:
        name = secure_filename(f.filename) or f.filename
        if not ingest.detect_kind(name):
            rejected.append({"name": f.filename, "error": f"Unsupported type. Supported: {', '.join(ingest.SUPPORTED)}"})
            continue
        suffix = os.path.splitext(name)[1]
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=app.config["UPLOAD_FOLDER"])
        os.close(fd); f.save(tmp)
        accepted.append({"path": tmp, "name": name, "doc_id": custom or _doc_id_from(name)})
    if not accepted:
        return jsonify({"error": "None of the files can be indexed.", "rejected": rejected}), 400

    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "status": "running", "done": 0, "total": len(accepted), "message": "Starting", "results": [], "error": None}
    INGEST_JOBS[jid] = job

    def work():
        global active_doc_id
        from indexer import index_blocks
        for i, item in enumerate(accepted):
            job["done"] = i
            def prog(d, t, msg, _i=i, _n=item["name"]):
                job["message"] = f"{_n}: {msg}" + (f" ({d + 1}/{t})" if t else "")
            try:
                prog(0, 0, "Parsing")
                res = ingest.parse_file(item["path"], item["doc_id"], item["name"], deep, prog)
                job["message"] = f"{item['name']}: indexing {len(res['blocks'])} blocks"
                index_blocks(item["doc_id"], res["blocks"], res["kind"], res["stats"], item["name"])
                active_doc_id = item["doc_id"]
                job["message"] = f"{item['name']}: writing summary, entities and tags"
                try:
                    enrich.enrich_document(item["doc_id"])
                except Exception as e:                   # noqa: BLE001
                    res["warnings"].append("The automatic summary, entities and tags could not be created "
                                           f"({str(e)[:120]}). Open the document's Details and press Regenerate.")
                job["results"].append({"doc_id": item["doc_id"], "name": item["name"], "kind": res["kind"],
                                       "stats": res["stats"], "warnings": res["warnings"]})
            except Exception as e:                       # noqa: BLE001
                traceback.print_exc()
                job["results"].append({"doc_id": item["doc_id"], "name": item["name"], "error": str(e)})
            finally:
                try: os.remove(item["path"])
                except OSError: pass
        job["done"] = len(accepted)
        job["status"] = "done"
        job["message"] = "Finished"

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": jid, "accepted": [a["name"] for a in accepted], "rejected": rejected})


@app.route("/api/ingest/jobs/<jid>")
def ingest_job(jid):
    job = INGEST_JOBS.get(jid)
    if not job: return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/docs/<path:doc_id>/enrichment")
def doc_enrichment(doc_id):
    from page_index import get_enrichment
    e = get_enrichment(doc_id)
    return jsonify({"ready": bool(e), **(e or {"summary": "", "entities": {}, "tags": []})})


@app.route("/api/docs/<path:doc_id>/enrich", methods=["POST"])
def doc_enrich(doc_id):
    try:
        return jsonify({"ready": True, **enrich.enrich_document(doc_id)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not generate the summary: {e}"}), 500


@app.route("/api/quiz", methods=["POST"])
def api_quiz():
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id") or active_doc_id
    if not doc_id:
        return jsonify({"error": "Choose a document first."}), 400
    try:
        return jsonify({"doc_id": doc_id, "questions": quiz_mod.make_quiz(doc_id, data.get("count", 5))})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not create the quiz: {e}"}), 500


@app.route("/chat-turns/delete", methods=["POST"])
def delete_chat_turns_route():
    """Removes whole turns (the question plus every stored answer to it). Analytics logs are kept."""
    from database import delete_chat_turns
    data = request.get_json(silent=True) or {}
    ids  = [i for i in data.get("user_message_ids", []) if str(i).isdigit()]
    if not ids:
        return jsonify({"error": "No messages given."}), 400
    removed = delete_chat_turns(data.get("session_id") or active_session_id, ids)
    return jsonify({"status": "success", "removed": removed})


@app.route("/figure/<doc_dir>/<fig>")
def figure_file(doc_dir, fig):
    if not re.fullmatch(r"f\d{3,}\.png", fig) or not re.fullmatch(r"[A-Za-z0-9._-]+", doc_dir):
        return jsonify({"error": "Not found"}), 404
    resp = send_from_directory(str(ingest.FIG_DIR.resolve() / doc_dir), fig)
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@app.route("/api/docs/<path:doc_id>/figures")
def doc_figures(doc_id):
    from indexer import list_figures
    return jsonify({"figures": list_figures(doc_id)})


@app.route("/api/docs/<path:doc_id>/data")
def doc_data(doc_id):
    """Preview of a spreadsheet document: columns and the first rows of each sheet."""
    out = []
    for name, df in tabular.load_sheets(doc_id).items():
        out.append({"sheet": name, "rows": int(len(df)), "columns": [str(c) for c in df.columns],
                    "head": json.loads(df.head(8).to_json(orient="values", date_format="iso"))})
    return jsonify({"sheets": out})


@app.route("/api/general-chat", methods=["POST"])
def general_chat():
    """
    Direct conversation with the LLM, NOT grounded in the documents and NOT logged to the analytics
    (so it never pollutes the evaluation metrics). Streams tokens as server-sent events.
    """
    data     = request.get_json(silent=True) or {}
    msgs     = conversation.clean(data.get("messages"), limit=40)
    model    = str(data.get("model") or "").strip()
    if not msgs or msgs[-1]["role"] != "user":
        return jsonify({"error": "The last message must be from the user."}), 400
    if model and not re.fullmatch(r"[A-Za-z0-9._:/\-]{1,80}", model):
        return jsonify({"error": "Invalid model name."}), 400
    from llm import chat as _chat, GEN_MODEL
    system = ("You are a helpful, accurate assistant for students and researchers. Explain step by step when it "
              "helps. Put code in fenced blocks with the language name and write mathematics in LaTeX ($...$ inline, "
              "$$...$$ display). If you are not sure, say so instead of guessing.")
    full = [{"role": "system", "content": system}] + msgs

    def gen():
        send = lambda o: f"data: {json.dumps(o)}\n\n"
        try:
            stream = _chat(full, model=model or GEN_MODEL, temperature=0.4, timeout=90, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield send({"type": "token", "content": chunk.choices[0].delta.content})
            yield send({"type": "done"})
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc()
            yield send({"type": "error", "message": str(e)})

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/library", methods=["GET"])
def api_library():
    """One call for the Library view: every document with size, chunk counts and tags."""
    from indexer import get_indexed_docs
    from page_index import get_conn
    from database import get_doc_tags
    docs = get_indexed_docs()
    words = {}
    try:
        conn = get_conn()
        words = dict(conn.execute("SELECT doc_id, word_count FROM documents").fetchall())
        terms = dict(conn.execute("SELECT doc_id, COUNT(*) FROM index_term GROUP BY doc_id").fetchall())
        conn.close()
    except Exception:
        terms = {}
    tags = {}
    for t in get_doc_tags():
        tags.setdefault(t["doc_id"], []).append({"id": t["id"], "tag": t["tag"]})
    from page_index import get_all_doc_meta
    meta = get_all_doc_meta()
    out = []
    for d in docs:
        large, small = get_doc_chunk_counts(d)
        mt = meta.get(d, {"kind": "text", "stats": {}, "source": ""})
        out.append({"doc_id": d, "word_count": words.get(d, 0), "term_count": terms.get(d, 0),
                    "chunks": large, "small_chunks": small, "tags": tags.get(d, []),
                    "active": d == active_doc_id, "kind": mt["kind"], "stats": mt["stats"], "source": mt["source"]})
    return jsonify({"docs": out, "active_doc_id": active_doc_id})


@app.route("/set-active-doc", methods=["POST"])
def set_active_doc():
    global active_doc_id
    data   = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "").strip()
    active_doc_id = doc_id if doc_id else None
    return jsonify({"status": "success", "active_doc_id": active_doc_id})


# ── General LLM fallback (no RAG context) ────────────────────────────────────

@app.route("/answer-general", methods=["POST"])
def answer_general():
    """
    Answers a question using the LLM's own knowledge — no document retrieval.
    Clearly separate from RAG answers. Used as fallback when RAG finds nothing.
    """
    data  = request.get_json(silent=True) or {}
    query = data.get("query","").strip()
    if not query: return jsonify({"error": "query required"}), 400

    from llm import chat as _llm_chat, GEN_MODEL
    try:
        resp = _llm_chat(
            model    = GEN_MODEL,
            messages = [{"role":"user","content":
                f"Answer this question clearly and concisely using your general knowledge.\n"
                f"If you are not confident, say so.\n\nQuestion: {query}\n\nAnswer:"}],
            temperature = 0.4,
            timeout     = 30
        )
        answer = resp.choices[0].message.content.strip()

        # Save to chat history as fallback type
        save_chat_message("fallback", answer, metadata={
            "query" : query,
            "source": "general_llm",
            "note"  : "Not from indexed documents — LLM general knowledge"
        })

        return jsonify({"answer": answer, "source": "general_llm"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Persistent chat history (session-aware) ───────────────────────────────────

@app.route("/chat-history", methods=["GET"])
def get_history():
    try:
        limit      = max(1, min(int(request.args.get("limit", 100)), 500))
        session_id = request.args.get("session_id") or active_session_id
        return jsonify({"messages": get_chat_history(limit=limit, session_id=session_id)})
    except Exception as e:
        return jsonify({"messages": [], "error": str(e)})


@app.route("/chat-history", methods=["POST"])
def save_history():
    data       = request.get_json(silent=True) or {}
    role       = data.get("role",       "").strip()
    content    = data.get("content",    "").strip()
    metadata   = data.get("metadata",   None)
    session_id = data.get("session_id") or active_session_id
    if not role or not content:
        return jsonify({"error": "role and content required"}), 400
    save_chat_message(role, content, metadata, session_id=session_id)
    return jsonify({"status": "saved"})


@app.route("/chat-history", methods=["DELETE"])
def delete_history():
    session_id = request.args.get("session_id") or active_session_id
    clear_chat_history(session_id=session_id)
    return jsonify({"status": "cleared"})


# ── Chat Session CRUD ─────────────────────────────────────────────────────────

@app.route("/chat-sessions", methods=["GET"])
def list_chat_sessions():
    from database import get_all_chat_sessions, get_session_preview
    try:
        sessions = get_all_chat_sessions()
        for s in sessions:
            s["preview"] = get_session_preview(s["id"])
        return jsonify({"sessions": sessions, "active_session_id": active_session_id})
    except Exception as e:
        return jsonify({"sessions": [], "error": str(e)})


@app.route("/chat-sessions", methods=["POST"])
def new_chat_session():
    global active_session_id
    from database import create_chat_session
    data = request.get_json(silent=True) or {}
    name = data.get("name", "New Chat").strip() or "New Chat"
    try:
        sid               = create_chat_session(name=name)
        active_session_id = sid
        return jsonify({"status": "created", "session_id": sid, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat-sessions/<sid>", methods=["PATCH"])
def update_chat_session(sid):
    from database import rename_chat_session, toggle_pin_session
    data   = request.get_json(silent=True) or {}
    action = data.get("action", "rename")
    try:
        if action == "rename":
            name = data.get("name", "").strip()
            if not name: return jsonify({"error": "name required"}), 400
            rename_chat_session(sid, name)
            return jsonify({"status": "renamed", "name": name})
        elif action == "pin":
            toggle_pin_session(sid)
            return jsonify({"status": "toggled"})
        return jsonify({"error": "unknown action"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat-sessions/<sid>", methods=["DELETE"])
def remove_chat_session(sid):
    global active_session_id
    from database import delete_chat_session, get_all_chat_sessions, create_chat_session
    try:
        delete_chat_session(sid)
        if active_session_id == sid:
            # switch to most recent remaining session or create new one
            remaining = get_all_chat_sessions()
            if remaining:
                active_session_id = remaining[0]["id"]
            else:
                active_session_id = create_chat_session()
        return jsonify({"status": "deleted", "active_session_id": active_session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat-sessions/switch/<sid>", methods=["POST"])
def switch_chat_session(sid):
    global active_session_id
    from database import get_all_chat_sessions
    try:
        sessions = get_all_chat_sessions()
        ids      = [s["id"] for s in sessions]
        if sid not in ids:
            return jsonify({"error": "Session not found"}), 404
        active_session_id = sid
        return jsonify({"status": "switched", "session_id": sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat-sessions/active", methods=["GET"])
def get_active_session():
    global active_session_id
    from database import create_chat_session, get_all_chat_sessions
    try:
        # auto-create a session if none exists
        if not active_session_id:
            sessions = get_all_chat_sessions()
            if sessions:
                active_session_id = sessions[0]["id"]
            else:
                active_session_id = create_chat_session(name="Chat 1")
        return jsonify({"session_id": active_session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main query (non-streaming) ────────────────────────────────────────────────

@app.route("/query", methods=["POST"])
def query():
    global active_doc_id
    data          = request.get_json(silent=True) or {}
    user_query    = data.get("query",         "").strip()
    multi_doc     = data.get("multi_doc",     False)
    selected_docs = data.get("selected_docs", None)   # list or None
    threshold     = _clamp(data.get("threshold", SCORE_THRESHOLD), 1.0, 5.0, SCORE_THRESHOLD)

    if not user_query: return jsonify({"error": "Query is required"}), 400

    from indexer import get_indexed_docs, get_partial_matches
    indexed_docs = get_indexed_docs() or []
    if not indexed_docs:
        return jsonify({"answer":None,"not_found":True,"reason":"no_docs",
                        "indexed_docs":[],"partial_matches":[],
                        "suggestion":"No documents indexed yet."})

    trace = []; step = 0
    def add_trace(label, detail=""):
        nonlocal step; step += 1
        trace.append({"step":step,"label":label,"detail":str(detail)})

    add_trace("Query received", user_query)

    attempt=0; top_k=5; answer=None; scores=None; last_error=None
    reformulated_query=None; current_query=user_query
    chunks=[]; labeled_chunks=[]; cited_sources=[]

    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            if selected_docs:
                chunks = retrieve_selective(current_query, selected_docs, top_k_per_doc=top_k)
                add_trace(f"Retrieval (attempt {attempt})",
                          f"Selective — {len(chunks)} chunks from {len(selected_docs)} selected docs")
            elif multi_doc:
                labeled_chunks = _retrieve_multi_doc(current_query, top_k)
                chunks = [c["text"] for c in labeled_chunks]
                add_trace(f"Retrieval (attempt {attempt})",
                          f"Multi-doc — {len(chunks)} chunks")
            else:
                chunks = retrieve(current_query, top_k=top_k, doc_id=active_doc_id)
                add_trace(f"Retrieval (attempt {attempt})",
                          f"Single-doc '{active_doc_id}' — {len(chunks)} chunks")
        except Exception as e:
            last_error = f"Retrieval failed: {e}"; break

        if not chunks: break

        try:
            if multi_doc and labeled_chunks:
                result = generate_with_citations(current_query, labeled_chunks)
                answer = result["answer"]; cited_sources = result["cited_sources"]
            else:
                answer = generate(current_query, chunks)
        except Exception as e:
            last_error = f"Generation failed: {e}"; break

        try:
            scores = evaluate(current_query, chunks, answer)
        except Exception:
            scores = {"retrieval_relevance":0,"answer_grounding":0,
                      "answer_quality":0,"average_score":0,"reasoning":"unavailable"}
            break

        if scores["average_score"] >= threshold:
            add_trace("Result", f"✓ Score {scores['average_score']} passed"); break

        if attempt < MAX_RETRIES:
            new_q = reformulate_query(current_query, f"score {scores['average_score']}/5")
            if new_q: reformulated_query=new_q; current_query=new_q; add_trace("Reformulated", new_q)
            else: top_k += 3

    if last_error and not answer: return jsonify({"error": last_error}), 500

    not_found = (scores is None or scores["average_score"] < threshold)
    if not_found:
        partial = get_partial_matches(user_query, doc_id=active_doc_id if not multi_doc else None)
        return jsonify({"answer":None,"not_found":True,"reason":"low_confidence",
                        "indexed_docs":indexed_docs,"partial_matches":partial,
                        "suggestion":"This topic doesn't appear in the indexed documents.",
                        "trace":trace,"reformulated_query":reformulated_query})

    h_result = {"sentences":[],"hallucination_score":0.0,"unsupported_count":0,"total_count":0}
    try: h_result = detect_hallucination(answer, chunks)
    except Exception: pass

    try:
        log_query(query=user_query, answer=answer,
                  retrieval_relevance=scores["retrieval_relevance"],
                  answer_grounding=scores["answer_grounding"],
                  answer_quality=scores["answer_quality"],
                  average_score=scores["average_score"], attempts=attempt,
                  reformulated_query=reformulated_query,
                  hallucination_score=h_result["hallucination_score"],
                  multi_doc_mode=multi_doc,
                  cited_sources=cited_sources if multi_doc else [], trace=trace)
    except Exception as e: print(f"Log warning: {e}")

    return jsonify({"answer":answer,"not_found":False,"scores":scores,"attempts":attempt,
                    "active_doc":active_doc_id,"multi_doc":multi_doc,
                    "reformulated_query":reformulated_query,
                    "hallucination":h_result,"cited_sources":cited_sources,"trace":trace})


# ── Streaming chat ────────────────────────────────────────────────────────────

@app.route("/chat-stream", methods=["POST"])
def chat_stream():
    data          = request.get_json(silent=True) or {}
    user_query    = data.get("query",         "").strip()
    history       = data.get("history",       [])
    multi_doc     = data.get("multi_doc",     False)
    selected_docs = data.get("selected_docs", None)   # NEW — list of doc_ids or None
    threshold     = _clamp(data.get("threshold", SCORE_THRESHOLD), 1.0, 5.0, SCORE_THRESHOLD)
    use_hybrid    = bool(data.get("hybrid", False))     # chunk-level BM25 + dense RRF
    use_repair    = bool(data.get("repair", False))     # sentence-level verify & repair
    is_regen      = bool(data.get("regenerate", False)) # a new version of an existing answer: do not repeat the question
    memory_turns  = int(_clamp(data.get("memory_turns", 4), 1, 12, 4))
    history       = conversation.clean(history if isinstance(history, list) else [])

    if not user_query: return jsonify({"error": "Query required"}), 400

    # ── Small talk: answer directly and skip the RAG pipeline ────────────────
    _msg = chitchat.reply_for(user_query)
    if _msg:
        def greeting_stream():
            uid = aid = None
            try:
                uid = save_chat_message("user",     user_query, session_id=active_session_id)
                aid = save_chat_message("greeting", _msg,       session_id=active_session_id)
            except Exception:
                pass
            for word in _msg.split(" "):
                yield f"data: {json.dumps({'type':'token','content':word+' '})}\n\n"
            yield f"data: {json.dumps({'type':'done','scores':None,'hallucination':None,'cited_sources':[],'attempts':0,'multi_doc':False,'reformulated_query':None,'is_greeting':True,'user_msg_id':uid,'assistant_msg_id':aid})}\n\n"

        return Response(stream_with_context(greeting_stream()), mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    def event_stream():
        from indexer import get_indexed_docs, get_partial_matches

        def send(obj): return f"data: {json.dumps(obj)}\n\n"
        step = 0
        def trace(label, detail=""):
            nonlocal step; step += 1
            return send({"type":"trace_step","step":step,"label":label,"detail":str(detail)})

        try:
            indexed_docs = get_indexed_docs()
            if not indexed_docs:
                yield send({"type":"not_found","suggestion":"No documents indexed yet.",
                            "partial_matches":[],"indexed_docs":[]})
                return

            yield trace("Query received", user_query)

            if selected_docs:
                mode_label = f"Selective — {len(selected_docs)} chosen docs"
            elif multi_doc:
                mode_label = "Multi-doc — all docs"
            else:
                mode_label = f"Single-doc — '{active_doc_id}'"
            yield trace("Mode", mode_label)

            # memory: recent turns verbatim + a rolling summary of anything older
            older, recent = conversation.split(history, memory_turns)
            summary = conversation.summarize(older) if older else None
            if history:
                yield trace("Memory", f"{len(recent) // 2} recent turn(s)" + (f" + summary of {len(older) // 2} older" if summary else ""))
            retrieval_query = None
            if conversation.looks_like_follow_up(user_query, recent):
                retrieval_query = conversation.condense(user_query, recent)
                if retrieval_query:
                    yield trace("Follow-up understood as", retrieval_query)
            scope_ids = selected_docs or ([active_doc_id] if (active_doc_id and not multi_doc) else None)

            current_query = retrieval_query or user_query; reformulated_query=None
            user_msg_id = assistant_msg_id = None

            # "Give me only the diagram": answer with the figure itself, not an essay about it
            if multimodal.wants_figure_only(user_query):
                figs = multimodal.related_figures(current_query, scope_ids, [], max_figs=2, strict=True)
                if figs:
                    yield trace("Figure lookup", f"{len(figs)} matching figure(s)")
                    def _cap(f):
                        return (f.get("caption") or (f"Figure {f['label']}" if f.get("label") else "Figure")) + (f" (page {f['page']})" if f.get("page") else "")
                    text = ("Here is the figure you asked for: **" + _cap(figs[0]) + "**." if len(figs) == 1
                            else "Here are the closest matching figures: " + "; ".join("**" + _cap(f) + "**" for f in figs) + ".")
                    for w in text.split(" "):
                        yield send({"type": "token", "content": w + " "})
                    try:
                        if not is_regen:
                            user_msg_id = save_chat_message("user", user_query, session_id=active_session_id)
                        assistant_msg_id = save_chat_message("assistant", text, session_id=active_session_id, metadata={
                            "scores": None, "hallucination": None, "cited_sources": [], "attempts": 0, "sources": [],
                            "figures": figs, "figure_only": True, "follow_up": retrieval_query})
                    except Exception as e:
                        print(f"Save warning: {e}")
                    yield send({"type": "done", "scores": None, "hallucination": None, "cited_sources": [], "attempts": 0,
                                "multi_doc": multi_doc, "reformulated_query": None, "query_id": None, "answer": text,
                                "repaired": False, "sources": [], "figures": figs, "follow_up": retrieval_query,
                                "figure_only": True, "user_msg_id": user_msg_id, "assistant_msg_id": assistant_msg_id})
                    return
            chunks=[]; labeled_chunks=[]; cited_sources=[]
            attempt=0; top_k=5; scores=None; full_answer=""; query_id=None
            MAX_STREAM_RETRIES = 2

            while attempt < MAX_STREAM_RETRIES:
                attempt += 1

                if attempt > 1:
                    yield send({"type": "reset"})     # client drops the first attempt's tokens

                # ── Retrieval ─────────────────────────────────────────────
                if use_hybrid and not multi_doc:
                    scope = selected_docs or ([active_doc_id] if active_doc_id else None)
                    chunks = research.hybrid_retrieve(current_query, top_k=top_k, doc_ids=scope)
                    yield trace(f"Retrieval (attempt {attempt})",
                                f"Hybrid BM25 + dense (RRF) — {len(chunks)} chunks")
                elif selected_docs:
                    chunks = retrieve_selective(current_query, selected_docs,
                                                top_k_per_doc=top_k)
                    yield trace(f"Retrieval (attempt {attempt})",
                                f"Selective — {len(chunks)} chunks from {len(selected_docs)} docs")
                elif multi_doc:
                    labeled_chunks = _retrieve_multi_doc(current_query, top_k)
                    chunks = [c["text"] for c in labeled_chunks]
                    yield trace(f"Retrieval (attempt {attempt})",
                                f"Multi-doc — {len(chunks)} chunks")
                else:
                    chunks = retrieve(current_query, top_k=top_k, doc_id=active_doc_id)
                    yield trace(f"Retrieval (attempt {attempt})",
                                f"TF-IDF + vector — {len(chunks)} chunks")

                if not chunks: break

                # ── Metadata, exact spreadsheet maths ─────────────────────
                from indexer import meta_for_texts
                metas = meta_for_texts(chunks, scope_ids)
                try:
                    sheet_docs = [d for d in (scope_ids or indexed_docs) if tabular.has_data(d)]
                    calc = tabular.answer_from_tables(current_query, sheet_docs) if sheet_docs else None
                    if calc:
                        chunks = [calc[0]] + chunks
                        metas  = [{"block_type": "sheet", "doc_id": calc[1].split("/")[0], "page": -1, "file": calc[1].split(":")[0]}] + metas
                        yield trace("Calculated from spreadsheet", calc[1])
                except Exception as e:
                    yield trace("Spreadsheet calculation skipped", str(e)[:120])
                labels = [multimodal.passage_label(m) for m in metas]

                yield trace("Generating answer", "Streaming tokens...")

                # ── Generation ────────────────────────────────────────────
                if multi_doc and labeled_chunks:
                    result=generate_with_citations(current_query,labeled_chunks,recent)
                    full_answer=result["answer"]; cited_sources=result["cited_sources"]
                    for word in full_answer.split(" "):
                        yield send({"type":"token","content":word+" "})
                else:
                    full_answer=""
                    for token in generate_stream(current_query, chunks, recent, labels=labels,
                                                 memory_turns=memory_turns, summary=summary):
                        full_answer += token
                        yield send({"type":"token","content":token})

                yield trace("Generation complete", f"{len(full_answer.split())} words")

                # ── Evaluation ────────────────────────────────────────────
                yield trace("Self-evaluating...")
                try:
                    scores = evaluate(current_query, chunks, full_answer)
                    yield trace("Self-evaluation",
                                f"R:{scores['retrieval_relevance']} "
                                f"G:{scores['answer_grounding']} "
                                f"Q:{scores['answer_quality']} "
                                f"→ Avg:{scores['average_score']} | {scores['reasoning']}")
                except Exception as e:
                    scores={"retrieval_relevance":0,"answer_grounding":0,
                            "answer_quality":0,"average_score":0,"reasoning":"unavailable"}
                    yield trace("Self-evaluation", f"Skipped: {e}"); break

                if scores["average_score"] >= threshold:
                    yield trace("Result", f"✓ Score {scores['average_score']} passed (threshold {threshold})")
                    break

                if attempt < MAX_STREAM_RETRIES:
                    yield trace("Score below threshold",
                                f"{scores['average_score']} < {threshold} — reformulating")
                    new_q = reformulate_query(current_query,
                                              f"R:{scores['retrieval_relevance']} G:{scores['answer_grounding']}")
                    if new_q:
                        reformulated_query=new_q; current_query=new_q; full_answer=""
                        yield send({"type":"reformulated","query":new_q})
                        yield trace("Query reformulated", f"'{user_query}' → '{new_q}'")
                    else:
                        top_k += 3

            # ── Not found ─────────────────────────────────────────────────
            if not scores or scores["average_score"] < threshold:
                yield trace("Result", "Not found — below threshold after retries")
                partial = get_partial_matches(user_query,
                                              doc_id=active_doc_id if not multi_doc and not selected_docs else None)
                try:
                    if not is_regen:
                        user_msg_id = save_chat_message("user", user_query, session_id=active_session_id)
                        assistant_msg_id = save_chat_message("notfound", "This topic doesn't appear in the indexed documents.",
                                          session_id=active_session_id,
                                          metadata={"partial_matches": partial, "query": user_query})
                except Exception as e:
                    print(f"Save warning: {e}")
                yield send({"type":"not_found","user_msg_id":user_msg_id,"assistant_msg_id":assistant_msg_id,
                            "suggestion":"This topic doesn't appear in the indexed documents.",
                            "partial_matches":partial,"indexed_docs":indexed_docs})
                return

            # ── Hallucination detection ────────────────────────────────────
            yield trace("Hallucination check", "Scanning sentences...")
            h_result={"sentences":[],"hallucination_score":0.0,"unsupported_count":0,"total_count":0}
            repair_info = None
            try:
                if use_repair:
                    h_result = research.verify_answer(full_answer, chunks)
                    yield trace("Cascaded verification",
                                f"{h_result['unsupported_count']}/{h_result['total_count']} flagged · "
                                f"{h_result['embedding_decided']} decided by embeddings, "
                                f"{h_result['llm_calls']} LLM call")
                    if h_result["unsupported_count"]:
                        yield trace("Repairing", "Rewriting unsupported sentences")
                        repair_info = research.repair_answer(user_query, full_answer, chunks, h_result)
                        if repair_info["changed"]:
                            full_answer = repair_info["answer"]
                            h_result    = repair_info["verification"]
                            yield trace("Repair applied",
                                        f"{repair_info['method']} · flagged {round(repair_info['before']*100)}% → "
                                        f"{round(repair_info['after']*100)}%")
                            yield send({"type": "repaired", "answer": full_answer,
                                        "method": repair_info["method"],
                                        "before": repair_info["before"], "after": repair_info["after"]})
                        else:
                            yield trace("Repair skipped", "No candidate improved on the original")
                else:
                    h_result = detect_hallucination(full_answer, chunks)
                    yield trace("Hallucination check",
                                f"{h_result['unsupported_count']}/{h_result['total_count']} flagged")
            except Exception as e:
                yield trace("Hallucination check", f"Skipped: {e}")

            # ── Sources and related figures ────────────────────────────────
            sources, figures = [], []
            try:
                sources = multimodal.build_sources(chunks, metas)
                figures = multimodal.related_figures(user_query, scope_ids, metas)
                yield trace("Sources", f"{len(sources)} passage(s), {len(figures)} related figure(s)")
            except Exception as e:
                yield trace("Sources skipped", str(e)[:120])

            # ── Save to DB ─────────────────────────────────────────────────
            try:
                query_id = log_query(query=user_query, answer=full_answer,
                          retrieval_relevance=scores["retrieval_relevance"],
                          answer_grounding=scores["answer_grounding"],
                          answer_quality=scores["answer_quality"],
                          average_score=scores["average_score"], attempts=attempt,
                          reformulated_query=reformulated_query,
                          hallucination_score=h_result["hallucination_score"],
                          multi_doc_mode=multi_doc,
                          cited_sources=cited_sources if multi_doc else [], trace=None,
                          session_id=active_session_id)

                # Also persist to chat history (session-aware)
                if not is_regen:
                    user_msg_id = save_chat_message("user", user_query, session_id=active_session_id)
                assistant_msg_id = save_chat_message("assistant", full_answer, session_id=active_session_id, metadata={
                    "scores"            : scores,
                    "hallucination"     : h_result,
                    "cited_sources"     : cited_sources,
                    "reformulated_query": reformulated_query,
                    "multi_doc"         : multi_doc,
                    "attempts"          : attempt,
                    "selected_docs"     : selected_docs,
                    "query_id"          : query_id,
                    "repaired"          : bool(repair_info and repair_info["changed"]),
                    "sources"           : sources,
                    "figures"           : figures,
                    "follow_up"         : retrieval_query,
                })
            except Exception as e:
                print(f"Save warning: {e}")

            yield send({"type":"done","scores":scores,"hallucination":h_result,
                        "cited_sources":cited_sources,"attempts":attempt,
                        "multi_doc":multi_doc,"reformulated_query":reformulated_query,
                        "query_id":query_id,"answer":full_answer,
                        "repaired": bool(repair_info and repair_info["changed"]),
                        "sources":sources,"figures":figures,"follow_up":retrieval_query,
                        "user_msg_id":user_msg_id,"assistant_msg_id":assistant_msg_id})

        except Exception as e:
            traceback.print_exc()
            yield send({"type":"error","message":str(e)})

    return Response(stream_with_context(event_stream()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


def _retrieve_multi_doc(query, top_k=8):
    """Dense search over ALL documents, keeping the doc_id so the LLM can cite sources."""
    try:
        from indexer import model, large_col
        qe  = model.encode([query], show_progress_bar=False).tolist()
        n   = max(1, min(top_k, large_col.count()))
        res = large_col.query(query_embeddings=qe, n_results=n)
        return [{"doc_id": m.get("doc_id", "?"), "text": d, "score": float(s)}
                for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
    except Exception as e:
        print(f"Multi-doc retrieval error: {e}"); return []


# ── Benchmark ─────────────────────────────────────────────────────────────────

BENCH_FILE = "benchmark_runs.json"


def _record_benchmark(result, doc_id, source="benchmark"):
    """Keeps the last 30 accuracy measurements so the Insights page can show them."""
    try:
        runs = json.load(open(BENCH_FILE)) if os.path.exists(BENCH_FILE) else []
    except Exception:
        runs = []
    rs = result.get("results", [])
    runs.append({"ts": time.strftime("%Y-%m-%d %H:%M"), "doc_id": doc_id or "all documents", "source": source,
                 "total": result.get("total", 0), "correct": result.get("correct", 0),
                 "accuracy_pct": result.get("accuracy_pct", 0.0),
                 "avg_self_score": round(sum((r.get("avg_score") or 0) for r in rs) / len(rs), 2) if rs else 0})
    try:
        json.dump(runs[-30:], open(BENCH_FILE, "w"))
    except Exception as e:
        print(f"benchmark history warning: {e}")


@app.route("/api/accuracy")
def api_accuracy():
    """Everything the Insights page needs to answer 'how accurate is the system?'."""
    try:
        runs = json.load(open(BENCH_FILE)) if os.path.exists(BENCH_FILE) else []
    except Exception:
        runs = []
    return jsonify({"runs": runs[-20:], "latest": runs[-1] if runs else None})


@app.route("/benchmark", methods=["POST"])
def run_benchmark_route():
    data=request.get_json(silent=True) or {}
    questions=data.get("questions",[]); doc_id=data.get("doc_id",active_doc_id)
    if not questions: return jsonify({"error":"No questions"}), 400
    if len(questions)>50: return jsonify({"error":"Max 50 questions"}), 400
    for i,q in enumerate(questions):
        if not q.get("question") or not q.get("expected"):
            return jsonify({"error":f"Question {i+1} missing 'question' or 'expected'"}), 400
    try:
        from benchmark import run_benchmark
        res = run_benchmark(questions, doc_id=doc_id)
        _record_benchmark(res, doc_id, data.get("source", "benchmark"))
        return jsonify(res)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":f"Benchmark failed: {e}"}), 500


@app.route("/benchmark-upload", methods=["POST"])
def benchmark_upload():
    if "file" not in request.files: return jsonify({"error":"No file"}), 400
    file=request.files["file"]; doc_id=request.form.get("doc_id",active_doc_id)
    if not file.filename.lower().endswith(".json"):
        return jsonify({"error":"Only .json files"}), 400
    try:
        qs=json.loads(file.read().decode("utf-8"))
        if not isinstance(qs,list): return jsonify({"error":"JSON must be an array"}), 400
        from benchmark import run_benchmark
        res = run_benchmark(qs, doc_id=doc_id or None)
        _record_benchmark(res, doc_id or None, "benchmark-upload")
        return jsonify(res)
    except json.JSONDecodeError as e:
        return jsonify({"error":f"Invalid JSON: {e}"}), 400
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":f"Benchmark failed: {e}"}), 500


# ── Comparisons & analysis ────────────────────────────────────────────────────

@app.route("/compare-retrievers", methods=["POST"])
def compare_retrievers():
    data=request.get_json(silent=True) or {}
    query=data.get("query","").strip(); doc_id=data.get("doc_id",active_doc_id)
    if not query: return jsonify({"error":"Query required"}), 400
    from retriever import retrieve_with_scores, retrieve_bm25
    result={"query":query,"tfidf":{},"bm25":{}}
    for algo,fn in [("tfidf",retrieve_with_scores),("bm25",retrieve_bm25)]:
        try:
            ch,ranked=fn(query,top_k=5,doc_id=doc_id)
            ans=generate(query,ch) if ch else "No relevant content found."
            sc=evaluate(query,ch,ans) if ch else {"retrieval_relevance":0,"answer_grounding":0,"answer_quality":0,"average_score":0.0,"reasoning":"No chunks"}
            result[algo]={"answer":ans,"scores":sc,
                          "ranked_docs":[{"doc_id":d,"score":round(float(s),4)} for d,s in ranked[:5]],
                          "chunk_count":len(ch)}
        except Exception as e:
            result[algo]={"error":str(e)}
    ta=result["tfidf"].get("scores",{}).get("average_score",0)
    ba=result["bm25"].get("scores",{}).get("average_score",0)
    result["winner"]="tfidf" if ta>ba else "bm25" if ba>ta else "tie"
    return jsonify(result)


@app.route("/explain-answer", methods=["POST"])
def explain_answer():
    data=request.get_json(silent=True) or {}
    query=data.get("query","").strip(); answer=data.get("answer","").strip()
    doc_id=data.get("doc_id",active_doc_id)
    if not query or not answer: return jsonify({"error":"query and answer required"}), 400
    try:
        import re, numpy as np
        from indexer import model
        a_sents=[s.strip() for s in re.split(r'(?<=[.?!])\s+',answer) if len(s.strip())>15]
        chunks=retrieve(query,top_k=8,doc_id=doc_id)
        s_sents=[]
        for ch in chunks:
            s_sents.extend([s.strip() for s in re.split(r'(?<=[.?!])\s+',ch) if len(s.strip())>15])
        if not a_sents or not s_sents: return jsonify({"provenance":[]})
        embs=model.encode(a_sents+s_sents)
        a_embs=embs[:len(a_sents)]; s_embs=embs[len(a_sents):]
        def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        provenance=[]
        for i,ae in enumerate(a_embs):
            best=max(((cos(ae,se),s_sents[j]) for j,se in enumerate(s_embs)),key=lambda x:x[0])
            provenance.append({"answer_sentence":a_sents[i],"source_chunk":best[1],"similarity":round(best[0],3)})
        return jsonify({"provenance":provenance})
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


@app.route("/reindex-diff", methods=["POST"])
def reindex_diff():
    global active_doc_id
    data=request.get_json(silent=True) or {}
    doc_id=data.get("doc_id","").strip(); text=data.get("text","").strip()
    if not doc_id or not text: return jsonify({"error":"doc_id and text required"}), 400
    from page_index import get_term_snapshot, compute_term_diff
    old_snap=get_term_snapshot(doc_id); is_new=len(old_snap)==0
    try:
        index_document(doc_id,text); active_doc_id=doc_id
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":f"Indexing failed: {e}"}), 500
    if is_new: return jsonify({"status":"success","is_new":True,"doc_id":doc_id,"diff":None})
    return jsonify({"status":"success","is_new":False,"doc_id":doc_id,
                    "diff":compute_term_diff(old_snap,get_term_snapshot(doc_id))})


# ── Export & sessions ─────────────────────────────────────────────────────────

@app.route("/export-pdf")
def export_pdf():
    from database import get_all_logs
    from exporter import generate_pdf_report
    try:
        import datetime as dt
        pdf=generate_pdf_report(get_all_logs())
        ts=dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Response(pdf, mimetype="application/pdf",
                        headers={"Content-Disposition":f"attachment; filename=selfrag_{ts}.pdf"})
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":f"PDF failed: {e}"}), 500


@app.route("/sessions", methods=["GET"])
def list_sessions():
    from database import get_all_sessions; return jsonify({"sessions":get_all_sessions()})

@app.route("/sessions", methods=["POST"])
def create_session_route():
    from database import create_session, assign_queries_to_session
    data=request.get_json(silent=True) or {}
    name=data.get("name","Unnamed").strip() or "Unnamed"
    query_ids=data.get("query_ids",[])
    sid=create_session(name)
    if query_ids: assign_queries_to_session(query_ids,sid)
    return jsonify({"session_id":sid,"name":name})

@app.route("/sessions/<int:sid>")
def get_session_route(sid):
    from database import get_session_queries; return jsonify({"queries":get_session_queries(sid)})


# ── Page index stats ──────────────────────────────────────────────────────────

@app.route("/page-index-stats")
def page_index_stats():
    try:    return jsonify({"docs":get_all_indexed_docs(),"total":len(get_all_indexed_docs())})
    except Exception as e: return jsonify({"error":str(e),"docs":[],"total":0}), 500

@app.route("/page-index-terms")
def page_index_terms():
    from page_index import get_conn
    doc_id=request.args.get("doc_id","").strip()
    if not doc_id: return jsonify({"error":"doc_id required"}), 400
    try:
        top_k=int(request.args.get("top_k",15))
        conn=get_conn(); c=conn.cursor()
        rows=c.execute("""SELECT it.term,it.tf,COALESCE(idf.idf,1.0),
            it.tf*COALESCE(idf.idf,1.0) FROM index_term it
            LEFT JOIN idf_scores idf ON it.term=idf.term
            WHERE it.doc_id=? ORDER BY 4 DESC LIMIT ?""",(doc_id,top_k)).fetchall()
        conn.close()
        return jsonify({"doc_id":doc_id,"terms":[{"term":r[0],"tf":r[1],"idf":r[2],"tfidf":r[3]} for r in rows]})
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/page-index-lookup")
def page_index_lookup():
    from page_index import get_conn
    term=request.args.get("term","").strip().lower()
    if not term: return jsonify({"error":"term required"}), 400
    try:
        conn=get_conn(); c=conn.cursor()
        rows=c.execute("""SELECT it.doc_id,it.tf,COALESCE(idf.idf,1.0),
            it.tf*COALESCE(idf.idf,1.0),json_array_length(it.positions)
            FROM index_term it LEFT JOIN idf_scores idf ON it.term=idf.term
            WHERE it.term=? ORDER BY 4 DESC""",(term,)).fetchall()
        conn.close()
        return jsonify({"term":term,"results":[{"doc_id":r[0],"tf":r[1],"idf":r[2],"tfidf":r[3],"positions":r[4]} for r in rows]})
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/api/logs")
def api_logs():
    from database import get_all_logs
    try:    return jsonify({"logs":get_all_logs()})
    except Exception as e: return jsonify({"logs":[],"error":str(e)})

@app.route("/indexed-docs")
def indexed_docs_list():
    from indexer import get_indexed_docs
    try:    return jsonify({"docs":get_indexed_docs()})
    except Exception as e: return jsonify({"docs":[],"error":str(e)})


# ── Answer Regeneration ───────────────────────────────────────────────────────

@app.route("/regenerate", methods=["POST"])
def regenerate():
    """
    Re-runs the full RAG pipeline for a previous query.
    Returns a new streaming SSE response — identical to /chat-stream
    but always bypasses any cached state.
    Body: {query, history, multi_doc, selected_docs, threshold}
    """
    # Regeneration is identical to chat-stream — just forward to it
    return chat_stream()


# ── User Feedback ─────────────────────────────────────────────────────────────

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    from database import save_feedback
    data     = request.get_json(silent=True) or {}
    query    = data.get("query",    "").strip()
    answer   = data.get("answer",   "").strip()
    try:    rating = int(data.get("rating", 0))
    except (TypeError, ValueError): rating = 0
    query_id = data.get("query_id", None)
    comment  = data.get("comment",  None)

    if not query or rating not in (1, -1):
        return jsonify({"error": "query and rating (1 or -1) are required"}), 400

    try:
        save_feedback(query=query, answer=answer, rating=rating,
                      query_id=query_id, comment=comment)
        return jsonify({"status": "saved", "rating": rating})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["GET"])
def get_feedback():
    from database import get_all_feedback, get_feedback_stats
    try:
        return jsonify({
            "feedback": get_all_feedback(),
            "stats"   : get_feedback_stats()
        })
    except Exception as e:
        return jsonify({"feedback": [], "stats": {}, "error": str(e)})


# ── Auto Q&A Generator ────────────────────────────────────────────────────────

@app.route("/generate-qa", methods=["POST"])
def generate_qa():
    """
    Auto-generates benchmark Q&A pairs from an indexed document.
    """
    data   = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", active_doc_id)
    count  = min(int(data.get("count", 10)), 20)

    if not doc_id:
        return jsonify({"error": "doc_id required — index a document first"}), 400

    try:
        # Use a broad query to pull diverse representative chunks
        chunks = retrieve("key concepts main topics details overview examples",
                          top_k=8, doc_id=doc_id)
        if not chunks:
            from indexer import model, large_col
            qe  = model.encode(["overview"]).tolist()
            res = large_col.query(query_embeddings=qe, n_results=6,
                                  where={"doc_id": doc_id})
            chunks = res["documents"][0] if res["documents"] else []
        if not chunks:
            return jsonify({"error": "No content found for this document"}), 400

        context = "\n\n---\n\n".join(chunks[:6])
        from llm import chat as _llm_chat, GEN_MODEL
        resp = _llm_chat(
            model    = GEN_MODEL,
            messages = [{"role": "user", "content":
                f"""You are creating a benchmark test set for a document Q&A system.

Read the following document excerpts carefully and generate exactly {count} question-answer pairs.

Rules:
- Questions must be answerable ONLY from the provided text
- Answers must be specific and concise (1-2 sentences max)
- Cover different sections/topics of the document
- Vary question types: factual, definitional, list-based
- Return ONLY valid JSON array, no other text

Format:
[
  {{"question": "...", "expected": "..."}},
  ...
]

Document excerpts:
{context}

Generate {count} Q&A pairs:"""}],
            temperature = 0.3,
            max_tokens  = 2000,
            timeout     = 45
        )

        raw   = resp.choices[0].message.content.strip()
        raw   = raw.replace("```json","").replace("```","").strip()
        pairs = json.loads(raw)
        if not isinstance(pairs, list):
            raise ValueError("LLM returned non-array JSON")

        return jsonify({"status":"success","doc_id":doc_id,
                        "pairs":pairs[:count],"count":len(pairs[:count])})

    except json.JSONDecodeError:
        return jsonify({"error": "LLM returned invalid JSON — try again"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Document Summary on Index ─────────────────────────────────────────────────

@app.route("/doc-summary/<path:doc_id>", methods=["GET"])
def doc_summary(doc_id):
    """
    Generates a 3-sentence summary of an indexed document.
    Called automatically after indexing in the frontend.
    """
    if not doc_id or doc_id.strip() == "":
        return jsonify({"error": "doc_id required"}), 400
    try:
        # Use a broad query to get representative chunks from the document
        chunks = retrieve("main topics overview summary key concepts introduction",
                          top_k=6, doc_id=doc_id)
        if not chunks:
            # fallback: try direct vector search
            from indexer import model, large_col
            qe  = model.encode(["overview"]).tolist()
            res = large_col.query(query_embeddings=qe, n_results=5,
                                  where={"doc_id": doc_id})
            chunks = res["documents"][0] if res["documents"] else []
        if not chunks:
            return jsonify({"error": "No content found for this document"}), 404

        context = "\n\n".join(chunks[:5])
        from llm import chat as _llm_chat, GEN_MODEL
        resp = _llm_chat(
            model    = GEN_MODEL,
            messages = [{"role": "user", "content":
                f"""Summarize this document in exactly 3 sentences.
Be specific — mention the actual topics, names, and key points covered.
Do not use vague language. Return only the 3 sentences, nothing else.

Document:
{context}

3-sentence summary:"""}],
            temperature = 0.2,
            max_tokens  = 200,
            timeout     = 20
        )
        summary = resp.choices[0].message.content.strip()
        return jsonify({"status": "success", "doc_id": doc_id, "summary": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Named Entity Extraction ───────────────────────────────────────────────────

@app.route("/extract-entities/<path:doc_id>", methods=["GET"])
def extract_entities(doc_id):
    """
    Extracts named entities from an indexed document.
    """
    if not doc_id or doc_id.strip() == "":
        return jsonify({"error": "doc_id required"}), 400
    try:
        # Use a broad query to get representative chunks
        chunks = retrieve("people organizations technology dates locations terms",
                          top_k=8, doc_id=doc_id)
        if not chunks:
            from indexer import model, large_col
            qe  = model.encode(["overview"]).tolist()
            res = large_col.query(query_embeddings=qe, n_results=6,
                                  where={"doc_id": doc_id})
            chunks = res["documents"][0] if res["documents"] else []
        if not chunks:
            return jsonify({"error": "No content found for this document"}), 404

        context = "\n\n".join(chunks[:6])
        from llm import chat as _llm_chat, GEN_MODEL
        resp = _llm_chat(
            model    = GEN_MODEL,
            messages = [{"role": "user", "content":
                f"""Extract named entities from this document text.
Return ONLY valid JSON with these exact keys. Empty array if none found.
No text outside the JSON.

{{
  "people"       : ["name1", "name2"],
  "organizations": ["org1", "org2"],
  "locations"    : ["place1", "place2"],
  "dates"        : ["date1", "date2"],
  "technologies" : ["tech1", "tech2"],
  "key_terms"    : ["term1", "term2"]
}}

Document:
{context}

JSON:"""}],
            temperature = 0.1,
            max_tokens  = 600,
            timeout     = 25
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json","").replace("```","").strip()
        entities = json.loads(raw)
        return jsonify({"status": "success", "doc_id": doc_id, "entities": entities})
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse entities — try again"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Answer Source Highlighting ────────────────────────────────────────────────

@app.route("/source-highlight", methods=["POST"])
def source_highlight():
    """
    For each sentence in an answer, finds the most similar retrieved chunk
    and returns the mapping with similarity scores.
    More detailed than hallucination detection — shows exact source passage.
    Body: {query, answer, doc_id}
    """
    data   = request.get_json(silent=True) or {}
    query  = data.get("query",  "").strip()
    answer = data.get("answer", "").strip()
    doc_id = data.get("doc_id", active_doc_id)

    if not query or not answer:
        return jsonify({"error": "query and answer required"}), 400

    try:
        import re
        import numpy as np
        from indexer import model as emb_model

        a_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
                   if len(s.strip()) > 15]
        chunks  = retrieve(query, top_k=8, doc_id=doc_id)
        if not a_sents or not chunks:
            return jsonify({"mappings": []})

        all_texts = a_sents + chunks
        embs      = emb_model.encode(all_texts)
        a_embs    = embs[:len(a_sents)]
        c_embs    = embs[len(a_sents):]

        def cos(a, b):
            return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

        mappings = []
        for i, ae in enumerate(a_embs):
            scores   = [(cos(ae, ce), ci) for ci, ce in enumerate(c_embs)]
            best_s, best_ci = max(scores, key=lambda x: x[0])
            mappings.append({
                "sentence"   : a_sents[i],
                "source"     : chunks[best_ci][:300],
                "similarity" : round(best_s, 3),
                "supported"  : best_s > 0.45
            })

        return jsonify({"mappings": mappings, "chunk_count": len(chunks)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Query Analytics (legacy score-chart data, used by dashboard score charts) ──

@app.route("/api/query-analytics", methods=["GET"])
def legacy_score_analytics():
    """Returns analytics data from query_logs for dashboard score charts."""
    from database import get_all_logs
    from collections import Counter
    import re as _re

    try:
        logs = get_all_logs()
        if not logs:
            return jsonify({"error": "No queries logged yet"})

        stopwords = {"what","is","the","a","an","how","does","are","in","of",
                     "to","for","and","or","this","that","it","with","on","at"}
        word_freq = Counter()
        for log in logs:
            words = [w.lower() for w in _re.findall(r'\b\w+\b', log["query"] or "")
                     if len(w) > 2 and w.lower() not in stopwords]
            word_freq.update(words)

        from collections import defaultdict
        hour_dist = defaultdict(int)
        for log in logs:
            try:
                hour = int(log["created_at"].split(" ")[1].split(":")[0])
                hour_dist[hour] += 1
            except Exception:
                pass

        score_trend = [{"created_at": l["created_at"],
                        "average_score": l["average_score"],
                        "hallucination": round((l["hallucination_score"] or 0)*100)}
                       for l in reversed(logs[:30])]

        attempts_dist = Counter(l["attempts"] for l in logs)
        reformulated  = sum(1 for l in logs if l["reformulated_query"])

        return jsonify({
            "total_queries"     : len(logs),
            "top_words"         : word_freq.most_common(20),
            "hour_distribution" : dict(hour_dist),
            "score_trend"       : score_trend,
            "attempts_dist"     : dict(attempts_dist),
            "reformulation_rate": round(reformulated / len(logs) * 100) if logs else 0,
            "avg_score"         : round(sum(l["average_score"] for l in logs) / len(logs), 2),
            "avg_hallucination" : round(sum((l["hallucination_score"] or 0) for l in logs) / len(logs) * 100, 1),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Query Suggestions (autocomplete) ─────────────────────────────────────────

@app.route("/api/query-suggestions", methods=["GET"])
def query_suggestions():
    """Returns autocomplete suggestions based on past queries and TF-IDF terms."""
    from database import get_all_logs
    q      = request.args.get("q", "").strip().lower()
    doc_id = request.args.get("doc_id", active_doc_id)

    if len(q) < 2:
        return jsonify({"suggestions": []})

    try:
        # Past queries matching prefix
        logs = get_all_logs()
        past = [l["query"] for l in logs
                if l["query"] and q in l["query"].lower()]
        past = list(dict.fromkeys(past))[:5]   # dedupe, max 5

        # TF-IDF terms from active doc
        terms = []
        if doc_id:
            from page_index import get_conn
            conn = get_conn(); c = conn.cursor()
            rows = c.execute("""
                SELECT term FROM index_term it
                LEFT JOIN idf_scores idf ON it.term = idf.term
                WHERE it.doc_id = ? AND it.term LIKE ?
                ORDER BY it.tf * COALESCE(idf.idf, 1.0) DESC LIMIT 8
            """, (doc_id, f"{q}%")).fetchall()
            conn.close()
            terms = [r[0] for r in rows]

        suggestions = (past + [t for t in terms if t not in past])[:8]
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"suggestions": [], "error": str(e)})


# ── Document Comparison Mode ──────────────────────────────────────────────────

@app.route("/compare-docs", methods=["POST"])
def compare_docs():
    """
    Runs the same query against two different documents and returns both answers.
    Body: {query, doc_id_a, doc_id_b}
    """
    data     = request.get_json(silent=True) or {}
    query    = data.get("query",    "").strip()
    doc_id_a = data.get("doc_id_a", "").strip()
    doc_id_b = data.get("doc_id_b", "").strip()

    if not query:
        return jsonify({"error": "query required"}), 400
    if not doc_id_a or not doc_id_b:
        return jsonify({"error": "Both doc_id_a and doc_id_b are required"}), 400

    result = {"query": query, "a": {}, "b": {}}

    for side, doc_id in [("a", doc_id_a), ("b", doc_id_b)]:
        try:
            chunks = retrieve(query, top_k=5, doc_id=doc_id)
            if not chunks:
                result[side] = {"error": f"No content found in '{doc_id}'",
                                "doc_id": doc_id}
                continue

            answer = generate(query, chunks)
            scores = evaluate(query, chunks, answer)
            h      = detect_hallucination(answer, chunks)

            result[side] = {
                "doc_id"            : doc_id,
                "answer"            : answer,
                "scores"            : scores,
                "hallucination_pct" : round(h["hallucination_score"] * 100),
                "chunk_count"       : len(chunks),
            }
        except Exception as e:
            result[side] = {"doc_id": doc_id, "error": str(e)}

    return jsonify(result)


# ── Answer Export ─────────────────────────────────────────────────────────────

@app.route("/export-answer", methods=["POST"])
def export_answer():
    """
    Exports a single answer as a formatted PDF.
    Body: {query, answer, scores, hallucination_pct, reformulated_query, cited_sources}
    """
    data   = request.get_json(silent=True) or {}
    query  = data.get("query",  "")
    answer = data.get("answer", "")

    if not query or not answer:
        return jsonify({"error": "query and answer required"}), 400

    try:
        import io
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        import datetime as dt

        buf   = io.BytesIO()
        doc   = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=2.2*cm, rightMargin=2.2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
        story = []
        from exporter import _x
        C_ACC = colors.HexColor("#00629B")
        C_TXT = colors.HexColor("#14213D")
        C_MUT = colors.HexColor("#5B6577")

        def ps(name, **kw):
            return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **kw)

        story.append(Paragraph("Self-RAG", ps("h1", fontSize=22,
            textColor=C_ACC, fontName="Helvetica-Bold", spaceAfter=3)))
        story.append(Paragraph("Answer Export", ps("sub", fontSize=13,
            textColor=C_TXT, fontName="Helvetica", spaceAfter=4)))
        story.append(Paragraph(dt.datetime.now().strftime("%d %B %Y, %H:%M"),
            ps("dt", fontSize=9, textColor=C_MUT, spaceAfter=14)))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=C_ACC, spaceAfter=14))

        story.append(Paragraph("Query", ps("ql", fontSize=9, textColor=C_MUT,
            spaceAfter=4)))
        story.append(Paragraph(_x(query), ps("qq", fontSize=13, textColor=C_TXT,
            fontName="Helvetica-Bold", spaceAfter=14)))

        if data.get("reformulated_query"):
            story.append(Paragraph(
                f"Query rewritten to: {_x(data['reformulated_query'])}",
                ps("ref", fontSize=10, textColor=colors.HexColor("#7c9fff"), spaceAfter=10)))

        story.append(Paragraph("Answer", ps("al", fontSize=9, textColor=C_MUT,
            spaceAfter=6)))
        # Clean markdown for PDF
        import re as _re
        clean_ans = _re.sub(r'\*\*(.+?)\*\*', r'\1', answer)
        clean_ans = _re.sub(r'\*(.+?)\*',     r'\1', clean_ans)
        clean_ans = _x(clean_ans)
        for para in clean_ans.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip().replace("\n"," "),
                    ps("ap", fontSize=11, textColor=C_TXT,
                       fontName="Helvetica", spaceAfter=8, leading=17)))

        scores = data.get("scores")
        if scores:
            story.append(Spacer(1, 0.4*cm))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#D9DEE7"), spaceAfter=8))
            avg = scores.get("average_score", 0) or 0
            story.append(Paragraph(
                f"Retrieval: {scores.get('retrieval_relevance',0)}/5 &nbsp;|&nbsp; "
                f"Grounding: {scores.get('answer_grounding',0)}/5 &nbsp;|&nbsp; "
                f"Quality: {scores.get('answer_quality',0)}/5 &nbsp;|&nbsp; "
                f"Average: {avg}/5",
                ps("sc", fontSize=9, textColor=C_MUT, spaceAfter=4)))
            h_pct = data.get("hallucination_pct", 0)
            story.append(Paragraph(
                f"Hallucination: {h_pct}% flagged",
                ps("hl", fontSize=9,
                   textColor=C_ACC if h_pct == 0 else colors.HexColor("#9A6700"),
                   spaceAfter=4)))

        if data.get("cited_sources"):
            story.append(Paragraph(
                f"Sources cited: {_x(', '.join(data['cited_sources']))}",
                ps("cs", fontSize=9, textColor=colors.HexColor("#7c9fff"), spaceAfter=4)))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Response(pdf_bytes, mimetype="application/pdf",
                        headers={"Content-Disposition":
                                 f"attachment; filename=answer_{ts}.pdf"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"PDF generation failed: {e}"}), 500


# ── Confidence Calibration ────────────────────────────────────────────────────

@app.route("/api/calibration", methods=["GET"])
def calibration_report():
    """
    Compares self-evaluation scores against user feedback ratings.
    Shows how well the self-evaluator predicts actual user satisfaction.
    """
    from database import get_all_logs, get_all_feedback

    try:
        logs     = {l["query"]: l for l in get_all_logs()}
        feedback = get_all_feedback()

        if not feedback:
            return jsonify({"error": "No feedback yet. Rate some answers first."})

        helpful     = [f for f in feedback if f["rating"] == 1]
        not_helpful = [f for f in feedback if f["rating"] == -1]

        def avg_score(items):
            matched = [logs[f["query"]]["average_score"]
                       for f in items if f["query"] in logs]
            return round(sum(matched)/len(matched), 2) if matched else None

        helpful_avg     = avg_score(helpful)
        not_helpful_avg = avg_score(not_helpful)

        # Per-query calibration data
        calibration_data = []
        for f in feedback:
            if f["query"] in logs:
                l = logs[f["query"]]
                calibration_data.append({
                    "query"         : f["query"][:80],
                    "self_eval_score": l["average_score"],
                    "user_rating"   : f["rating"],
                    "hallucination" : round((l["hallucination_score"] or 0)*100),
                    "calibrated"    : (f["rating"] == 1 and l["average_score"] >= 3.5) or
                                      (f["rating"] == -1 and l["average_score"] < 3.5),
                })

        well_calibrated = sum(1 for d in calibration_data if d["calibrated"])
        calibration_pct = round(well_calibrated / len(calibration_data) * 100) \
                          if calibration_data else 0

        # Score buckets vs satisfaction
        buckets = {}
        for d in calibration_data:
            bucket = f"{int(d['self_eval_score'])}-{int(d['self_eval_score'])+1}"
            if bucket not in buckets:
                buckets[bucket] = {"helpful": 0, "not_helpful": 0}
            if d["user_rating"] == 1:
                buckets[bucket]["helpful"] += 1
            else:
                buckets[bucket]["not_helpful"] += 1

        return jsonify({
            "total_rated"       : len(feedback),
            "helpful_count"     : len(helpful),
            "not_helpful_count" : len(not_helpful),
            "avg_score_helpful"    : helpful_avg,
            "avg_score_not_helpful": not_helpful_avg,
            "calibration_pct"   : calibration_pct,
            "well_calibrated"   : well_calibrated,
            "score_buckets"     : buckets,
            "data"              : calibration_data,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Bookmarks ─────────────────────────────────────────────────────────────────

@app.route("/bookmark", methods=["POST"])
def add_bookmark():
    from database import save_bookmark
    data      = request.get_json(silent=True) or {}
    query     = data.get("query",     "").strip()
    answer    = data.get("answer",    "").strip()
    doc_id    = data.get("doc_id",    active_doc_id)
    avg_score = data.get("avg_score", None)
    note      = data.get("note",      None)
    if not query or not answer:
        return jsonify({"error": "query and answer required"}), 400
    bid = save_bookmark(query=query, answer=answer, doc_id=doc_id,
                        avg_score=avg_score, note=note)
    return jsonify({"status": "saved", "id": bid})

@app.route("/bookmarks", methods=["GET"])
def list_bookmarks():
    from database import get_all_bookmarks
    return jsonify({"bookmarks": get_all_bookmarks()})

@app.route("/bookmark/<int:bid>", methods=["DELETE"])
def remove_bookmark(bid):
    from database import delete_bookmark
    delete_bookmark(bid)
    return jsonify({"status": "deleted"})

@app.route("/bookmark/<int:bid>", methods=["PATCH"])
def update_bookmark_note(bid):
    from database import DBSession, Bookmark
    note = (request.get_json(silent=True) or {}).get("note", "")
    db   = DBSession()
    bm   = db.query(Bookmark).filter_by(id=bid).first()
    if bm: bm.note = note; db.commit()
    db.close()
    return jsonify({"status": "updated"})


# ── Document Tags ─────────────────────────────────────────────────────────────

@app.route("/doc-tags", methods=["GET"])
def get_tags():
    from database import get_doc_tags, get_all_tags
    doc_id = request.args.get("doc_id", "").strip()
    if doc_id:
        return jsonify({"tags": get_doc_tags(doc_id)})
    return jsonify({"tags": get_all_tags()})

@app.route("/doc-tags", methods=["POST"])
def add_tag():
    from database import add_doc_tag
    data   = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "").strip()
    tag    = data.get("tag",    "").strip().lower()
    if not doc_id or not tag:
        return jsonify({"error": "doc_id and tag required"}), 400
    add_doc_tag(doc_id, tag)
    return jsonify({"status": "added"})

@app.route("/doc-tags/<int:tid>", methods=["DELETE"])
def remove_tag(tid):
    from database import remove_doc_tag
    remove_doc_tag(tid)
    return jsonify({"status": "deleted"})


# ── Query Analytics ───────────────────────────────────────────────────────────

@app.route("/analytics", methods=["GET"])
def query_analytics():
    """
    Analyses query_logs to return:
    - word frequency across all queries (for word cloud)
    - daily query counts
    - score distribution
    - top queries by frequency
    """
    from database import get_all_logs
    from collections import Counter
    import re as _re

    logs = get_all_logs()
    if not logs:
        return jsonify({"word_freq": {}, "daily": [], "top_queries": [],
                        "score_dist": [], "total": 0})

    stopwords = {"what","is","the","a","an","of","in","how","does","do",
                 "this","that","can","will","are","was","were","to","for",
                 "and","or","it","its","be","by","with","from","about","which"}

    word_counts = Counter()
    for log in logs:
        words = _re.findall(r'\b[a-z]{3,}\b', (log["query"] or "").lower())
        word_counts.update(w for w in words if w not in stopwords)

    # daily counts
    from collections import defaultdict
    daily = defaultdict(int)
    for log in logs:
        day = log["created_at"][:10]
        daily[day] += 1
    daily_list = [{"date": d, "count": c}
                  for d, c in sorted(daily.items())]

    # top queries
    q_counts = Counter(log["query"] for log in logs)
    top_queries = [{"query": q, "count": c}
                   for q, c in q_counts.most_common(10)]

    # score distribution buckets
    buckets = {"1-2": 0, "2-3": 0, "3-4": 0, "4-5": 0}
    for log in logs:
        avg = log["average_score"] or 0
        if   avg < 2: buckets["1-2"] += 1
        elif avg < 3: buckets["2-3"] += 1
        elif avg < 4: buckets["3-4"] += 1
        else:         buckets["4-5"] += 1

    return jsonify({
        "word_freq"  : dict(word_counts.most_common(60)),
        "daily"      : daily_list,
        "top_queries": top_queries,
        "score_dist" : [{"range": k, "count": v} for k, v in buckets.items()],
        "total"      : len(logs),
        "avg_score"  : round(sum(l["average_score"] or 0 for l in logs) / len(logs), 2)
    })


# ── Answer Comparison Mode ────────────────────────────────────────────────────

@app.route("/compare-thresholds", methods=["POST"])
def compare_thresholds():
    """
    Runs the same query at three confidence thresholds (1.0, 3.0, 5.0)
    and returns all three answers for side-by-side comparison.
    """
    data      = request.get_json(silent=True) or {}
    query     = data.get("query", "").strip()
    doc_id    = data.get("doc_id", active_doc_id)

    if not query:
        return jsonify({"error": "query required"}), 400

    results = {}
    try:
        chunks = retrieve(query, top_k=5, doc_id=doc_id)
        if not chunks:
            return jsonify({"query": query, "doc_id": doc_id,
                            "results": {str(t): {"error": "No content found"} for t in (1.0, 3.0, 5.0)}})
        answer = generate(query, chunks)
        scores = evaluate(query, chunks, answer)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    for threshold in [1.0, 3.0, 5.0]:
        results[str(threshold)] = {"answer": answer, "scores": scores,
                                   "passed": scores["average_score"] >= threshold,
                                   "threshold": threshold}
    return jsonify({"query": query, "doc_id": doc_id, "results": results})


# ── Smart Re-index Alert ──────────────────────────────────────────────────────

@app.route("/check-url-changes", methods=["POST"])
def check_url_changes():
    """
    Re-fetches a URL and compares content hash to detect changes.
    Returns: {changed: bool, old_words: int, new_words: int, similarity: float}
    """
    data   = request.get_json(silent=True) or {}
    url    = data.get("url",    "").strip()
    doc_id = data.get("doc_id", "").strip()

    if not url or not doc_id:
        return jsonify({"error": "url and doc_id required"}), 400

    try:
        from scraper import scrape_url
        from page_index import get_conn
        import hashlib

        new_text  = scrape_url(url)
        new_hash  = hashlib.md5(new_text.encode()).hexdigest()
        new_words = len(new_text.split())

        # get old word count from page index
        conn = get_conn(); c = conn.cursor()
        row  = c.execute("SELECT word_count FROM documents WHERE doc_id=?",
                         (doc_id,)).fetchone()
        conn.close()
        old_words = row[0] if row else 0

        # simple change detection: significant word count diff
        diff_pct = abs(new_words - old_words) / max(old_words, 1) * 100
        changed  = diff_pct > 5   # >5% word count change = likely updated

        return jsonify({
            "changed"  : changed,
            "old_words": old_words,
            "new_words": new_words,
            "diff_pct" : round(diff_pct, 1),
            "doc_id"   : doc_id,
            "url"      : url,
            "new_text" : new_text if changed else None
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Export Chat as HTML ───────────────────────────────────────────────────────

@app.route("/export-chat-html", methods=["GET"])
def export_chat_html():
    """Exports chat history as a clean standalone HTML page."""
    from database import get_chat_history, get_all_chat_sessions
    import datetime as dt

    try:
        sid       = request.args.get("session_id", active_session_id)
        msgs      = get_chat_history(limit=200, session_id=sid)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        # Get session name for the title
        session_name = "Chat Export"
        if sid:
            sessions = get_all_chat_sessions()
            match    = next((s for s in sessions if s["id"] == sid), None)
            if match: session_name = match["name"]

        esc = lambda t: _html.escape(str(t or "")).replace("\n", "<br/>")
        msg_html = ""
        for m in msgs:
            meta = m.get("metadata") or {}
            ts   = esc(m.get("created_at", ""))
            if m["role"] == "user":
                msg_html += f'<div class="msg user"><div class="bubble user-bubble">{esc(m["content"])}</div><div class="meta">{ts}</div></div>'
            elif m["role"] == "assistant":
                scores = meta.get("scores") or {}
                avg    = scores.get("average_score")
                h_pct  = round(((meta.get("hallucination") or {}).get("hallucination_score") or 0) * 100)
                chips  = (f'<span class="chip green">Score {avg}/5</span>' if avg is not None else "")
                chips += (f'<span class="chip red">{h_pct}% flagged</span>' if h_pct > 0
                          else '<span class="chip green">0% flagged</span>')
                msg_html += (f'<div class="msg bot"><div class="bubble bot-bubble">{esc(m["content"])}</div>'
                             f'<div class="answer-meta">{chips}</div><div class="meta">{ts}</div></div>')
            elif m["role"] == "fallback":
                msg_html += (f'<div class="msg bot"><div class="bubble fallback-bubble"><div class="fallback-tag">'
                             f'General knowledge, not from your documents</div>{esc(m["content"])}</div>'
                             f'<div class="meta">{ts}</div></div>')
            else:   # greeting / notfound
                msg_html += f'<div class="msg bot"><div class="bubble greeting-bubble">{esc(m["content"])}</div><div class="meta">{ts}</div></div>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Self-RAG chat export {timestamp}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#09090b;color:#f0f0f2;padding:32px 16px;}}
  .header{{max-width:760px;margin:0 auto 28px;border-bottom:1px solid #2e2e36;padding-bottom:16px;}}
  .header h1{{font-size:1.4rem;font-weight:700;color:#00e5a0;margin-bottom:4px;}}
  .header p{{font-size:.82rem;color:#606070;}}
  .chat{{max-width:760px;margin:0 auto;display:flex;flex-direction:column;gap:18px;}}
  .msg{{display:flex;flex-direction:column;gap:4px;}}
  .msg.user{{align-items:flex-end;}}
  .msg.bot {{align-items:flex-start;}}
  .bubble{{padding:12px 16px;border-radius:12px;font-size:.9rem;line-height:1.7;max-width:82%;word-break:break-word;}}
  .user-bubble{{background:#00e5a0;color:#000;border-radius:14px 14px 3px 14px;font-weight:500;}}
  .bot-bubble{{background:#1a1a1f;border:1px solid #2e2e36;border-radius:3px 14px 14px 14px;}}
  .greeting-bubble{{background:#1a1a1f;border:1px solid #3a3a44;border-radius:3px 14px 14px 14px;color:#a0a0ae;}}
  .fallback-bubble{{background:#1a1a1f;border:1px solid #ffb34030;border-radius:3px 14px 14px 14px;}}
  .fallback-tag{{font-size:.7rem;font-weight:700;color:#ffb340;margin-bottom:6px;}}
  .answer-meta{{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px;}}
  .chip{{font-size:.7rem;font-weight:600;padding:2px 8px;border-radius:20px;}}
  .chip.green{{background:rgba(0,229,160,.12);color:#00e5a0;}}
  .chip.red  {{background:rgba(255,92,92,.12); color:#ff5c5c;}}
  .meta{{font-size:.68rem;color:#606070;padding:0 2px;}}
  .footer{{max-width:760px;margin:36px auto 0;border-top:1px solid #2e2e36;padding-top:14px;font-size:.75rem;color:#606070;}}
</style>
</head>
<body>
<div class="header">
  <h1>Self-RAG: {_html.escape(session_name)}</h1>
  <p>Exported: {timestamp} · {len(msgs)} messages</p>
</div>
<div class="chat">{msg_html}</div>
<div class="footer">Generated by Self-RAG — Document AI System</div>
</body>
</html>"""

        return Response(
            html,
            mimetype="text/html",
            headers={"Content-Disposition":
                     f"attachment; filename=selfrag_chat_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.html"}
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Export failed: {e}"}), 500


# ── Research API ──────────────────────────────────────────────────────────────

@app.route("/api/research/ablation", methods=["POST"])
def research_ablation_start():
    data      = request.get_json(silent=True) or {}
    questions = [q for q in data.get("questions", [])
                 if str(q.get("question", "")).strip() and str(q.get("expected", "")).strip()]
    if not questions:
        return jsonify({"error": "Provide questions as [{question, expected}]."}), 400
    if len(questions) > 100:
        return jsonify({"error": "Max 100 questions per run."}), 400
    docs = data.get("doc_ids") or ([active_doc_id] if active_doc_id else None)
    ids  = data.get("configs") or None
    valid = {c[0] for c in research.ABLATION_LADDER}
    if ids and not set(ids) <= valid:
        return jsonify({"error": f"Unknown configuration. Valid: {sorted(valid)}"}), 400
    jid = research.start_ablation_job(questions, docs, ids, _clamp(data.get("threshold", 3.0), 1, 5, 3.0))
    return jsonify({"job_id": jid, "configs": ids or sorted(valid), "questions": len(questions)})


@app.route("/api/research/ladder")
def research_ladder():
    return jsonify({"ladder": [{"id": i, "name": n, "flags": c} for i, n, c in research.ABLATION_LADDER]})


@app.route("/api/research/jobs/<jid>")
def research_job(jid):
    job = research.get_job(jid)
    if not job: return jsonify({"error": "Job not found"}), 404
    out = {k: job[k] for k in ("id", "status", "done", "total", "message", "error")}
    out["elapsed"] = round(time.time() - job["started"], 1)
    if job["status"] in ("done", "stopped") and job["result"]:
        out["rows"]        = job["result"]["rows"]
        out["calibration"] = job["calibration"]
    return jsonify(out)


@app.route("/api/research/jobs/<jid>/stop", methods=["POST"])
def research_job_stop(jid):
    job = research.get_job(jid)
    if not job: return jsonify({"error": "Job not found"}), 404
    job["stop"] = True
    return jsonify({"status": "stopping"})


@app.route("/api/research/jobs/<jid>/export")
def research_job_export(jid):
    job = research.get_job(jid)
    if not job or not job.get("result"):
        return jsonify({"error": "No finished result for this job"}), 404
    fmt = request.args.get("fmt", "json")
    res = job["result"]
    if fmt == "tex":  body, mime, ext = res["latex"], "text/plain", "tex"
    elif fmt == "csv": body, mime, ext = res["csv"], "text/csv", "csv"
    else:
        body = json.dumps({"rows": res["rows"], "records": res["records"],
                           "calibration": job["calibration"]}, indent=2, default=str)
        mime, ext = "application/json", "json"
    return Response(body, mimetype=mime,
                    headers={"Content-Disposition": f"attachment; filename=ablation_{jid}.{ext}"})


@app.route("/api/research/verify", methods=["POST"])
def research_verify():
    """Cascaded sentence verification (+ optional repair) for any query/answer pair."""
    data   = request.get_json(silent=True) or {}
    query  = data.get("query", "").strip(); answer = data.get("answer", "").strip()
    scope  = data.get("doc_ids") or ([active_doc_id] if active_doc_id else None)
    if not query or not answer: return jsonify({"error": "query and answer required"}), 400
    try:
        chunks = research.hybrid_retrieve(query, top_k=5, doc_ids=scope)
        if not chunks: return jsonify({"error": "No evidence found for this query"}), 404
        v = research.verify_answer(answer, chunks)
        out = {"verification": v}
        if data.get("repair") and v["unsupported_count"]:
            r = research.repair_answer(query, answer, chunks, v)
            out["repair"] = {k: r[k] for k in ("answer", "method", "changed", "before", "after")}
            out["repair"]["verification"] = r["verification"]
        return jsonify(out)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500


@app.route("/api/research/calibration-live")
def research_calibration_live():
    """Calibration of the live system: self-eval score vs. user thumbs (no ground truth needed)."""
    from database import get_all_logs, get_all_feedback
    logs = {l["id"]: l for l in get_all_logs()}
    by_q = {l["query"]: l for l in logs.values()}
    recs = []
    for f in get_all_feedback():
        l = logs.get(f.get("query_id")) or by_q.get(f["query"])
        if not l: continue
        recs.append({"correct": 1 if f["rating"] == 1 else 0,
                     "features": [(l["retrieval_relevance"] or 0) / 5, (l["answer_grounding"] or 0) / 5,
                                  (l["answer_quality"] or 0) / 5, (l["average_score"] or 0) / 5]})
    return jsonify(research.calibration_report(recs))


if __name__ == "__main__":
    # use_reloader=False: the debug reloader starts the app twice, which loads the
    # embedding model and the cross-encoder twice and races on the SQLite files.
    app.run(debug=False, threaded=True, use_reloader=False, port=int(os.getenv("PORT", 5000)))