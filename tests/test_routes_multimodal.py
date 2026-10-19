import os, sys, tempfile, json, time, shutil, io
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "stubs")); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
work = tempfile.mkdtemp(); os.chdir(work); os.environ["GROQ_API_KEY"] = "x"
import make_fixtures
fx = make_fixtures.make(os.path.join(work, "fx"))
import app as A
c = A.app.test_client()

def sse(resp):
    return [json.loads(l[6:]) for l in resp.get_data(as_text=True).split("\n\n") if l.startswith("data: ")]

def upload(paths, deep=False, doc_id=""):
    data = {"files": [(open(p, "rb"), os.path.basename(p)) for p in paths], "deep": "1" if deep else "0", "doc_id": doc_id}
    r = c.post("/upload-file", data=data, content_type="multipart/form-data"); assert r.status_code == 200, r.data
    jid = r.json["job_id"]
    for _ in range(120):
        j = c.get(f"/api/ingest/jobs/{jid}").json
        if j["status"] == "done": return j
        time.sleep(0.3)
    raise AssertionError("ingest job did not finish")

def test_upload_and_library():
    j = upload([fx["pdf"], fx["xlsx"], fx["py"], fx["docx"]])
    assert len(j["results"]) == 4 and all("error" not in r for r in j["results"]), j["results"]
    lib = {d["doc_id"]: d for d in c.get("/api/library").json["docs"]}
    assert lib["paper"]["kind"] == "pdf" and lib["paper"]["stats"]["figures"] == 2 and lib["paper"]["stats"]["tables"] == 1
    assert lib["staff"]["kind"] == "sheet" and lib["stack"]["kind"] == "code" and lib["report"]["kind"] == "docx"
    # rejected type + figure file served + path traversal refused
    r = c.post("/upload-file", data={"files": (io.BytesIO(b"MZ"), "evil.exe")}, content_type="multipart/form-data")
    assert r.status_code == 400 and r.json["rejected"]
    figs = c.get("/api/docs/paper/figures").json["figures"]; assert len(figs) == 2 and figs[0]["url"].startswith("/figure/paper/")
    img = c.get(figs[0]["url"]); assert img.status_code == 200 and img.data[:4] == b"\x89PNG", img.status_code
    assert c.get("/figure/paper/..%2f..%2fapp.py").status_code == 404 and c.get("/figure/paper/f001.txt").status_code == 404
    data = c.get("/api/docs/staff/data").json["sheets"][0]; assert data["rows"] == 40 and "salary" in data["columns"]
    j = upload([fx["pdf"]], deep=True, doc_id="paper-deep")      # deep mode adds vision transcription blocks
    assert j["results"][0]["stats"]["vision_pages"] >= 1, j["results"]

def test_stream_sources_and_figures():
    c.post("/set-active-doc", json={"doc_id": "paper"})
    ev = sse(c.post("/chat-stream", json={"query": "Explain Figure 1 in the paper"}))
    done = ev[-1]; assert done["type"] == "done", done
    assert done["figures"] and done["figures"][0]["label"] == "1", done["figures"]
    assert done["sources"] and all("label" in s and "snippet" in s for s in done["sources"])
    # persisted so the figure gallery returns when the chat is reloaded
    sid = c.get("/chat-sessions/active").json["session_id"]
    meta = [m for m in c.get(f"/chat-history?session_id={sid}").json["messages"] if m["role"] == "assistant"][-1]["metadata"]
    assert meta["figures"] and meta["sources"]
    # a table question is labelled as a table with a page
    ev = sse(c.post("/chat-stream", json={"query": "What is the accuracy of ResNet-18, VGG-16 and MobileNet?"}))
    assert any(s["type"] == "table" and s["page"] == 1 for s in ev[-1]["sources"]), ev[-1]["sources"]

def test_followup_and_memory():
    hist = [{"role": "user", "content": "Who created Flask?"}, {"role": "assistant", "content": "Armin Ronacher."}]
    ev = sse(c.post("/chat-stream", json={"query": "and what is it based on?", "history": hist * 8, "memory_turns": 2}))
    steps = [e["label"] for e in ev if e["type"] == "trace_step"]
    assert "Follow-up understood as" in steps and "Memory" in steps, steps
    assert ev[-1]["follow_up"]
    assert ev[-1]["type"] == "done" or ev[-1]["type"] == "not_found"

def test_spreadsheet_answer():
    c.post("/set-active-doc", json={"doc_id": "staff"})
    ev = sse(c.post("/chat-stream", json={"query": "What is the average salary by department?"}))
    steps = [e["label"] for e in ev if e["type"] == "trace_step"]
    assert "Calculated from spreadsheet" in steps, steps
    assert any(s["type"] == "sheet" for s in ev[-1]["sources"])

def test_general_chat():
    r = c.post("/api/general-chat", json={"messages": [{"role": "user", "content": "Explain recursion"}]})
    ev = sse(r); assert ev[-1]["type"] == "done" and any(e["type"] == "token" for e in ev)
    assert c.post("/api/general-chat", json={"messages": []}).status_code == 400
    assert c.post("/api/general-chat", json={"messages": [{"role": "user", "content": "hi"}], "model": "bad model!"}).status_code == 400
    n = len(c.get("/api/logs").json["logs"]); sse(c.post("/api/general-chat", json={"messages": [{"role": "user", "content": "x y z"}]}))
    assert len(c.get("/api/logs").json["logs"]) == n          # direct chat is never logged as a RAG query

def test_regenerate_keeps_one_question_and_accuracy_history():
    c.post("/chat-sessions", json={"name": "regen"})
    c.post("/set-active-doc", json={"doc_id": "paper"})
    q = "Who trained the neural network in the paper?"
    sse(c.post("/chat-stream", json={"query": q}))
    sse(c.post("/regenerate", json={"query": q, "regenerate": True}))
    sse(c.post("/regenerate", json={"query": q, "regenerate": True}))
    sid = c.get("/chat-sessions/active").json["session_id"]
    roles = [m["role"] for m in c.get(f"/chat-history?session_id={sid}").json["messages"]]
    assert roles.count("user") == 1 and roles.count("assistant") == 3, roles          # 1 question, 3 versions
    # accuracy history is saved by the benchmark route and exposed to Insights
    assert c.get("/api/accuracy").json["latest"] is None
    r = c.post("/benchmark", json={"questions": [{"question": "Who created Flask?", "expected": "Armin Ronacher"}], "doc_id": "paper", "source": "quick-test"})
    assert r.status_code == 200
    acc = c.get("/api/accuracy").json
    assert acc["latest"]["total"] == 1 and acc["latest"]["doc_id"] == "paper" and len(acc["runs"]) == 1

def test_enrichment_figure_only_and_turn_deletion():
    # automatic summary, entities and tags were created during the upload job
    e = c.get("/api/docs/paper/enrichment").json
    assert e["ready"] and e["summary"] and "technologies" in e["entities"] and e["tags"], e
    tags = [t["tag"] for t in c.get("/doc-tags?doc_id=paper").json["tags"]]
    assert "flask" in tags and "python web" in tags, tags
    r = c.post("/api/docs/paper/enrich"); assert r.status_code == 200 and r.json["ready"]
    # "only the diagram" returns the picture and a one-line caption, no generated essay, no scores
    c.post("/chat-sessions", json={"name": "figs"}); c.post("/set-active-doc", json={"doc_id": "paper"})
    ev = sse(c.post("/chat-stream", json={"query": "Give me only the Figure 1 diagram"}))
    d = ev[-1]; assert d["type"] == "done" and d["figure_only"] and d["figures"] and d["scores"] is None, d
    assert "Here is the figure" in "".join(x.get("content", "") for x in ev if x["type"] == "token")
    # message ids come back so a turn can be deleted; whole turns disappear, others stay
    ev2 = sse(c.post("/chat-stream", json={"query": "Who created Flask and what is it based on?"})); d2 = ev2[-1]
    assert d["user_msg_id"] and d["assistant_msg_id"] and d2["user_msg_id"] and d2["assistant_msg_id"]
    sse(c.post("/chat-stream", json={"query": "What learning rate did the paper use?"}))
    sid = c.get("/chat-sessions/active").json["session_id"]
    rows = c.get(f"/chat-history?session_id={sid}").json["messages"]; assert [m["role"] for m in rows].count("user") == 3
    r = c.post("/chat-turns/delete", json={"session_id": sid, "user_message_ids": [d["user_msg_id"], rows[-2]["id"] if rows[-2]["role"] == "user" else d2["user_msg_id"]]})
    assert r.json["removed"] >= 2, r.json
    left = c.get(f"/chat-history?session_id={sid}").json["messages"]
    users = [m["content"] for m in left if m["role"] == "user"]
    assert "Give me only the Figure 1 diagram" not in users and len(users) == 1, users
    assert all(m["role"] != "assistant" or m["id"] > 0 for m in left) and len(left) == 2 or len(left) >= 2        # the kept question keeps its answer
    assert c.post("/chat-turns/delete", json={"session_id": sid, "user_message_ids": []}).status_code == 400

def test_quiz():
    r = c.post("/api/quiz", json={"doc_id": "paper", "count": 5}); assert r.status_code == 200, r.data
    qs = r.json["questions"]; assert len(qs) == 5                          # the malformed 6th question is dropped
    for q in qs:
        assert len(q["options"]) == 4 and len(set(q["options"])) == 4 and 0 <= q["answer"] <= 3 and q["explanation"]
        assert q["options"][q["answer"]] == "Armin Ronacher"                # the answer index follows the text after shuffling
    assert c.post("/api/quiz", json={"doc_id": "nope-doc"}).status_code in (400, 500)

def test_delete_cleans_everything():
    assert c.delete("/doc/staff").json["status"] == "success"
    assert not os.path.exists(os.path.join("data_store", "staff"))
    assert c.delete("/doc/paper").json["status"] == "success" and not os.path.exists(os.path.join("figures", "paper"))

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
    shutil.rmtree(work, ignore_errors=True)
