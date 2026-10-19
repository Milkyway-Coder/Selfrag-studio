"""Route-level smoke tests with stubbed embedder + LLM."""
import os, sys, tempfile, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "stubs")); sys.path.insert(0, ROOT)
work = tempfile.mkdtemp(); os.chdir(work); os.environ["GROQ_API_KEY"] = "x"
os.makedirs("templates", exist_ok=True)
import app as A
c = A.app.test_client()
TEXT = ("Flask is a lightweight web framework for Python. It is designed to make getting started quick and easy, with the ability to scale up to complex applications.\n\n"
        "Flask was created by Armin Ronacher and is based on the Werkzeug toolkit and the Jinja2 template engine. It does not require particular tools or libraries.\n\n"
        "Flask has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.")

def sse(resp):
    out = []
    for line in resp.get_data(as_text=True).split("\n\n"):
        if line.startswith("data: "): out.append(json.loads(line[6:]))
    return out

def test_flow():
    assert c.get("/api/status").json["docs"] == 0
    r = c.post("/index-doc", json={"doc_id": "flask-intro", "text": TEXT}); assert r.status_code == 200, r.data
    r = c.post("/index-doc", json={"doc_id": "second", "text": TEXT.replace("Flask", "Quart")}); assert r.status_code == 200
    assert sorted(c.get("/indexed-docs").json["docs"]) == ["flask-intro", "second"]
    info = c.get("/doc-info/flask-intro").json; assert info["chunk_count"] >= 1 and info["word_count"] > 20
    # greeting vs real question starting with "hi"
    ev = sse(c.post("/chat-stream", json={"query": "hello"})); assert ev[-1].get("is_greeting")
    ev = sse(c.post("/set-active-doc", json={"doc_id": "flask-intro"}) and c.post("/chat-stream", json={"query": "history of Flask creation"}))
    assert not ev[-1].get("is_greeting"), ev[-1]
    # full stream, classic + hybrid/repair
    for body in ({"query": "Who created Flask and what is it based on?"},
                 {"query": "Who created Flask and what is it based on?", "hybrid": True, "repair": True}):
        ev = sse(c.post("/chat-stream", json=body)); types = [e["type"] for e in ev]
        assert "token" in types and types[-1] == "done", types
        done = ev[-1]; assert done["scores"]["average_score"] > 0 and done["query_id"]
    assert "repaired" in types, types
    # feedback + calibration live
    assert c.post("/feedback", json={"query": "Who created Flask and what is it based on?", "answer": "x", "rating": 1, "query_id": done["query_id"]}).status_code == 200
    # sessions & history isolation
    sid1 = c.get("/chat-sessions/active").json["session_id"]
    h1 = c.get(f"/chat-history?session_id={sid1}").json["messages"]; assert len(h1) >= 2
    sid2 = c.post("/chat-sessions", json={"name": "Second"}).json["session_id"]
    assert c.get(f"/chat-history?session_id={sid2}").json["messages"] == []
    # delete really deletes
    r = c.delete("/doc/second"); assert r.json["status"] == "success"
    assert c.get("/indexed-docs").json["docs"] == ["flask-intro"]
    # other routes
    for url in ["/analytics", "/api/logs", "/bookmarks", "/feedback", "/page-index-stats", "/sessions",
                "/api/query-analytics", "/api/calibration", "/doc-tags", "/api/research/ladder",
                "/doc-summary/flask-intro", "/extract-entities/flask-intro", "/export-pdf",
                "/export-chat-html", "/api/query-suggestions?q=fl", "/page-index-terms?doc_id=flask-intro",
                "/page-index-lookup?term=flask"]:
        r = c.get(url); assert r.status_code == 200, (url, r.status_code, r.data[:200])
    r = c.post("/export-answer", json={"query": "a <b> c", "answer": "x < y & z", "scores": {"average_score": 4}}); assert r.status_code == 200
    r = c.post("/compare-retrievers", json={"query": "Who created Flask?"}); assert r.status_code == 200
    r = c.post("/compare-docs", json={"query": "Who created Flask?", "doc_id_a": "flask-intro", "doc_id_b": "flask-intro"}); assert r.status_code == 200
    r = c.post("/api/research/verify", json={"query": "Who created Flask?", "answer": "Flask was created by Armin Ronacher. The moon is cheese made.", "repair": True}); assert r.status_code == 200, r.data
    assert c.post("/bookmark", json={"query": "q", "answer": "a"}).status_code == 200
    assert c.get("/nope").status_code == 404 and c.post("/api/status").status_code == 405

def test_research_job():
    import time
    qs = [{"question": "Who created Flask and what is it based on?", "expected": "Armin Ronacher"}] * 10
    jid = c.post("/api/research/ablation", json={"questions": qs, "configs": ["A0", "A5"]}).json["job_id"]
    for _ in range(120):
        j = c.get(f"/api/research/jobs/{jid}").json
        if j["status"] != "running": break
        time.sleep(0.5)
    assert j["status"] == "done", j
    assert len(j["rows"]) == 2 and j["calibration"]
    for fmt in ("csv", "tex", "json"):
        assert c.get(f"/api/research/jobs/{jid}/export?fmt={fmt}").status_code == 200

if __name__ == "__main__":
    test_flow(); print("PASS flow"); test_research_job(); print("PASS research job"); shutil.rmtree(work, ignore_errors=True)
