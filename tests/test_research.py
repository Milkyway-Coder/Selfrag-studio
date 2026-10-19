"""Offline unit + integration tests (uses tests/stubs for the embedder and the LLM)."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "stubs"))
sys.path.insert(0, ROOT)
work = tempfile.mkdtemp(); os.chdir(work)
os.environ["GROQ_API_KEY"] = "x"

import numpy as np
import research as R
from indexer import index_document, delete_document, get_indexed_docs

TEXT_A = """Flask is a lightweight web framework for Python. It is designed to make getting started quick and easy, with the ability to scale up to complex applications.

Flask was created by Armin Ronacher and is based on the Werkzeug toolkit and the Jinja2 template engine. It does not require particular tools or libraries.

Flask has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions."""
TEXT_B = """PostgreSQL is a powerful open source relational database system. It has more than thirty five years of active development and a strong reputation for reliability.

PostgreSQL supports advanced data types, full text search, and transactional DDL. Many organisations choose it for analytics workloads and geospatial data."""

def test_math():
    assert abs(R.ece([1, 1, 0, 0], [1, 1, 0, 0])) < 1e-9
    assert R.ece([0.9] * 10, [1] * 5 + [0] * 5) > 0.3
    assert R.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    pts, aurc = R.risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert pts[-1]["risk"] == 0.5 and aurc < 0.5
    ranked = R.rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert {ranked[0][0], ranked[1][0]} == {"a", "b"}
    lo, hi = R.wilson(8, 10); assert 0.4 < lo < 0.8 < hi <= 1
    assert R.token_f1("Armin Ronacher created it", "Armin Ronacher") > 0.5
    assert R.paired_bootstrap_p([1]*10, [0]*10) is not None
    m = R.fit_logistic(np.array([[0.1], [0.2], [0.8], [0.9]] * 5), np.array([0, 0, 1, 1] * 5))
    p = R.predict_logistic(m, np.array([[0.1], [0.9]]))
    assert p[0] < 0.5 < p[1]

def test_bm25():
    bm = R._BM25([["flask", "python"], ["postgres", "database"], ["flask", "flask", "jinja"]])
    s = bm.scores(["flask"])
    assert s[2] > s[0] > s[1] == 0

def test_index_retrieve_delete():
    index_document("flask-intro", TEXT_A); index_document("pg-intro", TEXT_B)
    assert get_indexed_docs() == ["flask-intro", "pg-intro"]
    ch = R.hybrid_retrieve("Who created Flask?", top_k=3, doc_ids=["flask-intro"])
    assert ch and "Ronacher" in ch[0]
    ch = R.hybrid_retrieve("transactional DDL", top_k=2)
    assert any("PostgreSQL" in c for c in ch)
    for cid, name, cfg in R.ABLATION_LADDER:
        got, _ = R.retrieve_cfg("Who created Flask?", cfg, ["flask-intro"], top_k=3)
        assert got, cid
    delete_document("pg-intro")
    assert get_indexed_docs() == ["flask-intro"]
    from page_index import get_all_indexed_docs
    assert [d["doc_id"] for d in get_all_indexed_docs()] == ["flask-intro"]

def test_verify_and_repair():
    chunks = [TEXT_A]
    ans = "Flask was created by Armin Ronacher and is based on Werkzeug. The moon is made of cheese."
    v = R.verify_answer(ans, chunks)
    flags = {s["text"]: s["supported"] for s in v["sentences"]}
    assert flags["The moon is made of cheese."] is False
    assert v["total_count"] == 2 and v["unsupported_count"] == 1
    out = R.repair_answer("Who created Flask?", ans, chunks, v)
    assert out["changed"] and out["after"] < out["before"]
    strict = R.verify_answer(ans, chunks, mode="llm_only")
    assert strict["llm_calls"] == 1

def test_ablation_and_calibration():
    qs = [{"question": "Who created Flask and what is it based on?", "expected": "Armin Ronacher"}] * 12
    res = R.run_ablation(qs, ["flask-intro"])
    assert len(res["rows"]) == 6
    assert all(r["n"] == 12 for r in res["rows"])
    assert "\\begin{tabular}" in R.to_latex(res["rows"])
    assert R.to_csv(res["rows"]).count("\n") == 6
    cal = R.calibration_report(res["records"]["A5"])
    assert "raw" in cal and "calibrated" in cal

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    shutil.rmtree(work, ignore_errors=True)
