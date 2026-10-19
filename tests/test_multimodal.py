import os, sys, tempfile, shutil, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "stubs")); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
work = tempfile.mkdtemp(); os.chdir(work); os.environ["GROQ_API_KEY"] = "x"
import make_fixtures, ingest, tabular, multimodal
from indexer import index_blocks, delete_document, get_indexed_docs, meta_for_texts, list_figures
import research
fx = make_fixtures.make(os.path.join(work, "fx"))
kinds = lambda r: sorted({b["type"] for b in r["blocks"]})

def test_pdf():
    r = ingest.parse_file(fx["pdf"], "paper", "paper.pdf")
    print("pdf stats", r["stats"], kinds(r), r["warnings"])
    assert r["stats"]["tables"] >= 1 and r["stats"]["figures"] >= 2, r["stats"]
    tb = [b for b in r["blocks"] if b["type"] == "table"][0]
    assert "ResNet-18" in tb["text"] and "Table 1" in tb["text"], tb["text"]
    figs = [b for b in r["blocks"] if b["type"] == "figure"]
    assert any(b["meta"]["fig_label"] == "1" for b in figs) and any(b["meta"]["fig_label"] == "2" for b in figs)
    f1 = [b for b in figs if b["meta"]["fig_label"] == "1"][0]
    assert "Training loss over epochs" in f1["text"] and "decreases quickly" in f1["text"]   # caption + sentence that cites it
    assert os.path.exists(os.path.join("figures", "paper", f1["meta"]["fig_id"] + ".png"))
    # the table text must not be duplicated inside the running prose
    prose = " ".join(b["text"] for b in r["blocks"] if b["type"] == "text")
    assert "ResNet-18" not in prose and "mc^2" in prose

def test_pdf_deep():
    r = ingest.parse_file(fx["pdf"], "paper-deep", "paper.pdf", vision_on=True)
    v = [b for b in r["blocks"] if b["type"] == "vision"]
    assert v and "E = mc^2" in v[0]["text"], kinds(r)
    assert r["stats"]["vision_candidates"] == 2 and r["stats"]["vision_pages"] == 2    # both pages hold a table/figure/formula

def test_office():
    d = ingest.parse_file(fx["docx"], "report", "report.docx")
    assert d["stats"]["tables"] == 1 and d["stats"]["figures"] == 1 and any("Quick sort" in b["text"] for b in d["blocks"] if b["type"] == "table")
    assert any(b["type"] == "figure" and "Running time" in b["text"] for b in d["blocks"])
    p = ingest.parse_file(fx["pptx"], "deck", "deck.pptx")
    assert p["stats"]["figures"] == 1 and any("Slide 2" in b["text"] and "converges" in b["text"] for b in p["blocks"])

def test_sheet_and_dsl():
    for k in ("xlsx", "csv"):
        r = ingest.parse_file(fx[k], "staff-" + k, os.path.basename(fx[k]))
        assert r["stats"]["rows"] == 40 and "sheet" in kinds(r) and "table" in kinds(r), r["stats"]
    sh = [b for b in r["blocks"] if b["type"] == "sheet"][0]["text"]
    assert "salary" in sh and "mean" in sh
    # exact maths: mean salary by dept
    out = tabular.answer_from_tables("what is the average salary by dept?", ["staff-xlsx"])
    assert out, "planner should fire"
    ctx, detail, spec = out
    assert "mean(salary)" in ctx and "eng" in ctx, ctx
    import pandas as pd
    df = list(tabular.load_sheets("staff-xlsx").values())[0]
    expect = round(df[df.dept == "eng"].salary.mean(), 1)
    assert str(int(expect)) in ctx.replace(",", ""), (expect, ctx)
    assert tabular.answer_from_tables("tell me a story", ["staff-xlsx"]) is None    # not an aggregation question
    # filters + safe failure on a made-up column
    res, _ = tabular.run_spec(df, {"filters": [{"column": "salary", "op": "gt", "value": 80000}], "aggregations": [{"column": "name", "fn": "count"}]})
    assert int(res.iloc[0, 0]) == 9
    try: tabular.run_spec(df, {"filters": [{"column": "nope", "op": "eq", "value": 1}]}); assert False
    except ValueError: pass

def test_code():
    c = ingest.parse_file(fx["py"], "stack", "stack.py")
    assert kinds(c) == ["code"] and "Stack" in c["blocks"][0]["text"] and "helper_3" in c["blocks"][0]["text"]
    assert all(("lines" in b["text"]) for b in c["blocks"][1:]) and len(c["blocks"]) >= 3
    z = ingest.parse_file(fx["zip"], "proj", "project.zip")
    names = " ".join(b["text"] for b in z["blocks"])
    assert "node_modules" not in names and "proj/main.py" in names and "app.js" in names
    n = ingest.parse_file(fx["ipynb"], "nb", "nb.ipynb")
    assert any("np.mean" in b["text"] and "2.0" in b["text"] for b in n["blocks"])
    try: ingest.parse_file(fx["pdf"], "x", "evil.exe"); assert False
    except ValueError: pass

def test_index_retrieve_figures():
    for did, key in (("paper", "pdf"), ("staff", "csv"), ("stack", "py")):
        r = ingest.parse_file(fx[key], did, os.path.basename(fx[key]))
        index_blocks(did, r["blocks"], r["kind"], r["stats"])
    assert {"paper", "staff", "stack"} <= set(get_indexed_docs())
    from page_index import get_all_doc_meta
    assert get_all_doc_meta()["paper"]["kind"] == "pdf" and get_all_doc_meta()["staff"]["kind"] == "sheet"
    # table chunk is retrievable and carries its page
    chunks = research.hybrid_retrieve("accuracy of ResNet-18 VGG-16 MobileNet", top_k=4, doc_ids=["paper"])
    metas = meta_for_texts(chunks, ["paper"])
    assert any(m and m["block_type"] == "table" and m["page"] == 1 for m in metas), [m and m["block_type"] for m in metas]
    assert any("table" in multimodal.passage_label(m) and "page 1" in multimodal.passage_label(m) for m in metas if m)
    # figures: named, visual intent, and none for a plain question
    f = multimodal.related_figures("Explain Figure 1", ["paper"], [])
    assert f and f[0]["label"] == "1" and f[0]["url"].startswith("/figure/paper/") and "named" in f[0]["reason"]
    f = multimodal.related_figures("show me the pipeline architecture diagram", ["paper"], [])
    assert any(x["label"] == "2" for x in f), f
    assert multimodal.related_figures("what learning rate was used", ["paper"], []) == []
    assert len(list_figures("paper")) >= 2
    # code chunks keep file + line range
    cc = research.hybrid_retrieve("Stack push pop method", top_k=3, doc_ids=["stack"])
    cm = meta_for_texts(cc, ["stack"])
    assert any(m and m["block_type"] == "code" and m["file"] == "stack.py" for m in cm)
    delete_document("paper")
    assert not os.path.exists(os.path.join("figures", "paper")) and "paper" not in get_indexed_docs()

def test_memory():
    import conversation as C
    hist = []
    for i in range(10):
        hist += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    older, recent = C.split(hist, 4)
    assert len(recent) == 8 and len(older) == 12 and recent[0]["content"] == "question 6"
    assert C.summarize(older) and C.summarize(older) == C.summarize(older)
    assert C.looks_like_follow_up("what about the second one?", hist) and not C.looks_like_follow_up("Explain how gradient descent updates weights in a network", hist)
    assert C.condense("and who made it?", recent) == "Who created Flask and what is it based on?"
    assert len(C.clean(hist * 5)) == 60

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
    shutil.rmtree(work, ignore_errors=True)
