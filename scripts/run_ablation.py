"""
Headless, reproducible ablation run for the paper.

  python scripts/run_ablation.py --docs corpus/ --qa qa.json --out results/

corpus/ : .txt or .pdf files, one document each (doc_id = file name)
qa.json : [{"question": "...", "expected": "..."}, ...]
Writes results.json, results.csv and results.tex (booktabs table) to --out.
Set JUDGE_MODEL in .env to a model different from GEN_MODEL for unbiased judging.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from indexer import index_document, index_pdf          # noqa: E402
import research                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True); ap.add_argument("--qa", required=True)
    ap.add_argument("--out", default="results"); ap.add_argument("--configs", default="")
    ap.add_argument("--threshold", type=float, default=3.0)
    a = ap.parse_args()

    ids = []
    for name in sorted(os.listdir(a.docs)):
        path, doc_id = os.path.join(a.docs, name), os.path.splitext(name)[0]
        if name.lower().endswith(".pdf"):   index_pdf(doc_id, path)
        elif name.lower().endswith(".txt"): index_document(doc_id, open(path, encoding="utf-8").read())
        else: continue
        ids.append(doc_id)
    qa = json.load(open(a.qa, encoding="utf-8"))
    print(f"{len(ids)} documents, {len(qa)} questions")

    res = research.run_ablation(
        qa, ids, a.configs.split(",") if a.configs else None, a.threshold,
        on_progress=lambda d, t, m: print(f"[{d}/{t}] {m}"))
    last = list(res["records"])[-1]
    cal = research.calibration_report([r for r in res["records"][last] if not r.get("error")])

    os.makedirs(a.out, exist_ok=True)
    json.dump({"rows": res["rows"], "records": res["records"], "calibration": cal},
              open(os.path.join(a.out, "results.json"), "w"), indent=2, default=str)
    open(os.path.join(a.out, "results.csv"), "w").write(research.to_csv(res["rows"]))
    open(os.path.join(a.out, "results.tex"), "w").write(research.to_latex(res["rows"]))
    print(f"Saved to {a.out}/")

if __name__ == "__main__":
    main()
