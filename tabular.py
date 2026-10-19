"""
tabular.py — spreadsheets and CSV files.

Retrieval over row chunks cannot answer "what is the average salary by department?", because the answer
depends on EVERY row. So sheets get two things:
  1. a statistics chunk (columns, types, min/max/mean, top values) that is retrievable like any text, and
  2. an exact query path: the LLM plans a small JSON spec (filters, group-by, aggregations), and pandas
     executes it. No model-written code is ever executed.
"""
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd
from llm import chat_text, GEN_MODEL

DATA_DIR = Path(os.getenv("DATA_DIR", "data_store"))
MAX_ROWS_STORED = 200_000

AGG_RE = re.compile(
    r"\b(average|mean|median|total|sum|count|how many|number of|maximum|minimum|max|min|highest|lowest|"
    r"top \d*|bottom \d*|most|least|per|by each|group|percent\w*|ratio|rank\w*|sort\w*|distinct|unique|"
    r"greater|less than|more than|above|below|between|filter|where|list all|which \w+ (has|have|had))\b", re.I)


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def save_sheet(doc_id, sheet, df):
    d = DATA_DIR / safe(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    df.head(MAX_ROWS_STORED).to_csv(d / f"{safe(sheet)}.csv", index=False)


def load_sheets(doc_id):
    d = DATA_DIR / safe(doc_id)
    out = {}
    if d.is_dir():
        for p in sorted(d.glob("*.csv")):
            try:
                out[p.stem] = pd.read_csv(p)
            except Exception:
                pass
    return out


def delete_data(doc_id):
    shutil.rmtree(DATA_DIR / safe(doc_id), ignore_errors=True)


def has_data(doc_id):
    return (DATA_DIR / safe(doc_id)).is_dir()


# ── describing ────────────────────────────────────────────────────────────────

def to_markdown(df, max_rows=None):
    d = df if max_rows is None else df.head(max_rows)
    cols = [str(c) for c in d.columns]
    esc = lambda v: ("" if pd.isna(v) else str(v)).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(lines)


def clean_frame(df):
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    df.columns = [str(c).strip() if str(c).strip() else f"col{i}" for i, c in enumerate(df.columns)]
    return df


def describe_frame(name, df):
    lines = [f'Sheet "{name}": {len(df):,} rows × {len(df.columns)} columns.', "Columns:"]
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            lines.append(f"- {c} (number; {miss} missing): min {s.min():g}, max {s.max():g}, "
                         f"mean {s.mean():.4g}, median {s.median():.4g}, sum {s.sum():.6g}")
        else:
            vc = s.astype(str).value_counts().head(5)
            top = ", ".join(f"{k} ({v})" for k, v in vc.items())
            lines.append(f"- {c} (text; {miss} missing; {s.nunique()} unique): top {top}")
    lines += ["First rows:", to_markdown(df, 5)]
    return "\n".join(lines)


# ── safe query execution ──────────────────────────────────────────────────────

AGG_FNS = {"sum", "mean", "min", "max", "count", "nunique", "median"}
OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "isnull", "notnull"}


def _col(df, name):
    if name in df.columns:
        return name
    low = {str(c).lower(): c for c in df.columns}
    if str(name).lower() in low:
        return low[str(name).lower()]
    raise ValueError(f"Unknown column: {name}")


def run_spec(df, spec):
    """Executes a validated spec. Returns (result_df, human-readable description)."""
    d, notes = df, []
    for f in spec.get("filters") or []:
        c, op, v = _col(d, f.get("column")), f.get("op"), f.get("value")
        if op not in OPS:
            raise ValueError(f"Unsupported filter operator: {op}")
        s = d[c]
        if op in ("isnull", "notnull"):
            d = d[s.isna() if op == "isnull" else s.notna()]
        elif op == "contains":
            d = d[s.astype(str).str.contains(str(v), case=False, regex=False, na=False)]
        elif op == "in":
            vals = v if isinstance(v, list) else [v]
            d = d[s.astype(str).str.lower().isin([str(x).lower() for x in vals])] if not pd.api.types.is_numeric_dtype(s) \
                else d[s.isin([float(x) for x in vals])]
        else:
            if pd.api.types.is_numeric_dtype(s):
                v = float(v)
            else:
                s, v = s.astype(str).str.lower(), str(v).lower()
            m = {"eq": s == v, "ne": s != v, "gt": s > v, "gte": s >= v, "lt": s < v, "lte": s <= v}[op]
            d = d[m]
        notes.append(f"{c} {op} {v}")
    group = [_col(d, g) for g in (spec.get("group_by") or [])]
    aggs = spec.get("aggregations") or []
    if aggs:
        named = {}
        for a in aggs:
            fn = a.get("fn")
            if fn not in AGG_FNS:
                raise ValueError(f"Unsupported aggregation: {fn}")
            col = _col(d, a.get("column"))
            if fn in ("sum", "mean", "median") and not pd.api.types.is_numeric_dtype(d[col]):
                d = d.assign(**{col: pd.to_numeric(d[col], errors="coerce")})
            named[f"{fn}({col})"] = (col, fn)
        res = d.groupby(group, dropna=False).agg(**named).reset_index() if group else \
            pd.DataFrame({k: [getattr(d[c], f)() if f != "nunique" else d[c].nunique()] for k, (c, f) in named.items()})
    else:
        res = d
        if spec.get("select"):
            res = res[[_col(res, c) for c in spec["select"]]]
    sb = spec.get("sort_by")
    if sb:
        key = sb if sb in res.columns else _col(res, sb)
        res = res.sort_values(key, ascending=bool(spec.get("ascending", False)))
    limit = max(1, min(int(spec.get("limit") or 20), 50))
    return res.head(limit), (("filters: " + "; ".join(notes)) if notes else "no filters") + f"; {len(d):,} rows matched"


def _parse_json(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    a, b = raw.find("{"), raw.rfind("}") + 1
    return json.loads(raw[a:b]) if a >= 0 and b > a else None


def plan_query(question, frames):
    schema = []
    for label, df in frames.items():
        cols = []
        for c in df.columns[:40]:
            kind = "number" if pd.api.types.is_numeric_dtype(df[c]) else "text"
            ex = ", ".join(map(str, df[c].dropna().astype(str).unique()[:3]))
            cols.append(f"{c} ({kind}; e.g. {ex})")
        schema.append(f'- "{label}" [{len(df):,} rows]: ' + "; ".join(cols))
    prompt = f"""You plan a computation over spreadsheet data. Reply with ONLY a JSON object.

Tables:
{chr(10).join(schema)}

Question: {question}

JSON format:
{{"source": "<table label, or null if the question cannot be answered by computing over a table>",
  "filters": [{{"column": "...", "op": "eq|ne|gt|gte|lt|lte|contains|in|isnull|notnull", "value": ...}}],
  "group_by": ["..."],
  "aggregations": [{{"column": "...", "fn": "sum|mean|min|max|count|nunique|median"}}],
  "select": ["..."],
  "sort_by": "column or fn(column)", "ascending": false, "limit": 10}}
Use only column names from the table. Omit keys you do not need."""
    raw = chat_text([{"role": "user", "content": prompt}], model=GEN_MODEL, temperature=0.0, timeout=30)
    return _parse_json(raw)


def answer_from_tables(question, doc_ids):
    """Returns (context_text, detail) with the exact computed result, or None when not applicable."""
    frames = {}
    for did in doc_ids or []:
        for sheet, df in load_sheets(did).items():
            frames[f"{did}/{sheet}"] = df
    if not frames or not AGG_RE.search(question):
        return None
    spec = plan_query(question, frames)
    if not spec or not spec.get("source") or spec["source"] not in frames:
        return None
    res, desc = run_spec(frames[spec["source"]], spec)
    ctx = (f'Computed result (exact, calculated with pandas from table "{spec["source"]}"; {desc}):\n'
           f"{to_markdown(res)}")
    return ctx, f'{spec["source"]}: {desc}', spec
