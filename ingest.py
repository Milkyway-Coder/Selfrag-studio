"""
ingest.py — turns files into TYPED BLOCKS so tables, figures, formulas and code are not flattened into
plain text.

A block is {"type", "text", "page", "meta"} with type one of:
  text    ordinary prose (page = page / slide number, or None)
  table   a Markdown table (PDF/Word/PowerPoint tables, spreadsheet row groups)
  figure  an image saved to disk; text = caption + description + sentences that mention it
  vision  page-level transcription from a vision model: formulas as LaTeX, tables, chart descriptions
  sheet   statistics for a spreadsheet sheet
  code    a source-code window with file name and line range

What each format gives you:
  PDF     text (tables removed from the text and kept as tables), tables, embedded images and vector
          figures with their captions; optional "deep" mode sends every page to a vision model.
  DOCX    headings, lists, tables, inline images, equations (as text)
  PPTX    slide text, tables, pictures, speaker notes
  XLSX/CSV  statistics + row groups + exact pandas queries (see tabular.py)
  Code, .ipynb, .zip projects  language-aware windows with file path and line numbers
"""
import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import tabular
import vision

FIG_DIR = Path(os.getenv("FIGURE_DIR", "figures"))
MAX_VISION_PAGES = int(os.getenv("MAX_VISION_PAGES", "40"))
VISION_ALL_PAGES = os.getenv("VISION_ALL_PAGES", "").lower() in ("1", "true", "yes")
MATH_CHARS = set("∑∫√∞≈≠≤≥±∂∇∈∉⊂⊆∪∩→⇒⇔×÷αβγδεζηθλμνξπρσφχψωΔΘΛΞΠΣΦΨΩ")


def math_score(text):
    """Rough count of mathematical content in extracted text (symbols, equation numbers, many '=' lines)."""
    n = sum(1 for ch in text if ch in MATH_CHARS)
    n += 2 * len(re.findall(r"=.*\(\d+(?:\.\d+)?\)\s*$", text, re.M))
    if len(re.findall(r"^.*\S\s*=\s*\S.*$", text, re.M)) >= 3:
        n += 2
    return n


def needs_vision(info):
    """Only pages that can contain something plain text extraction misses are sent to the vision model."""
    return VISION_ALL_PAGES or bool(info["tables"] or info["images"] or info["drawings"] >= 8
                                    or info["math"] >= 2 or info["words"] < 40)
MAX_SHEET_ROWS_INDEXED = 3000

CODE_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift",
    ".sql": "sql", ".sh": "bash", ".bat": "batch", ".ps1": "powershell", ".r": "r", ".m": "matlab",
    ".html": "html", ".css": "css", ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml",
    ".toml": "toml", ".ini": "ini", ".tex": "latex", ".v": "verilog", ".vhd": "vhdl", ".asm": "asm",
}
TEXT_EXT = {".txt", ".md", ".rst", ".log"}
SHEET_EXT = {".xlsx", ".xlsm", ".csv", ".tsv"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "env", "dist", "build", ".idea",
             ".vscode", "target", "bin", "obj", ".next", ".mypy_cache"}
SUPPORTED = (sorted(CODE_EXT) + sorted(TEXT_EXT) + sorted(SHEET_EXT)
             + [".pdf", ".docx", ".pptx", ".ipynb", ".zip"])

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.?|Table)\s+(\d+[A-Za-z]?)\s*[.:\-–—]?\s*(.*)", re.I | re.S)


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def detect_kind(filename):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf": return "pdf"
    if ext == ".docx": return "docx"
    if ext == ".pptx": return "pptx"
    if ext in SHEET_EXT: return "sheet"
    if ext == ".ipynb": return "notebook"
    if ext == ".zip": return "project"
    if ext in CODE_EXT: return "code"
    if ext in TEXT_EXT: return "text"
    return None


class Ctx:
    """Per-document parsing state."""
    def __init__(self, doc_id, vision_on=False, progress=None):
        self.doc_id, self.vision, self.progress = doc_id, vision_on, progress
        self.fig_dir = FIG_DIR / safe(doc_id)
        self.fig_n = 0
        self.warnings = []
        self.vision_used = 0
        self.on_wait = lambda w: self.tick(0, 0, f"Waiting {w:.0f}s for the Groq rate limit")
        self.page_offset = 0
        self.vision_blocked = False

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def tick(self, done, total, msg):
        if self.progress:
            self.progress(done, total, msg)


def clear_artifacts(doc_id):
    shutil.rmtree(FIG_DIR / safe(doc_id), ignore_errors=True)
    tabular.delete_data(doc_id)


# ── helpers ───────────────────────────────────────────────────────────────────

def rows_to_markdown(rows):
    rows = [[("" if c is None else re.sub(r"\s+", " ", str(c)).strip().replace("|", "\\|")) for c in r] for r in rows or []]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * w) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _shrink_png(png, max_side=1600):
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(png))
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            return buf.getvalue(), im.size
        return png, im.size
    except Exception:
        return png, (0, 0)


def _is_noise(png):
    """Near-blank or banner-shaped images (watermark strips, separators, solid blocks) are not figures."""
    try:
        from PIL import Image, ImageStat
        im = Image.open(io.BytesIO(png))
        w, h = im.size
        if h and (w / h > 8 or w / h < 0.125):
            return True
        return ImageStat.Stat(im.convert("L")).stddev[0] < 6
    except Exception:
        return False


def add_figure(ctx, blocks, png, page, caption="", label="", mention="", min_side=0, y=0.0):
    png, (w, h) = _shrink_png(png)
    if min_side and (w < min_side or h < min_side * 0.66) and not caption:
        return None
    if not caption and _is_noise(png):
        return None
    ctx.fig_n += 1
    fig_id = f"f{ctx.fig_n:03d}"
    ctx.fig_dir.mkdir(parents=True, exist_ok=True)
    (ctx.fig_dir / f"{fig_id}.png").write_bytes(png)
    head = f"Figure {label}: {caption}".strip() if label else (caption or (f"Figure on page {page}" if page else "Figure"))
    parts = [head]
    if mention:
        parts.append("Mentioned in the text: " + mention)
    blocks.append({"type": "figure", "text": "\n".join(parts), "page": page,
                   "meta": {"fig_id": fig_id, "fig_label": label or "", "caption": caption or "", "y": float(y), "_pn": page,
                            "fig_url": f"/figure/{safe(ctx.doc_id)}/{fig_id}.png"}})
    return fig_id


_DESC_START = re.compile(r"^(this|the|a|an)\s+(\w+\s+){0,4}(figure|image|chart|graph|plot|diagram|screenshot|photo|map|table|illustration|composite)\b|^figure\b", re.I)


def clean_caption(text, max_len=90):
    """
    A short, readable caption from a model description. Skips code fences, LaTeX, Markdown tables and list
    markers, prefers a sentence that actually describes a figure, and never returns raw markup.
    """
    t = re.sub(r"```\w*", "\n", text or "")
    cands = []
    for line in t.splitlines():
        l = line.strip()
        if not l or l[0] in "|$\\#{}[<" or l.startswith(("---", "```")):
            continue
        l = re.sub(r"^[-*\u2022]+\s*|^\d+[.)]\s+", "", l)
        l = re.sub(r"^\W*(figure|fig\.?)\s*[:\-]\s*", "", l, flags=re.I)
        l = re.sub(r"[*_`]+", "", l).strip()
        if len(l.split()) < 3 or re.search(r"[\\{}$]|\\text|\\quad|\|", l):
            continue
        if sum(c.isalpha() or c == " " for c in l) / len(l) < 0.85:
            continue
        cands.append(l)
    if not cands:
        return ""
    pick = next((c for c in cands if _DESC_START.match(c)), cands[0])
    first = re.split(r"(?<=[.!?])\s", pick)[0].rstrip(" .:;")
    if len(first) > max_len:
        first = first[:max_len].rsplit(" ", 1)[0].rstrip(" ,;:-") + "\u2026"
    return first


def display_caption(caption):
    """Captions saved by an earlier version may hold raw markup; show clean text (or nothing) instead."""
    c = caption or ""
    if re.search(r"```|[\\{}$|]", c) or c.lstrip().startswith(("#", "[")):
        return clean_caption(c)
    return c


def attach_description(block, desc, title=None):
    """Adds a vision description to a figure block and, when possible, a meaningful caption."""
    desc = re.sub(r"```\w*", "", (desc or "")).strip()
    if not desc or desc.upper() == "NONE":
        return
    cap = clean_caption(title) if title else ""
    cap = cap or clean_caption(desc)
    if cap and not block["meta"].get("caption"):
        block["meta"]["caption"] = cap
    old = block["text"].split("\n")
    mention = [l for l in old[1:] if l.startswith("Mentioned in the text")]
    head = f"Figure: {block['meta']['caption']}" if block["meta"].get("caption") else old[0]
    block["text"] = "\n".join([head, "Description: " + desc[:700]] + mention)


def _mention(page_text, label, exclude=""):
    if not label:
        return ""
    pat = re.compile(rf"\b(?:Figure|Fig\.?)\s*{re.escape(label)}\b", re.I)
    for sent in re.split(r"(?<=[.!?])\s+", page_text):
        if pat.search(sent) and sent.strip() != exclude.strip() and not sent.strip().lower().startswith(("figure", "fig")):
            return sent.strip()[:260]
    return ""


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(path, ctx):
    import fitz
    doc = fitz.open(path)
    n = len(doc)
    blocks, stats = [], {"pages": n, "tables": 0, "figures": 0, "vision_pages": 0}
    native_words, seen_xref, page_info, printed = 0, set(), [], []

    for pno in range(n):
        page, pn = doc[pno], pno + 1
        ctx.tick(pno, n, f"Reading page {pn} of {n}")

        # tables first, so their text can be removed from the running text
        tables = []
        try:
            for t in page.find_tables().tables:
                md = rows_to_markdown(t.extract())
                if md:
                    tables.append((fitz.Rect(t.bbox), md))
        except Exception:
            pass

        text_blocks, captions = [], []
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0:
                continue
            r = fitz.Rect(b[:4])
            if any(r.intersects(tr) and (r & tr).get_area() > 0.5 * max(r.get_area(), 1) for tr, _ in tables):
                continue
            raw_lines = [l.strip() for l in b[4].split("\n") if l.strip()]
            if not raw_lines:
                continue
            # a caption is the first line(s) up to the first sentence end; whatever follows in the same
            # PDF block is ordinary prose (PyMuPDF often merges the caption with the next paragraph)
            if CAPTION_RE.match(raw_lines[0]):
                k = 1
                while k < len(raw_lines) and not re.search(r"[.?!:]$", raw_lines[k - 1]):
                    k += 1
                cap_text = re.sub(r"\s+", " ", " ".join(raw_lines[:k])).strip()
                m = CAPTION_RE.match(cap_text)
                if m and len(cap_text) < 500:
                    captions.append((r, m.group(1).lower().startswith("t"), m.group(2), m.group(3).strip(), cap_text))
                rest = re.sub(r"\s+", " ", " ".join(raw_lines[k:])).strip()
                text_blocks.append(cap_text)
                if rest:
                    text_blocks.append(rest)
                continue
            text_blocks.append(re.sub(r"\s+", " ", " ".join(raw_lines)).strip())
        page_text = "\n\n".join(text_blocks)
        native_words += len(page_text.split())
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        mpg = re.search(r"Page\s*\|\s*(\d+)\s*$", page.get_text().strip(), re.I | re.M) or re.search(r"Page\s*\|\s*(\d+)", page.get_text(), re.I)
        printed.append(int(mpg.group(1)) if mpg else None)
        page_info.append({"pn": pn, "words": len(page_text.split()), "images": len(page.get_images()),
                          "drawings": len(drawings), "tables": len(tables), "math": math_score(page_text)})

        for rect, md in tables:
            cap = ""
            near = [c for c in captions if c[1] and (-10 <= rect.y0 - c[0].y1 < 70 or -10 <= c[0].y0 - rect.y1 < 40)]
            if near:
                cap = f"Table {near[0][2]}: {near[0][3]}\n"
            blocks.append({"type": "table", "text": cap + md, "page": pn, "meta": {}})
            stats["tables"] += 1

        if page_text:
            blocks.append({"type": "text", "text": page_text, "page": pn, "meta": {}})

        # raster figures
        fig_caps = [c for c in captions if not c[1]]
        got_raster = False
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xref:
                continue
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png = pix.tobytes("png")
            except Exception:
                continue
            seen_xref.add(xref)
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            cap = None
            if rects and fig_caps:
                r0 = rects[0]
                cap = min(fig_caps, key=lambda c: min(abs(c[0].y0 - r0.y1), abs(r0.y0 - c[0].y1)))
            label, caption = (cap[2], cap[3]) if cap else ("", "")
            if add_figure(ctx, blocks, png, pn, caption, label, _mention(page_text, label, cap[4] if cap else ""),
                          min_side=150, y=(rects[0].y0 if rects else 0)):
                stats["figures"] += 1
                got_raster = True

        # vector figures (charts and diagrams drawn with paths, which are not images)
        if not got_raster and fig_caps:
            try:
                draws = [d["rect"] for d in drawings if d["rect"].width > 0 or d["rect"].height > 0]
                if len(draws) >= 8:
                    u = fitz.Rect(draws[0])
                    for d in draws[1:]:
                        u |= d
                    u &= page.rect
                    if u.width > 120 and u.height > 80 and u.get_area() < 0.92 * page.rect.get_area():
                        u = fitz.Rect(u.x0 - 6, u.y0 - 6, u.x1 + 6, u.y1 + 6) & page.rect
                        png = page.get_pixmap(dpi=130, clip=u).tobytes("png")
                        cap = min(fig_caps, key=lambda c: min(abs(c[0].y0 - u.y1), abs(u.y0 - c[0].y1)))
                        if add_figure(ctx, blocks, png, pn, cap[3], cap[2], _mention(page_text, cap[2], cap[4]), y=u.y0):
                            stats["figures"] += 1
            except Exception:
                pass

    scanned = native_words / max(n, 1) < 20
    # deep mode: vision transcription of the pages that can hold formulas, figures, tables or scanned text
    if ctx.vision and n:
        todo = [i["pn"] for i in page_info if needs_vision(i)]
        stats["vision_candidates"] = len(todo)
        if len(todo) > MAX_VISION_PAGES:
            ctx.warn(f"{len(todo)} pages need vision; only the first {MAX_VISION_PAGES} were sent (MAX_VISION_PAGES in .env).")
            todo = todo[:MAX_VISION_PAGES]
        failed = 0
        figs_on = {}
        for b_ in blocks:
            if b_["type"] == "figure":
                figs_on.setdefault(b_["meta"].get("_pn"), []).append(b_)
        lonely = []                                        # figures the page pass could not describe on its own
        for k, pn in enumerate(todo):
            if ctx.vision_blocked:
                break
            ctx.tick(k, len(todo), f"Vision: page {pn} ({k + 1} of {len(todo)} pages with formulas, figures or tables)")
            png = doc[pn - 1].get_pixmap(dpi=110).tobytes("png")
            out = (_vision(ctx, png, vision.PAGE_PROMPT, 900) or "").strip()
            if out and out.upper() != "NONE":
                blocks.append({"type": "vision", "text": f"[Page {pn} transcription]\n{out}", "page": pn, "meta": {}})
                stats["vision_pages"] += 1
                fl = sorted(figs_on.get(pn, []), key=lambda x: x["meta"].get("y", 0))
                todo_figs = [f for f in fl if not f["meta"].get("caption")]
                titles = [m.group(1).strip() for m in (re.match(r"\s*[-*\u2022]?\s*\**\s*Figure\s*\**\s*:\s*(.+)", l, re.I) for l in out.splitlines()) if m]
                if len(todo_figs) == 1:
                    attach_description(todo_figs[0], out, titles[0] if titles else None)
                elif todo_figs and len(titles) == len(fl) == len(todo_figs):
                    for f_, t_ in zip(fl, titles):
                        attach_description(f_, t_, t_)
                else:
                    lonely += todo_figs
        for j, f_ in enumerate(lonely[:6]):               # pages with several untitled figures: describe them one by one
            if ctx.vision_blocked:
                break
            ctx.tick(j, min(len(lonely), 6), f"Vision: figure {j + 1} of {min(len(lonely), 6)}")
            png_bytes = (ctx.fig_dir / f"{f_['meta']['fig_id']}.png").read_bytes()
            d_ = _vision(ctx, png_bytes, vision.FIGURE_PROMPT, 400)
            if d_:
                attach_description(f_, d_)
    elif scanned:
        ocr = _ocr(path)
        if ocr:
            blocks = [b for b in blocks if b["type"] != "text"] + [
                {"type": "text", "text": t, "page": i + 1, "meta": {}} for i, t in enumerate(ocr) if t.strip()]
        else:
            ctx.warn("This PDF has almost no extractable text (scanned?). Install Tesseract for OCR, "
                     "or tick 'Deep analysis' to read the pages with a vision model.")
    doc.close()
    # if the document prints its own page numbers ("Page | 14") and they differ from the PDF index by a constant,
    # show the printed numbers so "page 14" in an answer matches what the reader sees on the page
    offs = {p - (i + 1) for i, p in enumerate(printed) if p is not None}
    if len([p for p in printed if p is not None]) >= max(2, n // 2) and len(offs) == 1 and offs != {0}:
        off = offs.pop()
        for b_ in blocks:
            if b_.get("page"):
                b_["page"] += off
                if b_["type"] == "figure":
                    b_["text"] = re.sub(r"^Figure on page \d+", f"Figure on page {b_['page']}", b_["text"])
                elif b_["type"] == "vision":
                    b_["text"] = re.sub(r"^\[Page \d+ transcription\]", f"[Page {b_['page']} transcription]", b_["text"])
        stats["page_offset"] = off
    for b_ in blocks:
        b_["meta"].pop("_pn", None)
    return blocks, stats


def _vision(ctx, png, prompt, max_tokens):
    """One vision call. Returns text, or None when vision is (now) unavailable for this document."""
    if ctx.vision_blocked:
        return None
    try:
        out = vision.describe(png, prompt, max_tokens, ctx.on_wait)
        ctx.vision_used += 1
        return out
    except vision.VisionUnavailable as e:
        ctx.vision_blocked = True; ctx.warn(str(e)); return None
    except Exception as e:                                  # noqa: BLE001
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            ctx.vision_blocked = True
            ctx.warn("Groq's rate limit was reached while reading pages, so the remaining pages and figures were not "
                     "analysed. Your plan allows a limited number of vision tokens per minute or per day. Re-upload later, "
                     "or lower MAX_VISION_PAGES in .env.")
        else:
            ctx.warn(f"A vision call failed: {msg[:140]}")
        return None


def _ocr(path):
    try:
        import fitz
        import pytesseract
        from PIL import Image
        doc = fitz.open(path)
        out = []
        for page in doc:
            img = Image.open(io.BytesIO(page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).tobytes("png")))
            out.append(pytesseract.image_to_string(img))
        doc.close()
        return out
    except Exception:
        return []


# ── Word ──────────────────────────────────────────────────────────────────────

def parse_docx(path, ctx):
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(path)
    children = list(d.element.body.iterchildren())
    blocks, stats = [], {"tables": 0, "figures": 0, "equations": 0}
    for i, el in enumerate(children):
        ctx.tick(i, len(children), "Reading Word document")
        if el.tag == qn("w:tbl"):
            md = rows_to_markdown(_docx_rows(Table(el, d)))
            if md:
                blocks.append({"type": "table", "text": md, "page": None, "meta": {}}); stats["tables"] += 1
        elif el.tag == qn("w:p"):
            p = Paragraph(el, d)
            text = p.text.strip()
            math = ["".join(t.text or "" for t in m.iter(qn("m:t"))) for m in el.xpath(".//m:oMath")]
            math = [m for m in math if m.strip()]
            if math:
                text = (text + " " + " ".join(f"[formula: {m}]" for m in math)).strip(); stats["equations"] += len(math)
            style = (p.style.name if p.style is not None else "") or ""
            if text:
                if style.startswith("Heading") or style == "Title":
                    lvl = int(style[-1]) if style[-1:].isdigit() else 1
                    text = "#" * min(lvl, 4) + " " + text
                elif style.startswith("List"):
                    text = "- " + text
                blocks.append({"type": "text", "text": text, "page": None, "meta": {}})
            for rid in el.xpath(".//a:blip/@r:embed"):
                try:
                    png = d.part.related_parts[rid].blob
                except Exception:
                    continue
                cap, label = "", ""
                for j in (i + 1, i - 1):
                    if 0 <= j < len(children) and children[j].tag == qn("w:p"):
                        m = CAPTION_RE.match(Paragraph(children[j], d).text.strip())
                        if m and not m.group(1).lower().startswith("t"):
                            cap, label = m.group(3).strip(), m.group(2); break
                if add_figure(ctx, blocks, _to_png(png), None, cap, label, min_side=100):
                    stats["figures"] += 1
    return blocks, stats


def _docx_rows(table):
    rows = []
    for r in table.rows:
        seen, cells = set(), []
        for c in r.cells:
            if id(c._tc) in seen:
                continue
            seen.add(id(c._tc)); cells.append(c.text)
        rows.append(cells)
    return rows


def _to_png(blob):
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(blob)).convert("RGBA" if blob[:4] == b"\x89PNG" else "RGB")
        buf = io.BytesIO(); im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return blob


# ── PowerPoint ────────────────────────────────────────────────────────────────

def parse_pptx(path, ctx):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    prs = Presentation(path)
    blocks, stats = [], {"pages": len(prs.slides), "tables": 0, "figures": 0}
    for n, slide in enumerate(prs.slides, 1):
        ctx.tick(n - 1, len(prs.slides), f"Reading slide {n}")
        title = slide.shapes.title.text.strip() if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""
        lines = []
        for sh in slide.shapes:
            if sh.has_text_frame and sh is not slide.shapes.title:
                lines += [p.text.strip() for p in sh.text_frame.paragraphs if p.text.strip()]
            elif getattr(sh, "has_table", False) and sh.has_table:
                md = rows_to_markdown([[c.text for c in r.cells] for r in sh.table.rows])
                if md:
                    blocks.append({"type": "table", "text": f"Slide {n} table:\n{md}", "page": n, "meta": {}}); stats["tables"] += 1
            elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    if add_figure(ctx, blocks, _to_png(sh.image.blob), n, f"Picture on slide {n}" + (f" ({title})" if title else ""), "", min_side=100):
                        stats["figures"] += 1
                except Exception:
                    pass
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        body = f"Slide {n}: {title}".strip() + ("\n" + "\n".join("- " + l for l in lines) if lines else "")
        if notes:
            body += f"\nSpeaker notes: {notes}"
        blocks.append({"type": "text", "text": body, "page": n, "meta": {}})
    return blocks, stats


# ── Spreadsheets ──────────────────────────────────────────────────────────────

def parse_sheet(path, ctx, filename):
    import pandas as pd
    ext = Path(filename).suffix.lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                frames = {Path(filename).stem: pd.read_csv(path, sep=sep, engine="python", encoding=enc)}
                break
            except UnicodeDecodeError:
                continue
    else:
        frames = pd.read_excel(path, sheet_name=None)
    blocks, stats = [], {"sheets": 0, "rows": 0, "tables": 0}
    for name, df in frames.items():
        df = tabular.clean_frame(df)
        if df.empty:
            continue
        tabular.save_sheet(ctx.doc_id, name, df)
        stats["sheets"] += 1; stats["rows"] += len(df)
        blocks.append({"type": "sheet", "text": tabular.describe_frame(name, df), "page": None, "meta": {"sheet": str(name)}})
        capped = df.head(MAX_SHEET_ROWS_INDEXED)
        if len(df) > MAX_SHEET_ROWS_INDEXED:
            ctx.warn(f'Sheet "{name}" has {len(df):,} rows; the first {MAX_SHEET_ROWS_INDEXED:,} are searchable as text. '
                     f"Calculations (averages, counts, top-N) use all rows.")
        step = 15
        for s in range(0, len(capped), step):
            part = capped.iloc[s:s + step]
            blocks.append({"type": "table", "text": f'Sheet "{name}", rows {s + 1}-{s + len(part)}:\n{tabular.to_markdown(part)}',
                           "page": None, "meta": {"sheet": str(name)}})
            stats["tables"] += 1
    if not blocks:
        raise ValueError("The spreadsheet has no data.")
    return blocks, stats


# ── Code ──────────────────────────────────────────────────────────────────────

_BOUNDARY = re.compile(r"^\s{0,4}(def |class |async def |function |export |public |private |protected |static |"
                       r"void |int |double |float |char |fn |func |impl |struct |interface |enum |@|#include|"
                       r"import |from |package |using |namespace |SELECT |CREATE |INSERT |ALTER )")
_DEFS = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|fn|func|interface|struct|enum|public\s+(?:static\s+)?[\w<>\[\]]+|"
                   r"private\s+(?:static\s+)?[\w<>\[\]]+|static\s+[\w<>\[\]]+|void|int|double|float|char|bool|String)\s+(\w+)\s*[(:{<]")


def chunk_code(text, max_lines=60, overlap=6):
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        end = min(i + max_lines, len(lines))
        if end < len(lines):
            for j in range(end - 1, max(i + 20, end - 16), -1):
                if _BOUNDARY.match(lines[j]):
                    end = j; break
        out.append((i + 1, end, "\n".join(lines[i:end])))
        if end >= len(lines):
            break
        i = max(end - overlap, i + 1)
    return out


def code_blocks(text, filename, lang):
    text = text.replace("\r\n", "\n").replace("\t", "    ")
    if not text.strip():
        return []
    blocks = []
    defs = [m.group(1) for line in text.splitlines() if (m := _DEFS.match(line))]
    n_lines = len(text.splitlines())
    outline = f"File: {filename} ({lang}), {n_lines} lines."
    if defs:
        outline += " Defines: " + ", ".join(dict.fromkeys(defs))[:600] + "."
    blocks.append({"type": "code", "text": outline, "page": None, "meta": {"file": filename, "lang": lang, "outline": True}})
    for a, b, body in chunk_code(text):
        blocks.append({"type": "code", "text": f"File: {filename} ({lang}), lines {a}-{b}\n```{lang}\n{body}\n```",
                       "page": None, "meta": {"file": filename, "lang": lang, "start": a, "end": b}})
    return blocks


def parse_notebook(path, ctx, filename):
    nb = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    blocks, n_code = [], 0
    for i, cell in enumerate(nb.get("cells", []), 1):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            blocks.append({"type": "text", "text": src, "page": None, "meta": {}})
        elif cell.get("cell_type") == "code":
            n_code += 1
            out = ""
            for o in cell.get("outputs", []):
                t = "".join(o.get("text", [])) or "".join((o.get("data", {}) or {}).get("text/plain", []))
                if t:
                    out += t[:500]
            body = f"Notebook cell {i} ({filename})\n```python\n{src}\n```"
            if out.strip():
                body += f"\nOutput:\n{out.strip()[:500]}"
            blocks.append({"type": "code", "text": body, "page": None, "meta": {"file": filename, "lang": "python", "start": i, "end": i}})
    return blocks, {"code_cells": n_code}


def parse_zip(path, ctx):
    blocks, tree, files = [], [], 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1]) or info.file_size > 1_000_000:
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in CODE_EXT and ext not in TEXT_EXT and ext != ".ipynb":
                continue
            files += 1
            if files > 300:
                ctx.warn("Only the first 300 files of the archive were indexed."); break
            tree.append(info.filename)
            raw = z.read(info).decode("utf-8", errors="replace")
            ctx.tick(files, files, info.filename)
            if ext == ".ipynb":
                tmp = FIG_DIR / "_tmp.ipynb"; tmp.parent.mkdir(parents=True, exist_ok=True); tmp.write_text(raw, encoding="utf-8")
                try:
                    bl, _ = parse_notebook(tmp, ctx, info.filename); blocks += bl
                finally:
                    tmp.unlink(missing_ok=True)
            elif ext in TEXT_EXT:
                blocks.append({"type": "text", "text": f"File: {info.filename}\n{raw}", "page": None, "meta": {}})
            else:
                blocks += code_blocks(raw, info.filename, CODE_EXT[ext])
    if not files:
        raise ValueError("No source or text files found in the archive.")
    listing = "Project files:\n" + "\n".join(f"- {t}" for t in sorted(tree))
    return [{"type": "code", "text": listing, "page": None, "meta": {"file": "(project tree)", "lang": "", "outline": True}}] + blocks, \
        {"code_files": files}


# ── entry point ───────────────────────────────────────────────────────────────

def parse_file(path, doc_id, filename=None, vision_on=False, progress=None):
    """
    Returns {"blocks", "kind", "stats", "warnings"}. Raises ValueError for unsupported or empty files.
    """
    filename = filename or Path(path).name
    kind = detect_kind(filename)
    if kind is None:
        raise ValueError(f"Unsupported file type: {Path(filename).suffix or filename}. Supported: {', '.join(SUPPORTED)}")
    clear_artifacts(doc_id)
    ctx = Ctx(doc_id, vision_on, progress)
    try:
        if kind == "pdf":        blocks, stats = parse_pdf(path, ctx)
        elif kind == "docx":     blocks, stats = parse_docx(path, ctx)
        elif kind == "pptx":     blocks, stats = parse_pptx(path, ctx)
        elif kind == "sheet":    blocks, stats = parse_sheet(path, ctx, filename)
        elif kind == "notebook": blocks, stats = parse_notebook(path, ctx, filename)
        elif kind == "project":  blocks, stats = parse_zip(path, ctx)
        else:
            raw = Path(path).read_bytes().decode("utf-8", errors="replace")
            if kind == "code":
                ext = Path(filename).suffix.lower()
                blocks, stats = code_blocks(raw, filename, CODE_EXT[ext]), {"code_files": 1}
            else:
                blocks, stats = [{"type": "text", "text": raw, "page": None, "meta": {}}], {}
    except Exception:
        clear_artifacts(doc_id)
        raise
    blocks = [b for b in blocks if b["text"].strip()]
    words = sum(len(b["text"].split()) for b in blocks)
    if words < 5:
        clear_artifacts(doc_id)
        raise ValueError("No readable content was found in this file.")
    stats["words"] = words
    stats["vision_calls"] = ctx.vision_used
    return {"blocks": blocks, "kind": kind, "stats": stats, "warnings": ctx.warnings}
