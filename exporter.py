"""
exporter.py — Generates a formatted PDF session report using ReportLab.

Exports all logged queries with answers, scores, hallucination data,
and a summary statistics page.
"""

import io
import json
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)


# ── Colour palette ────────────────────────────────────────────────────────────
C_BG      = colors.white
C_SURFACE = colors.HexColor("#F3F6FA")
C_ACCENT  = colors.HexColor("#00629B")
C_RED     = colors.HexColor("#B3261E")
C_BLUE    = colors.HexColor("#00629B")
C_AMBER   = colors.HexColor("#9A6700")
C_GREEN   = colors.HexColor("#1B7F5C")
C_TEXT    = colors.HexColor("#14213D")
C_MUTED   = colors.HexColor("#5B6577")
C_WHITE   = colors.white
C_BLACK   = colors.black
C_LINE    = colors.HexColor("#D9DEE7")


def _x(s):
    """Escape text for ReportLab's mini-HTML Paragraph parser (raw '<' used to crash the export)."""
    from xml.sax.saxutils import escape
    return escape(str(s if s is not None else ""))


# ── Style helpers ─────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "title"    : ps("title",     fontSize=28, textColor=C_ACCENT,
                         fontName="Helvetica-Bold", spaceAfter=6),
        "subtitle" : ps("subtitle",  fontSize=12, textColor=C_MUTED,
                         fontName="Helvetica",      spaceAfter=20),
        "h2"       : ps("h2",        fontSize=13, textColor=C_ACCENT,
                         fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6),
        "h3"       : ps("h3",        fontSize=10, textColor=C_BLUE,
                         fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4),
        "body"     : ps("body",      fontSize=9,  textColor=C_TEXT,
                         fontName="Helvetica",      spaceAfter=4, leading=14),
        "muted"    : ps("muted",     fontSize=8,  textColor=C_MUTED,
                         fontName="Helvetica",      spaceAfter=3),
        "answer"   : ps("answer",    fontSize=9,  textColor=C_TEXT,
                         fontName="Helvetica",      spaceAfter=6,
                         leftIndent=12, leading=14),
        "tag_good" : ps("tag_good",  fontSize=8,  textColor=C_ACCENT,
                         fontName="Helvetica-Bold"),
        "tag_bad"  : ps("tag_bad",   fontSize=8,  textColor=C_RED,
                         fontName="Helvetica-Bold"),
        "tag_mid"  : ps("tag_mid",   fontSize=8,  textColor=C_AMBER,
                         fontName="Helvetica-Bold"),
        "label"    : ps("label",     fontSize=7,  textColor=C_MUTED,
                         fontName="Helvetica",      spaceAfter=2,
                         textTransform="uppercase"),
    }


def _score_color(v):
    if v >= 4:  return C_GREEN
    if v >= 3:  return C_AMBER
    return C_RED


def _halluc_color(pct):
    if pct == 0:   return C_GREEN
    if pct < 30:   return C_AMBER
    return C_RED


def _score_table(scores):
    """3-column mini table: Retrieval | Grounding | Quality."""
    def cell(label, val):
        col = _score_color(val)
        return [
            Paragraph(label, ParagraphStyle("sl", fontSize=7,
                textColor=C_MUTED, fontName="Helvetica",
                textTransform="uppercase")),
            Paragraph(f"{val}/5", ParagraphStyle("sv", fontSize=14,
                textColor=col, fontName="Helvetica-Bold")),
        ]

    r = scores.get("retrieval_relevance", 0)
    g = scores.get("answer_grounding", 0)
    q = scores.get("answer_quality", 0)
    a = scores.get("average_score", 0)

    data = [[
        cell("Retrieval",  r),
        cell("Grounding",  g),
        cell("Quality",    q),
        cell("Average",    a),
    ]]

    t = Table(data, colWidths=[3.8*cm, 3.8*cm, 3.8*cm, 3.8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_SURFACE),
        ("BOX",        (0,0), (-1,-1), 0.5, C_MUTED),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, C_LINE),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    return t


def _summary_stats_table(logs):
    """Top summary table on the first page."""
    total     = len(logs)
    avg_score = round(sum((l["average_score"] or 0) for l in logs) / total, 2) if total else 0
    good      = sum(1 for l in logs if (l["average_score"] or 0) >= 4)
    retried   = sum(1 for l in logs if l.get("attempts", 1) > 1)
    reformed  = sum(1 for l in logs if l.get("reformulated_query"))
    h_scores  = [l["hallucination_score"] for l in logs if l.get("hallucination_score") is not None]
    avg_halluc = round(sum(h_scores) / len(h_scores) * 100, 1) if h_scores else None

    rows = [
        ["Metric", "Value"],
        ["Total queries",          str(total)],
        ["Average eval score",     f"{avg_score}/5"],
        ["High quality (≥4)",      f"{good}  ({round(good/total*100) if total else 0}%)"],
        ["Queries retried",        str(retried)],
        ["Queries reformulated",   str(reformed)],
    ]
    if avg_halluc is not None:
        rows.append(["Avg hallucination rate", f"{avg_halluc}%"])

    t = Table(rows, colWidths=[8*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_ACCENT),
        ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 9),
        ("BACKGROUND",    (0,1), (-1,-1), C_SURFACE),
        ("TEXTCOLOR",     (0,1), (-1,-1), C_TEXT),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_SURFACE, colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.5, C_LINE),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    return t


# ── Main export function ──────────────────────────────────────────────────────

def generate_pdf_report(logs, title="Self-RAG Session Report"):
    """
    Generates a PDF report from a list of query log dicts.

    Returns bytes of the PDF.
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm
    )
    S      = _styles()
    story  = []
    now    = datetime.now().strftime("%d %B %Y, %H:%M")

    # ── Cover / summary page ─────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("Self-RAG", S["title"]))
    story.append(Paragraph("Session Report", ParagraphStyle(
        "sr", fontSize=20, textColor=C_TEXT, fontName="Helvetica-Bold", spaceAfter=4
    )))
    story.append(Paragraph(f"Generated: {now}", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=20))

    if logs:
        story.append(Paragraph("Summary", S["h2"]))
        story.append(_summary_stats_table(logs))
        story.append(Spacer(1, 0.8*cm))
    else:
        story.append(Paragraph("No queries have been logged yet.", S["body"]))
        doc.build(story)
        return buf.getvalue()

    story.append(PageBreak())

    # ── Per-query pages ───────────────────────────────────────────────────────
    for i, log in enumerate(logs):
        query    = log.get("query", "—")
        answer   = log.get("answer", "—") or "—"
        scores   = {
            "retrieval_relevance": log.get("retrieval_relevance") or 0,
            "answer_grounding"   : log.get("answer_grounding") or 0,
            "answer_quality"     : log.get("answer_quality") or 0,
            "average_score"      : log.get("average_score") or 0,
        }
        avg      = scores["average_score"] or 0
        attempts = log.get("attempts", 1)
        reform   = log.get("reformulated_query")
        h_score  = log.get("hallucination_score")
        mode     = "Multi-document" if log.get("multi_doc_mode") else "Single-document"
        cited    = log.get("cited_sources", [])
        ts       = log.get("created_at", "")

        # header row
        conf_label = ("HIGH CONFIDENCE" if avg >= 4.5 else
                      "MODERATE"        if avg >= 3.5 else
                      "LOW — VERIFY")
        conf_color = (_score_color(avg) if avg >= 3.5 else C_RED)

        story.append(Paragraph(
            f"Query {i+1} of {len(logs)}",
            ParagraphStyle("qn", fontSize=8, textColor=C_MUTED,
                           fontName="Helvetica", spaceAfter=2)
        ))

        # query text
        story.append(Paragraph("Question", S["label"]))
        story.append(Paragraph(_x(query), ParagraphStyle(
            "q", fontSize=10, textColor=C_TEXT, fontName="Helvetica-Bold",
            spaceAfter=8, leading=16
        )))

        # confidence badge row
        meta_parts = [
            f"<font color='#{conf_color.hexval()[2:]}'>{conf_label}</font>",
            f"Mode: {mode}",
            f"Attempts: {attempts}",
        ]
        if ts:
            meta_parts.append(_x(ts))
        story.append(Paragraph(" &nbsp;|&nbsp; ".join(meta_parts),
                                ParagraphStyle("meta", fontSize=8, textColor=C_MUTED,
                                               fontName="Helvetica", spaceAfter=8)))

        # reformulated query note
        if reform:
            story.append(Paragraph(
                f"Query was rewritten to: <i>{_x(reform)}</i>",
                ParagraphStyle("ref", fontSize=8, textColor=C_BLUE,
                               fontName="Helvetica", spaceAfter=6)
            ))

        # score table
        story.append(_score_table(scores))
        story.append(Spacer(1, 0.3*cm))

        # hallucination
        if h_score is not None:
            h_pct = round(h_score * 100)
            hc    = _halluc_color(h_pct)
            story.append(Paragraph(
                f"Hallucination check: "
                f"<font color='#{hc.hexval()[2:]}'><b>{h_pct}% flagged</b></font>",
                ParagraphStyle("hl", fontSize=8, textColor=C_MUTED,
                               fontName="Helvetica", spaceAfter=6)
            ))

        # cited sources
        if cited:
            story.append(Paragraph(
                f"Sources cited: {_x(', '.join(cited))}",
                ParagraphStyle("cs", fontSize=8, textColor=C_BLUE,
                               fontName="Helvetica", spaceAfter=6)
            ))

        # answer
        story.append(Paragraph("Answer", S["label"]))
        # truncate very long answers to keep report manageable
        answer_display = answer if len(answer) <= 800 else answer[:800] + " …[truncated]"
        story.append(Paragraph(_x(answer_display).replace("\n", "<br/>"), S["answer"]))

        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=C_LINE,
                                spaceBefore=12, spaceAfter=12))

        # page break every 2 queries to avoid crowding
        if (i + 1) % 2 == 0 and i < len(logs) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()