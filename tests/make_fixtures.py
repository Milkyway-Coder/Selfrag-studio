"""Builds sample PDF / DOCX / PPTX / XLSX / CSV / ipynb / zip files for the multimodal tests."""
import io, json, os, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def chart_png(path):
    fig, ax = plt.subplots(figsize=(4, 2.6), dpi=100)
    ax.plot([1, 2, 3, 4, 5], [2.1, 1.4, 0.9, 0.6, 0.5], marker="o"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def make(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    png = os.path.join(dirpath, "chart.png"); chart_png(png)
    out = {}
    # ---- PDF with prose, a table, a raster figure, a vector figure and a formula ----
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing, Rect, Line, String
    st = getSampleStyleSheet()
    pdf = os.path.join(dirpath, "paper.pdf")
    story = [Paragraph("A Study of Neural Training", st["Title"]),
             Paragraph("This paper studies how the training loss of a small neural network decreases over epochs. "
                       "We use stochastic gradient descent with a learning rate of 0.01 and report results on a held-out set. " * 2, st["Normal"]),
             Spacer(1, 12),
             Paragraph("Table 1: Accuracy of three models on the test set.", st["Normal"]),
             Table([["Model", "Accuracy", "Parameters"], ["ResNet-18", "91.2", "11M"], ["VGG-16", "89.5", "138M"], ["MobileNet", "88.1", "3M"]],
                   style=TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black)])),
             Spacer(1, 12),
             Image(png, width=260, height=170),
             Paragraph("Figure 1: Training loss over epochs for the baseline model.", st["Normal"]),
             Paragraph("As shown in Figure 1, the loss decreases quickly in the first three epochs and then flattens.", st["Normal"]),
             PageBreak(),
             Paragraph("The energy relation is E = mc^2, and the loss is defined as L = (1/N) sum (y - y_hat)^2.", st["Normal"])]
    d = Drawing(300, 140)
    for i in range(6):
        d.add(Rect(10 + i * 45, 40, 36, 30 + i * 12, fillColor=colors.lightblue, strokeColor=colors.black))
        d.add(Line(10 + i * 45, 20, 46 + i * 45, 20))
        d.add(String(14 + i * 45, 8, f"L{i}", fontSize=8))
    for i in range(8):
        d.add(Line(10, 100 + i, 290, 100 + i, strokeColor=colors.grey))
    story += [Spacer(1, 12), d, Paragraph("Figure 2: Pipeline architecture with six processing stages.", st["Normal"]),
              Paragraph("Figure 2 shows the six stages of the pipeline.", st["Normal"])]
    SimpleDocTemplate(pdf, pagesize=A4).build(story); out["pdf"] = pdf
    # ---- DOCX ----
    import docx
    from docx.shared import Inches
    dd = docx.Document(); dd.add_heading("Lab Report", 1)
    dd.add_paragraph("This report compares three sorting algorithms by running time on random arrays of increasing size.")
    t = dd.add_table(rows=3, cols=3)
    for r, row in enumerate([["Algorithm", "Time (ms)", "Space"], ["Quick sort", "12", "O(log n)"], ["Merge sort", "15", "O(n)"]]):
        for c, v in enumerate(row): t.cell(r, c).text = v
    dd.add_picture(png, width=Inches(3)); dd.add_paragraph("Figure 1: Running time versus array size.")
    p = os.path.join(dirpath, "report.docx"); dd.save(p); out["docx"] = p
    # ---- PPTX ----
    from pptx import Presentation
    from pptx.util import Inches as PI
    pr = Presentation(); s = pr.slides.add_slide(pr.slide_layouts[5]); s.shapes.title.text = "Results overview"
    s.shapes.add_picture(png, PI(1), PI(1.8), width=PI(4))
    s2 = pr.slides.add_slide(pr.slide_layouts[1]); s2.shapes.title.text = "Conclusions"; s2.placeholders[1].text = "Loss converges after five epochs"
    p = os.path.join(dirpath, "deck.pptx"); pr.save(p); out["pptx"] = p
    # ---- spreadsheet ----
    df = pd.DataFrame({"name": [f"emp{i}" for i in range(40)], "dept": ["eng", "ops", "hr", "eng"] * 10,
                       "salary": [50000 + i * 1000 for i in range(40)]})
    p = os.path.join(dirpath, "staff.xlsx"); df.to_excel(p, index=False); out["xlsx"] = p
    p = os.path.join(dirpath, "staff.csv"); df.to_csv(p, index=False); out["csv"] = p
    # ---- code ----
    code = "import math\n\nclass Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, x):\n        self.items.append(x)\n\n    def pop(self):\n        return self.items.pop()\n\n" + \
           "\n".join(f"def helper_{i}(x):\n    return x * {i}\n" for i in range(40))
    p = os.path.join(dirpath, "stack.py"); open(p, "w").write(code); out["py"] = p
    z = os.path.join(dirpath, "project.zip")
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("proj/main.py", code); zf.writestr("proj/web/app.js", "function add(a,b){return a+b}\nexport default add\n" * 3)
        zf.writestr("proj/node_modules/x/index.js", "SKIPPED"); zf.writestr("proj/README.md", "# Demo project\nThis project implements a stack and helpers for testing the ingestion pipeline.")
    out["zip"] = z
    nb = {"cells": [{"cell_type": "markdown", "source": ["# Analysis\n", "Plot the loss."]},
                    {"cell_type": "code", "source": ["import numpy as np\n", "print(np.mean([1,2,3]))"], "outputs": [{"output_type": "stream", "text": ["2.0\n"]}]}]}
    p = os.path.join(dirpath, "nb.ipynb"); json.dump(nb, open(p, "w")); out["ipynb"] = p
    return out
