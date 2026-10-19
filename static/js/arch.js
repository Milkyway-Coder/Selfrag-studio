// arch.js — how the system works, written for someone seeing it for the first time.
import { esc } from "./core.js";

const STAGES = [
  ["Ingest", "PDF, Word, PowerPoint, Excel, CSV, code, notebooks, zip projects, web pages",
   "Files become typed blocks instead of flat text: prose, tables (kept as Markdown), figures (saved as images with their captions and the sentences that cite them), spreadsheet statistics, and code windows with file name and line range. Deep analysis sends pages to a vision model to read formulas, charts and scans.", true],
  ["Chunk", "Semantic sentence chunker, parent and child sizes",
   "Each document is cut into parent chunks of about 400 words, and each parent into child chunks of about 100 words. Children are precise to search; parents give the model enough surrounding text to answer.", false],
  ["Index", "ChromaDB (vectors) + SQLite (inverted index)",
   "Children and parents are embedded with all-MiniLM-L6-v2. A SQLite inverted index stores TF-IDF weights and term positions. Deleting a document now removes it from every store and refreshes IDF for the rest.", false],
  ["Retrieve", "Dense + BM25 fused with reciprocal rank fusion",
   "Classic mode ranks documents by TF-IDF, then searches vectors inside the best ones. Hybrid mode ranks CHUNKS by BM25 and by dense similarity (child match mapped to its parent) and fuses the rankings, so an exact keyword hit and a semantic hit both survive.", true],
  ["Rerank", "Cross-encoder ms-marco-MiniLM-L-6-v2",
   "A cross-encoder scores each (question, chunk) pair jointly and keeps the best five. It loads in the background, so the first requests may skip it.", false],
  ["Understand the question", "Follow-up rewrite, rolling memory summary",
   "A follow-up such as \"and the second one?\" is rewritten into a standalone question for searching. The last N turns (you choose 2 to 12) are sent verbatim and older turns are summarised, so long conversations keep their context.", true],
  ["Calculate", "Safe pandas plans for spreadsheets",
   "Averages, counts and top-N questions over a sheet cannot be answered from a few retrieved rows. The model plans a small JSON query (filters, group-by, aggregations) and pandas executes it exactly. No model-written code is ever run.", true],
  ["Generate", "Groq LLM, grounded system prompt, streamed",
   "The model sees numbered passages labelled with their type and location (table, figure, code file and lines, page) and a strict instruction to answer only from them. Tokens stream to the browser as they are produced. Code, tables and LaTeX formulas are rendered properly.", false],
  ["Cite and illustrate", "Sources and related figures",
   "Each answer lists the passages it used (document, page, table or code lines). Figures are shown when you name one, when one was used as evidence, or when the question is visual (diagram, chart, architecture).", true],
  ["Self-evaluate", "LLM judge: relevance, grounding, quality",
   "A second call scores the answer from 1 to 5 on three dimensions. The average is the system's own confidence.", false],
  ["Reflect", "Reformulate and retry below threshold",
   "If the average falls under your threshold, the question is rewritten as a keyword-focused query and the loop runs again. If it still fails, the system reports that the topic is not in the documents rather than guessing.", false],
  ["Verify", "Cascaded sentence verification",
   "Every answer sentence is compared with evidence sentences by embedding similarity. Clear matches and clear misses are decided without a model call. Only the ambiguous middle goes to the LLM, in one batched call.", true],
  ["Repair", "Sentence-level corrective regeneration",
   "Unsupported sentences are rewritten against the evidence, or pruned. The repaired answer is re-verified and kept only if it is measurably better. Supported sentences are never touched.", true],
  ["Calibrate", "ECE, Brier, AUROC, risk–coverage",
   "The Lab compares the system's confidence with real correctness and with user ratings, learns a cross-validated recalibration, and shows how much error you avoid by abstaining on low-confidence questions.", true],
];

export function mount(root) {
  root.innerHTML = `
  <div class="page">
    <div><h1>Architecture</h1>
      <p class="muted" style="max-width:62ch">A question passes through retrieval, generation and two layers of checking before you see a confidence score. Stages marked <b>new</b> are the research contributions that the Lab ablates.</p></div>
    <div class="pipe">
      ${STAGES.map(([name, tech, body, isNew], i) => `
        <div class="stage ${isNew ? "new" : ""}">
          <div class="rail-line"><div class="node">${i + 1}</div></div>
          <details ${i === 3 ? "open" : ""}>
            <summary>${esc(name)} ${isNew ? '<span class="chip brand">new</span>' : ""}<span class="tech">${esc(tech)}</span></summary>
            <div class="body"><p style="margin:0">${esc(body)}</p></div>
          </details>
        </div>`).join("")}
    </div>
    <section class="card stack" style="max-width:820px"><h2>Research contributions</h2>
      <p class="small"><b>Chunk-level hybrid retrieval.</b> Keyword and semantic rankings are fused per chunk instead of per document.</p>
      <p class="small"><b>Cascaded verification.</b> Embedding similarity decides the easy sentences, so the LLM judge is called only for ambiguous ones. The Lab reports how many calls this saves.</p>
      <p class="small"><b>Corrective regeneration.</b> Repair acts on the unsupported sentences instead of retrying the whole query.</p>
      <p class="small"><b>Calibrated abstention.</b> Confidence is treated as a prediction of correctness and evaluated as one.</p>
    </section>
  </div>`;
}
export function show() {}
