// lab.js — experiments: ablation study (the research contribution), benchmark, comparisons,
// verify & repair, index explorer.
import {
  $, $$, esc, fmt, icon, api, download, toast, fail, on, state, persist, downloadText,
} from "./core.js";
import { hbars, lineChart, reliability, legend, col } from "./charts.js";

let root, current = "ablation", status = null;
let job = { id: sessionStorage.getItem("sr.job"), timer: null };
const built = new Set();

const TABS = [
  ["ablation", "Ablation study"], ["benchmark", "Benchmark"], ["compare", "Compare"],
  ["verify", "Verify and repair"], ["index", "Index explorer"],
];

export function mount(el) {
  root = el;
  root.innerHTML = `
  <div class="page">
    <div><h1>Lab</h1>
      <p class="muted">Measure the system. The ablation study adds one component at a time to a plain dense retriever and reports accuracy, faithfulness, calibration and cost for each step.</p></div>
    <div class="tabs" role="tablist" id="lab-tabs">${TABS.map(([k, v]) => `<button role="tab" data-tab="${k}" aria-selected="${k === current}">${v}</button>`).join("")}</div>
    ${TABS.map(([k]) => `<section id="lab-${k}" class="stack" ${k === current ? "" : "hidden"}></section>`).join("")}
  </div>`;
  on($("#lab-tabs", root), "click", "[data-tab]", (e, b) => switchTab(b.dataset.tab));
  document.addEventListener("docs:changed", () => built.forEach((k) => fillDocSelects()));
  switchTab(current);
}
export function show() { fillDocSelects(); if (job.id && !job.timer && !jobDone) poll(); }

function switchTab(k) {
  current = k;
  $$("#lab-tabs button", root).forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === k)));
  TABS.forEach(([t]) => ($(`#lab-${t}`, root).hidden = t !== k));
  if (!built.has(k)) { built.add(k); ({ ablation: buildAblation, benchmark: buildBench, compare: buildCompare, verify: buildVerify, index: buildIndex })[k](); fillDocSelects(); }
}

const docOptions = (withAll, sel) =>
  (withAll ? `<option value="">All documents</option>` : "") +
  state.docs.map((d) => `<option value="${esc(d)}" ${d === sel ? "selected" : ""}>${esc(d)}</option>`).join("");
function fillDocSelects() {
  $$("select[data-docs]", root).forEach((s) => {
    const keep = s.value || state.activeDoc || "";
    s.innerHTML = docOptions(s.dataset.docs === "all", keep);
    if ([...s.options].some((o) => o.value === keep)) s.value = keep;
  });
}

function parseQuestions(text) {
  let arr;
  try { arr = JSON.parse(text); } catch { throw new Error("The question set is not valid JSON. Expected [{\"question\": \"…\", \"expected\": \"…\"}]."); }
  if (!Array.isArray(arr) || !arr.length) throw new Error("Provide a non-empty JSON array of questions.");
  arr.forEach((q, i) => { if (!q.question || !q.expected) throw new Error(`Item ${i + 1} needs both "question" and "expected".`); });
  return arr;
}

function questionSetUI(prefix) {
  return `
    <div class="field"><label for="${prefix}-qs">Question set (JSON)</label>
      <textarea id="${prefix}-qs" class="inp mono" rows="7" spellcheck="false" placeholder='[{"question": "Who created Flask?", "expected": "Armin Ronacher"}]'></textarea></div>
    <div class="row wrap">
      <label class="btn sm">${icon("file")} Load file<input type="file" accept=".json,application/json" data-qfile="${prefix}" hidden></label>
      <button class="btn sm" data-qgen="${prefix}">Generate from document</button>
      <button class="btn sm" data-qsaved="${prefix}">Use questions from the Library</button>
      <span class="muted small" id="${prefix}-qc"></span>
    </div>`;
}
function bindQuestionSet(scope, prefix, docSelectSel) {
  const ta = $(`#${prefix}-qs`, scope), count = () => {
    try { $(`#${prefix}-qc`, scope).textContent = `${parseQuestions(ta.value).length} questions`; } catch { $(`#${prefix}-qc`, scope).textContent = ta.value.trim() ? "Invalid JSON" : ""; }
  };
  ta.addEventListener("input", count);
  on(scope, "change", `[data-qfile="${prefix}"]`, async (e, inp) => {
    const f = inp.files[0]; if (!f) return; ta.value = await f.text(); count(); inp.value = "";
  });
  on(scope, "click", `[data-qgen="${prefix}"]`, async (e, b) => {
    const doc = $(docSelectSel, scope).value || state.activeDoc; if (!doc) return toast("Choose a document first.", "bad");
    const t = b.textContent; b.disabled = true; b.innerHTML = `<span class="spinner"></span> Generating…`;
    try { const r = await api("/generate-qa", { json: { doc_id: doc, count: 10 } }); ta.value = JSON.stringify(r.pairs, null, 2); count(); }
    catch (err) { fail(err); } finally { b.disabled = false; b.textContent = t; }
  });
  on(scope, "click", `[data-qsaved="${prefix}"]`, () => {
    let saved = null; try { saved = JSON.parse(localStorage.getItem("sr.qa")); } catch {}
    if (!saved || !saved.pairs) return toast("Nothing saved yet. Generate questions from a document in the Library.", "bad");
    ta.value = JSON.stringify(saved.pairs, null, 2); count();
  });
  let saved = null; try { saved = JSON.parse(localStorage.getItem("sr.qa")); } catch {}
  if (saved && saved.pairs && !ta.value) { ta.value = JSON.stringify(saved.pairs, null, 2); count(); }
}

/* ═══════════════════════════ Ablation ═══════════════════════════ */
let jobDone = false;

async function buildAblation() {
  const el = $("#lab-ablation", root);
  let ladder = [];
  try { ladder = (await api("/api/research/ladder")).ladder; } catch (e) { fail(e); }
  try { status = await api("/api/status"); } catch {}
  const sameModel = status && status.gen_model === status.judge_model;
  el.innerHTML = `
    <div class="two" style="align-items:start">
      <section class="card stack">
        <h2>Experiment setup</h2>
        <div class="field"><label for="ab-doc">Documents to search</label><select id="ab-doc" class="inp" data-docs="all"></select></div>
        ${questionSetUI("ab")}
        <div class="field"><label>Configurations <span class="muted small">(each adds one component to the row above)</span></label>
          <div class="config-list">${ladder.map((c) => `<label class="check"><input type="checkbox" data-cfg="${c.id}" checked>
            <span><b>${esc(c.id)}</b> ${esc(c.name)}</span></label>`).join("")}</div></div>
        <div class="field"><label for="ab-thr">Reflection threshold: <b id="ab-thr-v" class="num">3</b></label><input id="ab-thr" type="range" min="1" max="5" step="0.5" value="3"></div>
        <div class="row"><button class="btn primary" id="ab-run">${icon("play")} Run ablation</button>
          <button class="btn" id="ab-stop" hidden>Stop</button><span class="muted small" id="ab-cost"></span></div>
      </section>
      <section class="card stack">
        <h2>How it is measured</h2>
        <p class="small">Each configuration answers every question. Correctness comes from an LLM judge plus token-level F1 against your expected answer. Faithfulness is measured with a strict verifier that sends <b>every</b> answer sentence to the judge model, independently of the cheaper cascade the system itself uses.</p>
        <p class="small">Calibration compares the system's own confidence (average self-evaluation score ÷ 5) with actual correctness, using cross-validated recalibration so nothing leaks between folds.</p>
        ${status ? `<div class="notice ${sameModel ? "warn" : "ok"}">Generator: <b>${esc(status.gen_model)}</b>. Judge: <b>${esc(status.judge_model)}</b>.
          ${sameModel ? "They are the same model, so the judge may favour its own style. For results you intend to publish, set JUDGE_MODEL to a different, stronger model." : ""}</div>` : ""}
        <div class="notice info small">Use at least 50 questions for the confidence intervals to mean anything. With 10 questions the intervals are wide enough that most differences will not be significant.</div>
      </section>
    </div>
    <div id="ab-progress"></div>
    <div id="ab-results" class="stack"></div>`;
  fillDocSelects();
  bindQuestionSet(el, "ab", "#ab-doc");
  const upd = () => {
    const n = (() => { try { return parseQuestions($("#ab-qs", el).value).length; } catch { return 0; } })();
    const k = $$("[data-cfg]:checked", el).length;
    $("#ab-cost", el).textContent = n && k ? `About ${n * k * 7} model calls` : "";
  };
  el.addEventListener("input", upd); el.addEventListener("change", upd);
  $("#ab-thr", el).oninput = (e) => ($("#ab-thr-v", el).textContent = e.target.value);
  $("#ab-run", el).onclick = startAblation;
  $("#ab-stop", el).onclick = async () => { try { await api(`/api/research/jobs/${job.id}/stop`, { method: "POST" }); toast("Stopping after the current question"); } catch (e) { fail(e); } };
  if (job.id) poll();
}

async function startAblation() {
  const el = $("#lab-ablation", root);
  try {
    const questions = parseQuestions($("#ab-qs", el).value);
    const configs = $$("[data-cfg]:checked", el).map((c) => c.dataset.cfg);
    if (!configs.length) throw new Error("Select at least one configuration.");
    const doc = $("#ab-doc", el).value;
    if (!doc && state.docs.length === 0) throw new Error("Index a document first.");
    const r = await api("/api/research/ablation", { json: { questions, configs, doc_ids: doc ? [doc] : state.docs, threshold: +$("#ab-thr", el).value } });
    job.id = r.job_id; sessionStorage.setItem("sr.job", job.id); jobDone = false;
    $("#ab-results", el).innerHTML = ""; poll();
  } catch (e) { fail(e); }
}

function poll() {
  clearInterval(job.timer);
  const tick = async () => {
    const el = $("#lab-ablation", root); if (!el || !job.id) return;
    try {
      const j = await api(`/api/research/jobs/${job.id}`);
      const running = j.status === "running";
      $("#ab-run", el).disabled = running; $("#ab-stop", el).hidden = !running;
      const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      $("#ab-progress", el).innerHTML = j.status === "error" ? `<div class="notice bad">The run failed: ${esc(j.error)}</div>` : `
        <div class="card stack"><div class="row"><b class="grow">${running ? "Running" : j.status === "stopped" ? "Stopped" : "Finished"}</b>
          <span class="muted small num">${j.done} / ${j.total || "?"} · ${Math.round(j.elapsed)}s</span></div>
          <div class="progress" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><i style="width:${pct}%"></i></div>
          <span class="muted small">${esc(j.message || "")}</span></div>`;
      if (!running) {
        clearInterval(job.timer); job.timer = null; jobDone = true;
        if (j.rows) renderResults(j);
      }
    } catch (e) {
      clearInterval(job.timer); job.timer = null;
      if (/not found/i.test(e.message)) { sessionStorage.removeItem("sr.job"); job.id = null; } else fail(e);
    }
  };
  tick(); job.timer = setInterval(tick, 1500);
}

function renderResults(j) {
  const el = $("#ab-results", $("#lab-ablation", root));
  const rows = j.rows || [];
  if (!rows.length) { el.innerHTML = `<div class="notice warn">No configuration finished. Check the progress message above.</div>`; return; }
  const errs = rows.reduce((a, r) => a + r.errors, 0);
  el.innerHTML = `
    ${errs ? `<div class="notice warn">${errs} question runs failed (rate limit or timeout) and are excluded from the averages. Re-run or reduce the question count.</div>` : ""}
    <section class="card stack">
      <div class="row wrap"><h2 class="grow">Results</h2>
        <a class="btn sm" href="/api/research/jobs/${job.id}/export?fmt=csv">${icon("download")} CSV</a>
        <a class="btn sm" href="/api/research/jobs/${job.id}/export?fmt=json">${icon("download")} Full JSON</a>
        <button class="btn sm" id="ab-tex">${icon("copy")} LaTeX table</button></div>
      <div class="table-wrap"><table>
        <thead><tr><th>ID</th><th>Configuration</th><th class="n">Accuracy %</th><th class="n">95% CI</th><th class="n">F1</th><th class="n">Faithful %</th><th class="n">ECE ↓</th><th class="n">AURC ↓</th><th class="n">LLM calls</th><th class="n">Latency s</th><th class="n">p vs previous</th></tr></thead>
        <tbody>${rows.map((r) => `<tr><td><b>${esc(r.id)}</b></td><td>${esc(r.name)}</td>
          <td class="n"><b>${fmt.n(r.accuracy, 1)}</b></td><td class="n">${r.ci[0]}–${r.ci[1]}</td><td class="n">${fmt.n(r.token_f1, 3)}</td>
          <td class="n">${fmt.n(r.faithfulness, 1)}</td><td class="n">${fmt.n(r.ece, 3)}</td><td class="n">${fmt.n(r.aurc, 3)}</td>
          <td class="n">${fmt.n(r.llm_calls, 1)}</td><td class="n">${fmt.n(r.latency, 1)}</td>
          <td class="n">${r.p_vs_prev === null ? "–" : `<span class="${r.p_vs_prev < 0.05 ? "judge-ok" : ""}">${r.p_vs_prev.toFixed(3)}${r.p_vs_prev < 0.05 ? " *" : ""}</span>`}</td></tr>`).join("")}</tbody></table></div>
      <p class="small muted" style="margin:0">n = ${rows[0].n} questions per configuration. * marks a paired-bootstrap p-value under 0.05 against the previous row. ECE and AURC are lower-is-better.</p>
    </section>
    <div class="two">
      <section class="card stack"><h3>Accuracy with 95% intervals</h3>
        ${hbars({ items: rows.map((r, i) => ({ label: `${r.id} ${r.name}`.slice(0, 26), value: r.accuracy, ci: r.ci, color: col(0) })), max: 100, fmt: (v) => v.toFixed(1), label: "accuracy by configuration" })}</section>
      <section class="card stack"><h3>Faithfulness and cost</h3>
        ${hbars({ items: rows.map((r) => ({ label: `${r.id} faithfulness`, value: r.faithfulness, color: "var(--ok)" })), max: 100, fmt: (v) => v.toFixed(1), label: "faithfulness" })}
        <div class="small muted">LLM calls per question: ${rows.map((r) => `${r.id} ${fmt.n(r.llm_calls, 1)}`).join(" · ")}</div></section>
    </div>
    ${calibrationHtml(j.calibration, rows[rows.length - 1])}`;
  $("#ab-tex", el).onclick = async () => {
    try {
      const txt = await (await fetch(`/api/research/jobs/${job.id}/export?fmt=tex`)).text();
      await navigator.clipboard.writeText(txt).catch(() => downloadText("ablation.tex", txt));
      toast("LaTeX table copied", "ok");
    } catch (e) { fail(e); }
  };
}

export function calibrationHtml(c, row) {
  if (!c) return "";
  if (c.error) return `<div class="notice warn">Calibration: ${esc(c.error)}</div>`;
  const k = (name, v, d = 3) => `<div class="kpi"><b class="num">${fmt.n(v, d)}</b><span>${name}</span></div>`;
  return `
    <section class="card stack">
      <div><h2>Calibration of the self-evaluation${row ? ` (${esc(row.id)})` : ""}</h2>
        <p class="muted small" style="margin:4px 0 0">Does a confidence of 0.8 mean the answer is right about 80% of the time? Raw is the system's own score. Recalibrated is a cross-validated logistic model over the four self-evaluation scores.</p></div>
      <div class="kpis">
        ${k("ECE, raw", c.raw.ece)}${k("ECE, recalibrated", c.calibrated.ece)}
        ${k("AUROC, raw", c.raw.auroc, 2)}${k("AUROC, recalibrated", c.calibrated.auroc, 2)}
        ${k("AURC, raw", c.raw.aurc)}${k("AURC, recalibrated", c.calibrated.aurc)}
        ${k("Accuracy", c.accuracy, 2)}${k("Questions", c.n, 0)}
      </div>
      <div class="two">
        <div><h3>Reliability</h3>${reliability(c.raw.bins)}<p class="small muted">Points on the dashed diagonal are perfectly calibrated.</p></div>
        <div><h3>Risk and coverage</h3>
          ${lineChart({ series: [
            { name: "raw", points: c.raw.risk_coverage.map((p) => ({ x: p.coverage, y: p.risk })) },
            { name: "recalibrated", points: c.calibrated.risk_coverage.map((p) => ({ x: p.coverage, y: p.risk })), color: "var(--ok)" }],
            xLabel: "Coverage (share of questions answered)", yLabel: "Error rate among answered", label: "risk coverage" })}
          ${legend([{ name: "Raw", color: col(0) }, { name: "Recalibrated", color: "var(--ok)" }])}
          <p class="small muted">A good confidence signal lets the system abstain on its worst answers, so error falls as coverage shrinks. Oracle AURC is ${fmt.n(c.oracle_aurc, 3)}.</p></div>
      </div>
    </section>`;
}

/* ═══════════════════════════ Benchmark ═══════════════════════════ */
function buildBench() {
  const el = $("#lab-benchmark", root);
  el.innerHTML = `
    <section class="card stack">
      <h2>Accuracy benchmark</h2>
      <p class="muted small" style="margin:0">Runs the standard pipeline on each question and has an LLM judge compare the answer with your expected answer.</p>
      <div class="field"><label for="bm-doc">Documents</label><select id="bm-doc" class="inp" data-docs="all" style="max-width:320px"></select></div>
      ${questionSetUI("bm")}
      <div class="row"><button class="btn primary" id="bm-run">${icon("play")} Run benchmark</button><label class="btn">${icon("file")} Upload and run<input type="file" id="bm-up" accept=".json" hidden></label></div>
    </section>
    <div id="bm-out"></div>`;
  fillDocSelects(); bindQuestionSet(el, "bm", "#bm-doc");
  const run = async (fn) => {
    const out = $("#bm-out", el); out.innerHTML = `<div class="card"><span class="spinner"></span> Running. Each question makes several model calls, so this takes a while.</div>`;
    $("#bm-run", el).disabled = true;
    try { renderBench(await fn()); } catch (e) { out.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; } finally { $("#bm-run", el).disabled = false; }
  };
  $("#bm-run", el).onclick = () => { let qs; try { qs = parseQuestions($("#bm-qs", el).value); } catch (e) { return fail(e); }
    run(() => api("/benchmark", { json: { questions: qs, doc_id: $("#bm-doc", el).value || null } })); };
  $("#bm-up", el).onchange = (e) => { const f = e.target.files[0]; if (!f) return; const fd = new FormData(); fd.append("file", f); fd.append("doc_id", $("#bm-doc", el).value || "");
    run(() => api("/benchmark-upload", { body: fd, method: "POST" })); e.target.value = ""; };
}
function renderBench(r) {
  const out = $("#bm-out", root);
  const avg = r.results.length ? r.results.reduce((a, x) => a + (x.avg_score || 0), 0) / r.results.length : 0;
  out.innerHTML = `
    <div class="kpis"><div class="kpi"><b>${r.accuracy_pct}%</b><span>Accuracy</span></div><div class="kpi"><b class="num">${r.correct}/${r.total}</b><span>Correct</span></div>
      <div class="kpi"><b class="num">${avg.toFixed(2)}</b><span>Mean self-evaluation score</span></div></div>
    <div class="table-wrap"><table><thead><tr><th></th><th>Question</th><th>Expected</th><th>Answer</th><th>Judge</th><th class="n">Score</th></tr></thead>
    <tbody>${r.results.map((x) => `<tr><td class="${x.correct ? "judge-ok" : "judge-bad"}">${x.correct ? "✓" : "✗"}</td><td>${esc(x.question)}</td><td>${esc(x.expected)}</td>
      <td class="small">${esc((x.actual || "").slice(0, 280))}</td><td class="small muted">${esc(x.reason)}</td><td class="n">${fmt.n(x.avg_score, 1)}</td></tr>`).join("")}</tbody></table></div>`;
}

/* ═══════════════════════════ Compare ═══════════════════════════ */
function buildCompare() {
  const el = $("#lab-compare", root);
  el.innerHTML = `
    <div class="seg" id="cmp-mode"><button data-m="ret" aria-pressed="true">TF-IDF vs BM25</button><button data-m="doc" aria-pressed="false">Document vs document</button></div>
    <section class="card stack">
      <div class="field"><label for="cmp-q">Question</label><input id="cmp-q" class="inp" placeholder="Who created Flask?"></div>
      <div class="row wrap" id="cmp-docs-ret"><div class="field"><label>Document</label><select id="cmp-d0" class="inp" data-docs="all"></select></div></div>
      <div class="row wrap" id="cmp-docs-doc" hidden>
        <div class="field"><label>Document A</label><select id="cmp-da" class="inp" data-docs="only"></select></div>
        <div class="field"><label>Document B</label><select id="cmp-db" class="inp" data-docs="only"></select></div></div>
      <div><button class="btn primary" id="cmp-run">Compare</button></div>
    </section><div id="cmp-out"></div>`;
  fillDocSelects();
  let mode = "ret";
  on($("#cmp-mode", el), "click", "[data-m]", (e, b) => {
    mode = b.dataset.m; $$("#cmp-mode button", el).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    $("#cmp-docs-ret", el).hidden = mode !== "ret"; $("#cmp-docs-doc", el).hidden = mode !== "doc";
  });
  const side = (title, s, extra = "") => s && s.error ? `<div class="card"><h3>${esc(title)}</h3><div class="notice bad">${esc(s.error)}</div></div>` : `
    <div class="card stack"><div class="row"><h3 class="grow">${esc(title)}</h3>${extra}</div>
      <div class="row wrap"><span class="chip brand">Score ${fmt.n(s.scores.average_score, 2)}/5</span><span class="chip">R ${s.scores.retrieval_relevance}</span><span class="chip">G ${s.scores.answer_grounding}</span><span class="chip">Q ${s.scores.answer_quality}</span>
      ${s.hallucination_pct !== undefined ? `<span class="chip ${s.hallucination_pct ? "bad" : "ok"}">${s.hallucination_pct}% flagged</span>` : ""}</div>
      <div style="font-family:var(--font-read)">${esc(s.answer).replace(/\n/g, "<br>")}</div>
      ${s.ranked_docs ? `<div class="small muted">Ranked: ${s.ranked_docs.map((d) => `${esc(d.doc_id)} (${d.score})`).join(", ") || "none"}</div>` : ""}</div>`;
  $("#cmp-run", el).onclick = async (e) => {
    const q = $("#cmp-q", el).value.trim(); if (!q) return toast("Enter a question.", "bad");
    const out = $("#cmp-out", el), b = e.currentTarget; b.disabled = true; out.innerHTML = `<div class="card"><span class="spinner"></span> Comparing…</div>`;
    try {
      if (mode === "ret") {
        const r = await api("/compare-retrievers", { json: { query: q, doc_id: $("#cmp-d0", el).value || null } });
        const w = (k) => (r.winner === k ? `<span class="chip ok">Higher score</span>` : r.winner === "tie" ? `<span class="chip">Tie</span>` : "");
        out.innerHTML = `<div class="two">${side("TF-IDF + dense", r.tfidf, w("tfidf"))}${side("BM25", r.bm25, w("bm25"))}</div>`;
      } else {
        const a = $("#cmp-da", el).value, bb = $("#cmp-db", el).value; if (!a || !bb) throw new Error("Choose two documents.");
        const r = await api("/compare-docs", { json: { query: q, doc_id_a: a, doc_id_b: bb } });
        out.innerHTML = `<div class="two">${side(a, r.a)}${side(bb, r.b)}</div>`;
      }
    } catch (err) { out.innerHTML = `<div class="notice bad">${esc(err.message)}</div>`; } finally { b.disabled = false; }
  };
}

/* ═══════════════════════════ Verify ═══════════════════════════ */
function buildVerify() {
  const el = $("#lab-verify", root);
  el.innerHTML = `
    <section class="card stack">
      <h2>Verify an answer against the evidence</h2>
      <p class="muted small" style="margin:0">Paste any question and answer. The cascade checks each sentence against retrieved passages, asking an LLM only when the embedding similarity is ambiguous.</p>
      <div class="field"><label for="vf-doc">Documents</label><select id="vf-doc" class="inp" data-docs="all" style="max-width:320px"></select></div>
      <div class="field"><label for="vf-q">Question</label><input id="vf-q" class="inp"></div>
      <div class="field"><label for="vf-a">Answer to check</label><textarea id="vf-a" class="inp" rows="5"></textarea></div>
      <label class="check"><input type="checkbox" id="vf-rep" checked><span>Also repair unsupported sentences</span></label>
      <div class="row"><button class="btn primary" id="vf-run">Verify</button><button class="btn" id="vf-prov">Show source passage per sentence</button></div>
    </section><div id="vf-out" class="stack"></div>`;
  fillDocSelects();
  const out = $("#vf-out", el);
  const body = () => ({ query: $("#vf-q", el).value.trim(), answer: $("#vf-a", el).value.trim(), doc_ids: $("#vf-doc", el).value ? [$("#vf-doc", el).value] : null });
  $("#vf-run", el).onclick = async (e) => {
    const b = body(); if (!b.query || !b.answer) return toast("Enter both a question and an answer.", "bad");
    e.currentTarget.disabled = true; out.innerHTML = `<div class="card"><span class="spinner"></span> Verifying…</div>`;
    try {
      const r = await api("/api/research/verify", { json: { ...b, repair: $("#vf-rep", el).checked } });
      const v = r.verification;
      out.innerHTML = `<section class="card stack"><div class="row wrap"><h3 class="grow">Sentence verdicts</h3>
        <span class="chip ${v.unsupported_count ? "bad" : "ok"}">${v.unsupported_count} of ${v.total_count} unsupported</span>
        <span class="chip">${v.embedding_decided} decided by embeddings</span><span class="chip">${v.llm_calls} LLM call</span></div>
        ${v.sentences.map((s) => `<div class="sent ${s.supported ? "ok" : "bad"}"><span>${esc(s.text)}</span>
          <small>${s.supported ? "Supported" : "Not supported"} · similarity ${fmt.n(s.similarity, 2)} · ${esc(s.method)}</small>
          <small>Closest evidence: ${esc((s.evidence || "").slice(0, 200))}</small></div>`).join("") || `<p class="muted">No checkable sentences.</p>`}</section>
        ${r.repair ? (r.repair.changed ? `<section class="card stack"><div class="row"><h3 class="grow">Repaired answer</h3><span class="chip ok">${esc(r.repair.method)} · ${Math.round(r.repair.before * 100)}% → ${Math.round(r.repair.after * 100)}% flagged</span></div>
          <div style="font-family:var(--font-read)">${esc(r.repair.answer).replace(/\n/g, "<br>")}</div></section>` : `<div class="notice info">No repair candidate improved on the original.</div>`) : ""}`;
    } catch (err) { out.innerHTML = `<div class="notice bad">${esc(err.message)}</div>`; } finally { e.currentTarget.disabled = false; }
  };
  $("#vf-prov", el).onclick = async () => {
    const b = body(); if (!b.query || !b.answer) return toast("Enter both a question and an answer.", "bad");
    out.innerHTML = `<div class="card"><span class="spinner"></span> Matching sentences…</div>`;
    try {
      const r = await api("/explain-answer", { json: { query: b.query, answer: b.answer, doc_id: b.doc_ids ? b.doc_ids[0] : null } });
      out.innerHTML = `<section class="card stack"><h3>Source passage per sentence</h3>${(r.provenance || []).map((p) => `<div class="sent ${p.similarity > 0.45 ? "ok" : "bad"}">
        <b>${esc(p.answer_sentence)}</b><small>Similarity ${fmt.n(p.similarity, 2)}</small><span class="small">${esc(p.source_chunk)}</span></div>`).join("") || "<p class='muted'>Nothing to match.</p>"}</section>`;
    } catch (err) { out.innerHTML = `<div class="notice bad">${esc(err.message)}</div>`; }
  };
}

/* ═══════════════════════════ Index explorer ═══════════════════════════ */
function buildIndex() {
  const el = $("#lab-index", root);
  el.innerHTML = `
    <div class="two" style="align-items:start">
      <section class="card stack"><h2>Top terms by TF-IDF</h2>
        <select id="ix-doc" class="inp" data-docs="only" style="max-width:320px"></select><div id="ix-terms"></div></section>
      <section class="card stack"><h2>Where does a term appear?</h2>
        <div class="row"><input id="ix-term" class="inp" placeholder="flask"><button class="btn" id="ix-go">Look up</button></div><div id="ix-res"></div></section>
    </div>`;
  fillDocSelects();
  const load = async () => {
    const id = $("#ix-doc", el).value; const box = $("#ix-terms", el);
    if (!id) { box.innerHTML = `<p class="muted">Add a document first.</p>`; return; }
    try {
      const r = await api(`/page-index-terms?doc_id=${encodeURIComponent(id)}&top_k=20`); const max = Math.max(...r.terms.map((t) => t.tfidf), 0.0001);
      box.innerHTML = hbars({ items: r.terms.map((t) => ({ label: t.term, value: t.tfidf, color: col(0) })), max, fmt: (v) => v.toFixed(3), label: "top terms" });
    } catch (e) { box.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  };
  $("#ix-doc", el).onchange = load; setTimeout(load, 0);
  document.addEventListener("docs:changed", () => setTimeout(load, 0));
  const look = async () => {
    const t = $("#ix-term", el).value.trim().toLowerCase(); if (!t) return;
    try {
      const r = await api(`/page-index-lookup?term=${encodeURIComponent(t)}`);
      $("#ix-res", el).innerHTML = r.results.length ? `<div class="table-wrap"><table><thead><tr><th>Document</th><th class="n">TF</th><th class="n">IDF</th><th class="n">TF-IDF</th><th class="n">Occurrences</th></tr></thead><tbody>
        ${r.results.map((x) => `<tr><td>${esc(x.doc_id)}</td><td class="n">${x.tf.toFixed(4)}</td><td class="n">${x.idf.toFixed(2)}</td><td class="n">${x.tfidf.toFixed(4)}</td><td class="n">${x.positions}</td></tr>`).join("")}</tbody></table></div>` : `<p class="muted">“${esc(t)}” is not in the index (stop words are removed).</p>`;
    } catch (e) { $("#ix-res", el).innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  };
  $("#ix-go", el).onclick = look; $("#ix-term", el).onkeydown = (e) => { if (e.key === "Enter") look(); };
}
