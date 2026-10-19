// insights.js — usage analytics, query log, feedback calibration, saved answers, saved sessions.
import { $, $$, esc, fmt, icon, api, download, toast, fail, dialog, confirmDialog, promptDialog, on, debounce, state } from "./core.js";
import { lineChart, columns, legend, col } from "./charts.js";
import { calibrationHtml } from "./lab.js";

let root, logs = [], selected = new Set();

export function mount(el) {
  root = el;
  root.innerHTML = `
  <div class="page">
    <div class="row wrap"><div class="grow"><h1>Insights</h1>
      <p class="muted" style="margin:4px 0 0">What has been asked, how well it was answered, and whether the system's confidence can be trusted.</p></div>
      <button class="btn" id="ins-pdf">${icon("download")} PDF report</button>
      <button class="btn" id="ins-refresh">${icon("refresh")} Refresh</button></div>
    <section class="card stack" id="ins-acc"></section>
    <div id="ins-kpis" class="kpis"></div>
    <div id="ins-charts" class="stack"></div>
    <section class="card stack"><div class="row wrap"><h2 class="grow">Query log</h2>
      <input class="inp" id="log-q" placeholder="Search questions" style="width:220px" aria-label="Search questions">
      <button class="btn sm" id="log-save" disabled>Save selection as session</button></div>
      <div id="log-table"></div></section>
    <div class="two" style="align-items:start">
      <section class="card stack"><h2>Saved answers</h2><div id="bm-list"></div></section>
      <section class="card stack"><h2>Saved sessions</h2><div id="ss-list"></div></section>
    </div>
    <div id="ins-cal"></div>
  </div>`;
  $("#ins-refresh", root).onclick = () => load();
  $("#ins-pdf", root).onclick = () => download("/export-pdf", {}, "report.pdf").catch(fail);
  $("#log-q", root).oninput = debounce(renderLog, 150);
  $("#log-save", root).onclick = saveSession;
  on($("#log-table", root), "click", "tr[data-i]", (e, tr) => { if (!e.target.closest("input")) openLog(+tr.dataset.i); });
  on($("#log-table", root), "change", "input[data-sel]", (e, c) => {
    const id = +c.dataset.sel; c.checked ? selected.add(id) : selected.delete(id); $("#log-save", root).disabled = !selected.size;
  });
  on($("#bm-list", root), "click", "[data-bm]", async (e, b) => {
    const id = b.dataset.id;
    try {
      if (b.dataset.bm === "del") { await api(`/bookmark/${id}`, { method: "DELETE" }); loadBookmarks(); }
      else if (b.dataset.bm === "note") {
        const note = await promptDialog("Note", "Add a note to this answer", b.dataset.note || "");
        if (note !== null) { await api(`/bookmark/${id}`, { method: "PATCH", json: { note } }); loadBookmarks(); }
      } else if (b.dataset.bm === "open") {
        const bm = bookmarks.find((x) => String(x.id) === id);
        dialog(bm.query, `<div style="font-family:var(--font-read);line-height:1.7">${esc(bm.answer).replace(/\n/g, "<br>")}</div>`);
      }
    } catch (err) { fail(err); }
  });
  on($("#ss-list", root), "click", "[data-ss]", async (e, b) => {
    try {
      const r = await api(`/sessions/${b.dataset.ss}`);
      dialog(b.dataset.name, `<div class="stack">${r.queries.map((q) => `<div class="sent"><b>${esc(q.query)}</b><span class="small">${esc((q.answer || "").slice(0, 400))}</span>
        <small>Score ${fmt.n(q.average_score, 1)} · ${fmt.date(q.created_at)}</small></div>`).join("") || "<p class='muted'>Empty session.</p>"}</div>`, { wide: true });
    } catch (err) { fail(err); }
  });
}
export function show() { load(); }

let bookmarks = [];

async function loadAccuracy() {
  const r = await api("/api/accuracy");
  const el = $("#ins-acc", root), L = r.latest, runs = r.runs || [];
  const tone = !L ? "" : L.accuracy_pct >= 80 ? "ok" : L.accuracy_pct >= 60 ? "brand" : "warn";
  el.innerHTML = `
    <div class="row wrap"><div class="grow"><h2>How accurate is it?</h2>
      <p class="muted small" style="margin:4px 0 0;max-width:78ch">Accuracy is measured by asking questions whose correct answers are known and letting a judge model compare each answer with the expected one.
        The score shown under each answer is different: it is the system grading itself, so treat it as confidence, not accuracy.</p></div>
      <button class="btn primary" id="acc-run">${icon("play")} Test accuracy now</button>
      <a class="btn" href="#/lab">Full tests in the Lab</a></div>
    ${L ? `<div class="row wrap" style="gap:24px">
        <div><div style="font-size:2.4rem;font-weight:650;letter-spacing:-.02em" class="num">${L.accuracy_pct}%</div>
          <span class="chip ${tone}">${L.correct} of ${L.total} questions answered correctly</span></div>
        <div class="small muted">Last measured ${esc(L.ts)}<br>Document: ${esc(L.doc_id)}<br>Mean self-evaluation score ${fmt.n(L.avg_self_score, 2)} / 5</div>
        ${runs.length > 1 ? `<div style="flex:1;min-width:260px;max-width:460px">${lineChart({ series: [{ name: "Accuracy", points: runs.map((x, i) => ({ x: i + 1, y: x.accuracy_pct })) }],
          xMin: 1, xMax: Math.max(2, runs.length), yMin: 0, yMax: 100, w: 460, h: 160, xTicks: Math.min(runs.length - 1, 5) || 1, yTicks: 4, xLabel: "Test run", label: "accuracy over runs" })}</div>` : ""}
      </div>` : `<div class="notice info">Not measured yet. Click "Test accuracy now": it writes questions from one of your documents, answers them and reports the result (about a minute).</div>`}`;
  $("#acc-run", root).onclick = quickTest;
}

function quickTest() {
  if (!state.docs.length) return toast("Add a document in the Library first.", "bad");
  const dlg = dialog("Test accuracy", `
    <p class="small">The model writes question and answer pairs from the document you choose, the system answers them, and a judge compares each answer with the expected one.
      Because the questions are model-written this is a quick sanity check. For a result you can defend, write your own questions in the Lab.</p>
    <div class="field"><label for="qt-doc">Document</label><select id="qt-doc" class="inp">${state.docs.map((d) => `<option ${d === state.activeDoc ? "selected" : ""}>${esc(d)}</option>`).join("")}</select></div>
    <div class="field"><label for="qt-n">Number of questions</label><select id="qt-n" class="inp"><option>5</option><option selected>10</option><option>15</option></select></div>
    <div id="qt-out"></div>`, { foot: `<button class="btn" data-close>Close</button><button class="btn primary" id="qt-go">Start</button>` });
  $("#qt-go", dlg.el).onclick = async (e) => {
    const btn = e.currentTarget, out = $("#qt-out", dlg.el), doc = $("#qt-doc", dlg.el).value; btn.disabled = true;
    try {
      out.innerHTML = `<span class="spinner"></span> Writing questions from the document…`;
      const qa = await api("/generate-qa", { json: { doc_id: doc, count: +$("#qt-n", dlg.el).value } });
      out.innerHTML = `<span class="spinner"></span> Answering and judging ${qa.pairs.length} questions. This takes about a minute…`;
      const res = await api("/benchmark", { json: { questions: qa.pairs, doc_id: doc, source: "quick-test" } });
      out.innerHTML = `<div class="notice ${res.accuracy_pct >= 70 ? "ok" : "warn"}"><b>${res.accuracy_pct}%</b> correct (${res.correct} of ${res.total}).</div>
        <div class="stack" style="margin-top:10px">${res.results.map((x) => `<div class="sent ${x.correct ? "ok" : "bad"}"><b>${esc(x.question)}</b>
          <span class="small">Expected: ${esc(x.expected)}</span><span class="small">Answer: ${esc((x.actual || "").slice(0, 200))}</span><small>${esc(x.reason || "")}</small></div>`).join("")}</div>`;
      loadAccuracy();
    } catch (err) { out.innerHTML = `<div class="notice bad">${esc(err.message)}</div>`; }
    finally { btn.disabled = false; btn.textContent = "Run again"; }
  };
}
async function load() {
  await Promise.allSettled([loadAccuracy(), loadAnalytics(), loadLogs(), loadBookmarks(), loadSessions(), loadCalibration()]);
}

async function loadAnalytics() {
  const [qa, an, fb] = await Promise.all([
    api("/api/query-analytics").catch(() => ({ error: "x" })),
    api("/analytics").catch(() => ({})),
    api("/feedback").catch(() => ({ stats: {} })),
  ]);
  const k = $("#ins-kpis", root), c = $("#ins-charts", root);
  if (qa.error) {
    k.innerHTML = ""; c.innerHTML = `<div class="card empty"><h3>No queries yet</h3><p>Ask a question and the charts appear here.</p><a class="btn primary" href="#/ask">Go to Ask</a></div>`; return;
  }
  const st = fb.stats || {};
  k.innerHTML = `
    <div class="kpi"><b class="num">${qa.total_queries}</b><span>Questions answered</span></div>
    <div class="kpi"><b class="num">${qa.avg_score}</b><span>Self-evaluation score out of 5 (the system's own opinion)</span></div>
    <div class="kpi"><b class="num">${qa.avg_hallucination}%</b><span>Sentences flagged, on average</span></div>
    <div class="kpi"><b class="num">${qa.reformulation_rate}%</b><span>Questions rewritten</span></div>
    <div class="kpi"><b class="num">${st.total ? st.satisfaction_pct + "%" : "–"}</b><span>Helpful ratings${st.total ? ` (${st.total})` : ""}</span></div>`;
  const trend = qa.score_trend || [];
  const words = Object.entries(an.word_freq || {}).slice(0, 30); const wmax = words.length ? words[0][1] : 1;
  c.innerHTML = `
    <div class="two">
      <section class="card stack"><h3>Score and flagged sentences, last ${trend.length} questions</h3>
        ${lineChart({ series: [{ name: "Score", points: trend.map((t, i) => ({ x: i + 1, y: t.average_score })) }], xMin: 1, xMax: Math.max(2, trend.length), yMin: 0, yMax: 5, xLabel: "Question (oldest to newest)", yLabel: "Score", label: "score trend" })}</section>
      <section class="card stack"><h3>Score distribution</h3>
        ${columns({ items: (an.score_dist || []).map((d) => ({ label: d.range, value: d.count, color: d.range === "4-5" ? "var(--ok)" : d.range === "1-2" ? "var(--bad)" : col(0) })), label: "score distribution" })}</section>
    </div>
    <div class="two">
      <section class="card stack"><h3>Questions per day</h3>
        ${columns({ items: (an.daily || []).slice(-14).map((d) => ({ label: d.date.slice(5), value: d.count })), label: "daily questions" })}</section>
      <section class="card stack"><h3>Attempts needed</h3>
        ${columns({ items: Object.entries(qa.attempts_dist || {}).map(([a, n]) => ({ label: `${a}×`, value: n })), label: "attempts" })}
        <p class="small muted" style="margin:0">More than one attempt means the first answer scored below the threshold and the question was rewritten.</p></section>
    </div>
    <section class="card stack"><h3>What people ask about</h3>
      <div class="cloud">${words.map(([w, n]) => `<span style="font-size:${(0.85 + (n / wmax) * 1.3).toFixed(2)}rem;color:${n / wmax > 0.5 ? "var(--brand)" : "var(--ink-2)"}">${esc(w)}</span>`).join("") || "<span class='muted'>No terms yet.</span>"}</div></section>`;
}

async function loadLogs() {
  logs = (await api("/api/logs")).logs || []; selected.clear(); $("#log-save", root).disabled = true; renderLog();
}
function renderLog() {
  const q = $("#log-q", root).value.trim().toLowerCase();
  const rows = logs.map((l, i) => [l, i]).filter(([l]) => !q || (l.query || "").toLowerCase().includes(q));
  $("#log-table", root).innerHTML = rows.length ? `<div class="table-wrap" style="max-height:420px;overflow:auto"><table>
    <thead><tr><th></th><th>Question</th><th class="n">Score</th><th class="n">Flagged</th><th class="n">Tries</th><th>When</th></tr></thead><tbody>
    ${rows.slice(0, 200).map(([l, i]) => `<tr class="click" data-i="${i}"><td><input type="checkbox" data-sel="${l.id}" aria-label="Select" ${selected.has(l.id) ? "checked" : ""}></td>
      <td>${esc((l.query || "").slice(0, 110))}</td><td class="n"><span class="chip ${l.average_score >= 4 ? "ok" : l.average_score >= 3 ? "brand" : "warn"}">${fmt.n(l.average_score, 1)}</span></td>
      <td class="n">${l.hallucination_score === null ? "–" : Math.round(l.hallucination_score * 100) + "%"}</td><td class="n">${l.attempts}</td><td class="small muted">${fmt.date(l.created_at)}</td></tr>`).join("")}
    </tbody></table></div>` : `<div class="empty"><p>${logs.length ? "No questions match." : "Nothing logged yet."}</p></div>`;
}
function openLog(i) {
  const l = logs[i]; if (!l) return;
  dialog(l.query, `
    <div class="row wrap"><span class="chip brand">Score ${fmt.n(l.average_score, 1)}</span><span class="chip">Retrieval ${l.retrieval_relevance}</span><span class="chip">Grounding ${l.answer_grounding}</span><span class="chip">Quality ${l.answer_quality}</span>
      ${l.hallucination_score !== null ? `<span class="chip ${l.hallucination_score ? "bad" : "ok"}">${Math.round(l.hallucination_score * 100)}% flagged</span>` : ""}
      ${(l.cited_sources || []).map((s) => `<span class="chip brand">${esc(s)}</span>`).join("")}</div>
    ${l.reformulated_query ? `<div class="notice info">Rewritten to: ${esc(l.reformulated_query)}</div>` : ""}
    <div style="font-family:var(--font-read);line-height:1.7">${esc(l.answer || "").replace(/\n/g, "<br>")}</div>
    <small class="muted">${fmt.date(l.created_at)}</small>`, { wide: true });
}
async function saveSession() {
  const name = await promptDialog("Save session", "Session name", `Session ${new Date().toISOString().slice(0, 10)}`); if (!name) return;
  try { await api("/sessions", { json: { name, query_ids: [...selected] } }); toast("Session saved", "ok"); selected.clear(); renderLog(); $("#log-save", root).disabled = true; loadSessions(); } catch (e) { fail(e); }
}

async function loadBookmarks() {
  bookmarks = (await api("/bookmarks")).bookmarks || [];
  $("#bm-list", root).innerHTML = bookmarks.length ? `<div class="stack">${bookmarks.map((b) => `<div class="sent"><b>${esc(b.query)}</b>
    <span class="small">${esc((b.answer || "").slice(0, 180))}${b.answer && b.answer.length > 180 ? "…" : ""}</span>
    ${b.note ? `<small>Note: ${esc(b.note)}</small>` : ""}
    <div class="row"><button class="btn sm" data-bm="open" data-id="${b.id}">Read</button><button class="btn sm" data-bm="note" data-id="${b.id}" data-note="${esc(b.note || "")}">${b.note ? "Edit note" : "Add note"}</button>
    <button class="btn sm danger" data-bm="del" data-id="${b.id}">Remove</button></div></div>`).join("")}</div>`
    : `<div class="empty small"><p>Use Save under an answer to keep it here.</p></div>`;
}
async function loadSessions() {
  const s = (await api("/sessions")).sessions || [];
  $("#ss-list", root).innerHTML = s.length ? `<div class="stack">${s.map((x) => `<button class="btn" style="justify-content:space-between" data-ss="${x.id}" data-name="${esc(x.name)}"><span>${esc(x.name)}</span><span class="muted small">${x.query_count} questions</span></button>`).join("")}</div>`
    : `<div class="empty small"><p>Select questions in the log and save them as a session.</p></div>`;
}
async function loadCalibration() {
  const c = await api("/api/research/calibration-live").catch(() => null);
  const el = $("#ins-cal", root);
  if (!c || c.error) { el.innerHTML = `<section class="card"><h2>Does the confidence score match what users think?</h2><p class="muted small">${esc((c && c.error) || "Rate some answers with the thumbs and a live calibration report appears here.")}</p></section>`; return; }
  el.innerHTML = calibrationHtml(c, null).replace("Calibration of the self-evaluation", "Live calibration against user ratings");
}
