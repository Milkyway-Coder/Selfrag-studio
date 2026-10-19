// ask.js — conversation view: sessions, streaming answers, sentence-level evidence, feedback, inspector.
import { openQuiz } from "./quiz.js";
import {
  $, $$, esc, fmt, icon, api, download, toast, fail, dialog, confirmDialog, promptDialog, on, debounce,
  markdown, markdownWithEvidence, renderMath, state, persist, refreshDocs, setActiveDoc, SAMPLE_DOC,
} from "./core.js";

let root, thread, inner, input, sendBtn, popEl;
const msgs = new Map();           // message id -> message model
let selectedId = null;
let abortCtl = null;
let llmHistory = [];              // [{role, content}] sent for multi-turn memory
let sessions = [];
let seq = 0;
let inspTab = "scores";
let bookmarks = [];
let ready = Promise.resolve();

const DEFAULT_NAME = /^(new chat|chat \d+|earlier chat)$/i;

/* ═════════════════════════════ mount ═════════════════════════════ */
export function mount(el) {
  root = el;
  root.innerHTML = `
  <div class="ask" id="ask">
    <aside class="sessions" id="sessions" aria-label="Conversations">
      <div class="sessions-head">
        <button class="btn primary" data-act="new-chat">${icon("plus")} New chat</button>
        <input class="inp" id="session-search" placeholder="Search chats" aria-label="Search chats">
      </div>
      <div class="session-list" id="session-list"></div>
    </aside>

    <section class="convo">
      <div class="scope">
        <button class="btn icon only-mobile" data-act="toggle-sessions" aria-label="Show chats">${icon("menu")}</button>
        <button class="btn icon ghost not-mobile" data-act="toggle-sessions-desk" aria-label="Show or hide chats" title="Show or hide chats">${icon("menu")}</button>

        <div class="pop-wrap">
          <button class="btn sm" data-pop="scope-pop" aria-haspopup="true" aria-expanded="false" title="Which documents to search">
            <span class="muted">Search in</span> <b id="scope-label"></b> ${icon("chevron")}</button>
          <div class="popover" id="scope-pop">
            <div class="seg" role="group" aria-label="Search scope" id="scope-seg">
              <button data-scope="active">One document</button>
              <button data-scope="all">All</button>
              <button data-scope="selected">Several</button>
            </div>
            <div id="scope-active" class="field"><label for="doc-select">Document</label><select class="inp" id="doc-select"></select></div>
            <div id="scope-selected" class="stack">
              <input class="inp" id="doc-filter" placeholder="Filter documents" aria-label="Filter documents">
              <div class="checklist" id="doc-checks"></div>
              <div class="row"><button class="btn sm ghost" data-act="sel-all">Select all</button><button class="btn sm ghost" data-act="sel-none">Clear</button></div>
            </div>
            <p class="small muted" id="scope-all-note" style="margin:0" hidden>Searches every document you have added and labels each source.</p>
          </div>
        </div>

        <div class="pop-wrap">
          <button class="btn sm" data-pop="opts-pop" aria-haspopup="true" aria-expanded="false">Options ${icon("chevron")}</button>
          <div class="popover" id="opts-pop">
            <label class="check" title="Fuse BM25 and dense ranking at chunk level (reciprocal rank fusion)">
              <input type="checkbox" id="opt-hybrid"><span>Hybrid retrieval<small>BM25 plus dense search, fused per chunk</small></span></label>
            <label class="check" title="Rewrite or prune sentences the verifier cannot support">
              <input type="checkbox" id="opt-repair"><span>Verify and repair<small>Fix sentences the evidence does not support</small></span></label>
            <div class="field"><label for="opt-thr">Confidence threshold: <b class="num" id="opt-thr-v"></b></label>
              <input type="range" id="opt-thr" min="1" max="5" step="0.5"><span class="small muted">Answers scoring below this are retried, then reported as not found.</span></div>
            <div class="field"><label for="opt-mem">Memory</label>
              <select class="inp" id="opt-mem">${[2, 4, 8, 12].map((n) => `<option value="${n}">${n} recent turns, older ones summarised</option>`).join("")}</select></div>
          </div>
        </div>

        <span class="muted small hide-narrow" id="mem-note"></span>
        <span class="right row">
          <button class="btn sm ghost" data-act="select-mode" id="sel-btn" title="Select questions to delete">${icon("check")}<span class="hide-narrow"> Select</span></button>
          <button class="btn sm ghost" data-act="export-chat" title="Export chat">${icon("download")}<span class="hide-narrow"> Export</span></button>
          <button class="btn sm ghost" data-act="clear-chat" title="Clear chat">${icon("trash")}<span class="hide-narrow"> Clear</span></button>
          <button class="btn sm ghost" data-act="toggle-inspector" aria-label="Toggle inspector" title="Show or hide the inspector">${icon("sidebar")}<span class="hide-narrow"> Inspector</span></button>
        </span>
      </div>

      <div class="thread" id="thread"><div class="thread-inner" id="thread-inner"></div></div>

      <div class="sel-bar" id="sel-bar" hidden>
        <b id="sel-count">0 selected</b>
        <button class="btn sm" data-act="sel-turns-all">Select all</button>
        <span class="right row"><button class="btn sm danger" data-act="del-selected" id="sel-del" disabled>${icon("trash")} Delete selected</button>
          <button class="btn sm" data-act="select-mode">Done</button></span>
      </div>
      <div class="composer">
        <div class="composer-inner">
          <div class="suggest" id="suggest" hidden></div>
          <div class="compose-box">
            <textarea id="q" rows="1" placeholder="Ask a question about your documents" aria-label="Question"></textarea>
            <button class="btn primary" id="send" aria-label="Send">${icon("send")}</button>
          </div>
          <div class="compose-hint"><span>Enter to send, Shift+Enter for a new line</span><span>Ctrl+K to focus</span>
            <span>Underlines: green is supported by the evidence, red is not</span></div>
        </div>
      </div>
    </section>

    <aside class="inspector" id="inspector" aria-label="Answer inspector">
      <div class="tabs" role="tablist" id="insp-tabs">
        <button role="tab" data-tab="scores" aria-selected="true">Scores</button>
        <button role="tab" data-tab="trace" aria-selected="false">Trace</button>
        <button role="tab" data-tab="evidence" aria-selected="false">Evidence</button>
      </div>
      <div class="inspector-body" id="insp-body"></div>
    </aside>
  </div>
  <div class="evpop" id="evpop" hidden></div>`;

  thread = $("#thread", root); inner = $("#thread-inner", root); input = $("#q", root);
  sendBtn = $("#send", root); popEl = $("#evpop", root);

  bindControls(); bindThread(); bindInspector(); bindSessions();
  document.addEventListener("docs:changed", renderScope);
  renderScope(); renderInspector();
  ready = init().catch(fail);
}

export function show() {
  renderScope();
  const pending = sessionStorage.getItem("sr.ask.pending");       // e.g. "Ask my documents" from the Chat view
  if (pending) { sessionStorage.removeItem("sr.ask.pending"); ready.then(() => send(pending)); }
}
export function focusComposer() { input && input.focus(); }

async function init() {
  const a = await api("/chat-sessions/active");
  state.sessionId = a.session_id;
  await Promise.all([loadSessions(), loadHistory(), loadBookmarks()]);
}

/* ═════════════════════════════ controls ═════════════════════════════ */
function bindControls() {
  const hy = $("#opt-hybrid", root), rp = $("#opt-repair", root), th = $("#opt-thr", root), thv = $("#opt-thr-v", root);
  hy.checked = state.hybrid; rp.checked = state.repair; th.value = state.threshold; thv.textContent = state.threshold;
  const mem = $("#opt-mem", root); mem.value = String(state.memory);
  mem.onchange = () => { state.memory = +mem.value; persist("sr.memory", state.memory); updateMemNote(); };
  hy.onchange = () => { state.hybrid = hy.checked; persist("sr.hybrid", state.hybrid); };
  rp.onchange = () => { state.repair = rp.checked; persist("sr.repair", state.repair); };
  th.oninput = () => { state.threshold = +th.value; thv.textContent = th.value; persist("sr.threshold", state.threshold); };

  on($("#scope-seg", root), "click", "[data-scope]", (e, b) => {
    state.scope = b.dataset.scope; persist("sr.scope", state.scope); renderScope();
  });
  $("#doc-select", root).onchange = (e) => setActiveDoc(e.target.value).catch(fail);
  on($("#doc-checks", root), "change", "input", (e, i) => {
    i.checked ? state.selected.add(i.value) : state.selected.delete(i.value);
    persist("sr.selected", [...state.selected]);
    $("#scope-label", root).textContent = state.selected.size ? `${state.selected.size} of ${state.docs.length} selected` : "Choose documents";
  });

  on(root, "click", "[data-act]", (e, b) => {
    const a = b.dataset.act;
    if (a === "new-chat") newChat();
    else if (a === "clear-chat") clearChat();
    else if (a === "export-chat") download(`/export-chat-html?session_id=${encodeURIComponent(state.sessionId)}`, {}, "chat.html").catch(fail);
    else if (a === "toggle-inspector") toggleInspector();
    else if (a === "toggle-sessions") $("#sessions", root).classList.toggle("open");
    else if (a === "toggle-sessions-desk") { const c = $("#ask", root).classList.toggle("no-sessions"); persist("sr.sessions.closed", c); }
    else if (a === "select-mode") toggleSelect();
    else if (a === "sel-turns-all") { $$(".turn", inner).forEach((t) => pick(t, true)); updateSel(); }
    else if (a === "del-selected") deleteTurns($$('.turn[data-picked="true"]', inner));
    else if (a === "sel-all") { state.selected = new Set(state.docs); persist("sr.selected", [...state.selected]); renderScope(); }
    else if (a === "sel-none") { state.selected = new Set(); persist("sr.selected", []); renderScope(); }
  });
  bindPopovers();
  $("#doc-filter", root).oninput = () => renderScope();
  if (localStorage.getItem("sr.sessions.closed") === "true") $("#ask", root).classList.add("no-sessions");
  const insPref = localStorage.getItem("sr.inspector.closed");
  if (insPref === "true" || (insPref === null && window.innerWidth < 1500)) $("#ask", root).classList.add("no-inspector");

  sendBtn.onclick = () => (abortCtl ? abortCtl.abort() : send());
  input.addEventListener("input", () => { autosize(); suggest(); });
  input.addEventListener("keydown", onKey);
  document.addEventListener("click", (e) => { if (!e.target.closest(".composer-inner")) hideSuggest(); });
}

function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; }

function toggleInspector() {
  const ask = $("#ask", root), insp = $("#inspector", root);
  if (window.matchMedia("(max-width: 1180px)").matches) insp.classList.toggle("open");
  else { const c = ask.classList.toggle("no-inspector"); persist("sr.inspector.closed", c); }
}

function bindPopovers() {
  const closeAll = () => { $$(".popover", root).forEach((p) => p.classList.remove("open")); $$("[data-pop]", root).forEach((b) => b.setAttribute("aria-expanded", "false")); };
  on(root, "click", "[data-pop]", (e, b) => {
    const pop = $("#" + b.dataset.pop, root), open = !pop.classList.contains("open"); closeAll(); pop.classList.toggle("open", open); b.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (e) => { if (root && !e.target.closest(".pop-wrap")) closeAll(); });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !root) return;
    closeAll(); if (thread && thread.classList.contains("selecting")) toggleSelect(false);
  });
}

function renderScope() {
  if (!root) return;
  $$("#scope-seg button", root).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.scope === state.scope)));
  const n = state.docs.length;
  $("#scope-label", root).textContent = state.scope === "all" ? `All documents (${n})`
    : state.scope === "selected" ? (state.selected.size ? `${state.selected.size} of ${n} selected` : "Choose documents")
    : (state.activeDoc || (n ? "Choose a document" : "No documents"));
  $("#scope-active", root).hidden = state.scope !== "active";
  $("#scope-selected", root).hidden = state.scope !== "selected";
  $("#scope-all-note", root).hidden = state.scope !== "all";
  const sel = $("#doc-select", root);
  sel.innerHTML = n
    ? `<option value="">${state.activeDoc ? "" : "Choose a document…"}</option>` +
      state.docs.map((d) => `<option value="${esc(d)}" ${d === state.activeDoc ? "selected" : ""}>${esc(d)}</option>`).join("")
    : `<option value="">No documents yet</option>`;
  if (state.activeDoc) sel.value = state.activeDoc;
  const f = ($("#doc-filter", root).value || "").trim().toLowerCase();
  const list = state.docs.filter((d) => !f || d.toLowerCase().includes(f));
  $("#doc-checks", root).innerHTML = list.map((d) =>
    `<label class="check"><input type="checkbox" value="${esc(d)}" ${state.selected.has(d) ? "checked" : ""}><span>${esc(d)}${state.docInfo[d] ? `<small>${esc(state.docInfo[d].kind)}</small>` : ""}</span></label>`).join("")
    || `<span class="muted small">${n ? "No match." : "Add documents in the Library first."}</span>`;
  renderStarter();
}

/* ═════════════════════════════ starter ═════════════════════════════ */
function renderStarter() {
  if (!inner) return;
  let st = $("#starter", inner);
  if ([...inner.children].some((c) => c.id !== "starter")) { st && st.remove(); return; }
  if (!st) { st = document.createElement("div"); st.id = "starter"; st.className = "starter"; inner.prepend(st); }
  if (!state.docs.length) {
    st.innerHTML = `<h2>Ask your documents</h2>
      <p class="muted">Add a PDF, a web page or pasted text, then ask questions. Every answer is scored and checked sentence by sentence against the evidence it was built from.</p>
      <div class="row"><a class="btn primary" href="#/library">Add a document</a>
      <button class="btn" data-starter="sample">Load a sample document</button></div>`;
  } else {
    const name = state.scope === "all" ? "all your documents" : state.scope === "selected" ? "the selected documents" : (state.activeDoc || "your documents");
    const info = state.scope === "active" ? state.docInfo[state.activeDoc] : null, kind = info ? info.kind : "text", sx = (info && info.stats) || {};
    const prompts = {
      sheet: ["Summarize the columns and what this data contains", "What is the average of each numeric column?", "Which rows have the highest values?"],
      code: ["Explain what this code does", "List the functions and classes and what each does", "Find possible bugs or edge cases", "What is the time complexity of the main functions?"],
    }[kind === "project" || kind === "notebook" ? "code" : kind] || ["Summarize this document", "What are the key concepts?", "List the main topics covered"];
    if (["pdf", "docx", "pptx"].includes(kind)) {
      if (sx.figures) prompts.push("Which figures are in this document and what do they show?");
      if (sx.tables) prompts.push("Summarize the tables");
    }
    st.innerHTML = `<h2>Ask about ${esc(name)}</h2>
      <div class="prompts">${prompts.slice(0, 5).map((p) => `<button class="btn" data-prompt="${esc(p)}">${esc(p)}</button>`).join("")}</div>
      ${state.activeDoc && state.scope === "active" ? `<div class="row wrap" style="justify-content:center"><button class="btn primary sm" data-starter="quiz">${icon("check")} Quiz me on this document</button><button class="btn ghost sm" data-starter="summary">Show the summary</button></div><div id="starter-sum" class="muted"></div>` : ""}`;
  }
}

/* ═════════════════════════════ thread events ═════════════════════════════ */
function bindThread() {
  on(inner, "click", "[data-prompt]", (e, b) => { input.value = b.dataset.prompt; autosize(); send(); });
  on(inner, "click", "[data-starter]", async (e, b) => {
    try {
      if (b.dataset.starter === "sample") {
        b.disabled = true; b.textContent = "Indexing…";
        await api("/index-doc", { json: { doc_id: SAMPLE_DOC.id, text: SAMPLE_DOC.text } });
        await refreshDocs(); toast("Sample document indexed", "ok");
      } else if (b.dataset.starter === "quiz") {
        openQuiz(state.activeDoc);
      } else if (b.dataset.starter === "summary") {
        const box = $("#starter-sum", inner); box.innerHTML = `<span class="spinner"></span> Loading…`;
        let e = await api(`/api/docs/${encodeURIComponent(state.activeDoc)}/enrichment`);
        if (!e.ready) e = await api(`/api/docs/${encodeURIComponent(state.activeDoc)}/enrich`, { method: "POST" });     // not created yet: make it now
        box.textContent = e.summary;
      }
    } catch (err) { fail(err); }
  });

  inner.addEventListener("click", (e) => {                          // select mode: a click anywhere on a turn picks it
    if (!thread.classList.contains("selecting")) return;
    const t = e.target.closest(".turn"); if (!t) return;
    if (e.target.closest(".turn-check")) { pick(t, e.target.checked); updateSel(); return; }
    pick(t); updateSel();
  });
  on(inner, "click", "[data-act]", (e, b) => {
    const card = b.closest("[data-msg]"); const m = card && msgs.get(card.dataset.msg);
    const act = b.dataset.act;
    if (act === "del-turn") { deleteTurns([b.closest(".turn")]); return; }
    if (act === "select") return;
    if (!m && !["general", "search-all", "to-chat"].includes(act)) return;
    ({
      copy: () => copyAnswer(m, b),
      regen: () => { if (m.done) send(m.query, m); },
      "v-prev": () => applyVersion(m, m.vi - 1),
      "v-next": () => applyVersion(m, m.vi + 1),
      up: () => rate(m, 1),
      down: () => openFeedbackForm(m),
      "fb-send": () => submitFeedback(m, -1, $("textarea", card).value.trim()),
      "fb-skip": () => submitFeedback(m, -1, ""),
      bookmark: () => toggleBookmark(m),
      evidence: () => { setInspTab("evidence"); selectMsg(m.id); openInspector(); loadSources(m); },
      pdf: () => exportPdf(m),
      marks: () => { m.marks = !m.marks; paintBody(m); },
      inspect: () => { selectMsg(m.id); openInspector(); },
      fig: () => openFigure(m, +b.dataset.i),
      src: () => openSource(m, +b.dataset.i),
      "to-chat": () => { sessionStorage.setItem("sr.chat.pending", b.dataset.q); location.hash = "#/chat"; },
      general: () => answerGeneral(b.dataset.q, b),
      "search-all": () => { state.scope = "all"; persist("sr.scope", "all"); renderScope(); send(b.dataset.q); },
    }[act] || (() => {}))();
  });

  inner.addEventListener("click", (e) => {
    const card = e.target.closest("[data-msg]");
    if (card && !e.target.closest("button, a, textarea") && msgs.get(card.dataset.msg)?.kind === "answer") selectMsg(card.dataset.msg);
  });

  // evidence popover
  const showPop = (e) => {
    const span = e.target.closest(".ev"); if (!span) return;
    const m = msgs.get(span.closest("[data-msg]").dataset.msg);
    const s = m && m.hallucination && m.hallucination.sentences[+span.dataset.ev]; if (!s) return;
    const r = span.getBoundingClientRect();
    popEl.innerHTML = `<b style="color:var(--${s.supported ? "ok" : "bad"})">${s.supported ? "Supported by the evidence" : "Not supported by the evidence"}</b>` +
      (s.similarity !== undefined ? `Similarity ${fmt.n(s.similarity, 2)}${s.method ? ` · decided by ${esc(s.method)}` : ""}` : "") +
      (s.evidence ? `<i>${esc(s.evidence.slice(0, 220))}${s.evidence.length > 220 ? "…" : ""}</i>` : "");
    popEl.hidden = false;
    const w = popEl.offsetWidth, h = popEl.offsetHeight;
    popEl.style.left = Math.max(8, Math.min(r.left, innerWidth - w - 8)) + "px";
    popEl.style.top = (r.top - h - 8 > 8 ? r.top - h - 8 : r.bottom + 8) + "px";
  };
  const hidePop = (e) => { if (e.target.closest(".ev")) popEl.hidden = true; };
  inner.addEventListener("mouseover", showPop); inner.addEventListener("mouseout", hidePop);
  inner.addEventListener("focusin", showPop); inner.addEventListener("focusout", hidePop);
}

/* ═════════════════════════════ rendering ═════════════════════════════ */
function nearBottom() { return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 140; }
function toBottom() { thread.scrollTop = thread.scrollHeight; }

const lastTurn = () => { const t = inner.lastElementChild; return t && t.classList.contains("turn") ? t : null; };

function addUser(text, uid) {
  $("#starter", inner)?.remove();
  const turn = document.createElement("div"); turn.className = "turn"; if (uid) turn.dataset.uid = uid;
  const u = document.createElement("div"); u.className = "msg-user"; u.textContent = text;
  turn.innerHTML = `<input type="checkbox" class="turn-check" aria-label="Select this question" tabindex="-1">
    <div class="turn-tools"><button class="btn sm ghost icon" data-act="del-turn" aria-label="Delete this question and its answer" title="Delete this question and its answer">${icon("trash")}</button></div>`;
  turn.append(u); inner.append(turn); return turn;
}
function addNote(text) {
  $("#starter", inner)?.remove();
  const d = document.createElement("div"); d.className = "msg-note"; d.textContent = text; (lastTurn() || inner).append(d); return d;
}

function newModel(query) {
  const id = `m${++seq}`;
  const m = { id, kind: "answer", query, raw: "", trace: [], scores: null, hallucination: null, cited: [], attempts: 1,
    reformulated: null, repaired: null, queryId: null, rating: 0, bookmarkId: null, marks: true, done: false, sources: null, step: "",
    srcs: [], figs: [], followUp: null, versions: [], vi: 0, regen: false, figureOnly: false };
  msgs.set(id, m); return m;
}

function cardEl(m) {
  const el = document.createElement("article");
  el.className = "answer"; el.dataset.msg = m.id; el.id = `a-${m.id}`;
  (lastTurn() || inner).append(el); return el;
}
const $card = (m) => document.getElementById(`a-${m.id}`);

function chipsFor(m) {
  const out = [];
  if (m.fallback) return `<span class="chip warn">General knowledge</span>`;
  if (m.figureOnly) return `<span class="chip brand">Figure lookup</span>`;
  const s = m.scores;
  if (s) {
    const a = s.average_score;
    out.push(a >= 4.5 ? `<span class="chip ok">High confidence ${fmt.n(a, 1)}/5</span>`
      : a >= 3.5 ? `<span class="chip brand">Moderate confidence ${fmt.n(a, 1)}/5</span>`
      : `<span class="chip warn">Low confidence ${fmt.n(a, 1)}/5, verify</span>`);
  }
  const h = m.hallucination;
  if (h && h.total_count) {
    const p = Math.round((h.hallucination_score || 0) * 100);
    out.push(p === 0 ? `<span class="chip ok">All ${h.total_count} sentences supported</span>`
      : `<span class="chip bad">${h.unsupported_count} of ${h.total_count} sentences unsupported</span>`);
  }
  if (m.repaired) out.push(`<span class="chip brand">Repaired (${esc(m.repaired.method)}): ${Math.round(m.repaired.before * 100)}% → ${Math.round(m.repaired.after * 100)}% flagged</span>`);
  if (m.attempts > 1) out.push(`<span class="chip">${m.attempts} attempts</span>`);
  if (m.followUp) out.push(`<span class="chip" title="Your follow-up was rewritten for searching: ${esc(m.followUp)}">Read as: ${esc(m.followUp.slice(0, 48))}${m.followUp.length > 48 ? "…" : ""}</span>`);
  if (m.reformulated) out.push(`<span class="chip" title="${esc(m.reformulated)}">Question rewritten</span>`);
  (m.cited || []).forEach((c) => out.push(`<span class="chip brand">${esc(c)}</span>`));
  return out.join("");
}

const srcName = (s) => [s.doc_id, s.label || (s.page ? `page ${s.page}` : "")].filter(Boolean).join(" · ");
const chipName = (s, multi) => (multi ? `${s.doc_id} · ` : "") + (s.chip || s.label || (s.page ? `p.${s.page}` : "Passage"));

function extraHtml(m) {
  let h = "";
  if (m.figs && m.figs.length) {
    h += `<div class="figs${m.figureOnly ? " big" : ""}" aria-label="Related figures">${m.figs.map((f, i) => `
      <button class="fig-card" data-act="fig" data-i="${i}" title="Shown because: ${esc(f.reason)}">
        <img loading="lazy" src="${esc(f.url)}" alt="${esc(f.caption || "Figure")}">
        <figcaption><b>${f.label ? "Figure " + esc(f.label) : "Figure"}</b>${f.page ? " · page " + f.page : ""}${f.caption && !/^Figure on page/i.test(f.caption) ? "<br>" + esc(f.caption.slice(0, 90)) : ""}</figcaption></button>`).join("")}</div>`;
  }
  if (m.srcs && m.srcs.length) {
    const multi = new Set(m.srcs.map((s) => s.doc_id)).size > 1;
    h += `<div class="sources"><span class="lbl">Sources</span>${m.srcs.map((s, i) =>
      `<button class="src" data-act="src" data-i="${i}" title="${esc(s.snippet)}">${esc(chipName(s, multi))}</button>`).join("")}</div>`;
  }
  return h;
}

function openFigure(m, i) {
  const f = m.figs[i]; if (!f) return;
  dialog(f.label ? `Figure ${f.label}` : "Figure", `
    <img class="lightbox-img" src="${esc(f.url)}" alt="${esc(f.caption || "Figure")}">
    <p>${esc(f.caption || "")}</p>
    <div class="row wrap"><span class="chip">${esc(f.doc_id || "")}</span>${f.page ? `<span class="chip">page ${f.page}</span>` : ""}<span class="chip brand">Shown because: ${esc(f.reason)}</span></div>`, { wide: true });
}

function openSource(m, i) {
  const s = m.srcs[i]; if (!s) return;
  dialog(`Passage ${s.n}: ${srcName(s)}`, `
    <div class="row wrap"><span class="chip">${esc(s.type)}</span>${s.page ? `<span class="chip">page ${s.page}</span>` : ""}${s.file ? `<span class="chip">${esc(s.file)}</span>` : ""}${s.lines ? `<span class="chip">lines ${esc(s.lines)}</span>` : ""}</div>
    <div class="snippet">${esc(s.snippet)}${s.snippet.length >= 240 ? "…" : ""}</div>
    <p class="small muted" style="margin:0">This is the start of the passage the answer was built from.</p>`);
}

function barHtml(m) {
  const bm = m.bookmarkId ? "aria-pressed=true" : "";
  const vers = m.versions.length > 1 ? `<span class="vers" role="group" aria-label="Answer versions">
      <button class="btn sm icon ghost" data-act="v-prev" ${m.vi === 0 ? "disabled" : ""} aria-label="Previous version">‹</button>
      <span class="num small">${m.vi + 1}/${m.versions.length}</span>
      <button class="btn sm icon ghost" data-act="v-next" ${m.vi === m.versions.length - 1 ? "disabled" : ""} aria-label="Next version">›</button></span>` : "";
  const hasMarks = m.hallucination && m.hallucination.sentences && m.hallucination.sentences.length;
  return `${vers}
    <button class="btn sm ghost" data-act="copy">${icon("copy")} Copy</button>
    <button class="btn sm ghost" data-act="regen">${icon("refresh")} Regenerate</button>
    <button class="btn sm ghost" data-act="up" ${m.rating ? "disabled" : ""} aria-label="Helpful" ${m.rating === 1 ? 'style="color:var(--ok)"' : ""}>${icon("up")}</button>
    <button class="btn sm ghost" data-act="down" ${m.rating ? "disabled" : ""} aria-label="Not helpful" ${m.rating === -1 ? 'style="color:var(--bad)"' : ""}>${icon("down")}</button>
    <button class="btn sm ghost" data-act="bookmark" ${bm} style="${m.bookmarkId ? "color:var(--brand)" : ""}">${icon("bookmark")} ${m.bookmarkId ? "Saved" : "Save"}</button>
    ${m.figureOnly ? "" : `<button class="btn sm ghost" data-act="evidence">${icon("search")} Trace sources</button>`}
    ${hasMarks && !m.figureOnly ? `<button class="btn sm ghost" data-act="marks">${m.marks ? "Hide" : "Show"} underlines</button>` : ""}
    <button class="btn sm ghost" data-act="pdf">${icon("download")} PDF</button>
    <button class="btn sm ghost only-mobile" data-act="inspect">${icon("info")} Details</button>`;
}

function paintBody(m) {
  const el = $card(m); if (!el) return;
  const body = $("[data-body]", el); if (!body) return;
  const marked = m.done && m.marks && m.hallucination && m.hallucination.sentences && m.hallucination.sentences.length;
  body.innerHTML = marked ? markdownWithEvidence(m.raw, m.hallucination.sentences) : markdown(m.raw);
  body.classList.toggle("cursor", !m.done);
  if (m.done) renderMath(body);
  const bar = $("[data-bar]", el); if (bar && m.done) bar.innerHTML = barHtml(m);
}

function paintCard(m) {
  const el = $card(m); if (!el) return;
  el.classList.toggle("selected", selectedId === m.id);
  el.classList.toggle("fallback", !!m.fallback);
  el.innerHTML = `
    ${m.fallback ? `<div class="answer-tag">General knowledge. This answer did not come from your documents.</div>` : ""}
    <div class="answer-body" data-body></div>
    <div class="live" data-live ${m.done ? "hidden" : ""}><span class="spinner"></span><span data-step>${esc(m.step || "Retrieving evidence…")}</span></div>
    <div class="answer-meta" data-meta>${m.done ? chipsFor(m) : ""}</div>
    <div data-extra>${m.done ? extraHtml(m) : ""}</div>
    <div data-fb></div>
    <div class="answer-bar" data-bar ${m.done ? "" : "hidden"}>${m.done ? barHtml(m) : ""}</div>`;
  paintBody(m);
}

let raf = 0;
function schedulePaint(m) {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0; const stick = nearBottom();
    const el = $card(m); if (!el) return;
    const body = $("[data-body]", el); if (body) { body.innerHTML = markdown(m.raw); body.classList.add("cursor"); }
    if (stick) toBottom();
  });
}

/* ═════════════════════════════ sending ═════════════════════════════ */
async function* sse(resp) {
  const reader = resp.body.getReader(), dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      const line = block.split("\n").find((l) => l.startsWith("data: "));
      if (line) { try { yield JSON.parse(line.slice(6)); } catch { /* ignore a malformed frame */ } }
    }
  }
}

/* Answer versions: regenerating keeps the earlier answer so you can step between them (1/2, 2/2). */
const SNAP = ["figureOnly", "raw", "scores", "hallucination", "cited", "attempts", "reformulated", "repaired", "queryId", "srcs", "figs", "followUp", "trace", "rating", "bookmarkId", "sources", "marks"];
const clone = (v) => (v === undefined ? v : JSON.parse(JSON.stringify(v)));
const snapshot = (m) => Object.fromEntries(SNAP.map((k) => [k, clone(m[k])]));
function applyVersion(m, i) {
  if (!m.versions[i]) return;
  Object.assign(m, clone(m.versions[i])); m.vi = i; m.done = true; m.error = null; paintCard(m);
  if (selectedId === m.id) renderInspector();
}
function resetForRegen(m) {
  Object.assign(m, { raw: "", scores: null, hallucination: null, cited: [], attempts: 1, reformulated: null, repaired: null, queryId: null,
    srcs: [], figs: [], followUp: null, trace: [], rating: 0, bookmarkId: null, sources: null, done: false, error: null, step: "", marks: true });
}
function fillFromRow(m, row) {
  const meta = row.metadata || {};
  Object.assign(m, { raw: row.content, scores: meta.scores || null, hallucination: meta.hallucination || null, cited: meta.cited_sources || [],
    attempts: meta.attempts || 1, reformulated: meta.reformulated_query || null, queryId: meta.query_id || null, done: true, repaired: null,
    srcs: meta.sources || [], figs: meta.figures || [], followUp: meta.follow_up || null, figureOnly: !!meta.figure_only, trace: [], rating: 0, sources: null });
  const bm = bookmarks.find((b) => b.query === m.query && b.answer === m.raw); m.bookmarkId = bm ? bm.id : null;
}

async function send(forcedQuery, regenOf) {
  if (abortCtl) return;
  const query = (forcedQuery ?? input.value).trim();
  if (!query) return;
  if (state.scope === "selected" && state.selected.size === 0) { toast("Tick at least one document in the scope bar.", "bad"); return; }

  let m, histForRequest = llmHistory.slice(-60), prevRaw = null, turnEl = null;
  if (regenOf) {                                  // a new VERSION of the same answer, in the same card
    m = regenOf; prevRaw = m.raw;
    if (!m.versions.length) m.versions = [snapshot(m)];
    const at = (() => { for (let i = llmHistory.length - 1; i >= 0; i--) if (llmHistory[i].role === "assistant" && llmHistory[i].content === prevRaw) return i; return -1; })();
    if (at > 0) histForRequest = llmHistory.slice(Math.max(0, at - 1 - 60), at - 1);   // the model must not see its own earlier answer to this question
    resetForRegen(m); m.regen = true; turnEl = $card(m)?.closest(".turn") || null;
  } else {
    turnEl = addUser(query); m = newModel(query); cardEl(m);
    input.value = ""; autosize(); hideSuggest();
  }
  selectedId = m.id; paintCard(m); renderInspector(); toBottom();

  abortCtl = new AbortController();
  sendBtn.innerHTML = icon("stop"); sendBtn.setAttribute("aria-label", "Stop");
  const body = {
    query, history: histForRequest, memory_turns: state.memory, regenerate: !!regenOf, multi_doc: state.scope === "all",
    selected_docs: state.scope === "selected" ? [...state.selected] : null,
    threshold: state.threshold, hybrid: state.hybrid, repair: state.repair,
  };
  let finished = false;
  try {
    const resp = await fetch(regenOf ? "/regenerate" : "/chat-stream", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: abortCtl.signal,
    });
    if (!resp.ok) { let msg = `Request failed (${resp.status})`; try { msg = (await resp.json()).error || msg; } catch {} throw new Error(msg); }
    for await (const ev of sse(resp)) {
      if (ev.type === "trace_step") {
        m.trace.push({ label: ev.label, detail: ev.detail }); m.step = ev.label;
        const s = $("[data-step]", $card(m) || document); if (s) s.textContent = ev.label + (ev.detail && ev.label.startsWith("Retrieval") ? `: ${ev.detail}` : "");
        if (selectedId === m.id && inspTab === "trace") renderInspector();
      } else if (ev.type === "token") { m.raw += ev.content; schedulePaint(m); }
      else if (ev.type === "reset") { m.raw = ""; m.step = "Retrying with a rewritten question…"; schedulePaint(m); }
      else if (ev.type === "reformulated") m.reformulated = ev.query;
      else if (ev.type === "repaired") { m.raw = ev.answer; m.repaired = { method: ev.method, before: ev.before, after: ev.after }; schedulePaint(m); }
      else if (ev.type === "not_found") {
        finished = true; if (turnEl && ev.user_msg_id) turnEl.dataset.uid = ev.user_msg_id;
        if (regenOf) { m.regen = false; applyVersion(m, m.versions.length - 1); toast("Nothing was found this time, so the previous answer was kept."); }
        else renderNotFound(m, ev);
      }
      else if (ev.type === "error") throw new Error(ev.message || "The server reported an error.");
      else if (ev.type === "done") { finished = true; if (turnEl && ev.user_msg_id) turnEl.dataset.uid = ev.user_msg_id; finalize(m, ev); }
    }
    if (!finished) throw new Error("The connection closed before the answer finished.");
  } catch (err) {
    const keepPrev = m.regen && !m.raw;
    if (keepPrev) { m.regen = false; applyVersion(m, m.versions.length - 1); }
    if (err.name === "AbortError") {
      if (!keepPrev) { m.done = true; m.raw = m.raw || "(stopped)"; m.stopped = true; if (m.regen) { m.versions.push(snapshot(m)); m.vi = m.versions.length - 1; m.regen = false; } paintCard(m); }
      toast("Stopped");
    } else {
      if (!keepPrev) {
        m.done = true; m.error = err.message; m.raw = m.raw || ""; paintCard(m);
        const meta = $("[data-meta]", $card(m)); if (meta) meta.innerHTML = `<div class="notice bad" style="width:100%">${esc(err.message)}</div>`;
        const bar = $("[data-bar]", $card(m)); if (bar) { bar.hidden = false; bar.innerHTML = `<button class="btn sm" data-act="regen">${icon("refresh")} Try again</button>`; }
      }
      fail(err);
    }
  } finally {
    abortCtl = null; sendBtn.innerHTML = icon("send"); sendBtn.setAttribute("aria-label", "Send");
    renderInspector(); toBottom(); loadSessions().catch(() => {});
  }
}

function finalize(m, ev) {
  if (ev.is_greeting) {                           // small talk is not an answer
    const el = $card(m); const note = document.createElement("div");
    note.className = "msg-note"; note.textContent = m.raw.trim(); el.replaceWith(note); msgs.delete(m.id);
    llmHistory.push({ role: "user", content: m.query }); llmHistory.push({ role: "assistant", content: m.raw.trim() });
    return;
  }
  if (ev.answer) m.raw = ev.answer;
  m.scores = ev.scores; m.hallucination = ev.hallucination; m.cited = ev.cited_sources || [];
  m.attempts = ev.attempts; m.reformulated = ev.reformulated_query || m.reformulated; m.queryId = ev.query_id; m.done = true;
  m.srcs = ev.sources || []; m.figs = ev.figures || []; m.followUp = ev.follow_up || null; m.figureOnly = !!ev.figure_only;
  if (m.regen) {                                  // store this answer as the newest version and show "n/n"
    const old = m.versions.length ? m.versions[m.versions.length - 1].raw : null;
    m.versions.push(snapshot(m)); m.vi = m.versions.length - 1; m.regen = false;
    const at = (() => { for (let i = llmHistory.length - 1; i >= 0; i--) if (llmHistory[i].role === "assistant" && llmHistory[i].content === old) return i; return -1; })();
    if (at >= 0) llmHistory[at].content = m.raw; else llmHistory.push({ role: "user", content: m.query }, { role: "assistant", content: m.raw });
  } else {
    llmHistory.push({ role: "user", content: m.query }); llmHistory.push({ role: "assistant", content: m.raw });
  }
  paintCard(m); updateMemNote(); autoName(m.query);
}

function renderNotFound(m, ev) {
  const el = $card(m);
  const partial = (ev.partial_matches || []).slice(0, 3);
  el.className = "notfound"; el.removeAttribute("id");
  el.innerHTML = `
    <div><h3>Not found in your documents</h3>
    <p class="muted" style="margin:4px 0 0">${esc(ev.suggestion || "The documents do not appear to cover this.")}</p></div>
    ${partial.length ? `<div class="stack"><span class="lbl">Closest passages</span>${partial.map((p) => `<div class="snippet">${esc(p)}…</div>`).join("")}</div>` : ""}
    ${(ev.indexed_docs || []).length ? `<div class="row wrap"><span class="muted small">Searched:</span>${ev.indexed_docs.slice(0, 8).map((d) => `<span class="chip">${esc(d)}</span>`).join("")}</div>` : ""}
    <div class="row wrap">
      ${state.scope !== "all" && state.docs.length > 1 ? `<button class="btn sm" data-act="search-all" data-q="${esc(m.query)}">Search all documents</button>` : ""}
      <button class="btn sm" data-act="general" data-q="${esc(m.query)}">Answer from general knowledge</button>
      <button class="btn sm" data-act="to-chat" data-q="${esc(m.query)}">Continue in Chat</button>
      <a class="btn sm ghost" href="#/library">Add a document</a>
    </div>`;
  msgs.delete(m.id);
}

async function answerGeneral(q, btn) {
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Thinking…`;
  try {
    const r = await api("/answer-general", { json: { query: q } });
    const m = newModel(q); m.fallback = true; m.raw = r.answer; m.done = true; cardEl(m); paintCard(m);
    $("[data-bar]", $card(m)).innerHTML = `<button class="btn sm ghost" data-act="copy">${icon("copy")} Copy</button>`;
    toBottom();
  } catch (e) { fail(e); }
  finally { btn.disabled = false; btn.textContent = "Answer from general knowledge"; }
}

/* ═════════════════════════════ actions ═════════════════════════════ */
function copyAnswer(m, btn) {
  navigator.clipboard.writeText(m.raw).then(() => { btn.innerHTML = `${icon("check")} Copied`; setTimeout(() => (btn.innerHTML = `${icon("copy")} Copy`), 1500); },
    () => toast("Copy was blocked by the browser", "bad"));
}

async function rate(m, rating, comment = "") {
  try {
    await api("/feedback", { json: { query: m.query, answer: m.raw, rating, query_id: m.queryId, comment: comment || null } });
    m.rating = rating; $("[data-fb]", $card(m)).innerHTML = ""; $("[data-bar]", $card(m)).innerHTML = barHtml(m);
    toast(rating === 1 ? "Thanks, marked helpful" : "Thanks, feedback saved", "ok");
  } catch (e) { fail(e); }
}
const submitFeedback = (m, r, c) => rate(m, r, c);
function openFeedbackForm(m) {
  $("[data-fb]", $card(m)).innerHTML = `<div class="fb-form"><label class="lbl" for="fb-${m.id}">What was wrong? (optional)</label>
    <textarea id="fb-${m.id}" class="inp" rows="2" style="min-height:0"></textarea>
    <div class="row"><button class="btn sm primary" data-act="fb-send">Send feedback</button><button class="btn sm ghost" data-act="fb-skip">Skip</button></div></div>`;
  $(`#fb-${m.id}`).focus();
}

async function loadBookmarks() { try { bookmarks = (await api("/bookmarks")).bookmarks || []; } catch { bookmarks = []; } }
async function toggleBookmark(m) {
  try {
    if (m.bookmarkId) { await api(`/bookmark/${m.bookmarkId}`, { method: "DELETE" }); m.bookmarkId = null; toast("Removed from saved answers"); }
    else {
      const r = await api("/bookmark", { json: { query: m.query, answer: m.raw, doc_id: state.activeDoc, avg_score: m.scores ? m.scores.average_score : null } });
      m.bookmarkId = r.id; toast("Saved. Find it under Insights", "ok");
    }
    $("[data-bar]", $card(m)).innerHTML = barHtml(m);
  } catch (e) { fail(e); }
}

async function exportPdf(m) {
  try {
    await download("/export-answer", { json: { query: m.query, answer: m.raw, scores: m.scores,
      hallucination_pct: m.hallucination ? Math.round((m.hallucination.hallucination_score || 0) * 100) : 0,
      reformulated_query: m.reformulated, cited_sources: m.cited } }, "answer.pdf");
  } catch (e) { fail(e); }
}

/* ═════════════════════════════ inspector ═════════════════════════════ */
function bindInspector() {
  on($("#insp-tabs", root), "click", "[data-tab]", (e, b) => setInspTab(b.dataset.tab));
  on($("#insp-body", root), "click", "[data-load-sources]", () => { const m = msgs.get(selectedId); m && loadSources(m); });
}
function openInspector() { if (window.matchMedia("(max-width: 1180px)").matches) $("#inspector", root).classList.add("open"); else $("#ask", root).classList.remove("no-inspector"); }
function setInspTab(t) {
  inspTab = t; $$("#insp-tabs button", root).forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t))); renderInspector();
}
function selectMsg(id) {
  selectedId = id; $$(".answer", inner).forEach((a) => a.classList.toggle("selected", a.dataset.msg === id)); renderInspector();
}

const barRow = (label, v, max = 5) => {
  const cls = v / max >= 0.8 ? "ok" : v / max >= 0.6 ? "" : v / max >= 0.4 ? "warn" : "bad";
  return `<div class="bar-row"><span>${label}</span><div class="bar ${cls}"><i style="width:${Math.max(0, Math.min(100, (v / max) * 100))}%"></i></div><b class="num">${fmt.n(v, 1)}</b></div>`;
};

function renderInspector() {
  const body = $("#insp-body", root); if (!body) return;
  const m = msgs.get(selectedId);
  if (!m) { body.innerHTML = `<div class="empty"><h3>Nothing selected</h3><p>Ask a question, or click an answer to see how it was produced.</p></div>`; return; }
  if (inspTab === "scores") {
    const s = m.scores, h = m.hallucination;
    body.innerHTML = m.figureOnly ? `<div class="notice info">This was a figure lookup. The picture was found by matching your request to figure descriptions, so there is no generated answer to score or fact-check.</div>
      ${m.figs.map((f) => `<div class="sent"><b>${f.label ? "Figure " + esc(f.label) : "Figure"} · page ${f.page || "?"}</b><span class="small">${esc(f.caption)}</span><small>Shown because: ${esc(f.reason)}</small></div>`).join("")}`
      : !s ? `<div class="empty"><span class="spinner"></span><p>Scores appear when the answer finishes.</p></div>` : `
      <div class="stack">
        ${barRow("Retrieval", s.retrieval_relevance)}${barRow("Grounding", s.answer_grounding)}${barRow("Quality", s.answer_quality)}${barRow("Average", s.average_score)}
      </div>
      <p class="small muted" style="margin:0">${esc(s.reasoning || "")}</p>
      <div class="kpis" style="grid-template-columns:1fr 1fr">
        <div class="kpi"><b>${h && h.total_count ? Math.round(h.hallucination_score * 100) + "%" : "–"}</b><span>Sentences flagged</span></div>
        <div class="kpi"><b>${m.attempts}</b><span>Attempts</span></div>
      </div>
      ${m.reformulated ? `<div class="notice info">Question rewritten to: <i>${esc(m.reformulated)}</i></div>` : ""}
      ${m.repaired ? `<div class="notice ok">Repaired by ${esc(m.repaired.method)}: flagged sentences went from ${Math.round(m.repaired.before * 100)}% to ${Math.round(m.repaired.after * 100)}%.</div>` : ""}
      ${h && h.llm_calls !== undefined && h.embedding_decided !== undefined ? `<p class="small muted">Verification: ${h.embedding_decided} of ${h.total_count} sentences decided by embeddings, ${h.llm_calls} LLM call.</p>` : ""}`;
  } else if (inspTab === "trace") {
    body.innerHTML = m.trace.length ? `<ol class="steps">${m.trace.map((t) => `<li><b>${esc(t.label)}</b>${t.detail ? `<span>${esc(t.detail)}</span>` : ""}</li>`).join("")}</ol>`
      : `<div class="empty"><p>No pipeline trace was recorded for this answer.</p></div>`;
  } else {
    const sents = (m.hallucination && m.hallucination.sentences) || [];
    body.innerHTML = `
      ${sents.length ? `<div class="stack">${sents.map((s) => `<div class="sent ${s.supported ? "ok" : "bad"}">
          <span>${esc(s.text)}</span>
          <small>${s.supported ? "Supported" : "Not supported"}${s.similarity !== undefined ? ` · similarity ${fmt.n(s.similarity, 2)}` : ""}${s.method ? ` · ${esc(s.method)}` : ""}</small>
          ${s.evidence ? `<small>Closest evidence: ${esc(s.evidence.slice(0, 160))}…</small>` : ""}</div>`).join("")}</div>`
        : `<div class="empty"><p>No sentence-level verdicts for this answer.</p></div>`}
      <button class="btn" data-load-sources>${icon("search")} Match each sentence to its source passage</button>
      ${m.sources ? (m.sources.length ? `<div class="stack">${m.sources.map((x) => `<div class="sent ${x.supported ? "ok" : "bad"}"><b>${esc(x.sentence)}</b>
          <small>Similarity ${fmt.n(x.similarity, 2)}</small><span class="small">${esc(x.source)}…</span></div>`).join("")}</div>` : `<p class="muted">No source passages could be matched.</p>`) : ""}`;
  }
}

async function loadSources(m) {
  if (!m.raw) return;
  const body = $("#insp-body", root); body.insertAdjacentHTML("beforeend", `<div class="row" id="src-wait"><span class="spinner"></span> Matching sentences…</div>`);
  try {
    const scope = state.scope === "active" ? state.activeDoc : null;
    const r = await api("/source-highlight", { json: { query: m.query, answer: m.raw, doc_id: scope } });
    m.sources = r.mappings || [];
  } catch (e) { fail(e); }
  renderInspector();
}

/* ═════════════════════════════ memory note, naming ═════════════════════════════ */
function updateMemNote() {
  const turns = Math.floor(llmHistory.length / 2);
  const recent = Math.min(state.memory, turns), older = Math.max(0, turns - state.memory);
  $("#mem-note", root).textContent = turns ? `Using ${recent} recent turn${recent > 1 ? "s" : ""}${older ? ` + a summary of ${older} earlier` : ""}` : "";
}
async function autoName(q) {
  const cur = sessions.find((s) => s.id === state.sessionId);
  if (cur && DEFAULT_NAME.test(cur.name)) {
    try { await api(`/chat-sessions/${state.sessionId}`, { method: "PATCH", json: { action: "rename", name: q.slice(0, 42) } }); await loadSessions(); } catch {}
  }
}

/* ═════════════════════════════ suggestions ═════════════════════════════ */
let sugIdx = -1, sugItems = [];
const suggest = debounce(async () => {
  const q = input.value.trim(); if (q.length < 2 || !state.docs.length) return hideSuggest();
  try {
    const r = await api(`/api/query-suggestions?q=${encodeURIComponent(q)}${state.activeDoc ? `&doc_id=${encodeURIComponent(state.activeDoc)}` : ""}`);
    sugItems = (r.suggestions || []).filter((s) => s.toLowerCase() !== q.toLowerCase());
    if (!sugItems.length) return hideSuggest();
    sugIdx = -1; const box = $("#suggest", root);
    box.innerHTML = sugItems.map((s, i) => `<button data-i="${i}">${esc(s)}</button>`).join(""); box.hidden = false;
    on(box, "click", "button", (e, b) => { input.value = sugItems[+b.dataset.i]; autosize(); hideSuggest(); input.focus(); });
  } catch { hideSuggest(); }
}, 250);
function hideSuggest() { const b = $("#suggest", root); if (b) b.hidden = true; sugIdx = -1; }
function onKey(e) {
  const box = $("#suggest", root);
  if (!box.hidden && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
    e.preventDefault(); sugIdx = (sugIdx + (e.key === "ArrowDown" ? 1 : -1) + sugItems.length) % sugItems.length;
    $$("button", box).forEach((b, i) => b.setAttribute("aria-selected", String(i === sugIdx))); return;
  }
  if (e.key === "Escape") return hideSuggest();
  if (e.key === "Tab" && !box.hidden && sugIdx >= 0) { e.preventDefault(); input.value = sugItems[sugIdx]; autosize(); hideSuggest(); return; }
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!box.hidden && sugIdx >= 0) { input.value = sugItems[sugIdx]; autosize(); hideSuggest(); }
    else send();
  }
}

/* ═════════════════════════════ sessions ═════════════════════════════ */
function bindSessions() {
  const list = $("#session-list", root);
  on(list, "click", "[data-sid]", async (e, el) => {
    const act = e.target.closest("[data-sact]")?.dataset.sact; const sid = el.dataset.sid;
    try {
      if (act === "pin") { await api(`/chat-sessions/${sid}`, { method: "PATCH", json: { action: "pin" } }); await loadSessions(); }
      else if (act === "rename") {
        const cur = sessions.find((s) => s.id === sid); const name = await promptDialog("Rename chat", "Chat name", cur ? cur.name : "");
        if (name) { await api(`/chat-sessions/${sid}`, { method: "PATCH", json: { action: "rename", name } }); await loadSessions(); }
      } else if (act === "delete") {
        if (await confirmDialog("Delete this chat?", "Its messages will be removed. Logged queries and analytics are kept.")) {
          const r = await api(`/chat-sessions/${sid}`, { method: "DELETE" });
          if (sid === state.sessionId) { state.sessionId = r.active_session_id; await loadHistory(); }
          await loadSessions();
        }
      } else if (sid !== state.sessionId) {
        await api(`/chat-sessions/switch/${sid}`, { method: "POST" }); state.sessionId = sid; await loadHistory(); await loadSessions();
        $("#sessions", root).classList.remove("open");
      }
    } catch (err) { fail(err); }
  });
  $("#session-search", root).oninput = () => renderSessions();
}

async function loadSessions() {
  const r = await api("/chat-sessions"); sessions = r.sessions || []; if (r.active_session_id) state.sessionId = r.active_session_id; renderSessions();
}
function renderSessions() {
  const q = $("#session-search", root).value.trim().toLowerCase();
  const list = sessions.filter((s) => !q || s.name.toLowerCase().includes(q) || (s.preview || "").toLowerCase().includes(q));
  $("#session-list", root).innerHTML = list.length ? list.map((s) => `
    <div class="session" data-sid="${esc(s.id)}" aria-current="${s.id === state.sessionId}" tabindex="0">
      <b>${s.is_pinned ? "📌 " : ""}${esc(s.name)}</b>
      <span class="acts">
        <button class="btn icon ghost sm" data-sact="pin" aria-label="Pin">${icon("pin")}</button>
        <button class="btn icon ghost sm" data-sact="rename" aria-label="Rename">${icon("edit")}</button>
        <button class="btn icon ghost sm" data-sact="delete" aria-label="Delete">${icon("trash")}</button>
      </span>
      <small>${esc(s.preview || "Empty chat")}</small></div>`).join("")
    : `<div class="empty small">No chats match.</div>`;
}

async function newChat() {
  try {
    const r = await api("/chat-sessions", { json: { name: "New chat" } });
    state.sessionId = r.session_id; resetThread(); await loadSessions(); input.focus();
    $("#sessions", root).classList.remove("open");
  } catch (e) { fail(e); }
}
/* ═════════════════════════════ delete / select turns ═════════════════════════════ */
function pick(t, on_) {
  const val = on_ === undefined ? t.dataset.picked !== "true" : !!on_;
  t.dataset.picked = String(val); const c = $(".turn-check", t); if (c) c.checked = val;
}
function updateSel() {
  const n = $$('.turn[data-picked="true"]', inner).length;
  $("#sel-count", root).textContent = `${n} selected`; $("#sel-del", root).disabled = !n;
}
function toggleSelect(force) {
  const on_ = force === undefined ? !thread.classList.contains("selecting") : force;
  thread.classList.toggle("selecting", on_); $("#sel-bar", root).hidden = !on_;
  $("#sel-btn", root).setAttribute("aria-pressed", String(on_));
  $$(".turn", inner).forEach((t) => pick(t, false)); updateSel();
}

function rebuildHistory() {                                         // the assistant's memory follows what is left on screen
  llmHistory = [];
  for (const t of $$(".turn", inner)) {
    const q = $(".msg-user", t)?.textContent, card = $("[data-msg]", t), m = card && msgs.get(card.dataset.msg), note = $(".msg-note", t);
    const a = m ? (m.fallback ? "" : m.raw) : note ? note.textContent : "";
    if (q && a) llmHistory.push({ role: "user", content: q }, { role: "assistant", content: a });
  }
  updateMemNote();
}

async function deleteTurns(turns) {
  turns = turns.filter(Boolean); if (!turns.length) return;
  const n = turns.length;
  if (!(await confirmDialog(n === 1 ? "Delete this question and its answer?" : `Delete ${n} questions and their answers?`,
    "They are removed from this chat and from the memory the assistant uses. Your analytics history is kept.", "Delete"))) return;
  try {
    const ids = turns.map((t) => t.dataset.uid).filter(Boolean);
    if (ids.length) await api("/chat-turns/delete", { json: { session_id: state.sessionId, user_message_ids: ids } });
    turns.forEach((t) => t.classList.add("leaving"));
    await new Promise((r) => setTimeout(r, 230));
    turns.forEach((t) => { $$("[data-msg]", t).forEach((c) => { msgs.delete(c.dataset.msg); if (selectedId === c.dataset.msg) selectedId = null; }); t.remove(); });
    rebuildHistory(); renderInspector(); renderStarter(); updateSel();
    if (!$$(".turn", inner).length) toggleSelect(false);
    toast(n === 1 ? "Deleted" : `Deleted ${n} questions`, "ok"); loadSessions().catch(() => {});
  } catch (e) { fail(e); }
}

async function clearChat() {
  if (!(await confirmDialog("Clear this chat?", "All messages in this conversation will be deleted.", "Clear"))) return;
  try { await api(`/chat-history?session_id=${encodeURIComponent(state.sessionId)}`, { method: "DELETE" }); resetThread(); await loadSessions(); } catch (e) { fail(e); }
}
function resetThread() {
  if (thread) toggleSelect(false);
  msgs.clear(); selectedId = null; llmHistory = []; inner.innerHTML = ""; updateMemNote(); renderStarter(); renderInspector();
}

async function loadHistory() {
  resetThread();
  const r = await api(`/chat-history?session_id=${encodeURIComponent(state.sessionId)}&limit=300`);
  let lastQuery = "", lastModel = null, prevRole = "";
  for (const row of r.messages || []) {
    const meta = row.metadata || {};
    if (row.role === "user") { addUser(row.content, row.id); lastQuery = row.content; lastModel = null; }
    else if (row.role === "greeting") { addNote(row.content); llmHistory.push({ role: "user", content: lastQuery }, { role: "assistant", content: row.content }); }
    else if (row.role === "notfound") renderNotFoundStored(row, meta);
    else if (row.role === "fallback") {
      const m = newModel(meta.query || lastQuery); m.fallback = true; m.raw = row.content; m.done = true; cardEl(m); paintCard(m);
      $("[data-bar]", $card(m)).innerHTML = `<button class="btn sm ghost" data-act="copy">${icon("copy")} Copy</button>`;
    } else if (row.role === "assistant") {
      if (lastModel && prevRole === "assistant") {            // a regenerated answer to the same question
        const old = lastModel.raw;
        if (!lastModel.versions.length) lastModel.versions = [snapshot(lastModel)];
        fillFromRow(lastModel, row); lastModel.versions.push(snapshot(lastModel)); lastModel.vi = lastModel.versions.length - 1;
        paintCard(lastModel);
        const at = llmHistory.map((x) => x.content).lastIndexOf(old); if (at >= 0) llmHistory[at].content = row.content;
      } else {
        const m = newModel(lastQuery); fillFromRow(m, row); cardEl(m); paintCard(m); lastModel = m;
        llmHistory.push({ role: "user", content: lastQuery }, { role: "assistant", content: row.content });
      }
    }
    prevRole = row.role;
  }
  updateMemNote(); renderStarter(); toBottom();
}
function renderNotFoundStored(row, meta) {
  const m = newModel(meta.query || ""); const el = cardEl(m); msgs.delete(m.id);
  renderNotFound(m, { suggestion: row.content, partial_matches: meta.partial_matches || [], indexed_docs: [] });
  el.removeAttribute("data-msg");
}
