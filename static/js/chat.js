// chat.js — talk to the model directly. Not grounded in documents, not scored, not logged.
// Conversations live in this browser (localStorage); each assistant reply can have several versions.
import { $, $$, esc, icon, api, toast, fail, confirmDialog, promptDialog, on, markdown, renderMath, sse, persist } from "./core.js";

const KEY = "sr.chats", ACTIVE = "sr.chats.active";
let root, thread, inner, input, sendBtn, abort = null;
let chats = [], activeId = null, memory = 8, model = "", selecting = false, picked = new Set();

const uid = () => Math.random().toString(36).slice(2, 10);
const now = () => Date.now();
const cur = () => chats.find((c) => c.id === activeId);
const save = () => {
  try {
    localStorage.setItem(KEY, JSON.stringify(chats.slice(0, 60).map((c) => ({ ...c, msgs: c.msgs.slice(-120) }))));
    localStorage.setItem(ACTIVE, activeId);
  } catch { /* storage full or blocked: the chat still works for this session */ }
};
const blank = () => ({ id: uid(), name: "New chat", pinned: false, updated: now(), msgs: [] });

function load() {
  try { chats = JSON.parse(localStorage.getItem(KEY)) || []; } catch { chats = []; }
  if (!chats.length) {                                              // migrate the single-thread version
    let old = []; try { old = JSON.parse(localStorage.getItem("sr.chat")) || []; } catch {}
    chats = [{ ...blank(), name: old.length ? "Earlier chat" : "New chat",
      msgs: old.map((m) => (m.role === "assistant" ? { role: "assistant", content: m.content, versions: [m.content], vi: 0, error: m.error } : m)) }];
  }
  activeId = localStorage.getItem(ACTIVE);
  if (!chats.some((c) => c.id === activeId)) activeId = chats[0].id;
}

export function mount(el) {
  root = el; load();
  memory = +localStorage.getItem("sr.chat.memory") || 8; model = localStorage.getItem("sr.chat.model") || "";
  root.innerHTML = `
  <div class="talk" id="talk">
    <aside class="sessions" id="tk-sessions" aria-label="Chats">
      <div class="sessions-head">
        <button class="btn primary" data-act="new">${icon("plus")} New chat</button>
        <input class="inp" id="tk-search" placeholder="Search chats" aria-label="Search chats">
      </div>
      <div class="session-list" id="tk-list"></div>
    </aside>
    <div class="talk-main">
      <div class="talk-bar">
        <button class="btn icon only-mobile" data-act="toggle-list" aria-label="Show chats">${icon("menu")}</button>
        <button class="btn icon ghost not-mobile" data-act="toggle-list-desk" aria-label="Show or hide chats">${icon("menu")}</button>
        <span class="chip warn">Not from your documents</span>
        <label class="row small" for="tk-model">Model <input class="inp" id="tk-model" list="tk-models" style="width:220px" autocomplete="off"><datalist id="tk-models"></datalist></label>
        <label class="row small" for="tk-mem">Memory <select class="inp" id="tk-mem" style="width:auto;padding:3px 6px">${[4, 8, 12, 20].map((n) => `<option value="${n}">${n} turns</option>`).join("")}</select></label>
        <span class="right row">
          <button class="btn sm ghost" data-act="select-mode" id="tk-selbtn" title="Select questions to delete">${icon("check")}<span class="hide-narrow"> Select</span></button>
          <button class="btn sm ghost" data-act="clear">${icon("trash")}<span class="hide-narrow"> Clear</span></button></span>
      </div>
      <div class="unverified">Answers here come from the model's general knowledge. They are not taken from your documents, scored or checked, so verify anything important. To answer from your files, use Ask.</div>
      <div class="thread" id="tk-thread"><div class="thread-inner" id="tk-inner"></div></div>
      <div class="sel-bar" id="tk-selbar" hidden>
        <b id="tk-selcount">0 selected</b><button class="btn sm" data-act="sel-all">Select all</button>
        <span class="right row"><button class="btn sm danger" data-act="del-selected" id="tk-seldel" disabled>${icon("trash")} Delete selected</button><button class="btn sm" data-act="select-mode">Done</button></span>
      </div>
      <div class="composer"><div class="composer-inner">
        <div class="compose-box">
          <textarea id="tk-q" rows="1" placeholder="Ask anything: concepts, code, maths, an explanation of something in your course" aria-label="Message"></textarea>
          <button class="btn primary" id="tk-send" aria-label="Send">${icon("send")}</button>
        </div>
        <div class="compose-hint"><span>Enter to send, Shift+Enter for a new line</span></div>
      </div></div>
    </div>
  </div>`;
  thread = $("#tk-thread", root); inner = $("#tk-inner", root); input = $("#tk-q", root); sendBtn = $("#tk-send", root);
  if (localStorage.getItem("sr.chats.closed") === "true") $("#talk", root).classList.add("no-sessions");
  $("#tk-mem", root).value = String(memory);
  $("#tk-mem", root).onchange = (e) => { memory = +e.target.value; localStorage.setItem("sr.chat.memory", memory); };
  const mi = $("#tk-model", root); mi.value = model;
  mi.onchange = () => { model = mi.value.trim(); localStorage.setItem("sr.chat.model", model); };
  api("/api/status").then((s) => {
    $("#tk-models", root).innerHTML = [s.gen_model, s.judge_model].filter(Boolean).map((m) => `<option value="${esc(m)}">`).join("");
    if (!model) { model = s.gen_model; mi.value = model; }
  }).catch(() => {});

  sendBtn.onclick = () => (abort ? abort.abort() : send());
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
  $("#tk-search", root).oninput = renderList;

  on(root, "click", "[data-act]", async (e, b) => {
    const a = b.dataset.act, chat = cur();
    if (a === "new") newChat();
    else if (a === "toggle-list") $("#tk-sessions", root).classList.toggle("open");
    else if (a === "toggle-list-desk") persist("sr.chats.closed", $("#talk", root).classList.toggle("no-sessions"));
    else if (a === "clear") { if (chat.msgs.length && await confirmDialog("Clear this chat?", "Its messages are removed from this browser.", "Clear")) { chat.msgs = []; touch(); render(); } }
    else if (a === "prompt") send(b.dataset.q);
    else if (a === "copy") { const m = chat.msgs[+b.dataset.i]; navigator.clipboard.writeText(m.content).then(() => toast("Copied", "ok"), () => toast("Copy was blocked", "bad")); }
    else if (a === "regen") regenerate(+b.dataset.i);
    else if (a === "v-prev" || a === "v-next") {
      const m = chat.msgs[+b.dataset.i]; m.vi += a === "v-next" ? 1 : -1; m.content = m.versions[m.vi]; m.error = null; save(); render(false);
    } else if (a === "docs") { sessionStorage.setItem("sr.ask.pending", b.dataset.q); location.hash = "#/ask"; }
    else if (a === "select-mode") toggleSelect();
    else if (a === "sel-all") { picked = new Set(chat.msgs.map((m, i) => (m.role === "user" ? i : -1)).filter((i) => i >= 0)); render(false); }
    else if (a === "del-selected") deleteTurns([...picked]);
    else if (a === "del-turn") deleteTurns([+b.dataset.i]);
  });
  inner.addEventListener("click", (e) => {                           // select mode: a click on a turn picks it
    if (!selecting) return;
    const t = e.target.closest(".turn"); if (!t) return;
    const i = +t.dataset.i; picked.has(i) ? picked.delete(i) : picked.add(i); render(false);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && selecting && root && !root.hidden) toggleSelect(false); });
  on($("#tk-list", root), "click", "[data-cid]", async (e, el) => {
    const act = e.target.closest("[data-sact]")?.dataset.sact, id = el.dataset.cid, c = chats.find((x) => x.id === id);
    if (act === "pin") { c.pinned = !c.pinned; save(); renderList(); }
    else if (act === "rename") { const n = await promptDialog("Rename chat", "Chat name", c.name); if (n) { c.name = n; save(); renderList(); } }
    else if (act === "delete") {
      if (await confirmDialog("Delete this chat?", "It is removed from this browser.")) {
        chats = chats.filter((x) => x.id !== id); if (!chats.length) chats = [blank()];
        if (activeId === id) activeId = chats[0].id;
        save(); render();
      }
    } else if (id !== activeId) { activeId = id; selecting = false; picked = new Set(); save(); render(); $("#tk-sessions", root).classList.remove("open"); }
  });
  render();
}

function toggleSelect(force) {
  selecting = force === undefined ? !selecting : force; picked = new Set(); render(false);
  $("#tk-selbtn", root).setAttribute("aria-pressed", String(selecting));
}

async function deleteTurns(idxs) {
  const chat = cur(), n = idxs.length; if (!n) return;
  if (!(await confirmDialog(n === 1 ? "Delete this question and its answer?" : `Delete ${n} questions and their answers?`,
    "They are removed from this chat and from the memory the model uses.", "Delete"))) return;
  const drop = new Set();
  idxs.forEach((i) => { drop.add(i); for (let j = i + 1; j < chat.msgs.length && chat.msgs[j].role !== "user"; j++) drop.add(j); });
  chat.msgs = chat.msgs.filter((_, i) => !drop.has(i)); picked = new Set();
  if (!chat.msgs.length) selecting = false;
  touch(); render(false); toast(n === 1 ? "Deleted" : `Deleted ${n} questions`, "ok");
}

function newChat() {
  if (cur() && !cur().msgs.length) { input.focus(); return; }       // reuse an empty chat instead of piling up blanks
  const c = blank(); chats.unshift(c); activeId = c.id; selecting = false; picked = new Set(); save(); render(); input.focus();
  $("#tk-sessions", root).classList.remove("open");
}
function touch() { const c = cur(); c.updated = now(); save(); }

export function show() {
  const p = sessionStorage.getItem("sr.chat.pending");                // handed over from a "not found" card in Ask
  if (p) { sessionStorage.removeItem("sr.chat.pending"); newChat(); send(p); }
  input && input.focus();
}

function renderList() {
  const q = ($("#tk-search", root).value || "").trim().toLowerCase();
  const list = [...chats].sort((a, b) => (b.pinned - a.pinned) || (b.updated - a.updated))
    .filter((c) => !q || c.name.toLowerCase().includes(q) || c.msgs.some((m) => m.content && m.content.toLowerCase().includes(q)));
  $("#tk-list", root).innerHTML = list.length ? list.map((c) => {
    const last = [...c.msgs].reverse().find((m) => m.role === "user");
    return `<div class="session" data-cid="${c.id}" aria-current="${c.id === activeId}" tabindex="0">
      <b>${c.pinned ? "📌 " : ""}${esc(c.name)}</b>
      <span class="acts">
        <button class="btn icon ghost sm" data-sact="pin" aria-label="Pin">${icon("pin")}</button>
        <button class="btn icon ghost sm" data-sact="rename" aria-label="Rename">${icon("edit")}</button>
        <button class="btn icon ghost sm" data-sact="delete" aria-label="Delete">${icon("trash")}</button></span>
      <small>${esc(last ? last.content.slice(0, 70) : "Empty chat")}</small></div>`;
  }).join("") : `<div class="empty small">No chats match.</div>`;
}

function render(scroll = true) {
  renderList();
  const chat = cur(), msgs = chat.msgs;
  if (!msgs.length) {
    inner.innerHTML = `<div class="starter"><h2>Ask the model directly</h2>
      <p class="muted">Use this for general questions, for checking your understanding, or when the answer is not in your documents.</p>
      <div class="prompts">${["Explain recursion with a Python example", "What is the time complexity of quicksort and why?", "Derive the gradient of mean squared error", "Explain the difference between TCP and UDP"]
        .map((q) => `<button class="btn" data-act="prompt" data-q="${esc(q)}">${esc(q)}</button>`).join("")}</div></div>`;
    return;
  }
  const turns = [];                                                   // [{i: index of the user message, html: [...]}]
  msgs.forEach((m, i) => {
    if (m.role === "user") { turns.push({ i, html: [`<div class="msg-user">${esc(m.content)}</div>`] }); return; }
    if (!turns.length) turns.push({ i: -1, html: [] });
    const prevUser = [...msgs.slice(0, i)].reverse().find((x) => x.role === "user");
    const n = (m.versions || []).length, last = i === msgs.length - 1;
    turns[turns.length - 1].html.push(`<div class="talk-msg" id="tm-${i}"><div class="bubble" data-body>${m.error ? `<div class="notice bad">${esc(m.error)}</div>` : markdown(m.content || "")}</div>
      <div class="acts">
        ${n > 1 ? `<span class="vers"><button class="btn sm icon ghost" data-act="v-prev" data-i="${i}" ${m.vi === 0 ? "disabled" : ""} aria-label="Previous version">‹</button><span class="num small">${m.vi + 1}/${n}</span><button class="btn sm icon ghost" data-act="v-next" data-i="${i}" ${m.vi === n - 1 ? "disabled" : ""} aria-label="Next version">›</button></span>` : ""}
        <button class="btn sm ghost" data-act="copy" data-i="${i}">${icon("copy")} Copy</button>
        ${last && !abort ? `<button class="btn sm ghost" data-act="regen" data-i="${i}">${icon("refresh")} ${m.error ? "Try again" : "Regenerate"}</button>` : ""}
        ${prevUser ? `<button class="btn sm ghost" data-act="docs" data-q="${esc(prevUser.content)}">${icon("search")} Ask my documents instead</button>` : ""}</div></div>`);
  });
  inner.innerHTML = turns.map((t) => `<div class="turn" data-i="${t.i}" data-picked="${picked.has(t.i)}">
      <input type="checkbox" class="turn-check" ${picked.has(t.i) ? "checked" : ""} aria-label="Select this question" tabindex="-1">
      ${t.i >= 0 ? `<div class="turn-tools"><button class="btn sm ghost icon" data-act="del-turn" data-i="${t.i}" aria-label="Delete this question and its answer" title="Delete this question and its answer">${icon("trash")}</button></div>` : ""}
      ${t.html.join("")}</div>`).join("");
  $("#tk-thread", root).classList.toggle("selecting", selecting);
  $("#tk-selbar", root).hidden = !selecting;
  $("#tk-selcount", root).textContent = `${picked.size} selected`; $("#tk-seldel", root).disabled = !picked.size;
  renderMath(inner);
  if (scroll) thread.scrollTop = thread.scrollHeight;
}

let raf = 0;
function paint(i) {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0; const el = $(`#tm-${i} [data-body]`, root); if (!el) return;
    const stick = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 140;
    el.innerHTML = markdown(cur().msgs[i].content); if (stick) thread.scrollTop = thread.scrollHeight;
  });
}

function regenerate(i) {
  const chat = cur();
  if (!chat.msgs[i] || abort) return;
  stream(chat, i, true);
}

async function send(text) {
  if (abort) return;
  text = (text ?? input.value).trim(); if (!text) return;
  const chat = cur();
  chat.msgs.push({ role: "user", content: text });
  if (/^(new chat|earlier chat)$/i.test(chat.name)) chat.name = text.slice(0, 42);
  input.value = ""; input.style.height = "auto";
  chat.msgs.push({ role: "assistant", content: "", versions: [""], vi: 0 });
  touch();
  stream(chat, chat.msgs.length - 1, false);
}

/** Streams a reply into assistant message `i`. For a regeneration the reply becomes a NEW version. */
async function stream(chat, i, regen) {
  const m = chat.msgs[i];
  const prev = { content: m.content, error: m.error, vi: m.vi };
  if (regen) { m.versions.push(""); m.vi = m.versions.length - 1; m.content = ""; m.error = null; }
  abort = new AbortController(); render();
  sendBtn.innerHTML = icon("stop"); sendBtn.setAttribute("aria-label", "Stop");
  try {
    const hist = chat.msgs.slice(0, i).filter((x) => !x.error && x.content).slice(-(memory * 2 + 1)).map(({ role, content }) => ({ role, content }));
    const resp = await fetch("/api/general-chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: hist, model: model || undefined }), signal: abort.signal });
    if (!resp.ok) { let msg = `Request failed (${resp.status})`; try { msg = (await resp.json()).error || msg; } catch {} throw new Error(msg); }
    let done = false;
    for await (const ev of sse(resp)) {
      if (ev.type === "token") { m.content += ev.content; m.versions[m.vi] = m.content; paint(i); }
      else if (ev.type === "error") throw new Error(ev.message);
      else if (ev.type === "done") done = true;
    }
    if (!done) throw new Error("The connection closed before the reply finished.");
  } catch (err) {
    const empty = !m.content;
    if (regen && empty) { m.versions.pop(); m.vi = prev.vi; m.content = prev.content; m.error = prev.error; }   // keep the earlier version
    else if (err.name !== "AbortError") m.error = err.message;
    else if (empty) chat.msgs.splice(i, 1);
    if (err.name !== "AbortError") fail(err);
  } finally {
    abort = null; sendBtn.innerHTML = icon("send"); sendBtn.setAttribute("aria-label", "Send"); touch(); render();
  }
}
