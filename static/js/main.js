// main.js — shell: hash router, theme, status pill, global shortcuts.
import { $, $$, api, state, refreshDocs, icon, toast, fail } from "./core.js";
import * as ask from "./ask.js";
import * as chat from "./chat.js";
import * as library from "./library.js";
import * as lab from "./lab.js";
import * as insights from "./insights.js";
import * as arch from "./arch.js";

const VIEWS = {
  ask:          { mod: ask,      title: "Ask" },
  chat:         { mod: chat,     title: "Chat" },
  library:      { mod: library,  title: "Library" },
  lab:          { mod: lab,      title: "Lab" },
  insights:     { mod: insights, title: "Insights" },
  architecture: { mod: arch,     title: "Architecture" },
};
const mounted = new Set();

function route() {
  const name = (location.hash.replace(/^#\/?/, "").split("?")[0]) || "ask";
  const key = VIEWS[name] ? name : "ask";
  for (const k of Object.keys(VIEWS)) {
    const el = $(`#view-${k}`);
    el.hidden = k !== key;
  }
  $$(".nav-item").forEach((a) => { if (a.dataset.nav === key) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  $("#page-title").textContent = VIEWS[key].title;
  document.title = `${VIEWS[key].title} · Self-RAG`;
  const v = VIEWS[key];
  try {
    if (!mounted.has(key)) { mounted.add(key); v.mod.mount($(`#view-${key}`)); }
    v.mod.show && v.mod.show();
  } catch (e) { fail(e); }
}

async function status() {
  const pill = $("#status-pill");
  try {
    const s = await api("/api/status");
    const bad = !s.groq_key;
    pill.className = `pill ${bad ? "bad" : s.reranker === "loading" ? "warn" : "ok"}`;
    pill.innerHTML = `<span class="dot"></span>${bad ? "GROQ_API_KEY missing" : `${s.docs} document${s.docs === 1 ? "" : "s"}${s.reranker === "loading" ? " · loading reranker" : ""}`}`;
    pill.title = `Generator ${s.gen_model} · Judge ${s.judge_model} · Reranker ${s.reranker}`;
    if (bad && !status.warned) { status.warned = true; toast("Set GROQ_API_KEY in your .env file, then restart the server.", "bad", 8000); }
  } catch { pill.className = "pill bad"; pill.innerHTML = `<span class="dot"></span>Server unreachable`; }
}

function theme() {
  const saved = localStorage.getItem("sr.theme");
  const dark = saved ? saved === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  const btn = $("#theme-btn");
  btn.innerHTML = icon(dark ? "sun" : "moon");
  btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
}

document.addEventListener("DOMContentLoaded", async () => {
  theme();
  $("#theme-btn").onclick = () => {
    const dark = document.documentElement.dataset.theme !== "dark";
    localStorage.setItem("sr.theme", dark ? "dark" : "light"); theme();
  };
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); location.hash = "#/ask"; setTimeout(() => ask.focusComposer(), 30); }
  });
  document.addEventListener("docs:changed", status);
  window.addEventListener("hashchange", route);
  try { await refreshDocs(); } catch (e) { fail(e); }
  route(); status(); setInterval(status, 30000);
});
