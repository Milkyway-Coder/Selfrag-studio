// core.js — shared helpers and global state. No framework; every view renders with template strings
// and ALWAYS passes dynamic text through esc().

export const $  = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export const esc = (s) => String(s ?? "")
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

export const fmt = {
  n: (v, d = 0) => (v === null || v === undefined || Number.isNaN(+v)) ? "–" : (+v).toFixed(d),
  pct: (v, d = 0) => (v === null || v === undefined || Number.isNaN(+v)) ? "–" : `${(+v * 100).toFixed(d)}%`,
  date: (s) => s ? String(s).replace("T", " ").slice(0, 16) : "",
};

export const icon = (name) => {
  const p = {
    chat: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    library: '<path d="M4 4h5v16H4zM11 4h5v16h-5zM18 6l3 .8-3.6 13.2-3-.8z"/>',
    flask: '<path d="M9 3h6M10 3v6l-5.5 9.5A1.5 1.5 0 0 0 5.8 21h12.4a1.5 1.5 0 0 0 1.3-2.5L14 9V3"/><path d="M7.5 15h9"/>',
    chart: '<path d="M4 20V4M4 20h16M8 16v-5M12 16V8M16 16v-8"/>',
    layers: '<path d="M12 3 3 8l9 5 9-5-9-5zM3 13l9 5 9-5M3 17.5 12 22l9-4.5"/>',
    send: '<path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
    up: '<path d="M7 11v9H4v-9zM7 11l4-8a2 2 0 0 1 2 2v4h6a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 17.8 20H7"/>',
    down: '<path d="M17 13V4h3v9zM17 13l-4 8a2 2 0 0 1-2-2v-4H5a2 2 0 0 1-2-2.3l1.2-6A2 2 0 0 1 6.2 4H17"/>',
    bookmark: '<path d="M6 3h12v18l-6-4-6 4z"/>',
    refresh: '<path d="M20 11a8 8 0 0 0-14.5-4M4 4v4h4M4 13a8 8 0 0 0 14.5 4M20 20v-4h-4"/>',
    download: '<path d="M12 3v12M7 10l5 5 5-5M4 20h16"/>',
    pin: '<path d="M9 3h6l-1 6 4 4H6l4-4zM12 13v8"/>',
    edit: '<path d="M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    sidebar: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10z"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    check: '<path d="m5 12 5 5 9-10"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    play: '<path d="M7 4v16l13-8z"/>',
    file: '<path d="M6 3h8l4 4v14H6zM14 3v4h4"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    talk: '<path d="M4 5h16v11H9l-5 4zM8 9h8M8 12h5"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-8 8"/>',
  }[name] || "";
  return `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">${p}</svg>`;
};

/* ── API ───────────────────────────────────────────────────────────────────── */
export async function api(url, opts = {}) {
  const init = { method: opts.method || (opts.json !== undefined ? "POST" : "GET"), headers: {}, signal: opts.signal };
  if (opts.json !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.json); }
  if (opts.body) init.body = opts.body;
  let res;
  try { res = await fetch(url, init); }
  catch (e) { if (e.name === "AbortError") throw e; throw new Error("Could not reach the server. Is it still running?"); }
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text.slice(0, 200) }; }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}

export async function download(url, opts = {}, filename = "download") {
  const init = { method: opts.json !== undefined ? "POST" : "GET", headers: {} };
  if (opts.json !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.json); }
  const res = await fetch(url, init);
  if (!res.ok) { let m = `Download failed (${res.status})`; try { m = (await res.json()).error || m; } catch {} throw new Error(m); }
  const cd = res.headers.get("Content-Disposition") || "";
  const m = /filename=([^;]+)/.exec(cd);
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = m ? m[1].replace(/"/g, "") : filename;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

/* ── Toast + dialog ────────────────────────────────────────────────────────── */
export function toast(msg, kind = "", ms = 3200) {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`; el.setAttribute("role", "status"); el.textContent = msg;
  box.append(el);
  setTimeout(() => el.remove(), ms);
}
export const fail = (e) => { if (e && e.name === "AbortError") return; console.error(e); toast(e.message || String(e), "bad", 5000); };

/** Opens a <dialog>. Returns {el, body, close}. `html` is trusted markup built with esc(). */
export function dialog(title, html, { foot = "", wide = false, onClose } = {}) {
  const d = document.createElement("dialog");
  if (wide) d.style.width = "min(960px, calc(100vw - 24px))";
  d.innerHTML = `<div class="dlg-head"><h2 class="grow">${esc(title)}</h2>
      <button class="btn ghost icon" data-close aria-label="Close">${icon("x")}</button></div>
    <div class="dlg-body">${html}</div>${foot ? `<div class="dlg-foot">${foot}</div>` : ""}`;
  document.body.append(d);
  const close = () => { if (d.open) d.close(); };
  d.addEventListener("click", (e) => { if (e.target === d || e.target.closest("[data-close]")) close(); });
  d.addEventListener("close", () => { d.remove(); onClose && onClose(); });
  d.showModal();
  return { el: d, body: $(".dlg-body", d), close };
}

export function confirmDialog(title, message, okLabel = "Delete") {
  return new Promise((resolve) => {
    let result = false;
    const dlg = dialog(title, `<p>${esc(message)}</p>`, {
      foot: `<button class="btn" data-close>Cancel</button><button class="btn danger" data-ok>${esc(okLabel)}</button>`,
      onClose: () => resolve(result),
    });
    $("[data-ok]", dlg.el).addEventListener("click", () => { result = true; dlg.close(); });
  });
}

export function promptDialog(title, label, value = "", okLabel = "Save") {
  return new Promise((resolve) => {
    let result = null;
    const dlg = dialog(title, `<div class="field"><label for="pd">${esc(label)}</label><input id="pd" class="inp" value="${esc(value)}"></div>`, {
      foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" data-ok>${esc(okLabel)}</button>`,
      onClose: () => resolve(result),
    });
    const inp = $("#pd", dlg.el);
    const ok = () => { result = inp.value.trim(); dlg.close(); };
    $("[data-ok]", dlg.el).addEventListener("click", ok);
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") ok(); });
    inp.focus(); inp.select();
  });
}

/** Parses a text/event-stream response into JSON events. */
export async function* sse(resp) {
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

/* ── Delegated events ──────────────────────────────────────────────────────── */
export function on(root, type, selector, handler) {
  root.addEventListener(type, (e) => {
    const t = e.target.closest(selector);
    if (t && root.contains(t)) handler(e, t);
  });
}

export const debounce = (fn, ms = 250) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

/* ── Markdown (small, safe subset): headings, nested lists, tables, quotes, links, fenced code, LaTeX ───── */
function inline(s) {
  // formatting never applies inside `code`, and the only HTML allowed through is <br> (models like to put it in table cells)
  return s.split(/(`[^`\n]+`)/).map((seg, i) => {
    if (i % 2) return `<code>${seg.slice(1, -1)}</code>`;
    return seg
      .replace(/&lt;br\s*\/?&gt;/gi, "<br>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
      .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  }).join("");
}

const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
const LIST_RE = /^(\s*)([-*•]|\d+[.)])\s+(.*)$/;
const HR_RE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const looksMath = (t) => /[\\^_{}=a-zA-Z]/.test(t);          // "$5-$" is money, "$x^2$" is maths
const splitRow = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));

export function markdown(raw) {
  const codes = [], maths = [];
  let src = String(raw ?? "").replace(/\r\n/g, "\n");
  // fenced code first (an unterminated fence is allowed so streaming text renders as it arrives)
  src = src.replace(/```([\w+#.-]*)[ \t]*\n([\s\S]*?)```/g, (_, lang, code) => { codes.push({ lang, code }); return `\n\u0004${codes.length - 1}\u0005\n`; });
  src = src.replace(/```([\w+#.-]*)[ \t]*\n([\s\S]*)$/, (_, lang, code) => { codes.push({ lang, code }); return `\n\u0004${codes.length - 1}\u0005\n`; });
  // maths: $$...$$ and \[...\] are display; $...$ and \(...\) are inline
  src = src.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g, (_, a, b) => { maths.push({ tex: (a ?? b).trim(), block: true }); return `\n\u0006${maths.length - 1}\u0007\n`; });
  src = src.replace(/\\\(([\s\S]+?)\\\)|\$(?!\s)([^$\n]+?)(?<!\s)\$/g, (m, a, b) => {
    const tex = (a ?? b); if (b !== undefined && !looksMath(tex)) return m;
    maths.push({ tex: tex.trim(), block: false }); return `\u0006${maths.length - 1}\u0007`;
  });

  const lines = esc(src).split("\n");
  const out = []; const stack = []; let para = [], quote = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join("<br>"))}</p>`); para = []; } };
  const flushQuote = () => { if (quote.length) { out.push(`<blockquote><p>${inline(quote.join("<br>"))}</p></blockquote>`); quote = []; } };
  const closeLists = () => { while (stack.length) out.push("</li>", `</${stack.pop().type}>`); };
  const openList = (type, indent, num) => { stack.push({ type, indent }); out.push(type === "ol" && num > 1 ? `<ol start="${num}">` : `<${type}>`); };
  const task = (t) => t.replace(/^\[( |x|X)\]\s+/, (_, c) => (c === " " ? "☐ " : "☑ "));

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]; let m;
    if (!line.trim()) { flushPara(); flushQuote(); continue; }                // a blank line does NOT end a list ("1. a\n\n2. b" is one list)
    if (stack.length && !LIST_RE.test(line) && !/^\s{2,}\S/.test(line)) closeLists();
    if ((m = /^\s*&gt;\s?(.*)$/.exec(line))) { flushPara(); closeLists(); quote.push(m[1]); continue; }
    flushQuote();
    if (HR_RE.test(line) && !LIST_RE.test(line)) { flushPara(); closeLists(); out.push("<hr>"); continue; }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]) && line.trim().startsWith("|")) {
      flushPara(); closeLists();
      const head = splitRow(line), align = splitRow(lines[i + 1]).map((c) => (/^:-+:$/.test(c) ? "center" : /-:$/.test(c) ? "right" : "left"));
      const rows = []; i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(splitRow(lines[i])); i++; }
      i--;
      const cell = (tag, c, k) => `<${tag} style="text-align:${align[k] || "left"}">${inline(c)}</${tag}>`;
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c, k) => cell("th", c, k)).join("")}</tr></thead><tbody>` +
        rows.map((r) => `<tr>${head.map((_, k) => cell("td", r[k] ?? "", k)).join("")}</tr>`).join("") + `</tbody></table></div>`);
      continue;
    }
    if ((m = /^\s*(#{1,4})\s+(.*)$/.exec(line))) { flushPara(); closeLists(); const h = m[1].length <= 2 ? "h3" : "h4"; out.push(`<${h}>${inline(m[2])}</${h}>`); continue; }
    if ((m = LIST_RE.exec(line))) {
      flushPara();
      const indent = m[1].replace(/\t/g, "    ").length, type = /\d/.test(m[2]) ? "ol" : "ul", num = parseInt(m[2], 10) || 1;
      while (stack.length > 1 && indent < stack[stack.length - 1].indent) out.push("</li>", `</${stack.pop().type}>`);
      const top = stack[stack.length - 1];
      if (!top || indent > top.indent + 0) openList(type, indent, num);             // deeper: nested inside the open <li>
      else if (top.type !== type) { out.push("</li>", `</${stack.pop().type}>`); openList(type, indent, num); }
      else out.push("</li>");
      out.push(`<li>${inline(task(m[3]))}`);
      continue;
    }
    if (stack.length) { out.push(`<br>${inline(line.trim())}`); continue; }        // an indented continuation of a list item
    para.push(line);
  }
  flushPara(); flushQuote(); closeLists();
  return out.join("")
    .replace(/<p>\u0004(\d+)\u0005<\/p>/g, "\u0004$1\u0005")
    .replace(/\u0004(\d+)\u0005/g, (_, i) => {
      const c = codes[+i];
      return `<pre class="codeblock"><div class="cb-head"><span>${esc(c.lang || "code")}</span><button class="cb-copy" data-copy>Copy</button></div><code>${esc(c.code.replace(/\n$/, ""))}</code></pre>`;
    })
    .replace(/<p>\u0006(\d+)\u0007<\/p>/g, "\u0006$1\u0007")
    .replace(/\u0006(\d+)\u0007/g, (_, i) => {
      const x = maths[+i];
      return `<span class="math${x.block ? " math-block" : ""}" data-tex="${esc(x.tex)}"><code>${esc(x.tex)}</code></span>`;
    });
}

/* KaTeX is loaded lazily from a CDN; if it cannot load, the LaTeX source stays visible. */
let katexReady = null;
function loadKatex() {
  if (window.katex) return Promise.resolve(true);
  if (!katexReady) katexReady = new Promise((res) => {
    const css = document.createElement("link"); css.rel = "stylesheet"; css.href = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"; document.head.append(css);
    const js = document.createElement("script"); js.src = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js";
    js.onload = () => res(true); js.onerror = () => res(false); document.head.append(js);
  });
  return katexReady;
}
export async function renderMath(root) {
  const els = [...root.querySelectorAll(".math:not([data-done])")];
  if (!els.length) return;
  const ok = await loadKatex();
  els.forEach((el) => {
    el.dataset.done = "1";
    if (!ok || !window.katex) return;
    try { window.katex.render(el.dataset.tex, el, { displayMode: el.classList.contains("math-block"), throwOnError: false }); } catch { /* keep the source */ }
  });
}

document.addEventListener("click", (e) => {                       // copy buttons on code blocks
  const b = e.target.closest("[data-copy]"); if (!b) return;
  const code = b.closest("pre")?.querySelector("code"); if (!code) return;
  navigator.clipboard.writeText(code.textContent).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1400); }, () => {});
});

/** Markdown with verified-sentence marks. `sentences` = [{text, supported, similarity, method, evidence}] */
export function markdownWithEvidence(raw, sentences) {
  if (!sentences || !sentences.length) return markdown(raw);
  let text = raw;
  sentences.forEach((s, i) => {
    let words = String(s.text).trim().split(/\s+/).filter(Boolean);
    while (words.length && /^(\d+[.)]|[-*•]|#{1,4}|>)$/.test(words[0])) words.shift();      // keep "2." / "-" outside the mark so lists survive
    if (!words.length) return;
    const re = new RegExp(words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+"));
    const m = re.exec(text);
    if (!m || /\n/.test(m[0]) || /\|/.test(m[0]) || /[\u0001-\u0007]/.test(m[0])) return;   // never wrap across list items, table cells or other marks
    text = text.slice(0, m.index) + `\u0001${i}\u0002` + m[0] + "\u0003" + text.slice(m.index + m[0].length);
  });
  return markdown(text)
    .replace(/\u0001(\d+)\u0002/g, (_, i) => {
      const s = sentences[+i];
      return `<span class="ev ${s.supported ? "ev-ok" : "ev-bad"}" tabindex="0" data-ev="${i}">`;
    })
    .replace(/\u0003/g, "</span>");
}

/* ── Global state + doc registry ───────────────────────────────────────────── */
const store = (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } };
export const persist = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

export const state = {
  docs: [], docInfo: {}, activeDoc: null,
  memory: store("sr.memory", 4),
  scope: store("sr.scope", "active"),            // active | all | selected
  selected: new Set(store("sr.selected", [])),
  threshold: store("sr.threshold", 3),
  hybrid: store("sr.hybrid", false),
  repair: store("sr.repair", false),
  sources: store("sr.sources", {}),             // doc_id -> url (for change checks)
  sessionId: null,
};

export async function refreshDocs() {
  const [lib, a] = await Promise.all([api("/api/library"), api("/active-doc")]);
  state.docInfo = Object.fromEntries((lib.docs || []).map((d) => [d.doc_id, d]));
  state.docs = (lib.docs || []).map((d) => d.doc_id);
  state.activeDoc = a.active_doc_id && state.docs.includes(a.active_doc_id) ? a.active_doc_id : null;
  state.selected = new Set([...state.selected].filter((x) => state.docs.includes(x)));
  document.dispatchEvent(new CustomEvent("docs:changed"));
  return state.docs;
}

export async function setActiveDoc(id) {
  await api("/set-active-doc", { json: { doc_id: id || "" } });
  state.activeDoc = id || null;
  document.dispatchEvent(new CustomEvent("docs:changed"));
}

export function rememberSource(docId, url) {
  state.sources[docId] = url; persist("sr.sources", state.sources);
}

export function downloadText(name, text, mime = "text/plain") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime })); a.download = name;
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

export const SAMPLE_DOC = {
  id: "flask-intro",
  text: `Flask is a lightweight web framework for Python. It is designed to make getting started quick and easy, with the ability to scale up to complex applications. Flask began as a simple wrapper around Werkzeug and Jinja2 and has become one of the most popular Python web frameworks.

Flask was created by Armin Ronacher and is based on the Werkzeug toolkit and the Jinja2 template engine. It does not require particular tools or libraries. Flask is called a microframework because it keeps the core simple but extensible.

Flask has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions. However, Flask supports extensions that can add application features as if they were implemented in Flask itself. Extensions exist for object-relational mappers, form validation, upload handling, various open authentication technologies and several common framework related tools.

Routing in Flask is declared with the route decorator, which binds a URL rule to a Python function. A view function returns the response body, and Flask builds the response object around it. Flask also provides a development server and debugger, integrated support for unit testing, and RESTful request dispatching.

Sessions in Flask use signed cookies, so the user can look at the contents of the session cookie but cannot modify it unless they know the secret key used for signing. Templates are rendered with Jinja2, which supports template inheritance, macros and automatic escaping of HTML to prevent cross-site scripting.`,
};
