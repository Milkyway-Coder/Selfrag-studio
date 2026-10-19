// library.js — add content (text, PDF, web page, crawl), manage documents, per-document tools.
import { openQuiz } from "./quiz.js";
import {
  $, $$, esc, fmt, icon, api, toast, fail, dialog, confirmDialog, on, debounce,
  state, persist, refreshDocs, setActiveDoc, rememberSource, SAMPLE_DOC,
} from "./core.js";

let root, docs = [], tab = "text", picked = [];
const ACCEPT = ".pdf,.docx,.pptx,.xlsx,.xlsm,.csv,.tsv,.ipynb,.zip,.txt,.md,.py,.js,.ts,.tsx,.jsx,.java,.c,.h,.cpp,.cc,.hpp,.cs,.go,.rs,.rb,.php,.kt,.swift,.sql,.sh,.html,.css,.json,.yaml,.yml,.xml,.toml,.ini,.tex,.r,.m";
const KIND = { pdf: "PDF", docx: "Word", pptx: "PowerPoint", sheet: "Spreadsheet", code: "Code", project: "Project", notebook: "Notebook", text: "Text" };

export function mount(el) {
  root = el;
  root.innerHTML = `
  <div class="page">
    <div>
      <h1>Library</h1>
      <p class="muted">Everything the system can answer from. Each document is chunked, embedded and indexed for both dense and keyword search.</p>
    </div>
    <div class="lib-grid">
      <section class="card" aria-labelledby="add-h">
        <h2 id="add-h">Add content</h2>
        <div class="tabs" role="tablist" id="add-tabs" style="margin:10px 0 14px">
          <button role="tab" data-tab="text" aria-selected="true">Text</button>
          <button role="tab" data-tab="files" aria-selected="false">Files</button>
          <button role="tab" data-tab="url" aria-selected="false">Web page</button>
          <button role="tab" data-tab="crawl" aria-selected="false">Crawl site</button>
        </div>

        <div data-pane="text" class="stack">
          <div class="field"><label for="t-id">Document name</label><input id="t-id" class="inp" placeholder="flask-docs"></div>
          <div class="field"><label for="t-text">Text</label><textarea id="t-text" class="inp" rows="8" placeholder="Paste the document text (at least 10 words)"></textarea></div>
          <button class="btn primary" data-go="text">Index text</button>
        </div>

        <div data-pane="files" class="stack" hidden>
          <div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose files">
            <div><b id="drop-main">Drop files here or click to browse</b></div>
            <div class="small">PDF, Word, PowerPoint, Excel, CSV, code files, notebooks, or a .zip project</div>
            <input type="file" id="p-file" multiple accept="${ACCEPT}" hidden>
          </div>
          <ul class="files-list" id="file-list"></ul>
          <label class="check"><input type="checkbox" id="p-deep"><span><b>Deep analysis</b> (slower, uses more API quota)
            <small>Sends PDF pages and figures to a vision model so formulas, charts and scanned pages are read and described. Without it, tables and figures are still extracted, but text inside images and formulas drawn as images are not.</small></span></label>
          <div class="field"><label for="p-id">Document name (optional, only when uploading one file)</label><input id="p-id" class="inp" placeholder="Uses the file name"></div>
          <button class="btn primary" data-go="files" disabled id="p-go">Upload and index</button>
        </div>

        <div data-pane="url" class="stack" hidden>
          <div class="field"><label for="u-url">Page address</label><input id="u-url" class="inp" placeholder="https://example.com/article" inputmode="url"></div>
          <div class="field"><label for="u-id">Document name (optional)</label><input id="u-id" class="inp" placeholder="Uses the site name"></div>
          <button class="btn primary" data-go="url">Fetch and index</button>
        </div>

        <div data-pane="crawl" class="stack" hidden>
          <div class="field"><label for="c-url">Starting address</label><input id="c-url" class="inp" placeholder="https://docs.python.org/3/library/" inputmode="url"></div>
          <div class="field"><label for="c-max">Maximum pages: <b id="c-max-v" class="num">10</b></label><input id="c-max" type="range" min="1" max="50" value="10"></div>
          <p class="small muted" style="margin:0">Follows links on the same site only. Each page becomes its own document.</p>
          <button class="btn primary" data-go="crawl">Crawl and index</button>
        </div>

        <div id="add-status" style="margin-top:14px"></div>
        <hr style="border:0;border-top:1px solid var(--line);margin:16px 0">
        <button class="btn" data-go="sample">Load a sample document</button>
      </section>

      <section class="stack">
        <div class="row">
          <h2 class="grow">Documents <span class="chip num" id="doc-count">0</span></h2>
          <input class="inp" id="doc-filter" placeholder="Filter" aria-label="Filter documents" style="width:180px">
        </div>
        <div id="doc-list" class="stack"></div>
      </section>
    </div>
  </div>`;

  bind();
  document.addEventListener("docs:changed", () => root && !root.hidden && load().catch(fail));
}

export function show() { load().catch(fail); }

async function load() {
  const r = await api("/api/library");
  docs = r.docs || []; render();
}

function render() {
  const q = $("#doc-filter", root).value.trim().toLowerCase();
  const list = docs.filter((d) => !q || d.doc_id.toLowerCase().includes(q) || d.tags.some((t) => t.tag.includes(q)));
  $("#doc-count", root).textContent = docs.length;
  $("#doc-list", root).innerHTML = list.length ? list.map((d) => `
    <article class="doc" data-id="${esc(d.doc_id)}" data-active="${d.doc_id === state.activeDoc}">
      <input type="checkbox" data-sel aria-label="Include ${esc(d.doc_id)} in selected search" title="Include in selected search" ${state.selected.has(d.doc_id) ? "checked" : ""}>
      <h3 title="${esc(d.doc_id)}">${esc(d.doc_id)}</h3>
      <span class="acts">
        <button class="btn sm ${d.doc_id === state.activeDoc ? "" : "primary"}" data-act="use" ${d.doc_id === state.activeDoc ? "disabled" : ""}>${d.doc_id === state.activeDoc ? "Active" : "Use"}</button>
        <button class="btn sm" data-act="quiz" title="Quiz yourself on this document">Quiz</button>
        <button class="btn sm" data-act="details">Details</button>
        <button class="btn sm icon danger" data-act="delete" aria-label="Delete ${esc(d.doc_id)}">${icon("trash")}</button>
      </span>
      <div class="meta"><span class="chip brand">${esc(KIND[d.kind] || d.kind)}</span>
        ${statChips(d.kind, d.stats || {}).map((c) => `<span class="chip">${esc(c)}</span>`).join("")}
        <span class="num">${d.word_count.toLocaleString()} words</span><span>·</span><span class="num">${d.chunks} chunks</span>
        ${d.tags.map((t) => `<span class="chip">${esc(t.tag)}</span>`).join("")}</div>
    </article>`).join("")
    : `<div class="card empty"><h3>${docs.length ? "No match" : "No documents yet"}</h3><p>${docs.length ? "Try a different filter." : "Add text, a PDF or a web page on the left, or load the sample document."}</p></div>`;
}

/* ───────────────────────── bindings ───────────────────────── */
function bind() {
  on($("#add-tabs", root), "click", "[data-tab]", (e, b) => {
    tab = b.dataset.tab;
    $$("#add-tabs button", root).forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $$("[data-pane]", root).forEach((p) => (p.hidden = p.dataset.pane !== tab));
    $("#add-status", root).innerHTML = "";
  });
  $("#c-max", root).oninput = (e) => ($("#c-max-v", root).textContent = e.target.value);
  $("#doc-filter", root).oninput = debounce(render, 120);

  // file picker + drag-and-drop (several files at once)
  const drop = $("#drop", root), file = $("#p-file", root);
  const sz = (n) => (n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");
  const pick = (list) => {
    picked = [...list];
    $("#file-list", root).innerHTML = picked.map((f) => `<li><span>${esc(f.name)}</span><span class="muted">${sz(f.size)}</span></li>`).join("");
    $("#drop-main", root).textContent = picked.length ? `${picked.length} file${picked.length > 1 ? "s" : ""} selected` : "Drop files here or click to browse";
    $("#p-go", root).disabled = !picked.length;
  };
  drop.onclick = () => file.click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } };
  file.onchange = () => pick(file.files);
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => pick(e.dataTransfer.files));

  on(root, "click", "[data-go]", (e, b) => go(b.dataset.go, b));

  const list = $("#doc-list", root);
  on(list, "click", "[data-act]", async (e, b) => {
    const id = b.closest("[data-id]").dataset.id;
    try {
      if (b.dataset.act === "use") { await setActiveDoc(id); toast(`${id} is now the active document`, "ok"); }
      else if (b.dataset.act === "details") openDetails(id);
      else if (b.dataset.act === "quiz") openQuiz(id);
      else if (b.dataset.act === "delete") {
        if (await confirmDialog(`Delete “${id}”?`, "Its chunks, embeddings and keyword index are removed. This cannot be undone.")) {
          await api(`/doc/${encodeURIComponent(id)}`, { method: "DELETE" });
          state.selected.delete(id); persist("sr.selected", [...state.selected]);
          await refreshDocs(); toast("Document deleted", "ok");
        }
      }
    } catch (err) { fail(err); }
  });
  on(list, "change", "[data-sel]", (e, c) => {
    const id = c.closest("[data-id]").dataset.id;
    c.checked ? state.selected.add(id) : state.selected.delete(id);
    persist("sr.selected", [...state.selected]);
  });
}

/* ───────────────────────── adding ───────────────────────── */
const slug = (s) => s.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase();

function status(html, kind = "info") { $("#add-status", root).innerHTML = html ? `<div class="notice ${kind}">${html}</div>` : ""; }
const doneLinks = `<br><a href="#/ask">Ask a question</a>`;

async function go(kind, btn) {
  const label = btn.textContent; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Working…`;
  try {
    if (kind === "text") {
      const id = slug($("#t-id", root).value), text = $("#t-text", root).value.trim();
      if (!id) throw new Error("Give the document a name.");
      if (text.split(/\s+/).length < 10) throw new Error("Add at least 10 words of text.");
      await api("/index-doc", { json: { doc_id: id, text } });
      status(`Indexed <b>${esc(id)}</b>.${doneLinks}`, "ok"); $("#t-text", root).value = "";
    } else if (kind === "files") {
      if (!picked.length) throw new Error("Choose at least one file first.");
      const fd = new FormData(); picked.forEach((f) => fd.append("files", f));
      fd.append("deep", $("#p-deep", root).checked ? "1" : "0"); fd.append("doc_id", slug($("#p-id", root).value));
      const job = await api("/upload-file", { body: fd, method: "POST" });
      const res = await waitIngest(job.job_id);
      showResults(res, job.rejected || []);
      picked = []; $("#file-list", root).innerHTML = ""; $("#drop-main", root).textContent = "Drop files here or click to browse";
    } else if (kind === "url") {
      const url = $("#u-url", root).value.trim();
      if (!/^https?:\/\//i.test(url)) throw new Error("The address must start with http:// or https://");
      const r = await api("/scrape-url", { json: { url, doc_id: slug($("#u-id", root).value) } });
      rememberSource(r.doc_id, url);
      status(`Indexed <b>${esc(r.doc_id)}</b> (${(r.word_count || 0).toLocaleString()} words).${doneLinks}`, "ok");
    } else if (kind === "crawl") {
      const url = $("#c-url", root).value.trim();
      if (!/^https?:\/\//i.test(url)) throw new Error("The address must start with http:// or https://");
      status(`<span class="spinner"></span> Crawling. This can take a while for many pages.`);
      const r = await api("/crawl", { json: { url, max_pages: +$("#c-max", root).value } });
      (r.pages || []).forEach((p) => rememberSource(p.doc_id, p.url));
      status(`Indexed <b>${r.indexed_count}</b> pages, ${r.failed_count} failed, ${r.skipped_count} skipped (too little content).${doneLinks}`, "ok");
    } else if (kind === "sample") {
      await api("/index-doc", { json: { doc_id: SAMPLE_DOC.id, text: SAMPLE_DOC.text } });
      status(`Sample document <b>${SAMPLE_DOC.id}</b> indexed.${doneLinks}`, "ok");
    }
    await refreshDocs();
  } catch (e) { status(esc(e.message), "bad"); }
  finally { btn.disabled = kind === "files" && !picked.length; btn.textContent = label; }
}

async function waitIngest(jid) {
  for (;;) {
    const j = await api(`/api/ingest/jobs/${jid}`);
    const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
    status(`<div class="stack"><span><span class="spinner"></span> ${esc(j.message || "Working")}</span><div class="progress"><i style="width:${pct}%"></i></div></div>`);
    if (j.status === "done") return j.results;
    await new Promise((r) => setTimeout(r, 900));
  }
}

function statChips(kind, st) {
  const c = [];
  if (st.pages) c.push(`${st.pages} ${kind === "pptx" ? "slides" : "pages"}`);
  if (st.figures) c.push(`${st.figures} figure${st.figures > 1 ? "s" : ""}`);
  if (st.tables && kind !== "sheet") c.push(`${st.tables} table${st.tables > 1 ? "s" : ""}`);
  if (st.vision_pages) c.push(`${st.vision_pages} pages read by vision`);
  if (st.sheets) c.push(`${st.sheets} sheet${st.sheets > 1 ? "s" : ""}, ${(st.rows || 0).toLocaleString()} rows`);
  if (st.code_files) c.push(`${st.code_files} file${st.code_files > 1 ? "s" : ""}`);
  if (st.code_cells) c.push(`${st.code_cells} code cells`);
  return c;
}

function showResults(results, rejected) {
  const ok = results.filter((r) => !r.error);
  status(`<div class="results-list">
    ${ok.length ? `<b>Indexed ${ok.length} file${ok.length > 1 ? "s" : ""}.</b> <a href="#/ask">Ask a question</a>` : ""}
    ${results.map((r) => r.error
      ? `<div><b>${esc(r.name)}</b>: ${esc(r.error)}</div>`
      : `<div><b>${esc(r.doc_id)}</b> <span class="chip">${esc(KIND[r.kind] || r.kind)}</span> ${statChips(r.kind, r.stats).map((c) => `<span class="chip">${esc(c)}</span>`).join(" ")}
         ${(r.warnings || []).map((w) => `<div class="small" style="margin-top:4px">Note: ${esc(w)}</div>`).join("")}</div>`).join("")}
    ${rejected.map((r) => `<div><b>${esc(r.name)}</b>: ${esc(r.error)}</div>`).join("")}</div>`,
    ok.length && !results.some((r) => r.error) && !rejected.length ? "ok" : "warn");
}

/* ───────────────────────── details dialog ───────────────────────── */
async function openDetails(id) {
  const d = docs.find((x) => x.doc_id === id); if (!d) return;
  const url = state.sources[id];
  const dlg = dialog(id, `
    <div class="kpis">
      <div class="kpi"><b class="num">${d.word_count.toLocaleString()}</b><span>Words indexed</span></div>
      <div class="kpi"><b class="num">${d.chunks}</b><span>Parent chunks (about 400 words)</span></div>
      <div class="kpi"><b class="num">${d.small_chunks}</b><span>Child chunks (about 100 words)</span></div>
      <div class="kpi"><b class="num">${fmt.pct(d.term_count / Math.max(d.word_count, 1), 0)}</b><span>Vocabulary diversity</span></div>
    </div>

    <section class="stack" id="dd-figs-sec" hidden><h3>Figures</h3><div class="gallery" id="dd-figs"></div></section>
    <section class="stack" id="dd-data-sec" hidden><h3>Data preview</h3><div id="dd-data"></div></section>

    <section class="stack"><h3>Top terms</h3><div class="terms" id="dd-terms"><span class="spinner"></span></div></section>

    <section class="stack"><div class="row"><h3 class="grow">Summary</h3><button class="btn sm" data-d="regen">Regenerate</button></div>
      <div id="dd-summary" class="small"><span class="spinner"></span> Loading…</div></section>

    <section class="stack"><h3>Named entities</h3><div id="dd-ents" class="small"><span class="spinner"></span></div></section>

    <section class="stack"><h3>Tags</h3>
      <div class="row wrap" id="dd-tags"></div>
      <div class="row"><input class="inp" id="dd-tag-in" placeholder="Add a tag" style="max-width:220px"><button class="btn sm" data-d="tag">Add</button></div></section>

    <section class="stack"><h3>Update this document</h3>
      <p class="small muted" style="margin:0">Paste a newer version to re-index it and see which terms were added, removed or re-weighted.</p>
      <textarea class="inp" id="dd-new" rows="4" placeholder="Updated text"></textarea>
      <div class="row"><button class="btn sm" data-d="reindex">Re-index and compare</button>
        ${url ? `<button class="btn sm" data-d="check-url">Check ${esc(new URL(url).host)} for changes</button>` : ""}</div>
      <div id="dd-diff"></div></section>

    <section class="stack"><div class="row"><h3 class="grow">Benchmark questions</h3>
      <select class="inp" id="dd-n" style="width:auto"><option>5</option><option selected>10</option><option>15</option><option>20</option></select>
      <button class="btn sm" data-d="qa">Generate</button></div>
      <div id="dd-qa" class="muted small">Generate question and answer pairs from this document, then run them in the Lab.</div></section>`, { wide: true });

  const q = (s) => $(s, dlg.el);
  const renderTags = async () => {
    const r = await api(`/doc-tags?doc_id=${encodeURIComponent(id)}`);
    q("#dd-tags").innerHTML = (r.tags || []).length ? r.tags.map((t) => `<span class="chip">${esc(t.tag)}<button data-rm="${t.id}" aria-label="Remove tag ${esc(t.tag)}">×</button></span>`).join("") : `<span class="muted small">No tags.</span>`;
  };
  api(`/page-index-terms?doc_id=${encodeURIComponent(id)}&top_k=18`).then((r) => {
    q("#dd-terms").innerHTML = (r.terms || []).map((t) => `<span class="chip" title="tf-idf ${t.tfidf.toFixed(4)}">${esc(t.term)}</span>`).join("") || "<span class='muted small'>No terms.</span>";
  }).catch((e) => (q("#dd-terms").textContent = e.message));
  renderTags().catch(fail);

  // summary, entities and tags are produced automatically after upload; show them, or wait for them
  const ENT = { people: "People", organizations: "Organizations", locations: "Places", dates: "Dates", technologies: "Technologies", key_terms: "Key terms" };
  const showEnrichment = (e) => {
    q("#dd-summary").textContent = e.summary || "No summary available.";
    const groups = Object.entries(ENT).filter(([k]) => (e.entities?.[k] || []).length);
    q("#dd-ents").innerHTML = groups.length
      ? groups.map(([k, label]) => `<div style="margin-bottom:8px"><span class="lbl">${label}</span><div class="terms">${e.entities[k].map((x) => `<span class="chip brand">${esc(x)}</span>`).join("")}</div></div>`).join("")
      : `<span class="muted">No named entities found.</span>`;
    renderTags().catch(() => {}); load();
  };
  let tries = 0;
  const loadEnrichment = async () => {
    try {
      const e = await api(`/api/docs/${encodeURIComponent(id)}/enrichment`);
      if (e.ready) return showEnrichment(e);
      if (!dlg.el.isConnected) return;
      q("#dd-summary").innerHTML = `<span class="spinner"></span> Writing the summary, entities and tags…`;
      q("#dd-ents").innerHTML = `<span class="muted">Waiting for the summary.</span>`;
      if (tries++ < 4) return setTimeout(loadEnrichment, 2500);           // a fresh upload is usually being processed already
      showEnrichment(await api(`/api/docs/${encodeURIComponent(id)}/enrich`, { method: "POST" }));   // older document, or it failed earlier
    } catch (err) {
      q("#dd-summary").innerHTML = `<span class="muted">It could not be generated (${esc(err.message)}). Press Regenerate.</span>`; q("#dd-ents").innerHTML = "";
    }
  };
  loadEnrichment();
  if ((d.stats || {}).figures) api(`/api/docs/${encodeURIComponent(id)}/figures`).then((r) => {
    q("#dd-figs-sec").hidden = !r.figures.length;
    q("#dd-figs").innerHTML = r.figures.map((f) => `<figure class="fig-card"><img loading="lazy" src="${esc(f.url)}" alt="${esc(f.caption)}"><figcaption><b>${f.label ? "Figure " + esc(f.label) : "Figure"}</b>${f.page ? " · page " + f.page : ""}${f.caption && !/^Figure on page/i.test(f.caption) ? "<br>" + esc(f.caption.slice(0, 80)) : ""}</figcaption></figure>`).join("");
  }).catch(() => {});
  if (d.kind === "sheet") api(`/api/docs/${encodeURIComponent(id)}/data`).then((r) => {
    q("#dd-data-sec").hidden = !r.sheets.length;
    q("#dd-data").innerHTML = r.sheets.map((s) => `<p class="small muted" style="margin:6px 0">${esc(s.sheet)}: ${s.rows.toLocaleString()} rows. Ask questions such as "average ${esc(s.columns[s.columns.length - 1])} by ${esc(s.columns[1] || s.columns[0])}" and the numbers are calculated exactly.</p>
      <div class="table-wrap"><table><thead><tr>${s.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${s.head.map((row) => `<tr>${row.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`).join("");
  }).catch(() => {});

  on(dlg.el, "click", "[data-rm]", async (e, b) => { try { await api(`/doc-tags/${b.dataset.rm}`, { method: "DELETE" }); await renderTags(); load(); } catch (x) { fail(x); } });
  on(dlg.el, "click", "[data-d]", async (e, b) => {
    const act = b.dataset.d, prev = b.textContent; b.disabled = true;
    try {
      if (act === "regen") {
        q("#dd-summary").innerHTML = `<span class="spinner"></span> Regenerating…`; q("#dd-ents").innerHTML = "";
        showEnrichment(await api(`/api/docs/${encodeURIComponent(id)}/enrich`, { method: "POST" }));
      } else if (act === "tag") {
        const tag = q("#dd-tag-in").value.trim().toLowerCase(); if (!tag) return;
        await api("/doc-tags", { json: { doc_id: id, tag } }); q("#dd-tag-in").value = ""; await renderTags(); load();
      } else if (act === "reindex") {
        const text = q("#dd-new").value.trim(); if (!text) throw new Error("Paste the updated text first.");
        const r = await api("/reindex-diff", { json: { doc_id: id, text } });
        q("#dd-diff").innerHTML = r.diff ? diffHtml(r.diff) : `<div class="notice ok">Indexed as a new document.</div>`;
        await refreshDocs(); load();
      } else if (act === "check-url") {
        q("#dd-diff").innerHTML = `<span class="spinner"></span> Fetching…`;
        const r = await api("/check-url-changes", { json: { url, doc_id: id } });
        q("#dd-diff").innerHTML = r.changed
          ? `<div class="notice warn">Changed: ${r.old_words} → ${r.new_words} words (${r.diff_pct}%). <button class="btn sm" data-d="apply-url">Re-index with the new content</button></div>`
          : `<div class="notice ok">No significant change (${r.diff_pct}% word-count difference).</div>`;
        dlg.newText = r.new_text;
      } else if (act === "apply-url") {
        const r = await api("/reindex-diff", { json: { doc_id: id, text: dlg.newText } });
        q("#dd-diff").innerHTML = r.diff ? diffHtml(r.diff) : ""; await refreshDocs(); load();
      } else if (act === "qa") {
        q("#dd-qa").innerHTML = `<span class="spinner"></span> Generating…`;
        const r = await api("/generate-qa", { json: { doc_id: id, count: +q("#dd-n").value } });
        const pairs = r.pairs || [];
        q("#dd-qa").innerHTML = `<div class="stack">${pairs.map((p) => `<div class="sent"><b>${esc(p.question)}</b><span>${esc(p.expected)}</span></div>`).join("")}
          <div class="row"><button class="btn primary sm" data-d="send-lab">Use in the Lab</button></div></div>`;
        dlg.pairs = pairs;
      } else if (act === "send-lab") {
        persist("sr.qa", { doc_id: id, pairs: dlg.pairs }); dlg.close(); location.hash = "#/lab"; toast("Questions loaded into the Lab", "ok");
      }
    } catch (err) { fail(err); if (act !== "reindex") { /* keep the section readable */ } }
    finally { if (b.isConnected) { b.disabled = false; b.textContent = prev; } }
  });
}

function diffHtml(d) {
  const t = (arr, cls, sign) => arr.length ? `<div class="terms">${arr.slice(0, 14).map((x) => `<span class="chip ${cls}">${sign}${esc(x.term)}</span>`).join("")}</div>` : `<span class="muted small">None</span>`;
  return `<div class="stack"><div class="small muted">Vocabulary ${d.stats.old_vocab} → ${d.stats.new_vocab} terms · ${d.stats.changed_count} re-weighted</div>
    <div><span class="lbl">Added</span>${t(d.added, "ok", "+")}</div>
    <div><span class="lbl">Removed</span>${t(d.removed, "bad", "−")}</div>
    <div><span class="lbl">Most re-weighted</span><div class="terms">${d.changed.slice(0, 10).map((x) => `<span class="chip ${x.delta > 0 ? "ok" : "warn"}">${esc(x.term)} ${x.delta > 0 ? "▲" : "▼"}</span>`).join("") || "<span class='muted small'>None</span>"}</div></div></div>`;
}
