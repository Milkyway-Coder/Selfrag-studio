// quiz.js — "Quiz me": multiple-choice revision questions generated from a document.
import { $, $$, esc, api, dialog, fail, markdown } from "./core.js";

const inlineMd = (t) => markdown(String(t)).replace(/^<p>|<\/p>$/g, "");

export function openQuiz(docId) {
  const dlg = dialog(`Quiz: ${docId}`, `<div id="qz"></div>`, { foot: `<button class="btn" data-close>Close</button>` });
  const box = $("#qz", dlg.el);
  let qs = [], i = 0, score = 0, wrong = [], answered = false, off = null;

  const setup = () => {
    box.innerHTML = `<p class="small muted" style="margin:0">Questions are written from passages spread across the whole document, each with four options and a short explanation.</p>
      <div class="field"><label for="qz-n">Number of questions</label><select id="qz-n" class="inp" style="max-width:160px"><option>5</option><option>8</option><option>10</option></select></div>
      <div><button class="btn primary" id="qz-go">Start quiz</button></div>`;
    $("#qz-go", box).onclick = async () => {
      box.innerHTML = `<div class="row"><span class="spinner"></span> Writing questions from the document…</div>`;
      try {
        qs = (await api("/api/quiz", { json: { doc_id: docId, count: +$("#qz-n", box)?.value || 5 } })).questions;
        i = 0; score = 0; wrong = []; show();
      } catch (e) { box.innerHTML = `<div class="notice bad">${esc(e.message)}</div><div style="margin-top:10px"><button class="btn" id="qz-retry">Try again</button></div>`; $("#qz-retry", box).onclick = setup; }
    };
  };

  const show = () => {
    const q = qs[i]; answered = false;
    box.innerHTML = `
      <div class="row"><b class="grow">Question ${i + 1} of ${qs.length}</b><span class="muted small num">Score ${score}</span></div>
      <div class="progress" aria-hidden="true"><i style="width:${(i / qs.length) * 100}%"></i></div>
      <p style="font-size:1.05rem;font-weight:600;margin:6px 0 2px">${inlineMd(q.question)}</p>
      <div class="stack" id="qz-opts">${q.options.map((o, k) => `<button class="quiz-opt" data-k="${k}"><span class="quiz-key">${"ABCD"[k]}</span><span>${inlineMd(o)}</span></button>`).join("")}</div>
      <div id="qz-fb"></div>`;
  };

  const choose = (k) => {
    if (answered) return; answered = true;
    const q = qs[i], ok = k === q.answer;
    if (ok) score++; else wrong.push({ q, picked: k });
    $$(".quiz-opt", box).forEach((b, n) => { b.disabled = true; b.classList.add(n === q.answer ? "correct" : n === k ? "wrong" : "dim"); });
    $("#qz-fb", box).innerHTML = `<div class="notice ${ok ? "ok" : "bad"}"><b>${ok ? "Correct." : `Not quite. The answer is ${"ABCD"[q.answer]}.`}</b> ${esc(q.explanation || "")}</div>
      <div style="margin-top:10px"><button class="btn primary" id="qz-next">${i + 1 < qs.length ? "Next question" : "See result"}</button></div>`;
    $("#qz-next", box).onclick = next; $("#qz-next", box).focus();
  };

  const next = () => { if (i + 1 < qs.length) { i++; show(); } else result(); };

  const result = () => {
    const pct = Math.round((score / qs.length) * 100);
    box.innerHTML = `<div class="row wrap" style="gap:18px"><div style="font-size:2.6rem;font-weight:650" class="num">${score}/${qs.length}</div>
        <div><span class="chip ${pct >= 80 ? "ok" : pct >= 50 ? "brand" : "warn"}">${pct}%</span><p class="small muted" style="margin:4px 0 0">${pct >= 80 ? "Strong grasp of this material." : pct >= 50 ? "A decent start. Review the ones you missed." : "Worth another read before trying again."}</p></div></div>
      ${wrong.length ? `<h3>Review</h3><div class="stack">${wrong.map(({ q, picked }) => `<div class="sent bad"><b>${inlineMd(q.question)}</b>
          <span class="small">Your answer: ${inlineMd(q.options[picked])}</span><span class="small">Correct: <b>${inlineMd(q.options[q.answer])}</b></span><small>${esc(q.explanation || "")}</small></div>`).join("")}</div>` : `<div class="notice ok">You answered every question correctly.</div>`}
      <div class="row"><button class="btn primary" id="qz-again">New quiz</button></div>`;
    $("#qz-again", box).onclick = setup;
  };

  box.addEventListener("click", (e) => { const b = e.target.closest(".quiz-opt"); if (b) choose(+b.dataset.k); });
  off = (e) => {
    if (!dlg.el.isConnected) return document.removeEventListener("keydown", off);
    if (/^[1-4]$/.test(e.key) && $("#qz-opts", box)) choose(+e.key - 1);
  };
  document.addEventListener("keydown", off);
  setup();
}
