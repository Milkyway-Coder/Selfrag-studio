// charts.js — tiny SVG charts that follow the theme through CSS variables. No libraries.
import { esc } from "./core.js";

const COLORS = ["var(--brand)", "#c98a00", "var(--ok)", "var(--bad)", "#7a5af8", "#0e9aa7"];
const col = (i) => COLORS[i % COLORS.length];

function frame(w, h, inner, label) {
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label || "chart")}">${inner}</svg>`;
}

/** Numeric line chart. series: [{name, points:[{x,y}], color?, dashed?}] */
export function lineChart({ series, xMin = 0, xMax = 1, yMin = 0, yMax = 1, xLabel = "", yLabel = "",
                            w = 520, h = 280, diagonal = false, xTicks = 5, yTicks = 5, label = "" }) {
  const m = { l: 44, r: 14, t: 12, b: 38 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const X = (v) => m.l + ((v - xMin) / ((xMax - xMin) || 1)) * pw;
  const Y = (v) => m.t + ph - ((v - yMin) / ((yMax - yMin) || 1)) * ph;
  let g = "";
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + (i * (yMax - yMin)) / yTicks;
    g += `<line class="grid" x1="${m.l}" x2="${w - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${+v.toFixed(2)}</text>`;
  }
  for (let i = 0; i <= xTicks; i++) {
    const v = xMin + (i * (xMax - xMin)) / xTicks;
    g += `<text x="${X(v)}" y="${h - m.b + 16}" text-anchor="middle">${+v.toFixed(2)}</text>`;
  }
  g += `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${m.t + ph}" y2="${m.t + ph}"/><line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ph}"/>`;
  if (diagonal) g += `<line x1="${X(xMin)}" y1="${Y(yMin)}" x2="${X(xMax)}" y2="${Y(yMax)}" stroke="var(--muted)" stroke-dasharray="4 4"/>`;
  series.forEach((s, i) => {
    const pts = s.points.filter((p) => p.y !== null && p.y !== undefined);
    if (!pts.length) return;
    const d = pts.map((p, k) => `${k ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
    const c = s.color || col(i);
    g += `<path d="${d}" fill="none" stroke="${c}" stroke-width="2.2" ${s.dashed ? 'stroke-dasharray="5 4"' : ""} stroke-linejoin="round"/>`;
    if (pts.length <= 40) g += pts.map((p) => `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="3" fill="${c}"/>`).join("");
  });
  if (xLabel) g += `<text x="${m.l + pw / 2}" y="${h - 4}" text-anchor="middle">${esc(xLabel)}</text>`;
  if (yLabel) g += `<text transform="translate(12 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
  return frame(w, h, g, label);
}

/** Vertical columns. items: [{label, value, color?}] */
export function columns({ items, max, w = 520, h = 240, fmt = (v) => v, label = "" }) {
  const m = { l: 34, r: 8, t: 18, b: 30 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const top = max || Math.max(1, ...items.map((i) => i.value));
  const bw = pw / items.length;
  let g = `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${m.t + ph}" y2="${m.t + ph}"/>`;
  [0, 0.5, 1].forEach((f) => {
    const y = m.t + ph - f * ph;
    g += `<line class="grid" x1="${m.l}" x2="${w - m.r}" y1="${y}" y2="${y}"/><text x="${m.l - 6}" y="${y + 4}" text-anchor="end">${Math.round(top * f)}</text>`;
  });
  items.forEach((it, i) => {
    const bh = (it.value / top) * ph, x = m.l + i * bw + bw * 0.18;
    g += `<rect x="${x}" y="${m.t + ph - bh}" width="${bw * 0.64}" height="${bh}" rx="3" fill="${it.color || col(0)}"/>`;
    g += `<text x="${x + bw * 0.32}" y="${m.t + ph - bh - 5}" text-anchor="middle">${esc(fmt(it.value))}</text>`;
    g += `<text x="${x + bw * 0.32}" y="${h - 10}" text-anchor="middle">${esc(it.label)}</text>`;
  });
  return frame(w, h, g, label);
}

/** Horizontal bars with optional CI whiskers. items: [{label, value, ci?:[lo,hi]}] (values 0–max) */
export function hbars({ items, max = 100, w = 520, rowH = 30, fmt = (v) => v, label = "" }) {
  const m = { l: 170, r: 56, t: 6, b: 6 };
  const h = m.t + m.b + items.length * rowH, pw = w - m.l - m.r;
  const X = (v) => m.l + (v / max) * pw;
  let g = "";
  items.forEach((it, i) => {
    const y = m.t + i * rowH;
    g += `<text x="${m.l - 8}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(it.label)}</text>`;
    g += `<rect x="${m.l}" y="${y + 6}" width="${pw}" height="${rowH - 12}" rx="4" fill="var(--line-2)"/>`;
    g += `<rect x="${m.l}" y="${y + 6}" width="${Math.max(0, X(it.value) - m.l)}" height="${rowH - 12}" rx="4" fill="${it.color || col(0)}"/>`;
    if (it.ci) g += `<line x1="${X(it.ci[0])}" x2="${X(it.ci[1])}" y1="${y + rowH / 2}" y2="${y + rowH / 2}" stroke="var(--ink)" stroke-width="1.5"/>` +
      `<line x1="${X(it.ci[0])}" x2="${X(it.ci[0])}" y1="${y + 9}" y2="${y + rowH - 9}" stroke="var(--ink)"/><line x1="${X(it.ci[1])}" x2="${X(it.ci[1])}" y1="${y + 9}" y2="${y + rowH - 9}" stroke="var(--ink)"/>`;
    g += `<text x="${w - m.r + 6}" y="${y + rowH / 2 + 4}">${esc(fmt(it.value))}</text>`;
  });
  return frame(w, h, g, label);
}

/** Reliability diagram from bins [{confidence, accuracy, n}] */
export function reliability(bins, label = "reliability diagram") {
  const pts = bins.filter((b) => b.confidence !== null).map((b) => ({ x: b.confidence, y: b.accuracy }));
  return lineChart({ series: [{ name: "accuracy", points: pts }], diagonal: true, xLabel: "Confidence", yLabel: "Accuracy", label });
}

export function legend(entries) {
  return `<div class="legend">${entries.map((e, i) => `<span><i style="background:${e.color || col(i)}"></i>${esc(e.name)}</span>`).join("")}</div>`;
}
export { col };
