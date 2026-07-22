// Series panel: per-frame scalar signals of one channel across the active
// frame range (whole dataset, or the sequence restricted in the inspector).
// Signal mode plots one component with an optional rolling mean/variance
// window (the imu-roughness recipe); path mode plots component X against
// component Y (trajectories from pose channels). Values are reduced
// server-side (see registry.series): 1-D frames index directly (imu, speed
// vectors), matrix frames index flat (a 4x4 pose trajectory is cols 3 and
// 7), per-point clouds reduce a column to its finite mean.

import { el } from "./datapanel.js";
import { fillSelect } from "./panels.js";
import * as store from "./store.js";
import { fmt } from "./views.js";

const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name);

// NaN/null-aware rolling mean or variance over a trailing window.
export function rolling(values, window, stat) {
  const out = new Array(values.length).fill(null);
  const buf = [];
  for (let i = 0; i < values.length; i++) {
    buf.push(Number.isFinite(values[i]) ? values[i] : null);
    if (buf.length > window) buf.shift();
    const xs = buf.filter((x) => x !== null);
    if (!xs.length) continue;
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    out[i] = stat === "variance"
      ? xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length
      : mean;
  }
  return out;
}

const finiteBounds = (arrays) => {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) {
    for (const v of a) {
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo)) return null;
  if (hi <= lo) hi = lo + 1;
  return [lo, hi];
};

export function seriesSpec(binding) {
  const { nodeId, nodeLabel, channel, len } = binding;
  return {
    key: `series:${nodeId}:${channel}`,
    tag: "SERIES",
    title: `${channel} series @ ${nodeLabel}`,
    binding,
    build(panel) {
      panel.body.classList.add("data-body");
      const meta = el("div", "chan-values");
      const stage = el("div", "data-stage");
      const canvas = el("canvas", "chan-canvas series-canvas");
      canvas.width = 640;
      canvas.height = 260;
      stage.appendChild(canvas);
      const foot = el("div", "chan-values");
      panel.body.append(meta, stage, foot);

      const modeSel = el("select", "chan-color");
      fillSelect(modeSel, [["signal", "signal"], ["path", "path (x/y)"]]);
      const colX = el("input", "opt-input frame-input");
      colX.type = "number";
      colX.value = "0";
      colX.title = "component (1-D: index, matrix: flat index, cloud: column)";
      const colY = el("input", "opt-input frame-input");
      colY.type = "number";
      colY.value = "1";
      colY.title = "path only: the component plotted on y";
      const winInput = el("input", "opt-input frame-input");
      winInput.type = "number";
      winInput.min = "0";
      winInput.value = "0";
      winInput.title = "rolling window in frames (0 = raw signal)";
      const statSel = el("select", "chan-color");
      fillSelect(statSel, [["mean", "rolling mean"], ["variance", "rolling variance"]]);
      panel.controls.append(modeSel, colX, colY, winInput, statSel);

      let data = null; // {start, stop, truncated, cols:{"0":[...]}}
      let seq = 0;

      const range = () => {
        const r = store.getRange();
        return r ? [r.start, r.stop] : [0, len ?? 1];
      };

      const paintControls = () => {
        const path = modeSel.value === "path";
        colY.hidden = !path;
        winInput.hidden = path;
        statSel.hidden = path || Number(winInput.value) <= 0;
      };

      const draw = () => {
        const ctx = canvas.getContext("2d");
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        if (!data) return;
        const { start, stop } = data;
        const count = stop - start;
        const accent = cssVar("--accent"), mutedC = cssVar("--muted");
        ctx.font = "10px ui-monospace, monospace";

        if (modeSel.value === "path") {
          const xs = data.cols[colX.value] || [];
          const ys = data.cols[colY.value] || [];
          const bx = finiteBounds([xs]), by = finiteBounds([ys]);
          if (!bx || !by) return;
          const span = Math.max(bx[1] - bx[0], by[1] - by[0], 1e-9);
          const scale = (Math.min(w, h) - 24) / span;
          const cx = (bx[0] + bx[1]) / 2, cy = (by[0] + by[1]) / 2;
          const px = (x) => w / 2 + (x - cx) * scale;
          const py = (y) => h / 2 - (y - cy) * scale;
          ctx.strokeStyle = accent;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          let started = false;
          for (let i = 0; i < count; i++) {
            if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue;
            if (started) ctx.lineTo(px(xs[i]), py(ys[i]));
            else { ctx.moveTo(px(xs[i]), py(ys[i])); started = true; }
          }
          ctx.stroke();
          if (Number.isFinite(xs[0]) && Number.isFinite(ys[0])) {
            ctx.fillStyle = mutedC;
            ctx.fillRect(px(xs[0]) - 3, py(ys[0]) - 3, 6, 6); // start marker
          }
          const f = store.getFrame() - start;
          if (f >= 0 && f < count && Number.isFinite(xs[f]) && Number.isFinite(ys[f])) {
            ctx.fillStyle = accent;
            ctx.beginPath();
            ctx.arc(px(xs[f]), py(ys[f]), 4, 0, 2 * Math.PI);
            ctx.fill();
          }
          foot.textContent =
            `x: c${colX.value} [${fmt(bx[0])}, ${fmt(bx[1])}] · ` +
            `y: c${colY.value} [${fmt(by[0])}, ${fmt(by[1])}]`;
          return;
        }

        const raw = data.cols[colX.value] || [];
        const win = Number(winInput.value) || 0;
        const rolled = win > 1 ? rolling(raw, win, statSel.value) : null;
        const shown = rolled ?? raw;
        const bounds = finiteBounds(rolled ? [raw, rolled] : [raw]);
        if (!bounds) { foot.textContent = "no finite values"; return; }
        const [lo, hi] = bounds;
        const px = (i) => (count > 1 ? (i / (count - 1)) * (w - 8) + 4 : w / 2);
        const py = (v) => h - 16 - ((v - lo) / (hi - lo)) * (h - 28);
        const trace = (values, color, width) => {
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          ctx.beginPath();
          let started = false;
          for (let i = 0; i < count; i++) {
            if (!Number.isFinite(values[i])) { started = false; continue; }
            if (started) ctx.lineTo(px(i), py(values[i]));
            else { ctx.moveTo(px(i), py(values[i])); started = true; }
          }
          ctx.stroke();
        };
        if (rolled) trace(raw, mutedC, 1);
        trace(shown, accent, 1.4);

        // Current-frame marker: the chart doubles as a navigator (click to
        // seek), so show where the global slider sits.
        const f = store.getFrame() - start;
        if (f >= 0 && f < count) {
          ctx.strokeStyle = mutedC;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(px(f), 4);
          ctx.lineTo(px(f), h - 14);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.fillStyle = mutedC;
        ctx.fillText(String(fmt(hi)), 4, 10);
        ctx.fillText(String(fmt(lo)), 4, h - 18);
        ctx.fillText(String(start), 4, h - 4);
        const endText = String(stop - 1);
        ctx.fillText(endText, w - ctx.measureText(endText).width - 4, h - 4);
        foot.textContent =
          `c${colX.value} over [${start}, ${stop})` +
          (win > 1 ? ` · ${statSel.value} over ${win} frames` : "") +
          (data.truncated ? " · truncated" : "");
      };

      const refetch = async () => {
        const mySeq = ++seq;
        const [start, stop] = range();
        const cols = modeSel.value === "path"
          ? `${Number(colX.value) || 0},${Number(colY.value) || 0}`
          : String(Number(colX.value) || 0);
        meta.textContent = `${channel} · loading [${start}, ${stop})…`;
        try {
          data = await store.api(
            `/api/node/${nodeId}/series/${encodeURIComponent(channel)}` +
            `?cols=${cols}&start=${start}&stop=${stop}`);
          if (mySeq !== seq) return;
          meta.textContent = `${channel} @ ${nodeLabel} · ${data.stop - data.start} frames`;
          draw();
        } catch (err) {
          if (mySeq === seq) meta.textContent = String(err);
        }
      };

      canvas.addEventListener("click", (e) => {
        if (!data || modeSel.value === "path") return;
        const r = canvas.getBoundingClientRect();
        const count = data.stop - data.start;
        const i = Math.round(((e.clientX - r.left) / r.width) * (count - 1));
        store.setFrame(data.start + Math.max(0, Math.min(i, count - 1)));
      });

      modeSel.addEventListener("change", () => { paintControls(); refetch(); });
      colX.addEventListener("change", refetch);
      colY.addEventListener("change", refetch);
      winInput.addEventListener("change", () => { paintControls(); draw(); });
      statSel.addEventListener("change", draw);

      const offRange = store.onRange(refetch);
      const offFrame = store.onFrame(draw); // marker only: no refetch
      panel.onRemove = () => { offRange(); offFrame(); };
      paintControls();
      refetch();
    },
  };
}
