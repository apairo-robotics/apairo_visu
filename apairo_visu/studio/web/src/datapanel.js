// Panel content builders: pipeline graph, node inspector, data views and
// frame metrics. Each returns a spec for PanelManager.add(); every panel
// pulls what it needs from the shared store and cleans up via onRemove.

import { fillSelect } from "./panels.js";
import * as store from "./store.js";
import { Viewer } from "./engine/viewer.js";
import {
  cloudColorMapper, cloudColorValues, cloudFit, columnStats, columnStatsLine,
  drawCloud, drawHist, drawImage, drawRaster, fmt, previewKind, statsLine,
  uniqueCounts, valuesText,
} from "./views.js";

export const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const accent = () => getComputedStyle(document.documentElement).getPropertyValue("--accent");
const muted = () => getComputedStyle(document.documentElement).getPropertyValue("--muted");

/* ------------------------------------------------------------- pipeline */

// Adopts the server-rendered SVG (SSR-lite), binds pan/zoom and node clicks.
export function pipelineSpec(svg) {
  return {
    key: "pipeline",
    tag: "GRAPH",
    title: "pipeline",
    build(panel) {
      panel.body.classList.add("graph-body");
      panel.body.appendChild(svg);
      panel.body.appendChild(el("div", "hint", "drag to pan · scroll to zoom · click a node"));
      let suppressClick = false;

      const vb = svg.viewBox.baseVal;
      const toWorld = (e) => {
        const r = svg.getBoundingClientRect();
        return {
          x: vb.x + ((e.clientX - r.left) / r.width) * vb.width,
          y: vb.y + ((e.clientY - r.top) / r.height) * vb.height,
        };
      };
      svg.addEventListener("wheel", (e) => {
        e.preventDefault();
        const k = e.deltaY > 0 ? 1.12 : 1 / 1.12;
        const p = toWorld(e);
        vb.x = p.x - (p.x - vb.x) * k;
        vb.y = p.y - (p.y - vb.y) * k;
        vb.width *= k;
        vb.height *= k;
      }, { passive: false });
      let drag = null;
      svg.addEventListener("pointerdown", (e) => {
        drag = { x: e.clientX, y: e.clientY, vx: vb.x, vy: vb.y, moved: false };
        svg.setPointerCapture(e.pointerId);
      });
      svg.addEventListener("pointermove", (e) => {
        if (!drag) return;
        if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) drag.moved = true;
        const r = svg.getBoundingClientRect();
        vb.x = drag.vx - ((e.clientX - drag.x) / r.width) * vb.width;
        vb.y = drag.vy - ((e.clientY - drag.y) / r.height) * vb.height;
      });
      svg.addEventListener("pointerup", () => {
        suppressClick = Boolean(drag && drag.moved);
        drag = null;
      });

      svg.querySelectorAll("[data-node]").forEach((g) => {
        g.addEventListener("click", (e) => {
          e.stopPropagation();
          if (suppressClick) { suppressClick = false; return; }
          store.setSelected(g.dataset.node);
        });
      });
      panel.onRemove = store.onSelect((nodeId) => {
        svg.querySelectorAll(".node.selected").forEach((g) => g.classList.remove("selected"));
        const g = svg.querySelector(`[data-node="${nodeId}"]`);
        if (g) g.classList.add("selected");
      });
    },
  };
}

/* ------------------------------------------------------------ inspector */

function kvTable(rows) {
  const table = el("table", "kv");
  for (const [key, value] of rows) {
    const tr = el("tr");
    tr.appendChild(el("th", null, key));
    tr.appendChild(el("td", /^[\d\s,./×-]+$/.test(String(value)) ? "num" : null, String(value)));
    table.appendChild(tr);
  }
  return table;
}

function section(title, body) {
  const wrap = el("div", "insp-section");
  wrap.appendChild(el("h3", null, title));
  wrap.appendChild(body);
  return wrap;
}

// openData / openSeries (nodeId, label, channel, len) -> open a panel.
export function inspectorSpec(openData, openSeries) {
  return {
    key: "inspector",
    tag: "INFO",
    title: "inspector",
    build(panel) {
      panel.body.classList.add("scroll-body");
      const render = async (nodeId) => {
        if (!nodeId) {
          panel.body.replaceChildren(el("div", "placeholder", "Click a node to inspect it."));
          return;
        }
        let detail;
        try {
          detail = await store.detailAt(nodeId);
        } catch (err) {
          panel.body.replaceChildren(el("div", "error", String(err)));
          return;
        }
        if (store.getSelected() !== nodeId) return;
        panel.body.replaceChildren();

        const head = el("div", "insp-head");
        head.appendChild(el("h2", null, detail.label));
        head.appendChild(el("span", `kind-badge ${detail.kind}`, detail.kind));
        panel.body.appendChild(head);
        if (detail.error) panel.body.appendChild(el("div", "error", detail.error));

        const params = Object.entries(detail.params || {});
        if (detail.len !== undefined && !params.some(([k]) => k === "frames")) {
          params.unshift(["frames", detail.len]);
        }
        if (detail.kind === "transform") {
          if (detail.owner) params.push(["registered on", detail.owner]);
          if (detail.callable) params.push(["callable", detail.callable]);
          if (detail.signature) params.push(["signature", detail.signature]);
        }
        if (params.length) panel.body.appendChild(section("parameters", kvTable(params)));

        if (detail.channels) {
          const table = el("table", "kv");
          const head2 = el("tr");
          for (const h of ["channel", "dtype", "shape", ""]) head2.appendChild(el("th", null, h));
          table.appendChild(head2);
          for (const ch of detail.channels) {
            const tr = el("tr");
            if (ch.error) {
              const td = el("td", "error", ch.error);
              td.colSpan = 4;
              tr.appendChild(td);
              table.appendChild(tr);
              continue;
            }
            tr.appendChild(el("td", null, ch.key));
            tr.appendChild(el("td", null, ch.dtype));
            tr.appendChild(el("td", "num", ch.shape ? ch.shape.join(" × ") : "—"));
            const td = el("td");
            const open = el("button", "tbtn", "open");
            open.title = `Open ${ch.key} in a data panel`;
            open.addEventListener("click", () =>
              openData(nodeId, detail.label, ch.key, detail.len));
            const plot = el("button", "tbtn", "plot");
            plot.title = `Plot ${ch.key} across frames (series panel)`;
            plot.addEventListener("click", () =>
              openSeries(nodeId, detail.label, ch.key, detail.len, ch.shape));
            const designate = el("button", "tbtn", "try");
            designate.title = `Designate ${ch.key} as the catalog try target`;
            designate.addEventListener("click", () => store.setTarget({
              nodeId, nodeLabel: detail.label, channel: ch.key, len: detail.len,
              index: Math.max(0, Math.min(store.getFrame(), (detail.len ?? 1) - 1)),
            }));
            td.append(open, plot, designate);
            tr.appendChild(td);
            table.appendChild(tr);
          }
          panel.body.appendChild(
            section("channels — open: data panel · plot: series · try: catalog target", table));
        }

        if (detail.sequences && detail.sequences.length) {
          const table = el("table", "kv");
          const headRow = el("tr");
          for (const h of ["sequence", "frames", "range", ""]) {
            headRow.appendChild(el("th", null, h));
          }
          table.appendChild(headRow);
          for (const s of detail.sequences) {
            const tr = el("tr");
            tr.appendChild(el("td", null, s.id));
            tr.appendChild(el("td", "num", String(s.stop - s.start)));
            tr.appendChild(el("td", "num", `${s.start} – ${s.stop - 1}`));
            const td = el("td");
            const viewBtn = el("button", "tbtn", "view");
            viewBtn.title = `Restrict the frame slider to sequence ${s.id}`;
            viewBtn.addEventListener("click", () => {
              store.setRange({ start: s.start, stop: s.stop, label: s.id });
              store.setFrame(s.start);
            });
            td.appendChild(viewBtn);
            tr.appendChild(td);
            table.appendChild(tr);
          }
          panel.body.appendChild(
            section("sequences — view: restrict the frame slider", table));
        }
        if (detail.doc) panel.body.appendChild(section("docstring", el("pre", "doc", detail.doc)));
      };
      panel.onRemove = store.onSelect(render);
      render(store.getSelected());
    },
  };
}

/* ----------------------------------------------------------- data panel */

const VIEW_MODES = [
  ["auto", "auto"], ["cloud", "BEV"], ["cloud3d", "3D"], ["image", "image"],
  ["raster", "raster"], ["hist", "histogram"], ["values", "values"],
];

// Feed an (N, C>=3) per-point array to the engine viewer: positions from
// cols 0-2 (NaN rows dropped), colors from a cloudColorValues spec (own
// column or a per-point label channel) — the same rules the 2D BEV uses.
// Streaming options keep the camera and skip the octree; the caller frames
// once and builds the octree when the frame slider pauses.
function feedCloud3d(viewer, arr, spec) {
  const [n, stride] = arr.shape;
  const toColor = cloudColorMapper(spec);
  const pos = new Float32Array(n * 3);
  const rgb = new Float32Array(n * 3);
  const alpha = new Float32Array(n).fill(1);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const x = Number(arr.data[i * stride]);
    const y = Number(arr.data[i * stride + 1]);
    const z = Number(arr.data[i * stride + 2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    pos[k * 3] = x; pos[k * 3 + 1] = y; pos[k * 3 + 2] = z;
    const c = toColor(spec.values[i]);
    rgb[k * 3] = c[0] / 255; rgb[k * 3 + 1] = c[1] / 255; rgb[k * 3 + 2] = c[2] / 255;
    k++;
  }
  viewer.setCloud(pos.subarray(0, k * 3), { octree: false, frame: false });
  viewer.setColors(rgb.subarray(0, k * 3), alpha.subarray(0, k));
  return k;
}

export function dataSpec(binding) {
  const { nodeId, nodeLabel, channel, len } = binding;
  return {
    key: `data:${nodeId}:${channel}`,
    tag: "DATA",
    title: `${channel} @ ${nodeLabel}`,
    binding,
    build(panel) {
      panel.body.classList.add("data-body");
      const meta = el("div", "chan-values");
      const stage = el("div", "data-stage");
      const foot = el("div", "chan-values");
      panel.body.append(meta, stage, foot);

      const modeSel = el("select", "chan-color");
      fillSelect(modeSel, VIEW_MODES);
      const colorSel = el("select", "chan-color");
      const bgrBtn = el("button", "tbtn", localStorage.getItem("studio.bgr") === "1" ? "BGR" : "RGB");
      bgrBtn.title = "Swap the red and blue channels (rosbag images are BGR)";
      const sizeInput = el("input", "opt-input frame-input");
      sizeInput.type = "number";
      sizeInput.min = "0.5";
      sizeInput.step = "0.5";
      sizeInput.value = localStorage.getItem("studio.ptsize") || "2";
      sizeInput.title = "Point size (px)";
      // Per-panel frame: type an index to diverge from the global slider.
      // Moving the global slider resynchronizes every panel EXCEPT the
      // locked ones -- lock to compare two moments side by side.
      const frameInput = el("input", "opt-input frame-input");
      frameInput.type = "number";
      frameInput.min = "0";
      frameInput.max = String(Math.max(0, (len ?? 1) - 1));
      frameInput.title = "This panel's frame (lock to survive global slider moves)";
      const lockBtn = el("button", "tbtn");
      let local = null;   // panel-specific frame index, null = global
      let locked = false;
      const paintLock = () => {
        lockBtn.textContent = locked ? "locked" : "follow";
        lockBtn.title = locked
          ? "Locked on its own frame — the global slider does not move this panel"
          : "Follows the global slider — set a frame and lock to keep it";
      };
      paintLock();
      // Designates THIS data (node+channel+current sample) as the target
      // a catalog try applies to.
      const tryBtn = el("button", "tbtn", "try");
      tryBtn.title = "Try a transform on this data (designates it as the catalog target)";
      tryBtn.addEventListener("click", () => {
        store.setTarget({
          nodeId, nodeLabel, channel, len,
          index: local ?? Math.max(0, Math.min(store.getFrame(), (len ?? 1) - 1)),
        });
      });
      panel.controls.append(
        modeSel, colorSel, bgrBtn, sizeInput, frameInput, lockBtn, tryBtn);

      let arr = null;
      // Channels of the current sample: the color select offers per-point
      // siblings (labels) and cloudColorValues resolves them.
      let chans = null;
      // BEV zoom state: null = auto-fit. Survives frame changes so a zoom
      // holds while scrubbing; double-click resets it.
      let bevView = null;

      const colorSpec = () => {
        const fallback = String(Math.min(2, arr.shape[1] - 1));
        return cloudColorValues(arr, colorSel.value || fallback, chans)
            ?? cloudColorValues(arr, fallback, chans);
      };
      const colorFoot = (spec) => (spec.label.startsWith("col ")
        ? columnStatsLine(arr, Number(spec.label.slice(4)), spec.label)
        : `colored by ${spec.label}`);

      // Color options: the cloud's own columns plus per-point sibling
      // channels of the same length (labels colors categorically). Row
      // counts stay aligned under decimation: same cap, same stride.
      const fillColorOptions = () => {
        if (!arr || !arr.shape || arr.shape.length !== 2) return;
        const opts = [...Array(Math.min(arr.shape[1], 8)).keys()]
          .map((c) => [String(c), `color: col ${c}${c === 2 ? " (z)" : ""}`]);
        for (const [name, ch] of Object.entries(chans || {})) {
          if (name === channel || !ch || !ch.shape) continue;
          if (ch.shape.length > 2 || ch.shape[0] !== arr.shape[0]) continue;
          if (ch.shape.length === 2 && ch.shape[1] > 4) continue;
          opts.push([`ch:${name}`, `color: ${name}`]);
        }
        const prev = colorSel.value;
        fillSelect(colorSel, opts);
        colorSel.value = opts.some(([v]) => v === prev)
          ? prev
          : String(Math.min(2, arr.shape[1] - 1));
      };

      // Wheel = zoom at the cursor, drag = rubber-band zoom to the box,
      // shift-drag = pan, double-click = reset to the auto fit.
      const bindBev = (canvas, spec) => {
        const w = canvas.width, h = canvas.height;
        const redraw = () =>
          drawCloud(canvas, arr, spec, bevView, Number(sizeInput.value) || 1.7);
        const view = () => bevView ?? cloudFit(arr, w, h);
        const toPx = (e) => {
          const r = canvas.getBoundingClientRect();
          return {
            x: ((e.clientX - r.left) / r.width) * w,
            y: ((e.clientY - r.top) / r.height) * h,
          };
        };
        const toWorld = (p, v) => ({
          x: v.cx + (p.x - w / 2) / v.scale,
          y: v.cy - (p.y - h / 2) / v.scale,
        });

        canvas.addEventListener("wheel", (e) => {
          e.preventDefault();
          const v = view();
          if (!v) return;
          const p = toPx(e);
          const wpt = toWorld(p, v);
          const scale = v.scale * (e.deltaY > 0 ? 1 / 1.2 : 1.2);
          // Keep the world point under the cursor fixed while scaling.
          bevView = {
            cx: wpt.x - (p.x - w / 2) / scale,
            cy: wpt.y + (p.y - h / 2) / scale,
            scale,
          };
          redraw();
        }, { passive: false });

        let drag = null;
        canvas.addEventListener("pointerdown", (e) => {
          const v = view();
          if (!v || e.button !== 0) return;
          canvas.setPointerCapture(e.pointerId);
          const p = toPx(e);
          drag = e.shiftKey
            ? { pan: true, px: p, view: v }
            : { pan: false, px: p, to: p,
                snap: canvas.getContext("2d").getImageData(0, 0, w, h) };
        });
        canvas.addEventListener("pointermove", (e) => {
          if (!drag) return;
          const p = toPx(e);
          if (drag.pan) {
            bevView = {
              cx: drag.view.cx - (p.x - drag.px.x) / drag.view.scale,
              cy: drag.view.cy + (p.y - drag.px.y) / drag.view.scale,
              scale: drag.view.scale,
            };
            redraw();
            return;
          }
          drag.to = p;
          const ctx = canvas.getContext("2d");
          ctx.putImageData(drag.snap, 0, 0);
          ctx.strokeStyle = accent();
          ctx.setLineDash([4, 3]);
          ctx.strokeRect(drag.px.x, drag.px.y, p.x - drag.px.x, p.y - drag.px.y);
          ctx.setLineDash([]);
        });
        canvas.addEventListener("pointerup", () => {
          if (!drag) return;
          const d = drag;
          drag = null;
          if (d.pan) return;
          const v = view();
          const bw = Math.abs(d.to.x - d.px.x), bh = Math.abs(d.to.y - d.px.y);
          if (!v || bw < 8 || bh < 8) { redraw(); return; } // a click, not a box
          const a = toWorld(d.px, v), b = toWorld(d.to, v);
          bevView = {
            cx: (a.x + b.x) / 2,
            cy: (a.y + b.y) / 2,
            scale: 0.95 * Math.min(w / Math.abs(b.x - a.x), h / Math.abs(b.y - a.y)),
          };
          redraw();
        });
        canvas.addEventListener("dblclick", () => { bevView = null; redraw(); });
      };

      // 3D viewer: created on first use, kept alive across frames of the
      // stream, torn down when the mode leaves 3D or the panel closes.
      let v3d = null;
      let v3dTimer = null;
      const drop3d = () => {
        if (v3dTimer) { clearTimeout(v3dTimer); v3dTimer = null; }
        if (v3d) { v3d.viewer.dispose(); v3d = null; }
      };
      const ensure3d = () => {
        if (v3d) return v3d;
        const host = el("div", "cloud3d-host");
        const viewer = new Viewer(host, { lodBudget: 500000 });
        viewer.setBackground(0x14161c);
        v3d = { host, viewer, framed: false };
        return v3d;
      };

      const drawAs = (kind) => {
        if (kind !== "cloud3d") drop3d();
        stage.replaceChildren();
        foot.textContent = "";
        colorSel.hidden = kind !== "cloud" && kind !== "cloud3d";
        sizeInput.hidden = kind !== "cloud" && kind !== "cloud3d";
        bgrBtn.hidden = kind !== "image";
        if (!arr) { stage.appendChild(el("div", "placeholder", "no data")); return; }
        if (arr.repr !== undefined) { stage.appendChild(el("div", "chan-values", arr.repr)); return; }
        try {
          if (kind === "cloud3d" && arr.shape.length === 2 && arr.shape[1] >= 3) {
            const inst = ensure3d();
            stage.appendChild(inst.host);
            inst.viewer.resize(); // the host just got its size from the stage
            inst.viewer.setPointSize(Number(sizeInput.value) || 2);
            const spec = colorSpec();
            const k = feedCloud3d(inst.viewer, arr, spec);
            if (!inst.framed && k > 0) { inst.framed = true; inst.viewer.frame(); }
            // Frame slider paused → octree for exact-fast picking + LOD cut.
            if (v3dTimer) clearTimeout(v3dTimer);
            v3dTimer = setTimeout(() => { if (v3d) v3d.viewer.buildOctree(); }, 400);
            foot.textContent = colorFoot(spec) +
              (arr.fullRows ? `  ·  sample of ${arr.fullRows} pts` : "");
          } else if (kind === "cloud" && arr.shape.length === 2) {
            const canvas = el("canvas", "chan-canvas");
            canvas.width = 640;
            canvas.height = 480;
            stage.appendChild(canvas);
            const spec = colorSpec();
            drawCloud(canvas, arr, spec, bevView, Number(sizeInput.value) || 1.7);
            bindBev(canvas, spec);
            stage.appendChild(el("div", "hint",
              "scroll: zoom · drag: zoom to box · shift-drag: pan · double-click: reset"));
            foot.textContent = colorFoot(spec);
          } else if (kind === "image" && arr.shape.length === 3) {
            const canvas = el("canvas", "chan-canvas img");
            stage.appendChild(canvas);
            drawImage(canvas, arr, localStorage.getItem("studio.bgr") === "1");
          } else if (kind === "raster" && arr.shape.length === 2) {
            const canvas = el("canvas", "chan-canvas img");
            stage.appendChild(canvas);
            drawRaster(canvas, arr);
          } else if (kind === "hist") {
            const canvas = el("canvas", "chan-canvas");
            canvas.width = 640;
            canvas.height = 160;
            stage.appendChild(canvas);
            drawHist(canvas, arr, accent(), muted());
            foot.textContent = statsLine(arr);
          } else {
            stage.appendChild(el("div", "chan-values", valuesText(arr)));
          }
        } catch (err) {
          stage.replaceChildren(el("div", "error", `cannot draw as ${kind}: ${err}`));
        }
      };

      const currentKind = () => {
        const mode = modeSel.value;
        if (mode !== "auto") return mode;
        return arr && arr.shape ? previewKind(arr.shape, arr.dtype) : "values";
      };

      // Async timelines: the bound channel may be absent at the current
      // frame. Hold the latest earlier frame that carries it, and say how
      // old it is relative to the current frame (timestamps when the
      // dataset has them, frame distance otherwise).
      const holdLast = async (sample) => {
        try {
          const { indices } = await store.channelFrames(nodeId, channel);
          if (!indices.length) return null;
          const at = indices[
            (() => {
              let lo = 0, hi = indices.length - 1, best = -1;
              while (lo <= hi) {
                const m = (lo + hi) >> 1;
                if (indices[m] <= sample.index) { best = m; lo = m + 1; }
                else hi = m - 1;
              }
              return best;
            })()
          ];
          if (at === undefined) return null;
          const held = await store.sampleOne(nodeId, at, channel);
          if (!held.arr) return null;
          const age = held.timestamp != null && sample.timestamp != null
            ? `${(sample.timestamp - held.timestamp).toFixed(3)} s old`
            : `${sample.index - held.index} frames old`;
          return { arr: held.arr, note: ` · held from frame ${held.index} (${age})` };
        } catch {
          return null; // transform node or no timeline: plain "channel absent"
        }
      };

      let refreshSeq = 0;
      const refresh = async () => {
        const mySeq = ++refreshSeq;
        try {
          const sample = await store.sampleAt(nodeId, len, local);
          let held = null;
          if ((sample.channels[channel] ?? null) === null) {
            held = await holdLast(sample);
          }
          if (mySeq !== refreshSeq) return;
          frameInput.value = String(sample.index);
          arr = sample.channels[channel] ?? held?.arr ?? null;
          chans = held ? { ...sample.channels, [channel]: held.arr } : sample.channels;
          // Frame provenance: the sequence this global index falls in, and
          // the event's channel-relative row (async timelines) -- `frame
          // 503961 (seq_b · camera 2)` reads as "camera frame 2 of seq_b".
          const ref = sample.frame;
          const rowInfo = ref?.channel != null && ref?.row != null
            ? ` · ${ref.channel} ${ref.row}` : "";
          const prov = ref?.sequence ? ` (${ref.sequence}${rowInfo})` : "";
          meta.textContent = arr && arr.shape
            ? `frame ${sample.index}${prov} · ${arr.dtype} · ${arr.shape.join(" × ")}` +
              (arr.fullRows ? ` (showing ${arr.shape[0]} of ${arr.fullRows})` : "") +
              (held ? held.note : "")
            : `frame ${sample.index}${prov} · channel absent`;
          fillColorOptions();
          drawAs(currentKind());
        } catch (err) {
          if (mySeq === refreshSeq) {
            stage.replaceChildren(el("div", "error", String(err)));
          }
        }
      };

      modeSel.addEventListener("change", () => drawAs(currentKind()));
      colorSel.addEventListener("change", () => drawAs(currentKind()));
      bgrBtn.addEventListener("click", () => {
        const bgr = localStorage.getItem("studio.bgr") !== "1";
        localStorage.setItem("studio.bgr", bgr ? "1" : "0");
        bgrBtn.textContent = bgr ? "BGR" : "RGB";
        drawAs(currentKind());
      });
      sizeInput.addEventListener("change", () => {
        localStorage.setItem("studio.ptsize", sizeInput.value);
        drawAs(currentKind());
      });
      frameInput.addEventListener("change", () => {
        local = Math.max(0, Math.min(Number(frameInput.value) || 0, (len ?? 1) - 1));
        frameInput.value = String(local);
        refresh();
      });
      lockBtn.addEventListener("click", () => {
        locked = !locked;
        if (locked && local === null) {
          local = Math.max(0, Math.min(store.getFrame(), (len ?? 1) - 1));
        }
        if (!locked) {
          local = null; // unlocking resynchronizes on the global frame
          refresh();
        }
        paintLock();
      });

      // The global slider resynchronizes every panel except the locked
      // ones (a typed-in frame diverges only until the next global move).
      const offFrame = store.onFrame(() => {
        if (locked) return;
        local = null;
        refresh();
      });
      panel.onRemove = () => {
        offFrame();
        drop3d();
      };
      refresh();
    },
  };
}

/* -------------------------------------------------------------- metrics */

// Frame metrics for the SELECTED node: per-channel column stats, and the
// value distribution for label-like integer channels. All computed
// client-side on the decoded arrays.
export function metricsSpec() {
  return {
    key: "metrics",
    tag: "METRICS",
    title: "frame metrics",
    build(panel) {
      panel.body.classList.add("scroll-body");
      let seq = 0;

      const render = async () => {
        const nodeId = store.getSelected();
        const mySeq = ++seq;
        if (!nodeId) {
          panel.body.replaceChildren(el("div", "placeholder", "Select a node."));
          return;
        }
        let detail, sample;
        try {
          detail = await store.detailAt(nodeId);
          if (!detail.len) {
            panel.body.replaceChildren(el("div", "placeholder", "No frames at this node."));
            return;
          }
          sample = await store.sampleAt(nodeId, detail.len);
        } catch (err) {
          panel.body.replaceChildren(el("div", "error", String(err)));
          return;
        }
        if (mySeq !== seq) return;
        panel.body.replaceChildren();
        panel.body.appendChild(
          el("div", "chan-values", `${detail.label} · frame ${sample.index}`));

        for (const [name, arr] of Object.entries(sample.channels)) {
          if (!arr || arr.repr !== undefined) continue;
          const table = el("table", "kv");

          const isMatrix = arr.shape.length === 2 && arr.shape[1] <= 32;
          if (isMatrix) {
            const head = el("tr");
            for (const h of ["col", "min", "mean", "max"]) head.appendChild(el("th", null, h));
            table.appendChild(head);
            for (let c = 0; c < Math.min(arr.shape[1], 8); c++) {
              const s = columnStats(arr, c);
              if (!s) continue;
              const tr = el("tr");
              tr.appendChild(el("td", null, String(c)));
              for (const v of [s.min, s.mean, s.max]) tr.appendChild(el("td", "num", String(fmt(v))));
              table.appendChild(tr);
            }
          } else {
            const s = columnStats(arr);
            if (s) {
              for (const [k, v] of [["min", s.min], ["mean", s.mean], ["max", s.max]]) {
                const tr = el("tr");
                tr.appendChild(el("th", null, k));
                tr.appendChild(el("td", "num", String(fmt(v))));
                table.appendChild(tr);
              }
              if (s.nan) {
                const tr = el("tr");
                tr.appendChild(el("th", null, "non-finite"));
                tr.appendChild(el("td", "num", String(s.nan)));
                table.appendChild(tr);
              }
            }
          }

          // Label-like channels: show the class distribution.
          if (arr.shape.length === 1 && arr.dtype.startsWith("uint")) {
            const counts = uniqueCounts(arr);
            if (counts) {
              const total = arr.data.length;
              for (const [value, count] of counts) {
                const tr = el("tr");
                tr.appendChild(el("th", null, `label ${value}`));
                tr.appendChild(el("td", "num",
                  `${count} (${((100 * count) / total).toFixed(1)}%)`));
                table.appendChild(tr);
              }
            }
          }
          panel.body.appendChild(section(`${name} — ${arr.shape.join(" × ")}`, table));
        }
      };

      const unsubFrame = store.onFrame(render);
      const unsubSelect = store.onSelect(render);
      panel.onRemove = () => { unsubFrame(); unsubSelect(); };
      render();
    },
  };
}
