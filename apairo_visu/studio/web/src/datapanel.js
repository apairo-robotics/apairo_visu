// Panel content builders: pipeline graph, node inspector, data views and
// frame metrics. Each returns a spec for PanelManager.add(); every panel
// pulls what it needs from the shared store and cleans up via onRemove.

import { fillSelect } from "./panels.js";
import * as store from "./store.js";
import { Viewer } from "./engine/viewer.js";
import {
  cloudColorMapper, cloudColorValues, cloudFit, columnStats, columnStatsLine,
  drawCloud, drawHist, drawSource, fmt, imageCanvas, previewKind,
  rasterCanvas, sourceFit, statsLine, uniqueCounts, valuesText,
} from "./views.js";

export const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const accent = () => getComputedStyle(document.documentElement).getPropertyValue("--accent");
const muted = () => getComputedStyle(document.documentElement).getPropertyValue("--muted");

/* ------------------------------------------------------- view settings */
// How clouds are *looked at* -- background, point shape and size, ground
// grid, camera style. Shared by every cloud panel and remembered across
// sessions, because it is a property of the viewer's eyes, not of one
// channel: setting a readable background once should not have to be redone
// for the next panel.

const VIEW_KEY = "studio.view";
// The two stage backgrounds worth naming: near-black for a dark page, a
// pale slate for a light one. Anything else is a colour the user picked.
const STAGE_DARK = "#14161c";
const STAGE_LIGHT = "#e6eaf1";
const VIEW_DEFAULTS = {
  // "auto" tracks the page theme, "dark"/"light" pin one, else a #rrggbb.
  background: "auto",
  round: true,
  attenuate: false,   // point size in metres (shrinks with distance) vs pixels
  size: 2,
  controls: "trackball",
  grid: true,
};

export const viewSettings = { ...VIEW_DEFAULTS };
try {
  Object.assign(viewSettings, JSON.parse(localStorage.getItem(VIEW_KEY) || "{}"));
  // Point size used to live on its own key; carry it over rather than
  // silently resetting a size the user had already dialled in.
  const legacy = Number(localStorage.getItem("studio.ptsize"));
  if (!localStorage.getItem(VIEW_KEY) && legacy > 0) viewSettings.size = legacy;
} catch { /* corrupt entry: the defaults stand */ }

const viewListeners = new Set();

export function onViewSettings(fn) {
  viewListeners.add(fn);
  return () => viewListeners.delete(fn);
}

export function setViewSettings(patch) {
  Object.assign(viewSettings, patch);
  try {
    localStorage.setItem(VIEW_KEY, JSON.stringify(viewSettings));
  } catch { /* private mode: the session still honours the change */ }
  for (const fn of viewListeners) fn(viewSettings);
}

// A near-black stage under a light UI is what makes a viridis cloud hard to
// read: dark points on a dark plane. Following the page theme keeps the
// contrast the colormap was designed for, whichever theme is on.
function themeBackground() {
  const set = document.documentElement.dataset.theme;
  const dark = set === "dark"
    || (!set && window.matchMedia("(prefers-color-scheme: dark)").matches);
  return dark ? STAGE_DARK : STAGE_LIGHT;
}

export function backgroundColor() {
  const choice = viewSettings.background;
  if (choice === "auto") return themeBackground();
  if (choice === "dark") return STAGE_DARK;
  if (choice === "light") return STAGE_LIGHT;
  return choice;
}

// Push the current settings onto a viewer. Re-applied after every setCloud:
// the engine rebuilds the ground grid with the cloud, so a hidden grid would
// reappear on the next frame otherwise.
function applyViewSettings(viewer) {
  viewer.setBackground(Number.parseInt(backgroundColor().slice(1), 16));
  viewer.setRound(viewSettings.round);
  viewer.setSizeAttenuation(viewSettings.attenuate);
  viewer.setPointSize(viewSettings.size);
  viewer.setControlStyle(viewSettings.controls);
  // Reaching for the overlay: the shared engine exposes no grid controls, so
  // drive its uniforms from here. All of it optional -- a future engine
  // without a ReferenceGrid simply ignores us rather than throwing.
  const mesh = viewer.grid && viewer.grid.mesh;
  if (mesh) {
    mesh.visible = viewSettings.grid;
    // The engine's grid is tuned for its dark stage; left as-is on a light
    // background those near-navy lines dominate the cloud they exist to
    // give depth to. Pick the pair that reads as "quiet floor" either way.
    const u = mesh.material && mesh.material.uniforms;
    if (u) {
      const light = isLight(backgroundColor());
      u.uMinorColor.value.set(light ? 0xb9c1ce : 0x333c4d);
      u.uMajorColor.value.set(light ? 0x93a0b4 : 0x5a6478);
      u.uOpacity.value = light ? 0.5 : 0.35;
    }
  }
  viewer.requestRender();
}

// Perceived lightness of a #rrggbb, the cheap Rec.601 way.
function isLight(hex) {
  const v = Number.parseInt(hex.slice(1), 16);
  const r = (v >> 16) & 255, g = (v >> 8) & 255, b = v & 255;
  return 0.299 * r + 0.587 * g + 0.114 * b > 140;
}

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

// openData / openSeries / openEntries (nodeId, label, channel, len, shape)
// -> open a panel.
export function inspectorSpec(openData, openSeries, openEntries) {
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
            const list = el("button", "tbtn", "list");
            list.title = `List the frames of ${ch.key} (file, index, size)`;
            list.addEventListener("click", () =>
              openEntries(nodeId, detail.label, ch.key, detail.len));
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
            td.append(open, list, plot, designate);
            tr.appendChild(td);
            table.appendChild(tr);
          }
          panel.body.appendChild(section(
            "channels — open: data · list: frames · plot: series · try: catalog",
            table));
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
      const gearBtn = el("button", "tbtn", "view");
      gearBtn.title = "Background, point shape and size, grid, camera";
      // This panel's frame, counted in ITS channel's own frames rather than
      // in the interleaved global index: a lidar panel sits at "lidar
      // 000850", and 000850 is the number a label file is named after. The
      // global index is an artefact of how the events happen to be merged
      // -- it even shifts when a label is written to another channel --
      // so nobody can act on it. Type a frame and press Enter to seek;
      // where that seek lands is the sync checkbox's business.
      const frameInput = el("input", "opt-input frame-input");
      frameInput.type = "text";
      frameInput.title =
        "This panel's frame, in this channel's own frames (the file stem, "
        + "e.g. 000850). Type one and press Enter to seek.";
      // "sync": this panel and the global timeline drive each OTHER. Ticked
      // (the default), seeking here moves the global frame -- and with it
      // the topbar slider and every other synced panel -- and a global move
      // brings this panel along. Unticked, the panel is an island: seek it
      // freely without disturbing anything, then tick sync again to bring
      // everyone onto the frame you found.
      const syncBox = el("input");
      syncBox.type = "checkbox";
      syncBox.checked = true;
      const syncLabel = el("label", "sync-toggle on", "sync");
      syncLabel.prepend(syncBox);
      syncLabel.title =
        "Synced: seeking here moves the global timeline and every other "
        + "synced panel, and follows it back. Unticked: this panel alone moves.";
      let local = null;   // panel-specific frame index, null = follows global
      const synced = () => syncBox.checked;
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
        modeSel, colorSel, bgrBtn, gearBtn, frameInput, syncLabel, tryBtn);

      // View settings, as a popover over the stage rather than yet more
      // header controls: they are set once and then left alone, unlike the
      // frame and the colour, which are handled every few seconds.
      const settings = el("div", "settings-pop");
      settings.hidden = true;
      panel.body.appendChild(settings);
      const only3d = [];

      const optRow = (label, control) => {
        const row = el("label", "opt");
        row.append(el("span", null, label), control);
        settings.appendChild(row);
        return control;
      };
      const check = (label, key, onChange) => {
        const box = el("input");
        box.type = "checkbox";
        box.checked = Boolean(viewSettings[key]);
        box.addEventListener("change", () => {
          setViewSettings({ [key]: box.checked });
          if (onChange) onChange();
        });
        return optRow(label, box);
      };

      const sizeInput = el("input", "opt-input");
      sizeInput.type = "number";
      sizeInput.min = "0.5";
      sizeInput.step = "0.5";
      sizeInput.value = String(viewSettings.size);
      optRow("point size", sizeInput);
      sizeInput.addEventListener("change", () => {
        setViewSettings({ size: Math.max(0.1, Number(sizeInput.value) || 1) });
        drawAs(currentKind()); // the BEV draws its own points
      });

      only3d.push(check("round points", "round").closest(".opt"));
      only3d.push(check("size in metres", "attenuate").closest(".opt"));
      only3d.push(check("ground grid", "grid").closest(".opt"));

      // Named choices rather than a "use the theme" tickbox: that tickbox
      // only gated the colour picker, and since the picker already held the
      // theme's colour, toggling it changed precisely nothing on screen.
      // Every option here says what it does and does it.
      const bgSel = el("select", "chan-color");
      fillSelect(bgSel, [
        ["auto", "match page theme"],
        ["dark", "dark"],
        ["light", "light"],
        ["custom", "custom colour"],
      ]);
      const bgColor = el("input");
      bgColor.type = "color";
      const bgChoice = () =>
        (["auto", "dark", "light"].includes(viewSettings.background)
          ? viewSettings.background : "custom");
      const paintBackground = () => {
        bgSel.value = bgChoice();
        bgColor.value = backgroundColor();
        bgColor.disabled = bgSel.value !== "custom";
      };
      paintBackground();
      const bgWrap = el("span", "opt-pair");
      bgWrap.append(bgSel, bgColor);
      only3d.push(optRow("background", bgWrap).closest(".opt"));
      bgSel.addEventListener("change", () => {
        // Entering "custom" keeps whatever is on screen as the starting
        // colour, so the picker opens on the shade you are looking at.
        setViewSettings({
          background: bgSel.value === "custom" ? backgroundColor() : bgSel.value,
        });
        paintBackground();
      });
      bgColor.addEventListener("input", () => {
        setViewSettings({ background: bgColor.value });
        paintBackground();
      });

      const camSel = el("select", "chan-color");
      fillSelect(camSel, [["trackball", "trackball (free)"], ["orbit", "orbit (upright)"]]);
      camSel.value = viewSettings.controls;
      camSel.addEventListener("change", () => setViewSettings({ controls: camSel.value }));
      only3d.push(optRow("camera", camSel).closest(".opt"));

      const recenter = el("button", "tbtn", "frame cloud");
      recenter.title = "Bring the camera back onto the cloud";
      recenter.addEventListener("click", () => { if (v3d) v3d.viewer.frame(); });
      only3d.push(optRow("recenter", recenter).closest(".opt"));

      const paintSettings = () => {
        const is3d = currentKind() === "cloud3d";
        for (const row of only3d) row.hidden = !is3d;
        sizeInput.value = String(viewSettings.size);
        paintBackground();
        camSel.value = viewSettings.controls;
      };
      gearBtn.addEventListener("click", () => {
        settings.hidden = !settings.hidden;
        gearBtn.classList.toggle("primed", !settings.hidden);
        if (!settings.hidden) paintSettings();
      });
      // Another panel (or the theme toggle) changed the settings: follow.
      const offView = onViewSettings(() => {
        if (v3d) applyViewSettings(v3d.viewer);
        if (!settings.hidden) paintSettings();
      });

      let arr = null;
      // Channels of the current sample: the color select offers per-point
      // siblings (labels) and cloudColorValues resolves them.
      let chans = null;

      // This channel's own timeline: {indices, stems} over the global frame
      // axis, or false for a synchronous dataset (every frame carries every
      // channel, so the global index already IS the channel's frame) and for
      // transform nodes, which have no event timeline. Fetched once, cached
      // by the store.
      let track = null;
      const loadTrack = async () => {
        if (track !== null) return track;
        try {
          const t = await store.channelFrames(nodeId, channel);
          track = t.indices.length ? t : false;
        } catch {
          track = false;
        }
        return track;
      };
      // Position of the channel event at (or before) a global frame; -1 when
      // the channel has no event that early.
      const posOf = (indices, frame) => {
        let lo = 0, hi = indices.length - 1, best = -1;
        while (lo <= hi) {
          const m = (lo + hi) >> 1;
          if (indices[m] <= frame) { best = m; lo = m + 1; } else hi = m - 1;
        }
        return best;
      };
      // How this panel names a global frame: the file stem when the dataset
      // has one, the channel-relative row otherwise, null when the frame is
      // not one of this channel's events at all.
      const nameOf = (globalIndex) => {
        if (!track) return null;
        const k = posOf(track.indices, globalIndex);
        if (k < 0 || track.indices[k] !== globalIndex) return null;
        return track.stems ? track.stems[k] : String(k);
      };
      // The inverse, for what the user types: a stem (000850, or 850 without
      // the padding nobody types) and, failing that, a plain channel row.
      const seek = (t, text) => {
        const numeric = /^\d+$/.test(text);
        if (t.stems) {
          const k = t.stems.findIndex((s) => s === text ||
            (numeric && /^\d+$/.test(s) && Number(s) === Number(text)));
          if (k >= 0) return k;
        }
        if (numeric && Number(text) < t.indices.length) return Number(text);
        return -1;
      };
      // Zoom state per 2-D stage: null = auto-fit. Survives frame changes so
      // a zoom holds while scrubbing (watch one bush across a sequence);
      // double-click resets it. The BEV works in world metres, the image /
      // raster stage in source pixels, so they cannot share one state.
      let bevView = null;
      let imgView = null;

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

      // Wheel = zoom at the cursor, drag = rubber-band zoom to the box (or
      // pan when `box` is off), shift-drag = pan, double-click = reset to the
      // auto fit. One implementation for every 2-D stage: the BEV works in
      // world metres with y up, an image in source pixels with y down, and
      // both carry the same {cx, cy, scale} view -- `yUp` is the only
      // difference, so zooming into a corner of a camera frame is the same
      // gesture as zooming into a corner of a scan.
      const bindViewport = (canvas, ctl) => {
        const { fit, redraw, setView, yUp = true, box = true } = ctl;
        const w = canvas.width, h = canvas.height;
        const sy = yUp ? -1 : 1;
        const view = () => ctl.getView() ?? fit();
        const toPx = (e) => {
          const r = canvas.getBoundingClientRect();
          return {
            x: ((e.clientX - r.left) / r.width) * w,
            y: ((e.clientY - r.top) / r.height) * h,
          };
        };
        const toWorld = (p, v) => ({
          x: v.cx + (p.x - w / 2) / v.scale,
          y: v.cy + sy * (p.y - h / 2) / v.scale,
        });

        canvas.addEventListener("wheel", (e) => {
          e.preventDefault();
          const v = view();
          if (!v) return;
          const p = toPx(e);
          const wpt = toWorld(p, v);
          const scale = v.scale * (e.deltaY > 0 ? 1 / 1.2 : 1.2);
          // Keep the world point under the cursor fixed while scaling.
          setView({
            cx: wpt.x - (p.x - w / 2) / scale,
            cy: wpt.y - sy * (p.y - h / 2) / scale,
            scale,
          });
          redraw();
        }, { passive: false });

        let drag = null;
        canvas.addEventListener("pointerdown", (e) => {
          const v = view();
          if (!v || e.button !== 0) return;
          canvas.setPointerCapture(e.pointerId);
          const p = toPx(e);
          drag = e.shiftKey || !box
            ? { pan: true, px: p, view: v }
            : { pan: false, px: p, to: p,
                snap: canvas.getContext("2d").getImageData(0, 0, w, h) };
        });
        canvas.addEventListener("pointermove", (e) => {
          if (!drag) return;
          const p = toPx(e);
          if (drag.pan) {
            setView({
              cx: drag.view.cx - (p.x - drag.px.x) / drag.view.scale,
              cy: drag.view.cy - sy * (p.y - drag.px.y) / drag.view.scale,
              scale: drag.view.scale,
            });
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
          setView({
            cx: (a.x + b.x) / 2,
            cy: (a.y + b.y) / 2,
            scale: 0.95 * Math.min(w / Math.abs(b.x - a.x), h / Math.abs(b.y - a.y)),
          });
          redraw();
        });
        canvas.addEventListener("dblclick", () => { setView(null); redraw(); });
      };

      // The keyboard acts on the FOCUSED panel: click a panel to give it the
      // keys, and it keeps them until another panel is clicked -- the
      // pointer is then free to go anywhere (a control, another panel, off
      // the window) without the camera changing owner mid-gesture. The
      // binding travels with the focus so the frame arrows can step along
      // THIS panel's channel; the outline shows which panel holds the keys.
      // seekBy is declared further down (it needs the track helpers and
      // refresh): wrap it so the binding does not read it before it exists.
      const binding = {
        key: panel.key, nodeId, channel, synced, seekBy: (d) => seekBy(d),
      };
      const hasKeys = () => store.getFocused()?.key === panel.key;
      panel.el.addEventListener("pointerdown", () => store.setFocused(binding));
      const offFocus = store.onFocus((f) =>
        panel.el.classList.toggle("focused", f?.key === panel.key));
      store.setFocused(binding); // a panel just opened is the one you meant

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
        // The engine's fly keys are window-level: several data panels can
        // each hold a viewer, so gate them on the focused one (the hook the
        // engine documents for multi-viewer hosts).
        viewer.flyGate = hasKeys;
        applyViewSettings(viewer);
        v3d = { host, viewer, framed: false };
        return v3d;
      };

      // Shift+arrows step-rotate the view: ←/→ roll, ↑/↓ pitch, so a scan
      // that came in tilted can be turned upright without dragging (yaw
      // stays on the orbit drag). Toaster puts this on the plain arrows,
      // but here those walk the frame timeline -- stepping through lidar
      // events is the more frequent move, and it must stay one key away.
      // The shared engine exposes rotateView and deliberately binds no keys
      // (it knows nothing about its host's DOM), so every consumer wires
      // this itself.
      const ROTATE_STEP = Math.PI / 36; // 5 degrees per press
      const onArrowKey = (e) => {
        if (!v3d || !hasKeys() || !e.shiftKey || !e.key.startsWith("Arrow")) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const rotate = (kind, sign) => v3d.viewer.rotateView(kind, sign * ROTATE_STEP);
        if (e.key === "ArrowLeft") rotate("roll", 1);
        else if (e.key === "ArrowRight") rotate("roll", -1);
        else if (e.key === "ArrowUp") rotate("pitch", 1);
        else if (e.key === "ArrowDown") rotate("pitch", -1);
        else return;
        e.preventDefault();
      };
      window.addEventListener("keydown", onArrowKey);

      // A canvas whose backing store matches the stage box: a panel taking
      // half the page draws at half-page resolution instead of a fixed
      // 640x480 thumbnail stretched to fit. Height can be pinned -- the
      // histogram is a strip, not a viewport.
      const stageCanvas = (cls, fixedHeight = null) => {
        const canvas = el("canvas", cls);
        canvas.width = Math.max(160, Math.round(stage.clientWidth) || 640);
        canvas.height = fixedHeight
          ?? Math.max(120, Math.round(stage.clientHeight) || 480);
        return canvas;
      };

      const drawAs = (kind) => {
        if (kind !== "cloud3d") drop3d();
        stage.replaceChildren();
        foot.textContent = "";
        colorSel.hidden = kind !== "cloud" && kind !== "cloud3d";
        gearBtn.hidden = kind !== "cloud" && kind !== "cloud3d";
        if (gearBtn.hidden) { settings.hidden = true; gearBtn.classList.remove("primed"); }
        bgrBtn.hidden = kind !== "image";
        if (!arr) { stage.appendChild(el("div", "placeholder", "no data")); return; }
        if (arr.repr !== undefined) { stage.appendChild(el("div", "chan-values", arr.repr)); return; }
        try {
          if (kind === "cloud3d" && arr.shape.length === 2 && arr.shape[1] >= 3) {
            const inst = ensure3d();
            stage.appendChild(inst.host);
            inst.viewer.resize(); // the host just got its size from the stage
            const spec = colorSpec();
            const k = feedCloud3d(inst.viewer, arr, spec);
            // After setCloud, not before: it rebuilds the ground grid, which
            // would come back visible on every frame otherwise.
            applyViewSettings(inst.viewer);
            if (!inst.framed && k > 0) { inst.framed = true; inst.viewer.frame(); }
            // Frame slider paused → octree for exact-fast picking + LOD cut.
            if (v3dTimer) clearTimeout(v3dTimer);
            v3dTimer = setTimeout(() => { if (v3d) v3d.viewer.buildOctree(); }, 400);
            foot.textContent = colorFoot(spec) +
              (arr.fullRows ? `  ·  sample of ${arr.fullRows} pts` : "") +
              "  ·  drag: orbit · WASD/QE: fly · shift+arrows: rotate " +
              "(keys go to the clicked panel)";
          } else if (kind === "cloud" && arr.shape.length === 2) {
            const canvas = stageCanvas("chan-canvas");
            stage.appendChild(canvas);
            const spec = colorSpec();
            const redraw = () =>
              drawCloud(canvas, arr, spec, bevView, viewSettings.size);
            redraw();
            bindViewport(canvas, {
              fit: () => cloudFit(arr, canvas.width, canvas.height),
              getView: () => bevView,
              setView: (v) => { bevView = v; },
              redraw,
            });
            stage.appendChild(el("div", "hint",
              "scroll: zoom · drag: zoom to box · shift-drag: pan · double-click: reset"));
            foot.textContent = colorFoot(spec);
          } else if ((kind === "image" && arr.shape.length === 3)
                  || (kind === "raster" && arr.shape.length === 2)) {
            // Pixels live in an offscreen source at native resolution; the
            // visible canvas is panel-sized and only blits a view of it, so
            // zooming shows real pixels instead of an upscaled thumbnail.
            const source = kind === "image"
              ? imageCanvas(arr, localStorage.getItem("studio.bgr") === "1")
              : rasterCanvas(arr);
            const canvas = stageCanvas("chan-canvas img");
            stage.appendChild(canvas);
            const fit = () =>
              sourceFit(source.width, source.height, canvas.width, canvas.height);
            const redraw = () => {
              const view = imgView ?? fit();
              drawSource(canvas, source, view);
              foot.textContent = `${source.width} × ${source.height} px · ` +
                `zoom ${view.scale.toFixed(2)}x`;
            };
            redraw();
            bindViewport(canvas, {
              fit,
              getView: () => imgView,
              setView: (v) => { imgView = v; },
              redraw,
              yUp: false,
              box: false, // a drag pans: an image has no rubber-band selection
            });
            stage.appendChild(el("div", "hint",
              "scroll: zoom · drag: pan · double-click: reset"));
          } else if (kind === "hist") {
            const canvas = stageCanvas("chan-canvas", 160);
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

      // Panel resized (a gutter dragged, a panel docked or closed): the
      // stage canvas was sized from the stage box, so redraw at the new
      // resolution rather than let the browser stretch a stale one. The 3D
      // host fills its box by itself and only needs the viewport told.
      let resizeTimer = null;
      const stageObserver = new ResizeObserver(() => {
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          resizeTimer = null;
          if (v3d) v3d.viewer.resize();
          else drawAs(currentKind());
        }, 80);
      });
      stageObserver.observe(stage);

      // Async timelines: the bound channel may be absent at the current
      // frame. Hold the nearest frame that carries it -- the latest earlier
      // one, which is what a sensor stream actually shows at that instant;
      // failing that (before the channel's first event, e.g. frame 0 of a
      // raw root whose lidar starts at 19) the first later one, so opening a
      // channel never lands on an empty panel. The note says how far off it
      // is (timestamps when the dataset has them, frame distance otherwise).
      const holdLast = async (sample) => {
        const t = await loadTrack();
        if (!t) return null; // transform node or no timeline: "channel absent"
        try {
          const best = posOf(t.indices, sample.index);
          const ahead = best < 0;
          const at = t.indices[ahead ? 0 : best];
          if (at === undefined) return null;
          const held = await store.sampleOne(nodeId, at, channel);
          if (!held.arr) return null;
          const delta = held.timestamp != null && sample.timestamp != null
            ? `${Math.abs(sample.timestamp - held.timestamp).toFixed(3)} s`
            : `${Math.abs(sample.index - held.index)} frames`;
          return {
            arr: held.arr,
            index: held.index,
            note: ` · ${ahead ? "first" : "held"} (${delta} ${ahead ? "ahead" : "old"})`,
          };
        } catch {
          return null;
        }
      };

      let refreshSeq = 0;
      const refresh = async () => {
        const mySeq = ++refreshSeq;
        try {
          await loadTrack();
          const sample = await store.sampleAt(nodeId, len, local);
          let held = null;
          if ((sample.channels[channel] ?? null) === null) {
            held = await holdLast(sample);
          }
          if (mySeq !== refreshSeq) return;
          arr = sample.channels[channel] ?? held?.arr ?? null;
          chans = held ? { ...sample.channels, [channel]: held.arr } : sample.channels;
          // The data on screen is the held frame's when the channel is absent
          // here, so name THAT one -- the box must always read as the frame
          // you are looking at, never the one you merely scrolled past.
          const shown = held ? held.index : sample.index;
          const name = nameOf(shown);
          frameInput.value = name ?? String(shown);
          // The panel's own channel leads (`lidar 000850`); the global index
          // and the sequence follow as context, because the interleaved
          // index is the one thing nobody can act on.
          const head = name != null
            ? `${channel} ${name} · frame ${shown}` : `frame ${shown}`;
          const seq = sample.frame?.sequence ? ` · ${sample.frame.sequence}` : "";
          meta.textContent = arr && arr.shape
            ? `${head}${seq} · ${arr.dtype} · ${arr.shape.join(" × ")}` +
              (arr.fullRows ? ` (showing ${arr.shape[0]} of ${arr.fullRows})` : "") +
              (held ? held.note : "")
            : `${head}${seq} · channel absent`;
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
      // Land on a global frame index: shared when synced (the topbar and
      // every other synced panel move with it), private otherwise.
      const goTo = (target) => {
        if (!synced()) {
          local = target;
          refresh();
          return;
        }
        local = null;
        // setFrame is a no-op when the frame is already there, and then
        // nothing would repaint this panel -- refresh it ourselves.
        if (target === store.getFrame()) refresh();
        else store.setFrame(target);
      };

      // Seek this panel to one of ITS channel's frames. `change` alone would
      // ignore a re-typed identical value and only fire on blur for some
      // inputs, and "I pressed Enter and nothing moved" is exactly the bug
      // to avoid here -- so Enter seeks explicitly too.
      const seekTyped = async () => {
        const text = frameInput.value.trim();
        const t = await loadTrack();
        if (!t) { // synchronous dataset: the global index is the frame
          goTo(Math.max(0, Math.min(Number(text) || 0, (len ?? 1) - 1)));
          return;
        }
        const k = seek(t, text);
        if (k < 0) { refresh(); return; } // unknown: restore what is shown
        goTo(t.indices[k]);
      };
      frameInput.addEventListener("change", seekTyped);
      frameInput.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        e.preventDefault();
        // Hand the keyboard back: a focused text input swallows the arrows,
        // and stepping on from the frame you just typed is the whole point.
        frameInput.blur();
        seekTyped();
      });

      // Step this panel by *delta* of ITS channel's frames. Used by the
      // arrow keys when this panel holds the keys and is NOT synced -- a
      // synced panel routes its arrows through the global timeline instead.
      const seekBy = (delta) => {
        const at = local ?? store.getFrame();
        if (!track) {
          goTo(Math.max(0, Math.min(at + delta, (len ?? 1) - 1)));
          return;
        }
        const base = posOf(track.indices, at);
        const off = delta < 0 && track.indices[base] !== at ? 1 : 0;
        const next = Math.max(
          0, Math.min(base + delta + off, track.indices.length - 1));
        goTo(track.indices[next]);
      };

      syncBox.addEventListener("change", () => {
        syncLabel.classList.toggle("on", synced());
        if (!synced()) {
          // Unticking must never move the picture: freeze where it stands.
          if (local === null) {
            local = Math.max(0, Math.min(store.getFrame(), (len ?? 1) - 1));
          }
          return;
        }
        // Re-synced: the frame this panel wandered to becomes everyone's.
        const own = local;
        local = null;
        if (own === null || own === store.getFrame()) refresh();
        else store.setFrame(own);
      });
      // A global move brings every synced panel along; an unsynced one stays
      // on the frame it was left at.
      const offFrame = store.onFrame(() => {
        if (!synced()) return;
        local = null;
        refresh();
      });
      panel.onRemove = () => {
        offFrame();
        offFocus();
        offView();
        if (hasKeys()) store.setFocused(null);
        window.removeEventListener("keydown", onArrowKey);
        stageObserver.disconnect();
        if (resizeTimer) clearTimeout(resizeTimer);
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
