// apairo studio front -- vanilla JS, ES modules, zero build (house pattern).
// A multi-panel workbench: pipeline graph, node inspector, dockable data
// panels (one per node+channel, several view modes) and a frame-metrics
// panel. Panels share one store (global frame index, selection, per-node
// sample cache) and dock/drag like projector's views.

import { catalogSpec, trySpec } from "./catalogpanel.js";
import { dataSpec, el, inspectorSpec, metricsSpec, pipelineSpec } from "./datapanel.js";
import { entriesSpec } from "./entriespanel.js";
import { seriesSpec } from "./seriespanel.js";
import { PanelManager, fillSelect } from "./panels.js";
import * as store from "./store.js";

const $ = (sel) => document.querySelector(sel);
// Versioned: bump when the default panel set changes, so stale saved
// layouts don't hide newly introduced core panels.
const LAYOUT_KEY = "studio.layout.v2";

/* ------------------------------------------------------------- theming */

function initTheme() {
  const btn = $("#theme");
  const modes = ["auto", "light", "dark"];
  let mode = localStorage.getItem("studio.theme") || "auto";
  const apply = () => {
    if (mode === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = mode;
    btn.textContent = mode;
    localStorage.setItem("studio.theme", mode);
  };
  btn.addEventListener("click", () => {
    mode = modes[(modes.indexOf(mode) + 1) % modes.length];
    apply();
  });
  apply();
}

/* ---------------------------------------------------------- frame slider */

// Global: every data/metrics panel follows it. Its range tracks the longest
// node bound so far; per-node clamping happens in the store.
let maxLen = 1;

// Last position in *idxs* (sorted) whose frame is <= *frame*.
const floorPos = (idxs, frame) => {
  let lo = 0, hi = idxs.length - 1, best = 0;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (idxs[m] <= frame) { best = m; lo = m + 1; } else hi = m - 1;
  }
  return best;
};

function initSlider() {
  const slider = $("#frame-slider");
  const counter = $("#frame-counter");
  const rangeClear = $("#range-clear");
  const gotoInput = $("#frame-goto");
  // Active bounds: the whole dataset, or the sequence range when one is
  // selected in the inspector.
  const bounds = () => {
    const r = store.getRange();
    return r ? [r.start, Math.max(r.start, r.stop - 1)]
             : [0, Math.max(0, maxLen - 1)];
  };
  const label = () => {
    const r = store.getRange();
    return r ? ` · ${r.label}` : "";
  };
  // Positions the slider walks: the per-channel track when one is chosen
  // (intersected with the sequence range), else the plain frame range.
  // Cached — the intersection is O(track) and paint() runs on every scrub,
  // while the inputs only change when the track or the range does.
  let posCache = null; // {indices, stems|null} or null (no track)
  const positions = () => {
    if (posCache) return posCache;
    const t = store.getTrack();
    if (!t) return null;
    const r = store.getRange();
    if (!r) {
      posCache = { indices: t.indices, stems: t.stems || null };
      return posCache;
    }
    const indices = [];
    const stems = t.stems ? [] : null;
    for (let k = 0; k < t.indices.length; k++) {
      const i = t.indices[k];
      if (i < r.start || i >= r.stop) continue;
      indices.push(i);
      if (stems) stems.push(t.stems[k]);
    }
    posCache = { indices, stems };
    return posCache;
  };
  // Channel mode reads as "lidar 000850" — the on-disk file name leads,
  // because that is what a label file is called; the channel-relative
  // position and the global timeline index are secondary context.
  const trackText = (p, pos) => {
    const stem = p.stems ? ` ${p.stems[pos]}` : "";
    return `${store.getTrack().channel}${stem} · ${pos + 1}/${p.indices.length}` +
      ` · frame ${p.indices[pos]}${label()}`;
  };
  const paint = () => {
    const p = positions();
    if (p && p.indices.length) {
      const pos = floorPos(p.indices, store.getFrame());
      slider.min = 0;
      slider.max = p.indices.length - 1;
      slider.value = pos;
      counter.textContent = trackText(p, pos);
    } else {
      const [lo, hi] = bounds();
      slider.min = lo;
      slider.max = hi;
      slider.value = Math.max(lo, Math.min(store.getFrame(), hi));
      counter.textContent = `${slider.value} / ${hi}${label()}`;
    }
    rangeClear.hidden = !store.getRange();
  };
  let pending = null;
  slider.addEventListener("input", () => {
    const p = positions();
    const value = Number(slider.value);
    counter.textContent = p && p.indices.length
      ? trackText(p, Math.min(value, p.indices.length - 1))
      : `${value} / ${bounds()[1]}${label()}`;
    // Trailing throttle: at most one store update per 60ms while scrubbing.
    if (pending) return;
    pending = setTimeout(() => {
      pending = null;
      const now = positions();
      const v = Number(slider.value);
      store.setFrame(now && now.indices.length
        ? now.indices[Math.min(v, now.indices.length - 1)]
        : v);
    }, 60);
  });
  rangeClear.addEventListener("click", () => store.setRange(null));

  // The focused data panel's own event timeline, prefetched on focus so an
  // arrow press stays synchronous -- stepping has to feel like a key repeat,
  // not a request.
  let focusTrack = null;
  store.onFocus(async (f) => {
    focusTrack = null;
    if (!f) return;
    try {
      const t = await store.channelFrames(f.nodeId, f.channel);
      if (store.getFocused()?.key === f.key && t.indices.length) {
        focusTrack = t.indices;
      }
    } catch { /* transform node or synchronous data: global stepping */ }
  });

  // Arrow keys step the timeline. Which timeline, in order: the topbar
  // channel track when one is set (the slider walks it, so the arrows must
  // agree with what the counter shows), else the FOCUSED panel's own channel
  // -- an arrow on a lidar view steps one scan, not one interleaved imu
  // message -- else one global frame. It moves the GLOBAL frame either way,
  // so every other panel follows and the camera, the metrics and the series
  // marker stay in step with the lidar view.
  const step = (delta) => {
    const p = positions();
    const idxs = p && p.indices.length ? p.indices : focusTrack;
    if (idxs && idxs.length) {
      const base = floorPos(idxs, store.getFrame());
      // Between two events, stepping back should land on the one behind, not
      // one further: floorPos already points there.
      const off = delta < 0 && idxs[base] !== store.getFrame() ? 1 : 0;
      const next = Math.max(0, Math.min(base + delta + off, idxs.length - 1));
      store.setFrame(idxs[next]);
      return;
    }
    const [lo, hi] = bounds();
    store.setFrame(Math.max(lo, Math.min(store.getFrame() + delta, hi)));
  };
  window.addEventListener("keydown", (e) => {
    // Shift+arrows rotate the hovered 3D view (bound in the data panel), and
    // a focused text control keeps its own arrows -- the go-to box, a
    // panel's frame input. The slider is deliberately NOT excluded: its
    // native stepping would move one position where ↑/↓ should move ten.
    if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && (t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
    if (t && t.tagName === "INPUT" && t.type !== "range") return;
    // Up goes forward, down goes back: the arrows read as a throttle here,
    // not as a list cursor -- up is more, down is less.
    const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: 10, ArrowDown: -10 }[e.key];
    if (delta === undefined) return;
    e.preventDefault();
    // An unsynced panel is an island: its arrows move only itself, never the
    // shared timeline. A synced one routes through the global frame below,
    // so every other synced panel comes along.
    const f = store.getFocused();
    if (f && f.synced && !f.synced()) f.seekBy(delta);
    else step(delta);
  });

  // "go to": a file stem (000850 — what a label file is named after), or a
  // plain frame index when no channel track narrows the timeline. Stems
  // restart at 000000 in every sequence and every channel has its own, so
  // the search starts scoped to what the user is looking at and only widens
  // from there.
  const where = (hit) => [hit.sequence, hit.channel].filter(Boolean).join(" · ");
  const land = (stem, hit, note = "") => {
    store.setFrame(hit.index);
    const at = where(hit);
    $("#status").textContent =
      `${stem} → frame ${hit.index}${at ? ` (${at})` : ""}${note}`;
  };
  const jumpTo = async (raw) => {
    const text = raw.trim();
    if (!text) return;
    const track = store.getTrack();
    const range = store.getRange();
    const numeric = /^\d+$/.test(text);

    // No channel track and a bare number: that is a frame index.
    if (!track && numeric) {
      const [lo, hi] = bounds();
      store.setFrame(Math.max(lo, Math.min(Number(text), hi)));
      return;
    }
    // Track loaded: its stems are already here, match them without a request
    // (and accept 850 for 000850 -- nobody types the padding).
    const p = positions();
    if (p && p.stems) {
      const k = p.stems.findIndex((s) => s === text ||
        (numeric && /^\d+$/.test(s) && Number(s) === Number(text)));
      if (k >= 0) {
        store.setFrame(p.indices[k]);
        $("#status").textContent =
          `${track.channel} ${p.stems[k]} → frame ${p.indices[k]}`;
        return;
      }
    }

    const nodeId = (track && track.nodeId) || store.getSelected();
    if (!nodeId) return;
    const channel = track ? track.channel : undefined;
    const stems = numeric && text.length < 6
      ? [text.padStart(6, "0"), text] : [text];
    try {
      for (const stem of stems) {
        // 1. Exactly what is on screen: this channel, this sequence.
        let { matches } = await store.locate(
          nodeId, stem, { channel, sequence: range && range.label });
        if (matches.length) return land(stem, matches[0]);
        // 2. Same channel, another sequence. The sequence restriction would
        //    clamp the jump straight back, so drop it and say so.
        ({ matches } = await store.locate(nodeId, stem, { channel }));
        if (matches.length) {
          store.setRange(null);
          return land(stem, matches[0], " — sequence restriction cleared");
        }
        // 3. Anywhere: report where it lives, but do not drag the slider off
        //    the channel timeline it is walking.
        ({ matches } = await store.locate(nodeId, stem, {}));
        if (matches.length) {
          $("#status").textContent =
            `'${stem}' is not on ${channel ?? "this timeline"} — ` +
            `it exists as ${where(matches[0])}`;
          return;
        }
      }
    } catch (err) {
      $("#status").textContent = String(err);
      return;
    }
    $("#status").textContent = `no frame named '${text}' in this dataset`;
  };
  gotoInput.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    jumpTo(gotoInput.value);
  });

  // Frame changes can come from panels too (a series chart click): keep the
  // slider in sync, not only the other way around.
  store.onFrame(paint);
  store.onRange(() => {
    posCache = null;
    const [lo, hi] = bounds();
    store.setFrame(Math.max(lo, Math.min(store.getFrame(), hi)));
    paint();
  });
  store.onTrack(() => {
    posCache = null;
    // Snap the current frame onto the track so panels show a real event.
    const p = positions();
    if (p && p.indices.length) {
      store.setFrame(p.indices[floorPos(p.indices, store.getFrame())]);
    }
    paint();
  });
  paint();
  return { paint, bump(len) { if (len > maxLen) { maxLen = len; paint(); } } };
}

// Topbar channel select: bind the slider to one channel's event timeline of
// the SELECTED node. "all channels" restores the plain global timeline.
function initTrackSelect() {
  const sel = $("#track-select");
  let nodeId = null;
  const populate = async (selected) => {
    if (selected === nodeId) return;
    nodeId = selected;
    if (store.getTrack()) store.setTrack(null);
    fillSelect(sel, [["", "all channels"]]);
    if (!selected) return;
    try {
      const detail = await store.detailAt(selected);
      const keys = (detail.channels || []).filter((c) => c.key).map((c) => c.key);
      fillSelect(sel, [["", "all channels"], ...keys.map((k) => [k, `only ${k}`])]);
    } catch { /* inspector shows the error */ }
  };
  sel.addEventListener("change", async () => {
    if (!sel.value || !nodeId) { store.setTrack(null); return; }
    try {
      const { indices, stems } = await store.channelFrames(nodeId, sel.value);
      if (!indices.length) {
        $("#status").textContent =
          `${sel.value}: every frame carries it (synchronous) — nothing to filter`;
        sel.value = "";
        store.setTrack(null);
        return;
      }
      store.setTrack({ nodeId, channel: sel.value, indices, stems: stems || null });
    } catch (err) {
      $("#status").textContent = String(err);
      sel.value = "";
      store.setTrack(null);
    }
  });
  store.onSelect(populate);
  populate(store.getSelected());
}

/* ----------------------------------------------------------------- boot */

async function init() {
  initTheme();
  const slider = initSlider();
  initTrackSelect();
  const manager = new PanelManager($("#stack"));

  // Adopt the server-rendered SVG (SSR-lite) before its host is removed.
  const ssr = $("#graph-ssr");
  const svg = ssr.querySelector("svg");
  svg.removeAttribute("font-family");

  const openData = (nodeId, nodeLabel, channel, len) => {
    const key = `data:${nodeId}:${channel}`;
    if (manager.byKey(key)) return;
    slider.bump(len ?? 1);
    // First data panel opens its own column; the next ones stack into it
    // (drag a panel's header to re-dock it anywhere).
    const first = !manager.panels.some((p) => p.key.startsWith("data:"));
    manager.add(dataSpec({ nodeId, nodeLabel, channel, len }), first);
  };

  const openSeries = (nodeId, nodeLabel, channel, len, shape) => {
    const key = `series:${nodeId}:${channel}`;
    if (manager.byKey(key)) return;
    slider.bump(len ?? 1);
    manager.add(seriesSpec({ nodeId, nodeLabel, channel, len, shape }), true);
  };

  const openEntries = (nodeId, nodeLabel, channel, len) => {
    const key = `entries:${nodeId}:${channel}`;
    if (manager.byKey(key)) return;
    slider.bump(len ?? 1);
    manager.add(entriesSpec({ nodeId, nodeLabel, channel, len }), true);
  };

  const openTry = (binding) => {
    const spec = trySpec(binding);
    const existing = manager.byKey(spec.key);
    if (existing) manager.remove(existing.id); // re-try with fresh kwargs
    slider.bump(binding.len ?? 1);
    manager.add(spec, true);
  };

  // Panel factory -- used both for the default layout and for restores.
  const make = (key, binding) => {
    if (key === "pipeline") return pipelineSpec(svg);
    if (key === "inspector") return inspectorSpec(openData, openSeries, openEntries);
    if (key === "metrics") return metricsSpec();
    if (key === "catalog") return catalogSpec(openTry);
    const specFor = {
      "data:": dataSpec, "try:": trySpec, "series:": seriesSpec,
      "entries:": entriesSpec,
    };
    const prefix = key && Object.keys(specFor).find((p) => key.startsWith(p));
    if (prefix && binding) {
      if (!svg.querySelector(`[data-node="${binding.nodeId}"]`)) return null;
      slider.bump(binding.len ?? 1);
      return specFor[prefix](binding);
    }
    return null;
  };

  let restored = false;
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "null");
    if (saved) restored = manager.restore(saved, make);
  } catch { /* corrupt layout: fall through to the default */ }
  if (!restored || !manager.byKey("pipeline")) {
    manager.panels.slice().forEach((p) => manager.remove(p.id));
    manager.add(make("pipeline")).el.style.flex = "2.2 1 0";
    manager.add(make("inspector"), true);
    manager.add(make("metrics"));
    manager.add(make("catalog"));
  }
  manager.onChanged = () =>
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(manager.serialize()));
  manager.onChanged();

  // Re-open core panels from the topbar menu.
  const addSel = $("#add-panel");
  fillSelect(addSel, [
    ["", "+ panel"], ["pipeline", "pipeline graph"],
    ["inspector", "inspector"], ["metrics", "frame metrics"],
    ["catalog", "transform catalog"],
  ]);
  addSel.addEventListener("change", () => {
    if (addSel.value && !manager.byKey(addSel.value)) {
      manager.add(make(addSel.value), addSel.value === "pipeline");
    }
    addSel.value = "";
  });

  $("#status").textContent =
    `${svg.querySelectorAll(".node.dataset").length} datasets · ` +
    `${svg.querySelectorAll(".node.transform").length} transforms`;

  // Track node lengths so the global slider spans the longest known node.
  store.onSelect(async (nodeId) => {
    try {
      const detail = await store.detailAt(nodeId);
      if (detail.len) slider.bump(detail.len);
    } catch { /* inspector shows the error */ }
  });

  ssr.remove();

  // Auto-select the terminal dataset (first rendered = first passed dataset).
  const first = svg.querySelector(".node.dataset");
  if (first) store.setSelected(first.dataset.node);

  // ?open=nodeId:channel,... -- pre-open data panels (shareable workspaces).
  const open = new URLSearchParams(location.search).get("open");
  for (const entry of open ? open.split(",") : []) {
    const [nodeId, channel] = entry.split(":");
    if (!nodeId || !channel) continue;
    try {
      const detail = await store.detailAt(nodeId);
      openData(nodeId, detail.label, channel, detail.len);
    } catch { /* unknown node: skip */ }
  }

  // ?try=entryId:nodeId:channel[:k=v;k=v] -- pre-open a try panel.
  const tryParam = new URLSearchParams(location.search).get("try");
  if (tryParam) {
    const [entryId, nodeId, channel, kvs] = tryParam.split(":");
    const kwargs = {};
    for (const kv of kvs ? kvs.split(";") : []) {
      const [k, v] = kv.split("=");
      if (k && v !== undefined) kwargs[k] = v;
    }
    try {
      const detail = await store.detailAt(nodeId);
      const len = detail.len ?? 1;
      openTry({
        entryId, kwargs, nodeId, channel,
        entryName: entryId.split(".").pop(),
        nodeLabel: detail.label, len,
        index: Math.max(0, Math.min(store.getFrame(), len - 1)),
        pinned: true,
      });
    } catch { /* unknown node: skip */ }
  }
}

init().catch((err) => {
  $("#stack").replaceChildren(el("div", "error", String(err)));
});
