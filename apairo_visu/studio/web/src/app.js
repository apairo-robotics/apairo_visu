// apairo studio front -- vanilla JS, ES modules, zero build (house pattern).
// A multi-panel workbench: pipeline graph, node inspector, dockable data
// panels (one per node+channel, several view modes) and a frame-metrics
// panel. Panels share one store (global frame index, selection, per-node
// sample cache) and dock/drag like projector's views.

import { catalogSpec, trySpec } from "./catalogpanel.js";
import { dataSpec, el, inspectorSpec, metricsSpec, pipelineSpec } from "./datapanel.js";
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
  const positions = () => {
    const t = store.getTrack();
    if (!t) return null;
    const r = store.getRange();
    return r ? t.indices.filter((i) => i >= r.start && i < r.stop) : t.indices;
  };
  const paint = () => {
    const idxs = positions();
    if (idxs && idxs.length) {
      const pos = floorPos(idxs, store.getFrame());
      slider.min = 0;
      slider.max = idxs.length - 1;
      slider.value = pos;
      // Channel mode: the channel-relative index leads; the global
      // timeline index is secondary context.
      counter.textContent =
        `${store.getTrack().channel} ${pos + 1}/${idxs.length}` +
        ` · frame ${idxs[pos]}${label()}`;
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
    const idxs = positions();
    const value = Number(slider.value);
    counter.textContent = idxs && idxs.length
      ? `${store.getTrack().channel} ${value + 1}/${idxs.length}` +
        ` · frame ${idxs[value]}${label()}`
      : `${value} / ${bounds()[1]}${label()}`;
    // Trailing throttle: at most one store update per 60ms while scrubbing.
    if (pending) return;
    pending = setTimeout(() => {
      pending = null;
      const now = positions();
      const v = Number(slider.value);
      store.setFrame(now && now.length ? now[Math.min(v, now.length - 1)] : v);
    }, 60);
  });
  rangeClear.addEventListener("click", () => store.setRange(null));
  // Frame changes can come from panels too (a series chart click): keep the
  // slider in sync, not only the other way around.
  store.onFrame(paint);
  store.onRange(() => {
    const [lo, hi] = bounds();
    store.setFrame(Math.max(lo, Math.min(store.getFrame(), hi)));
    paint();
  });
  store.onTrack(() => {
    // Snap the current frame onto the track so panels show a real event.
    const idxs = positions();
    if (idxs && idxs.length) store.setFrame(idxs[floorPos(idxs, store.getFrame())]);
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
      const { indices } = await store.channelFrames(nodeId, sel.value);
      if (!indices.length) {
        $("#status").textContent =
          `${sel.value}: every frame carries it (synchronous) — nothing to filter`;
        sel.value = "";
        store.setTrack(null);
        return;
      }
      store.setTrack({ nodeId, channel: sel.value, indices });
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

  const openSeries = (nodeId, nodeLabel, channel, len) => {
    const key = `series:${nodeId}:${channel}`;
    if (manager.byKey(key)) return;
    slider.bump(len ?? 1);
    manager.add(seriesSpec({ nodeId, nodeLabel, channel, len }), true);
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
    if (key === "inspector") return inspectorSpec(openData, openSeries);
    if (key === "metrics") return metricsSpec();
    if (key === "catalog") return catalogSpec(openTry);
    const specFor = { "data:": dataSpec, "try:": trySpec, "series:": seriesSpec };
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
