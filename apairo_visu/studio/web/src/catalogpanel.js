// Catalog panel (browse apairo_transform / apairo_preprocess, set kwargs,
// try on a channel of the selected node) and the try panel (before / after
// on the current frame + the code snippet to paste into the script).

import { decodeArray } from "./arrays.js";
import { el } from "./datapanel.js";
import { fillSelect } from "./panels.js";
import * as store from "./store.js";
import {
  columnStatsLine, drawCloud, drawHist, drawImage, drawRaster,
  previewKind, statsLine,
} from "./views.js";

const accent = () => getComputedStyle(document.documentElement).getPropertyValue("--accent");
const muted = () => getComputedStyle(document.documentElement).getPropertyValue("--muted");

let catalogPromise = null;
const getCatalog = () => (catalogPromise ??= store.api("/api/catalog"));

/* -------------------------------------------------------------- catalog */

function paramControl(param) {
  if (param.type === "bool") {
    const input = el("input");
    input.type = "checkbox";
    input.checked = Boolean(param.default);
    input.dataset.param = param.name;
    return input;
  }
  const input = el("input", "opt-input");
  input.type = param.type === "int" || param.type === "float" ? "number" : "text";
  if (param.type === "int") input.step = "1";
  if (param.type === "float") input.step = "any";
  if (param.default !== null && param.default !== undefined) input.value = String(param.default);
  input.placeholder = param.required ? "required" : "default";
  input.dataset.param = param.name;
  return input;
}

function readKwargs(form) {
  const kwargs = {};
  for (const input of form.querySelectorAll("[data-param]")) {
    const name = input.dataset.param;
    if (input.type === "checkbox") kwargs[name] = input.checked;
    else if (input.value !== "") kwargs[name] = input.value;
  }
  return kwargs;
}

// openTry(binding) -> opens/updates a try panel.
export function catalogSpec(openTry) {
  return {
    key: "catalog",
    tag: "CATALOG",
    title: "transforms",
    build(panel) {
      panel.body.classList.add("scroll-body");
      const targetBar = el("div", "cat-target");
      const search = el("input", "opt-input");
      search.type = "search";
      search.placeholder = "filter…";
      const list = el("div");
      panel.body.append(targetBar, search, list);
      let expanded = null;

      const paintTarget = () => {
        const t = store.getTarget();
        targetBar.textContent = t
          ? `target: ${t.channel} @ ${t.nodeLabel} · sample ${t.index}`
          : "target: none — press 'try' on a data panel or an inspector row";
        targetBar.classList.toggle("set", Boolean(t));
      };
      paintTarget();

      const render = async () => {
        let cat;
        try {
          cat = await getCatalog();
        } catch (err) {
          list.replaceChildren(el("div", "error", String(err)));
          return;
        }
        const needle = search.value.trim().toLowerCase();
        list.replaceChildren();
        for (const missing of cat.missing || []) {
          list.appendChild(el("div", "chan-values", `${missing}: not installed`));
        }
        for (const group of cat.modules) {
          const entries = group.entries.filter((e) =>
            !needle || e.id.toLowerCase().includes(needle));
          if (!entries.length) continue;
          list.appendChild(el("div", "cat-module",
            group.module.replace(/^apairo_/, "")));
          for (const entry of entries) {
            const row = el("div", "cat-entry");
            const head = el("div", "cat-entry-head");
            head.appendChild(el("span", "cat-name", entry.name));
            head.appendChild(el("span", "chan-meta",
              entry.is_preprocessor ? "preprocessor" : entry.kind));
            row.appendChild(head);
            head.addEventListener("click", () => {
              expanded = expanded === entry.id ? null : entry.id;
              render();
            });
            if (expanded === entry.id) row.appendChild(buildForm(entry));
            list.appendChild(row);
          }
        }
      };

      const buildForm = (entry) => {
        const form = el("div", "cat-form");
        if (entry.doc) {
          form.appendChild(el("div", "chan-values", entry.doc.split("\n\n")[0]));
        }
        for (const param of entry.params) {
          const lab = el("label", "opt");
          lab.appendChild(el("span", null, param.name));
          lab.appendChild(paramControl(param));
          form.appendChild(lab);
        }
        const channelSel = el("select", "chan-color");
        const tryBtn = el("button", "ghost", "try on target");
        const actions = el("div", "cat-actions");
        actions.append(channelSel, tryBtn);
        form.appendChild(actions);

        // The try applies to the DESIGNATED target (try button); without one it
        // falls back to the selected node. The channel select is scoped to
        // that node, preset to the target's channel.
        const fillChannels = async () => {
          const t = store.getTarget();
          const nodeId = t ? t.nodeId : store.getSelected();
          if (!nodeId) return;
          try {
            const detail = await store.detailAt(nodeId);
            const keys = (detail.channels || []).filter((c) => c.key).map((c) => c.key);
            fillSelect(channelSel, keys.map((k) => [k, k]));
            if (t && keys.includes(t.channel)) channelSel.value = t.channel;
            channelSel.dataset.nodeId = nodeId;
            channelSel.dataset.nodeLabel = detail.label;
            channelSel.dataset.len = detail.len ?? 1;
          } catch { /* inspector shows the error */ }
        };
        fillChannels();

        tryBtn.addEventListener("click", () => {
          if (!channelSel.value) return;
          const t = store.getTarget();
          const len = Number(channelSel.dataset.len) || 1;
          openTry({
            entryId: entry.id,
            entryName: entry.name,
            kwargs: readKwargs(form),
            nodeId: channelSel.dataset.nodeId,
            nodeLabel: channelSel.dataset.nodeLabel,
            channel: channelSel.value,
            len,
            // Designated sample: pin the target's index; else capture the
            // current frame (unpin from the try panel to follow the slider).
            index: t && t.nodeId === channelSel.dataset.nodeId
              ? t.index
              : Math.max(0, Math.min(store.getFrame(), len - 1)),
            pinned: true,
          });
        });
        return form;
      };

      search.addEventListener("input", render);
      const unsubSelect = store.onSelect(render);
      const unsubTarget = store.onTarget(() => { paintTarget(); render(); });
      panel.onRemove = () => { unsubSelect(); unsubTarget(); };
      render();
    },
  };
}

/* ------------------------------------------------------------ try panel */

function drawInto(stage, foot, arr) {
  stage.replaceChildren();
  foot.textContent = "";
  if (!arr) { stage.appendChild(el("div", "placeholder", "—")); return; }
  const kind = previewKind(arr.shape, arr.dtype);
  // Try cells are static before/after snapshots: clouds render as BEV here
  // even when the data panels would open the 3D viewer.
  if (kind === "cloud" || kind === "cloud3d") {
    const canvas = el("canvas", "chan-canvas");
    canvas.width = 420;
    canvas.height = 320;
    stage.appendChild(canvas);
    const col = Math.min(2, arr.shape[1] - 1);
    drawCloud(canvas, arr, col);
    foot.textContent = `${arr.shape.join(" × ")} · ${columnStatsLine(arr, col, `col ${col}`)}`;
  } else if (kind === "image") {
    const canvas = el("canvas", "chan-canvas img");
    stage.appendChild(canvas);
    drawImage(canvas, arr, localStorage.getItem("studio.bgr") === "1");
    foot.textContent = arr.shape.join(" × ");
  } else if (kind === "raster") {
    const canvas = el("canvas", "chan-canvas img");
    stage.appendChild(canvas);
    drawRaster(canvas, arr);
    foot.textContent = arr.shape.join(" × ");
  } else {
    const canvas = el("canvas", "chan-canvas");
    canvas.width = 420;
    canvas.height = 120;
    stage.appendChild(canvas);
    drawHist(canvas, arr, accent(), muted());
    foot.textContent = `${arr.shape.join(" × ")} · ${statsLine(arr)}`;
  }
}

export function trySpec(binding) {
  const { entryId, entryName, kwargs, nodeId, nodeLabel, channel, len } = binding;
  // The designated sample: pinned tries stay on binding.index whatever the
  // global slider does; unpinned tries follow it.
  if (binding.index === undefined || binding.index === null) binding.index = 0;
  if (binding.pinned === undefined) binding.pinned = true;
  return {
    key: `try:${entryId}:${nodeId}:${channel}`,
    tag: "TRY",
    title: `${entryName} on ${channel} @ ${nodeLabel}`,
    binding,
    build(panel) {
      panel.body.classList.add("scroll-body");
      const meta = el("div", "chan-values");
      const pair = el("div", "try-pair");
      const beforeBox = el("div", "try-cell");
      const afterBox = el("div", "try-cell");
      const mkCell = (box, label) => {
        box.appendChild(el("h3", null, label));
        const stage = el("div", "data-stage");
        const foot = el("div", "chan-values");
        box.append(stage, foot);
        return { stage, foot };
      };
      const before = mkCell(beforeBox, "before");
      const after = mkCell(afterBox, "after");
      pair.append(beforeBox, afterBox);
      const errorBox = el("div", "error");
      const snippetBox = el("pre", "doc snippet");
      const copyBtn = el("button", "ghost", "copy snippet");
      copyBtn.addEventListener("click", () => {
        navigator.clipboard?.writeText(snippetBox.textContent);
        copyBtn.textContent = "copied";
        setTimeout(() => { copyBtn.textContent = "copy snippet"; }, 1200);
      });
      panel.body.append(meta, pair, errorBox, snippetBox, copyBtn);

      // Sample designation controls: a frame input + pin toggle in the head.
      const frameInput = el("input", "opt-input frame-input");
      frameInput.type = "number";
      frameInput.min = "0";
      frameInput.max = String(len - 1);
      frameInput.value = String(binding.index);
      frameInput.title = "Designated sample index";
      const pinBtn = el("button", "tbtn");
      const paintPin = () => {
        pinBtn.textContent = binding.pinned ? "pinned" : "follow";
        pinBtn.title = binding.pinned
          ? "Pinned to this sample — click to follow the global slider"
          : "Following the global slider — click to pin the current sample";
        frameInput.disabled = !binding.pinned;
      };
      panel.controls.append(frameInput, pinBtn);

      let seq = 0;
      let lastIndex = null;
      const refresh = async (force = false) => {
        const mySeq = ++seq;
        const index = binding.pinned
          ? Math.max(0, Math.min(binding.index, len - 1))
          : Math.max(0, Math.min(store.getFrame(), len - 1));
        if (!force && index === lastIndex) return; // pinned: ignore slider moves
        lastIndex = index;
        if (!binding.pinned) frameInput.value = String(index);
        meta.textContent =
          `sample ${index}${binding.pinned ? " (pinned)" : " (follows slider)"} · ${entryId}`;
        try {
          const result = await store.apiPost("/api/try", {
            entry_id: entryId, node_id: nodeId, channel, kwargs, index,
          });
          if (mySeq !== seq) return;
          drawInto(before.stage, before.foot, decodeArray(result.before));
          errorBox.textContent = result.error || "";
          drawInto(after.stage, after.foot,
            result.after ? decodeArray(result.after) : null);
          snippetBox.textContent = result.snippet || "";
        } catch (err) {
          if (mySeq === seq) errorBox.textContent = String(err);
        }
      };

      frameInput.addEventListener("change", () => {
        binding.index = Math.max(0, Math.min(Number(frameInput.value) || 0, len - 1));
        frameInput.value = String(binding.index);
        refresh(true);
      });
      pinBtn.addEventListener("click", () => {
        binding.pinned = !binding.pinned;
        if (binding.pinned) binding.index = Math.max(0, Math.min(store.getFrame(), len - 1));
        frameInput.value = String(binding.index);
        paintPin();
        refresh(true);
      });
      paintPin();
      panel.onRemove = store.onFrame(() => refresh());
      refresh(true);
    },
  };
}
