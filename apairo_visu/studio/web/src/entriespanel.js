// Entries panel: what is actually IN a channel, one row per frame.
//
// The inspector's channel table says a channel exists and what one sample
// looks like; this says which frames it holds -- row, on-disk file and its
// size, timestamp, and the global index to seek to. That is the view that
// answers "which frames have I already labelled, and which are left", which
// no single sample can show. Paged server-side (see registry.entries): every
// row costs an os.stat, so the whole listing is never materialized at once.

import { el } from "./datapanel.js";
import { fillSelect } from "./panels.js";
import * as store from "./store.js";

const PAGE = 200;

// Byte counts read at a glance, not as seven digits.
function humanBytes(n) {
  if (n === null || n === undefined) return "—";
  const units = ["B", "kB", "MB", "GB"];
  let v = n, u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u += 1; }
  return `${u === 0 ? v : v.toFixed(v < 10 ? 1 : 0)} ${units[u]}`;
}

// Timestamps are absolute epoch seconds; what matters when scanning a list
// is the offset from the channel's first frame.
function relTime(t, origin) {
  if (t === null || t === undefined || origin === null) return "—";
  return `${(t - origin).toFixed(3)} s`;
}

export function entriesSpec(binding) {
  const { nodeId, nodeLabel, channel } = binding;
  return {
    key: `entries:${nodeId}:${channel}`,
    tag: "LIST",
    title: `${channel} entries @ ${nodeLabel}`,
    binding,
    build(panel) {
      panel.body.classList.add("scroll-body");
      const head = el("div", "chan-values");
      const host = el("div");
      panel.body.append(head, host);

      const seqSel = el("select", "chan-color");
      seqSel.title = "Restrict the listing to one sequence";
      const findInput = el("input", "opt-input frame-input");
      findInput.type = "text";
      findInput.placeholder = "find";
      findInput.title = "Page to a file stem (000850) or a channel row";
      const prevBtn = el("button", "tbtn", "prev");
      const nextBtn = el("button", "tbtn", "next");
      const hereBtn = el("button", "tbtn", "here");
      hereBtn.title = "Page to the frame the global timeline is on";
      panel.controls.append(seqSel, findInput, prevBtn, nextBtn, hereBtn);

      let start = 0;
      let page = null;   // last payload
      let origin = null; // timestamp of the channel's first listed frame
      let seq = 0;

      const sequence = () => seqSel.value || null;

      const paint = () => {
        if (!page) return;
        const frame = store.getFrame();
        const table = el("table", "kv entries");
        const hrow = el("tr");
        for (const h of ["row", "file", "frame", "t", "size"]) {
          hrow.appendChild(el("th", null, h));
        }
        table.appendChild(hrow);
        for (const e of page.entries) {
          const tr = el("tr", e.index === frame ? "at-frame" : null);
          tr.appendChild(el("td", "num", e.row ?? e.pos));
          tr.appendChild(el("td", null, e.file ?? e.stem ?? "—"));
          tr.appendChild(el("td", "num", String(e.index)));
          tr.appendChild(el("td", "num", relTime(e.timestamp ?? null, origin)));
          tr.appendChild(el("td", "num", humanBytes(e.bytes)));
          tr.title = "Seek the timeline to this frame";
          tr.addEventListener("click", () => store.setFrame(e.index));
          table.appendChild(tr);
        }
        host.replaceChildren(table);
        head.textContent =
          `${channel} · ${page.total} frames` +
          (sequence() ? ` in ${sequence()}` : "") +
          ` · showing ${page.start}–${Math.max(page.start, page.stop - 1)}`;
        prevBtn.disabled = page.start <= 0;
        nextBtn.disabled = page.stop >= page.total;
      };

      const load = async (from) => {
        const mySeq = ++seq;
        start = Math.max(0, from);
        head.textContent = `${channel} · loading…`;
        const query = new URLSearchParams({ start, stop: start + PAGE });
        if (sequence()) query.set("sequence", sequence());
        try {
          const payload = await store.api(
            `/api/node/${nodeId}/entries/${encodeURIComponent(channel)}?${query}`);
          if (mySeq !== seq) return;
          page = payload;
          // Relative times need a fixed origin: the first frame of the
          // listing, fetched once so paging does not re-baseline the column.
          if (origin === null && payload.entries.length && start === 0) {
            origin = payload.entries[0].timestamp ?? null;
          }
          paint();
        } catch (err) {
          if (mySeq === seq) head.textContent = String(err);
        }
      };

      // Page to whichever entry holds a global frame index (the "here"
      // button and the current-frame highlight both need it).
      const pageToFrame = async (frame) => {
        // The track is already cached client-side by the data panels; reuse
        // it rather than asking the server to walk the listing again.
        try {
          const { indices } = await store.channelFrames(nodeId, channel);
          if (!indices.length) return load(Math.max(0, frame - PAGE / 2));
          let lo = 0, hi = indices.length - 1, best = 0;
          while (lo <= hi) {
            const m = (lo + hi) >> 1;
            if (indices[m] <= frame) { best = m; lo = m + 1; } else hi = m - 1;
          }
          // `best` counts the whole track; a sequence filter shifts the page
          // origin, so let the server-side total clamp it.
          await load(Math.max(0, best - Math.floor(PAGE / 4)));
        } catch {
          await load(0);
        }
      };

      const find = async () => {
        const text = findInput.value.trim();
        if (!text) return;
        try {
          const { matches } = await store.locate(nodeId, text, {
            channel, sequence: sequence(),
          });
          if (matches.length) {
            await pageToFrame(matches[0].index);
            store.setFrame(matches[0].index);
            return;
          }
        } catch { /* fall through to the row reading */ }
        if (/^\d+$/.test(text)) await load(Number(text));
      };

      prevBtn.addEventListener("click", () => load(start - PAGE));
      nextBtn.addEventListener("click", () => load(start + PAGE));
      hereBtn.addEventListener("click", () => pageToFrame(store.getFrame()));
      seqSel.addEventListener("change", () => { origin = null; load(0); });
      findInput.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        e.preventDefault();
        find();
      });

      // Repaint only: the highlighted row moves with the timeline, but
      // scrubbing must not drag the page around under the user.
      const offFrame = store.onFrame(paint);
      panel.onRemove = offFrame;

      (async () => {
        try {
          const detail = await store.detailAt(nodeId);
          const seqs = detail.sequences || [];
          fillSelect(seqSel, [
            ["", "all sequences"],
            ...seqs.map((s) => [s.id, s.id]),
          ]);
          seqSel.hidden = seqs.length < 2;
        } catch { /* the listing works without the sequence filter */ }
        load(0);
      })();
    },
  };
}
