// Panel manager: addable / closable / dockable panels in a split tree.
// Tree, drag-dock and relayout algorithms adapted from projector's
// web/src/panels.js; panel content is generic here (a spec with a build()
// hook) instead of projector's channel-bound views.
//
// Layout: {t:"leaf", id} | {t:"split", dir:"row"|"col", kids}. Docking
// left/right of a panel splits THAT cell, not the whole workspace.

import { growGutter } from "./resize.js";

export class PanelManager {
  constructor(stackEl) {
    this.stack = stackEl;
    this.panels = [];
    this.tree = null;
    this._seq = 0;
    this._hintEl = null;
    this.onChanged = null; // layout mutated (add/close/dock) -> persist hook
  }

  byKey(key) {
    return this.panels.find((p) => p.key === key);
  }

  // spec: {key, tag, title, build(panel), state?(), binding?} -- build fills
  // panel.body and may append controls to panel.controls; it may set
  // panel.onRemove for cleanup.
  add(spec, newColumn = false) {
    const panel = this._createPanel(spec);
    this._place(panel.id, newColumn);
    this._relayout();
    if (this.onChanged) this.onChanged();
    return panel;
  }

  remove(id) {
    const i = this.panels.findIndex((p) => p.id === id);
    if (i < 0) return;
    const [p] = this.panels.splice(i, 1);
    if (p.onRemove) p.onRemove();
    this._dropFromLayout(id);
    this._relayout();
    if (this.onChanged) this.onChanged();
  }

  _createPanel(spec) {
    const id = ++this._seq;
    const el = document.createElement("section");
    el.className = "vpanel";
    el.style.flex = "1 1 0";

    const head = document.createElement("div");
    head.className = "vpanel-head";
    const tag = document.createElement("span");
    tag.className = "vpanel-tag";
    tag.textContent = spec.tag;
    const title = document.createElement("span");
    title.className = "vpanel-title";
    title.textContent = spec.title;
    const controls = document.createElement("span");
    controls.className = "vpanel-controls";
    const close = document.createElement("button");
    close.className = "tbtn";
    close.textContent = "×";
    close.title = "Close panel";
    head.append(tag, title, controls, close);

    const body = document.createElement("div");
    body.className = "vpanel-body";
    el.append(head, body);

    const panel = {
      id, el, head, body, controls, title,
      key: spec.key, tag: spec.tag, spec,
      onRemove: null,
    };
    close.onclick = () => this.remove(id);
    this._bindDrag(panel, head);
    this.panels.push(panel);
    spec.build(panel);
    return panel;
  }

  // ------------------------------------------------- split-tree layout
  // Root is a row of columns; default placement stacks into the last column,
  // newColumn=true appends a column on the right.
  _place(id, newColumn) {
    const leaf = { t: "leaf", id };
    if (!this.tree) { this.tree = leaf; return; }
    if (this.tree.t !== "split" || this.tree.dir !== "row") {
      this.tree = { t: "split", dir: "row", kids: [this.tree] };
    }
    const kids = this.tree.kids;
    if (newColumn) { kids.push(leaf); return; }
    const target = kids[kids.length - 1];
    if (target.t === "split" && target.dir === "col") target.kids.push(leaf);
    else kids[kids.length - 1] = { t: "split", dir: "col", kids: [target, leaf] };
  }

  _dropFromLayout(id) {
    const prune = (node) => {
      if (!node) return null;
      if (node.t === "leaf") return node.id === id ? null : node;
      node.kids = node.kids.map(prune).filter(Boolean);
      if (!node.kids.length) return null;
      if (node.kids.length === 1) return node.kids[0];
      return node;
    };
    this.tree = prune(this.tree);
  }

  _byId(id) {
    return this.panels.find((p) => p.id === id);
  }

  // ------------------------------------------------- drag-dock
  // Drag a panel by its header onto another: left/right quarter = split into
  // a row beside it, top/bottom half = stack above/below it.
  _bindDrag(panel, head) {
    head.style.cursor = "grab";
    head.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest("select, button, input")) return;
      e.preventDefault();
      const start = [e.clientX, e.clientY];
      let dragging = false;
      const move = (ev) => {
        if (!dragging && Math.hypot(ev.clientX - start[0], ev.clientY - start[1]) < 6) return;
        if (!dragging) { dragging = true; panel.el.classList.add("dragging"); }
        this._showHint(ev, panel);
      };
      const up = (ev) => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        this._clearHint();
        if (!dragging) return;
        panel.el.classList.remove("dragging");
        const t = this._dropTarget(ev, panel);
        if (t) this._dock(panel.id, t.panel.id, t.zone);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
  }

  _dropTarget(ev, dragPanel) {
    const el = document.elementFromPoint(ev.clientX, ev.clientY);
    const host = el && el.closest(".vpanel");
    if (!host) return null;
    const p = this.panels.find((q) => q.el === host);
    if (!p || p.id === dragPanel.id) return null;
    const r = host.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width;
    const fy = (ev.clientY - r.top) / r.height;
    const zone = fx < 0.25 ? "left" : fx > 0.75 ? "right" : fy < 0.5 ? "top" : "bottom";
    return { panel: p, zone };
  }

  _showHint(ev, dragPanel) {
    const t = this._dropTarget(ev, dragPanel);
    this._clearHint();
    if (!t) return;
    if (!this._hintEl) {
      this._hintEl = document.createElement("div");
      this._hintEl.className = "drop-hint";
    }
    this._hintEl.dataset.zone = t.zone;
    t.panel.el.appendChild(this._hintEl);
  }

  _clearHint() {
    if (this._hintEl) this._hintEl.remove();
  }

  _dock(dragId, targetId, zone) {
    if (dragId === targetId) return;
    this._dropFromLayout(dragId);
    const dir = zone === "left" || zone === "right" ? "row" : "col";
    const first = zone === "left" || zone === "top";
    const drag = { t: "leaf", id: dragId };
    const place = (node, parent) => {
      if (node.t === "leaf") {
        if (node.id !== targetId) return false;
        if (parent && parent.dir === dir) {
          const i = parent.kids.indexOf(node);
          parent.kids.splice(first ? i : i + 1, 0, drag);
        } else {
          const split = { t: "split", dir, kids: first ? [drag, node] : [node, drag] };
          if (parent) parent.kids[parent.kids.indexOf(node)] = split;
          else this.tree = split;
        }
        return true;
      }
      return node.kids.some((k) => place(k, node));
    };
    if (!this.tree) this.tree = drag;
    else if (!place(this.tree, null)) {
      this.tree = { t: "split", dir: "row", kids: [this.tree, drag] };
    }
    this._relayout();
    if (this.onChanged) this.onChanged();
  }

  // ------------------------------------------------- persistence
  serialize() {
    const enc = (node) => {
      if (!node) return null;
      if (node.t === "leaf") {
        const p = this._byId(node.id);
        return p ? { key: p.key, binding: p.spec.binding ?? null } : null;
      }
      return { dir: node.dir, kids: node.kids.map(enc).filter(Boolean) };
    };
    return enc(this.tree);
  }

  // factory(key, binding) -> spec | null; false when nothing was restored.
  restore(saved, factory) {
    let added = 0;
    const build = (node) => {
      if (!node) return null;
      if (node.dir) {
        const kids = (node.kids || []).map(build).filter(Boolean);
        if (!kids.length) return null;
        if (kids.length === 1) return kids[0];
        return { t: "split", dir: node.dir === "col" ? "col" : "row", kids };
      }
      const spec = factory(node.key, node.binding);
      if (!spec) return null;
      const p = this._createPanel(spec);
      added++;
      return { t: "leaf", id: p.id };
    };
    const tree = build(saved);
    if (tree && added) {
      this.tree = tree;
      this._relayout();
    }
    return added > 0;
  }

  // ------------------------------------------------- relayout
  _relayout() {
    const render = (node) => {
      if (node.t === "leaf") {
        const p = this._byId(node.id);
        return p ? p.el : null;
      }
      const parts = node.kids.map(render).filter(Boolean);
      if (!parts.length) return null;
      if (parts.length === 1) return parts[0];
      if (!node.el) {
        node.el = document.createElement("div");
        node.el.style.flex = "1 1 0";
      }
      node.el.className = `vsplit vsplit-${node.dir}`;
      node.el.replaceChildren();
      parts.forEach((el, i) => {
        if (i > 0) {
          const g = document.createElement("div");
          g.className = node.dir === "row" ? "gutter gutter-v" : "gutter gutter-h";
          node.el.appendChild(g);
          growGutter(g, parts[i - 1], el, node.dir === "row" ? "x" : "y", node.dir === "row" ? 160 : 90);
        }
        node.el.appendChild(el);
      });
      return node.el;
    };
    this.stack.replaceChildren();
    if (!this.tree) return;
    const rootEl = render(this.tree);
    if (rootEl) this.stack.appendChild(rootEl);
  }
}

export function fillSelect(sel, pairs) {
  sel.replaceChildren();
  for (const [value, label] of pairs) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    sel.appendChild(o);
  }
}
