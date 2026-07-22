// Shared front state: global frame index, selected node, and a per-node
// sample cache so several panels bound to the same node share one request
// and one decode.

import { decodeArray } from "./arrays.js";

// ?snapshot: fetch synchronously so the page is fully rendered by the load
// event -- deterministic captures (headless screenshots, visual tests).
const SNAPSHOT = new URLSearchParams(location.search).has("snapshot");

export async function api(path) {
  if (SNAPSHOT) {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", path, false);
    xhr.send();
    if (xhr.status !== 200) throw new Error(`${path} -> HTTP ${xhr.status}`);
    return JSON.parse(xhr.responseText);
  }
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

export async function apiPost(path, body) {
  if (SNAPSHOT) {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path, false);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify(body || {}));
    if (xhr.status !== 200) throw new Error(`${path} -> HTTP ${xhr.status} ${xhr.responseText}`);
    return JSON.parse(xhr.responseText);
  }
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status} ${await res.text()}`);
  return res.json();
}

/* ----------------------------------------------------------- frame index */

let frame = Number(new URLSearchParams(location.search).get("frame")) || 0;
const frameListeners = new Set();
const sampleCache = new Map(); // nodeId -> {index, promise}

export const getFrame = () => frame;

export function setFrame(index) {
  if (index === frame) return;
  frame = index;
  sampleCache.clear();
  for (const fn of frameListeners) fn(frame);
}

export function onFrame(fn) {
  frameListeners.add(fn);
  return () => frameListeners.delete(fn);
}

/* ---------------------------------------------------------- frame range */
// Optional restriction of the global slider to one sequence's frames:
// {start, stop, label} or null (all frames). Set from the inspector's
// sequences table; the topbar shows the active label with a clear button.

let range = null;
const rangeListeners = new Set();

export const getRange = () => range;

export function setRange(next) {
  range = next;
  for (const fn of rangeListeners) fn(range);
}

export function onRange(fn) {
  rangeListeners.add(fn);
  return () => rangeListeners.delete(fn);
}

/* ---------------------------------------------------------- frame track */
// Optional per-channel timeline: {nodeId, channel, indices} or null. When
// set, the topbar slider steps only through those global frame indices
// (lidar events only, camera events only… on asynchronous datasets).

let track = null;
const trackListeners = new Set();

export const getTrack = () => track;

export function setTrack(next) {
  track = next;
  for (const fn of trackListeners) fn(track);
}

export function onTrack(fn) {
  trackListeners.add(fn);
  return () => trackListeners.delete(fn);
}

/* ------------------------------------------------------------ selection */

let selected = null;
const selectListeners = new Set();

export const getSelected = () => selected;

export function setSelected(nodeId) {
  selected = nodeId;
  for (const fn of selectListeners) fn(nodeId);
}

export function onSelect(fn) {
  selectListeners.add(fn);
  return () => selectListeners.delete(fn);
}

/* --------------------------------------------------------------- target */
// The DESIGNATED example a catalog try applies to: one channel of one node,
// optionally pinned to a precise sample index. Set from data panels (try) or
// inspector rows; the catalog panel displays and uses it.

let target = null; // {nodeId, nodeLabel, channel, len, index|null}
const targetListeners = new Set();

export const getTarget = () => target;

export function setTarget(next) {
  target = next;
  for (const fn of targetListeners) fn(target);
}

export function onTarget(fn) {
  targetListeners.add(fn);
  return () => targetListeners.delete(fn);
}

/* ----------------------------------------------------------------- data */

const detailCache = new Map(); // nodeId -> promise (details are frame-free)

export function detailAt(nodeId) {
  if (!detailCache.has(nodeId)) {
    detailCache.set(nodeId, api(`/api/node/${nodeId}`));
  }
  return detailCache.get(nodeId);
}

// Global frame indices carrying one channel at a node (the per-channel
// timeline). Cached for the session: the mapping is static.
const framesCache = new Map(); // `${nodeId}:${channel}` -> promise

export function channelFrames(nodeId, channel) {
  const key = `${nodeId}:${channel}`;
  if (!framesCache.has(key)) {
    framesCache.set(key,
      api(`/api/node/${nodeId}/frames/${encodeURIComponent(channel)}`));
  }
  return framesCache.get(key);
}

// One channel of one PRECISE frame (not the global frame): used to hold the
// last available data when the bound channel is absent at the current
// frame. Cached for the session: frames are immutable while serving.
const oneCache = new Map(); // `${nodeId}:${index}:${channel}` -> promise

export function sampleOne(nodeId, index, channel) {
  const key = `${nodeId}:${index}:${channel}`;
  if (!oneCache.has(key)) {
    const promise = api(
      `/api/node/${nodeId}/sample/${index}?channels=${encodeURIComponent(channel)}`
    ).then((payload) => {
      const encoded = payload.channels[channel];
      return {
        index,
        timestamp: payload.timestamp ?? null,
        arr: encoded === undefined || encoded.repr !== undefined
          ? null
          : decodeArray(encoded),
      };
    });
    oneCache.set(key, promise);
  }
  return oneCache.get(key);
}

// Decoded channels of one frame at a node (clamped to its length): the
// CURRENT global frame by default, or an explicit override (panels locked
// on their own frame). Cached per node until the index changes; shared by
// every panel bound to the same node+index.
export function sampleAt(nodeId, len, override = null) {
  const index = Math.max(0, Math.min(override ?? frame, (len ?? 1) - 1));
  const hit = sampleCache.get(nodeId);
  if (hit && hit.index === index) return hit.promise;
  const promise = api(`/api/node/${nodeId}/sample/${index}`).then((payload) => {
    const channels = {};
    for (const [name, encoded] of Object.entries(payload.channels)) {
      channels[name] = encoded.repr !== undefined ? encoded : decodeArray(encoded);
    }
    return {
      index,
      len: payload.len,
      channels,
      frame: payload.frame ?? null,
      timestamp: payload.timestamp ?? null,
    };
  });
  sampleCache.set(nodeId, { index, promise });
  return promise;
}
