"""Transform / preprocessor catalog: introspection + sandboxed try-apply.

Walks the satellite packages (``apairo_transform``, ``apairo_preprocess``)
and lists their public callables with a parameter schema derived from each
signature (defaults give the control type, projector's kwargs->sliders
pattern).  ``apply()`` runs ONE catalog entry on one frame's channel -- a pure
function call on arrays already in memory, never a mutation of the served
datasets, and never free-form code: only catalog ids are accepted.

``snippet()`` renders the exact code to paste into the training script --
the GUI lets you *try* a transform, the script remains the source of truth.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass, field

import numpy as np

DEFAULT_PACKAGES = ("apairo_transform", "apairo_preprocess")

_SCALARS = {bool: "bool", int: "int", float: "float", str: "str"}


@dataclass
class Param:
    name: str
    type: str          # bool | int | float | str | any
    default: object = None
    required: bool = False


@dataclass
class Entry:
    id: str
    name: str
    module: str
    kind: str          # class | function
    is_preprocessor: bool
    doc: str | None
    signature: str | None
    params: list[Param] = field(default_factory=list)
    obj: object = None  # the class / function itself (not serialized)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "module": self.module,
            "kind": self.kind,
            "is_preprocessor": self.is_preprocessor,
            "doc": self.doc,
            "signature": self.signature,
            "params": [vars(p) for p in self.params],
        }


def _param_schema(obj) -> list[Param]:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    params: list[Param] = []
    for p in signature.parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            kind = _SCALARS.get(p.annotation, "any")
            params.append(Param(p.name, kind, None, required=True))
        else:
            kind = _SCALARS.get(type(p.default), "any")
            params.append(Param(p.name, kind, p.default if kind != "any" else None))
    return params


def _coerce(value, kind: str):
    if value is None or value == "":
        return None
    if kind == "bool":
        return value in (True, "true", "1", 1)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "str":
        return str(value)
    # "any": accept JSON scalars as-is, try to parse numbers out of strings.
    if isinstance(value, str):
        try:
            return float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError:
            return value
    return value


class Catalog:
    """Public callables of the satellite packages, indexed by entry id."""

    def __init__(self, packages: tuple[str, ...] = DEFAULT_PACKAGES) -> None:
        self.entries: dict[str, Entry] = {}
        self.missing: list[str] = []  # packages that failed to import
        for package in packages:
            self._scan_package(package)

    def _scan_package(self, package: str) -> None:
        try:
            root = importlib.import_module(package)
        except Exception:
            self.missing.append(package)
            return
        self._scan_module(root)
        for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
            try:
                self._scan_module(importlib.import_module(info.name))
            except Exception:
                continue  # a submodule with a heavy/broken import: skip it

    def _scan_module(self, module) -> None:
        for name, obj in vars(module).items():
            if name.startswith("_") or not callable(obj):
                continue
            if getattr(obj, "__module__", None) != module.__name__:
                continue  # re-export: listed where it is defined
            kind = "class" if inspect.isclass(obj) else "function"
            if kind == "function" and not inspect.isfunction(obj):
                continue
            entry = Entry(
                id=f"{module.__name__}.{name}",
                name=name,
                module=module.__name__,
                kind=kind,
                is_preprocessor=bool(
                    hasattr(obj, "process") and hasattr(obj, "input_keys")
                ),
                doc=inspect.getdoc(obj),
                signature=None,
                params=_param_schema(obj),
                obj=obj,
            )
            try:
                entry.signature = str(inspect.signature(obj))
            except (TypeError, ValueError):
                pass
            self.entries[entry.id] = entry

    # ----------------------------------------------------------------- API

    def to_dict(self) -> dict:
        groups: dict[str, list[dict]] = {}
        for entry in sorted(self.entries.values(), key=lambda e: e.id):
            groups.setdefault(entry.module, []).append(entry.to_dict())
        return {
            "modules": [{"module": m, "entries": es} for m, es in groups.items()],
            "missing": self.missing,
        }

    def _instantiate(self, entry: Entry, kwargs: dict):
        schema = {p.name: p.type for p in entry.params}
        coerced = {
            k: _coerce(v, schema.get(k, "any"))
            for k, v in (kwargs or {}).items()
            if v is not None and v != ""
        }
        if entry.kind == "class":
            return entry.obj(**coerced), coerced
        return functools.partial(entry.obj, **coerced), coerced

    def apply(self, entry_id: str, kwargs: dict, channels: dict, channel: str):
        """Run one entry on one frame -> the resulting array.

        Per-channel callables get the channel array; preprocessors (and
        sample-level callables, as a fallback) get a Sample carrying all the
        frame's channels.  Raises with the underlying message on failure --
        the UI shows it verbatim.
        """
        entry = self.entries[entry_id]
        fn, _ = self._instantiate(entry, kwargs)
        from apairo.core.sample import Sample

        if entry.is_preprocessor:
            return np.asarray(fn(Sample(data=dict(channels))))
        try:
            return np.asarray(fn(np.asarray(channels[channel])))
        except Exception as array_err:
            # Sample-level transform? Retry with the full sample and report
            # the changed / newly published channel.
            try:
                result = fn(Sample(data=dict(channels)))
            except Exception:
                raise array_err from None
            if hasattr(result, "data"):
                new = [k for k in result.data if k not in channels]
                key = new[0] if new else channel
                return np.asarray(result.data[key])
            return np.asarray(result)

    def snippet(self, entry_id: str, kwargs: dict, channel: str) -> str:
        entry = self.entries[entry_id]
        _, coerced = self._instantiate(entry, kwargs)
        args = ", ".join(f"{k}={v!r}" for k, v in coerced.items())
        output = f"{channel}_{_snake(entry.name)}"
        if entry.kind == "class":
            call = f"{entry.name}({args})"
            imports = f"from {entry.module} import {entry.name}"
        else:
            call = f"partial({entry.name}, {args})" if args else entry.name
            imports = f"from functools import partial\n" * bool(args) + \
                f"from {entry.module} import {entry.name}"
        if entry.is_preprocessor:
            return f"{imports}\n\nds.transform({call})"
        return (
            f"{imports}\n\n"
            f'ds.transform("{channel}", {call}, output="{output}")'
        )


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
