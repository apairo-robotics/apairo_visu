"""Interactive 3-D LiDAR viewer for apairo datasets -- multi-pipeline edition.

Usage (single viewport, no transform):
    from apairo_visu import LidarViewer, load_label_config
    cfg = load_label_config("goose")
    LidarViewer.launch(dataset, label_cfg=cfg)

Usage (compare multiple pipelines side-by-side):
    from apairo_visu import LidarViewer, Pipeline, load_label_config
    cfg = load_label_config("goose")
    LidarViewer.launch(dataset, label_cfg=cfg, pipelines=[
        Pipeline("Raw",     []),
        Pipeline("Model A", [preprocess, model_a]),
        Pipeline("Model B", [preprocess, model_b]),
    ])

Pipelines execute in parallel via a thread pool; each viewport updates as soon
as its pipeline finishes (useful when inference is slow).

Keyboard shortcuts:
    Right / L   next frame
    Left  / H   previous frame
    R           reset camera (all viewports)
    B           bird's-eye (top-down) view
    T           cycle colour mode  (Semantic -> Intensity -> Height)
    J           toggle trajectory overlay
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError as exc:  # an optional extra: only this viewer renders with Open3D
    raise ImportError(
        "The Open3D viewer (LidarViewer, `apairo visu`) needs Open3D, which is an "
        "optional extra: pip install 'apairo-visu[open3d]'. The studio and the "
        "Rerun backend do not need it."
    ) from exc

from .colors import (
    auto_color_map,
    height_colors,
    intensity_colors,
    labels_to_colors,
    normalize_color_map,
)
from .config import ViewConfig
from .geometry import (
    has_extent,
    make_lineset,
    make_point_cloud,
    pick_nearest,
    poses_to_local,
    to_numpy,
)
from .pipeline import Pipeline

PANEL_W = 290
POINT_SIZE = 2.5
DISPLAY_MODES = ["Semantic", "Intensity", "Height"]
BANNER_H = 22   # pixels -- pipeline-name banner above each viewport

C_TRAJ_PAST   = [0.20, 0.60, 1.00]
C_TRAJ_FUTURE = [1.00, 0.60, 0.10]


class LidarViewer:
    """Interactive 3-D LiDAR viewer for any apairo synchronous dataset.

    Supports one or more named pipelines displayed side-by-side for comparison.
    Each pipeline is a sequence of ``(pts, labels) -> (pts, labels)`` steps
    applied before rendering (see :class:`apairo_visu.Pipeline`).  Pipelines run
    in parallel via a thread pool; each viewport updates as soon as its pipeline
    finishes.

    Args:
        dataset:   Any apairo synchronous dataset (``dataset[idx]`` returns a
                   ``Sample`` with a ``data`` dict).
        view_cfg:  Which keys to extract from each Sample.  Defaults to
                   ``point_key="lidar"`` and ``label_key="labels"``.
        label_cfg: Dict with ``color_map``, ``semantic_map``, and optionally
                   ``traversable_map``.  Use ``load_label_config(name)`` for
                   built-in configs.  Pass ``None`` to skip label colouring.
        label_cfgs: Per-pipeline label configs; entries fall back to
                   ``label_cfg`` when missing.
        poses:     Optional list of 4x4 numpy pose matrices (``T_world_sensor``),
                   one per frame, for a trajectory overlay (viewport 0 only).
        start_idx: First frame to display.
        pipelines: List of :class:`apairo_visu.Pipeline` objects -- one viewport
                   per entry.  Defaults to ``[Pipeline("Raw", [])]``.
    """

    def __init__(
        self,
        dataset,
        view_cfg: ViewConfig | None = None,
        label_cfg: dict | None = None,
        label_cfgs: list[dict | None] | None = None,
        poses: list[np.ndarray] | None = None,
        start_idx: int = 0,
        pipelines: list[Pipeline] | None = None,
    ) -> None:
        self.dataset     = dataset
        self.cfg         = view_cfg or ViewConfig()
        self.current_idx = start_idx
        self._poses      = poses
        self._pipelines  = pipelines if pipelines is not None else [Pipeline("Raw", [])]

        # Per-pipeline label setup
        # label_cfgs overrides label_cfg per pipeline; missing entries fall back to label_cfg.
        n_pipe = len(self._pipelines)
        if label_cfgs is None:
            resolved_cfgs: list[dict | None] = [label_cfg] * n_pipe
        else:
            resolved_cfgs = list(label_cfgs)
            while len(resolved_cfgs) < n_pipe:
                resolved_cfgs.append(label_cfg)

        self._color_maps:          list[dict]          = []
        self._semantic_maps:       list[dict[int, str]] = []
        self._active_classes_list: list[set[int]]      = []
        self._class_ids_list:      list[list[int]]     = []
        self._label_fixed:         list[bool]          = []

        for cfg in resolved_cfgs:
            if cfg is not None:
                cmap = normalize_color_map(cfg["color_map"])
                smap = {int(k): v for k, v in cfg.get("semantic_map", {}).items()}
                cids = sorted(smap.keys())
                fixed = True
            else:
                cmap  = {}
                smap  = {}
                cids  = []
                fixed = False
            self._color_maps.append(cmap)
            self._semantic_maps.append(smap)
            self._active_classes_list.append(set(cids))
            self._class_ids_list.append(cids)
            self._label_fixed.append(fixed)

        self._has_labels    = self.cfg.label_key is not None
        self._display_mode: int = 0 if (self._has_labels and self._label_fixed[0]) else 2

        # Per-viewport state (one entry per pipeline)
        n = n_pipe
        self._scenes:            list[gui.SceneWidget]        = []
        self._banner_labels:     list[gui.Label]              = []
        self._mats:              list[rendering.MaterialRecord] = []
        self._cached_pts:        list[np.ndarray | None]      = [None] * n
        self._cached_labels:     list[np.ndarray | None]      = [None] * n
        self._camera_initialized: list[bool]                  = [False] * n
        self._pipeline_active:   list[bool]                   = [True]  * n

        # Raw frame cache -- shared across all pipelines, refreshed on navigation
        self._raw_pts:    np.ndarray | None = None
        self._raw_labels: np.ndarray | None = None

        # Parallel execution
        self._executor   = ThreadPoolExecutor(max_workers=max(n, 1))
        self._refresh_id = 0  # bumped on each navigation; discards stale callbacks

        # Shared panel widget refs
        self._window:              gui.Window | None       = None
        self._panel:               gui.Vert   | None       = None
        self._lbl_frame:           gui.Label  | None       = None
        self._lbl_npts:            gui.Label  | None       = None
        self._lbl_stats_list:      list[gui.Label | None]  = []
        self._slider_frame:        gui.Slider | None       = None
        self._timing_labels:       list[gui.Label]         = []
        self._pipeline_checkboxes: list[gui.Checkbox]      = []
        self._checkboxes:          dict[int, gui.Checkbox] = {}
        self._checkbox_swatches:   dict[int, gui.ImageWidget] = {}
        self._filter_pipe_idx:     int                     = 0
        self._filter_combo:        gui.Combobox | None     = None
        self._mode_combo:          gui.Combobox | None     = None
        self._cb_traj:             gui.Checkbox | None     = None
        self._show_traj:           bool                    = False
        self._hover_label:         gui.Label  | None       = None
        self._last_hover_pos:      tuple[int, int]         = (-1, -1)
        self._last_hover_time:     float                   = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def launch(
        dataset,
        view_cfg: ViewConfig | None = None,
        label_cfg: dict | None = None,
        label_cfgs: list[dict | None] | None = None,
        poses: list[np.ndarray] | None = None,
        start_idx: int = 0,
        pipelines: list[Pipeline] | None = None,
    ) -> None:
        """Create the GUI application and block until the window is closed.

        See the class docstring for the argument meanings.
        """
        app = gui.Application.instance
        app.initialize()
        viewer = LidarViewer(dataset, view_cfg, label_cfg, label_cfgs, poses, start_idx, pipelines)
        viewer._build_window()
        app.run()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        app = gui.Application.instance
        w   = app.create_window("apairo -- LiDAR Viewer", 1500, 900)
        self._window = w
        em = w.theme.font_size

        multi = len(self._pipelines) > 1

        # --- N viewports (one per pipeline) ---
        for pipeline in self._pipelines:
            mat = rendering.MaterialRecord()
            mat.shader     = "defaultUnlit"
            mat.point_size = POINT_SIZE
            self._mats.append(mat)

            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(w.renderer)
            scene.scene.set_background([0.08, 0.08, 0.08, 1.0])
            scene.set_on_key(self._on_key)
            self._scenes.append(scene)
            w.add_child(scene)

            if multi:
                banner = gui.Label(f"  {pipeline.name}")
                banner.text_color = gui.Color(0.9, 0.85, 0.5)
                self._banner_labels.append(banner)
                w.add_child(banner)

        # --- Left panel ---
        panel = gui.Vert(int(0.4 * em), gui.Margins(int(0.6 * em)))

        # Navigation
        panel.add_child(self._section_label("Navigation", em))

        self._lbl_frame = gui.Label("-- / --")
        self._lbl_frame.text_color = gui.Color(0.85, 0.85, 0.85)
        panel.add_child(self._lbl_frame)

        self._lbl_npts = gui.Label("Points: --")
        self._lbl_npts.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(self._lbl_npts)

        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(0, len(self.dataset) - 1)
        slider.int_value = self.current_idx
        slider.set_on_value_changed(self._on_slider_changed)
        self._slider_frame = slider
        panel.add_child(slider)

        nav_row = gui.Horiz(int(0.3 * em))
        btn_prev = gui.Button("< Prev"); btn_prev.set_on_clicked(self._on_prev)
        btn_next = gui.Button("Next >"); btn_next.set_on_clicked(self._on_next)
        nav_row.add_stretch(); nav_row.add_child(btn_prev)
        nav_row.add_child(btn_next); nav_row.add_stretch()
        panel.add_child(nav_row)

        cam_row = gui.Horiz(int(0.3 * em))
        btn_bev   = gui.Button("BEV  [B]");   btn_bev.set_on_clicked(self._look_bev)
        btn_reset = gui.Button("Reset  [R]"); btn_reset.set_on_clicked(self._reset_camera)
        cam_row.add_stretch(); cam_row.add_child(btn_bev)
        cam_row.add_child(btn_reset); cam_row.add_stretch()
        panel.add_child(cam_row)

        if multi:
            btn_sync = gui.Button("Sync cam")
            btn_sync.set_on_clicked(self._sync_cameras)
            row = gui.Horiz(int(0.3 * em))
            row.add_stretch(); row.add_child(btn_sync); row.add_stretch()
            panel.add_child(row)

        panel.add_child(gui.Label(""))

        # Colour mode
        panel.add_child(self._section_label("Colour mode  [T]", em))
        combo = gui.Combobox()
        for m in DISPLAY_MODES:
            combo.add_item(m)
        combo.selected_index = self._display_mode
        combo.set_on_selection_changed(self._on_mode_changed)
        self._mode_combo = combo
        panel.add_child(combo)

        panel.add_child(gui.Label(""))

        # Trajectory overlay (when poses provided, shown in viewport 0 only)
        if self._poses is not None:
            panel.add_child(self._section_label("Overlays", em))
            cb_traj = gui.Checkbox("Trajectory  [J]")
            cb_traj.checked = False
            cb_traj.set_on_checked(self._on_traj_toggle)
            self._cb_traj = cb_traj
            panel.add_child(cb_traj)
            panel.add_child(gui.Label(""))

        # Active pipeline toggles -- check/uncheck to show/hide individual viewports
        if multi:
            panel.add_child(self._section_label("Active pipelines", em))
            for i, pipeline in enumerate(self._pipelines):
                cb = gui.Checkbox(pipeline.name)
                cb.checked = True
                cb.set_on_checked(
                    lambda checked, idx=i: self._on_pipeline_toggle(idx, checked)
                )
                self._pipeline_checkboxes.append(cb)
                panel.add_child(cb)
            panel.add_child(gui.Label(""))

        # Pipelines section: timing + per-pipeline class distribution
        panel.add_child(self._section_label("Pipelines", em))
        for i, pipeline in enumerate(self._pipelines):
            timing_lbl = gui.Label(f"{pipeline.name}: --")
            timing_lbl.text_color = gui.Color(0.7, 0.7, 0.7)
            self._timing_labels.append(timing_lbl)
            panel.add_child(timing_lbl)

            if self._has_labels:
                stats_lbl = gui.Label("")
                stats_lbl.text_color = gui.Color(0.65, 0.65, 0.65)
                self._lbl_stats_list.append(stats_lbl)
                panel.add_child(stats_lbl)
            else:
                self._lbl_stats_list.append(None)

        panel.add_child(gui.Label(""))

        # Class filter — one filter section, pipeline selector if multiple configs differ
        if self._has_labels:
            # Collect class IDs for all fixed-config pipelines (union)
            fixed_indices = [i for i, f in enumerate(self._label_fixed) if f]
            all_class_ids: list[int] = sorted({
                cid
                for i in fixed_indices
                for cid in self._class_ids_list[i]
            })

            if all_class_ids:
                panel.add_child(self._section_label("Filter classes", em))

                # Pipeline selector (only when configs differ across pipelines)
                if multi and len(fixed_indices) > 1:
                    filter_row = gui.Horiz(int(0.3 * em))
                    filter_row.add_child(gui.Label("Pipeline:"))
                    fc = gui.Combobox()
                    for i in fixed_indices:
                        fc.add_item(self._pipelines[i].name)
                    fc.selected_index = 0
                    fc.set_on_selection_changed(self._on_filter_pipe_changed)
                    self._filter_combo = fc
                    self._filter_pipe_idx = fixed_indices[0]
                    filter_row.add_child(fc)
                    panel.add_child(filter_row)
                elif fixed_indices:
                    self._filter_pipe_idx = fixed_indices[0]

                toggle_row = gui.Horiz(int(0.3 * em))
                btn_show = gui.Button("Show all"); btn_show.set_on_clicked(self._on_show_all)
                btn_hide = gui.Button("Hide all"); btn_hide.set_on_clicked(self._on_hide_all)
                toggle_row.add_child(btn_show); toggle_row.add_child(btn_hide)
                panel.add_child(toggle_row)

                scroll = gui.ScrollableVert(
                    int(0.3 * em), gui.Margins(0, 0, int(0.3 * em), 0)
                )
                p0 = self._filter_pipe_idx
                for cls_id in all_class_ids:
                    name = self._semantic_maps[p0].get(cls_id, str(cls_id))
                    rgb  = self._color_maps[p0].get(cls_id, [128, 128, 128])
                    tile = np.full((14, 14, 3), rgb, dtype=np.uint8)

                    cb = gui.Checkbox(f"{cls_id}: {name}")
                    cb.checked = cls_id in self._active_classes_list[p0]
                    cb.set_on_checked(
                        lambda checked, cid=cls_id: self._on_class_toggle(cid, checked)
                    )
                    self._checkboxes[cls_id] = cb

                    img = gui.ImageWidget(o3d.geometry.Image(tile))
                    self._checkbox_swatches[cls_id] = img

                    row = gui.Horiz(int(0.2 * em))
                    row.add_child(img)
                    row.add_child(cb)
                    scroll.add_child(row)

                panel.add_child(scroll)

        w.add_child(panel)
        self._panel = panel

        # Hover info overlay (added last so it renders on top of the scene)
        hover_lbl = gui.Label("--")
        hover_lbl.text_color = gui.Color(1.0, 1.0, 0.5)
        self._hover_label = hover_lbl
        w.add_child(hover_lbl)
        self._scenes[0].set_on_mouse(self._on_mouse)

        w.set_on_layout(self._on_layout)

        self._refresh()

    @staticmethod
    def _section_label(text: str, em: float) -> gui.Label:
        lbl = gui.Label(text.upper())
        lbl.text_color = gui.Color(0.5, 0.8, 1.0)
        return lbl

    def _on_layout(self, _ctx) -> None:
        r     = self._window.content_rect
        self._panel.frame = gui.Rect(r.x, r.y, PANEL_W, r.height)

        n          = len(self._scenes)
        multi      = n > 1
        vp_total_w = r.width - PANEL_W
        lh         = BANNER_H if multi else 0

        # Distribute width only among active viewports; inactive ones are pushed
        # off-screen with a 1x1 frame to avoid a zero-size render target.
        active = [i for i in range(n) if self._pipeline_active[i]]
        vp_w   = vp_total_w / len(active) if active else vp_total_w
        off_x  = r.x + r.width + 10  # off-screen position for hidden viewports

        slot = 0
        for i, scene in enumerate(self._scenes):
            if not self._pipeline_active[i]:
                if multi:
                    self._banner_labels[i].frame = gui.Rect(off_x, r.y, 1, 1)
                scene.frame = gui.Rect(off_x, r.y, 1, 1)
            else:
                x = r.x + PANEL_W + slot * vp_w
                if multi:
                    self._banner_labels[i].frame = gui.Rect(x, r.y, vp_w, lh)
                scene.frame = gui.Rect(x, r.y + lh, vp_w, r.height - lh)
                slot += 1

        if self._hover_label is not None:
            ol_w, ol_h = 175, 115
            vp0_right = r.x + PANEL_W + (vp_w if 0 in active else vp_total_w)
            self._hover_label.frame = gui.Rect(
                vp0_right - ol_w - 8, r.y + lh + 8, ol_w, ol_h
            )

    # ------------------------------------------------------------------
    # Navigation event handlers
    # ------------------------------------------------------------------

    def _on_next(self) -> None:
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        self._refresh()

    def _on_prev(self) -> None:
        self.current_idx = (self.current_idx - 1) % len(self.dataset)
        self._refresh()

    def _on_slider_changed(self, value: float) -> None:
        new_idx = int(value)
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self._refresh()

    def _on_mouse(self, event) -> int:
        if event.type == gui.MouseEvent.Type.MOVE:
            now = time.perf_counter()
            if now - self._last_hover_time > 0.05:
                self._last_hover_time = now
                self._last_hover_pos = (event.x, event.y)
                self._update_hover_info(event.x, event.y)
        return gui.Widget.EventCallbackResult.IGNORED

    def _find_hover_point(self, lx: int, ly: int) -> int | None:
        """Index of the point nearest viewport-local ``(lx, ly)``, or ``None``."""
        pts = self._cached_pts[0]
        if pts is None or len(pts) == 0:
            return None
        try:
            frame = self._scenes[0].frame
            W, H = frame.width, frame.height
            if W <= 0 or H <= 0 or not (0 <= lx < W and 0 <= ly < H):
                return None
            cam  = self._scenes[0].scene.camera
            view = np.asarray(cam.get_view_matrix())         # world -> camera
            proj = np.asarray(cam.get_projection_matrix())   # camera -> clip
            return pick_nearest(pts, view, proj, W, H, lx, ly, radius=15.0)
        except Exception:
            return None

    def _update_hover_info(self, cx: int, cy: int) -> None:
        if self._hover_label is None:
            return

        frame = self._scenes[0].frame
        # cx, cy are window-local; convert to viewport-local
        lx = cx - frame.x
        ly = cy - frame.y

        if not (0 <= lx < frame.width and 0 <= ly < frame.height):
            self._hover_label.text = "--"
            self._window.post_redraw()
            return

        idx = self._find_hover_point(lx, ly)
        if idx is None:
            self._hover_label.text = "--"
            self._window.post_redraw()
            return

        pts    = self._cached_pts[0]
        labels = self._cached_labels[0]
        x, y, z = float(pts[idx, 0]), float(pts[idx, 1]), float(pts[idx, 2])
        dist = float(np.sqrt(x * x + y * y + z * z))

        lines = [
            f" x   {x:+.2f} m",
            f" y   {y:+.2f} m",
            f" z   {z:+.2f} m",
            f" d   {dist:.2f} m",
        ]
        if labels is not None:
            cls_id = int(labels[idx])
            name   = self._semantic_maps[0].get(cls_id, str(cls_id))
            lines.append(f" [{cls_id}: {name}]")
        if pts.shape[1] > self.cfg.intensity_channel:
            lines.append(f" i   {float(pts[idx, self.cfg.intensity_channel]):.3f}")

        self._hover_label.text = "\n".join(lines)
        self._window.post_redraw()

    def _on_mode_changed(self, _text: str, idx: int) -> None:
        self._display_mode = idx
        for i, (pts, labels) in enumerate(zip(self._cached_pts, self._cached_labels)):
            if pts is not None:
                self._update_cloud(i, pts, labels)

    def _on_traj_toggle(self, checked: bool) -> None:
        self._show_traj = checked
        if checked:
            self._update_trajectory()
        else:
            self._scenes[0].scene.remove_geometry("traj_past")
            self._scenes[0].scene.remove_geometry("traj_future")

    def _on_pipeline_toggle(self, i: int, checked: bool) -> None:
        self._pipeline_active[i] = checked
        self._on_layout(None)       # redistribute viewport widths immediately
        self._window.post_redraw()
        if checked and self._cached_pts[i] is None:
            self._run_single_pipeline(i)

    def _run_single_pipeline(self, i: int) -> None:
        """Run pipeline *i* on the cached raw frame and update its viewport."""
        if self._raw_pts is None:
            return
        pts    = self._raw_pts
        labels = self._raw_labels
        current_id = self._refresh_id
        frame_idx  = self.current_idx

        if i < len(self._timing_labels):
            self._timing_labels[i].text = f"{self._pipelines[i].name}: …"

        def _run() -> None:
            t0 = time.perf_counter()
            p_pts, p_labels = self._pipelines[i].run(
                pts.copy(),
                labels.copy() if labels is not None else None,
                frame_idx=frame_idx,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            gui.Application.instance.post_to_main_thread(
                self._window,
                lambda p=p_pts, l=p_labels, t=elapsed_ms:
                    self._on_pipeline_done(current_id, i, p, l, t),
            )

        self._executor.submit(_run)

    def _on_class_toggle(self, cls_id: int, checked: bool) -> None:
        p = self._filter_pipe_idx
        if checked:
            self._active_classes_list[p].add(cls_id)
        else:
            self._active_classes_list[p].discard(cls_id)
        pts, labels = self._cached_pts[p], self._cached_labels[p]
        if pts is not None and self._pipeline_active[p]:
            self._update_cloud(p, pts, labels)

    def _on_show_all(self) -> None:
        p = self._filter_pipe_idx
        self._active_classes_list[p] = set(self._class_ids_list[p])
        for cls_id, cb in self._checkboxes.items():
            cb.checked = cls_id in self._active_classes_list[p]
        pts, labels = self._cached_pts[p], self._cached_labels[p]
        if pts is not None and self._pipeline_active[p]:
            self._update_cloud(p, pts, labels)

    def _on_hide_all(self) -> None:
        p = self._filter_pipe_idx
        self._active_classes_list[p] = set()
        for cb in self._checkboxes.values():
            cb.checked = False
        pts, labels = self._cached_pts[p], self._cached_labels[p]
        if pts is not None and self._pipeline_active[p]:
            self._update_cloud(p, pts, labels)

    def _on_filter_pipe_changed(self, _text: str, combo_idx: int) -> None:
        fixed_indices = [i for i, f in enumerate(self._label_fixed) if f]
        if combo_idx >= len(fixed_indices):
            return
        p = fixed_indices[combo_idx]
        self._filter_pipe_idx = p
        active = self._active_classes_list[p]
        for cls_id, cb in self._checkboxes.items():
            cb.checked = cls_id in active
        for cls_id, img in self._checkbox_swatches.items():
            rgb  = self._color_maps[p].get(cls_id, [128, 128, 128])
            tile = np.full((14, 14, 3), rgb, dtype=np.uint8)
            img.update_image(o3d.geometry.Image(tile))

    def _recolor_all(self) -> None:
        for i, (pts, labels) in enumerate(zip(self._cached_pts, self._cached_labels)):
            if pts is not None and self._pipeline_active[i]:
                self._update_cloud(i, pts, labels)

    # Resolve key-down enum once, handling Open3D API differences across versions
    _KEY_DOWN = getattr(gui.KeyEvent, "DOWN", None) or gui.KeyEvent.Type.DOWN

    def _on_key(self, event) -> int:
        if event.type == self._KEY_DOWN:
            k = event.key
            if k in (gui.KeyName.RIGHT, ord("l"), ord("L")):
                self._on_next(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (gui.KeyName.LEFT, ord("h"), ord("H")):
                self._on_prev(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("r"), ord("R")):
                self._reset_camera(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("b"), ord("B")):
                self._look_bev(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("t"), ord("T")):
                self._cycle_mode(); return gui.Widget.EventCallbackResult.HANDLED
            if k in (ord("j"), ord("J")) and self._cb_traj is not None:
                self._cb_traj.checked = not self._cb_traj.checked
                self._on_traj_toggle(self._cb_traj.checked)
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _cycle_mode(self) -> None:
        self._display_mode = (self._display_mode + 1) % len(DISPLAY_MODES)
        if self._mode_combo is not None:
            self._mode_combo.selected_index = self._display_mode
        self._recolor_all()

    # ------------------------------------------------------------------
    # Data loading & parallel pipeline execution
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        sample = self.dataset[self.current_idx]

        pts_raw = sample.data.get(self.cfg.point_key)
        if pts_raw is None:
            return
        pts = to_numpy(pts_raw).astype(np.float32)

        labels: np.ndarray | None = None
        if self.cfg.label_key:
            lbl_raw = sample.data.get(self.cfg.label_key)
            if lbl_raw is not None:
                labels = to_numpy(lbl_raw).astype(np.int64)

        self._raw_pts    = pts
        self._raw_labels = labels

        self._lbl_frame.text = f"{self.current_idx + 1} / {len(self.dataset)}"
        self._lbl_npts.text  = f"Points: {len(pts):,}"
        if self._slider_frame is not None:
            self._slider_frame.int_value = self.current_idx

        for i, lbl in enumerate(self._timing_labels):
            if self._pipeline_active[i]:
                lbl.text = f"{self._pipelines[i].name}: …"

        # Bump ID so any in-flight callbacks from the previous frame are discarded
        self._refresh_id += 1
        current_id = self._refresh_id
        frame_idx  = self.current_idx

        def _run(i: int) -> None:
            t0 = time.perf_counter()
            p_pts, p_labels = self._pipelines[i].run(
                pts.copy(),
                labels.copy() if labels is not None else None,
                frame_idx=frame_idx,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            gui.Application.instance.post_to_main_thread(
                self._window,
                lambda i=i, p=p_pts, l=p_labels, t=elapsed_ms:
                    self._on_pipeline_done(current_id, i, p, l, t),
            )

        for i in range(len(self._pipelines)):
            if self._pipeline_active[i]:
                self._executor.submit(_run, i)
            else:
                # Clear stale cache so the viewport re-renders when re-enabled
                self._cached_pts[i]          = None
                self._cached_labels[i]       = None
                self._camera_initialized[i]  = False

    def _on_pipeline_done(
        self,
        refresh_id: int,
        i: int,
        pts: np.ndarray,
        labels: np.ndarray | None,
        elapsed_ms: float,
    ) -> None:
        if refresh_id != self._refresh_id:
            return  # user navigated away; discard stale result

        self._cached_pts[i]    = pts
        self._cached_labels[i] = labels

        if labels is not None and not self._label_fixed[i]:
            self._auto_detect_labels(i, labels)

        lbl_stats = self._lbl_stats_list[i] if i < len(self._lbl_stats_list) else None
        if labels is not None and lbl_stats is not None:
            lbl_stats.text = self._build_stats_text(i, labels)

        if i < len(self._timing_labels):
            self._timing_labels[i].text = f"{self._pipelines[i].name}: {elapsed_ms:.0f} ms"

        self._update_cloud(i, pts, labels)

        if self._show_traj and self._poses is not None and i == 0:
            self._update_trajectory()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _update_cloud(self, i: int, pts: np.ndarray, labels: np.ndarray | None) -> None:
        scene = self._scenes[i].scene

        if self._camera_initialized[i]:
            scene.remove_geometry("cloud")

        xyz  = pts[:, :3]
        mask = np.ones(len(xyz), dtype=bool)

        if self._display_mode == 0 and labels is not None:
            mask = np.isin(labels, list(self._active_classes_list[i]))

        xyz_f = xyz[mask].astype(np.float64)
        if len(xyz_f) == 0:
            return

        colors = self._compute_colors(i, pts[mask], labels[mask] if labels is not None else None)

        pcd = make_point_cloud(xyz_f, colors)
        scene.add_geometry("cloud", pcd, self._mats[i])

        if not self._camera_initialized[i]:
            self._setup_camera(i)
            self._camera_initialized[i] = True

    def _compute_colors(self, i: int, pts: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
        mode = self._display_mode

        if mode == 0 and labels is not None and self._color_maps[i]:
            return labels_to_colors(labels, self._color_maps[i])

        if mode == 1 and pts.shape[1] > self.cfg.intensity_channel:
            return intensity_colors(pts[:, self.cfg.intensity_channel])

        return height_colors(pts[:, 2])

    def _update_trajectory(self) -> None:
        if self._poses is None:
            return
        scene = self._scenes[0].scene
        scene.remove_geometry("traj_past")
        scene.remove_geometry("traj_future")

        pose_cur = self._poses[self.current_idx]

        mat_line = rendering.MaterialRecord()
        mat_line.shader     = "unlitLine"
        mat_line.line_width = 2.0

        if self.current_idx > 0:
            past_pts = poses_to_local(self._poses[: self.current_idx + 1], pose_cur)
            edges    = [[j, j + 1] for j in range(len(past_pts) - 1)]
            if edges and has_extent(past_pts):
                scene.add_geometry(
                    "traj_past", make_lineset(past_pts, edges, C_TRAJ_PAST), mat_line
                )

        if self.current_idx < len(self._poses) - 1:
            future_pts = poses_to_local(self._poses[self.current_idx:], pose_cur)
            edges      = [[j, j + 1] for j in range(len(future_pts) - 1)]
            if edges and has_extent(future_pts):
                scene.add_geometry(
                    "traj_future", make_lineset(future_pts, edges, C_TRAJ_FUTURE), mat_line
                )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _setup_camera(self, i: int) -> None:
        pts = self._cached_pts[i]
        if pts is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scenes[i].setup_camera(60, bounds, bounds.get_center())

    def _reset_camera(self) -> None:
        for i in range(len(self._scenes)):
            self._setup_camera(i)

    def _sync_cameras(self) -> None:
        """Reset all viewports to the bounding box of pipeline 0."""
        if self._cached_pts[0] is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self._cached_pts[0][:, :3].astype(np.float64)
        )
        bounds = pcd.get_axis_aligned_bounding_box()
        for scene in self._scenes:
            scene.setup_camera(60, bounds, bounds.get_center())

    def _look_bev(self) -> None:
        pts_list = [p for p in self._cached_pts if p is not None]
        if not pts_list:
            return
        z_max  = max(float(np.max(p[:, 2])) for p in pts_list)
        height = max(z_max + 15.0, 20.0)
        for scene in self._scenes:
            scene.scene.camera.look_at(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, height],
                [1.0, 0.0, 0.0],
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _build_stats_text(self, i: int, labels: np.ndarray) -> str:
        total  = max(len(labels), 1)
        unique, counts = np.unique(labels, return_counts=True)
        order  = np.argsort(-counts)
        smap   = self._semantic_maps[i]
        lines  = []
        for cls_id, cnt in zip(unique[order][:12], counts[order][:12]):
            name = smap.get(int(cls_id), str(cls_id))
            pct  = 100.0 * cnt / total
            bar  = "█" * int(pct / 5)
            lines.append(f"{name[:14]:<14} {pct:5.1f}% {bar}")
        return "\n".join(lines)

    def _auto_detect_labels(self, i: int, labels: np.ndarray) -> None:
        """Build or refresh color map for pipeline i from the observed label IDs."""
        unique_ids = sorted(int(v) for v in np.unique(labels))
        if unique_ids == self._class_ids_list[i]:
            return  # nothing changed
        n = len(unique_ids)
        cmap = auto_color_map(n)
        self._color_maps[i]          = {uid: cmap[j] for j, uid in enumerate(unique_ids)}
        self._semantic_maps[i]       = {uid: str(uid) for uid in unique_ids}
        self._class_ids_list[i]      = unique_ids
        self._active_classes_list[i] = set(unique_ids)
