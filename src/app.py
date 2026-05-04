#!/usr/bin/env python3
"""
SNA Project 8 — interactive viewer and control panel.

Usage (from project root):
    python src/app.py
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Pillow

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
SRC_DIR     = BASE_DIR / "src"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Use venv Python when available so subprocesses inherit installed packages
_venv = os.environ.get("VIRTUAL_ENV")
if _venv and Path(_venv).exists():
    _candidate = (
        Path(_venv) / "Scripts" / "python.exe"   # Windows
        if sys.platform == "win32"
        else Path(_venv) / "bin" / "python3"      # Linux / macOS
    )
    _PYTHON = str(_candidate) if _candidate.exists() else sys.executable
else:
    _PYTHON = sys.executable
_MAIN_PY = str(SRC_DIR / "main.py")

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
BG     = "#1e1e2e"
BG2    = "#313244"
BG3    = "#45475a"
FG     = "#cdd6f4"
FG_DIM = "#a6adc8"
BLUE   = "#89b4fa"
GREEN  = "#a6e3a1"
YELLOW = "#f9e2af"
PINK   = "#f38ba8"
MAUVE  = "#cba6f7"

# ── Plot pages: (title, [relative-paths-within-output-dir/plots]) ─────────────
# Each entry becomes one viewer page; images are displayed side-by-side (≤2).
PLOT_PAGES = [
    ("Sentiment distribution",
    ["sentiment_distribution.png"]),
    ("Step 5 — Thread Similarity: network & degree",
    ["network_thread.png", "degree_dist_thread.png"]),
    ("Step 5 — Thread Similarity: communities & k-core",
    ["communities_thread.png", "kcore_thread.png"]),
    ("Step 5 — Thread Similarity: centrality & k-core distribution",
    ["centrality_thread.png", "kcore_dist_thread.png"]),
    ("Step 6 — User Interaction: network & degree",
    ["network_user.png", "degree_dist_user.png"]),
    ("Step 6 — User Interaction: communities & sentiment",
    ["communities_user.png", "community_sentiment_user.png"]),
    ("Step 6 — User Interaction: centrality & k-core",
    ["centrality_user.png", "kcore_user.png"]),
    ("Step 6 — User Interaction: k-core distribution",
    ["kcore_dist_user.png"]),
    ("Step 7 — Topic Co-occurrence: network & degree",
    ["network_topic.png", "degree_dist_topic.png"]),
    ("Step 7 — Topic Co-occurrence: centrality & k-core distribution",
    ["centrality_topic.png", "kcore_dist_topic.png"]),
    ("Step 13 — Topic popularity vs influence",
    ["topic_influence.png"]),
]

# ── Default config values ─────────────────────────────────────────────────────
DEFAULT_THRESHOLD  = "0.15"
DEFAULT_THREADS    = "5000"
DEFAULT_MIN_POSTS  = "2"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _output_dir_name(threshold: str, threads: str, min_posts: str) -> str:
    return f"t{threshold}_n{threads}_p{min_posts}"


def _output_dir(label: str) -> Path:
    return OUTPUTS_DIR / label


def _has_results(label: str) -> bool:
    return (_output_dir(label) / "reports" / "network_stats.json").exists()


def _list_all_configs() -> list[str]:
    """Return all output subdirs that have at least a plots/ or reports/ child."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        d.name for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and (
            (d / "plots").exists() or (d / "reports").exists()
        )
    )


def _config_label(name: str) -> str:
    """Display name for the dropdown — adds '(partial)' if run did not finish."""
    return name if _has_results(name) else f"{name}  (partial)"


def _load_stats(label: str) -> dict:
    p = _output_dir(label) / "reports" / "network_stats.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _data_summary() -> str:
    csvs = sorted(DATA_DIR.glob("suomi24_filtered_data_*.csv")) if DATA_DIR.exists() else []
    if not csvs:
        return "No filtered CSVs found.  Run data_collection.py first."
    lines = [f"Filtered CSV files found: {len(csvs)}"]
    total = 0
    for p in csvs:
        try:
            with open(p, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1  # minus header
            total += n
            mtime = p.stat().st_mtime
            import time
            lines.append(f"  {p.name}  ({n:,} rows,  {time.strftime('%Y-%m-%d', time.localtime(mtime))})")
        except Exception:
            lines.append(f"  {p.name}  (unreadable)")
    lines.append(f"  Total rows: {total:,}")
    strict = DATA_DIR / "suomi24_STRICT_food_data.csv"
    if strict.exists():
        try:
            import pandas as pd
            n = len(pd.read_csv(strict, usecols=[0]))
            lines.append(f"\nStrict food filter output: {n:,} rows  ({strict.name})")
        except Exception:
            lines.append(f"\nStrict food filter output exists: {strict.name}")
    sentiment_out = DATA_DIR / "suomi24_sentiment_FINAL_results.csv"
    if sentiment_out.exists():
        lines.append(f"FinBERT sentiment output exists: {sentiment_out.name}")
    return "\n".join(lines)


# ── Main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SNA Project 8 — Suomi24 Food & Health Analysis")
        self.geometry("1400x860")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._idx = 0
        self._photo_refs: list = []  # keep Tk image refs alive

        # All pages: [0] = DATA MANAGEMENT, [1..] = plot pages
        self._frames: list[tk.Frame] = []
        self._titles: list[str] = (
            ["DATA MANAGEMENT"]
            + [p[0] for p in PLOT_PAGES]
            + ["Network stats summary"]
        )

        # ── Header ────────────────────────────────────────────────────────
        self._hdr = tk.Label(self, text="", font=("Helvetica", 17, "bold"),
                             bg=BG, fg=FG)
        self._hdr.pack(pady=(14, 0))

        self._area = tk.Frame(self, bg=BG)
        self._area.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)

        # ── Navigation ────────────────────────────────────────────────────
        nav = tk.Frame(self, bg=BG)
        nav.pack(pady=8)
        btn_kw = dict(bg=BG2, fg=FG, font=("Helvetica", 14), relief="flat",
                      padx=18, pady=4, cursor="hand2",
                      activebackground=BG3, activeforeground=FG)
        self._btn_prev = tk.Button(nav, text="<", command=self._prev, **btn_kw)
        self._btn_prev.pack(side=tk.LEFT, padx=8)
        self._page_lbl = tk.Label(nav, text="", bg=BG, fg=FG_DIM,
                                  font=("Helvetica", 11))
        self._page_lbl.pack(side=tk.LEFT, padx=8)
        self._btn_next = tk.Button(nav, text=">", command=self._next, **btn_kw)
        self._btn_next.pack(side=tk.LEFT, padx=8)

        self.bind("<Left>",  lambda _: self._prev())
        self.bind("<Right>", lambda _: self._next())

        # ── Build frames ──────────────────────────────────────────────────
        # Page 0: Data management
        f0 = tk.Frame(self._area, bg=BG)
        self._frames.append(f0)
        self._build_data_mgmt(f0)

        # Pages 1…N: plot viewers (built lazily on first visit)
        for _ in PLOT_PAGES:
            f = tk.Frame(self._area, bg=BG)
            self._frames.append(f)

        # Last page: stats summary
        f_stats = tk.Frame(self._area, bg=BG)
        self._frames.append(f_stats)
        self._stats_frame = f_stats
        self._stats_built = False

        self._show(0)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show(self, idx: int) -> None:
        for f in self._frames:
            f.pack_forget()
        self._frames[idx].pack(fill=tk.BOTH, expand=True)
        self._hdr.config(text=self._titles[idx])
        self._page_lbl.config(text=f"{idx + 1} / {len(self._frames)}")
        self._btn_prev.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        self._btn_next.config(
            state=tk.NORMAL if idx < len(self._frames) - 1 else tk.DISABLED)
        self._idx = idx

        # Lazy build of plot pages
        if 1 <= idx <= len(PLOT_PAGES):
            self._ensure_plot_page(idx)
        elif idx == len(self._frames) - 1:
            self._ensure_stats_page()

    def _prev(self) -> None:
        if self._idx > 0:
            self._show(self._idx - 1)

    def _next(self) -> None:
        if self._idx < len(self._frames) - 1:
            self._show(self._idx + 1)

    # ── Reusable widget helpers ───────────────────────────────────────────────

    def _label(self, parent, text, fg=None, font=None, **pack_kw) -> tk.Label:
        lbl = tk.Label(parent, text=text, bg=BG, fg=fg or FG,
                       font=font or ("Helvetica", 11))
        lbl.pack(**pack_kw)
        return lbl

    def _section_label(self, parent, text: str) -> None:
        tk.Label(parent, text=text, bg=BG, fg=BLUE,
                 font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(10, 2))

    def _textbox(self, parent, content: str = "", height: int = 6,
                 font_size: int = 10, scrollbar: bool = False,
                 **pack_kw) -> tk.Text:
        kw = dict(bg=BG2, fg=FG, font=("Courier", font_size),
                  relief="flat", padx=8, pady=6, wrap=tk.WORD,
                  insertbackground=FG)
        if height:
            kw["height"] = height
        if scrollbar:
            container = tk.Frame(parent, bg=BG)
            container.pack(**pack_kw)
            sb = tk.Scrollbar(container, bg=BG3, troughcolor=BG2)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            kw["yscrollcommand"] = sb.set
            tb = tk.Text(container, **kw)
            tb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.config(command=tb.yview)
        else:
            tb = tk.Text(parent, **kw)
            tb.pack(**pack_kw)
        if content:
            tb.insert(tk.END, content)
            tb.config(state=tk.DISABLED)
        return tb

    def _entry(self, parent, var: tk.StringVar, width: int = 10) -> tk.Entry:
        return tk.Entry(parent, textvariable=var, bg=BG2, fg=FG,
                        insertbackground=FG, relief="flat",
                        font=("Courier", 11), width=width)

    def _button(self, parent, text: str, cmd, fg=None, **pack_kw) -> tk.Button:
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=BG2, fg=fg or FG, relief="flat",
                        font=("Helvetica", 10), padx=10, pady=4,
                        cursor="hand2", activebackground=BG3,
                        activeforeground=FG)
        btn.pack(**pack_kw)
        return btn

    def _status_label(self, parent, **pack_kw) -> tk.Label:
        lbl = tk.Label(parent, text="", bg=BG, fg=FG_DIM,
                       font=("Courier", 9), anchor="w", justify="left")
        lbl.pack(**pack_kw)
        return lbl

    def _init_step_statuses(self) -> None:
        """Set step status labels at startup based on which output files already exist."""
        filtered = list(DATA_DIR.glob("suomi24_filtered_data_*.csv")) if DATA_DIR.exists() else []
        strict   = DATA_DIR / "suomi24_STRICT_food_data.csv"
        sentiment = DATA_DIR / "suomi24_sentiment_FINAL_results.csv"

        def _set(lbl: tk.Label, done: bool, detail: str = "") -> None:
            if done:
                lbl.config(text=f"Done  {detail}".strip(), fg=GREEN)
            else:
                lbl.config(text="Not done", fg=FG_DIM)

        _set(self._dc_status, bool(filtered),
             f"({len(filtered)} file(s))" if filtered else "")
        _set(self._ff_status, strict.exists())
        _set(self._fb_status, sentiment.exists())

    # ── Plot image embed ───────────────────────────────────────────────────────

    def _embed_image(self, parent: tk.Frame, img_path: Path) -> None:
        if not img_path.exists():
            tk.Label(parent, text=f"[image not found: {img_path.name}]",
                     bg=BG, fg=PINK, font=("Courier", 10)).pack(
                         fill=tk.BOTH, expand=True)
            return
        # Resize to fit available space while preserving aspect ratio
        parent.update_idletasks()
        w = max(parent.winfo_width(), 400)
        h = max(parent.winfo_height(), 300)
        img = Image.open(img_path)
        img.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._photo_refs.append(photo)
        lbl = tk.Label(parent, image=photo, bg=BG)
        lbl.pack(fill=tk.BOTH, expand=True)

    # ── Lazy plot-page builder ────────────────────────────────────────────────

    def _ensure_plot_page(self, page_idx: int) -> None:
        frame = self._frames[page_idx]
        for w in frame.winfo_children():
            w.destroy()

        plot_idx = page_idx - 1
        _, filenames = PLOT_PAGES[plot_idx]

        # Which output dir to use?
        cfg = self._selected_config.get()
        plots_dir = (_output_dir(cfg) / "plots") if cfg else (OUTPUTS_DIR / "plots")

        paths = [plots_dir / fn for fn in filenames]

        if len(paths) == 1:
            self._embed_image(frame, paths[0])
        else:
            frame.columnconfigure(0, weight=1, uniform="img")
            frame.columnconfigure(1, weight=1, uniform="img")
            frame.rowconfigure(0, weight=1)
            for col, p in enumerate(paths):
                cell = tk.Frame(frame, bg=BG)
                cell.grid(row=0, column=col, sticky="nsew",
                          padx=(0 if col else 0, 4 if col == 0 else 0))
                self._embed_image(cell, p)

    def _rebuild_plot_pages(self) -> None:
        """Clear and rebuild all plot pages (called when config selection changes)."""
        for i in range(1, len(PLOT_PAGES) + 1):
            frame = self._frames[i]
            for w in frame.winfo_children():
                w.destroy()
        self._photo_refs.clear()
        self._stats_built = False
        for w in self._stats_frame.winfo_children():
            w.destroy()
        # If currently on a plot page, re-render it immediately
        if self._idx >= 1:
            self._ensure_plot_page(self._idx)

    def _ensure_stats_page(self) -> None:
        if self._stats_built:
            return
        cfg = self._selected_config.get()
        stats = _load_stats(cfg) if cfg else {}
        if not stats and (OUTPUTS_DIR / "reports" / "network_stats.json").exists():
            stats = json.loads(
                (OUTPUTS_DIR / "reports" / "network_stats.json").read_text("utf-8"))

        if not stats:
            tk.Label(self._stats_frame,
                     text="No network_stats.json found for this config.",
                     bg=BG, fg=PINK, font=("Courier", 11)).pack(pady=20)
            self._stats_built = True
            return

        text = json.dumps(stats, indent=2, default=str)
        self._textbox(self._stats_frame, text, height=0, font_size=10,
                      scrollbar=True, fill=tk.BOTH, expand=True)
        self._stats_built = True

    # ── DATA MANAGEMENT page ──────────────────────────────────────────────────

    def _build_data_mgmt(self, parent: tk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)

        # ── Left column ──────────────────────────────────────────────────
        left = tk.Frame(parent, bg=BG)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))

        # Data status
        self._section_label(left, "DATA STATUS")
        summary = _data_summary()
        self._textbox(left, summary, height=8, scrollbar=False,
                      fill=tk.X, pady=(0, 6))

        # Data collection
        self._section_label(left, "STEP 1 — DATA COLLECTION  (data_collection.py)")
        row_dc = tk.Frame(left, bg=BG)
        row_dc.pack(fill=tk.X, pady=(0, 4))
        self._dc_status = self._status_label(row_dc, side=tk.RIGHT, padx=8)
        self._button(row_dc, "Run data_collection.py",
                     self._run_data_collection, fg=GREEN, side=tk.LEFT)

        self._dc_log = self._textbox(left, height=5, scrollbar=True,
                                     fill=tk.X, pady=(0, 8))
        self._dc_log.config(state=tk.NORMAL)

        # Strict food filter
        self._section_label(left, "STEP 2 — STRICT FOOD FILTER  (filter_food_data.py)")
        row_ff = tk.Frame(left, bg=BG)
        row_ff.pack(fill=tk.X, pady=(0, 4))
        self._ff_status = self._status_label(row_ff, side=tk.RIGHT, padx=8)
        self._button(row_ff, "Run filter_food_data.py",
                     self._run_filter_food, fg=YELLOW, side=tk.LEFT)

        self._ff_log = self._textbox(left, height=4, scrollbar=True,
                                     fill=tk.X, pady=(0, 8))
        self._ff_log.config(state=tk.NORMAL)

        # FinBERT sentiment
        self._section_label(left, "STEP 3 — FINBERT SENTIMENT  (sentiment_analysis.py)")
        row_fb = tk.Frame(left, bg=BG)
        row_fb.pack(fill=tk.X, pady=(0, 4))
        self._fb_status = self._status_label(row_fb, side=tk.RIGHT, padx=8)
        self._button(row_fb, "Run sentiment_analysis.py",
                     self._run_finbert, fg=MAUVE, side=tk.LEFT)

        self._fb_log = self._textbox(left, height=4, scrollbar=True,
                                     fill=tk.X, pady=(0, 8))
        self._fb_log.config(state=tk.NORMAL)

        self._init_step_statuses()

        # ── Right column ─────────────────────────────────────────────────
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")

        # Config chooser (previously run results)
        self._section_label(right, "CHOOSE CONFIGURATION  (view results)")
        cfg_row = tk.Frame(right, bg=BG)
        cfg_row.pack(fill=tk.X, pady=(0, 4))

        configs = _list_all_configs()
        self._cfg_var = tk.StringVar(value=configs[0] if configs else "")
        self._selected_config = self._cfg_var

        self._cfg_dropdown = ttk.Combobox(
            cfg_row, textvariable=self._cfg_var,
            values=configs, state="readonly", width=32,
            font=("Courier", 10))
        self._cfg_dropdown.pack(side=tk.LEFT, padx=(0, 8))
        self._cfg_dropdown.bind("<<ComboboxSelected>>", lambda _: self._load_config())
        self._button(cfg_row, "Load", self._load_config, fg=BLUE, side=tk.LEFT)

        self._cfg_info = self._textbox(right, height=5, scrollbar=False,
                                       fill=tk.X, pady=(0, 8))
        self._cfg_info.config(state=tk.NORMAL)
        if configs:
            self._refresh_cfg_info(configs[0])

        # Single config run
        self._section_label(right, "RUN SINGLE CONFIGURATION  (main.py)")
        grid = tk.Frame(right, bg=BG)
        grid.pack(fill=tk.X, pady=(0, 4))

        fields = [
            ("Similarity threshold", DEFAULT_THRESHOLD),
            ("Max threads",          DEFAULT_THREADS),
            ("Min user posts",       DEFAULT_MIN_POSTS),
        ]
        self._run_vars: list[tk.StringVar] = []
        for row_i, (label, default) in enumerate(fields):
            tk.Label(grid, text=label, bg=BG, fg=FG_DIM,
                     font=("Helvetica", 10)).grid(
                         row=row_i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            self._run_vars.append(var)
            self._entry(grid, var, width=12).grid(
                row=row_i, column=1, sticky="w", padx=(8, 0))

        run_row = tk.Frame(right, bg=BG)
        run_row.pack(fill=tk.X, pady=(4, 2))
        self._run_status = self._status_label(run_row, side=tk.RIGHT, padx=8)
        self._run_btn = self._button(run_row, "Run analysis",
                                     self._run_single, fg=GREEN, side=tk.LEFT)

        self._run_log = self._textbox(right, height=6, scrollbar=True,
                                      fill=tk.X, pady=(0, 8))
        self._run_log.config(state=tk.NORMAL)

        # Schedule queue
        self._section_label(right, "SCHEDULE CONFIGURATIONS  (overnight batch)")
        add_row = tk.Frame(right, bg=BG)
        add_row.pack(fill=tk.X, pady=(0, 4))

        self._q_vars: list[tk.StringVar] = []
        for i, (lbl, dflt) in enumerate([
                ("thr", DEFAULT_THRESHOLD),
                ("threads", DEFAULT_THREADS),
                ("posts", DEFAULT_MIN_POSTS)]):
            tk.Label(add_row, text=lbl, bg=BG, fg=FG_DIM,
                     font=("Helvetica", 9)).grid(row=0, column=i*2, padx=(4, 0))
            var = tk.StringVar(value=dflt)
            self._q_vars.append(var)
            self._entry(add_row, var, width=7).grid(
                row=0, column=i*2+1, padx=(2, 4))

        tk.Button(add_row, text="Add to queue", command=self._queue_add,
                  bg=BG2, fg=YELLOW, relief="flat", font=("Helvetica", 10),
                  padx=10, pady=4, cursor="hand2",
                  activebackground=BG3, activeforeground=FG,
                  ).grid(row=0, column=6, padx=4)

        # Queue list
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Queue.Treeview",
                        background=BG2, foreground=FG, fieldbackground=BG2,
                        rowheight=22, font=("Courier", 9))
        style.configure("Queue.Treeview.Heading",
                        background=BG3, foreground=FG,
                        font=("Helvetica", 9, "bold"))
        style.map("Queue.Treeview", background=[("selected", BG3)])

        q_container = tk.Frame(right, bg=BG)
        q_container.pack(fill=tk.X, pady=(0, 4))

        self._queue_tree = ttk.Treeview(
            q_container,
            columns=("threshold", "max_threads", "min_posts", "status"),
            show="headings", height=5, style="Queue.Treeview")
        for col, w in [("threshold", 90), ("max_threads", 100),
                       ("min_posts", 90), ("status", 120)]:
            self._queue_tree.heading(col, text=col)
            self._queue_tree.column(col, width=w, anchor="center")
        q_sb = ttk.Scrollbar(q_container, orient=tk.VERTICAL,
                              command=self._queue_tree.yview)
        self._queue_tree.configure(yscrollcommand=q_sb.set)
        q_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._queue_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        queue_btns = tk.Frame(right, bg=BG)
        queue_btns.pack(fill=tk.X, pady=(0, 4))
        self._button(queue_btns, "Remove selected",
                     self._queue_remove, side=tk.LEFT, padx=(0, 6))
        self._q_run_btn = self._button(queue_btns, "Run all (shortest first)",
                                       self._queue_run_all,
                                       fg=GREEN, side=tk.LEFT)

        self._q_log = self._textbox(right, height=5, scrollbar=True,
                                    fill=tk.BOTH, expand=True)
        self._q_log.config(state=tk.NORMAL)

        # Internal queue state
        self._queue_items: list[dict] = []
        self._queue_running = False

    # ── Config chooser actions ────────────────────────────────────────────────

    def _refresh_cfg_info(self, label: str) -> None:
        stats = _load_stats(label)
        partial = not _has_results(label)
        if not stats:
            text = f"Config: {label}\n" + ("(partial run: some information is still missing)" if partial else "(no results found)")
        else:
            lines = [f"Config: {label}"]
            for net_key, net_stats in stats.items():
                if isinstance(net_stats, dict):
                    name = net_stats.get("Network", net_key)
                    nodes = net_stats.get("Nodes", "?")
                    edges = net_stats.get("Edges", "?")
                    lines.append(f"  {name}: {nodes} nodes, {edges} edges")
            text = "\n".join(lines)
        self._cfg_info.config(state=tk.NORMAL)
        self._cfg_info.delete("1.0", tk.END)
        self._cfg_info.insert(tk.END, text)
        self._cfg_info.config(state=tk.DISABLED)

    def _load_config(self) -> None:
        label = self._cfg_var.get()
        if not label:
            return
        self._refresh_cfg_info(label)
        self._rebuild_plot_pages()

    # ── Subprocess runner ─────────────────────────────────────────────────────

    def _run_subprocess(self, cmd: list[str], log_widget: tk.Text,
                        status_label: tk.Label, done_cb=None) -> None:
        """Run cmd in a background thread, streaming output to log_widget."""
        def _set_status(text: str, color: str = FG_DIM) -> None:
            self.after(0, lambda: status_label.config(text=text, fg=color))

        def _append(text: str) -> None:
            def _do():
                log_widget.config(state=tk.NORMAL)
                log_widget.insert(tk.END, text)
                log_widget.see(tk.END)
                log_widget.config(state=tk.DISABLED)
            self.after(0, _do)

        def _worker():
            _set_status("Running...", YELLOW)
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding="utf-8",
                    cwd=str(BASE_DIR))
                for line in proc.stdout:
                    _append(line)
                proc.wait()
                if proc.returncode == 0:
                    _set_status("Done.", GREEN)
                elif proc.returncode == -9 or proc.returncode == 137:
                    _set_status("OOM — killed.", PINK)
                else:
                    _set_status(f"Exit {proc.returncode}", PINK)
            except Exception as e:
                _set_status(f"Error: {e}", PINK)
            if done_cb:
                self.after(0, done_cb)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Data pipeline button actions ──────────────────────────────────────────

    def _run_data_collection(self) -> None:
        self._run_subprocess(
            [_PYTHON, str(SRC_DIR / "data_collection.py")],
            self._dc_log, self._dc_status)

    def _run_filter_food(self) -> None:
        self._run_subprocess(
            [_PYTHON, str(SRC_DIR / "filter_food_data.py")],
            self._ff_log, self._ff_status)

    def _run_finbert(self) -> None:
        self._run_subprocess(
            [_PYTHON, str(SRC_DIR / "sentiment_analysis.py")],
            self._fb_log, self._fb_status)

    # ── Single config run ─────────────────────────────────────────────────────

    def _run_single(self) -> None:
        threshold, threads, min_posts = (v.get().strip() for v in self._run_vars)
        label = _output_dir_name(threshold, threads, min_posts)
        out   = _output_dir(label)

        if _has_results(label):
            self._set_run_status(f"Already done: {label}", YELLOW)
            return

        self._run_btn.config(state=tk.DISABLED)
        cmd = [
            _PYTHON, _MAIN_PY,
            "--threshold", threshold,
            "--max-threads", threads,
            "--min-posts", min_posts,
            "--output-dir", str(out),
        ]

        def _on_done():
            self._run_btn.config(state=tk.NORMAL)
            self._on_run_success(label)

        self._run_subprocess(cmd, self._run_log, self._run_status,
                             done_cb=_on_done)

    def _set_run_status(self, text: str, color: str = FG_DIM) -> None:
        self._run_status.config(text=text, fg=color)

    def _refresh_completed_dropdown(self) -> None:
        configs = _list_all_configs()
        self._cfg_dropdown["values"] = configs
        if configs and not self._cfg_var.get():
            self._cfg_var.set(configs[0])

    def _on_run_success(self, label: str) -> None:
        """Called on the main thread after any pipeline run succeeds."""
        self._refresh_completed_dropdown()
        self._cfg_var.set(label)
        self._rebuild_plot_pages()

    # ── Schedule queue ────────────────────────────────────────────────────────

    def _queue_add(self) -> None:
        threshold, threads, min_posts = (v.get().strip() for v in self._q_vars)
        label = _output_dir_name(threshold, threads, min_posts)
        existing = [item["label"] for item in self._queue_items]
        if label in existing:
            return
        item = {"threshold": threshold, "threads": threads,
                "min_posts": min_posts, "label": label, "status": "pending"}
        self._queue_items.append(item)
        self._queue_tree.insert("", tk.END, iid=label,
                                values=(threshold, threads, min_posts, "pending"))

    def _queue_remove(self) -> None:
        sel = self._queue_tree.selection()
        for iid in sel:
            self._queue_tree.delete(iid)
            self._queue_items = [i for i in self._queue_items
                                 if i["label"] != iid]

    def _queue_set_status(self, label: str, status: str, color: str) -> None:
        try:
            item = self._queue_tree.item(label)
            vals = list(item["values"])
            vals[3] = status
            self._queue_tree.item(label, values=vals,
                                  tags=(color,))
        except Exception:
            pass
        try:
            self._queue_tree.tag_configure("green", foreground=GREEN)
            self._queue_tree.tag_configure("yellow", foreground=YELLOW)
            self._queue_tree.tag_configure("pink", foreground=PINK)
        except Exception:
            pass

    def _queue_append_log(self, text: str) -> None:
        def _do():
            self._q_log.config(state=tk.NORMAL)
            self._q_log.insert(tk.END, text)
            self._q_log.see(tk.END)
            self._q_log.config(state=tk.DISABLED)
        self.after(0, _do)

    def _queue_run_all(self) -> None:
        if self._queue_running:
            return
        pending = [i for i in self._queue_items if i["status"] == "pending"]
        if not pending:
            return

        # Shortest-job-first: sort by max_threads ascending
        pending.sort(key=lambda i: int(i["threads"]))

        self._queue_running = True
        self._q_run_btn.config(state=tk.DISABLED)

        def _worker():
            for item in pending:
                label    = item["label"]
                threshold = item["threshold"]
                threads  = item["threads"]
                min_posts = item["min_posts"]
                out      = _output_dir(label)

                if _has_results(label):
                    self.after(0, lambda l=label: self._queue_set_status(
                        l, "skipped (done)", "yellow"))
                    continue

                self.after(0, lambda l=label: self._queue_set_status(
                    l, "running", "yellow"))
                item["status"] = "running"

                cmd = [
                    _PYTHON, _MAIN_PY,
                    "--threshold", threshold,
                    "--max-threads", threads,
                    "--min-posts", min_posts,
                    "--output-dir", str(out),
                ]
                self._queue_append_log(f"\n--- {label} ---\n")
                try:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=str(BASE_DIR))
                    for line in proc.stdout:
                        self._queue_append_log(line)
                    proc.wait()
                    rc = proc.returncode

                    if rc == 0:
                        item["status"] = "done"
                        self.after(0, lambda l=label: self._queue_set_status(
                            l, "done", "green"))
                        self.after(0, lambda l=label: self._on_run_success(l))
                    elif rc in (-9, 137):
                        # OOM: retry with halved thread count
                        halved = str(max(100, int(threads) // 2))
                        self._queue_append_log(
                            f"OOM (rc {rc}) — retrying with max_threads={halved}\n")
                        self.after(0, lambda l=label: self._queue_set_status(
                            l, f"OOM retry {halved}", "yellow"))
                        retry_cmd = [
                            _PYTHON, _MAIN_PY,
                            "--threshold", threshold,
                            "--max-threads", halved,
                            "--min-posts", min_posts,
                            "--output-dir", str(out),
                        ]
                        proc2 = subprocess.Popen(
                            retry_cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd=str(BASE_DIR))
                        for line in proc2.stdout:
                            self._queue_append_log(line)
                        proc2.wait()
                        if proc2.returncode == 0:
                            item["status"] = "done"
                            self.after(0, lambda l=label: self._queue_set_status(
                                l, "done (retry)", "green"))
                            self.after(0, lambda l=label: self._on_run_success(l))
                        else:
                            item["status"] = "failed"
                            self.after(0, lambda l=label: self._queue_set_status(
                                l, f"failed {proc2.returncode}", "pink"))
                    else:
                        item["status"] = "failed"
                        self.after(0, lambda l=label: self._queue_set_status(
                            l, f"failed rc={rc}", "pink"))
                except Exception as e:
                    item["status"] = "error"
                    self._queue_append_log(f"Error: {e}\n")
                    self.after(0, lambda l=label: self._queue_set_status(
                        l, "error", "pink"))

            self.after(0, self._on_queue_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_queue_done(self) -> None:
        self._queue_running = False
        self._q_run_btn.config(state=tk.NORMAL)
        self._refresh_completed_dropdown()
        self._queue_append_log("\nBatch complete.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
