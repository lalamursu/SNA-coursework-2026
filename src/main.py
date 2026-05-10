#!/usr/bin/env python3
"""
Main analysis pipeline — SNA Project 8, steps 5–13.

Usage (from project root):
    python src/main.py [--threshold T] [--max-threads N] [--min-posts P] [--output-dir DIR]

Progress is shown live in the terminal (stderr).
Detailed output is written to <output-dir>/reports/run.log.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from network_analysis import (
    assortativity_analysis,
    centrality_analysis,
    detect_communities,
    global_network_stats,
    kcore_decomposition,
    topic_influence_analysis,
)
from network_builder import (
    build_bipartite_user_topic_network,
    build_thread_similarity_network,
    build_user_interaction_network,
    _resolve_user_col,
)
from progress import ProgressTracker
from sentiment import add_sentiment_to_df
from visualization import (
    plot_bipartite_network,
    plot_centrality_bars,
    plot_community_profiles,
    plot_degree_distribution,
    plot_kcore,
    plot_kcore_distribution,
    plot_network_sample,
    plot_communities,
    plot_sentiment_distribution,
    plot_temporal_analysis,
    plot_topic_influence,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Configuration (overridable via CLI) ───────────────────────────────────────
SIMILARITY_THRESHOLD = 0.15
MAX_THREADS    = 5_000
MIN_USER_POSTS = 2

# ── Step definitions ──────────────────────────────────────────────────────────
# weight  = relative share of total work (drives overall %)
# estimate = expected wall-clock seconds (drives step-level ETA)
STEPS = [
    {"name": "Load & merge data",                           "weight": 1.0, "estimate":  20},  # 0
    {"name": "Sentiment analysis",                          "weight": 1.5, "estimate":  30},  # 1
    {"name": "[Step 3]  Temporal analysis",                 "weight": 0.5, "estimate":  10},  # 2
    {"name": "[Step 5]  Thread similarity network",         "weight": 5.0, "estimate":  90},  # 3
    {"name": "[Step 8]  Global stats — Thread",             "weight": 0.5, "estimate":  15},  # 4
    {"name": "[Step 9]  Centrality — Thread",               "weight": 2.0, "estimate":  60},  # 5
    {"name": "[Step 10] Communities — Thread",              "weight": 1.5, "estimate":  30},  # 6
    {"name": "[Step 11] Assortativity — Thread",            "weight": 0.5, "estimate":  10},  # 7
    {"name": "[Step 12] K-core — Thread",                   "weight": 0.5, "estimate":  10},  # 8
    {"name": "Plots — Thread",                              "weight": 1.0, "estimate":  30},  # 9
    {"name": "[Step 6]  User interaction network",          "weight": 5.0, "estimate": 120},  # 10
    {"name": "[Step 8]  Global stats — User",               "weight": 0.5, "estimate":  20},  # 11
    {"name": "[Step 9]  Centrality — User",                 "weight": 2.0, "estimate":  90},  # 12
    {"name": "[Step 10] Communities — User",                "weight": 2.0, "estimate":  60},  # 13
    {"name": "[Step 11] Assortativity — User",              "weight": 0.5, "estimate":  10},  # 14
    {"name": "[Step 12] K-core — User",                     "weight": 0.5, "estimate":  10},  # 15
    {"name": "Plots — User",                                "weight": 1.5, "estimate":  45},  # 16
    {"name": "[Step 7]  Bipartite user–topic network",      "weight": 2.0, "estimate":  45},  # 17
    {"name": "[Step 7]  Global stats — User projection",    "weight": 0.5, "estimate":  20},  # 18
    {"name": "[Step 7]  Centrality — User projection",      "weight": 1.0, "estimate":  30},  # 19
    {"name": "[Step 7]  Plots — Bipartite & projections",   "weight": 1.0, "estimate":  30},  # 20
    {"name": "[Step 8]  Global stats — Topic",              "weight": 0.3, "estimate":  10},  # 21
    {"name": "[Step 9]  Centrality — Topic",                "weight": 0.5, "estimate":  15},  # 22
    {"name": "[Step 10] Communities — Topic",               "weight": 0.5, "estimate":  15},  # 23
    {"name": "[Step 11] Assortativity — Topic",             "weight": 0.3, "estimate":   5},  # 24
    {"name": "[Step 12] K-core — Topic",                    "weight": 0.3, "estimate":   5},  # 25
    {"name": "Plots — Topic",                               "weight": 0.7, "estimate":  20},  # 26
    {"name": "[Step 13] Topic popularity vs influence",     "weight": 0.5, "estimate":  15},  # 27
    {"name": "Save reports",                                "weight": 0.3, "estimate":   5},  # 28
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules"}


def _read_csv_robust(filepath, **kwargs) -> pd.DataFrame:
    """Try UTF-8 (with BOM), then cp1252, then latin-1 — handles files from any OS."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(filepath, encoding="latin-1", **kwargs)


def _load_data() -> pd.DataFrame:
    """Load best available data using priority: FinBERT > strict filter > filtered CSVs."""

    def _skip(p: Path) -> bool:
        return any(s in p.parts for s in _SKIP_DIRS)

    # Priority 1: FinBERT sentiment output (richest — has Sentiment + Opinion_Category)
    sentiment_files = [p for p in BASE_DIR.rglob("suomi24_sentiment_FINAL_results.csv") if not _skip(p)]
    if sentiment_files:
        p = min(sentiment_files)
        print(f"Loading FinBERT sentiment output: {p.relative_to(BASE_DIR)}")
        df = _read_csv_robust(p, dtype=str)
        # Normalize column names so the rest of the pipeline finds them
        rename = {}
        if "Sentiment" in df.columns and "sentiment" not in df.columns:
            rename["Sentiment"] = "sentiment"
        if "Opinion_Category" in df.columns and "opinion_category" not in df.columns:
            rename["Opinion_Category"] = "opinion_category"
        if rename:
            df = df.rename(columns=rename)
        print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")
        return df

    # Priority 2: strict food filter output
    strict = DATA_DIR / "suomi24_STRICT_food_data.csv"
    if strict.exists():
        print(f"Loading strict food filter output: {strict.relative_to(BASE_DIR)}")
        df = _read_csv_robust(strict, dtype=str)
        print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")
        return df

    # Priority 3: raw filtered CSVs (one per VRT year file), merged
    filtered = sorted(p for p in BASE_DIR.rglob("suomi24_filtered_data_*.csv") if not _skip(p))
    if filtered:
        print(f"Merging {len(filtered)} filtered CSV(s):")
        frames = []
        for p in filtered:
            print(f"  {p.relative_to(BASE_DIR)}")
            frames.append(_read_csv_robust(p, dtype=str))
        df = pd.concat(frames, ignore_index=True)
        print(f"  Total rows: {len(df):,}")
        return df

    raise FileNotFoundError(
        f"No input data found under {BASE_DIR}.\n"
        "Run src/data_collection.py first, then optionally filter_food_data.py and sentiment_analysis.py."
    )


def _print_stats(stats: dict) -> None:
    for k, v in stats.items():
        if k != "Network":
            print(f"    {k}: {v}")


def _save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


def _centrality_csv(top_nodes: dict, path: Path) -> None:
    rows = [
        {"metric": m, "rank": r + 1, "node": n, "score": round(s, 6)}
        for m, entries in top_nodes.items()
        for r, (n, s) in enumerate(entries)
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path.name}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SNA Project 8 — analysis pipeline")
    ap.add_argument("--threshold",   type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--max-threads", type=int,   default=MAX_THREADS)
    ap.add_argument("--min-posts",   type=int,   default=MIN_USER_POSTS)
    ap.add_argument("--output-dir",  type=Path,  default=BASE_DIR / "outputs")
    args = ap.parse_args()

    out_dir    = args.output_dir
    plots_dir  = out_dir / "plots"
    reports_dir = out_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_log = open(reports_dir / "run.log", "w", buffering=1, encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = run_log

    tracker = ProgressTracker(STEPS)

    try:
        _run_pipeline(tracker, args.threshold, args.max_threads, args.min_posts,
                      plots_dir, reports_dir)
    finally:
        tracker.finish()
        sys.stdout = _orig_stdout
        run_log.close()

    elapsed = time.time() - tracker._t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    sys.stderr.write(
        f"\n✓  Done in {h}h {m:02d}m {s:02d}s\n"
        f"   Plots   → {plots_dir}\n"
        f"   Reports → {reports_dir}\n"
        f"   Run log → {reports_dir / 'run.log'}\n"
    )
    sys.stderr.flush()


_NET_THREAD = "Thread Similarity"
_NET_USER   = "User Interaction"
_NET_TOPIC  = "Topic Co-occurrence"
_NET_UPROJ  = "User Projection (bipartite)"


def _run_pipeline(  # noqa: C901
    t: ProgressTracker,
    threshold: float,
    max_threads: int,
    min_posts: int,
    plots_dir: Path,
    reports_dir: Path,
) -> None:
    all_stats: dict = {}

    # ── Load data ──────────────────────────────────────────────────────────
    t.start_step(0)
    df = _load_data()
    user_col = _resolve_user_col(df)

    # ── Sentiment ──────────────────────────────────────────────────────────
    t.start_step(1)
    if "sentiment" not in df.columns:
        df = add_sentiment_to_df(df)
    plot_sentiment_distribution(df, plots_dir / "sentiment_distribution.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 3  Temporal Analysis
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(2)
    print("\n[Step 3] Temporal Analysis")
    plot_temporal_analysis(df, plots_dir)

    # ══════════════════════════════════════════════════════════════════════
    # Step 5  Thread Similarity Network
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(3)
    print("\n[Step 5] Thread Similarity Network")
    g_thread = build_thread_similarity_network(
        df, threshold=threshold, max_threads=max_threads
    )

    t.start_step(4)
    print("[Step 8] Global stats — Thread")
    s = global_network_stats(g_thread, _NET_THREAD)
    all_stats["thread_similarity"] = s
    _print_stats(s)

    t.start_step(5)
    print("[Step 9] Centrality — Thread")
    _, top_thread = centrality_analysis(g_thread)
    plot_centrality_bars(top_thread, _NET_THREAD, plots_dir / "centrality_thread.png")
    _centrality_csv(top_thread, reports_dir / "centrality_top20_thread.csv")

    t.start_step(6)
    print("[Step 10] Community detection — Thread")
    part_thread, _, mod_thread = detect_communities(g_thread)
    nx.set_node_attributes(g_thread, part_thread, "community")
    all_stats["thread_modularity"] = mod_thread

    t.start_step(7)
    print("[Step 11] Assortativity — Thread")
    a = assortativity_analysis(g_thread)
    all_stats["thread_assortativity"] = a
    _print_stats(a)

    t.start_step(8)
    print("[Step 12] K-core — Thread")
    core_t, _, _, _ = kcore_decomposition(g_thread)
    pd.DataFrame(
        [{"node": n, "core_number": k} for n, k in core_t.items()]
    ).sort_values("core_number", ascending=False).to_csv(
        reports_dir / "kcore_thread.csv", index=False, encoding="utf-8"
    )

    t.start_step(9)
    print("Plots — Thread")
    plot_network_sample(g_thread, f"{_NET_THREAD} Network",
                        plots_dir / "network_thread.png", color_attr="community")
    plot_degree_distribution(g_thread, _NET_THREAD,
                             plots_dir / "degree_dist_thread.png")
    plot_communities(g_thread, part_thread, _NET_THREAD,
                     plots_dir / "communities_thread.png")
    plot_kcore(g_thread, core_t, _NET_THREAD,
               plots_dir / "kcore_thread.png")
    plot_kcore_distribution(core_t, _NET_THREAD,
                            plots_dir / "kcore_dist_thread.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 6  User Interaction Network
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(10)
    print("\n[Step 6] User Interaction Network")
    g_user = build_user_interaction_network(df, min_posts=min_posts)

    t.start_step(11)
    print("[Step 8] Global stats — User")
    s = global_network_stats(g_user, _NET_USER)
    all_stats["user_interaction"] = s
    _print_stats(s)

    t.start_step(12)
    print("[Step 9] Centrality — User")
    _, top_user = centrality_analysis(g_user)
    plot_centrality_bars(top_user, _NET_USER, plots_dir / "centrality_user.png")
    _centrality_csv(top_user, reports_dir / "centrality_top20_user.csv")

    t.start_step(13)
    print("[Step 10] Community detection — User")
    part_user, _, mod_user = detect_communities(g_user)
    nx.set_node_attributes(g_user, part_user, "community")
    all_stats["user_modularity"] = mod_user

    t.start_step(14)
    print("[Step 11] Assortativity — User")
    a = assortativity_analysis(g_user)
    all_stats["user_assortativity"] = a
    _print_stats(a)

    t.start_step(15)
    print("[Step 12] K-core — User")
    core_u, _, _, _ = kcore_decomposition(g_user)
    pd.DataFrame(
        [{"node": n, "core_number": k} for n, k in core_u.items()]
    ).sort_values("core_number", ascending=False).to_csv(
        reports_dir / "kcore_user.csv", index=False, encoding="utf-8"
    )

    t.start_step(16)
    print("Plots — User")
    plot_network_sample(g_user, f"{_NET_USER} Network",
                        plots_dir / "network_user.png", color_attr="community")
    plot_degree_distribution(g_user, _NET_USER,
                             plots_dir / "degree_dist_user.png")
    plot_communities(g_user, part_user, _NET_USER,
                     plots_dir / "communities_user.png")
    plot_kcore(g_user, core_u, _NET_USER,
               plots_dir / "kcore_user.png")
    plot_kcore_distribution(core_u, _NET_USER,
                            plots_dir / "kcore_dist_user.png")
    plot_community_profiles(df, part_user, user_col,
                            plots_dir / "community_sentiment_user.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 7  Bipartite User–Topic Network + projections
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(17)
    print("\n[Step 7] Bipartite User–Topic Network")
    g_bipartite, g_user_proj, g_topic_proj = build_bipartite_user_topic_network(df)

    t.start_step(18)
    print("[Step 7] Global stats — User projection")
    s = global_network_stats(g_user_proj, _NET_UPROJ)
    all_stats["bipartite_user_projection"] = s
    _print_stats(s)

    # NEW: Detect communities in User Projection for clean visualization
    print("  Detecting communities in User Projection...")
    part_uproj, _, _ = detect_communities(g_user_proj)
    nx.set_node_attributes(g_user_proj, part_uproj, "community")

    t.start_step(19)
    print("[Step 7] Centrality — User projection")
    _, top_uproj = centrality_analysis(g_user_proj)
    plot_centrality_bars(top_uproj, _NET_UPROJ,
                         plots_dir / "centrality_user_proj.png")
    _centrality_csv(top_uproj, reports_dir / "centrality_top20_user_proj.csv")

    t.start_step(20)
    print("[Step 7] Plots — Bipartite & projections")
    plot_bipartite_network(g_bipartite, plots_dir / "network_bipartite.png")
    
    # UPDATED: Use community coloring and filter low-weight edges for the projection
    plot_network_sample(g_user_proj, f"{_NET_UPROJ} Network",
                        plots_dir / "network_user_proj.png", 
                        color_attr="community", min_edge_weight=2)
    
    plot_degree_distribution(g_user_proj, _NET_UPROJ,
                             plots_dir / "degree_dist_user_proj.png")

    # ══════════════════════════════════════════════════════════════════════
    # Steps 8–12 on Topic Co-occurrence projection
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(21)
    print("[Step 8] Global stats — Topic")
    s = global_network_stats(g_topic_proj, _NET_TOPIC)
    all_stats["topic_projection"] = s
    _print_stats(s)

    t.start_step(22)
    print("[Step 9] Centrality — Topic")
    _, top_topic = centrality_analysis(g_topic_proj)
    plot_centrality_bars(top_topic, _NET_TOPIC, plots_dir / "centrality_topic.png")
    _centrality_csv(top_topic, reports_dir / "centrality_top20_topic.csv")

    t.start_step(23)
    print("[Step 10] Community detection — Topic")
    _, _, mod_topic = detect_communities(g_topic_proj)
    all_stats["topic_modularity"] = mod_topic

    t.start_step(24)
    print("[Step 11] Assortativity — Topic")
    a = assortativity_analysis(g_topic_proj)
    all_stats["topic_assortativity"] = a

    t.start_step(25)
    print("[Step 12] K-core — Topic")
    core_topic, _, _, _ = kcore_decomposition(g_topic_proj)

    t.start_step(26)
    print("Plots — Topic")
    plot_network_sample(g_topic_proj, f"{_NET_TOPIC} Network",
                        plots_dir / "network_topic.png", color_attr=None)
    plot_degree_distribution(g_topic_proj, _NET_TOPIC,
                             plots_dir / "degree_dist_topic.png")
    plot_kcore_distribution(core_topic, _NET_TOPIC,
                            plots_dir / "kcore_dist_topic.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 13  Topic Popularity vs Influence
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(27)
    print("\n[Step 13] Topic popularity vs influence")
    topic_df = topic_influence_analysis(g_topic_proj, df)
    print(topic_df.head(20).to_string(index=False))
    topic_df.to_csv(reports_dir / "topic_influence.csv", index=False, encoding="utf-8")
    plot_topic_influence(topic_df, plots_dir / "topic_influence.png")

    # ── Save consolidated report ───────────────────────────────────────────
    t.start_step(28)
    print("Save reports")
    _save_json(all_stats, reports_dir / "network_stats.json")

if __name__ == "__main__":
    main()