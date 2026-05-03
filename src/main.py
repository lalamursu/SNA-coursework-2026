#!/usr/bin/env python3
"""
Main analysis pipeline — SNA Project 8, steps 5–13.

Usage (from project root):
    python src/main.py

Progress is shown live in the terminal (stderr).
Detailed output is written to outputs/reports/run.log.
"""

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
    plot_centrality_bars,
    plot_community_profiles,
    plot_degree_distribution,
    plot_kcore,
    plot_kcore_distribution,
    plot_network_sample,
    plot_communities,
    plot_sentiment_distribution,
    plot_topic_influence,
)

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

# ── Configuration ─────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.15
MAX_THREADS  = 5_00
MIN_USER_POSTS = 2

# ── Step definitions ──────────────────────────────────────────────────────────
# weight  = relative share of total work (drives overall %)
# estimate = expected wall-clock seconds (drives step-level ETA)
STEPS = [
    {"name": "Load & merge data",                        "weight": 1.0, "estimate":  20},
    {"name": "Sentiment analysis",                       "weight": 1.5, "estimate":  30},
    {"name": "[Step 5]  Thread similarity network",      "weight": 5.0, "estimate":  90},
    {"name": "[Step 8]  Global stats — Thread",          "weight": 0.5, "estimate":  15},
    {"name": "[Step 9]  Centrality — Thread",            "weight": 2.0, "estimate":  60},
    {"name": "[Step 10] Communities — Thread",           "weight": 1.5, "estimate":  30},
    {"name": "[Step 11] Assortativity — Thread",         "weight": 0.5, "estimate":  10},
    {"name": "[Step 12] K-core — Thread",                "weight": 0.5, "estimate":  10},
    {"name": "Plots — Thread",                           "weight": 1.0, "estimate":  30},
    {"name": "[Step 6]  User interaction network",       "weight": 5.0, "estimate": 120},
    {"name": "[Step 8]  Global stats — User",            "weight": 0.5, "estimate":  20},
    {"name": "[Step 9]  Centrality — User",              "weight": 2.0, "estimate":  90},
    {"name": "[Step 10] Communities — User",             "weight": 2.0, "estimate":  60},
    {"name": "[Step 11] Assortativity — User",           "weight": 0.5, "estimate":  10},
    {"name": "[Step 12] K-core — User",                  "weight": 0.5, "estimate":  10},
    {"name": "Plots — User",                             "weight": 1.5, "estimate":  45},
    {"name": "[Step 7]  Bipartite user–topic network",   "weight": 2.0, "estimate":  45},
    {"name": "[Step 8]  Global stats — Topic",           "weight": 0.3, "estimate":  10},
    {"name": "[Step 9]  Centrality — Topic",             "weight": 0.5, "estimate":  15},
    {"name": "[Step 10] Communities — Topic",            "weight": 0.5, "estimate":  15},
    {"name": "[Step 11] Assortativity — Topic",          "weight": 0.3, "estimate":   5},
    {"name": "[Step 12] K-core — Topic",                 "weight": 0.3, "estimate":   5},
    {"name": "Plots — Topic",                            "weight": 0.7, "estimate":  20},
    {"name": "[Step 13] Topic popularity vs influence",  "weight": 0.5, "estimate":  15},
    {"name": "Save reports",                             "weight": 0.3, "estimate":   5},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules"}


def _find_filtered_csvs(base: Path) -> list[Path]:
    results = []
    for p in base.rglob("suomi24_filtered_data_*.csv"):
        if not any(skip in p.parts for skip in _SKIP_DIRS):
            results.append(p)
    return sorted(results)


def _load_data() -> pd.DataFrame:
    candidates = _find_filtered_csvs(BASE_DIR)
    if not candidates:
        raise FileNotFoundError(
            f"No suomi24_filtered_data_*.csv found under {BASE_DIR}.\n"
            "Run src/data_collection.py first."
        )
    if len(candidates) == 1:
        print(f"Loading: {candidates[0].relative_to(BASE_DIR)}")
        df = pd.read_csv(candidates[0], dtype=str)
        print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")
        return df

    print(f"Merging {len(candidates)} filtered CSV(s):")
    frames = []
    for p in candidates:
        print(f"  {p.relative_to(BASE_DIR)}")
        frames.append(pd.read_csv(p, dtype=str))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    return df


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
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path.name}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to run.log; progress tracker uses stderr (terminal)
    run_log = open(REPORTS_DIR / "run.log", "w", buffering=1, encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = run_log

    tracker = ProgressTracker(STEPS)

    try:
        _run_pipeline(tracker)
    finally:
        tracker.finish()
        sys.stdout = _orig_stdout
        run_log.close()

    elapsed = time.time() - tracker._t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    sys.stderr.write(
        f"\n✓  Done in {h}h {m:02d}m {s:02d}s\n"
        f"   Plots   → {PLOTS_DIR}\n"
        f"   Reports → {REPORTS_DIR}\n"
        f"   Run log → {REPORTS_DIR / 'run.log'}\n"
    )
    sys.stderr.flush()


def _run_pipeline(t: ProgressTracker) -> None:  # noqa: C901
    all_stats: dict = {}

    # ── Load data ──────────────────────────────────────────────────────────
    t.start_step(0)
    df = _load_data()
    user_col = _resolve_user_col(df)

    # ── Sentiment ──────────────────────────────────────────────────────────
    t.start_step(1)
    if "sentiment" not in df.columns:
        df = add_sentiment_to_df(df)
        plot_sentiment_distribution(df, PLOTS_DIR / "sentiment_distribution.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 5  Thread Similarity Network
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(2)
    print("\n[Step 5] Thread Similarity Network")
    G_thread = build_thread_similarity_network(
        df, threshold=SIMILARITY_THRESHOLD, max_threads=MAX_THREADS
    )

    t.start_step(3)
    print("[Step 8] Global stats — Thread")
    s = global_network_stats(G_thread, "Thread Similarity")
    all_stats["thread_similarity"] = s
    _print_stats(s)

    t.start_step(4)
    print("[Step 9] Centrality — Thread")
    _, top_thread = centrality_analysis(G_thread)
    plot_centrality_bars(top_thread, "Thread Similarity", PLOTS_DIR / "centrality_thread.png")
    _centrality_csv(top_thread, REPORTS_DIR / "centrality_top20_thread.csv")

    t.start_step(5)
    print("[Step 10] Community detection — Thread")
    part_thread, _, mod_thread = detect_communities(G_thread)
    nx.set_node_attributes(G_thread, part_thread, "community")
    all_stats["thread_modularity"] = mod_thread

    t.start_step(6)
    print("[Step 11] Assortativity — Thread")
    a = assortativity_analysis(G_thread)
    all_stats["thread_assortativity"] = a
    _print_stats(a)

    t.start_step(7)
    print("[Step 12] K-core — Thread")
    core_t, _, _, _ = kcore_decomposition(G_thread)
    pd.DataFrame(
        [{"node": n, "core_number": k} for n, k in core_t.items()]
    ).sort_values("core_number", ascending=False).to_csv(
        REPORTS_DIR / "kcore_thread.csv", index=False
    )

    t.start_step(8)
    print("Plots — Thread")
    plot_network_sample(G_thread, "Thread Similarity Network",
                        PLOTS_DIR / "network_thread.png", color_attr="community")
    plot_degree_distribution(G_thread, "Thread Similarity",
                            PLOTS_DIR / "degree_dist_thread.png")
    plot_communities(G_thread, part_thread, "Thread Similarity",
                    PLOTS_DIR / "communities_thread.png")
    plot_kcore(G_thread, core_t, "Thread Similarity",
                PLOTS_DIR / "kcore_thread.png")
    plot_kcore_distribution(core_t, "Thread Similarity",
                            PLOTS_DIR / "kcore_dist_thread.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 6  User Interaction Network
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(9)
    print("\n[Step 6] User Interaction Network")
    G_user = build_user_interaction_network(df, min_posts=MIN_USER_POSTS)

    t.start_step(10)
    print("[Step 8] Global stats — User")
    s = global_network_stats(G_user, "User Interaction")
    all_stats["user_interaction"] = s
    _print_stats(s)

    t.start_step(11)
    print("[Step 9] Centrality — User")
    _, top_user = centrality_analysis(G_user)
    plot_centrality_bars(top_user, "User Interaction", PLOTS_DIR / "centrality_user.png")
    _centrality_csv(top_user, REPORTS_DIR / "centrality_top20_user.csv")

    t.start_step(12)
    print("[Step 10] Community detection — User")
    part_user, _, mod_user = detect_communities(G_user)
    nx.set_node_attributes(G_user, part_user, "community")
    all_stats["user_modularity"] = mod_user

    t.start_step(13)
    print("[Step 11] Assortativity — User")
    a = assortativity_analysis(G_user)
    all_stats["user_assortativity"] = a
    _print_stats(a)

    t.start_step(14)
    print("[Step 12] K-core — User")
    core_u, _, _, _ = kcore_decomposition(G_user)
    pd.DataFrame(
        [{"node": n, "core_number": k} for n, k in core_u.items()]
    ).sort_values("core_number", ascending=False).to_csv(
        REPORTS_DIR / "kcore_user.csv", index=False
    )

    t.start_step(15)
    print("Plots — User")
    plot_network_sample(G_user, "User Interaction Network",
                        PLOTS_DIR / "network_user.png", color_attr="community")
    plot_degree_distribution(G_user, "User Interaction",
                            PLOTS_DIR / "degree_dist_user.png")
    plot_communities(G_user, part_user, "User Interaction",
                    PLOTS_DIR / "communities_user.png")
    plot_kcore(G_user, core_u, "User Interaction",
                PLOTS_DIR / "kcore_user.png")
    plot_kcore_distribution(core_u, "User Interaction",
                            PLOTS_DIR / "kcore_dist_user.png")
    plot_community_profiles(df, part_user, user_col,
                            PLOTS_DIR / "community_sentiment_user.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 7  Bipartite User–Topic Network + projections
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(16)
    print("\n[Step 7] Bipartite User–Topic Network")
    _, G_user_proj, G_topic_proj = build_bipartite_user_topic_network(df)

    t.start_step(17)
    print("[Step 8] Global stats — Topic")
    s = global_network_stats(G_topic_proj, "Topic Co-occurrence")
    all_stats["topic_projection"] = s
    _print_stats(s)

    t.start_step(18)
    print("[Step 9] Centrality — Topic")
    _, top_topic = centrality_analysis(G_topic_proj)
    plot_centrality_bars(top_topic, "Topic Co-occurrence", PLOTS_DIR / "centrality_topic.png")
    _centrality_csv(top_topic, REPORTS_DIR / "centrality_top20_topic.csv")

    t.start_step(19)
    print("[Step 10] Community detection — Topic")
    part_topic, _, mod_topic = detect_communities(G_topic_proj)
    all_stats["topic_modularity"] = mod_topic

    t.start_step(20)
    print("[Step 11] Assortativity — Topic")
    a = assortativity_analysis(G_topic_proj)
    all_stats["topic_assortativity"] = a

    t.start_step(21)
    print("[Step 12] K-core — Topic")
    core_topic, _, _, _ = kcore_decomposition(G_topic_proj)

    t.start_step(22)
    print("Plots — Topic")
    plot_network_sample(G_topic_proj, "Topic Co-occurrence Network",
                        PLOTS_DIR / "network_topic.png", color_attr=None)
    plot_degree_distribution(G_topic_proj, "Topic Co-occurrence",
                            PLOTS_DIR / "degree_dist_topic.png")
    plot_kcore_distribution(core_topic, "Topic Co-occurrence",
                            PLOTS_DIR / "kcore_dist_topic.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 13  Topic Popularity vs Influence
    # ══════════════════════════════════════════════════════════════════════
    t.start_step(23)
    print("\n[Step 13] Topic popularity vs influence")
    topic_df = topic_influence_analysis(G_topic_proj, df)
    print(topic_df.head(20).to_string(index=False))
    topic_df.to_csv(REPORTS_DIR / "topic_influence.csv", index=False)
    plot_topic_influence(topic_df, PLOTS_DIR / "topic_influence.png")

    # ── Save consolidated report ───────────────────────────────────────────
    t.start_step(24)
    print("Save reports")
    _save_json(all_stats, REPORTS_DIR / "network_stats.json")

if __name__ == "__main__":
    main()
