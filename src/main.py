#!/usr/bin/env python3
"""
Main analysis pipeline — SNA Project 8, steps 5–13.

Usage (from project root):
    python src/main.py

Output:
    outputs/plots/      — PNG figures
    outputs/reports/    — CSV and JSON reports
"""

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

# Ensure src/ is on the path when run as `python src/main.py`
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

# ── Configuration ─────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.15   # step 5: minimum cosine similarity for thread edge
MAX_THREADS = 5_000           # step 5: max threads to include in similarity graph
MIN_USER_POSTS = 2            # step 6: minimum posts for a user to be included


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    candidates = sorted(DATA_DIR.glob("suomi24_filtered_data_*.csv"))
    if not candidates:
        candidates = sorted(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV data found in {DATA_DIR}")

    path = candidates[-1]
    print(f"Loading: {path.name}")
    df = pd.read_csv(path, dtype=str)
    print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")
    return df


def _print_stats(stats: dict) -> None:
    for k, v in stats.items():
        if k != "Network":
            print(f"    {k}: {v}")


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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

    df = _load_data()
    user_col = _resolve_user_col(df)

    # ── Sentiment (step 4 supplement) ─────────────────────────────────────
    if "sentiment" not in df.columns:
        print("\n[Sentiment]")
        df = add_sentiment_to_df(df)
        plot_sentiment_distribution(df, PLOTS_DIR / "sentiment_distribution.png")

    all_stats: dict = {}

    # ══════════════════════════════════════════════════════════════════════
    # Step 5  Thread Similarity Network
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Step 5] Thread Similarity Network")
    G_thread = build_thread_similarity_network(
        df, threshold=SIMILARITY_THRESHOLD, max_threads=MAX_THREADS
    )

    print("[Step 8] Global stats")
    s = global_network_stats(G_thread, "Thread Similarity")
    all_stats["thread_similarity"] = s
    _print_stats(s)

    print("[Step 9] Centrality")
    _, top_thread = centrality_analysis(G_thread)
    plot_centrality_bars(top_thread, "Thread Similarity", PLOTS_DIR / "centrality_thread.png")
    _centrality_csv(top_thread, REPORTS_DIR / "centrality_top20_thread.csv")

    print("[Step 10] Community detection")
    part_thread, comm_thread, mod_thread = detect_communities(G_thread)
    nx.set_node_attributes(G_thread, part_thread, "community")
    all_stats["thread_modularity"] = mod_thread

    print("[Step 11] Assortativity")
    a = assortativity_analysis(G_thread)
    all_stats["thread_assortativity"] = a
    for k, v in a.items():
        print(f"    {k}: {v}")

    print("[Step 12] K-core decomposition")
    core_t, _, _, _ = kcore_decomposition(G_thread)

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

    pd.DataFrame(
        [{"node": n, "core_number": k} for n, k in core_t.items()]
    ).sort_values("core_number", ascending=False).to_csv(
        REPORTS_DIR / "kcore_thread.csv", index=False
    )

    # ══════════════════════════════════════════════════════════════════════
    # Step 6  User Interaction Network
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Step 6] User Interaction Network")
    G_user = build_user_interaction_network(df, min_posts=MIN_USER_POSTS)

    print("[Step 8] Global stats")
    s = global_network_stats(G_user, "User Interaction")
    all_stats["user_interaction"] = s
    _print_stats(s)

    print("[Step 9] Centrality")
    _, top_user = centrality_analysis(G_user)
    plot_centrality_bars(top_user, "User Interaction", PLOTS_DIR / "centrality_user.png")
    _centrality_csv(top_user, REPORTS_DIR / "centrality_top20_user.csv")

    print("[Step 10] Community detection")
    part_user, comm_user, mod_user = detect_communities(G_user)
    nx.set_node_attributes(G_user, part_user, "community")
    all_stats["user_modularity"] = mod_user

    print("[Step 11] Assortativity")
    a = assortativity_analysis(G_user)
    all_stats["user_assortativity"] = a
    for k, v in a.items():
        print(f"    {k}: {v}")

    print("[Step 12] K-core decomposition")
    core_u, _, _, _ = kcore_decomposition(G_user)

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

    pd.DataFrame(
        [{"node": n, "core_number": k} for n, k in core_u.items()]
    ).sort_values("core_number", ascending=False).to_csv(
        REPORTS_DIR / "kcore_user.csv", index=False
    )

    # ══════════════════════════════════════════════════════════════════════
    # Step 7  Bipartite User–Topic Network + projections
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Step 7] Bipartite User–Topic Network")
    _, G_user_proj, G_topic_proj = build_bipartite_user_topic_network(df)

    print("[Step 8] Global stats — topic projection")
    s = global_network_stats(G_topic_proj, "Topic Co-occurrence")
    all_stats["topic_projection"] = s
    _print_stats(s)

    print("[Step 9] Centrality — topic network")
    _, top_topic = centrality_analysis(G_topic_proj)
    plot_centrality_bars(top_topic, "Topic Co-occurrence", PLOTS_DIR / "centrality_topic.png")
    _centrality_csv(top_topic, REPORTS_DIR / "centrality_top20_topic.csv")

    print("[Step 10] Community detection — topic network")
    part_topic, _, mod_topic = detect_communities(G_topic_proj)
    all_stats["topic_modularity"] = mod_topic

    print("[Step 11] Assortativity — topic network")
    a = assortativity_analysis(G_topic_proj)
    all_stats["topic_assortativity"] = a

    print("[Step 12] K-core — topic network")
    core_topic, _, _, _ = kcore_decomposition(G_topic_proj)

    plot_network_sample(G_topic_proj, "Topic Co-occurrence Network",
                        PLOTS_DIR / "network_topic.png", color_attr=None)
    plot_degree_distribution(G_topic_proj, "Topic Co-occurrence",
                             PLOTS_DIR / "degree_dist_topic.png")
    plot_kcore_distribution(core_topic, "Topic Co-occurrence",
                            PLOTS_DIR / "kcore_dist_topic.png")

    # ══════════════════════════════════════════════════════════════════════
    # Step 13  Topic Popularity vs Influence
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Step 13] Topic Popularity vs Influence")
    topic_df = topic_influence_analysis(G_topic_proj, df)
    print(topic_df.head(20).to_string(index=False))
    topic_df.to_csv(REPORTS_DIR / "topic_influence.csv", index=False)
    plot_topic_influence(topic_df, PLOTS_DIR / "topic_influence.png")

    # ── Consolidated stats report ──────────────────────────────────────────
    _save_json(all_stats, REPORTS_DIR / "network_stats.json")

    print(f"\n✓ Done.  Plots → {PLOTS_DIR}   Reports → {REPORTS_DIR}")


if __name__ == "__main__":
    main()
