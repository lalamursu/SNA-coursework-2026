"""Visualization helpers — saves all plots to the outputs/plots/ directory."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Degree distribution ───────────────────────────────────────────────────────

def plot_degree_distribution(G: nx.Graph, title: str, output_path: Path) -> None:
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(degrees, bins=min(60, max(degrees) + 1), color="steelblue", edgecolor="white")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Degree Distribution — {title}")
    _save(fig, output_path)


# ── Network layout ────────────────────────────────────────────────────────────

def plot_network_sample(
    G: nx.Graph,
    title: str,
    output_path: Path,
    max_nodes: int = 500,
    color_attr: str | None = None,
    min_edge_weight: int = 1, # Added weight filtering to clean up the 'hairball'
) -> None:
    """Spring-layout plot focusing on structure and community coloring."""
    G_u = G.to_undirected() if G.is_directed() else G.copy()

    # 1. Edge weight filtering to reveal backbone structure
    if min_edge_weight > 1:
        low_weight_edges = [(u, v) for u, v, d in G_u.edges(data=True) if d.get('weight', 1) < min_edge_weight]
        G_u.remove_edges_from(low_weight_edges)

    if G_u.number_of_nodes() == 0:
        return

    # 2. Extract giant component for better visualization
    comps = list(nx.connected_components(G_u))
    giant = max(comps, key=len)
    sample = list(giant)[:max_nodes]
    G_plot = G_u.subgraph(sample).copy()

    # 3. Physics-tuned layout: 'k' pushes clusters away from each other
    pos = nx.spring_layout(
        G_plot, 
        seed=42, 
        k=2.0 / np.sqrt(len(G_plot)), 
        iterations=50
    )
    
    degrees = dict(G_plot.degree())
    node_sizes = [max(30, 10 * degrees[n]) for n in G_plot.nodes()]

    # 4. Coloring logic
    if color_attr:
        attrs = nx.get_node_attributes(G_plot, color_attr)
        unique_vals = sorted(set(attrs.values()))
        cmap = plt.cm.get_cmap("turbo", len(unique_vals))
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        node_colors = [cmap(val_to_idx.get(attrs.get(n, 0), 0)) for n in G_plot.nodes()]
    else:
        node_colors = "#89b4fa"

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#ffffff')
    
    # 5. Draw edges with transparency
    nx.draw_networkx_edges(
        G_plot, pos=pos, ax=ax,
        alpha=0.1,
        edge_color="#cccccc",
        width=0.4
    )

    # 6. Draw nodes with borders (Medium-style aesthetic)
    nx.draw_networkx_nodes(
        G_plot, pos=pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white", 
        linewidths=1.2,
        alpha=0.9
    )

    ax.set_title(f"{title}\n(Filtered weight < {min_edge_weight}, showing ≤{max_nodes} nodes)", 
                 fontsize=14, pad=20)
    ax.axis("off")
    _save(fig, output_path)


# ── Community visualization ───────────────────────────────────────────────────

def plot_communities(
    G: nx.Graph,
    partition: dict,
    title: str,
    output_path: Path,
    max_nodes: int = 400,
) -> None:
    G_u = G.to_undirected() if G.is_directed() else G
    if G_u.number_of_nodes() == 0 or not partition:
        return

    giant = max(nx.connected_components(G_u), key=len)
    sample = list(giant)[:max_nodes]
    G_plot = G_u.subgraph(sample).copy()

    pos = nx.spring_layout(G_plot, seed=42)
    cmap = plt.cm.tab20
    n_comm = max(partition.values()) + 1
    node_colors = [cmap((partition.get(n, 0) % 20) / 20) for n in G_plot.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx(
        G_plot, pos=pos, ax=ax,
        node_size=[max(15, 6 * G_plot.degree(n)) for n in G_plot.nodes()],
        node_color=node_colors,
        edge_color="gray",
        alpha=0.75,
        with_labels=False,
        width=0.3,
    )
    ax.set_title(f"{title}  ({n_comm} communities, ≤{max_nodes} nodes shown)")
    ax.axis("off")
    _save(fig, output_path)


# ── Centrality bar charts ─────────────────────────────────────────────────────

def plot_centrality_bars(
    top_nodes: dict[str, list],
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    metrics = list(top_nodes.keys())[:4]
    if not metrics:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics):
        entries = top_nodes[metric][:top_n]
        labels = [str(n)[:25] for n, _ in entries]
        values = [v for _, v in entries]

        axes_flat[i].barh(labels[::-1], values[::-1], color="steelblue")
        axes_flat[i].set_title(f"{metric.capitalize()} Centrality")
        axes_flat[i].set_xlabel("Score")

    for j in range(len(metrics), 4):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Top {top_n} Nodes — {title}", fontsize=13)
    plt.tight_layout()
    _save(fig, output_path)


# ── K-core visualization ──────────────────────────────────────────────────────

def plot_kcore(
    G: nx.Graph,
    core_numbers: dict,
    title: str,
    output_path: Path,
    max_nodes: int = 400,
) -> None:
    G_u = G.to_undirected() if G.is_directed() else G
    if G_u.number_of_nodes() == 0:
        return

    giant = max(nx.connected_components(G_u), key=len)
    sample = list(giant)[:max_nodes]
    G_plot = G_u.subgraph(sample).copy()

    max_k = max(core_numbers.values()) if core_numbers else 1
    node_colors = [core_numbers.get(n, 0) / max(max_k, 1) for n in G_plot.nodes()]
    node_sizes = [max(15, 12 * core_numbers.get(n, 1)) for n in G_plot.nodes()]

    pos = nx.spring_layout(G_plot, seed=42)
    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx(
        G_plot, pos=pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        vmin=0, vmax=1,
        edge_color="gray",
        alpha=0.8,
        with_labels=False,
        width=0.3,
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, max_k))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Core number (k)")
    ax.set_title(f"{title} — K-Core Structure  (max k={max_k})")
    ax.axis("off")
    _save(fig, output_path)


def plot_kcore_distribution(core_numbers: dict, title: str, output_path: Path) -> None:
    if not core_numbers:
        return
    counts = pd.Series(core_numbers.values()).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color="darkorange")
    ax.set_xlabel("Core number (k)")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"K-Core Distribution — {title}")
    _save(fig, output_path)


# ── Topic influence ───────────────────────────────────────────────────────────

def plot_topic_influence(topic_df: pd.DataFrame, output_path: Path) -> None:
    if topic_df.empty:
        return

    top = topic_df.head(60)

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        top["frequency"],
        top["degree_centrality"],
        c=top["rank_diff"],
        s=80,
        cmap="coolwarm",
        alpha=0.75,
        edgecolors="gray",
        linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="|Frequency rank − Centrality rank|")

    for _, row in top.head(20).iterrows():
        ax.annotate(
            row["topic"],
            (row["frequency"], row["degree_centrality"]),
            fontsize=7,
            ha="right",
            va="bottom",
        )

    ax.set_xlabel("Keyword Frequency  (popularity)")
    ax.set_ylabel("Degree Centrality in Topic Network")
    ax.set_title("Topic Popularity vs Network Centrality  (top 60 topics)")
    _save(fig, output_path)


# ── Sentiment ─────────────────────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if "sentiment" not in df.columns:
        return

    has_opinion = "opinion_category" in df.columns

    _SENTIMENT_COLORS = {
        "positive": "#a6e3a1",
        "negative": "#f38ba8",
        "neutral":  "#6c7086",
    }
    _OPINION_COLORS = {
        "pro-healthy":        "#a6e3a1",
        "anti-healthy":       "#f38ba8",
        "pro-sustainability": "#94e2d5",
        "skeptical":          "#fab387",
        "neutral":            "#6c7086",
        "positive":           "#89b4fa",
        "negative":           "#f38ba8",
    }

    def _annotate(ax, bars, values):
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:,}",
                ha="center", va="bottom", fontsize=8,
            )

    if has_opinion:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(6, 5))
        ax2 = None

    # Left panel: FinBERT base sentiment (Positive / Negative / Neutral)
    sent_counts = df["sentiment"].str.lower().value_counts()
    bars1 = ax1.bar(
        sent_counts.index,
        sent_counts.values,
        color=[_SENTIMENT_COLORS.get(s, "#89b4fa") for s in sent_counts.index],
    )
    _annotate(ax1, bars1, sent_counts.values)
    ax1.set_title("FinBERT Sentiment")
    ax1.set_ylabel("Posts")

    # Right panel: Opinion categories
    if has_opinion:
        op_counts = df["opinion_category"].value_counts()
        bars2 = ax2.bar(
            op_counts.index,
            op_counts.values,
            color=[_OPINION_COLORS.get(c.lower(), "#cba6f7") for c in op_counts.index],
        )
        _annotate(ax2, bars2, op_counts.values)
        ax2.set_title("Opinion Category")
        ax2.set_ylabel("Posts")
        ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save(fig, output_path)


# ── Temporal analysis (Step 3) ────────────────────────────────────────────────

_HEALTH_KW = {
    "terveellinen", "terveys", "ravitseva", "vitamiini", "kuitu", "proteiini",
    "vähärasvainen", "sokeriton", "epäterveellinen", "roskaruoka", "pikaruoka",
    "rasvainen", "sokeri", "lisäaine", "eines", "valmisruoka", "karkki", "sipsi",
    "limu", "mikroateria", "ravintoarvo", "kalori", "kilokalori", "hiilihydraatti", "rasva",
}

_SUSTAIN_KW = {
    "kestävä", "ekologinen", "luomu", "lähiruoka", "kasvisruoka", "vegaani",
    "vegaaninen", "kasvipohjainen", "ilmasto", "ilmastovaikutus", "hiilijalanjälki",
    "liha", "lihansyönti", "punainen liha", "nauta", "sika", "broileri",
}


def _kw_list(kw_str) -> list[str]:
    if pd.isna(kw_str) or not str(kw_str).strip():
        return []
    return [k.strip() for k in str(kw_str).split(",") if k.strip()]


def plot_temporal_analysis(df: pd.DataFrame, plots_dir: Path) -> None:
    """Step 3: keyword frequency over time and health vs sustainability comparison."""
    if "timestamp" not in df.columns or "matched_keywords" not in df.columns:
        print("  Skipping temporal analysis: missing timestamp or matched_keywords column")
        return

    df_t = df.copy()
    df_t["_ts"] = pd.to_datetime(df_t["timestamp"], errors="coerce")
    df_t = df_t.dropna(subset=["_ts"])
    if df_t.empty:
        print("  Skipping temporal analysis: no parseable timestamps")
        return

    df_t["_month"] = df_t["_ts"].dt.to_period("M")

    # ── Plot 1: monthly total post count ──────────────────────────────────────
    monthly = df_t.groupby("_month").size()
    x_ts = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_ts, monthly.values, marker="o", linewidth=1.8, markersize=3, color="steelblue")
    ax.fill_between(x_ts, monthly.values, alpha=0.15, color="steelblue")
    ax.set_xlabel("Month")
    ax.set_ylabel("Posts")
    ax.set_title("Monthly Post Frequency — Food & Health Discussions (Suomi24 2021–2023)")
    ax.grid(axis="y", alpha=0.35)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, plots_dir / "temporal_freq.png")

    # ── Plot 2: health vs sustainability over time ─────────────────────────────
    def _has_kw(series: pd.Series, kw_set: set) -> pd.Series:
        return series.apply(lambda s: bool(set(_kw_list(s)) & kw_set))

    df_t["_health"] = _has_kw(df_t["matched_keywords"], _HEALTH_KW)
    df_t["_sust"]   = _has_kw(df_t["matched_keywords"], _SUSTAIN_KW)

    h_monthly = df_t[df_t["_health"]].groupby("_month").size()
    s_monthly = df_t[df_t["_sust"]].groupby("_month").size()

    all_months = sorted(set(h_monthly.index) | set(s_monthly.index))
    x2 = pd.PeriodIndex(all_months).to_timestamp()
    h_vals = [int(h_monthly.get(m, 0)) for m in all_months]
    s_vals = [int(s_monthly.get(m, 0)) for m in all_months]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x2, h_vals, label="Health-related", color="#a6e3a1",
            marker="o", markersize=3, linewidth=1.8)
    ax.plot(x2, s_vals, label="Sustainability-related", color="#94e2d5",
            marker="s", markersize=3, linewidth=1.8)
    ax.fill_between(x2, h_vals, alpha=0.12, color="#a6e3a1")
    ax.fill_between(x2, s_vals, alpha=0.12, color="#94e2d5")
    ax.set_xlabel("Month")
    ax.set_ylabel("Posts")
    ax.set_title("Health vs Sustainability Discussion Volume Over Time")
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, plots_dir / "temporal_health_vs_sust.png")

    print(f"  Temporal range: {df_t['_ts'].min().date()} – {df_t['_ts'].max().date()}")
    print(f"  Health posts: {df_t['_health'].sum():,}  Sustainability posts: {df_t['_sust'].sum():,}")


# ── Bipartite network visualization (Step 7) ──────────────────────────────────

def plot_bipartite_network(
    graph: nx.Graph,
    output_path: Path,
    max_users: int = 60,
) -> None:
    """Bipartite layout: top-N users (right) ↔ all topic nodes (left)."""
    user_nodes  = [n for n, d in graph.nodes(data=True)
                   if d.get("bipartite") == 0 and graph.degree(n) > 0]
    topic_nodes = [n for n, d in graph.nodes(data=True) if d.get("bipartite") == 1]

    if not user_nodes or not topic_nodes:
        return

    top_users = sorted(user_nodes, key=lambda n: graph.degree(n), reverse=True)[:max_users]
    g_plot = graph.subgraph(set(top_users) | set(topic_nodes)).copy()

    n_t = len(topic_nodes)
    n_u = len(top_users)
    pos = {}
    for i, t in enumerate(sorted(topic_nodes)):
        pos[t] = (0.0, i / max(n_t - 1, 1))
    for i, u in enumerate(top_users):
        pos[u] = (1.0, i / max(n_u - 1, 1))

    node_colors = [
        "#89b4fa" if g_plot.nodes[n].get("bipartite") == 1 else "#a6e3a1"
        for n in g_plot.nodes()
    ]
    node_sizes = [
        220 if g_plot.nodes[n].get("bipartite") == 1 else 35
        for n in g_plot.nodes()
    ]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx(
        g_plot, pos=pos, ax=ax,
        node_size=node_sizes, node_color=node_colors,
        edge_color="gray", alpha=0.55,
        with_labels=False, width=0.15,
    )
    topic_sub = {n: pos[n] for n in topic_nodes if n in g_plot}
    nx.draw_networkx_labels(g_plot.subgraph(list(topic_sub)), pos=topic_sub,
                            ax=ax, font_size=7)

    handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#89b4fa",
                      markersize=10, label="Topic (keyword)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#a6e3a1",
                      markersize=8, label=f"User (top {n_u} by degree)"),
    ]
    ax.legend(handles=handles, loc="upper center")
    ax.set_title(
        f"Bipartite User–Topic Graph  ({len(user_nodes):,} users, {n_t} topics)"
        f"\n(showing top {n_u} users by degree)"
    )
    ax.axis("off")
    _save(fig, output_path)


def plot_community_profiles(
    df: pd.DataFrame,
    partition: dict,
    user_col: str,
    output_path: Path,
    max_communities: int = 15,
) -> None:
    """Stacked bar: sentiment fraction per community (top communities by size)."""
    if "sentiment" not in df.columns or not partition:
        return

    df_tmp = df.copy()
    df_tmp["_user"] = df_tmp[user_col].astype(str)
    df_tmp["community"] = df_tmp["_user"].map(partition)
    df_tmp = df_tmp.dropna(subset=["community"])
    df_tmp["community"] = df_tmp["community"].astype(int)

    # FIXED: Ensure sentiment strings are lowercase to match the reindex labels
    df_tmp["sentiment"] = df_tmp["sentiment"].astype(str).str.lower()

    # Keep only the largest communities
    top_comms = (
        df_tmp["community"].value_counts().head(max_communities).index.tolist()
    )
    df_tmp = df_tmp[df_tmp["community"].isin(top_comms)]

    if df_tmp.empty:
        return

    cross = (
        pd.crosstab(df_tmp["community"], df_tmp["sentiment"], normalize="index")
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(max(8, len(cross) * 0.6), 5))
    cross.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        # Colors updated to match the sentiment distribution plot (Catppuccin style)
        color=["#a6e3a1", "#6c7086", "#f38ba8"],
    )
    ax.set_title(f"Sentiment Distribution per Community  (top {max_communities})")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Fraction")
    ax.legend(title="Sentiment", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, output_path)