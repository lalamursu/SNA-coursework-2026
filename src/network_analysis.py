"""
Network analysis — steps 8–13.

Step 8 : Global network statistics
Step 9 : Centrality analysis
Step 10: Community detection (Louvain / greedy fallback)
Step 11: Assortativity and homophily
Step 12: K-core decomposition
Step 13: Topic popularity vs network centrality
"""

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain
    _HAS_LOUVAIN = True
except ImportError:
    _HAS_LOUVAIN = False
    print("python-louvain not found — using NetworkX greedy modularity fallback")


# ── Step 8 ────────────────────────────────────────────────────────────────────

def global_network_stats(G: nx.Graph, name: str = "Network") -> dict:
    """Return a dict of global statistics for the network."""
    stats: dict = {"Network": name, "Nodes": G.number_of_nodes(), "Edges": G.number_of_edges()}

    if G.number_of_nodes() == 0:
        return stats

    G_u = G.to_undirected() if G.is_directed() else G

    degrees = [d for _, d in G_u.degree()]
    stats["Avg Degree"] = round(float(np.mean(degrees)), 3)
    stats["Max Degree"] = int(max(degrees))
    stats["Avg Clustering Coeff"] = round(nx.average_clustering(G_u), 4)

    comps = list(nx.connected_components(G_u))
    stats["Num Components"] = len(comps)
    giant = max(comps, key=len)
    stats["Giant Component Size"] = len(giant)
    stats["Giant Component %"] = round(100 * len(giant) / G.number_of_nodes(), 1)

    G_giant = G_u.subgraph(giant).copy()
    n_g = len(giant)

    if n_g <= 1_500:
        try:
            stats["Diameter"] = nx.diameter(G_giant)
            stats["Avg Path Length"] = round(nx.average_shortest_path_length(G_giant), 3)
        except Exception:
            stats["Diameter"] = None
            stats["Avg Path Length"] = None
    else:
        # Approximate from a random sample of source nodes
        sample = list(np.random.choice(list(giant), min(300, n_g), replace=False))
        lengths: list[int] = []
        for node in sample:
            lengths.extend(nx.single_source_shortest_path_length(G_giant, node).values())
        stats["Diameter"] = int(max(lengths)) if lengths else None
        stats["Avg Path Length"] = round(float(np.mean(lengths)), 3) if lengths else None
        stats["Avg Path Length (note)"] = "approx. from 300-node sample"

    return stats


# ── Step 9 ────────────────────────────────────────────────────────────────────

def centrality_analysis(
    G: nx.Graph, top_n: int = 20
) -> tuple[dict[str, dict], dict[str, list]]:
    """
    Compute degree, closeness, betweenness, and eigenvector centrality.
    Returns (full_dicts, top_n_lists).
    """
    G_u = G.to_undirected() if G.is_directed() else G
    n = G_u.number_of_nodes()
    if n == 0:
        return {}, {}

    results: dict[str, dict] = {}
    results["degree"] = nx.degree_centrality(G_u)
    results["closeness"] = nx.closeness_centrality(G_u)

    k_sample = min(500, n) if n > 500 else None
    results["betweenness"] = nx.betweenness_centrality(G_u, k=k_sample, normalized=True)

    try:
        results["eigenvector"] = nx.eigenvector_centrality(G_u, max_iter=1_000, tol=1e-6)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        results["eigenvector"] = nx.pagerank(G_u)  # stable fallback

    top_nodes = {
        metric: sorted(vals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for metric, vals in results.items()
    }
    return results, top_nodes


# ── Step 10 ───────────────────────────────────────────────────────────────────

def detect_communities(
    G: nx.Graph,
) -> tuple[dict, dict[int, list], float]:
    """
    Community detection using Louvain (python-louvain) with greedy fallback.
    Returns (node→community_id dict, community_id→[nodes] dict, modularity).
    """
    G_u = G.to_undirected() if G.is_directed() else G.copy()
    G_u.remove_edges_from(nx.selfloop_edges(G_u))

    if G_u.number_of_nodes() == 0 or G_u.number_of_edges() == 0:
        return {}, {}, 0.0

    if _HAS_LOUVAIN:
        partition: dict = community_louvain.best_partition(G_u)
        modularity: float = community_louvain.modularity(partition, G_u)
    else:
        from networkx.algorithms.community import greedy_modularity_communities
        comm_sets = list(greedy_modularity_communities(G_u))
        partition = {node: i for i, comm in enumerate(comm_sets) for node in comm}
        modularity = nx.community.modularity(G_u, comm_sets)

    communities: dict[int, list] = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    print(f"  Communities: {len(communities)}  Modularity: {modularity:.4f}")
    return partition, dict(communities), modularity


# ── Step 11 ───────────────────────────────────────────────────────────────────

def assortativity_analysis(G: nx.Graph) -> dict[str, float]:
    """
    Compute degree assortativity and attribute assortativity for sentiment,
    community, post_count, and thread_count where data are available.
    """
    G_u = G.to_undirected() if G.is_directed() else G
    if G_u.number_of_edges() == 0:
        return {}

    results: dict[str, float] = {}
    results["degree"] = round(nx.degree_assortativity_coefficient(G_u), 4)

    n = G_u.number_of_nodes()
    coverage_threshold = 0.8

    for attr in ("sentiment", "community"):
        present = nx.get_node_attributes(G_u, attr)
        if len(present) >= n * coverage_threshold:
            try:
                r = nx.attribute_assortativity_coefficient(G_u, attr)
                results[attr] = round(r, 4)
            except Exception:
                pass

    for attr in ("post_count", "thread_count"):
        present = nx.get_node_attributes(G_u, attr)
        if len(present) >= n * coverage_threshold:
            try:
                r = nx.numeric_assortativity_coefficient(G_u, attr)
                results[attr] = round(r, 4)
            except Exception:
                pass

    return results


# ── Step 12 ───────────────────────────────────────────────────────────────────

def kcore_decomposition(
    G: nx.Graph,
) -> tuple[dict, nx.Graph | None, set, set]:
    """
    Compute k-core numbers for all nodes.
    Returns (core_number_dict, innermost_kcore_subgraph, core_nodes, peripheral_nodes).
    """
    G_u = G.to_undirected() if G.is_directed() else G.copy()
    G_u.remove_edges_from(nx.selfloop_edges(G_u))

    if G_u.number_of_nodes() == 0:
        return {}, None, set(), set()

    core_nums: dict = nx.core_number(G_u)
    max_k = max(core_nums.values())

    nx.set_node_attributes(G_u, core_nums, "core_number")
    kcore_sub = nx.k_core(G_u, k=max_k)

    core_nodes = {n for n, k in core_nums.items() if k == max_k}
    peripheral_nodes = {n for n, k in core_nums.items() if k == 1}

    print(f"  Max k: {max_k}  Core nodes: {len(core_nodes):,}  Peripheral: {len(peripheral_nodes):,}")
    return core_nums, kcore_sub, core_nodes, peripheral_nodes


# ── Step 13 ───────────────────────────────────────────────────────────────────

def topic_influence_analysis(topic_proj: nx.Graph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare topic popularity (raw frequency in corpus) vs network centrality
    (degree and betweenness in the topic co-occurrence projection).
    """
    # Frequency from raw data
    freq: dict[str, int] = defaultdict(int)
    for kws in df["matched_keywords"]:
        for kw in str(kws).split(","):
            kw = kw.strip()
            if kw:
                freq[kw] += 1

    if topic_proj.number_of_nodes() == 0:
        return pd.DataFrame(columns=["topic", "frequency", "degree_centrality", "betweenness_centrality"])

    deg_cent = nx.degree_centrality(topic_proj)
    n = topic_proj.number_of_nodes()
    k_s = min(200, n) if n > 200 else None
    betw_cent = nx.betweenness_centrality(topic_proj, k=k_s, normalized=True)

    rows = [
        {
            "topic": t,
            "frequency": freq.get(t, 0),
            "degree_centrality": round(deg_cent.get(t, 0.0), 6),
            "betweenness_centrality": round(betw_cent.get(t, 0.0), 6),
        }
        for t in set(freq) | set(deg_cent)
    ]

    result = (
        pd.DataFrame(rows)
        .query("frequency > 0")
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
    result["freq_rank"] = result["frequency"].rank(ascending=False, method="min").astype(int)
    result["centrality_rank"] = result["degree_centrality"].rank(ascending=False, method="min").astype(int)
    result["rank_diff"] = (result["freq_rank"] - result["centrality_rank"]).abs()

    return result
