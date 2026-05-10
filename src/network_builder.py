"""
Network construction — steps 5, 6, 7.

Step 5: Thread Similarity Network  (nodes=threads, edges=keyword cosine similarity)
Step 6: User Interaction Network   (nodes=users,   edges=co-participation in threads)
Step 7: Bipartite User–Topic Network and projections
"""

from itertools import combinations

import networkx as nx
import networkx.algorithms.bipartite as bipartite
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _parse_keywords(kw_str) -> list[str]:
    if pd.isna(kw_str) or not str(kw_str).strip():
        return []
    return [k.strip() for k in str(kw_str).split(",") if k.strip()]


# ── Step 5 ────────────────────────────────────────────────────────────────────

def build_thread_similarity_network(
    df: pd.DataFrame,
    threshold: float = 0.15,
    max_threads: int = 5_000,
) -> nx.Graph:
    """
    Nodes = threads.  Edges = cosine similarity of TF-IDF keyword vectors >= threshold.
    Limits to max_threads most-active threads for tractability.
    """
    print("Building thread similarity network...")

    # Aggregate keywords per thread into one text document
    thread_docs = (
        df.groupby("thread_id")["matched_keywords"]
        .apply(lambda x: " ".join(kw for s in x for kw in _parse_keywords(s)))
        .reset_index(name="kw_text")
    )
    thread_docs = thread_docs[thread_docs["kw_text"].str.strip() != ""]

    if len(thread_docs) > max_threads:
        original_count = len(thread_docs)
        thread_docs["kw_count"] = thread_docs["kw_text"].str.split().str.len()
        thread_docs = thread_docs.nlargest(max_threads, "kw_count")
        print(f"  Sampled {max_threads} most-active threads from {original_count}")

    threads = thread_docs["thread_id"].tolist()
    docs = thread_docs["kw_text"].tolist()
    print(f"  Computing pairwise similarity for {len(threads)} threads...")

    tfidf = TfidfVectorizer(min_df=1)
    mat = tfidf.fit_transform(docs)
    sim = cosine_similarity(mat)  # dense, OK for ≤5 000 threads

    G = nx.Graph()
    G.add_nodes_from(threads)

    n = len(threads)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sim[i, j])
            if w >= threshold:
                G.add_edge(threads[i], threads[j], weight=w)

    # Keyword list as node attribute
    thread_kws = df.groupby("thread_id")["matched_keywords"].apply(
        lambda x: list({kw for s in x for kw in _parse_keywords(s)})
    )
    for tid in threads:
        G.nodes[tid]["keywords"] = thread_kws.get(tid, [])

    if "sentiment" in df.columns:
        dominant = df.groupby("thread_id")["sentiment"].agg(
            lambda x: x.value_counts().index[0]
        )
        for tid in threads:
            if tid in G:
                G.nodes[tid]["sentiment"] = dominant.get(tid, "neutral")

    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    return G


# ── Step 6 ────────────────────────────────────────────────────────────────────

def build_user_interaction_network(
    df: pd.DataFrame,
    min_posts: int = 2,
) -> nx.Graph:
    """
    Nodes = users.  Edges = co-participation in threads, weight = shared thread count.
    Falls back to post_id when user_id is absent or all-unknown.
    """
    print("Building user interaction network...")

    user_col = _resolve_user_col(df)

    # Filter to active users
    counts = df[user_col].value_counts()
    active = counts[counts >= min_posts].index
    df_f = df[df[user_col].isin(active)]
    print(f"  Active users (≥{min_posts} posts): {len(active):,}")

    G = nx.Graph()
    thread_users = df_f.groupby("thread_id")[user_col].apply(
        lambda x: list({str(u) for u in x})
    )

    for users in thread_users:
        G.add_nodes_from(users)
        for u1, u2 in combinations(users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]["weight"] += 1
            else:
                G.add_edge(u1, u2, weight=1)

    post_cnt = df_f.groupby(user_col).size()
    thread_cnt = df_f.groupby(user_col)["thread_id"].nunique()
    for node in G.nodes():
        G.nodes[node]["post_count"] = int(post_cnt.get(node, 0))
        G.nodes[node]["thread_count"] = int(thread_cnt.get(node, 0))

    if "sentiment" in df_f.columns:
        dominant = df_f.groupby(user_col)["sentiment"].agg(
            lambda x: x.value_counts().index[0]
        )
        for node in G.nodes():
            G.nodes[node]["sentiment"] = dominant.get(node, "neutral")

    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    return G


# ── Step 7 ────────────────────────────────────────────────────────────────────

def build_bipartite_user_topic_network(
    df: pd.DataFrame,
) -> tuple[nx.Graph, nx.Graph, nx.Graph]:
    """
    Bipartite graph:  users (bipartite=0) ↔ topics/keywords (bipartite=1).
    Returns (bipartite_G, user_projection, topic_projection).
    """
    print("Building bipartite user–topic network...")

    user_col = _resolve_user_col(df)

    # Explode rows to (user, keyword) pairs efficiently
    df_tmp = df[[user_col, "matched_keywords"]].copy()
    df_tmp["kw_list"] = df_tmp["matched_keywords"].apply(_parse_keywords)
    df_exp = df_tmp.explode("kw_list").dropna(subset=["kw_list"])
    df_exp = df_exp[df_exp["kw_list"] != ""].rename(columns={"kw_list": "topic"})

    edge_weights = (
        df_exp.groupby([user_col, "topic"]).size().reset_index(name="weight")
    )

    B = nx.Graph()
    all_users = set(df[user_col].astype(str))
    B.add_nodes_from(all_users, bipartite=0)

    for _, row in edge_weights.iterrows():
        user = str(row[user_col])
        topic = row["topic"]
        if topic not in B:
            B.add_node(topic, bipartite=1)
        B.add_edge(user, topic, weight=int(row["weight"]))

    user_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0 and B.degree(n) > 0}
    topic_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1 and B.degree(n) > 0}

    print(f"  Bipartite: {len(user_nodes):,} users, {len(topic_nodes):,} topics")

    user_proj = bipartite.weighted_projected_graph(B, user_nodes)
    topic_proj = bipartite.weighted_projected_graph(B, topic_nodes)

    # We remove low-weight edges to reveal the true network backbone.
    # This allows centrality metrics to actually differentiate the nodes.
    def _filter_backbone(G: nx.Graph, min_weight: int) -> nx.Graph:
        G_filtered = G.copy()
        weak_edges = [(u, v) for u, v, d in G_filtered.edges(data=True) if d.get('weight', 1) < min_weight]
        G_filtered.remove_edges_from(weak_edges)
        # Remove nodes that became isolated after edge removal
        G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))
        return G_filtered

    # Apply the filter. You can increase min_weight (e.g., to 3 or 4) 
    # if the centrality charts still look too identical.
    user_proj = _filter_backbone(user_proj, min_weight=3)
    topic_proj = _filter_backbone(topic_proj, min_weight=3)

    print(f"  User projection (filtered) : {user_proj.number_of_nodes():,} nodes, {user_proj.number_of_edges():,} edges")
    print(f"  Topic projection (filtered): {topic_proj.number_of_nodes():,} nodes, {topic_proj.number_of_edges():,} edges")

    return B, user_proj, topic_proj


# ── Helper ────────────────────────────────────────────────────────────────────

def _resolve_user_col(df: pd.DataFrame) -> str:
    """Return 'user_id' if usable, otherwise fall back to 'post_id'."""
    if "user_id" in df.columns:
        unique_users = df["user_id"].nunique()
        total_rows = len(df)
        # If user_id is all 'unknown', fall back
        if unique_users > 1 or df["user_id"].iloc[0] != "unknown":
            return "user_id"
    print("  Note: falling back to 'post_id' as user proxy (re-run data_collection.py to get real user IDs)")
    return "post_id"
