"""
eval/contradiction_viz.py
--------------------------
Phase 11 — Contradiction Network Graph Generator

Reads eval/results/recon_linear.csv and produces:
    docs/contradiction_graph.png

The graph shows:
  Nodes  = retrieved papers (sized by citation count)
  Edges  = citation relationships between retrieved papers
  Color  = red edges where critic flagged CONTRADICTED verdict
           grey edges for standard citations (PASS/STALE)
  Labels = paper title (truncated) + year

This is the visual that goes in the README and spreads on LinkedIn.
It makes the system's reasoning visible at a glance.

Run from repo root:
    python eval/contradiction_viz.py
"""

import sys
import os
import csv
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Output paths ──────────────────────────────────────────────────────────────
EVAL_DIR  = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR  = os.path.join(os.path.dirname(EVAL_DIR), "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

RECON_LINEAR_CSV = os.path.join(EVAL_DIR, "results", "recon_linear.csv")
OUTPUT_PNG       = os.path.join(DOCS_DIR, "contradiction_graph.png")


def load_rows() -> list[dict]:
    if not os.path.exists(RECON_LINEAR_CSV):
        raise FileNotFoundError(
            f"recon_linear.csv not found.\nRun eval/run_eval.py first."
        )
    with open(RECON_LINEAR_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_citations(position_text: str) -> list[tuple[str, str]]:
    """
    Extract (author, year) citation pairs from synthesized position text.
    Matches patterns like [Smith et al., 2023] or [Smith, 2023].
    Returns list of (label, year) tuples.
    """
    pattern = r"\[([A-Za-z][^,\[\]]{1,40}?),?\s*(?:et al\.?)?,?\s*(\d{4})[a-z]?\]"
    matches = re.findall(pattern, position_text)
    return [(author.strip(), year.strip()) for author, year in matches]


def build_graph_data(rows: list[dict]) -> dict:
    """
    Build node and edge data from CSV rows.

    Nodes: unique (author, year) citation pairs found in synthesized positions.
    Edges: co-appearance in the same synthesized position.
    Red edges: positions where critic_verdict == CONTRADICTED.
    Grey edges: all other positions.

    Returns {
        "nodes": {node_id: {"label": str, "year": int, "count": int, "contested": bool}},
        "edges": [(node_a, node_b, {"color": str, "weight": int, "verdict": str})],
    }
    """
    from collections import defaultdict

    nodes   = defaultdict(lambda: {"label": "", "year": 0, "count": 0, "contested": False})
    edge_weights = defaultdict(lambda: {"weight": 0, "contested": False, "verdict": "PASS"})

    for row in rows:
        verdict  = row.get("critic_verdict", "") or ""
        position = row.get("synthesized_position", "") or ""

        if not position:
            continue

        citations = extract_citations(position)
        if len(citations) < 2:
            continue

        is_contested = (verdict == "CONTRADICTED")

        # Add/update nodes
        for author, year in citations:
            node_id = f"{author}_{year}"
            nodes[node_id]["label"] = f"{author}\n({year})"
            nodes[node_id]["year"]  = int(year) if year.isdigit() else 0
            nodes[node_id]["count"] += 1
            if is_contested:
                nodes[node_id]["contested"] = True

        # Add edges between all citation pairs in this position
        seen = list(set(f"{a}_{y}" for a, y in citations))
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                edge_key = tuple(sorted([seen[i], seen[j]]))
                edge_weights[edge_key]["weight"] += 1
                if is_contested:
                    edge_weights[edge_key]["contested"] = True
                    edge_weights[edge_key]["verdict"]   = "CONTRADICTED"

    # Convert to lists
    node_dict = dict(nodes)
    edge_list = [
        (a, b, {
            "weight":    data["weight"],
            "contested": data["contested"],
            "verdict":   data["verdict"],
            "color":     "#ef4444" if data["contested"] else "#334155",
        })
        for (a, b), data in edge_weights.items()
    ]

    return {"nodes": node_dict, "edges": edge_list}


def plot_graph(graph_data: dict, output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    import numpy as np

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    if not nodes:
        print("⚠ No citation nodes found — nothing to plot.")
        return

    # ── Build NetworkX graph ──────────────────────────────────────────────────
    G = nx.Graph()

    for node_id, data in nodes.items():
        G.add_node(node_id, **data)

    for a, b, data in edges:
        if a in nodes and b in nodes:
            G.add_edge(a, b, **data)

    # Prune: keep only nodes with degree >= 1, cap at 60 nodes for readability
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    if len(G.nodes()) > 60:
        # Keep top-60 by degree
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:60]
        remove    = [n for n in G.nodes() if n not in top_nodes]
        G.remove_nodes_from(remove)

    if len(G.nodes()) == 0:
        print("⚠ No connected nodes after pruning — nothing to plot.")
        return

    print(f"  Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # ── Layout ────────────────────────────────────────────────────────────────
    # Spring layout with seed for reproducibility
    # Use kamada_kawai for smaller graphs (cleaner), spring for larger
    if len(G.nodes()) <= 30:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.5)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.8, iterations=80)

    # ── Node sizing by citation count ────────────────────────────────────────
    counts    = [nodes.get(n, {}).get("count", 1) for n in G.nodes()]
    max_count = max(counts) if counts else 1
    node_sizes = [
        200 + 800 * (c / max_count) for c in counts
    ]

    # Node colors: red-tinted for contested nodes, blue otherwise
    node_colors = [
        "#7f1d1d" if nodes.get(n, {}).get("contested", False) else "#1e3a5f"
        for n in G.nodes()
    ]
    node_borders = [
        "#ef4444" if nodes.get(n, {}).get("contested", False) else "#3b82f6"
        for n in G.nodes()
    ]

    # ── Edge separation ───────────────────────────────────────────────────────
    red_edges  = [(a, b) for a, b, d in G.edges(data=True) if d.get("contested")]
    grey_edges = [(a, b) for a, b, d in G.edges(data=True) if not d.get("contested")]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    # Grey citation edges
    if grey_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=grey_edges,
            edge_color="#334155", width=0.8,
            alpha=0.5, ax=ax,
        )

    # Red contradiction edges (drawn on top)
    if red_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=red_edges,
            edge_color="#ef4444", width=2.2,
            alpha=0.85, ax=ax,
            style="solid",
        )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=1.5,
        ax=ax,
    )

    # Labels — only for higher-degree nodes to avoid clutter
    degree_threshold = 2 if len(G.nodes()) > 25 else 1
    label_nodes = {
        n: (nodes.get(n, {}).get("label", n)[:28])
        for n in G.nodes()
        if G.degree(n) >= degree_threshold
    }

    nx.draw_networkx_labels(
        G, pos,
        labels=label_nodes,
        font_size=6.5,
        font_color="#cbd5e1",
        font_weight="normal",
        ax=ax,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#1e3a5f", edgecolor="#3b82f6",
                       linewidth=1.5, label="Paper (citation node)"),
        mpatches.Patch(facecolor="#7f1d1d", edgecolor="#ef4444",
                       linewidth=1.5, label="Paper (in contested query)"),
        plt.Line2D([0], [0], color="#334155", linewidth=1.5,
                   label="Citation co-appearance"),
        plt.Line2D([0], [0], color="#ef4444", linewidth=2.5,
                   label="CONTRADICTED verdict"),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=8.5,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
        framealpha=0.9,
        borderpad=0.8,
    )

    # ── Stats annotation ──────────────────────────────────────────────────────
    n_red  = len(red_edges)
    n_grey = len(grey_edges)
    ax.text(
        0.02, 0.98,
        f"{len(G.nodes())} papers · {n_grey} citation edges · {n_red} contradicted",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8, color="#64748b",
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "RECON Citation & Contradiction Network",
        fontsize=15, fontweight="bold",
        color="#f1f5f9", pad=14,
    )
    ax.text(
        0.5, 1.02,
        "Nodes = cited papers · Red edges = CONTRADICTED verdict · "
        "Node size = citation frequency · recon_linear · 130 questions",
        transform=ax.transAxes,
        ha="center", fontsize=8, color="#64748b",
    )

    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"✅ Contradiction graph saved → {output_path}")


def main():
    print("=" * 55)
    print("RECON — Contradiction Network Graph Generator")
    print("=" * 55)

    rows = load_rows()
    print(f"Loaded {len(rows)} rows from recon_linear.csv")

    contradicted = [r for r in rows if r.get("critic_verdict") == "CONTRADICTED"]
    print(f"CONTRADICTED verdict rows: {len(contradicted)}")

    print("\nBuilding citation graph from synthesized positions...")
    graph_data = build_graph_data(rows)

    n_nodes = len(graph_data["nodes"])
    n_edges = len(graph_data["edges"])
    n_red   = sum(1 for _, _, d in graph_data["edges"] if d.get("contested"))

    print(f"  Nodes extracted:  {n_nodes}")
    print(f"  Edges extracted:  {n_edges}")
    print(f"  Red (contested):  {n_red}")
    print(f"  Grey (citation):  {n_edges - n_red}")

    if n_nodes == 0:
        print("\n⚠ No citation nodes found in synthesized positions.")
        print("  This means position text had no [Author, Year] citations.")
        print("  Check a few rows in recon_linear.csv to confirm.")
        return

    print("\nRendering graph...")
    plot_graph(graph_data, OUTPUT_PNG)

    print("\nDone. Embed in README with:")
    print("  ![Contradiction Graph](docs/contradiction_graph.png)")


if __name__ == "__main__":
    main()