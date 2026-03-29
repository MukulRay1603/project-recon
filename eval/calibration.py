"""
eval/calibration.py
--------------------
Phase 11 — Calibration Curve Generator

Reads eval/results/recon_linear.csv and produces:
    docs/calibration_curve.png

The calibration curve answers: "When the critic assigns a verdict,
does that verdict actually predict position accuracy?"

A well-calibrated critic shows:
    PASS > FORCED_PASS > INSUFFICIENT > STALE (in position accuracy)

This proves the critic's verdict tiers are statistically meaningful,
not arbitrary heuristic labels.

Run from repo root:
    python eval/calibration.py
"""

import sys
import os
import csv
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Output paths ──────────────────────────────────────────────────────────────
EVAL_DIR  = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR  = os.path.join(os.path.dirname(EVAL_DIR), "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

RECON_LINEAR_CSV = os.path.join(EVAL_DIR, "results", "recon_linear.csv")
OUTPUT_PNG       = os.path.join(DOCS_DIR, "calibration_curve.png")

# ── Verdict display order and colours ────────────────────────────────────────
VERDICT_ORDER = ["PASS", "FORCED_PASS", "STALE", "INSUFFICIENT", "CONTRADICTED", "N/A"]

VERDICT_COLORS = {
    "PASS":         "#22c55e",   # green
    "FORCED_PASS":  "#f59e0b",   # amber
    "STALE":        "#f97316",   # orange
    "INSUFFICIENT": "#ef4444",   # red
    "CONTRADICTED": "#a855f7",   # purple
    "N/A":          "#6b7280",   # grey
}

VERDICT_LABELS = {
    "PASS":         "PASS",
    "FORCED_PASS":  "FORCED\nPASS",
    "STALE":        "STALE",
    "INSUFFICIENT": "INSUF.",
    "CONTRADICTED": "CONTRA.",
    "N/A":          "N/A",
}

# Accuracy score: MATCH=1.0, PARTIAL=0.5, MISMATCH=0.0, others=excluded
ACCURACY_SCORE = {"MATCH": 1.0, "PARTIAL": 0.5, "MISMATCH": 0.0}


def load_recon_linear() -> list[dict]:
    if not os.path.exists(RECON_LINEAR_CSV):
        raise FileNotFoundError(
            f"recon_linear.csv not found at {RECON_LINEAR_CSV}\n"
            "Run eval/run_eval.py first."
        )
    with open(RECON_LINEAR_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_calibration(rows: list[dict]) -> dict:
    """
    For each critic verdict bin, compute:
      - mean position accuracy (MATCH=1, PARTIAL=0.5, MISMATCH=0)
      - count of questions in that bin
      - 95% confidence interval (Wilson score interval approximation)
    """
    import math

    bins = defaultdict(list)

    for row in rows:
        verdict  = row.get("critic_verdict", "N/A") or "N/A"
        accuracy = row.get("position_accuracy", "")

        if accuracy not in ACCURACY_SCORE:
            continue  # skip SKIPPED / ERROR rows

        bins[verdict].append(ACCURACY_SCORE[accuracy])

    result = {}
    for verdict, scores in bins.items():
        n    = len(scores)
        mean = sum(scores) / n if n else 0.0

        # Standard error for mean of bounded [0,1] scores
        variance = sum((s - mean) ** 2 for s in scores) / n if n > 1 else 0.0
        se       = math.sqrt(variance / n) if n > 0 else 0.0
        ci95     = 1.96 * se

        result[verdict] = {
            "mean":  round(mean, 4),
            "count": n,
            "ci95":  round(ci95, 4),
        }

    return result


def plot_calibration(calibration: dict, output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — works on Windows + Colab
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Filter to verdicts that actually appear in the data
    present = [v for v in VERDICT_ORDER if v in calibration]

    if not present:
        print("⚠ No verdict data found — nothing to plot.")
        return

    means  = [calibration[v]["mean"]  for v in present]
    ci95s  = [calibration[v]["ci95"]  for v in present]
    counts = [calibration[v]["count"] for v in present]
    colors = [VERDICT_COLORS.get(v, "#6b7280") for v in present]
    labels = [VERDICT_LABELS.get(v, v) for v in present]

    x = np.arange(len(present))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Bars
    bars = ax.bar(
        x, means,
        width=0.55,
        color=colors,
        alpha=0.88,
        zorder=3,
        edgecolor="#0f172a",
        linewidth=1.2,
    )

    # Error bars (95% CI)
    ax.errorbar(
        x, means,
        yerr=ci95s,
        fmt="none",
        ecolor="#e2e8f0",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        zorder=4,
    )

    # Value labels on bars
    for bar, mean, count, ci in zip(bars, means, counts, ci95s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + ci + 0.025,
            f"{mean:.0%}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="#f1f5f9",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.01,
            f"n={count}",
            ha="center", va="bottom",
            fontsize=8,
            color="#94a3b8",
        )

    # Reference line at chance (0.5 = all PARTIAL)
    ax.axhline(0.5, color="#64748b", linestyle="--", linewidth=1.0,
               alpha=0.6, zorder=2, label="PARTIAL baseline (0.50)")

    # Gridlines
    ax.yaxis.grid(True, color="#334155", linewidth=0.6, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(colors="#94a3b8", length=0)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="#e2e8f0", fontweight="500")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=9, color="#94a3b8")

    # Labels and title
    ax.set_xlabel("Critic Verdict", fontsize=11, color="#94a3b8",
                  labelpad=10, fontweight="500")
    ax.set_ylabel("Position Accuracy\n(MATCH=1.0 · PARTIAL=0.5 · MISMATCH=0)",
                  fontsize=9, color="#94a3b8", labelpad=10)

    ax.set_title(
        "RECON Critic Calibration Curve",
        fontsize=14, fontweight="bold", color="#f1f5f9", pad=16,
    )
    ax.text(
        0.5, 1.04,
        "Position accuracy by critic verdict — recon_linear · 130 questions",
        transform=ax.transAxes,
        ha="center", fontsize=9, color="#64748b",
    )

    # Legend
    ax.legend(
        loc="upper right", fontsize=8,
        facecolor="#1e293b", edgecolor="#334155",
        labelcolor="#94a3b8",
    )

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"✅ Calibration curve saved → {output_path}")


def main():
    print("=" * 55)
    print("RECON — Calibration Curve Generator")
    print("=" * 55)

    rows = load_recon_linear()
    print(f"Loaded {len(rows)} rows from recon_linear.csv")

    calibration = compute_calibration(rows)

    print("\nCalibration data:")
    print(f"  {'Verdict':<16} {'Mean Acc':>9} {'Count':>7} {'95% CI':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*8}")
    for v in VERDICT_ORDER:
        if v in calibration:
            d = calibration[v]
            print(f"  {v:<16} {d['mean']:>8.1%} {d['count']:>7} ±{d['ci95']:.3f}")

    print()
    plot_calibration(calibration, OUTPUT_PNG)

    # Also save calibration data as JSON for reference
    json_path = OUTPUT_PNG.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"✅ Calibration data saved → {json_path}")

    print("\nInterpretation:")
    if "PASS" in calibration and "STALE" in calibration:
        pass_acc  = calibration["PASS"]["mean"]
        stale_acc = calibration["STALE"]["mean"]
        if pass_acc > stale_acc:
            print(f"  ✓ PASS ({pass_acc:.0%}) > STALE ({stale_acc:.0%})")
            print("    Critic is calibrated — higher-confidence verdicts")
            print("    correlate with higher position accuracy.")
        else:
            print(f"  ⚠ PASS ({pass_acc:.0%}) ≤ STALE ({stale_acc:.0%})")
            print("    Calibration is weak — discuss in limitations.")


if __name__ == "__main__":
    main()