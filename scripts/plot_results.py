#!/usr/bin/env python3
"""Plot values transcribed from the manuscript; this script does not run experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ALBERT", "DistilBERT", "DistilRoBERTa", "Proposed Model"]
COLORS = ["#98a4b3", "#5b7dba", "#93b9d1", "#137d77"]
DOMAINS = ["movie", "restaurant", "taxi", "dstc1"]
DOMAIN_NAMES = {"movie": "Movie", "restaurant": "Restaurant", "taxi": "Taxi", "dstc1": "DSTC1", "mdc": "MDC"}
FOOTNOTE = "Manuscript-reported values · Not reproduced experiment results"


def read_table(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / name).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("source") != "manuscript" for row in rows):
        raise ValueError(f"Expected manuscript-only values in {name}")
    return rows


def save(figure, output: Path, name: str, formats: list[str]) -> None:
    figure.text(0.5, 0.015, FOOTNOTE, ha="center", fontsize=9, color="#546173")
    for extension in formats:
        metadata = {"Date": None} if extension == "svg" else ({"CreationDate": None, "ModDate": None} if extension == "pdf" else {})
        figure.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight", facecolor="white", metadata=metadata)
    plt.close(figure)


def clean_axes(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#cbd2db")
    axis.grid(axis="y", alpha=0.22, zorder=0)
    axis.set_axisbelow(True)


def in_domain(rows, output, formats) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
    for axis, domain in zip(axes, DOMAINS):
        lookup = {row["model"]: float(row["f1"]) for row in rows if row["setting"] == "in_domain" and row["test_domain"] == domain}
        values = [lookup[model] for model in MODELS]
        bars = axis.bar(np.arange(4), values, color=COLORS, width=0.68, zorder=3)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=9)
        axis.set_xticks([])
        axis.set_title(DOMAIN_NAMES[domain], fontsize=12, pad=10)
        axis.set_ylim(0, 1.085)
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        clean_axes(axis)
    axes[0].set_ylabel("F1 score")
    figure.suptitle("Politeness classification within each domain", fontsize=16, fontweight="bold", y=0.99)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    figure.legend(handles, MODELS, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4, frameon=False)
    figure.subplots_adjust(bottom=0.21, top=0.8, wspace=0.17)
    save(figure, output, "in_domain_f1", formats)


def distribution(rows, output, formats) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.8))
    palette = ["#ca6456", "#e6bd64", "#7aaec5", "#267e79"]
    counts = np.array([[int(next(row["count"] for row in rows if row["domain"] == domain and int(row["label"]) == label)) for label in range(4)] for domain in DOMAINS])
    shares = counts / counts.sum(axis=1)[:, None]
    left = np.zeros(4)
    names = ["0 · Impolite", "1 · Somewhat impolite", "2 · Somewhat polite", "3 · Polite"]
    for label in range(4):
        axis.barh(np.arange(4), shares[:, label], left=left, height=0.62, color=palette[label], label=names[label])
        for index in range(4):
            axis.text(left[index] + shares[index, label] / 2, index, f"{shares[index, label]:.1%}", ha="center", va="center", color="white" if label in (0, 3) else "#142334", fontsize=10)
        left += shares[:, label]
    axis.set_yticks(np.arange(4), [f"{DOMAIN_NAMES[d]}  (n = {int(n):,})" for d, n in zip(DOMAINS, counts.sum(axis=1))])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xlabel("Share of the class counts printed in the manuscript")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#cbd2db")
    axis.tick_params(axis="y", length=0)
    figure.suptitle("Four politeness levels across conversational domains", fontsize=16, fontweight="bold", y=0.99)
    figure.legend(loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=2, frameon=False, fontsize=10)
    figure.subplots_adjust(left=0.23, right=0.98, top=0.84, bottom=0.28)
    save(figure, output, "class_distribution", formats)


def cross_domain(rows, output, formats) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    domains = DOMAINS[:3]
    values = np.array([[float(next(row["f1"] for row in rows if row["model"] == "Proposed Model" and row["train_domain"] == train and row["test_domain"] == test)) for test in domains] for train in domains])
    artist = axis.imshow(values, vmin=0.8, vmax=1.0, cmap="YlGnBu")
    for i in range(3):
        for j in range(3):
            axis.text(j, i, f"{values[i, j]:.3f}" + ("\nin-domain" if i == j else ""), ha="center", va="center", fontsize=14, color="white" if values[i, j] >= 0.91 else "#142334")
    axis.set_xticks(range(3), [DOMAIN_NAMES[d] for d in domains])
    axis.set_yticks(range(3), [DOMAIN_NAMES[d] for d in domains])
    axis.set_xlabel("Test domain", labelpad=10)
    axis.set_ylabel("Training domain", labelpad=10)
    axis.tick_params(length=0)
    axis.spines[:].set_visible(False)
    colorbar = figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("F1 score (color scale: 0.80–1.00)")
    figure.suptitle("Transfer between MDC domains", fontsize=16, fontweight="bold", y=0.99)
    axis.set_title("Proposed hierarchical Transformer", fontsize=11, pad=16, color="#546173")
    figure.subplots_adjust(top=0.82, bottom=0.13, left=0.21, right=0.85)
    save(figure, output, "cross_domain_f1", formats)


def annotation_phases(rows, output, formats) -> None:
    """Compare printed marginal class counts; individual label transitions are unknown."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    phase_colors = ["#98a4b3", "#137d77"]
    class_names = ["0\nImpolite", "1\nSomewhat\nimpolite", "2\nSomewhat\npolite", "3\nPolite"]
    for axis, domain in zip(axes, ["mdc", "dstc1"]):
        totals = []
        for index, phase in enumerate(["1", "2"]):
            values = [int(next(row["count"] for row in rows if row["domain"] == domain and row["phase"] == phase and int(row["label"]) == label)) for label in range(4)]
            bars = axis.bar(np.arange(4) + (index - 0.5) * 0.37, values, width=0.34, color=phase_colors[index], zorder=3)
            axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4, fontsize=9)
            totals.append(sum(values))
        axis.set_title(f"{DOMAIN_NAMES[domain]}\nPhase 1 total {totals[0]:,} · Phase 2 total {totals[1]:,}", fontsize=11, pad=12)
        axis.set_xticks(range(4), class_names, fontsize=9)
        axis.set_ylim(0, 11000)
        axis.set_yticks(np.arange(0, 10001, 2000))
        axis.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        clean_axes(axis)
    axes[0].set_ylabel("Number of utterances printed in the phase figures")
    figure.suptitle("Class distributions across annotation phases", fontsize=16, fontweight="bold", y=0.99)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in phase_colors]
    figure.legend(handles, ["Phase 1", "Phase 2"], loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.28, wspace=0.14)
    save(figure, output, "annotation_phases", formats)


def cross_dataset(rows, output, formats) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for axis, train, test in zip(axes, ["mdc", "dstc1"], ["dstc1", "mdc"]):
        selected = {row["model"]: row for row in rows if row["setting"] == "cross_dataset" and row["train_domain"] == train}
        for index, (model, color) in enumerate(zip(MODELS, COLORS)):
            positions = np.arange(2) + (index - 1.5) * 0.19
            values = [float(selected[model][metric]) for metric in ["f1", "accuracy"]]
            bars = axis.bar(positions, values, width=0.17, color=color, zorder=3)
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
        axis.set_xticks([0, 1], ["F1", "Accuracy"])
        axis.set_ylim(0, 1.02)
        axis.set_title(f"{DOMAIN_NAMES[train]} → {DOMAIN_NAMES[test]}", fontsize=12, pad=12)
        clean_axes(axis)
    axes[0].set_ylabel("Score")
    figure.suptitle("Transfer between MDC and DSTC1", fontsize=16, fontweight="bold", y=0.99)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    figure.legend(handles, MODELS, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4, frameon=False)
    figure.subplots_adjust(top=0.8, bottom=0.23, wspace=0.18)
    save(figure, output, "cross_dataset", formats)


def hyperparameters(rows, output, formats, proposed: bool) -> None:
    panels = [("Proposed Model", "2"), ("Proposed Model", "4")] if proposed else [(model, "2") for model in MODELS[:3]]
    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.5), sharey=True)
    for axis, (model, epochs) in zip(axes, panels):
        selected = [row for row in rows if row["model"] == model and row["epochs"] == epochs]
        values = np.array([[float(next(row["f1"] for row in selected if row["learning_rate"] == lr and int(row["batch_size"]) == batch)) for batch in [8, 16, 32]] for lr in ["1e-5", "2e-5", "3e-5", "4e-5"]])
        artist = axis.imshow(values, vmin=0.8, vmax=0.98, cmap="YlGnBu", aspect="auto")
        for i in range(4):
            for j in range(3):
                axis.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=11, color="white" if values[i, j] >= 0.9 else "#142334")
        axis.set_xticks(range(3), ["8", "16", "32"])
        axis.set_yticks(range(4), ["1e-5", "2e-5", "3e-5", "4e-5"])
        axis.set_xlabel("Batch size")
        axis.set_title(f"{epochs} epochs" if proposed else model, fontsize=12, pad=10)
        axis.tick_params(length=0)
        axis.spines[:].set_visible(False)
    axes[0].set_ylabel("Learning rate")
    title = "Proposed model: validation F1" if proposed else "Baseline models: validation F1 at 2 epochs"
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    figure.subplots_adjust(top=0.81, bottom=0.2, left=0.1, right=0.87, wspace=0.15)
    color_axis = figure.add_axes([0.89, 0.2, 0.018, 0.61])
    figure.colorbar(artist, cax=color_axis, label="F1 (0.80–0.98)")
    save(figure, output, "hyperparameters_proposed" if proposed else "hyperparameters_baselines", formats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "docs/results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--formats", nargs="+", choices=["png", "svg", "pdf"], default=["png", "svg"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": "#273449", "text.color": "#172638", "xtick.color": "#46566b", "ytick.color": "#46566b", "svg.fonttype": "none", "svg.hashsalt": "pacd-manuscript"})
    classification = read_table(args.results_dir, "classification.csv")
    counts = read_table(args.results_dir, "class_distribution.csv")
    phases = read_table(args.results_dir, "annotation_phases.csv")
    parameters = read_table(args.results_dir, "hyperparameters.csv")
    in_domain(classification, args.output_dir, args.formats)
    distribution(counts, args.output_dir, args.formats)
    annotation_phases(phases, args.output_dir, args.formats)
    cross_domain(classification, args.output_dir, args.formats)
    cross_dataset(classification, args.output_dir, args.formats)
    hyperparameters(parameters, args.output_dir, args.formats, proposed=True)
    hyperparameters(parameters, args.output_dir, args.formats, proposed=False)
    print(f"Wrote seven manuscript figures in {', '.join(args.formats)} format.")


if __name__ == "__main__":
    main()
