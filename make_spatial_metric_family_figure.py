#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np


COLOR_STROMA = "#0072B2"
COLOR_TUMOR = "#E64B35"
COLOR_EDGE = "#000000"
COLOR_BG = "#FFFFFF"
COLOR_PANEL = "#FFFFFF"
COLOR_TEXT = "#1E293B"
COLOR_MUTED = "#5B6470"
COLOR_DIVIDER = "#CBD5E1"
PLOT_XLIM = (0.0, 1.0)
PLOT_YLIM = (0.14, 0.82)
FIG_WIDTH = 4.4
FIG_HEIGHT = FIG_WIDTH * (PLOT_YLIM[1] - PLOT_YLIM[0]) / (PLOT_XLIM[1] - PLOT_XLIM[0])


FAMILIES = [
    {
        "title": "Ripley's L Function",
        "drawer": "ripley",
    },
    {
        "title": "Label Neighborhood Structure",
        "drawer": "label_neighborhood",
    },
    {
        "title": "Graph Centrality",
        "drawer": "centrality",
    },
    {
        "title": "Cancer-Stroma Co-Occurrence",
        "drawer": "cooccurrence",
    },
    {
        "title": "Object-Object Correlation",
        "drawer": "object_object",
    },
]

LEGACY_FAMILY_TITLES = [
    "Cell Density",
    "Compartment-Specific Clustering",
    "Compartment Separation",
    "Mixing and Connectivity",
    "Newman's Assortativity",
    "Neighborhood Enrichment",
]


def sample_cluster(center: tuple[float, float], n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    pts = rng.normal(loc=np.array(center), scale=scale, size=(n, 2))
    return np.clip(pts, 0.06, 0.94)


def sample_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.08, 0.92, size=(n, 2))


def draw_canvas(ax, panel_facecolor: str) -> None:
    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (PLOT_XLIM[0], PLOT_YLIM[0]),
            PLOT_XLIM[1] - PLOT_XLIM[0],
            PLOT_YLIM[1] - PLOT_YLIM[0],
            boxstyle="square,pad=0",
            linewidth=0,
            facecolor=panel_facecolor,
            zorder=0,
        )
    )


def draw_split_canvas(ax, panel_facecolor: str, left_label: str = "low", right_label: str = "high") -> None:
    draw_canvas(ax, panel_facecolor)
    ax.plot([0.5, 0.5], [0.365, 0.615], color=COLOR_DIVIDER, linewidth=2.0, alpha=0.85, zorder=1)


def place_points(points: np.ndarray, side: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if side == "left":
        offset = np.array([0.085, 0.285])
    elif side == "right":
        offset = np.array([0.565, 0.285])
    else:
        raise ValueError(f"Unknown side: {side}")
    return offset + points * 0.35


def place_square_points(points: np.ndarray, side: str) -> np.ndarray:
    return place_points(points, side)


def scale_unit_points(points: np.ndarray, factor: float, center: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    center_arr = np.asarray(center, dtype=float)
    return np.clip(center_arr + (points - center_arr) * float(factor), 0.03, 0.97)


def scatter_points(ax, points: np.ndarray, color: str, size: float = 44) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=size, c=color, linewidths=0, alpha=0.95, zorder=3)


def radius_edge_indices(points: np.ndarray, radius: float) -> list[tuple[int, int]]:
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=2))
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dists[i, j] <= radius:
                edges.append((i, j))
    return edges


def draw_ripley(ax, rng: np.random.Generator, panel_facecolor: str) -> None:
    draw_split_canvas(ax, panel_facecolor)
    dispersed = place_points(sample_uniform(58, rng), "left")
    clustered = place_points(
        np.vstack(
            [
                sample_cluster((0.32, 0.66), 24, 0.055, rng),
                sample_cluster((0.66, 0.38), 24, 0.055, rng),
                sample_uniform(14, rng),
            ]
        ),
        "right",
    )

    for center in (dispersed[10], clustered[5]):
        for radius in (0.050, 0.095, 0.140):
            ax.add_patch(Circle(center, radius, fill=False, color=COLOR_TUMOR, linewidth=2.1, alpha=0.30, zorder=1))

    scatter_points(ax, dispersed, COLOR_TUMOR, size=32)
    scatter_points(ax, clustered, COLOR_TUMOR, size=32)


def edge_segments(points: np.ndarray, edges: list[tuple[int, int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(points[a], points[b]) for a, b in edges]


def knn_edge_indices(points: np.ndarray, k: int) -> list[tuple[int, int]]:
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    edges: set[tuple[int, int]] = set()
    for i in range(len(points)):
        nbrs = np.argsort(dists[i])[:k]
        for j in nbrs:
            a, b = sorted((i, int(j)))
            edges.add((a, b))
    return sorted(edges)


def draw_label_graph(ax, points: np.ndarray, labels: np.ndarray, k: int = 3) -> None:
    same_edges = []
    mixed_edges = []
    for a, b in knn_edge_indices(points, k=k):
        if labels[a] == labels[b]:
            same_edges.append((a, b))
        else:
            mixed_edges.append((a, b))
    if mixed_edges:
        ax.add_collection(LineCollection(edge_segments(points, mixed_edges), colors="#A3A3A3", linewidths=3.0, alpha=0.42, zorder=1))
    if same_edges:
        ax.add_collection(LineCollection(edge_segments(points, same_edges), colors=COLOR_EDGE, linewidths=3.8, alpha=0.50, zorder=2))


def draw_label_neighborhood(ax, rng: np.random.Generator, panel_facecolor: str) -> None:
    draw_split_canvas(ax, panel_facecolor)
    mixed_base = np.array(
        [
            [0.20, 0.72],
            [0.30, 0.76],
            [0.40, 0.70],
            [0.52, 0.74],
            [0.64, 0.66],
            [0.72, 0.56],
            [0.66, 0.46],
            [0.56, 0.38],
            [0.46, 0.34],
            [0.34, 0.28],
            [0.24, 0.34],
            [0.18, 0.46],
            [0.28, 0.54],
            [0.38, 0.58],
            [0.48, 0.54],
            [0.58, 0.50],
            [0.38, 0.44],
            [0.30, 0.64],
            [0.66, 0.32],
            [0.74, 0.40],
            [0.16, 0.60],
            [0.54, 0.24],
            [0.44, 0.22],
            [0.62, 0.78],
            [0.26, 0.22],
            [0.72, 0.70],
            [0.20, 0.28],
            [0.50, 0.64],
            [0.60, 0.60],
            [0.42, 0.78],
            [0.32, 0.40],
            [0.70, 0.24],
            [0.52, 0.44],
            [0.24, 0.68],
        ]
    )
    mixed_base = scale_unit_points(np.clip(mixed_base + rng.normal(0, 0.012, size=mixed_base.shape), 0.08, 0.92), 1.5)
    mixed_points = place_points(mixed_base, "left")
    mixed_labels = np.array(([1, 0] * 17)[: len(mixed_points)])

    red_region = np.array(
        [
            [0.28, 0.72],
            [0.34, 0.76],
            [0.40, 0.72],
            [0.46, 0.68],
            [0.30, 0.64],
            [0.38, 0.62],
            [0.45, 0.60],
            [0.52, 0.58],
            [0.33, 0.56],
            [0.42, 0.54],
            [0.50, 0.52],
            [0.56, 0.50],
            [0.25, 0.58],
            [0.36, 0.48],
            [0.46, 0.46],
        ]
    )
    bridge = np.array(
        [
            [0.50, 0.47],
            [0.55, 0.46],
            [0.58, 0.43],
            [0.61, 0.41],
            [0.53, 0.40],
            [0.59, 0.36],
            [0.64, 0.48],
            [0.66, 0.44],
            [0.68, 0.40],
            [0.70, 0.36],
            [0.57, 0.51],
            [0.62, 0.52],
            [0.66, 0.50],
        ]
    )
    blue_region = np.array(
        [
            [0.58, 0.36],
            [0.64, 0.39],
            [0.70, 0.38],
            [0.80, 0.34],
            [0.62, 0.30],
            [0.70, 0.30],
            [0.78, 0.27],
            [0.62, 0.24],
            [0.68, 0.22],
            [0.74, 0.23],
            [0.84, 0.26],
            [0.60, 0.29],
            [0.82, 0.42],
            [0.72, 0.46],
            [0.86, 0.36],
            [0.76, 0.32],
            [0.70, 0.34],
            [0.64, 0.31],
        ]
    )
    segregated_base = np.vstack([red_region, bridge, blue_region])
    segregated_base = scale_unit_points(
        np.clip(segregated_base + rng.normal(0, 0.012, size=segregated_base.shape), 0.08, 0.92),
        1.5,
    )
    segregated_points = place_points(segregated_base, "right")
    segregated_labels = np.array(
        [1] * len(red_region)
        + [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
        + [0] * len(blue_region)
    )

    draw_label_graph(ax, mixed_points, mixed_labels, k=3)
    draw_label_graph(ax, segregated_points, segregated_labels, k=3)
    scatter_points(ax, mixed_points[mixed_labels == 0], COLOR_STROMA, size=46)
    scatter_points(ax, mixed_points[mixed_labels == 1], COLOR_TUMOR, size=46)
    scatter_points(ax, segregated_points[segregated_labels == 0], COLOR_STROMA, size=46)
    scatter_points(ax, segregated_points[segregated_labels == 1], COLOR_TUMOR, size=46)


def draw_centrality(ax, rng: np.random.Generator, panel_facecolor: str) -> None:
    draw_split_canvas(ax, panel_facecolor)
    graph_radius = 0.080
    low_base = np.array(
        [
            [0.22, 0.72],
            [0.32, 0.76],
            [0.42, 0.72],
            [0.52, 0.68],
            [0.62, 0.62],
            [0.70, 0.54],
            [0.64, 0.44],
            [0.54, 0.38],
            [0.44, 0.34],
            [0.34, 0.28],
            [0.24, 0.32],
            [0.18, 0.42],
            [0.24, 0.52],
            [0.34, 0.58],
            [0.44, 0.56],
            [0.54, 0.52],
            [0.46, 0.46],
            [0.36, 0.44],
            [0.30, 0.66],
            [0.58, 0.30],
            [0.68, 0.36],
            [0.16, 0.60],
            [0.72, 0.70],
            [0.40, 0.20],
        ]
    )
    low_base = scale_unit_points(np.clip(low_base + rng.normal(0, 0.014, size=low_base.shape), 0.08, 0.92), 1.5)
    low_points = place_points(low_base, "left")
    low_edges = radius_edge_indices(low_points, radius=graph_radius)
    ax.add_collection(LineCollection(edge_segments(low_points, low_edges), colors=COLOR_EDGE, linewidths=2.5, alpha=0.25, zorder=1))
    scatter_points(ax, low_points, COLOR_TUMOR, size=38)

    high_base = np.vstack(
        [
            sample_cluster((0.50, 0.52), 22, 0.090, rng),
            sample_uniform(16, rng) * 0.78 + 0.11,
        ]
    )
    high_base = scale_unit_points(high_base, 1.5)
    high_points = place_points(high_base, "right")
    high_edges = radius_edge_indices(high_points, radius=graph_radius)
    ax.add_collection(LineCollection(edge_segments(high_points, high_edges), colors=COLOR_EDGE, linewidths=2.6, alpha=0.26, zorder=1))
    scatter_points(ax, high_points, COLOR_TUMOR, size=38)


def knn_edges(points: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    edges: set[tuple[int, int]] = set()
    for i in range(len(points)):
        nbrs = np.argsort(dists[i])[:k]
        for j in nbrs:
            a, b = sorted((i, int(j)))
            edges.add((a, b))
    return [(points[a], points[b]) for a, b in sorted(edges)]


def draw_cooccurrence(ax, rng: np.random.Generator, panel_facecolor: str) -> None:
    draw_split_canvas(ax, panel_facecolor)
    left_t = place_points(scale_unit_points(sample_cluster((0.40, 0.56), 26, 0.090, rng), 1.2), "left")
    left_s = place_points(scale_unit_points(sample_cluster((0.74, 0.44), 26, 0.090, rng), 1.2), "left")
    right_t = place_points(scale_unit_points(sample_uniform(30, rng) * 0.62 + 0.19, 1.2), "right")
    right_s = right_t + rng.normal(0, 0.030, size=right_t.shape)
    right_s[:, 0] = np.clip(right_s[:, 0], 0.55, 0.93)
    right_s[:, 1] = np.clip(right_s[:, 1], 0.14, 0.86)

    centers = [left_t[8], right_t[8]]
    for center in centers:
        ax.add_patch(Circle(center, 0.105, fill=False, color=COLOR_EDGE, linewidth=3.2, alpha=0.40, zorder=1))
    scatter_points(ax, left_s, COLOR_STROMA, size=38)
    scatter_points(ax, left_t, COLOR_TUMOR, size=38)
    scatter_points(ax, right_s, COLOR_STROMA, size=38)
    scatter_points(ax, right_t, COLOR_TUMOR, size=38)


def draw_neighborhood_enrichment(ax, rng: np.random.Generator, panel_facecolor: str) -> None:
    draw_split_canvas(ax, panel_facecolor, left_label="null-like", right_label="enriched")
    left_base = sample_uniform(32, rng) * 0.72 + 0.14
    left_points = place_points(left_base, "left")
    left_labels = np.array(([1, 0] * 16)[: len(left_points)])

    right_base = np.vstack(
        [
            sample_cluster((0.34, 0.62), 13, 0.07, rng),
            sample_cluster((0.70, 0.36), 13, 0.07, rng),
            sample_uniform(6, rng) * 0.62 + 0.19,
        ]
    )
    right_points = place_points(right_base, "right")
    right_labels = np.array([1] * 13 + [0] * 13 + [1, 0, 1, 0, 1, 0])

    draw_label_graph(ax, left_points, left_labels, k=3)
    draw_label_graph(ax, right_points, right_labels, k=3)
    scatter_points(ax, left_points[left_labels == 0], COLOR_STROMA, size=44)
    scatter_points(ax, left_points[left_labels == 1], COLOR_TUMOR, size=44)
    scatter_points(ax, right_points[right_labels == 0], COLOR_STROMA, size=44)
    scatter_points(ax, right_points[right_labels == 1], COLOR_TUMOR, size=44)


def draw_object_object(ax, rng: np.random.Generator, panel_facecolor: str) -> None:
    draw_split_canvas(ax, panel_facecolor, left_label="anti", right_label="co-local")
    for side in ("left", "right"):
        x0 = 0.085 if side == "left" else 0.565
        y0 = 0.285
        size = 0.35
        for i in range(5):
            pos = i / 4 * size
            ax.plot([x0 + pos, x0 + pos], [y0, y0 + size], color=COLOR_MUTED, linewidth=1.3, alpha=0.25, zorder=1)
            ax.plot([x0, x0 + size], [y0 + pos, y0 + pos], color=COLOR_MUTED, linewidth=1.3, alpha=0.25, zorder=1)

    left_t = place_square_points(sample_cluster((0.32, 0.68), 20, 0.09, rng), "left")
    left_s = place_square_points(sample_cluster((0.70, 0.32), 20, 0.09, rng), "left")
    right_t = place_square_points(
        np.vstack([sample_cluster((0.35, 0.66), 13, 0.08, rng), sample_cluster((0.68, 0.34), 13, 0.08, rng)]),
        "right",
    )
    right_s = right_t + rng.normal(0, 0.025, size=right_t.shape)
    right_s[:, 0] = np.clip(right_s[:, 0], 0.55, 0.93)
    right_s[:, 1] = np.clip(right_s[:, 1], 0.14, 0.86)

    cell = 0.35 / 4
    for xy in ((0.085 + cell, 0.285 + 2 * cell), (0.085 + 2 * cell, 0.285 + cell)):
        ax.add_patch(Rectangle(xy, cell, cell, facecolor="#D7E8F5", edgecolor="none", alpha=1.0, zorder=1))
    for xy in ((0.565 + cell, 0.285 + 2 * cell), (0.565 + 2 * cell, 0.285 + cell)):
        ax.add_patch(Rectangle(xy, cell, cell, facecolor="#D7E8F5", edgecolor="none", alpha=1.0, zorder=1))
    scatter_points(ax, left_s, COLOR_STROMA, size=42)
    scatter_points(ax, left_t, COLOR_TUMOR, size=42)
    scatter_points(ax, right_s, COLOR_STROMA, size=42)
    scatter_points(ax, right_t, COLOR_TUMOR, size=42)


def draw_family_panel(ax, family: dict[str, str], seed: int, panel_facecolor: str) -> None:
    rng = np.random.default_rng(seed)
    drawer = family["drawer"]
    if drawer == "ripley":
        draw_ripley(ax, rng, panel_facecolor)
    elif drawer == "label_neighborhood":
        draw_label_neighborhood(ax, rng, panel_facecolor)
    elif drawer == "centrality":
        draw_centrality(ax, rng, panel_facecolor)
    elif drawer == "cooccurrence":
        draw_cooccurrence(ax, rng, panel_facecolor)
    elif drawer == "neighborhood_enrichment":
        draw_neighborhood_enrichment(ax, rng, panel_facecolor)
    elif drawer == "object_object":
        draw_object_object(ax, rng, panel_facecolor)
    else:
        raise ValueError(f"Unknown drawer: {drawer}")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def make_family_panel(family: dict[str, str], out_dir: Path, seed: int) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLOR_BG)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=1,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )

    stem = slugify(family["title"])
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"

    ax_plot = fig.add_subplot(gs[0, 0])
    draw_family_panel(ax_plot, family, seed=seed, panel_facecolor=COLOR_PANEL)
    fig.savefig(png_path, dpi=300, facecolor=COLOR_BG, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="none")
    gs = fig.add_gridspec(
        nrows=1,
        ncols=1,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )
    ax_plot = fig.add_subplot(gs[0, 0])
    draw_family_panel(ax_plot, family, seed=seed, panel_facecolor="none")
    fig.savefig(svg_path, facecolor="none", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def remove_legacy_panels(out_dir: Path) -> None:
    current_stems = {slugify(family["title"]) for family in FAMILIES}
    stale_stems = {slugify(title) for title in LEGACY_FAMILY_TITLES} - current_stems
    for stem in stale_stems:
        for suffix in (".png", ".svg"):
            path = out_dir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a first-pass schematic figure for spatial metric families.")
    parser.add_argument("--out-dir", default="outputs/figures/paper_figures/06_metric_families")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_panels(out_dir)
    for idx, family in enumerate(FAMILIES, start=1):
        make_family_panel(family, out_dir, seed=idx)
    print(f"[DONE] Wrote metric family panels to {out_dir}")


if __name__ == "__main__":
    main()
