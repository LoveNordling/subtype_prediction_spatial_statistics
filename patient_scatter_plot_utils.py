from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from spatial_utils import CK_POSITIVE_EXTERNAL_CLASSES


COLOR_STROMA = "#0072B2"
COLOR_TUMOR = "#E64B35"
SCATTER_SIZE = 1.2
SCATTER_ALPHA = 0.92
ZOOM_SCATTER_SIZE = 10.0
BOMI1_PX_PER_UM = 2.0


@dataclass(frozen=True)
class ImageSpec:
    image: np.ndarray
    original_width: int
    original_height: int


def cache_key(path: Path, max_dim: int) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{max_dim}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def canonicalize_bomi2_sample_name(raw_sample_name: str) -> str:
    text = str(raw_sample_name)
    if "_" in text:
        text = text[: text.rfind("_")]
    return text.replace("Core[1,", "[")


def canonicalize_from_cellpose_filename(filename: str) -> str:
    text = str(filename)
    m = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", text)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    m = re.search(r"(.+?)_\[(\d+,[A-Z])\]", text)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    raise ValueError(f"Cannot parse sample name from filename: {filename}")


def load_bomi2_allowed_sample_names(samples_labels_path: str) -> set[str]:
    df = pd.read_csv(samples_labels_path, low_memory=False)
    required = {"sample_name", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"BOMI2 samples labels file missing required columns {sorted(missing)}: {samples_labels_path}"
        )
    return set(df["sample_name"].astype(str).map(canonicalize_bomi2_sample_name))


def scale_channel(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=float)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def downsample_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return image
    h, w = image.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    if step == 1:
        return image
    if image.ndim == 2:
        return image[::step, ::step]
    return image[::step, ::step, :]


def load_bomi2_image(path: Path, max_dim: int, cache_dir: Path | None = None) -> ImageSpec | None:
    if not path.exists():
        return None
    if cache_dir is not None:
        key = cache_key(path, max_dim)
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            return ImageSpec(
                image=cached["image"],
                original_width=int(cached["original_width"]),
                original_height=int(cached["original_height"]),
            )
    arr = tifffile.imread(path, key=[0, 6])
    if arr.ndim != 3 or arr.shape[0] != 2:
        return None
    original_height = int(arr.shape[1])
    original_width = int(arr.shape[2])
    dapi = scale_channel(arr[0])
    ck = scale_channel(arr[1])
    rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=float)
    rgb[..., 1] = ck
    rgb[..., 2] = dapi
    out = ImageSpec(
        image=downsample_image(rgb, max_dim=max_dim),
        original_width=original_width,
        original_height=original_height,
    )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            image=out.image,
            original_width=np.array(out.original_width, dtype=np.int32),
            original_height=np.array(out.original_height, dtype=np.int32),
        )
    return out


def load_bomi1_image(path: Path, max_dim: int, cache_dir: Path | None = None) -> ImageSpec | None:
    if not path.exists():
        return None
    if cache_dir is not None:
        key = cache_key(path, max_dim)
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            return ImageSpec(
                image=cached["image"],
                original_width=int(cached["original_width"]),
                original_height=int(cached["original_height"]),
            )
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        x = scale_channel(arr)
        out = ImageSpec(
            image=downsample_image(np.stack([x, x, x], axis=-1), max_dim=max_dim),
            original_width=int(arr.shape[1]),
            original_height=int(arr.shape[0]),
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                image=out.image,
                original_width=np.array(out.original_width, dtype=np.int32),
                original_height=np.array(out.original_height, dtype=np.int32),
            )
        return out
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(float) / 255.0 if np.issubdtype(arr.dtype, np.integer) else scale_channel(arr[..., :3])
        out = ImageSpec(
            image=downsample_image(rgb, max_dim=max_dim),
            original_width=int(arr.shape[1]),
            original_height=int(arr.shape[0]),
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                image=out.image,
                original_width=np.array(out.original_width, dtype=np.int32),
                original_height=np.array(out.original_height, dtype=np.int32),
            )
        return out
    if arr.ndim == 3 and arr.shape[0] >= 3:
        rgb = arr[:3].transpose(1, 2, 0)
        rgb = rgb.astype(float) / 255.0 if np.issubdtype(rgb.dtype, np.integer) else scale_channel(rgb)
        out = ImageSpec(
            image=downsample_image(rgb, max_dim=max_dim),
            original_width=int(rgb.shape[1]),
            original_height=int(rgb.shape[0]),
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                image=out.image,
                original_width=np.array(out.original_width, dtype=np.int32),
                original_height=np.array(out.original_height, dtype=np.int32),
            )
        return out
    return None


def load_bomi2_cells(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=1)
    ck_col = "CK" if "CK" in head.columns else ("ck" if "ck" in head.columns else None)
    if ck_col is None:
        raise ValueError("Cellpose file missing CK/ck column")
    df = pd.read_csv(path, usecols=["filename", "x", "y", ck_col], low_memory=False)
    df["sample_name"] = df["filename"].map(canonicalize_from_cellpose_filename)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df[ck_col], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_bomi1_cells(cells_path: str, meta_path: str) -> pd.DataFrame:
    head = pd.read_csv(cells_path, nrows=1)
    columns = set(head.columns)

    if {"sample_name", "Class", "CentroidX_um", "CentroidY_um"}.issubset(columns):
        raw = pd.read_csv(
            cells_path,
            usecols=["sample_name", "Class", "CentroidX_um", "CentroidY_um"],
            low_memory=False,
        )
        df = raw.copy()
        df["CentroidX_um"] = pd.to_numeric(df["CentroidX_um"], errors="coerce")
        df["CentroidY_um"] = pd.to_numeric(df["CentroidY_um"], errors="coerce")
        df["x"] = (df["CentroidX_um"] - df.groupby("sample_name")["CentroidX_um"].transform("min")) * BOMI1_PX_PER_UM
        df["y"] = (df["CentroidY_um"] - df.groupby("sample_name")["CentroidY_um"].transform("min")) * BOMI1_PX_PER_UM
    elif {"sample_name", "Class", "CentroidX_px", "CentroidY_px"}.issubset(columns):
        raw = pd.read_csv(
            cells_path,
            usecols=["sample_name", "Class", "CentroidX_px", "CentroidY_px"],
            low_memory=False,
        )
        meta = pd.read_csv(meta_path, usecols=["sample_name", "X", "Y"], low_memory=False).drop_duplicates(subset=["sample_name"])
        df = raw.merge(meta, on="sample_name", how="inner")
        df["x"] = pd.to_numeric(df["CentroidX_px"], errors="coerce") - pd.to_numeric(df["X"], errors="coerce")
        df["y"] = pd.to_numeric(df["CentroidY_px"], errors="coerce") - pd.to_numeric(df["Y"], errors="coerce")
    else:
        raise ValueError(
            "BOMI1 cells file must contain either "
            "['sample_name', 'Class', 'CentroidX_um', 'CentroidY_um'] "
            "or ['sample_name', 'Class', 'CentroidX_px', 'CentroidY_px']."
        )

    df["CK"] = df["Class"].astype(str).str.strip().str.lower().isin(CK_POSITIVE_EXTERNAL_CLASSES).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y", "sample_name"])


def plot_scatter_panel(
    ax,
    g: pd.DataFrame,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    marker_size: float,
    zoom_box: tuple[float, float, float, float] | None = None,
    max_points: int | None = None,
) -> None:
    if max_points is not None and len(g) > max_points:
        g = g.sample(n=max_points, random_state=0)
    stroma = g[g["CK"] == 0]
    tumor = g[g["CK"] == 1]
    if len(stroma) > 0:
        ax.scatter(stroma["x"], stroma["y"], color=COLOR_STROMA, s=marker_size, alpha=SCATTER_ALPHA, linewidths=0, rasterized=True)
    if len(tumor) > 0:
        ax.scatter(tumor["x"], tumor["y"], color=COLOR_TUMOR, s=marker_size, alpha=SCATTER_ALPHA, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STROMA, markeredgecolor="none", markersize=6, label="Stroma"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TUMOR, markeredgecolor="none", markersize=6, label="Tumor"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=True, framealpha=0.9)
    if zoom_box is not None:
        zx0, zx1, zy0, zy1 = zoom_box
        ax.add_patch(
            Rectangle(
                (zx0, zy0),
                zx1 - zx0,
                zy1 - zy0,
                fill=False,
                edgecolor="lime",
                linewidth=1.8,
                linestyle="--",
            )
        )


def build_zoom_bounds(g: pd.DataFrame, zoom_factor: float) -> tuple[float, float, float, float]:
    x_min = float(g["x"].min())
    x_max = float(g["x"].max())
    y_min = float(g["y"].min())
    y_max = float(g["y"].max())
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    width = max(1.0, (x_max - x_min) / zoom_factor)
    height = max(1.0, (y_max - y_min) / zoom_factor)
    return (cx - width / 2.0, cx + width / 2.0, cy - height / 2.0, cy + height / 2.0)
