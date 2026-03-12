#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=2",
#     "pandas>=2",
#     "scipy>=1.12",
# ]
# ///
"""Visualize co-occurrence matrix and bucket count histogram from mass-spectrometry-counts output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]
from matplotlib.colors import LogNorm


def load_data(results_dir: Path):
    with open(results_dir / "metadata.json") as f:
        metadata = json.load(f)

    npz_path = results_dir / "cooccurrence.npz"
    tsv_path = results_dir / "cooccurrence.tsv"

    if npz_path.exists():
        mat = sp.load_npz(npz_path)
        coo = mat.tocoo()
        df = pd.DataFrame({"i": coo.row, "j": coo.col, "count": coo.data})
    elif tsv_path.exists():
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=["i", "j", "count"],
            dtype={"i": np.int32, "j": np.int32, "count": np.int64},
        )
    else:
        raise FileNotFoundError(f"No cooccurrence.npz or cooccurrence.tsv in {results_dir}")

    return metadata, df


@overload
def bucket_to_mz(idx: int, min_mz: float, bin_width: float) -> float: ...


@overload
def bucket_to_mz(
    idx: npt.NDArray[np.integer[Any]],
    min_mz: float,
    bin_width: float,
) -> npt.NDArray[np.floating[Any]]: ...


def bucket_to_mz(
    idx: int | npt.NDArray[np.integer[Any]],
    min_mz: float,
    bin_width: float,
) -> float | npt.NDArray[np.floating[Any]]:
    return min_mz + (idx + 0.5) * bin_width


def plot_bucket_counts(df: pd.DataFrame, metadata: dict, output_dir: Path):
    diag = df[df["i"] == df["j"]].copy()
    diag = diag.sort_values("i")

    mz = bucket_to_mz(diag["i"].values, metadata["min_mz"], metadata["bin_width"])
    counts = diag["count"].values

    # Crop to active range with a small margin
    active = counts > 0
    mz_active = mz[active]
    counts_active = counts[active]
    xlim = (mz_active.min() - 5, mz_active.max() + 5)
    title_base = f"Per-bucket spectrum counts ({metadata['num_spectra']:,} spectra, {len(diag):,} active buckets)"

    # Log-scale plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(mz_active, counts_active, width=metadata["bin_width"], linewidth=0, color="#2196F3")
    ax.set_yscale("log")
    ax.set_xlabel("m/z (Da)")
    ax.set_ylabel("Spectrum count (log scale)")
    ax.set_title(f"{title_base} — log scale")
    ax.set_xlim(*xlim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    log_path = output_dir / "bucket_counts_log.png"
    fig.savefig(log_path, dpi=200)
    plt.close(fig)
    print(f"Saved bucket counts histogram (log) to {log_path}")

    # Linear-scale plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(mz_active, counts_active, width=metadata["bin_width"], linewidth=0, color="#2196F3")
    ax.set_xlabel("m/z (Da)")
    ax.set_ylabel("Spectrum count")
    ax.set_title(f"{title_base} — linear scale")
    ax.set_xlim(*xlim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    lin_path = output_dir / "bucket_counts_linear.png"
    fig.savefig(lin_path, dpi=200)
    plt.close(fig)
    print(f"Saved bucket counts histogram (linear) to {lin_path}")


def build_cooccurrence_matrix(df: pd.DataFrame, metadata: dict, downsample: int = 10):
    """Downsample and build a symmetric dense co-occurrence submatrix for the active region."""
    coarse_i = df["i"].values // downsample
    coarse_j = df["j"].values // downsample
    counts = df["count"].values

    coarse_df = pd.DataFrame({"i": coarse_i, "j": coarse_j, "count": counts})
    coarse_df = coarse_df.groupby(["i", "j"], sort=False)["count"].sum().reset_index()

    n_coarse = metadata["num_buckets"] // downsample

    mat = sp.coo_matrix(
        (coarse_df["count"].values, (coarse_df["i"].values, coarse_df["j"].values)),
        shape=(n_coarse, n_coarse),
    ).tocsr()

    mat_full = mat + mat.T - sp.diags(mat.diagonal(), dtype=mat.dtype)

    row_nnz = np.diff(mat_full.indptr)
    active_rows = np.where(row_nnz > 0)[0]
    if len(active_rows) == 0:
        return None, 0.0, 0.0, 0.0

    lo, hi = active_rows[0], active_rows[-1] + 1
    dense = mat_full[lo:hi, lo:hi].toarray().astype(np.float64)

    coarse_bin_width = metadata["bin_width"] * downsample
    mz_lo = metadata["min_mz"] + lo * coarse_bin_width
    mz_hi = metadata["min_mz"] + hi * coarse_bin_width

    return dense, mz_lo, mz_hi, coarse_bin_width


def plot_cooccurrence_heatmap(dense: np.ndarray, metadata: dict, mz_lo: float, mz_hi: float,
                              coarse_bin_width: float, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 9))
    plot_data = dense.copy()
    plot_data[plot_data == 0] = np.nan
    im = ax.imshow(
        plot_data,
        origin="lower",
        extent=(mz_lo, mz_hi, mz_lo, mz_hi),
        aspect="equal",
        cmap="inferno",
        norm=LogNorm(vmin=1, vmax=np.nanmax(plot_data)),
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Co-occurrence count", shrink=0.82)
    ax.set_xlabel("m/z (Da)")
    ax.set_ylabel("m/z (Da)")
    ax.set_title(
        f"Peak co-occurrence ({metadata['num_spectra']:,} spectra, "
        f"{coarse_bin_width:g} Da bins)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved co-occurrence heatmap to {output_path}")


def plot_pmi_heatmap(dense: np.ndarray, metadata: dict, mz_lo: float, mz_hi: float,
                     coarse_bin_width: float, output_path: Path):
    N = metadata["num_spectra"]
    diag = np.diag(dense).copy()
    # P(i) = diag(i) / N, P(i,j) = cooc(i,j) / N
    # PMI = log2(P(i,j) / (P(i)*P(j))) = log2(N * cooc(i,j) / (diag(i) * diag(j)))
    # Compute the denominator: outer product of marginals
    marginal_outer = np.outer(diag, diag)

    # Mask cells where either marginal is zero or co-occurrence is zero
    valid = (dense > 0) & (marginal_outer > 0)
    pmi = np.full_like(dense, np.nan)
    pmi[valid] = np.log2(N * dense[valid] / marginal_outer[valid])

    vmax = np.nanmax(np.abs(pmi))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(
        pmi,
        origin="lower",
        extent=(mz_lo, mz_hi, mz_lo, mz_hi),
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="PMI (bits)", shrink=0.82)
    ax.set_xlabel("m/z (Da)")
    ax.set_ylabel("m/z (Da)")
    ax.set_title(
        f"Peak co-occurrence PMI ({metadata['num_spectra']:,} spectra, "
        f"{coarse_bin_width:g} Da bins)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved PMI heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot mass-spectrometry-counts results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing cooccurrence.tsv and metadata.json (default: results/)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Downsample factor for heatmap (default: 10, i.e. 0.1 Da -> 1 Da bins)",
    )
    args = parser.parse_args()

    metadata, df = load_data(args.results_dir)
    print(f"Loaded {len(df):,} entries from {args.results_dir / 'cooccurrence.tsv'}")

    plot_bucket_counts(df, metadata, args.results_dir)

    dense, mz_lo, mz_hi, coarse_bin_width = build_cooccurrence_matrix(df, metadata, downsample=args.downsample)
    if dense is None:
        print("No active entries found, skipping heatmaps.")
        return
    plot_cooccurrence_heatmap(dense, metadata, mz_lo, mz_hi, coarse_bin_width, args.results_dir / "cooccurrence_heatmap.png")
    plot_pmi_heatmap(dense, metadata, mz_lo, mz_hi, coarse_bin_width, args.results_dir / "cooccurrence_pmi.png")


if __name__ == "__main__":
    main()
