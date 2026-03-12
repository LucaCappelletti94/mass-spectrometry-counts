#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=2",
#     "scipy>=1.12",
# ]
# ///
"""Analyze count and co-occurrence distributions from mass-spectrometry-counts output.

Produces:
  1. Bucket count distribution: rank-frequency, histogram of log-counts, Q-Q vs log-normal
  2. Top outlier fragments (unusually high counts)
  3. Co-occurrence distribution: histogram, top outlier pairs (highest raw and highest PMI)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp  # type: ignore[import-untyped]
import scipy.stats as stats  # type: ignore[import-untyped]


def load_data(results_dir: Path):
    with open(results_dir / "metadata.json") as f:
        metadata = json.load(f)

    npz_path = results_dir / "cooccurrence.npz"
    mat = sp.load_npz(npz_path)
    return metadata, mat


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


def analyze_counts(mat: sp.csr_matrix, metadata: dict, output_dir: Path, label: str):
    """Analyze diagonal (per-bucket counts) distribution."""
    diag = mat.diagonal()
    active_mask = diag > 0
    counts = diag[active_mask]
    active_indices = np.where(active_mask)[0]
    mz_values = bucket_to_mz(active_indices, metadata["min_mz"], metadata["bin_width"])

    n_active = len(counts)
    N = metadata["num_spectra"]
    print(f"\n{'='*60}")
    print(f"BUCKET COUNT ANALYSIS — {label}")
    print(f"{'='*60}")
    print(f"Active buckets: {n_active:,} / {metadata['num_buckets']:,}")
    print(f"Total spectra:  {N:,}")

    # Summary statistics
    log_counts = np.log(counts)
    mu, sigma = np.mean(log_counts), np.std(log_counts)
    print(f"\nSummary statistics (active buckets):")
    print(f"  Min:    {counts.min():>12,}")
    print(f"  P25:    {int(np.percentile(counts, 25)):>12,}")
    print(f"  Median: {int(np.median(counts)):>12,}")
    print(f"  Mean:   {int(np.mean(counts)):>12,}")
    print(f"  P75:    {int(np.percentile(counts, 75)):>12,}")
    print(f"  P99:    {int(np.percentile(counts, 99)):>12,}")
    print(f"  Max:    {counts.max():>12,}")
    print(f"  Log-normal params: mu={mu:.2f}, sigma={sigma:.2f}")

    # Find outliers: z-score in log-space
    z_scores = (log_counts - mu) / sigma
    # Top 20 by count
    top_idx = np.argsort(counts)[::-1][:20]
    print(f"\nTop 20 fragments by count:")
    print(f"  {'Rank':>4}  {'m/z':>8}  {'Count':>12}  {'% spectra':>10}  {'z-score':>8}")
    for rank, idx in enumerate(top_idx, 1):
        mz = mz_values[idx]
        c = counts[idx]
        pct = 100.0 * c / N
        z = z_scores[idx]
        print(f"  {rank:>4}  {mz:>8.2f}  {c:>12,}  {pct:>9.1f}%  {z:>8.2f}")

    # Outliers: z > 3 in log-space
    outlier_mask = z_scores > 3
    n_outliers = outlier_mask.sum()
    print(f"\nOutliers (z > 3 in log-space): {n_outliers} buckets")
    if n_outliers > 0 and n_outliers <= 50:
        outlier_idx = np.where(outlier_mask)[0]
        outlier_idx = outlier_idx[np.argsort(counts[outlier_idx])[::-1]]
        for idx in outlier_idx:
            mz = mz_values[idx]
            c = counts[idx]
            z = z_scores[idx]
            print(f"    m/z {mz:>8.2f}  count={c:>12,}  z={z:.2f}")

    # --- Plot 1: Rank-frequency ---
    sorted_counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.loglog(ranks, sorted_counts, ".", markersize=1.5, alpha=0.6, color="#2196F3")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Spectrum count")
    ax.set_title("Rank-frequency (log-log)")
    ax.grid(True, alpha=0.3)

    # Fit power law to middle portion (avoid head/tail)
    mid = (ranks > 10) & (ranks < len(ranks) * 0.9)
    if mid.sum() > 10:
        slope, intercept, r, _, _ = stats.linregress(np.log10(ranks[mid]), np.log10(sorted_counts[mid]))
        fit_line = 10 ** (intercept + slope * np.log10(ranks))
        ax.plot(ranks, fit_line, "r--", linewidth=1, label=f"power law slope={slope:.2f}, R²={r**2:.2f}")
        ax.legend(fontsize=8)

    # --- Plot 2: Histogram of log10(counts) ---
    ax = axes[1]
    log10_counts = np.log10(counts)
    ax.hist(log10_counts, bins=80, density=True, alpha=0.7, color="#2196F3", edgecolor="none")
    # Overlay log-normal PDF
    x = np.linspace(log10_counts.min(), log10_counts.max(), 200)
    # Convert log-normal params to log10 space
    mu10 = mu / np.log(10)
    sigma10 = sigma / np.log(10)
    pdf = stats.norm.pdf(x, mu10, sigma10)
    ax.plot(x, pdf, "r-", linewidth=2, label=f"Log-normal fit\n(μ={mu10:.2f}, σ={sigma10:.2f})")
    ax.set_xlabel("log₁₀(count)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of log₁₀(bucket counts)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Q-Q plot vs log-normal ---
    ax = axes[2]
    theoretical = np.sort(stats.norm.ppf(np.linspace(0.001, 0.999, n_active)))
    observed = np.sort((log_counts - mu) / sigma)
    # Subsample if too many points
    if len(observed) > 5000:
        idx = np.linspace(0, len(observed) - 1, 5000, dtype=int)
        theoretical_sub = theoretical[idx]
        observed_sub = observed[idx]
    else:
        theoretical_sub = theoretical
        observed_sub = observed
    ax.scatter(theoretical_sub, observed_sub, s=2, alpha=0.5, color="#2196F3")
    lim = max(abs(theoretical_sub.min()), abs(theoretical_sub.max()), abs(observed_sub.min()), abs(observed_sub.max()))
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
    ax.set_xlabel("Theoretical quantiles (normal)")
    ax.set_ylabel("Observed quantiles (log counts)")
    ax.set_title("Q-Q plot vs log-normal")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Bucket count distribution — {label}", fontsize=14)
    fig.tight_layout()
    path = output_dir / "count_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nSaved count distribution plot to {path}")

    return counts, active_indices, mz_values, z_scores


def analyze_cooccurrences(mat: sp.csr_matrix, metadata: dict, output_dir: Path, label: str):
    """Analyze off-diagonal co-occurrence distribution."""
    # Extract upper triangle (excluding diagonal)
    coo = sp.triu(mat, k=1).tocoo()
    offdiag_counts = coo.data
    rows = coo.row
    cols = coo.col

    N = metadata["num_spectra"]
    bin_width = metadata["bin_width"]
    min_mz = metadata["min_mz"]
    diag = mat.diagonal().astype(np.float64)

    n_pairs = len(offdiag_counts)
    print(f"\n{'='*60}")
    print(f"CO-OCCURRENCE ANALYSIS — {label}")
    print(f"{'='*60}")
    print(f"Off-diagonal nonzero pairs: {n_pairs:,}")

    if n_pairs == 0:
        print("No off-diagonal entries found.")
        return

    print(f"\nOff-diagonal count statistics:")
    print(f"  Min:    {offdiag_counts.min():>12,}")
    print(f"  P25:    {int(np.percentile(offdiag_counts, 25)):>12,}")
    print(f"  Median: {int(np.median(offdiag_counts)):>12,}")
    print(f"  Mean:   {int(np.mean(offdiag_counts)):>12,}")
    print(f"  P75:    {int(np.percentile(offdiag_counts, 75)):>12,}")
    print(f"  P99:    {int(np.percentile(offdiag_counts, 99)):>12,}")
    print(f"  Max:    {offdiag_counts.max():>12,}")

    # Top 20 by raw co-occurrence count
    top_raw_idx = np.argsort(offdiag_counts)[::-1][:20]
    print(f"\nTop 20 co-occurring pairs by raw count:")
    print(f"  {'Rank':>4}  {'m/z_i':>8}  {'m/z_j':>8}  {'Count':>12}  {'c_i':>10}  {'c_j':>10}  {'Expected':>12}  {'Obs/Exp':>8}")
    for rank, idx in enumerate(top_raw_idx, 1):
        i, j = rows[idx], cols[idx]
        c = offdiag_counts[idx]
        ci, cj = diag[i], diag[j]
        mz_i = bucket_to_mz(i, min_mz, bin_width)
        mz_j = bucket_to_mz(j, min_mz, bin_width)
        expected = ci * cj / N
        ratio = c / expected if expected > 0 else float("inf")
        print(f"  {rank:>4}  {mz_i:>8.2f}  {mz_j:>8.2f}  {c:>12,}  {int(ci):>10,}  {int(cj):>10,}  {expected:>12,.0f}  {ratio:>8.2f}")

    # Compute PMI for all pairs: PMI = log2(N * cooc / (ci * cj))
    ci = diag[rows]
    cj = diag[cols]
    valid = (ci > 0) & (cj > 0) & (offdiag_counts > 0)
    pmi = np.full(n_pairs, np.nan)
    pmi[valid] = np.log2(N * offdiag_counts[valid].astype(np.float64) / (ci[valid] * cj[valid]))

    # For PMI outlier analysis, require minimum support to avoid noise
    min_support = max(100, int(N * 1e-5))  # at least 100 or 0.001% of spectra
    high_support = valid & (offdiag_counts >= min_support)
    print(f"\nPMI analysis (min support = {min_support:,}):")
    print(f"  Pairs with sufficient support: {high_support.sum():,}")

    if high_support.sum() > 0:
        pmi_supported = pmi[high_support]
        pmi_mean = np.nanmean(pmi_supported)
        pmi_std = np.nanstd(pmi_supported)
        print(f"  PMI mean: {pmi_mean:.2f} bits")
        print(f"  PMI std:  {pmi_std:.2f} bits")

        # Top 20 by PMI (with support filter)
        supported_indices = np.where(high_support)[0]
        pmi_order = np.argsort(pmi[supported_indices])[::-1][:20]
        print(f"\nTop 20 co-occurring pairs by PMI (min support {min_support:,}):")
        print(f"  {'Rank':>4}  {'m/z_i':>8}  {'m/z_j':>8}  {'PMI':>8}  {'Count':>12}  {'c_i':>10}  {'c_j':>10}")
        for rank, oi in enumerate(pmi_order, 1):
            idx = supported_indices[oi]
            i, j = rows[idx], cols[idx]
            c = offdiag_counts[idx]
            mz_i = bucket_to_mz(i, min_mz, bin_width)
            mz_j = bucket_to_mz(j, min_mz, bin_width)
            print(f"  {rank:>4}  {mz_i:>8.2f}  {mz_j:>8.2f}  {pmi[idx]:>8.2f}  {c:>12,}  {int(diag[i]):>10,}  {int(diag[j]):>10,}")

        # Bottom 20 by PMI (most anti-correlated, with support)
        pmi_bottom = np.argsort(pmi[supported_indices])[:20]
        print(f"\nBottom 20 co-occurring pairs by PMI (most anti-correlated, min support {min_support:,}):")
        print(f"  {'Rank':>4}  {'m/z_i':>8}  {'m/z_j':>8}  {'PMI':>8}  {'Count':>12}  {'c_i':>10}  {'c_j':>10}")
        for rank, oi in enumerate(pmi_bottom, 1):
            idx = supported_indices[oi]
            i, j = rows[idx], cols[idx]
            c = offdiag_counts[idx]
            mz_i = bucket_to_mz(i, min_mz, bin_width)
            mz_j = bucket_to_mz(j, min_mz, bin_width)
            print(f"  {rank:>4}  {mz_i:>8.2f}  {mz_j:>8.2f}  {pmi[idx]:>8.2f}  {c:>12,}  {int(diag[i]):>10,}  {int(diag[j]):>10,}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Raw co-occurrence distribution
    log10_cooc = np.log10(offdiag_counts.astype(np.float64))

    ax = axes[0, 0]
    ax.hist(log10_cooc, bins=100, density=True, alpha=0.7, color="#FF9800", edgecolor="none")
    ax.set_xlabel("log₁₀(co-occurrence count)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of off-diagonal co-occurrences")
    ax.grid(True, alpha=0.3)

    # Fit log-normal
    mu_c = np.mean(np.log(offdiag_counts.astype(np.float64)))
    sigma_c = np.std(np.log(offdiag_counts.astype(np.float64)))
    x = np.linspace(log10_cooc.min(), log10_cooc.max(), 200)
    mu10 = mu_c / np.log(10)
    sigma10 = sigma_c / np.log(10)
    pdf = stats.norm.pdf(x, mu10, sigma10)
    ax.plot(x, pdf, "r-", linewidth=2, label=f"Log-normal\n(μ₁₀={mu10:.2f}, σ₁₀={sigma10:.2f})")
    ax.legend(fontsize=8)

    # Rank-frequency of co-occurrences (subsample for speed)
    ax = axes[0, 1]
    sorted_cooc = np.sort(offdiag_counts)[::-1]
    n_cooc = len(sorted_cooc)
    if n_cooc > 100_000:
        # Log-spaced subsample for plotting
        sample_idx = np.unique(np.geomspace(1, n_cooc, 50_000).astype(int) - 1)
        ax.loglog(sample_idx + 1, sorted_cooc[sample_idx], ".", markersize=1, alpha=0.4, color="#FF9800")
    else:
        ranks = np.arange(1, n_cooc + 1)
        ax.loglog(ranks, sorted_cooc, ".", markersize=1.5, alpha=0.5, color="#FF9800")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Co-occurrence count")
    ax.set_title("Rank-frequency (off-diagonal)")
    ax.grid(True, alpha=0.3)

    # Observed vs expected scatter (subsample)
    ax = axes[0, 2]
    ci_all = diag[rows].astype(np.float64)
    cj_all = diag[cols].astype(np.float64)
    expected_all = ci_all * cj_all / N
    if n_pairs > 100_000:
        rng = np.random.default_rng(42)
        sample = rng.choice(n_pairs, 100_000, replace=False)
    else:
        sample = np.arange(n_pairs)
    ax.scatter(
        expected_all[sample], offdiag_counts[sample],
        s=0.5, alpha=0.1, color="#FF9800", rasterized=True
    )
    lim_lo = max(0.5, min(expected_all[sample].min(), offdiag_counts[sample].min()))
    lim_hi = max(expected_all[sample].max(), offdiag_counts[sample].max()) * 1.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=0.8, label="y = x (independence)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expected count (c_i · c_j / N)")
    ax.set_ylabel("Observed co-occurrence count")
    ax.set_title("Observed vs expected")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: PMI distribution
    pmi_finite = pmi[np.isfinite(pmi)]

    ax = axes[1, 0]
    ax.hist(pmi_finite, bins=100, density=True, alpha=0.7, color="#4CAF50", edgecolor="none")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("PMI (bits)")
    ax.set_ylabel("Density")
    ax.set_title("PMI distribution (all pairs)")
    ax.grid(True, alpha=0.3)

    # PMI vs count (scatter)
    ax = axes[1, 1]
    if n_pairs > 100_000:
        sample2 = rng.choice(n_pairs, 100_000, replace=False)
    else:
        sample2 = np.arange(n_pairs)
    valid_sample = sample2[np.isfinite(pmi[sample2])]
    ax.scatter(
        offdiag_counts[valid_sample], pmi[valid_sample],
        s=0.5, alpha=0.1, color="#4CAF50", rasterized=True
    )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Co-occurrence count")
    ax.set_ylabel("PMI (bits)")
    ax.set_title("PMI vs co-occurrence count")
    ax.grid(True, alpha=0.3)

    # Obs/Expected ratio distribution
    ax = axes[1, 2]
    ratio = offdiag_counts.astype(np.float64) / np.maximum(expected_all, 1e-10)
    log_ratio = np.log2(ratio[np.isfinite(ratio) & (ratio > 0)])
    ax.hist(log_ratio, bins=100, density=True, alpha=0.7, color="#9C27B0", edgecolor="none")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", label="Obs = Expected")
    ax.set_xlabel("log₂(Observed / Expected)")
    ax.set_ylabel("Density")
    ax.set_title("Obs/Expected ratio distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Co-occurrence distribution — {label}", fontsize=14)
    fig.tight_layout()
    path = output_dir / "cooccurrence_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nSaved co-occurrence distribution plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze count and co-occurrence distributions")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing cooccurrence.npz and metadata.json",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for plots (default: inferred from bin_width)",
    )
    args = parser.parse_args()

    metadata, mat = load_data(args.results_dir)
    label = args.label or f"{metadata['bin_width']} Da bins"
    print(f"Loaded {mat.nnz:,} entries from {args.results_dir}")

    # Make symmetric for co-occurrence analysis
    mat_full = mat + mat.T - sp.diags(mat.diagonal(), dtype=mat.dtype)

    analyze_counts(mat_full, metadata, args.results_dir, label)
    analyze_cooccurrences(mat, metadata, args.results_dir, label)


if __name__ == "__main__":
    main()
