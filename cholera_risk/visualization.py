"""
Cholera Risk Assessment - Visualization Module

Provides plotting functions for visualizing Monte Carlo simulation results.
All functions accept the results dictionary from CholeraSimulator.run_simulation().

This module is intentionally kept separate from the model so the simulation
can run "headless" (e.g., on a server) without importing matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
from typing import Dict, Any, Optional, Tuple


# ─── Style Configuration ───
STYLE_CONFIG = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafbfc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "text.color": "#0f172a",
    "grid.color": "#e2e8f0",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
}


def apply_style():
    """Apply the default plotting style for cholera risk visualizations."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_style("whitegrid", {
        "axes.facecolor": "#fafbfc",
        "grid.color": "#e2e8f0",
    })


def plot_distribution(results: Dict[str, Any], 
                      bins: int = 50,
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[str] = None,
                      show: bool = True) -> plt.Figure:
    """Plot the distribution of simulated risk probabilities.
    
    Creates a histogram with KDE overlay, plus vertical lines for 
    mean, min, and max values.
    
    Args:
        results: Output from CholeraSimulator.run_simulation().
        bins: Number of histogram bins.
        figsize: Figure dimensions (width, height).
        save_path: If provided, save the figure to this file path.
        show: Whether to display the figure interactively.
        
    Returns:
        The matplotlib Figure object.
    """
    apply_style()
    
    samples = results["samples"]
    summary = results["summary"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram + KDE
    sns.histplot(samples, bins=bins, kde=True, color="#3b82f6", 
                 alpha=0.55, edgecolor="#2563eb", linewidth=0.5, ax=ax)
    
    # Mean line (RED - prominent)
    ax.axvline(summary["expected_prob"], color="#dc2626", linestyle="--", 
               linewidth=2.5, label=f'Mean = {summary["expected_prob"]:.2e}', zorder=5)
    
    # Min line
    ax.axvline(summary["min"], color="#059669", linestyle="--", 
               linewidth=1.5, label=f'Min = {summary["min"]:.2e}', alpha=0.7)
    
    # Max line
    ax.axvline(summary["max"], color="#7c3aed", linestyle="--", 
               linewidth=1.5, label=f'Max = {summary["max"]:.2e}', alpha=0.7)
    
    ax.set_xlabel("Probability of Cholera Infection (Transboundary Risk)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Simulated Risk Probabilities", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig


def plot_sensitivity(results: Dict[str, Any],
                     figsize: Tuple[int, int] = (10, 5),
                     save_path: Optional[str] = None,
                     show: bool = True) -> plt.Figure:
    """Plot sensitivity analysis as a horizontal bar chart (tornado diagram).
    
    Shows Spearman rank correlation coefficients for each stochastic input.
    
    Args:
        results: Output from CholeraSimulator.run_simulation().
        figsize: Figure dimensions (width, height).
        save_path: If provided, save the figure to this file path.
        show: Whether to display the figure interactively.
        
    Returns:
        The matplotlib Figure object.
    """
    apply_style()
    
    sensitivity = results["sensitivity"]
    
    # Extract names and rho values
    names = list(sensitivity.keys())
    rhos = [sensitivity[n]["rho"] for n in names]
    
    # Create DataFrame and sort
    df = pd.DataFrame({"Factor": names, "Spearman ρ": rhos})
    df = df.sort_values("Spearman ρ", ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color based on positive/negative correlation
    colors = ["#3b82f6" if v > 0 else "#e11d48" for v in df["Spearman ρ"]]
    
    bars = ax.barh(df["Factor"], df["Spearman ρ"], color=colors, 
                   edgecolor=[c.replace("0.5", "0.8") for c in colors], height=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, df["Spearman ρ"]):
        width = bar.get_width()
        x_pos = width + 0.02 * np.sign(width) if width != 0 else 0.02
        ha = "left" if width > 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Spearman Rank Correlation Coefficient", fontsize=12)
    ax.set_title("Sensitivity Analysis", fontsize=14, fontweight="bold")
    ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="-")
    ax.set_xlim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig


def plot_scatter(results: Dict[str, Any],
                 x_factor: str = "P2",
                 max_points: int = 500,
                 figsize: Tuple[int, int] = (8, 6),
                 save_path: Optional[str] = None,
                 show: bool = True) -> plt.Figure:
    """Plot a scatter plot of an input factor vs. final risk.
    
    Args:
        results: Output from CholeraSimulator.run_simulation().
        x_factor: Which input factor to plot on x-axis ('P2', 'P3', or 'P4').
        max_points: Maximum number of points to plot (subsampled if needed).
        figsize: Figure dimensions (width, height).
        save_path: If provided, save the figure to this file path.
        show: Whether to display the figure interactively.
        
    Returns:
        The matplotlib Figure object.
    """
    apply_style()
    
    samples = results["samples"]
    x_data = results["inputs"][x_factor]
    
    # Subsample if too many points
    n = len(samples)
    if n > max_points:
        step = n // max_points
        indices = np.arange(0, n, step)[:max_points]
        x_data = x_data[indices]
        samples = samples[indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x_data, samples, alpha=0.35, s=12, color="#059669", edgecolors="#047857")
    
    factor_labels = {
        "P2": "Infection Rate (Beta Distribution)",
        "P3": "False Negative Rate (Source Region)",
        "P4": "False Negative Rate (Target Region)",
    }
    
    ax.set_xlabel(f"{x_factor} — {factor_labels.get(x_factor, x_factor)}", fontsize=12)
    ax.set_ylabel("Final Risk Probability", fontsize=12)
    ax.set_title(f"{x_factor} vs Final Risk", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig


def plot_all(results: Dict[str, Any],
             save_dir: Optional[str] = None,
             show: bool = True) -> Dict[str, plt.Figure]:
    """Generate all standard plots for a simulation run.
    
    Creates:
        1. Risk distribution histogram
        2. Sensitivity tornado chart
        3. P2 vs Final Risk scatter plot
    
    Args:
        results: Output from CholeraSimulator.run_simulation().
        save_dir: If provided, save all plots to this directory.
        show: Whether to display plots interactively.
        
    Returns:
        Dictionary mapping plot names to Figure objects.
    """
    import os
    
    figures = {}
    
    save_path = lambda name: os.path.join(save_dir, name) if save_dir else None
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    figures["distribution"] = plot_distribution(
        results, save_path=save_path("distribution.png"), show=show
    )
    figures["sensitivity"] = plot_sensitivity(
        results, save_path=save_path("sensitivity.png"), show=show
    )
    figures["scatter_p2"] = plot_scatter(
        results, x_factor="P2", save_path=save_path("scatter_p2.png"), show=show
    )
    
    return figures


def print_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of simulation results to the console.
    
    Args:
        results: Output from CholeraSimulator.run_simulation().
    """
    s = results["summary"]
    cfg = results["config"]
    
    print("=" * 65)
    print("   CHOLERA RISK ASSESSMENT — Monte Carlo Simulation Results")
    print("=" * 65)
    print(f"  Samples:              {cfg.n_samples:,}")
    print(f"  Random Seed:          {cfg.seed}")
    print(f"  λ (Poisson rate):     {results['lambda_rate']:.4f}")
    print("-" * 65)
    print(f"  Expected Probability: {s['expected_prob']:.6e}")
    print(f"  Median:               {s['median']:.6e}")
    print(f"  Std Deviation:        {s['std_dev']:.6e}")
    print(f"  90% CI:               [{s['ci_5']:.2e}, {s['ci_95']:.2e}]")
    print(f"  Minimum:              {s['min']:.2e}")
    print(f"  Maximum:              {s['max']:.2e}")
    print("-" * 65)
    print("  Input Parameter Means:")
    for key, val in results["input_means"].items():
        print(f"    {key}: {val:.6e}")
    print("-" * 65)
    print("  Sensitivity Analysis (Spearman ρ):")
    for name, data in results["sensitivity"].items():
        print(f"    {name}: ρ = {data['rho']:.4f}  (p = {data['p_value']:.2e})")
    print("=" * 65)
