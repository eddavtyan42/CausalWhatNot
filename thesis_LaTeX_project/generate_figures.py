#!/usr/bin/env python
"""
Generate publication-quality figures for the thesis.
Figures are saved to thesis_LaTeX_project/figures/ in PDF and PNG formats.

Usage:
    cd thesis_LaTeX_project
    python generate_figures.py
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal_benchmark"))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from math import pi
import seaborn as sns
import ast

# Create output directory for figures
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

BENCHMARK_DIR = SCRIPT_DIR.parent / "causal_benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results" / "benchmark"

# Set style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Color palette (colorblind-friendly)
COLORS = {'pc': '#0072B2', 'ges': '#E69F00', 'notears': '#009E73', 'cosmo': '#CC79A7'}
ALGO_LABELS = {'pc': 'PC', 'ges': 'GES', 'notears': 'NOTEARS', 'cosmo': 'COSMO'}
ALGORITHMS = ['pc', 'ges', 'notears', 'cosmo']

# Node counts mapping for datasets
NODE_COUNTS = {
    'asia': 8,
    'sachs': 11,
    'child': 20,
    'insurance': 27,
    'alarm': 37
}


def load_benchmark_results():
    """Load the main benchmark results."""
    csv_path = RESULTS_DIR / "summary_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark results not found at {csv_path}")
    return pd.read_csv(csv_path)


def fig1_skeleton_f1_comparison(df):
    """Figure 1: Skeleton F1 by Dataset (grouped bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.2

    for i, algo in enumerate(ALGORITHMS):
        algo_data = df[df['algorithm'] == algo]
        f1_vals = [algo_data[algo_data['dataset'] == ds]['f1'].values[0] for ds in datasets]
        ax.bar(x + i*width - 1.5*width, f1_vals, width, 
               label=ALGO_LABELS[algo], color=COLORS[algo], 
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Skeleton F1 Score')
    ax.set_title('Skeleton Recovery Performance Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'skeleton_f1_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'skeleton_f1_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: skeleton_f1_comparison.pdf")


def fig2_skeleton_vs_directed_f1(df):
    """Figure 2: Skeleton F1 vs Directed F1 scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for algo in ALGORITHMS:
        algo_df = df[df['algorithm'] == algo]
        ax.scatter(algo_df['f1'], algo_df['directed_f1'], 
                   s=100, c=COLORS[algo], label=ALGO_LABELS[algo], 
                   edgecolors='black', linewidth=0.5, alpha=0.8)
        # Add dataset labels
        for _, row in algo_df.iterrows():
            ax.annotate(row['dataset'][:3].upper(), 
                       (row['f1'], row['directed_f1']),
                       fontsize=7, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect orientation')
    ax.set_xlabel('Skeleton F1 Score')
    ax.set_ylabel('Directed F1 Score')
    ax.set_title('Skeleton Recovery vs Edge Orientation Accuracy')
    ax.legend(loc='lower right')
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'skeleton_vs_directed_f1.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'skeleton_vs_directed_f1.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: skeleton_vs_directed_f1.pdf")


def fig3_runtime_comparison(df):
    """Figure 3: Runtime comparison (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.2

    for i, algo in enumerate(ALGORITHMS):
        algo_data = df[df['algorithm'] == algo]
        runtime_vals = [algo_data[algo_data['dataset'] == ds]['runtime_s'].values[0] for ds in datasets]
        ax.bar(x + i*width - 1.5*width, runtime_vals, width, 
               label=ALGO_LABELS[algo], color=COLORS[algo], 
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Runtime (seconds, log scale)')
    ax.set_title('Computational Cost Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend(loc='upper right')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'runtime_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'runtime_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: runtime_comparison.pdf")


def fig4_shd_heatmap(df):
    """Figure 4: SHD heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = df['dataset'].unique()
    shd_matrix = np.zeros((len(datasets), len(ALGORITHMS)))
    for i, ds in enumerate(datasets):
        for j, algo in enumerate(ALGORITHMS):
            shd_matrix[i, j] = df[(df['dataset'] == ds) & (df['algorithm'] == algo)]['shd'].values[0]

    im = ax.imshow(shd_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(np.arange(len(ALGORITHMS)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGORITHMS])
    ax.set_yticklabels([d.capitalize() for d in datasets])

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(ALGORITHMS)):
            text = ax.text(j, i, f'{int(shd_matrix[i, j])}',
                           ha='center', va='center', 
                           color='black' if shd_matrix[i, j] < 50 else 'white',
                           fontsize=11, fontweight='bold')

    ax.set_title('Structural Hamming Distance (SHD)')
    cbar = plt.colorbar(im, ax=ax, label='SHD')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shd_heatmap.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'shd_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: shd_heatmap.pdf")


def fig5_precision_recall_tradeoff(df):
    """Figure 5: Precision-Recall trade-off with iso-F1 curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for algo in ALGORITHMS:
        algo_df = df[df['algorithm'] == algo]
        ax.scatter(algo_df['recall'], algo_df['precision'], 
                   s=120, c=COLORS[algo], label=ALGO_LABELS[algo], 
                   edgecolors='black', linewidth=0.5, alpha=0.8, marker='o')

    # Add iso-F1 curves
    for f1 in [0.4, 0.6, 0.8]:
        recall_range = np.linspace(0.01, 1, 100)
        precision_iso = (f1 * recall_range) / (2 * recall_range - f1)
        valid = (precision_iso >= 0) & (precision_iso <= 1)
        ax.plot(recall_range[valid], precision_iso[valid], '--', 
                color='gray', alpha=0.4, linewidth=0.8)
        # Label the F1 curve
        idx = np.argmin(np.abs(recall_range - 0.9))
        if valid[idx] and precision_iso[idx] <= 1:
            ax.annotate(f'F1={f1}', (0.92, precision_iso[idx]), fontsize=8, color='gray')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off for Skeleton Recovery')
    ax.legend(loc='lower left')
    ax.set_xlim(0.25, 1.05)
    ax.set_ylim(0.35, 1.05)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'precision_recall_tradeoff.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'precision_recall_tradeoff.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: precision_recall_tradeoff.pdf")


def fig6_performance_by_datatype(df):
    """Figure 6: Algorithm performance by data type (3-panel)."""
    # Categorize datasets
    continuous_ds = ['sachs']
    discrete_small = ['asia']
    discrete_large = ['alarm', 'child', 'insurance']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Continuous
    ax = axes[0]
    cont_df = df[df['dataset'].isin(continuous_ds)]
    for i, algo in enumerate(ALGORITHMS):
        algo_data = cont_df[cont_df['algorithm'] == algo]
        if len(algo_data) > 0:
            ax.bar(i, algo_data['f1'].values[0], color=COLORS[algo], 
                   edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGORITHMS])
    ax.set_ylabel('Skeleton F1')
    ax.set_title('Continuous Data\n(Sachs)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Discrete small
    ax = axes[1]
    small_df = df[df['dataset'].isin(discrete_small)]
    for i, algo in enumerate(ALGORITHMS):
        algo_data = small_df[small_df['algorithm'] == algo]
        if len(algo_data) > 0:
            ax.bar(i, algo_data['f1'].values[0], color=COLORS[algo], 
                   edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGORITHMS])
    ax.set_title('Small Discrete\n(Asia, 8 nodes)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Discrete large (average)
    ax = axes[2]
    large_df = df[df['dataset'].isin(discrete_large)]
    avg_f1 = large_df.groupby('algorithm')['f1'].mean()
    for i, algo in enumerate(ALGORITHMS):
        if algo in avg_f1.index:
            ax.bar(i, avg_f1[algo], color=COLORS[algo], 
                   edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGORITHMS])
    ax.set_title('Large Discrete (avg)\n(Alarm, Child, Insurance)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Algorithm Performance by Data Characteristics', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'performance_by_datatype.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'performance_by_datatype.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: performance_by_datatype.pdf")


def fig7_algorithm_radar(df):
    """Figure 7: Algorithm summary radar chart."""
    metrics = ['f1', 'directed_f1', 'precision', 'recall']
    metric_labels = ['Skeleton F1', 'Directed F1', 'Precision', 'Recall']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Compute average for each algorithm
    algo_avgs = {}
    for algo in ALGORITHMS:
        algo_df = df[df['algorithm'] == algo]
        algo_avgs[algo] = {m: algo_df[m].mean() for m in metrics}
        # Add inverse of normalized runtime (so higher is better)
        max_runtime = df['runtime_s'].max()
        algo_avgs[algo]['speed'] = 1 - (algo_df['runtime_s'].mean() / max_runtime)

    metrics_ext = metrics + ['speed']
    metric_labels_ext = metric_labels + ['Speed']

    angles = [n / float(len(metrics_ext)) * 2 * pi for n in range(len(metrics_ext))]
    angles += angles[:1]

    for algo in ALGORITHMS:
        values = [algo_avgs[algo][m] for m in metrics_ext]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=ALGO_LABELS[algo], color=COLORS[algo])
        ax.fill(angles, values, alpha=0.1, color=COLORS[algo])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels_ext)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_title('Algorithm Performance Profile\n(Average Across Datasets)', y=1.08)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'algorithm_radar.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'algorithm_radar.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: algorithm_radar.pdf")


def fig8_sensitivity_ci_tests(sensitivity_df):
    """Figure 8: Conditional independence test results for mis-specification detection."""
    # Filter to CI test rows only
    ci_df = sensitivity_df[sensitivity_df['method'] == 'ci_test'].copy()
    
    if len(ci_df) == 0:
        print("No CI test results found in sensitivity data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Missing edges (should show strong dependence = high statistic)
    ax = axes[0]
    missing_df = ci_df[ci_df['edge_type'] == 'missing']
    datasets = missing_df['dataset'].unique()
    x = np.arange(len(datasets))
    
    # Use log scale for statistics since they vary greatly
    stats = [missing_df[missing_df['dataset'] == ds]['statistic'].values[0] for ds in datasets]
    colors_list = ['#d62728' if missing_df[missing_df['dataset'] == ds]['reject'].values[0] else '#1f77b4' 
                   for ds in datasets]
    
    bars = ax.bar(x, stats, color=colors_list, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel('Test Statistic (log scale)')
    ax.set_title('Missing Edge Detection\n(Red = Significant at α=0.05)')
    ax.set_yscale('log')
    ax.set_ylim(10, 1000)
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0]))
    
    # Panel B: Spurious edges (should show weak dependence = low statistic)
    ax = axes[1]
    spurious_df = ci_df[ci_df['edge_type'] == 'spurious']
    datasets = spurious_df['dataset'].unique()
    x = np.arange(len(datasets))
    
    stats = [spurious_df[spurious_df['dataset'] == ds]['statistic'].values[0] for ds in datasets]
    colors_list = ['#d62728' if spurious_df[spurious_df['dataset'] == ds]['reject'].values[0] else '#2ca02c' 
                   for ds in datasets]
    
    bars = ax.bar(x, stats, color=colors_list, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel('Test Statistic (log scale)')
    ax.set_title('Spurious Edge Detection\n(Green = Not Significant = Correctly Identified)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 1000)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0]))
    
    plt.suptitle('Conditional Independence Tests for Analyst Mis-specification', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sensitivity_ci_tests.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'sensitivity_ci_tests.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: sensitivity_ci_tests.pdf")


def fig9_sensitivity_algorithm_comparison(sensitivity_df):
    """Figure 9: Algorithm performance on mis-specified vs true graphs."""
    # Filter to algorithm rows with reference to truth
    algo_truth_df = sensitivity_df[
        (sensitivity_df['method'].isin(ALGORITHMS)) & 
        (sensitivity_df['reference'] == 'truth')
    ].copy()
    
    if len(algo_truth_df) == 0:
        print("No algorithm-vs-truth results found in sensitivity data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = algo_truth_df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, algo in enumerate(ALGORITHMS):
        algo_data = algo_truth_df[algo_truth_df['method'] == algo]
        f1_vals = []
        for ds in datasets:
            ds_data = algo_data[algo_data['dataset'] == ds]
            if len(ds_data) > 0:
                f1_vals.append(ds_data['f1'].values[0])
            else:
                f1_vals.append(0)
        ax.bar(x + i*width - 1.5*width, f1_vals, width, 
               label=ALGO_LABELS[algo], color=COLORS[algo], 
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Skeleton F1 Score (vs True Graph)')
    ax.set_title('Algorithm Performance in Sensitivity Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sensitivity_algorithm_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'sensitivity_algorithm_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: sensitivity_algorithm_comparison.pdf")


def fig10_critical_difference_diagram(df):
    """
    Figure 10: Critical Difference Diagram for algorithm rankings.
    Shows average ranks with critical difference bars from Nemenyi test.
    """
    # Calculate average ranks per algorithm across datasets
    datasets = df['dataset'].unique()
    ranks_data = []
    
    for dataset in datasets:
        ds_df = df[df['dataset'] == dataset].copy()
        # Rank algorithms by F1 (higher is better, so rank descending)
        ds_df = ds_df.sort_values('f1', ascending=False)
        ds_df['rank'] = range(1, len(ds_df) + 1)
        for _, row in ds_df.iterrows():
            ranks_data.append({
                'dataset': dataset,
                'algorithm': row['algorithm'],
                'f1': row['f1'],
                'rank': row['rank']
            })
    
    ranks_df = pd.DataFrame(ranks_data)
    
    # Calculate average rank per algorithm
    avg_ranks = ranks_df.groupby('algorithm')['rank'].mean().sort_values()
    
    # Critical difference for Nemenyi test (k=4 algorithms, n=5 datasets, alpha=0.05)
    # CD = q_alpha * sqrt(k*(k+1)/(6*n))
    # q_alpha for k=4 at alpha=0.05 is approximately 2.569
    q_alpha = 2.569
    k = len(ALGORITHMS)
    n = len(datasets)
    CD = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Draw axis
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axhline(y=0.5, color='black', linewidth=1)
    
    # Draw tick marks and labels
    for i in range(1, k + 1):
        ax.plot([i, i], [0.4, 0.6], 'k-', linewidth=1)
        ax.text(i, 0.2, str(i), ha='center', fontsize=11)
    
    ax.text((1 + k) / 2, 0.0, 'Average Rank', ha='center', fontsize=12)
    
    # Position algorithms
    y_positions = {'top': 1.0, 'bottom': -0.2}
    algo_positions = {}
    
    for i, (algo, rank) in enumerate(avg_ranks.items()):
        side = 'top' if i % 2 == 0 else 'bottom'
        y = y_positions[side]
        algo_positions[algo] = (rank, y)
        
        # Draw line to axis
        ax.plot([rank, rank], [0.5, y - 0.1 if side == 'top' else y + 0.1], 
                'k-', linewidth=1)
        
        # Draw algorithm name
        label = ALGO_LABELS.get(algo, algo)
        ax.text(rank, y, f'{label}\n({rank:.2f})', ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS.get(algo, 'white'),
                         edgecolor='black', alpha=0.8))
    
    # Draw critical difference bar (positioned at the top, but below where title will be)
    cd_y = 1.35
    ax.annotate('', xy=(1, cd_y), xytext=(1 + CD, cd_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(1 + CD/2, cd_y + 0.08, f'CD = {CD:.2f}', ha='center', fontsize=10, color='red')
    
    # Title (with extra padding)
    ax.set_title('Critical Difference Diagram (Skeleton F1)', fontsize=13, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'critical_difference.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'critical_difference.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: critical_difference.pdf")


def load_sensitivity_results():
    """Load sensitivity analysis results if available."""
    sensitivity_dir = BENCHMARK_DIR / "results" / "sensitivity_analysis"
    
    # Try multiple possible filenames
    for fname in ["phase3_results.csv", "sensitivity_analysis_results.csv", "sensitivity_results.csv", "sensitivity.csv"]:
        csv_path = sensitivity_dir / fname
        if csv_path.exists():
            return pd.read_csv(csv_path)
    
    return None


def fig11_runtime_scaling(df):
    """Figure 11: Runtime scaling with network size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate runtime by dataset and algorithm
    df_runtime = df.groupby(['dataset', 'algorithm'])['runtime_s'].mean().reset_index()
    df_runtime['nodes'] = df_runtime['dataset'].map(NODE_COUNTS)
    df_runtime = df_runtime.sort_values('nodes')
    
    markers = {'pc': 'o', 'ges': 's', 'notears': '^', 'cosmo': 'D'}
    
    for algo in ALGORITHMS:
        subset = df_runtime[df_runtime['algorithm'] == algo]
        ax.plot(subset['nodes'], subset['runtime_s'], marker=markers[algo], 
                label=ALGO_LABELS[algo], color=COLORS[algo], linewidth=2, markersize=8)
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Algorithm Scalability: Runtime vs Network Size')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'runtime_scaling.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'runtime_scaling.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: runtime_scaling.pdf")


def fig12_shd_breakdown(sensitivity_df):
    """Figure 12: SHD breakdown into error types."""
    # Filter for algorithm runs against truth
    df_algos = sensitivity_df[(sensitivity_df['reference'] == 'truth') & 
                               (sensitivity_df['method'].isin(ALGORITHMS))].copy()
    
    if len(df_algos) == 0:
        print("No algorithm-vs-truth results found for SHD breakdown")
        return
    
    # Parse lists to get counts
    def get_len(x):
        try:
            return len(ast.literal_eval(x))
        except:
            return 0
    
    df_algos['FP'] = df_algos['extra'].apply(get_len)
    df_algos['FN'] = df_algos['missing'].apply(get_len)
    df_algos['Rev'] = df_algos['reversed'].apply(get_len)
    
    # Melt for plotting
    df_melted = df_algos.melt(id_vars=['dataset', 'method'], 
                               value_vars=['FP', 'FN', 'Rev'], 
                               var_name='Error Type', value_name='Count')
    
    # Sort datasets by size
    df_melted['dataset'] = pd.Categorical(df_melted['dataset'], 
                                          categories=['asia', 'sachs', 'child', 'insurance', 'alarm'], 
                                          ordered=True)
    df_melted = df_melted.sort_values('dataset')
    
    # Use seaborn's catplot for faceted visualization
    import seaborn as sns
    g = sns.catplot(
        data=df_melted, kind="bar",
        x="dataset", y="Count", hue="Error Type", col="method",
        palette={'FP': '#d62728', 'FN': '#1f77b4', 'Rev': '#ff7f0e'},
        height=4, aspect=0.8, col_wrap=2, legend=True
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Number of Errors")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('SHD Breakdown: False Positives, False Negatives, and Reversals')
    
    plt.savefig(FIG_DIR / 'shd_breakdown.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'shd_breakdown.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: shd_breakdown.pdf")


def fig13_ci_test_strength(sensitivity_df):
    """Figure 13: CI test detection strength (log p-values)."""
    df_ci = sensitivity_df[sensitivity_df['method'] == 'ci_test'].copy()
    
    if len(df_ci) == 0:
        print("No CI test results found")
        return
    
    df_ci['log_p'] = -np.log10(df_ci['p_value'] + 1e-300)  # Avoid log(0)
    
    # Map edge types to readable names
    df_ci['Scenario'] = df_ci['edge_type'].map({
        'missing': 'Missing Edge (Target)', 
        'spurious': 'Spurious Edge (Target)'
    })
    df_ci['dataset'] = pd.Categorical(df_ci['dataset'], 
                                      categories=['asia', 'sachs', 'child', 'insurance', 'alarm'], 
                                      ordered=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    import seaborn as sns
    sns.barplot(data=df_ci, x='dataset', y='log_p', hue='Scenario', ax=ax,
                palette={'Missing Edge (Target)': '#2ca02c', 'Spurious Edge (Target)': '#d62728'})
    
    # Add threshold line
    threshold = -np.log10(0.05)
    ax.axhline(y=threshold, color='black', linestyle='--', label='p=0.05 Threshold')
    
    ax.set_ylabel(r'$-\log_{10}(p\text{-value})$')
    ax.set_xlabel('Dataset')
    ax.set_title('CI Test Sensitivity: Signal Strength for Mis-specification')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ci_test_log_p.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'ci_test_log_p.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: ci_test_log_p.pdf")


def fig14_recovery_matrix(sensitivity_df):
    """Figure 14: Target edge recovery matrix."""
    # Define target edges for each dataset
    target_edges = {
        'asia': ('Smoking', 'LungCancer'),
        'sachs': ('PKA', 'Mek'),
        'alarm': ('PVSAT', 'SAO2'),
        'child': ('Disease', 'LungParench'),
        'insurance': ('Age', 'DrivingSkill')
    }
    
    # Filter for algorithm runs against truth
    df_algos = sensitivity_df[(sensitivity_df['reference'] == 'truth') & 
                               (sensitivity_df['method'].isin(ALGORITHMS))].copy()
    
    if len(df_algos) == 0:
        print("No algorithm-vs-truth results found for recovery matrix")
        return
    
    # Initialize matrix
    datasets = ['asia', 'sachs', 'child', 'insurance', 'alarm']
    matrix = pd.DataFrame(index=datasets, columns=ALGORITHMS, dtype=int)
    
    for idx, row in df_algos.iterrows():
        ds = row['dataset']
        algo = row['method']
        
        if ds not in target_edges:
            continue
            
        target = target_edges[ds]
        
        missing_list = ast.literal_eval(row['missing'])
        reversed_list = ast.literal_eval(row['reversed'])
        
        # Check status: 0=Recovered, 1=Reversed, 2=Missing
        is_missing = any(t == target for t in missing_list)
        is_reversed = any(t == target for t in reversed_list)
        
        if is_missing:
            status = 2  # Missed
        elif is_reversed:
            status = 1  # Reversed
        else:
            status = 0  # Recovered
        
        matrix.loc[ds, algo] = status
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom colormap: 0=Green, 1=Yellow, 2=Red
    import seaborn as sns
    cmap = sns.color_palette(['#2ca02c', '#ff7f0e', '#d62728'])
    sns.heatmap(matrix, cmap=cmap, annot=False, cbar=False, 
                linewidths=1, linecolor='white', ax=ax)
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Recovered'),
        Patch(facecolor='#ff7f0e', label='Reversed'),
        Patch(facecolor='#d62728', label='Missed')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    ax.set_title('Target Edge Recovery by Algorithm')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'recovery_matrix.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'recovery_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: recovery_matrix.pdf")


def main():
    print("=" * 60)
    print("Generating thesis figures")
    print("=" * 60)
    print(f"Output directory: {FIG_DIR}")
    print()
    
    # Load benchmark results
    print("Loading benchmark results...")
    df = load_benchmark_results()
    print(f"  Loaded {len(df)} rows from benchmark results")
    print()
    
    # Generate main benchmark figures
    print("Generating benchmark figures...")
    fig1_skeleton_f1_comparison(df)
    fig2_skeleton_vs_directed_f1(df)
    fig3_runtime_comparison(df)
    fig4_shd_heatmap(df)
    fig5_precision_recall_tradeoff(df)
    fig6_performance_by_datatype(df)
    fig7_algorithm_radar(df)
    fig10_critical_difference_diagram(df)
    fig11_runtime_scaling(df)
    print()
    
    # Load and generate sensitivity analysis figures
    print("Loading sensitivity analysis results...")
    sensitivity_df = load_sensitivity_results()
    if sensitivity_df is not None:
        print(f"  Loaded {len(sensitivity_df)} rows from sensitivity results")
        fig8_sensitivity_ci_tests(sensitivity_df)
        fig9_sensitivity_algorithm_comparison(sensitivity_df)
        fig12_shd_breakdown(sensitivity_df)
        fig13_ci_test_strength(sensitivity_df)
        fig14_recovery_matrix(sensitivity_df)
    else:
        print("  Sensitivity results not found. Run sensitivity_analysis.py first.")
        print("  Skipping sensitivity figures.")
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
