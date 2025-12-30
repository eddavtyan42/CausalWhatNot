#!/usr/bin/env python3
"""
Generate methodology figures for thesis: Asia network DAG and benchmarking pipeline.
These figures are used in the Methods section to illustrate the experimental setup.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_asia_network():
    """
    Generate the Asia network DAG visualization.
    
    The Asia network (also known as the "Chest Clinic" network) contains 8 nodes:
    - VisitAsia: Whether the patient visited Asia
    - Smoking: Whether the patient smokes
    - Tuberculosis: Whether the patient has tuberculosis
    - LungCancer: Whether the patient has lung cancer
    - Bronchitis: Whether the patient has bronchitis
    - Either: Either tuberculosis or lung cancer (or-node)
    - Xray: X-ray result
    - Dyspnea: Whether the patient has difficulty breathing
    
    Edges (8 total):
    VisitAsia -> Tuberculosis
    Smoking -> LungCancer
    Smoking -> Bronchitis
    Tuberculosis -> Either
    LungCancer -> Either
    Either -> Xray
    Either -> Dyspnea
    Bronchitis -> Dyspnea
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Node positions (x, y) - arranged in layers for clarity
    # Layer 1 (top): Root causes
    # Layer 2: Intermediate diseases
    # Layer 3: Deterministic node
    # Layer 4 (bottom): Observable symptoms
    
    # Adjusted positions for better arrow aesthetics and spacing
    nodes = {
        'VisitAsia':     (1.5, 7.0),
        'Smoking':       (7.5, 7.0),
        'Tuberculosis':  (1.5, 5.0),
        'LungCancer':    (5.0, 5.0),
        'Bronchitis':    (8.5, 5.0),
        'Either':        (3.5, 3.0),
        'Xray':          (2.0, 1.0),
        'Dyspnea':       (6.0, 1.0),
    }
    
    # Short labels for display
    labels = {
        'VisitAsia': 'Visit to\nAsia',
        'Smoking': 'Smoking',
        'Tuberculosis': 'Tuberculosis',
        'LungCancer': 'Lung\nCancer',
        'Bronchitis': 'Bronchitis',
        'Either': 'Either',
        'Xray': 'X-ray\nResult',
        'Dyspnea': 'Dyspnea',
    }
    
    # Edges as (source, target) pairs
    edges = [
        ('VisitAsia', 'Tuberculosis'),
        ('Smoking', 'LungCancer'),
        ('Smoking', 'Bronchitis'),
        ('Tuberculosis', 'Either'),
        ('LungCancer', 'Either'),
        ('Either', 'Xray'),
        ('Either', 'Dyspnea'),
        ('Bronchitis', 'Dyspnea'),
    ]
    
    # Node styling
    node_width = 1.4
    node_height = 0.8
    node_color = '#E3F2FD'  # Light blue
    edge_color = '#1565C0'  # Dark blue
    deterministic_color = '#FFF3E0'  # Light orange for "Either" (or-node)
    
    # Draw nodes as rounded rectangles
    for name, (x, y) in nodes.items():
        # Use different color for the deterministic "Either" node
        color = deterministic_color if name == 'Either' else node_color
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x - node_width/2, y - node_height/2),
            node_width, node_height,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color,
            edgecolor='#333333',
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y, labels[name], ha='center', va='center',
                fontsize=9, fontweight='bold', color='#333333')
    
    # Draw edges as arrows
    def draw_arrow(src_name, tgt_name):
        sx, sy = nodes[src_name]
        tx, ty = nodes[tgt_name]
        
        # Calculate direction
        dx = tx - sx
        dy = ty - sy
        dist = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction
        dx_norm = dx / dist
        dy_norm = dy / dist
        
        # Offset start and end points to edge of nodes
        # Account for node size
        start_offset = 0.5  # roughly half the node size
        end_offset = 0.5
        
        start_x = sx + dx_norm * start_offset
        start_y = sy + dy_norm * start_offset
        end_x = tx - dx_norm * end_offset
        end_y = ty - dy_norm * end_offset
        
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle=ArrowStyle('->', head_length=8, head_width=5),
            color='#333333',
            linewidth=1.5,
            connectionstyle='arc3,rad=0.0',
            mutation_scale=1.0
        )
        ax.add_patch(arrow)
    
    for src, tgt in edges:
        draw_arrow(src, tgt)
    
    # Add title
    ax.set_title('Asia (Chest Clinic) Network', fontsize=14, fontweight='bold', pad=10)
    
    # Add legend for node types
    legend_elements = [
        mpatches.Patch(facecolor=node_color, edgecolor='#333333',
                       label='Random Variable'),
        mpatches.Patch(facecolor=deterministic_color, edgecolor='#333333',
                       label='Deterministic (OR-node)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    # Add annotation about the network
    ax.text(0.5, -0.05, '8 nodes, 8 edges • Binary variables',
            transform=ax.transAxes, ha='center', fontsize=10,
            style='italic', color='#666666')
    
    # Save figures
    plt.savefig(os.path.join(FIGURES_DIR, 'asia_network.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'asia_network.png'), format='png', dpi=300)
    plt.close()
    
    print("Generated: asia_network.pdf and asia_network.png")


def generate_pipeline_figure():
    """
    Generate the benchmarking pipeline flowchart.
    
    The pipeline consists of 6 main stages:
    1. Benchmark Networks - Load reference DAGs (Asia, Sachs, Alarm, Child, Insurance)
    2. Data Generation - Sample observational data from networks
    3. Causal Discovery - Run algorithms (PC, GES, NOTEARS, COSMO)
    4. Learned Graph - Output DAG from each algorithm
    5. Evaluation Metrics - Compare learned vs true graphs
    6. Sensitivity Analysis - Test mis-specification detection
    7. Results - Final benchmark tables and figures
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define box positions and sizes
    box_height = 1.2
    box_width = 1.6
    y_center = 3.0
    
    # Main pipeline stages
    stages = [
        {'x': 1.0, 'label': 'Benchmark\nNetworks', 'color': '#E8F5E9',
         'sublabel': 'Asia, Sachs,\nAlarm, Child,\nInsurance'},
        {'x': 3.2, 'label': 'Data\nGeneration', 'color': '#E3F2FD',
         'sublabel': 'n samples\nper network'},
        {'x': 5.4, 'label': 'Causal\nDiscovery', 'color': '#FFF3E0',
         'sublabel': 'PC, GES,\nNOTEARS,\nCOSMO'},
        {'x': 7.6, 'label': 'Learned\nGraph', 'color': '#F3E5F5',
         'sublabel': 'Estimated\nDAG'},
        {'x': 9.8, 'label': 'Evaluation\nMetrics', 'color': '#E0F7FA',
         'sublabel': 'F₁, SHD,\nPrecision,\nRecall'},
        {'x': 12.0, 'label': 'Results', 'color': '#FFEBEE',
         'sublabel': 'Tables &\nFigures'},
    ]
    
    # Draw main pipeline boxes
    for stage in stages:
        x = stage['x']
        
        # Main box
        rect = FancyBboxPatch(
            (x - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=stage['color'],
            edgecolor='#333333',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Main label
        ax.text(x, y_center + 0.1, stage['label'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='#333333')
        
        # Sublabel (below main label, larger for better readability)
        ax.text(x, y_center - 0.5, stage['sublabel'], ha='center', va='top',
                fontsize=9, color='#555555', style='italic')
    
    # Draw arrows between main stages
    arrow_style = ArrowStyle('->', head_length=10, head_width=6)
    for i in range(len(stages) - 1):
        x1 = stages[i]['x'] + box_width/2 + 0.05
        x2 = stages[i+1]['x'] - box_width/2 - 0.05
        
        arrow = FancyArrowPatch(
            (x1, y_center), (x2, y_center),
            arrowstyle=arrow_style,
            color='#333333',
            linewidth=2,
            mutation_scale=1.0
        )
        ax.add_patch(arrow)
    
    # Add Sensitivity Analysis branch
    sensitivity_x = 9.8
    sensitivity_y = 1.0
    sens_width = 2.0
    sens_height = 0.9
    
    # Sensitivity box
    sens_rect = FancyBboxPatch(
        (sensitivity_x - sens_width/2, sensitivity_y - sens_height/2),
        sens_width, sens_height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#FFF8E1',
        edgecolor='#333333',
        linewidth=2
    )
    ax.add_patch(sens_rect)
    ax.text(sensitivity_x, sensitivity_y, 'Sensitivity Analysis\n(Mis-specification)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')
    
    # Arrow from Learned Graph to Sensitivity
    arrow1 = FancyArrowPatch(
        (7.6, y_center - box_height/2 - 0.1), (sensitivity_x - 0.3, sensitivity_y + sens_height/2 + 0.1),
        arrowstyle=arrow_style,
        color='#666666',
        linewidth=1.5,
        connectionstyle='arc3,rad=-0.2',
        mutation_scale=1.0
    )
    ax.add_patch(arrow1)
    
    # Arrow from Sensitivity to Results
    arrow2 = FancyArrowPatch(
        (sensitivity_x + sens_width/2 + 0.1, sensitivity_y + 0.2),
        (12.0 - box_width/2 - 0.1, y_center - box_height/2 - 0.1),
        arrowstyle=arrow_style,
        color='#666666',
        linewidth=1.5,
        connectionstyle='arc3,rad=0.2',
        mutation_scale=1.0
    )
    ax.add_patch(arrow2)
    
    # Add True Graph reference (feeds into Evaluation)
    true_graph_x = 7.6
    true_graph_y = 5.2
    
    true_rect = FancyBboxPatch(
        (true_graph_x - 0.9, true_graph_y - 0.4),
        1.8, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor='#C8E6C9',
        edgecolor='#388E3C',
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(true_rect)
    ax.text(true_graph_x, true_graph_y, 'True Graph\n(Ground Truth)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#2E7D32')
    
    # Arrow from True Graph to Evaluation
    arrow3 = FancyArrowPatch(
        (true_graph_x + 0.9, true_graph_y - 0.4),
        (9.8 - 0.5, y_center + box_height/2 + 0.1),
        arrowstyle=arrow_style,
        color='#388E3C',
        linewidth=1.5,
        connectionstyle='arc3,rad=0.15',
        mutation_scale=1.0,
        linestyle='--'
    )
    ax.add_patch(arrow3)
    
    # Add Bootstrap annotation
    bootstrap_x = 5.4
    bootstrap_y = 5.2
    
    bootstrap_rect = FancyBboxPatch(
        (bootstrap_x - 1.0, bootstrap_y - 0.35),
        2.0, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor='#BBDEFB',
        edgecolor='#1976D2',
        linewidth=1.5,
        linestyle='--'
    )
    ax.add_patch(bootstrap_rect)
    ax.text(bootstrap_x, bootstrap_y, 'Bootstrap × k runs',
            ha='center', va='center', fontsize=10, color='#1565C0', style='italic')
    
    # Curved arrow back from Learned Graph to Data Generation (bootstrap loop)
    arrow4 = FancyArrowPatch(
        (7.6, y_center + box_height/2 + 0.1),
        (3.2, y_center + box_height/2 + 0.1),
        arrowstyle=arrow_style,
        color='#1976D2',
        linewidth=1.5,
        connectionstyle='arc3,rad=-0.4',
        mutation_scale=1.0,
        linestyle='--'
    )
    ax.add_patch(arrow4)
    
    # Add title
    ax.set_title('Causal Discovery Benchmarking Pipeline', fontsize=14, fontweight='bold', y=0.98)
    
    # Add caption annotation (larger for readability)
    ax.text(0.5, -0.02, 
            'Solid arrows: main pipeline flow • Dashed arrows: validation loops and comparisons',
            transform=ax.transAxes, ha='center', fontsize=10,
            style='italic', color='#555555')
    
    # Save figures
    plt.savefig(os.path.join(FIGURES_DIR, 'pipeline.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'pipeline.png'), format='png', dpi=300)
    plt.close()
    
    print("Generated: pipeline.pdf and pipeline.png")


if __name__ == '__main__':
    print("Generating methodology figures for thesis...")
    print(f"Output directory: {FIGURES_DIR}")
    print("-" * 50)
    
    generate_asia_network()
    generate_pipeline_figure()
    
    print("-" * 50)
    print("All methodology figures generated successfully!")
    print("\nFiles created:")
    print("  - figures/asia_network.pdf")
    print("  - figures/asia_network.png")
    print("  - figures/pipeline.pdf")
    print("  - figures/pipeline.png")
