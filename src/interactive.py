"""
Interactive Cultural Analysis Module

Provides functions for running analyses with different configurations
and generating comparison visualizations.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from .config import AnalysisConfig, TOPIC_PRESETS, STATE_NAMES, default_config
from .data_collection import collect_trends_data
from .preprocessing import preprocess_data
from .similarity import compute_similarity_matrix, analyze_similarity_matrix
from .spectral_analysis import spectral_decomposition, interpret_cultural_modes
from .clustering import cluster_regions, analyze_clusters

logger = logging.getLogger(__name__)

# Color palette for clusters
CLUSTER_COLORS = [
    '#66c2a5',  # teal
    '#fc8d62',  # coral
    '#8da0cb',  # periwinkle
    '#e78ac3',  # pink
    '#a6d854',  # lime
    '#ffd92f',  # yellow
    '#e5c494',  # tan
    '#b3b3b3',  # gray
]


class CulturalAnalysisRunner:
    """
    Runner for interactive cultural analysis with configurable parameters.

    Allows easy switching between:
    - Topic presets (music, politics, culture, all)
    - Geographic units (state, metro)
    - Number of clusters
    """

    def __init__(self, base_config: AnalysisConfig = None):
        self.base_config = base_config or AnalysisConfig()
        self._cache = {}

    def run_analysis(
        self,
        topic_preset: str = 'all',
        geo_unit: str = 'state',
        n_clusters: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Run a complete analysis with specified parameters.

        Args:
            topic_preset: One of 'music', 'politics', 'culture', 'all'
            geo_unit: 'state' or 'metro'
            n_clusters: Number of clusters (None for auto-selection)
            use_cache: Whether to cache and reuse data collection results

        Returns:
            Dictionary with all analysis results
        """
        # Create config for this run
        config = replace(
            self.base_config,
            topic_preset=topic_preset,
            geo_unit=geo_unit,
            n_clusters=n_clusters
        )

        cache_key = f"{topic_preset}_{geo_unit}"

        # Collect data (potentially cached)
        if use_cache and cache_key in self._cache:
            raw_data = self._cache[cache_key]['raw_data']
            processed_data = self._cache[cache_key]['processed_data']
            similarity_matrix = self._cache[cache_key]['similarity_matrix']
            spectral_result = self._cache[cache_key]['spectral_result']
        else:
            logger.info(f"Running analysis: preset={topic_preset}, geo={geo_unit}")

            raw_data = collect_trends_data(config)
            processed_data, metadata = preprocess_data(raw_data, config)
            similarity_matrix = compute_similarity_matrix(processed_data, config=config)
            spectral_result = spectral_decomposition(similarity_matrix, config)

            if use_cache:
                self._cache[cache_key] = {
                    'raw_data': raw_data,
                    'processed_data': processed_data,
                    'similarity_matrix': similarity_matrix,
                    'spectral_result': spectral_result
                }

        # Clustering (always recomputed based on n_clusters)
        clustering_result = cluster_regions(spectral_result, similarity_matrix, config)
        cluster_analysis = analyze_clusters(clustering_result, spectral_result, processed_data)
        mode_interpretations = interpret_cultural_modes(spectral_result, processed_data, n_modes=3)

        # Generate axis labels from mode interpretations
        axis_labels = self._generate_axis_labels(mode_interpretations, processed_data)

        return {
            'config': config,
            'raw_data': raw_data,
            'processed_data': processed_data,
            'similarity_matrix': similarity_matrix,
            'spectral_result': spectral_result,
            'clustering_result': clustering_result,
            'cluster_analysis': cluster_analysis,
            'mode_interpretations': mode_interpretations,
            'axis_labels': axis_labels,
        }

    def _generate_axis_labels(
        self,
        mode_interpretations: Dict,
        data: pd.DataFrame
    ) -> List[str]:
        """
        Generate interpretable axis labels based on mode analysis.

        Examines which topics load most heavily on each mode to create
        descriptive labels like "Urban ←→ Rural" or "South ←→ North".
        """
        labels = []

        for mode_name, interp in mode_interpretations.items():
            pos_topics = interp.get('positive_topic_correlations', [])
            neg_topics = interp.get('negative_topic_correlations', [])

            high_regions = interp.get('high_loading_regions', [])
            low_regions = interp.get('low_loading_regions', [])

            # Create label based on top correlating topics
            pos_words = [t[0].split()[0] for t in pos_topics[:2]] if pos_topics else ['High']
            neg_words = [t[0].split()[0] for t in neg_topics[:2]] if neg_topics else ['Low']

            # Also include region hints
            high_region_hint = high_regions[0][0] if high_regions else ''
            low_region_hint = low_regions[0][0] if low_regions else ''

            var_pct = interp['variance_explained'] * 100

            label = f"{', '.join(neg_words)} ←→ {', '.join(pos_words)} ({var_pct:.0f}%)"
            labels.append(label)

        return labels

    def clear_cache(self):
        """Clear the data cache."""
        self._cache = {}


def create_embedding_plot_with_labels(
    results: Dict,
    title: Optional[str] = None,
    highlight_regions: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    show_all_labels: bool = False
) -> plt.Figure:
    """
    Create 2D embedding plot with interpretable axis labels.

    Args:
        results: Results dictionary from CulturalAnalysisRunner
        title: Custom title (auto-generated if None)
        highlight_regions: Specific regions to label
        figsize: Figure size
        show_all_labels: Whether to label all points

    Returns:
        matplotlib Figure
    """
    spectral_result = results['spectral_result']
    clustering_result = results['clustering_result']
    axis_labels = results['axis_labels']
    config = results['config']

    embedding = spectral_result.embedding[:, :2]
    regions = spectral_result.region_labels
    labels = clustering_result.labels
    n_clusters = clustering_result.n_clusters

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points by cluster
    for k in range(n_clusters):
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            s=120,
            alpha=0.7,
            edgecolors='white',
            linewidth=1.5,
            label=f'Cluster {k + 1}'
        )

    # Label points
    if show_all_labels:
        for i, region in enumerate(regions):
            ax.annotate(
                region,
                (embedding[i, 0], embedding[i, 1]),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    elif highlight_regions:
        for region in highlight_regions:
            if region in regions:
                i = regions.index(region)
                ax.annotate(
                    region,
                    (embedding[i, 0], embedding[i, 1]),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )

    # Axis labels with interpretation
    if len(axis_labels) >= 2:
        ax.set_xlabel(f"Cultural Axis 1: {axis_labels[0]}", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Cultural Axis 2: {axis_labels[1]}", fontsize=11, fontweight='bold')
    else:
        var1 = spectral_result.variance_explained[0] * 100
        var2 = spectral_result.variance_explained[1] * 100
        ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Legend
    ax.legend(loc='best', fontsize=10)

    # Title
    if title is None:
        preset = config.topic_preset or 'custom'
        title = f"Latent Cultural Embedding ({preset.title()} Topics, {n_clusters} Clusters)"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    return fig


def create_comparison_figure(
    results_dict: Dict[str, Dict],
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comparison figure showing multiple topic preset analyses.

    Args:
        results_dict: Dict mapping preset name to results
        figsize: Figure size

    Returns:
        matplotlib Figure with subplots for each preset
    """
    n_plots = len(results_dict)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (preset_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]

        spectral_result = results['spectral_result']
        clustering_result = results['clustering_result']
        axis_labels = results['axis_labels']

        embedding = spectral_result.embedding[:, :2]
        regions = spectral_result.region_labels
        labels = clustering_result.labels
        n_clusters = clustering_result.n_clusters

        # Plot points
        for k in range(n_clusters):
            mask = labels == k
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color,
                s=80,
                alpha=0.7,
                edgecolors='white',
                linewidth=1,
                label=f'Cluster {k + 1}'
            )

        # Label key regions
        key_regions = ['NY', 'CA', 'TX', 'FL', 'IL']
        for region in key_regions:
            if region in regions:
                i = regions.index(region)
                ax.annotate(
                    region,
                    (embedding[i, 0], embedding[i, 1]),
                    fontsize=8,
                    fontweight='bold'
                )

        # Axis labels
        if len(axis_labels) >= 2:
            ax.set_xlabel(axis_labels[0], fontsize=9)
            ax.set_ylabel(axis_labels[1], fontsize=9)

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f"{preset_name.title()} Topics", fontsize=12, fontweight='bold')

    # Hide unused axes
    for idx in range(len(results_dict), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Cultural Similarity by Topic Category", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_cluster_map_with_embedding(
    results: Dict,
    figsize: Tuple[int, int] = (18, 8)
) -> plt.Figure:
    """
    Create side-by-side: (1) Geographic map with clusters, (2) Embedding plot.

    Args:
        results: Analysis results dictionary
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Try to import geo visualization
    try:
        from .geo_visualization import plot_state_clusters, HAS_GEOPANDAS
    except ImportError:
        HAS_GEOPANDAS = False

    fig = plt.figure(figsize=figsize)

    if HAS_GEOPANDAS:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Left: Geographic map
        clustering_result = results['clustering_result']
        cluster_labels = pd.Series(
            clustering_result.labels,
            index=clustering_result.region_labels
        )
        plot_state_clusters(cluster_labels, ax=ax1, title="Cultural Clusters (Map)")

        # Right: Embedding
        spectral_result = results['spectral_result']
        embedding = spectral_result.embedding[:, :2]
        regions = spectral_result.region_labels
        labels = clustering_result.labels
        n_clusters = clustering_result.n_clusters

        for k in range(n_clusters):
            mask = labels == k
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            ax2.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, s=100, alpha=0.7, edgecolors='white',
                label=f'Cluster {k + 1}'
            )

        # Labels
        for i, region in enumerate(regions):
            ax2.annotate(region, (embedding[i, 0], embedding[i, 1]),
                        fontsize=7, alpha=0.8)

        axis_labels = results.get('axis_labels', [])
        if len(axis_labels) >= 2:
            ax2.set_xlabel(f"Axis 1: {axis_labels[0]}", fontsize=10)
            ax2.set_ylabel(f"Axis 2: {axis_labels[1]}", fontsize=10)

        ax2.legend(loc='best')
        ax2.set_title("Cultural Clusters (Embedding)", fontsize=12, fontweight='bold')
    else:
        # Just show embedding if no geopandas
        ax = fig.add_subplot(111)
        _plot_embedding_on_ax(ax, results)

    plt.tight_layout()
    return fig


def _plot_embedding_on_ax(ax: plt.Axes, results: Dict):
    """Helper to plot embedding on an axis."""
    spectral_result = results['spectral_result']
    clustering_result = results['clustering_result']
    axis_labels = results.get('axis_labels', [])

    embedding = spectral_result.embedding[:, :2]
    regions = spectral_result.region_labels
    labels = clustering_result.labels
    n_clusters = clustering_result.n_clusters

    for k in range(n_clusters):
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, s=100, alpha=0.7, edgecolors='white',
            label=f'Cluster {k + 1}'
        )

    for i, region in enumerate(regions):
        ax.annotate(region, (embedding[i, 0], embedding[i, 1]),
                   fontsize=7, alpha=0.8)

    if len(axis_labels) >= 2:
        ax.set_xlabel(f"Cultural Axis 1: {axis_labels[0]}", fontsize=10)
        ax.set_ylabel(f"Cultural Axis 2: {axis_labels[1]}", fontsize=10)

    ax.legend(loc='best')
    ax.set_title("Latent Cultural Embedding", fontsize=12, fontweight='bold')


def run_preset_comparison(
    presets: List[str] = None,
    n_clusters: int = 5,
    output_dir: Path = None
) -> Dict[str, Dict]:
    """
    Run analysis for multiple presets and generate comparison.

    Args:
        presets: List of presets to compare (default: all)
        n_clusters: Number of clusters for each analysis
        output_dir: Optional directory to save figures

    Returns:
        Dictionary mapping preset name to results
    """
    if presets is None:
        presets = ['music', 'politics', 'culture', 'all']

    runner = CulturalAnalysisRunner()
    results = {}

    for preset in presets:
        logger.info(f"Running analysis for preset: {preset}")
        results[preset] = runner.run_analysis(
            topic_preset=preset,
            n_clusters=n_clusters
        )

    # Generate comparison figure
    fig = create_comparison_figure(results)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'preset_comparison.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to {output_dir / 'preset_comparison.png'}")

    return results
