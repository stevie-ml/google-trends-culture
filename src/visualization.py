"""
Visualization Module for Cultural Similarity Analysis

This module provides publication-quality visualizations for interpreting
the results of cultural similarity analysis.

Visualizations include:
1. Scree plots - eigenvalue decay to determine number of cultural modes
2. Embedding plots - 2D/3D scatter plots of the latent cultural space
3. Similarity heatmaps - visualize pairwise regional similarity
4. Cluster maps - geographic visualization of cultural clusters
5. Topic loading plots - interpret what each cultural mode represents
"""

import logging
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

from .config import AnalysisConfig, STATE_NAMES, default_config
from .spectral_analysis import SpectralResult
from .clustering import ClusteringResult

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.Set2.colors


def plot_scree(
    spectral_result: SpectralResult,
    ax: Optional[plt.Axes] = None,
    show_kaiser: bool = True,
    show_cumulative: bool = True,
    title: str = 'Scree Plot: Eigenvalue Decay'
) -> plt.Figure:
    """
    Plot scree diagram showing eigenvalue decay.

    The scree plot is fundamental for determining the number of meaningful
    cultural dimensions. The "elbow" in the curve suggests where adding
    more components provides diminishing returns.

    Mathematical interpretation:
    - Each eigenvalue λ_k represents variance explained by mode k
    - A sharp drop after k components suggests k cultural modes
    - The Kaiser criterion (λ > 1) is shown as a reference line

    Args:
        spectral_result: Result from spectral decomposition
        ax: Optional matplotlib axes
        show_kaiser: Whether to show Kaiser criterion line
        show_cumulative: Whether to show cumulative variance

    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    eigenvalues = spectral_result.eigenvalues
    n_components = len(eigenvalues)
    x = np.arange(1, n_components + 1)

    # Plot eigenvalues
    ax.plot(x, eigenvalues, 'o-', color=COLORS[0], linewidth=2,
            markersize=8, label='Eigenvalues')

    # Mark selected components
    n_selected = spectral_result.n_components_selected
    ax.axvline(x=n_selected + 0.5, color=COLORS[2], linestyle='--',
               linewidth=2, label=f'Selected: {n_selected} components')

    # Kaiser criterion
    if show_kaiser:
        ax.axhline(y=1, color=COLORS[3], linestyle=':', linewidth=2,
                   label='Kaiser criterion (λ=1)')

    # Cumulative variance on secondary axis
    if show_cumulative:
        ax2 = ax.twinx()
        cumvar = spectral_result.cumulative_variance * 100
        ax2.plot(x, cumvar, 's-', color=COLORS[1], linewidth=2,
                 markersize=6, alpha=0.7, label='Cumulative variance')
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
        ax2.set_ylim(0, 105)

        # Add legend for secondary axis
        ax2.legend(loc='center right')

    ax.set_xlabel('Component Number', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, min(20, n_components) + 0.5)  # Show first 20 at most
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_embedding_2d(
    spectral_result: SpectralResult,
    clustering_result: Optional[ClusteringResult] = None,
    highlight_regions: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Cultural Space: 2D Embedding'
) -> plt.Figure:
    """
    Plot 2D embedding of regions in latent cultural space.

    This visualization shows regions as points in a 2D space where
    distance corresponds to cultural dissimilarity. The axes are
    the first two principal components (cultural modes).

    Mathematical interpretation:
    - PC1 (x-axis): The primary dimension of cultural variation
    - PC2 (y-axis): The secondary dimension, orthogonal to PC1
    - Euclidean distance approximates cultural dissimilarity

    Args:
        spectral_result: Spectral decomposition result
        clustering_result: Optional clustering for coloring points
        highlight_regions: Optional list of regions to label
        ax: Optional matplotlib axes
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure

    embedding = spectral_result.embedding[:, :2]
    regions = spectral_result.region_labels

    # Determine colors
    if clustering_result is not None:
        colors = [COLORS[l % len(COLORS)] for l in clustering_result.labels]
        # Create legend
        unique_labels = sorted(set(clustering_result.labels))
        legend_elements = [Patch(facecolor=COLORS[l % len(COLORS)],
                                  label=f'Cluster {l}')
                          for l in unique_labels]
    else:
        colors = [COLORS[0]] * len(regions)
        legend_elements = None

    # Scatter plot
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=colors, s=100, alpha=0.7, edgecolors='white',
                        linewidth=1.5)

    # Label points
    if highlight_regions is None:
        # Label all points
        for i, region in enumerate(regions):
            ax.annotate(region, (embedding[i, 0], embedding[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    else:
        # Only label highlighted regions
        for region in highlight_regions:
            if region in regions:
                i = regions.index(region)
                ax.annotate(region, (embedding[i, 0], embedding[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='yellow', alpha=0.7))

    # Axis labels with variance explained
    var1 = spectral_result.variance_explained[0] * 100
    var2 = spectral_result.variance_explained[1] * 100
    ax.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if legend_elements:
        ax.legend(handles=legend_elements, loc='best')

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    return fig


def plot_embedding_3d(
    spectral_result: SpectralResult,
    clustering_result: Optional[ClusteringResult] = None,
    highlight_regions: Optional[List[str]] = None,
    title: str = 'Cultural Space: 3D Embedding'
) -> plt.Figure:
    """
    Plot 3D embedding of regions in latent cultural space.

    Extends the 2D visualization to include the third principal component,
    providing a richer view of the cultural space.

    Args:
        spectral_result: Spectral decomposition result
        clustering_result: Optional clustering for coloring
        highlight_regions: Optional regions to label
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    embedding = spectral_result.embedding[:, :3]
    regions = spectral_result.region_labels

    # Colors
    if clustering_result is not None:
        colors = [COLORS[l % len(COLORS)] for l in clustering_result.labels]
    else:
        colors = [COLORS[0]] * len(regions)

    # 3D scatter
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               c=colors, s=80, alpha=0.7, edgecolors='white')

    # Labels for highlighted regions
    if highlight_regions:
        for region in highlight_regions:
            if region in regions:
                i = regions.index(region)
                ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2],
                       region, fontsize=9, fontweight='bold')

    # Axis labels
    var = spectral_result.variance_explained * 100
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=10)
    ax.set_zlabel(f'PC3 ({var[2]:.1f}%)', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return fig


def plot_similarity_heatmap(
    similarity_matrix: pd.DataFrame,
    clustering_result: Optional[ClusteringResult] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Regional Cultural Similarity Matrix',
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Plot heatmap of the similarity matrix.

    This visualization shows pairwise cultural similarity between all
    regions. Red indicates high similarity, blue indicates low similarity.

    If clustering is provided, regions are reordered to show cluster
    structure (regions in the same cluster appear together).

    Args:
        similarity_matrix: The n×n similarity matrix
        clustering_result: Optional clustering for reordering
        ax: Optional matplotlib axes
        title: Plot title
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    S = similarity_matrix.copy()

    # Reorder by cluster if clustering provided
    if clustering_result is not None:
        # Sort regions by cluster label
        order = np.argsort(clustering_result.labels)
        regions_ordered = [clustering_result.region_labels[i] for i in order]
        S = S.loc[regions_ordered, regions_ordered]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 12))
    else:
        fig = ax.figure

    # Heatmap
    sns.heatmap(S, ax=ax, cmap=cmap, center=0,
                square=True, linewidths=0.5,
                cbar_kws={'label': 'Similarity (correlation)'})

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    return fig


def plot_cluster_comparison(
    regions_to_compare: List[str],
    similarity_matrix: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = 'Cultural Distance Comparison'
) -> plt.Figure:
    """
    Plot comparison of cultural distances between specific regions.

    Useful for answering questions like: "Is Miami closer to NYC or Atlanta?"

    Args:
        regions_to_compare: List of regions to compare
        similarity_matrix: The similarity matrix
        ax: Optional matplotlib axes
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    subset = similarity_matrix.loc[regions_to_compare, regions_to_compare]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sns.heatmap(subset, ax=ax, cmap='RdBu_r', center=0,
                annot=True, fmt='.2f', square=True,
                cbar_kws={'label': 'Similarity'})

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_mode_interpretation(
    spectral_result: SpectralResult,
    data: pd.DataFrame,
    mode_index: int = 0,
    n_regions: int = 10,
    n_topics: int = 8,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot interpretation of a cultural mode.

    Shows:
    - Left: Regions with highest/lowest loadings on this mode
    - Right: Topics most correlated with this mode

    This helps answer: "What does Cultural Mode 1 represent?"

    Args:
        spectral_result: Spectral decomposition result
        data: Original normalized data matrix
        mode_index: Which mode to interpret (0-indexed)
        n_regions: Number of top/bottom regions to show
        n_topics: Number of top topics to show
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    loadings = spectral_result.eigenvectors[:, mode_index]
    regions = spectral_result.region_labels
    var_explained = spectral_result.variance_explained[mode_index] * 100

    if title is None:
        title = f'Cultural Mode {mode_index + 1} ({var_explained:.1f}% variance)'

    # Sort regions by loading
    sorted_idx = np.argsort(loadings)

    # Top and bottom regions
    top_regions = [(regions[i], loadings[i]) for i in sorted_idx[-n_regions//2:][::-1]]
    bottom_regions = [(regions[i], loadings[i]) for i in sorted_idx[:n_regions//2]]

    # Combine for plotting
    display_regions = bottom_regions + top_regions
    region_names = [r[0] for r in display_regions]
    region_loadings = [r[1] for r in display_regions]

    # Bar plot of region loadings
    colors = ['#e74c3c' if l < 0 else '#3498db' for l in region_loadings]
    y_pos = np.arange(len(display_regions))
    ax1.barh(y_pos, region_loadings, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(region_names)
    ax1.set_xlabel('Loading on Mode', fontsize=11)
    ax1.set_title('Region Loadings', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)

    # Topic correlations with this mode
    topics = data.columns.tolist()
    topic_correlations = []
    for j, topic in enumerate(topics):
        corr = np.corrcoef(loadings, data.iloc[:, j].values)[0, 1]
        topic_correlations.append((topic, corr))

    topic_correlations.sort(key=lambda x: x[1])

    # Top positive and negative
    display_topics = topic_correlations[:n_topics//2] + topic_correlations[-n_topics//2:]
    topic_names = [t[0] for t in display_topics]
    topic_corrs = [t[1] for t in display_topics]

    # Bar plot of topic correlations
    colors = ['#e74c3c' if c < 0 else '#3498db' for c in topic_corrs]
    y_pos = np.arange(len(display_topics))
    ax2.barh(y_pos, topic_corrs, color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(topic_names)
    ax2.set_xlabel('Correlation with Mode', fontsize=11)
    ax2.set_title('Associated Topics', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.5)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_cluster_profiles(
    clustering_result: ClusteringResult,
    data: pd.DataFrame,
    n_topics: int = 10,
    title: str = 'Cluster Cultural Profiles'
) -> plt.Figure:
    """
    Plot the characteristic topic profiles for each cluster.

    Shows the mean search interest by topic for each cluster,
    highlighting what makes each cultural group distinctive.

    Args:
        clustering_result: Clustering result
        data: Normalized data matrix
        n_topics: Number of most variable topics to show
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    n_clusters = clustering_result.n_clusters

    # Find most variable topics across clusters
    cluster_means = []
    for k in range(n_clusters):
        members = clustering_result.get_cluster_members(k)
        cluster_means.append(data.loc[members].mean(axis=0))

    cluster_means_df = pd.DataFrame(cluster_means)
    topic_variance = cluster_means_df.var(axis=0)
    top_topics = topic_variance.nlargest(n_topics).index.tolist()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(top_topics))
    width = 0.8 / n_clusters

    for k in range(n_clusters):
        members = clustering_result.get_cluster_members(k)
        means = data.loc[members, top_topics].mean(axis=0)
        offset = (k - n_clusters/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=f'Cluster {k}',
               color=COLORS[k % len(COLORS)], alpha=0.8)

    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Mean Normalized Interest', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_topics, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linewidth=0.5)

    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path,
    dpi: int = 150,
    format: str = 'png'
) -> Path:
    """
    Save figure to file.

    Args:
        fig: matplotlib Figure
        filename: Base filename (without extension)
        output_dir: Output directory
        dpi: Resolution
        format: File format

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f'{filename}.{format}'
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved figure to {filepath}")

    return filepath


def create_all_visualizations(
    similarity_matrix: pd.DataFrame,
    spectral_result: SpectralResult,
    clustering_result: ClusteringResult,
    data: pd.DataFrame,
    config: AnalysisConfig = default_config,
    highlight_regions: Optional[List[str]] = None
) -> Dict[str, plt.Figure]:
    """
    Create all standard visualizations.

    Convenience function to generate the complete set of figures
    for a cultural similarity analysis.

    Args:
        similarity_matrix: The similarity matrix
        spectral_result: Spectral decomposition result
        clustering_result: Clustering result
        data: Original normalized data
        config: Configuration
        highlight_regions: Regions to highlight in embeddings

    Returns:
        Dictionary mapping figure names to Figure objects
    """
    figures = {}

    # 1. Scree plot
    figures['scree_plot'] = plot_scree(spectral_result)

    # 2. 2D embedding
    figures['embedding_2d'] = plot_embedding_2d(
        spectral_result, clustering_result, highlight_regions
    )

    # 3. 3D embedding
    figures['embedding_3d'] = plot_embedding_3d(
        spectral_result, clustering_result, highlight_regions
    )

    # 4. Similarity heatmap
    figures['similarity_heatmap'] = plot_similarity_heatmap(
        similarity_matrix, clustering_result
    )

    # 5. Mode interpretations (first 3 modes)
    for i in range(min(3, spectral_result.n_components_selected)):
        figures[f'mode_{i+1}_interpretation'] = plot_mode_interpretation(
            spectral_result, data, mode_index=i
        )

    # 6. Cluster profiles
    figures['cluster_profiles'] = plot_cluster_profiles(
        clustering_result, data
    )

    return figures


def save_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: Path,
    config: AnalysisConfig = default_config
) -> List[Path]:
    """
    Save all figures to disk.

    Args:
        figures: Dictionary of figure name -> Figure
        output_dir: Output directory
        config: Configuration

    Returns:
        List of paths to saved files
    """
    paths = []
    for name, fig in figures.items():
        path = save_figure(
            fig, name, output_dir,
            dpi=config.figure_dpi,
            format=config.figure_format
        )
        paths.append(path)
        plt.close(fig)

    return paths
