"""
Geographic Visualization Module for Cultural Similarity Analysis

This module provides choropleth map visualizations for both state-level
and metro/DMA-level cultural analysis results.

Requires:
- geopandas: For geographic data handling
- matplotlib: For plotting
- Census Bureau shapefiles (downloaded automatically)
"""

import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Suppress geopandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import geopandas
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    logger.warning("geopandas not installed. Geographic maps will not be available.")


# State FIPS codes for mapping
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
    'WY': '56'
}

# Reverse mapping
FIPS_TO_STATE = {v: k for k, v in STATE_FIPS.items()}


def get_us_states_geodata() -> Optional['gpd.GeoDataFrame']:
    """
    Load US states geodata from Census Bureau.

    Returns GeoDataFrame with state boundaries, or None if geopandas unavailable.
    """
    if not HAS_GEOPANDAS:
        logger.error("geopandas required for geographic visualization")
        return None

    try:
        # Use Census Bureau's cartographic boundary files
        url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip"
        states = gpd.read_file(url)

        # Add state abbreviation column
        states['state_abbr'] = states['STATEFP'].map(FIPS_TO_STATE)

        # Filter to continental US + Alaska + Hawaii (exclude territories)
        continental = states[states['STATEFP'].astype(int) < 60].copy()

        return continental

    except Exception as e:
        logger.warning(f"Could not load Census geodata: {e}")
        logger.info("Trying alternative source...")

        try:
            # Alternative: use naturalearth data via geopandas
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            us = world[world['name'] == 'United States of America']
            return us
        except Exception as e2:
            logger.error(f"Could not load any geodata: {e2}")
            return None


def get_us_metros_geodata() -> Optional['gpd.GeoDataFrame']:
    """
    Load US metropolitan statistical areas (approximate by CBSA).

    Note: Metro boundaries are approximated using county-level data.
    For precise DMA boundaries, commercial data would be needed.
    """
    if not HAS_GEOPANDAS:
        return None

    try:
        # Use CBSA (Core Based Statistical Areas) as proxy for metros
        url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_cbsa_20m.zip"
        metros = gpd.read_file(url)
        return metros
    except Exception as e:
        logger.warning(f"Could not load metro geodata: {e}")
        return None


def plot_state_choropleth(
    data: pd.Series,
    title: str = "Cultural Analysis by State",
    cmap: str = 'RdYlBu_r',
    ax: Optional[plt.Axes] = None,
    legend_title: str = "Value",
    show_labels: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[plt.Figure]:
    """
    Create a choropleth map of US states colored by a continuous variable.

    Args:
        data: Series with state abbreviations as index and values to plot
        title: Map title
        cmap: Matplotlib colormap name
        ax: Optional axes to plot on
        legend_title: Title for the colorbar
        show_labels: Whether to show state abbreviations
        figsize: Figure size

    Returns:
        matplotlib Figure, or None if geopandas unavailable
    """
    if not HAS_GEOPANDAS:
        logger.error("geopandas required for choropleth maps")
        return None

    # Load geodata
    states_geo = get_us_states_geodata()
    if states_geo is None:
        return None

    # Merge data with geodata
    states_geo = states_geo.merge(
        data.reset_index().rename(columns={'index': 'state_abbr', data.name or 0: 'value'}),
        on='state_abbr',
        how='left'
    )

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot continental US
    continental = states_geo[~states_geo['state_abbr'].isin(['AK', 'HI'])]
    continental.plot(
        column='value',
        cmap=cmap,
        linewidth=0.8,
        ax=ax,
        edgecolor='white',
        legend=True,
        legend_kwds={'label': legend_title, 'shrink': 0.6}
    )

    # Add state labels
    if show_labels:
        for idx, row in continental.iterrows():
            if pd.notna(row['state_abbr']) and row.geometry is not None:
                centroid = row.geometry.centroid
                ax.annotate(
                    row['state_abbr'],
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='black',
                    path_effects=[]
                )

    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    return fig


def plot_state_clusters(
    cluster_labels: pd.Series,
    cluster_names: Optional[Dict[int, str]] = None,
    title: str = "Cultural Clusters by State",
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[plt.Figure]:
    """
    Create a choropleth map of US states colored by cluster membership.

    Args:
        cluster_labels: Series with state abbreviations as index and cluster IDs
        cluster_names: Optional dict mapping cluster ID to descriptive name
        title: Map title
        colors: Optional list of colors for clusters
        ax: Optional axes
        show_labels: Whether to show state abbreviations
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if not HAS_GEOPANDAS:
        logger.error("geopandas required for choropleth maps")
        return None

    states_geo = get_us_states_geodata()
    if states_geo is None:
        return None

    # Get unique clusters
    n_clusters = cluster_labels.nunique()

    # Default colors
    if colors is None:
        cmap = plt.cm.Set2
        colors = [mcolors.rgb2hex(cmap(i)) for i in range(n_clusters)]

    # Create color mapping
    unique_clusters = sorted(cluster_labels.unique())
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(unique_clusters)}

    # Map cluster labels to colors
    cluster_colors = cluster_labels.map(color_map)

    # Merge with geodata
    merge_df = pd.DataFrame({
        'state_abbr': cluster_labels.index,
        'cluster': cluster_labels.values,
        'color': cluster_colors.values
    })
    states_geo = states_geo.merge(merge_df, on='state_abbr', how='left')

    # Fill missing with gray
    states_geo['color'] = states_geo['color'].fillna('#cccccc')

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot continental US
    continental = states_geo[~states_geo['state_abbr'].isin(['AK', 'HI'])]
    continental.plot(
        color=continental['color'],
        linewidth=0.8,
        ax=ax,
        edgecolor='white'
    )

    # Add state labels
    if show_labels:
        for idx, row in continental.iterrows():
            if pd.notna(row['state_abbr']) and row.geometry is not None:
                centroid = row.geometry.centroid
                ax.annotate(
                    row['state_abbr'],
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='black'
                )

    # Create legend
    if cluster_names is None:
        cluster_names = {c: f'Cluster {c}' for c in unique_clusters}

    legend_patches = [
        mpatches.Patch(color=color_map[c], label=cluster_names.get(c, f'Cluster {c}'))
        for c in unique_clusters
    ]
    ax.legend(
        handles=legend_patches,
        loc='lower left',
        fontsize=10,
        title='Cultural Clusters'
    )

    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    return fig


def plot_state_embedding_component(
    embedding_df: pd.DataFrame,
    component: int = 0,
    component_label: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = 'RdBu_r',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[plt.Figure]:
    """
    Create a choropleth showing a single embedding component (cultural axis).

    Args:
        embedding_df: DataFrame with regions as index and PC columns
        component: Which component to plot (0-indexed)
        component_label: Label for this component (e.g., "Urban-Rural")
        title: Map title
        cmap: Colormap (diverging recommended)
        ax: Optional axes
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    col_name = f'PC{component + 1}' if f'PC{component + 1}' in embedding_df.columns else embedding_df.columns[component]
    values = embedding_df[col_name]

    if component_label is None:
        component_label = col_name

    if title is None:
        title = f"Cultural Axis: {component_label}"

    return plot_state_choropleth(
        values,
        title=title,
        cmap=cmap,
        ax=ax,
        legend_title=component_label,
        figsize=figsize
    )


def plot_dual_map(
    embedding_df: pd.DataFrame,
    clustering_labels: pd.Series,
    component: int = 0,
    component_label: str = "PC1",
    cluster_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (18, 8)
) -> Optional[plt.Figure]:
    """
    Create side-by-side maps: one showing continuous embedding, one showing clusters.

    Args:
        embedding_df: Embedding coordinates
        clustering_labels: Cluster assignments
        component: Which embedding component to show
        component_label: Label for the component
        cluster_names: Optional names for clusters
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if not HAS_GEOPANDAS:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Continuous embedding
    col_name = f'PC{component + 1}' if f'PC{component + 1}' in embedding_df.columns else embedding_df.columns[component]
    plot_state_choropleth(
        embedding_df[col_name],
        title=f"Cultural Dimension: {component_label}",
        cmap='RdBu_r',
        ax=ax1,
        legend_title=component_label
    )

    # Right: Clusters
    plot_state_clusters(
        clustering_labels,
        cluster_names=cluster_names,
        title="Cultural Clusters",
        ax=ax2
    )

    plt.tight_layout()
    return fig


def create_metro_point_map(
    embedding_df: pd.DataFrame,
    clustering_labels: pd.Series,
    metro_coords: Dict[str, Tuple[float, float]],
    title: str = "Cultural Similarity: Metro Areas",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a map with metro areas as points, colored by cluster.

    This is useful when polygon boundaries for metros aren't available.

    Args:
        embedding_df: Embedding coordinates with metro names as index
        clustering_labels: Cluster assignments
        metro_coords: Dict mapping metro name to (longitude, latitude)
        title: Map title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Try to plot US outline as background
    if HAS_GEOPANDAS:
        states_geo = get_us_states_geodata()
        if states_geo is not None:
            continental = states_geo[~states_geo['state_abbr'].isin(['AK', 'HI'])]
            continental.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)

    # Get unique clusters and colors
    n_clusters = clustering_labels.nunique()
    cmap = plt.cm.Set2
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(n_clusters)]

    # Plot each metro
    for metro in embedding_df.index:
        if metro in metro_coords and metro in clustering_labels.index:
            lon, lat = metro_coords[metro]
            cluster = clustering_labels[metro]
            color = colors[cluster % len(colors)]

            ax.scatter(lon, lat, c=color, s=100, edgecolors='white',
                      linewidth=1, zorder=5, alpha=0.8)
            ax.annotate(metro, (lon, lat), xytext=(5, 5),
                       textcoords='offset points', fontsize=7)

    # Legend
    unique_clusters = sorted(clustering_labels.unique())
    legend_patches = [
        mpatches.Patch(color=colors[c % len(colors)], label=f'Cluster {c}')
        for c in unique_clusters
    ]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=9)

    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


# Metro coordinates (longitude, latitude) for point mapping
METRO_COORDINATES = {
    'New York': (-74.006, 40.7128),
    'Los Angeles': (-118.2437, 34.0522),
    'Chicago': (-87.6298, 41.8781),
    'Houston': (-95.3698, 29.7604),
    'Phoenix': (-112.0740, 33.4484),
    'Philadelphia': (-75.1652, 39.9526),
    'San Antonio': (-98.4936, 29.4241),
    'San Diego': (-117.1611, 32.7157),
    'Dallas': (-96.7970, 32.7767),
    'San Jose': (-121.8863, 37.3382),
    'Austin': (-97.7431, 30.2672),
    'Jacksonville': (-81.6557, 30.3322),
    'San Francisco': (-122.4194, 37.7749),
    'Columbus': (-82.9988, 39.9612),
    'Indianapolis': (-86.1581, 39.7684),
    'Charlotte': (-80.8431, 35.2271),
    'Seattle': (-122.3321, 47.6062),
    'Denver': (-104.9903, 39.7392),
    'Boston': (-71.0589, 42.3601),
    'Nashville': (-86.7816, 36.1627),
    'Detroit': (-83.0458, 42.3314),
    'Portland': (-122.6765, 45.5051),
    'Las Vegas': (-115.1398, 36.1699),
    'Memphis': (-90.0490, 35.1495),
    'Louisville': (-85.7585, 38.2527),
    'Baltimore': (-76.6122, 39.2904),
    'Milwaukee': (-87.9065, 43.0389),
    'Albuquerque': (-106.6504, 35.0844),
    'Tucson': (-110.9747, 32.2226),
    'Fresno': (-119.7871, 36.7378),
    'Sacramento': (-121.4944, 38.5816),
    'Atlanta': (-84.3880, 33.7490),
    'Miami': (-80.1918, 25.7617),
    'Minneapolis': (-93.2650, 44.9778),
    'Cleveland': (-81.6944, 41.4993),
    'New Orleans': (-90.0715, 29.9511),
    'Tampa': (-82.4572, 27.9506),
    'Pittsburgh': (-79.9959, 40.4406),
    'St. Louis': (-90.1994, 38.6270),
    'Cincinnati': (-84.5120, 39.1031),
    'Orlando': (-81.3792, 28.5383),
    'Kansas City': (-94.5786, 39.0997),
    'Raleigh': (-78.6382, 35.7796),
    'Salt Lake City': (-111.8910, 40.7608),
    'Richmond': (-77.4360, 37.5407),
    'Hartford': (-72.6851, 41.7658),
    'Birmingham': (-86.8025, 33.5207),
    'Oklahoma City': (-97.5164, 35.4676),
}
