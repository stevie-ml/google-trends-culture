"""
Similarity Matrix Computation for Cultural Analysis

This module computes pairwise similarity between regions based on their
search behavior profiles.

Mathematical Background:
------------------------
Given the normalized region × topic matrix X ∈ ℝ^(n×p), we compute
a similarity matrix S ∈ ℝ^(n×n) where S[i,j] measures how "culturally
similar" regions i and j are.

Three similarity metrics are supported:

1. Pearson Correlation:
   corr(x_i, x_j) = Σ(x_i - μ_i)(x_j - μ_j) / (σ_i * σ_j * (p-1))

   Range: [-1, 1]
   Interpretation: Captures whether regions have similar *patterns* of
   high/low search interest across topics, regardless of magnitude.
   Value of 1 means perfectly correlated profiles.
   Value of 0 means no linear relationship.
   Value of -1 means inversely correlated profiles.

2. Cosine Similarity:
   cos(x_i, x_j) = (x_i · x_j) / (||x_i|| * ||x_j||)

   Range: [-1, 1] for centered data, [0, 1] for non-negative data
   Interpretation: Measures the angle between region vectors in topic space.
   Similar to correlation but doesn't subtract means.

3. Negative Euclidean Distance:
   -||x_i - x_j||_2

   Range: (-∞, 0]
   Interpretation: Smaller distance (closer to 0) means more similar.
   Unlike correlation/cosine, this is sensitive to magnitude differences.

For cultural analysis, Pearson correlation is typically preferred because:
- It captures similarity in *preferences* not absolute levels
- It's invariant to the mean search interest of a region
- The resulting matrix can be directly interpreted as a covariance
  structure suitable for factor analysis

The similarity matrix S is symmetric and positive semi-definite (for
correlation and cosine), making it suitable for spectral decomposition.
"""

import logging
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
from scipy import spatial
from scipy.stats import pearsonr, spearmanr

from .config import AnalysisConfig, default_config

logger = logging.getLogger(__name__)


def compute_similarity_matrix(
    data: pd.DataFrame,
    metric: str = 'correlation',
    config: AnalysisConfig = None
) -> pd.DataFrame:
    """
    Compute pairwise similarity matrix between regions.

    Args:
        data: Normalized region × topic DataFrame
        metric: Similarity metric ('correlation', 'cosine', 'euclidean')
        config: Optional configuration (metric taken from config if provided)

    Returns:
        Symmetric similarity matrix as DataFrame with regions as both
        index and columns.

    Mathematical Note:
    ------------------
    For n regions, the similarity matrix S is n×n symmetric.
    The diagonal elements are always 1 (or 0 for euclidean) since
    each region is perfectly similar to itself.
    """
    if config is not None:
        metric = config.similarity_metric

    regions = data.index.tolist()
    X = data.values  # (n_regions, n_topics)
    n_regions = X.shape[0]

    logger.info(f"Computing {metric} similarity matrix for {n_regions} regions")

    if metric == 'correlation':
        # Pearson correlation using numpy
        # For z-scored data, correlation = X @ X.T / (p - 1)
        # But we use corrcoef for numerical stability
        S = np.corrcoef(X)

    elif metric == 'cosine':
        # Cosine similarity: 1 - cosine_distance
        # scipy.spatial.distance.cdist returns distances, not similarities
        S = 1 - spatial.distance.cdist(X, X, metric='cosine')

    elif metric == 'euclidean':
        # Negative Euclidean distance (so higher = more similar)
        dist = spatial.distance.cdist(X, X, metric='euclidean')
        # Normalize to [-1, 0] range for comparability
        S = -dist / dist.max() if dist.max() > 0 else -dist

    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    # Ensure symmetry (numerical precision)
    S = (S + S.T) / 2

    # Ensure diagonal is exactly 1 (or 0 for euclidean)
    if metric in ['correlation', 'cosine']:
        np.fill_diagonal(S, 1.0)
    else:
        np.fill_diagonal(S, 0.0)

    # Create DataFrame
    similarity_df = pd.DataFrame(S, index=regions, columns=regions)
    similarity_df.index.name = 'region_i'
    similarity_df.columns.name = 'region_j'

    return similarity_df


def get_pairwise_similarity(
    similarity_matrix: pd.DataFrame,
    region1: str,
    region2: str
) -> float:
    """
    Get similarity between two specific regions.

    Args:
        similarity_matrix: The computed similarity matrix
        region1: First region identifier
        region2: Second region identifier

    Returns:
        Similarity value between the two regions
    """
    return similarity_matrix.loc[region1, region2]


def get_most_similar_regions(
    similarity_matrix: pd.DataFrame,
    region: str,
    n: int = 5
) -> pd.Series:
    """
    Find the n most similar regions to a given region.

    Args:
        similarity_matrix: The computed similarity matrix
        region: Reference region
        n: Number of similar regions to return

    Returns:
        Series of similarity values for the n most similar regions
        (excluding the region itself)
    """
    similarities = similarity_matrix.loc[region].drop(region)
    return similarities.nlargest(n)


def get_most_dissimilar_regions(
    similarity_matrix: pd.DataFrame,
    region: str,
    n: int = 5
) -> pd.Series:
    """
    Find the n most dissimilar regions to a given region.

    Args:
        similarity_matrix: The computed similarity matrix
        region: Reference region
        n: Number of dissimilar regions to return

    Returns:
        Series of similarity values for the n most dissimilar regions
    """
    similarities = similarity_matrix.loc[region].drop(region)
    return similarities.nsmallest(n)


def compare_regions(
    similarity_matrix: pd.DataFrame,
    regions: List[str]
) -> pd.DataFrame:
    """
    Create a comparison matrix for a subset of regions.

    Useful for targeted comparisons like "How similar are Miami,
    NYC, and Atlanta to each other?"

    Args:
        similarity_matrix: Full similarity matrix
        regions: List of regions to compare

    Returns:
        Sub-matrix showing pairwise similarities among the specified regions
    """
    return similarity_matrix.loc[regions, regions]


def compute_distance_matrix(
    similarity_matrix: pd.DataFrame,
    transform: str = 'subtract'
) -> pd.DataFrame:
    """
    Convert similarity matrix to distance matrix.

    Useful for algorithms that require distances (e.g., MDS, hierarchical
    clustering, some graph algorithms).

    Args:
        similarity_matrix: Similarity matrix with values typically in [-1, 1]
        transform: How to convert similarity to distance
            - 'subtract': d = 1 - s (valid for s ∈ [0, 1])
            - 'sqrt': d = sqrt(2 * (1 - s)) (Euclidean-like, valid for correlations)
            - 'arccos': d = arccos(s) (angular distance)

    Returns:
        Distance matrix where smaller values = more similar

    Mathematical Note:
    ------------------
    The sqrt(2*(1-r)) transformation is particularly meaningful for
    correlation matrices. It's equivalent to the Euclidean distance
    between standardized vectors:
        ||z_i - z_j||^2 = 2(1 - r_ij)

    This allows interpreting the distance geometrically in the
    standardized feature space.
    """
    S = similarity_matrix.values

    if transform == 'subtract':
        D = 1 - S
    elif transform == 'sqrt':
        # sqrt(2*(1-r)) - valid for correlations in [-1, 1]
        D = np.sqrt(np.maximum(0, 2 * (1 - S)))
    elif transform == 'arccos':
        # Angular distance
        D = np.arccos(np.clip(S, -1, 1))
    else:
        raise ValueError(f"Unknown transform: {transform}")

    # Ensure non-negative and zero diagonal
    D = np.maximum(0, D)
    np.fill_diagonal(D, 0)

    return pd.DataFrame(
        D,
        index=similarity_matrix.index,
        columns=similarity_matrix.columns
    )


def analyze_similarity_matrix(similarity_matrix: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for the similarity matrix.

    Returns:
        Dictionary of statistics useful for understanding the
        overall structure of regional similarities.
    """
    S = similarity_matrix.values
    n = S.shape[0]

    # Get upper triangle (excluding diagonal)
    upper_indices = np.triu_indices(n, k=1)
    upper_values = S[upper_indices]

    stats = {
        'n_regions': n,
        'n_pairs': len(upper_values),

        # Distribution of pairwise similarities
        'mean_similarity': upper_values.mean(),
        'std_similarity': upper_values.std(),
        'min_similarity': upper_values.min(),
        'max_similarity': upper_values.max(),
        'median_similarity': np.median(upper_values),

        # Percentiles
        'percentile_25': np.percentile(upper_values, 25),
        'percentile_75': np.percentile(upper_values, 75),

        # Most/least similar pairs
        'max_pair_idx': np.unravel_index(
            np.argmax(S + np.eye(n) * -np.inf),  # Exclude diagonal
            S.shape
        ),
        'min_pair_idx': np.unravel_index(np.argmin(S), S.shape),
    }

    # Convert indices to region names
    regions = similarity_matrix.index.tolist()
    stats['most_similar_pair'] = (
        regions[stats['max_pair_idx'][0]],
        regions[stats['max_pair_idx'][1]],
        S[stats['max_pair_idx']]
    )
    stats['least_similar_pair'] = (
        regions[stats['min_pair_idx'][0]],
        regions[stats['min_pair_idx'][1]],
        S[stats['min_pair_idx']]
    )

    # Average similarity per region (how "typical" is each region)
    avg_by_region = (S.sum(axis=1) - 1) / (n - 1)  # Exclude self-similarity
    most_typical_idx = np.argmax(avg_by_region)
    least_typical_idx = np.argmin(avg_by_region)

    stats['most_typical_region'] = (regions[most_typical_idx], avg_by_region[most_typical_idx])
    stats['least_typical_region'] = (regions[least_typical_idx], avg_by_region[least_typical_idx])

    return stats


def cultural_distance(
    similarity_matrix: pd.DataFrame,
    region_a: str,
    region_b: str,
    region_c: str
) -> Dict:
    """
    Compare cultural distances between three regions.

    Answers questions like: "Is Miami culturally closer to NYC than Atlanta?"

    Args:
        similarity_matrix: The similarity matrix
        region_a: First region (the reference)
        region_b: Second region
        region_c: Third region

    Returns:
        Dictionary with distance comparisons and interpretation
    """
    sim_ab = similarity_matrix.loc[region_a, region_b]
    sim_ac = similarity_matrix.loc[region_a, region_c]
    sim_bc = similarity_matrix.loc[region_b, region_c]

    # Determine which is closer to region_a
    if sim_ab > sim_ac:
        closer_to_a = region_b
        farther_from_a = region_c
    else:
        closer_to_a = region_c
        farther_from_a = region_b

    result = {
        'reference': region_a,
        'comparison': [region_b, region_c],

        'similarity_a_b': sim_ab,
        'similarity_a_c': sim_ac,
        'similarity_b_c': sim_bc,

        'closer_to_reference': closer_to_a,
        'farther_from_reference': farther_from_a,
        'similarity_difference': abs(sim_ab - sim_ac),

        'interpretation': (
            f"{region_a} is culturally closer to {closer_to_a} "
            f"(similarity: {max(sim_ab, sim_ac):.3f}) than to {farther_from_a} "
            f"(similarity: {min(sim_ab, sim_ac):.3f}). "
            f"The difference is {abs(sim_ab - sim_ac):.3f}."
        )
    }

    return result
