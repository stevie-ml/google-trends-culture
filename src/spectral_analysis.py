"""
Spectral Analysis and Dimensionality Reduction for Cultural Analysis

This module performs eigenvalue decomposition of the similarity matrix
to discover latent cultural dimensions and embed regions in a
low-dimensional space.

Mathematical Framework:
-----------------------
Given the similarity matrix S ∈ ℝ^(n×n), the spectral decomposition is:

    S = VΛV^T

where:
- V ∈ ℝ^(n×n) is the matrix of eigenvectors (columns are eigenvectors)
- Λ = diag(λ_1, ..., λ_n) is the diagonal matrix of eigenvalues
- Eigenvalues are ordered: λ_1 ≥ λ_2 ≥ ... ≥ λ_n

Interpretation of eigenvalues:
- Each eigenvalue λ_k represents the variance explained by the k-th
  principal direction (cultural mode)
- For a correlation matrix, Σλ_k = n (trace equals dimension)
- The ratio λ_k / Σλ explains the proportion of total variance

Interpretation of eigenvectors:
- Each eigenvector v_k defines a "cultural mode" - a pattern of
  regional variation
- The i-th component v_k[i] tells us how region i loads on mode k
- Regions with similar loadings are culturally similar along that mode

Embedding regions:
- The k-dimensional embedding of region i is: (λ_1^(1/2) v_1[i], ..., λ_k^(1/2) v_k[i])
- This is equivalent to PCA on the original data matrix
- The embedding preserves pairwise distances optimally in the least-squares sense

Determining the number of cultural modes:
1. Kaiser criterion: Keep eigenvalues > 1 (for correlation matrices)
2. Scree test: Look for "elbow" in eigenvalue decay
3. Variance threshold: Keep enough modes to explain X% of variance
4. Parallel analysis: Compare to random data eigenvalues
"""

import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA

from .config import AnalysisConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class SpectralResult:
    """
    Container for spectral decomposition results.

    Attributes:
        eigenvalues: Array of eigenvalues in descending order
        eigenvectors: Matrix where columns are eigenvectors
        variance_explained: Proportion of variance explained by each component
        cumulative_variance: Cumulative variance explained
        n_components_selected: Number of components selected by criterion
        embedding: Low-dimensional embedding of regions
        region_labels: List of region names
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    variance_explained: np.ndarray
    cumulative_variance: np.ndarray
    n_components_selected: int
    embedding: np.ndarray
    region_labels: List[str]

    def get_embedding_df(self, n_components: Optional[int] = None) -> pd.DataFrame:
        """Get embedding as DataFrame with region labels."""
        if n_components is None:
            n_components = self.n_components_selected

        cols = [f'PC{i+1}' for i in range(n_components)]
        return pd.DataFrame(
            self.embedding[:, :n_components],
            index=self.region_labels,
            columns=cols
        )

    def get_loadings_df(self, n_components: Optional[int] = None) -> pd.DataFrame:
        """Get eigenvector loadings as DataFrame."""
        if n_components is None:
            n_components = self.n_components_selected

        cols = [f'Mode{i+1}' for i in range(n_components)]
        return pd.DataFrame(
            self.eigenvectors[:, :n_components],
            index=self.region_labels,
            columns=cols
        )


def spectral_decomposition(
    similarity_matrix: pd.DataFrame,
    config: AnalysisConfig = default_config
) -> SpectralResult:
    """
    Perform spectral decomposition of the similarity matrix.

    This is the core function for discovering latent cultural dimensions.

    Args:
        similarity_matrix: The n×n similarity matrix
        config: Configuration specifying component selection criteria

    Returns:
        SpectralResult containing eigenvalues, eigenvectors, and embedding

    Mathematical Details:
    ---------------------
    1. Eigendecomposition: S = VΛV^T
    2. Sort by eigenvalue magnitude (descending)
    3. Select number of components by specified criterion
    4. Compute embedding: E = V @ diag(sqrt(λ))
    """
    regions = similarity_matrix.index.tolist()
    S = similarity_matrix.values
    n = S.shape[0]

    logger.info(f"Performing spectral decomposition on {n}×{n} similarity matrix")

    # Eigendecomposition
    # Use eigh for symmetric matrices (more numerically stable)
    eigenvalues, eigenvectors = linalg.eigh(S)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Handle numerical issues: small negative eigenvalues from numerical error
    eigenvalues = np.maximum(eigenvalues, 0)

    # Compute variance explained
    total_variance = np.sum(eigenvalues)
    variance_explained = eigenvalues / total_variance if total_variance > 0 else eigenvalues
    cumulative_variance = np.cumsum(variance_explained)

    # Select number of components
    n_components = _select_n_components(
        eigenvalues,
        cumulative_variance,
        config
    )

    logger.info(f"Selected {n_components} components explaining "
                f"{cumulative_variance[n_components-1]*100:.1f}% of variance")

    # Compute embedding
    # E[i,k] = sqrt(λ_k) * v_k[i]
    # This gives the coordinates of region i in the principal component space
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    embedding = eigenvectors * sqrt_eigenvalues[np.newaxis, :]

    return SpectralResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        variance_explained=variance_explained,
        cumulative_variance=cumulative_variance,
        n_components_selected=n_components,
        embedding=embedding,
        region_labels=regions
    )


def _select_n_components(
    eigenvalues: np.ndarray,
    cumulative_variance: np.ndarray,
    config: AnalysisConfig
) -> int:
    """
    Select the number of components to retain.

    Methods:
    1. Kaiser criterion: eigenvalue > 1 (for correlation matrix)
    2. Variance threshold: cumulative variance > threshold
    3. Elbow detection: find the "elbow" in the scree plot

    The Kaiser criterion (eigenvalue > 1) has a statistical interpretation:
    a component with eigenvalue < 1 explains less variance than a single
    original variable would, so it's not worth keeping.
    """
    criterion = config.component_criterion

    if config.n_components is not None:
        # User specified exact number
        return min(config.n_components, len(eigenvalues))

    if criterion == 'kaiser':
        # Keep eigenvalues > 1
        n_above_one = np.sum(eigenvalues > 1)
        return max(1, n_above_one)

    elif criterion == 'variance':
        # Keep until cumulative variance exceeds threshold
        threshold = config.variance_threshold
        n_for_variance = np.argmax(cumulative_variance >= threshold) + 1
        return max(1, n_for_variance)

    elif criterion == 'elbow':
        # Detect elbow using second derivative
        return _detect_elbow(eigenvalues)

    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def _detect_elbow(eigenvalues: np.ndarray) -> int:
    """
    Detect the elbow point in the eigenvalue sequence.

    The elbow is where the rate of decrease in eigenvalues slows down,
    indicating diminishing returns from additional components.

    Method: Find the point of maximum curvature using second differences.
    """
    if len(eigenvalues) < 3:
        return len(eigenvalues)

    # First difference (rate of change)
    d1 = np.diff(eigenvalues)

    # Second difference (acceleration)
    d2 = np.diff(d1)

    # Elbow is where second derivative is most positive
    # (i.e., where decline is slowing most rapidly)
    elbow_idx = np.argmax(d2) + 2  # +2 to account for two diffs

    # Ensure at least 1 component
    return max(1, min(elbow_idx, len(eigenvalues)))


def parallel_analysis(
    data: pd.DataFrame,
    n_iterations: int = 100,
    percentile: float = 95,
    random_seed: int = 42
) -> np.ndarray:
    """
    Perform parallel analysis to determine significant eigenvalues.

    Parallel analysis compares observed eigenvalues to those from random
    data with the same dimensions. Eigenvalues that exceed the random
    threshold are considered significant.

    Mathematical Basis:
    -------------------
    Under the null hypothesis of no structure, eigenvalues follow a
    known distribution (Marchenko-Pastur for large matrices). Parallel
    analysis is a simulation-based approximation to this distribution.

    Args:
        data: The original region × topic data matrix
        n_iterations: Number of random datasets to generate
        percentile: Percentile of random eigenvalues to use as threshold

    Returns:
        Array of threshold eigenvalues (one per component)
    """
    np.random.seed(random_seed)
    n_regions, n_topics = data.shape

    random_eigenvalues = []

    for _ in range(n_iterations):
        # Generate random data with same dimensions
        random_data = np.random.normal(0, 1, (n_regions, n_topics))

        # Compute correlation matrix
        random_corr = np.corrcoef(random_data)

        # Get eigenvalues
        eigvals = linalg.eigvalsh(random_corr)
        eigvals = np.sort(eigvals)[::-1]
        random_eigenvalues.append(eigvals)

    random_eigenvalues = np.array(random_eigenvalues)

    # Get threshold at specified percentile
    thresholds = np.percentile(random_eigenvalues, percentile, axis=0)

    return thresholds


def embed_regions(
    spectral_result: SpectralResult,
    n_dimensions: int = 2
) -> pd.DataFrame:
    """
    Get the n-dimensional embedding of regions.

    This embedding positions regions in a low-dimensional space where
    Euclidean distance approximates cultural dissimilarity.

    Args:
        spectral_result: Result from spectral_decomposition
        n_dimensions: Number of embedding dimensions (2 or 3 for visualization)

    Returns:
        DataFrame with regions as index and PC1, PC2, ... as columns
    """
    return spectral_result.get_embedding_df(n_dimensions)


def interpret_cultural_modes(
    spectral_result: SpectralResult,
    data: pd.DataFrame,
    n_modes: int = 4,
    n_top_topics: int = 5
) -> Dict:
    """
    Interpret the cultural modes by identifying defining characteristics.

    For each principal component (cultural mode), identify:
    1. Which regions load highest/lowest on this mode
    2. Which topics correlate most strongly with this mode

    This helps answer: "What does cultural mode 1 represent?"

    Args:
        spectral_result: Result from spectral decomposition
        data: Original normalized data matrix
        n_modes: Number of modes to interpret
        n_top_topics: Number of top topics to show per mode

    Returns:
        Dictionary with interpretation for each mode
    """
    n_modes = min(n_modes, spectral_result.n_components_selected)
    regions = spectral_result.region_labels
    topics = data.columns.tolist()

    interpretations = {}

    for k in range(n_modes):
        mode_name = f'Mode_{k+1}'
        loadings = spectral_result.eigenvectors[:, k]

        # Sort regions by loading
        sorted_idx = np.argsort(loadings)
        high_regions = [(regions[i], loadings[i]) for i in sorted_idx[-5:][::-1]]
        low_regions = [(regions[i], loadings[i]) for i in sorted_idx[:5]]

        # Correlate mode with original topics
        # This tells us which topics drive this cultural mode
        topic_correlations = []
        for j, topic in enumerate(topics):
            corr = np.corrcoef(loadings, data.iloc[:, j].values)[0, 1]
            topic_correlations.append((topic, corr))

        topic_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        # Positive and negative topic associations
        positive_topics = [(t, c) for t, c in topic_correlations if c > 0][:n_top_topics]
        negative_topics = [(t, c) for t, c in topic_correlations if c < 0][:n_top_topics]

        interpretations[mode_name] = {
            'eigenvalue': spectral_result.eigenvalues[k],
            'variance_explained': spectral_result.variance_explained[k],
            'high_loading_regions': high_regions,
            'low_loading_regions': low_regions,
            'positive_topic_correlations': positive_topics,
            'negative_topic_correlations': negative_topics,
        }

    return interpretations


def compute_region_coordinates(
    spectral_result: SpectralResult,
    region: str
) -> Dict:
    """
    Get detailed coordinates and interpretation for a specific region.

    Args:
        spectral_result: Spectral decomposition result
        region: Region name

    Returns:
        Dictionary with coordinates and relative position information
    """
    if region not in spectral_result.region_labels:
        raise ValueError(f"Region '{region}' not found")

    idx = spectral_result.region_labels.index(region)
    n_modes = spectral_result.n_components_selected

    # Get coordinates
    coords = spectral_result.embedding[idx, :n_modes]

    # Compute percentile rank for each mode
    percentiles = []
    for k in range(n_modes):
        all_values = spectral_result.embedding[:, k]
        pct = 100 * np.mean(all_values <= coords[k])
        percentiles.append(pct)

    return {
        'region': region,
        'coordinates': coords,
        'percentile_ranks': percentiles,
        'mode_labels': [f'PC{i+1}' for i in range(n_modes)],
    }


def scree_analysis(spectral_result: SpectralResult) -> Dict:
    """
    Produce scree plot data and analysis.

    The scree plot shows eigenvalues vs. component number.
    The "elbow" in this plot suggests the number of meaningful components.

    Returns:
        Dictionary with data for scree plot and analysis
    """
    eigenvalues = spectral_result.eigenvalues
    variance_explained = spectral_result.variance_explained
    cumulative_variance = spectral_result.cumulative_variance

    # Detect potential elbows
    d2 = np.diff(np.diff(eigenvalues))
    potential_elbows = np.where(d2 > 0)[0] + 2  # +2 for diff offsets

    return {
        'eigenvalues': eigenvalues,
        'variance_explained': variance_explained,
        'cumulative_variance': cumulative_variance,
        'n_components': len(eigenvalues),
        'kaiser_threshold': 1.0,
        'n_above_kaiser': np.sum(eigenvalues > 1),
        'potential_elbows': potential_elbows.tolist(),
        'selected': spectral_result.n_components_selected,
    }
