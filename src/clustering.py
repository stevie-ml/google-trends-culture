"""
Clustering Module for Cultural Region Analysis

This module provides clustering algorithms to identify distinct "cultural
groups" of regions based on their search behavior similarity.

Mathematical Background:
------------------------
Clustering partitions the n regions into K groups such that regions within
a group are more similar to each other than to regions in other groups.

Three algorithms are implemented:

1. Gaussian Mixture Model (GMM):
   Assumes regions are drawn from a mixture of K Gaussian distributions
   in the embedded space. Provides "soft" cluster assignments (probabilities).

   p(x|θ) = Σ_k π_k * N(x | μ_k, Σ_k)

   where π_k are mixing weights, μ_k are cluster means, Σ_k are covariances.

   Advantages:
   - Probabilistic: gives cluster membership probabilities
   - Allows overlapping/fuzzy clusters (a region can be "60% South, 40% Midwest")
   - Naturally handles uncertainty

2. Spectral Clustering:
   Constructs a graph where regions are nodes and edge weights are similarities.
   Partitions the graph by cutting edges with small total weight.

   Steps:
   a) Build similarity graph from S
   b) Compute graph Laplacian L = D - S (or normalized variant)
   c) Find bottom K eigenvectors of L
   d) Cluster the eigenvector embedding with K-means

   Advantages:
   - Doesn't assume convex clusters
   - Naturally uses the similarity matrix
   - Can find complex cluster shapes

3. K-Means (baseline):
   Partitions regions to minimize within-cluster variance in the embedding space.

   Advantages:
   - Simple and interpretable
   - Fast
   - Good baseline for comparison

Selecting the Number of Clusters:
---------------------------------
- BIC (Bayesian Information Criterion): For GMM, lower is better
- Silhouette Score: Measures how well regions fit their cluster
- Gap Statistic: Compares within-cluster dispersion to random reference
"""

import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples

from .config import AnalysisConfig, default_config
from .spectral_analysis import SpectralResult

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """
    Container for clustering results.

    Attributes:
        labels: Hard cluster assignment for each region (0 to K-1)
        probabilities: Soft assignment probabilities (n_regions × K) for GMM
        n_clusters: Number of clusters
        method: Clustering method used
        region_labels: List of region names
        metrics: Dictionary of clustering quality metrics
        model: The fitted clustering model (for GMM/KMeans)
    """
    labels: np.ndarray
    probabilities: Optional[np.ndarray]
    n_clusters: int
    method: str
    region_labels: List[str]
    metrics: Dict
    model: object = None

    def get_cluster_df(self) -> pd.DataFrame:
        """Get cluster assignments as DataFrame."""
        df = pd.DataFrame({
            'region': self.region_labels,
            'cluster': self.labels
        })
        df = df.set_index('region')

        # Add probabilities if available (GMM)
        if self.probabilities is not None:
            for k in range(self.n_clusters):
                df[f'prob_cluster_{k}'] = self.probabilities[:, k]

        return df

    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Get list of regions in a specific cluster."""
        mask = self.labels == cluster_id
        return [r for r, m in zip(self.region_labels, mask) if m]


def cluster_regions(
    spectral_result: SpectralResult,
    similarity_matrix: pd.DataFrame,
    config: AnalysisConfig = default_config
) -> ClusteringResult:
    """
    Main clustering function.

    Clusters regions using the specified method and automatically
    selects the number of clusters if not provided.

    Args:
        spectral_result: Result from spectral decomposition (provides embedding)
        similarity_matrix: The similarity matrix (for spectral clustering)
        config: Configuration with clustering parameters

    Returns:
        ClusteringResult with cluster assignments and quality metrics
    """
    method = config.clustering_method
    regions = spectral_result.region_labels

    # Get embedding for clustering
    n_dims = min(spectral_result.n_components_selected, 10)  # Cap at 10 dims
    embedding = spectral_result.embedding[:, :n_dims]

    logger.info(f"Clustering {len(regions)} regions using {method}")

    # Determine number of clusters
    if config.n_clusters is not None:
        n_clusters = config.n_clusters
    else:
        n_clusters = _select_n_clusters(
            embedding,
            similarity_matrix.values,
            method,
            config.cluster_range,
            config.random_seed
        )

    logger.info(f"Using {n_clusters} clusters")

    # Perform clustering
    if method == 'gmm':
        return _gmm_clustering(embedding, n_clusters, regions, config.random_seed)
    elif method == 'spectral':
        return _spectral_clustering(similarity_matrix.values, n_clusters, regions, config.random_seed)
    elif method == 'kmeans':
        return _kmeans_clustering(embedding, n_clusters, regions, config.random_seed)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def _gmm_clustering(
    embedding: np.ndarray,
    n_clusters: int,
    regions: List[str],
    random_seed: int
) -> ClusteringResult:
    """
    Gaussian Mixture Model clustering.

    GMM is particularly appropriate for cultural analysis because:
    1. It provides probability of cluster membership, allowing for
       statements like "Region X is 70% Southern, 30% Midwestern"
    2. It can model clusters with different shapes and sizes
    3. It naturally handles uncertainty in cluster boundaries
    """
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',  # Allow each cluster its own covariance
        random_state=random_seed,
        n_init=10,  # Multiple initializations for stability
    )

    # Fit and predict
    labels = gmm.fit_predict(embedding)
    probabilities = gmm.predict_proba(embedding)

    # Compute metrics
    silhouette = silhouette_score(embedding, labels) if n_clusters > 1 else 0
    bic = gmm.bic(embedding)
    aic = gmm.aic(embedding)

    metrics = {
        'silhouette_score': silhouette,
        'bic': bic,
        'aic': aic,
        'log_likelihood': gmm.score(embedding) * len(embedding),
        'n_iterations': gmm.n_iter_,
        'converged': gmm.converged_,
    }

    return ClusteringResult(
        labels=labels,
        probabilities=probabilities,
        n_clusters=n_clusters,
        method='gmm',
        region_labels=regions,
        metrics=metrics,
        model=gmm
    )


def _spectral_clustering(
    similarity_matrix: np.ndarray,
    n_clusters: int,
    regions: List[str],
    random_seed: int
) -> ClusteringResult:
    """
    Spectral clustering directly on the similarity matrix.

    Spectral clustering works by:
    1. Converting similarity to an affinity matrix
    2. Computing the graph Laplacian
    3. Finding the bottom eigenvectors of the Laplacian
    4. Clustering the eigenvector embedding

    This is ideal when you want clusters to respect the similarity
    structure directly, without going through an intermediate embedding.
    """
    # Ensure similarity is non-negative for affinity
    affinity = similarity_matrix.copy()
    affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min())
    np.fill_diagonal(affinity, 1)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_seed,
        n_init=10,
    )

    labels = spectral.fit_predict(affinity)

    # Compute silhouette on the original embedding
    # (spectral clustering doesn't have a natural embedding to use)
    # Use the similarity-derived distance for silhouette
    distance = 1 - similarity_matrix
    distance = np.maximum(0, distance)
    np.fill_diagonal(distance, 0)

    silhouette = silhouette_score(distance, labels, metric='precomputed') if n_clusters > 1 else 0

    metrics = {
        'silhouette_score': silhouette,
    }

    return ClusteringResult(
        labels=labels,
        probabilities=None,  # Spectral clustering doesn't give probabilities
        n_clusters=n_clusters,
        method='spectral',
        region_labels=regions,
        metrics=metrics,
        model=spectral
    )


def _kmeans_clustering(
    embedding: np.ndarray,
    n_clusters: int,
    regions: List[str],
    random_seed: int
) -> ClusteringResult:
    """
    K-Means clustering on the embedding.

    K-Means is included as a simple baseline. It works well when
    clusters are roughly spherical in the embedding space.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=10,
    )

    labels = kmeans.fit_predict(embedding)

    # Compute metrics
    silhouette = silhouette_score(embedding, labels) if n_clusters > 1 else 0
    inertia = kmeans.inertia_

    metrics = {
        'silhouette_score': silhouette,
        'inertia': inertia,
        'n_iterations': kmeans.n_iter_,
    }

    return ClusteringResult(
        labels=labels,
        probabilities=None,
        n_clusters=n_clusters,
        method='kmeans',
        region_labels=regions,
        metrics=metrics,
        model=kmeans
    )


def _select_n_clusters(
    embedding: np.ndarray,
    similarity_matrix: np.ndarray,
    method: str,
    cluster_range: Tuple[int, int],
    random_seed: int
) -> int:
    """
    Automatically select the number of clusters.

    Uses BIC for GMM, silhouette score otherwise.
    """
    k_min, k_max = cluster_range
    k_values = range(k_min, k_max + 1)

    if method == 'gmm':
        # Use BIC (lower is better)
        bics = []
        for k in k_values:
            gmm = GaussianMixture(n_components=k, random_state=random_seed, n_init=5)
            gmm.fit(embedding)
            bics.append(gmm.bic(embedding))

        best_k = k_values[np.argmin(bics)]
        logger.info(f"BIC-selected k={best_k}")

    else:
        # Use silhouette score (higher is better)
        silhouettes = []
        for k in k_values:
            if method == 'spectral':
                affinity = similarity_matrix.copy()
                affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min())
                spectral = SpectralClustering(n_clusters=k, affinity='precomputed',
                                              random_state=random_seed)
                labels = spectral.fit_predict(affinity)
            else:
                kmeans = KMeans(n_clusters=k, random_state=random_seed)
                labels = kmeans.fit_predict(embedding)

            silhouettes.append(silhouette_score(embedding, labels))

        best_k = k_values[np.argmax(silhouettes)]
        logger.info(f"Silhouette-selected k={best_k}")

    return best_k


def analyze_clusters(
    clustering_result: ClusteringResult,
    spectral_result: SpectralResult,
    data: pd.DataFrame
) -> Dict:
    """
    Analyze and interpret the clusters.

    For each cluster, compute:
    - Member regions
    - Centroid in embedding space
    - Characteristic topics (what makes this cluster distinctive)
    - Size and cohesion

    Args:
        clustering_result: The clustering result
        spectral_result: Spectral decomposition result
        data: Original normalized data matrix

    Returns:
        Dictionary with detailed cluster analysis
    """
    n_clusters = clustering_result.n_clusters
    labels = clustering_result.labels
    regions = clustering_result.region_labels

    n_dims = min(spectral_result.n_components_selected, 10)
    embedding = spectral_result.embedding[:, :n_dims]

    analysis = {}

    for k in range(n_clusters):
        mask = labels == k
        members = [r for r, m in zip(regions, mask) if m]

        # Cluster centroid
        centroid = embedding[mask].mean(axis=0)

        # Within-cluster distances
        cluster_embedding = embedding[mask]
        if len(cluster_embedding) > 1:
            dists = np.linalg.norm(cluster_embedding - centroid, axis=1)
            cohesion = dists.mean()
        else:
            cohesion = 0

        # Characteristic topics
        # Compare mean topic values in cluster vs overall
        cluster_data = data.loc[members]
        cluster_means = cluster_data.mean(axis=0)
        overall_means = data.mean(axis=0)
        topic_diff = cluster_means - overall_means

        # Top distinctive topics (positive and negative)
        sorted_topics = topic_diff.sort_values()
        distinctive_positive = sorted_topics.tail(5).iloc[::-1].to_dict()
        distinctive_negative = sorted_topics.head(5).to_dict()

        analysis[f'cluster_{k}'] = {
            'members': members,
            'size': len(members),
            'centroid': centroid,
            'cohesion': cohesion,
            'distinctive_topics_high': distinctive_positive,
            'distinctive_topics_low': distinctive_negative,
        }

        # Add probability info if available
        if clustering_result.probabilities is not None:
            probs = clustering_result.probabilities[mask, k]
            analysis[f'cluster_{k}']['mean_probability'] = probs.mean()
            analysis[f'cluster_{k}']['min_probability'] = probs.min()

    # Overall statistics
    analysis['summary'] = {
        'n_clusters': n_clusters,
        'method': clustering_result.method,
        'silhouette_score': clustering_result.metrics.get('silhouette_score', None),
        'cluster_sizes': [np.sum(labels == k) for k in range(n_clusters)],
    }

    return analysis


def get_cluster_profile(
    cluster_id: int,
    clustering_result: ClusteringResult,
    data: pd.DataFrame
) -> pd.Series:
    """
    Get the average search profile for a cluster.

    This shows the typical search behavior of regions in this cluster.

    Args:
        cluster_id: The cluster number (0 to K-1)
        clustering_result: Clustering result
        data: Original data matrix

    Returns:
        Series with mean search interest for each topic
    """
    members = clustering_result.get_cluster_members(cluster_id)
    return data.loc[members].mean(axis=0)


def compare_clusters(
    cluster_a: int,
    cluster_b: int,
    clustering_result: ClusteringResult,
    data: pd.DataFrame
) -> Dict:
    """
    Compare two clusters to find what distinguishes them.

    Useful for understanding what makes, e.g., the "Southern" cluster
    different from the "Northeastern" cluster.

    Args:
        cluster_a: First cluster ID
        cluster_b: Second cluster ID
        clustering_result: Clustering result
        data: Original data matrix

    Returns:
        Dictionary with comparison statistics
    """
    members_a = clustering_result.get_cluster_members(cluster_a)
    members_b = clustering_result.get_cluster_members(cluster_b)

    mean_a = data.loc[members_a].mean(axis=0)
    mean_b = data.loc[members_b].mean(axis=0)

    diff = mean_a - mean_b

    return {
        'cluster_a': cluster_a,
        'cluster_b': cluster_b,
        'size_a': len(members_a),
        'size_b': len(members_b),
        'topics_higher_in_a': diff.nlargest(5).to_dict(),
        'topics_higher_in_b': (-diff).nlargest(5).to_dict(),
    }


def compute_silhouette_by_region(
    clustering_result: ClusteringResult,
    spectral_result: SpectralResult
) -> pd.Series:
    """
    Compute silhouette score for each region.

    Regions with low silhouette scores are on the boundary between
    clusters or may be misclassified.

    Returns:
        Series with silhouette score per region
    """
    n_dims = min(spectral_result.n_components_selected, 10)
    embedding = spectral_result.embedding[:, :n_dims]

    scores = silhouette_samples(embedding, clustering_result.labels)

    return pd.Series(scores, index=clustering_result.region_labels, name='silhouette')
