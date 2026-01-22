#!/usr/bin/env python3
"""
Cultural Similarity Analysis of U.S. Regions Using Google Trends

This script performs a complete analysis pipeline to infer latent cultural
similarity between U.S. regions using Google search behavior data.

Usage:
    python run_analysis.py                    # Run with defaults (mock data)
    python run_analysis.py --real-data        # Use real Google Trends API
    python run_analysis.py --n-clusters 5     # Specify number of clusters

The analysis proceeds as follows:
1. Data Collection: Fetch (or simulate) Google Trends data for multiple topics
2. Preprocessing: Normalize the region × topic matrix
3. Similarity: Compute pairwise cultural similarity between regions
4. Spectral Analysis: Decompose similarity matrix to find latent dimensions
5. Clustering: Identify distinct cultural groups
6. Visualization: Generate interpretable plots

Output:
    - Console: Summary statistics and interpretations
    - outputs/: Figures and data files

Author: Cultural Analytics Research
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import AnalysisConfig, STATE_NAMES
from src.data_collection import collect_trends_data
from src.preprocessing import preprocess_data, compute_data_quality_report
from src.similarity import (
    compute_similarity_matrix,
    analyze_similarity_matrix,
    cultural_distance,
    get_most_similar_regions,
    get_most_dissimilar_regions
)
from src.spectral_analysis import (
    spectral_decomposition,
    scree_analysis,
    interpret_cultural_modes,
    compute_region_coordinates
)
from src.clustering import (
    cluster_regions,
    analyze_clusters,
    compute_silhouette_by_region
)
from src.visualization import (
    plot_scree,
    plot_embedding_2d,
    plot_embedding_3d,
    plot_similarity_heatmap,
    plot_cluster_comparison,
    plot_mode_interpretation,
    plot_cluster_profiles,
    save_figure,
    create_all_visualizations,
    save_all_figures
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cultural Similarity Analysis using Google Trends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--real-data', action='store_true',
        help='Use real Google Trends API (requires pytrends)'
    )
    parser.add_argument(
        '--geo-unit', choices=['state', 'metro'], default='state',
        help='Geographic unit for analysis (default: state)'
    )
    parser.add_argument(
        '--similarity', choices=['correlation', 'cosine', 'euclidean'],
        default='correlation',
        help='Similarity metric (default: correlation)'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=None,
        help='Number of clusters (default: auto-select)'
    )
    parser.add_argument(
        '--clustering', choices=['gmm', 'spectral', 'kmeans'],
        default='gmm',
        help='Clustering algorithm (default: gmm)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('outputs'),
        help='Output directory (default: outputs/)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots'
    )

    return parser.parse_args()


def print_section(title: str, char: str = '='):
    """Print a section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def run_analysis(config: AnalysisConfig, args) -> dict:
    """
    Run the complete cultural similarity analysis.

    Returns:
        Dictionary containing all results
    """
    results = {}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: DATA COLLECTION
    # =========================================================================
    print_section("STEP 1: DATA COLLECTION")

    logger.info("Collecting Google Trends data...")
    raw_data = collect_trends_data(config)

    print(f"Collected data for {raw_data.shape[0]} regions and {raw_data.shape[1]} topics")
    print(f"\nRegions: {list(raw_data.index[:10])}...")
    print(f"Topics: {list(raw_data.columns[:5])}...")

    # Save raw data
    raw_data.to_csv(output_dir / 'raw_data.csv')
    results['raw_data'] = raw_data

    # =========================================================================
    # STEP 2: PREPROCESSING
    # =========================================================================
    print_section("STEP 2: PREPROCESSING")

    data, metadata = preprocess_data(raw_data, config)

    print(f"Original shape: {metadata['original_shape']}")
    print(f"Final shape: {metadata['final_shape']}")
    print(f"Normalization: {metadata['normalization']}")
    print(f"Topics dropped: {len(metadata['topics_dropped'])}")
    print(f"Regions dropped: {len(metadata['regions_dropped'])}")

    # Quality report
    quality = compute_data_quality_report(data)
    print(f"\nData quality:")
    print(f"  - Missing values: {quality['pct_missing']:.2f}%")
    print(f"  - Sparsity: {quality['sparsity']*100:.1f}%")

    if quality['issues']:
        print(f"  - Issues: {quality['issues']}")

    data.to_csv(output_dir / 'processed_data.csv')
    results['processed_data'] = data
    results['metadata'] = metadata

    # =========================================================================
    # STEP 3: SIMILARITY MATRIX
    # =========================================================================
    print_section("STEP 3: SIMILARITY MATRIX COMPUTATION")

    similarity_matrix = compute_similarity_matrix(data, config=config)
    sim_stats = analyze_similarity_matrix(similarity_matrix)

    print(f"Similarity metric: {config.similarity_metric}")
    print(f"\nSimilarity distribution:")
    print(f"  - Mean: {sim_stats['mean_similarity']:.3f}")
    print(f"  - Std:  {sim_stats['std_similarity']:.3f}")
    print(f"  - Range: [{sim_stats['min_similarity']:.3f}, {sim_stats['max_similarity']:.3f}]")

    print(f"\nMost similar pair: {sim_stats['most_similar_pair'][0]} & "
          f"{sim_stats['most_similar_pair'][1]} "
          f"(similarity: {sim_stats['most_similar_pair'][2]:.3f})")
    print(f"Least similar pair: {sim_stats['least_similar_pair'][0]} & "
          f"{sim_stats['least_similar_pair'][1]} "
          f"(similarity: {sim_stats['least_similar_pair'][2]:.3f})")

    print(f"\nMost typical region: {sim_stats['most_typical_region'][0]} "
          f"(avg similarity: {sim_stats['most_typical_region'][1]:.3f})")
    print(f"Least typical region: {sim_stats['least_typical_region'][0]} "
          f"(avg similarity: {sim_stats['least_typical_region'][1]:.3f})")

    similarity_matrix.to_csv(output_dir / 'similarity_matrix.csv')
    results['similarity_matrix'] = similarity_matrix
    results['similarity_stats'] = sim_stats

    # =========================================================================
    # STEP 4: SPECTRAL ANALYSIS
    # =========================================================================
    print_section("STEP 4: SPECTRAL ANALYSIS (Latent Cultural Dimensions)")

    spectral_result = spectral_decomposition(similarity_matrix, config)
    scree = scree_analysis(spectral_result)

    print(f"Total eigenvalues: {scree['n_components']}")
    print(f"Eigenvalues > 1 (Kaiser): {scree['n_above_kaiser']}")
    print(f"Components selected: {scree['selected']}")

    print(f"\nTop eigenvalues and variance explained:")
    for i in range(min(8, len(spectral_result.eigenvalues))):
        eig = spectral_result.eigenvalues[i]
        var = spectral_result.variance_explained[i] * 100
        cumvar = spectral_result.cumulative_variance[i] * 100
        marker = " *" if i < spectral_result.n_components_selected else ""
        print(f"  λ_{i+1} = {eig:6.3f} ({var:5.1f}% var, {cumvar:5.1f}% cumulative){marker}")

    # Interpret cultural modes
    print(f"\n--- Interpretation of Cultural Modes ---")
    interpretations = interpret_cultural_modes(spectral_result, data, n_modes=3)

    for mode_name, interp in interpretations.items():
        print(f"\n{mode_name} ({interp['variance_explained']*100:.1f}% variance):")
        print(f"  High-loading regions: {[r[0] for r in interp['high_loading_regions'][:3]]}")
        print(f"  Low-loading regions: {[r[0] for r in interp['low_loading_regions'][:3]]}")
        print(f"  Positive topics: {[t[0] for t in interp['positive_topic_correlations'][:3]]}")
        print(f"  Negative topics: {[t[0] for t in interp['negative_topic_correlations'][:3]]}")

    # Save embedding
    embedding_df = spectral_result.get_embedding_df(n_components=5)
    embedding_df.to_csv(output_dir / 'embedding.csv')
    results['spectral_result'] = spectral_result
    results['scree'] = scree
    results['mode_interpretations'] = interpretations

    # =========================================================================
    # STEP 5: CLUSTERING
    # =========================================================================
    print_section("STEP 5: CLUSTERING (Cultural Groups)")

    clustering_result = cluster_regions(spectral_result, similarity_matrix, config)

    print(f"Clustering method: {clustering_result.method}")
    print(f"Number of clusters: {clustering_result.n_clusters}")
    print(f"Silhouette score: {clustering_result.metrics.get('silhouette_score', 'N/A'):.3f}")

    # Analyze clusters
    cluster_analysis = analyze_clusters(clustering_result, spectral_result, data)

    print(f"\n--- Cluster Composition ---")
    for k in range(clustering_result.n_clusters):
        info = cluster_analysis[f'cluster_{k}']
        members = info['members']
        print(f"\nCluster {k} ({info['size']} regions):")
        print(f"  Members: {members[:8]}{'...' if len(members) > 8 else ''}")
        print(f"  Distinctive topics (high): {list(info['distinctive_topics_high'].keys())[:3]}")
        print(f"  Distinctive topics (low): {list(info['distinctive_topics_low'].keys())[:3]}")

    # Save clustering
    cluster_df = clustering_result.get_cluster_df()
    cluster_df.to_csv(output_dir / 'clusters.csv')
    results['clustering_result'] = clustering_result
    results['cluster_analysis'] = cluster_analysis

    # =========================================================================
    # STEP 6: SPECIFIC COMPARISONS
    # =========================================================================
    print_section("STEP 6: CULTURAL DISTANCE COMPARISONS")

    # Example comparisons
    comparisons = [
        ('FL', 'NY', 'GA'),  # Is Florida closer to New York or Georgia?
        ('CA', 'WA', 'TX'),  # Is California closer to Washington or Texas?
        ('IL', 'MI', 'IN'),  # Midwest comparison
        ('MA', 'CT', 'NH'),  # New England comparison
    ]

    # Filter to available regions
    available_regions = set(similarity_matrix.index)
    valid_comparisons = [
        (a, b, c) for a, b, c in comparisons
        if a in available_regions and b in available_regions and c in available_regions
    ]

    print("Cultural distance comparisons (Reference -> Comparison regions):\n")
    for a, b, c in valid_comparisons:
        result = cultural_distance(similarity_matrix, a, b, c)
        print(f"  {result['interpretation']}")

    results['comparisons'] = valid_comparisons

    # Most similar/dissimilar for key regions
    print("\n--- Regional Similarity Analysis ---")
    key_regions = ['NY', 'CA', 'TX', 'FL', 'IL', 'GA']
    key_regions = [r for r in key_regions if r in available_regions]

    for region in key_regions[:4]:
        most_similar = get_most_similar_regions(similarity_matrix, region, n=3)
        most_dissimilar = get_most_dissimilar_regions(similarity_matrix, region, n=3)
        name = STATE_NAMES.get(region, region)
        print(f"\n{name} ({region}):")
        print(f"  Most similar: {list(most_similar.index)}")
        print(f"  Most different: {list(most_dissimilar.index)}")

    # =========================================================================
    # STEP 7: VISUALIZATION
    # =========================================================================
    if not args.no_plots:
        print_section("STEP 7: VISUALIZATION")

        # Key regions to highlight in plots
        highlight = ['NY', 'CA', 'TX', 'FL', 'IL', 'GA', 'MA', 'WA', 'CO', 'AZ']
        highlight = [r for r in highlight if r in available_regions]

        # Generate all figures
        figures = create_all_visualizations(
            similarity_matrix,
            spectral_result,
            clustering_result,
            data,
            config,
            highlight_regions=highlight
        )

        # Add specific comparison plot
        compare_regions = ['NY', 'FL', 'GA', 'CA', 'TX']
        compare_regions = [r for r in compare_regions if r in available_regions]
        if len(compare_regions) >= 3:
            figures['region_comparison'] = plot_cluster_comparison(
                compare_regions, similarity_matrix,
                title='Cultural Similarity: Key Regions'
            )

        # Save all figures
        paths = save_all_figures(figures, output_dir, config)
        print(f"Saved {len(paths)} figures to {output_dir}/")

        results['figures'] = list(figures.keys())

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("ANALYSIS SUMMARY", char='#')

    print(f"""
Cultural Similarity Analysis Complete
=====================================

Data:
  - Regions analyzed: {data.shape[0]}
  - Topics used: {data.shape[1]}
  - Similarity metric: {config.similarity_metric}

Latent Structure:
  - Cultural modes identified: {spectral_result.n_components_selected}
  - Variance explained by top modes: {spectral_result.cumulative_variance[spectral_result.n_components_selected-1]*100:.1f}%

Clustering:
  - Number of cultural clusters: {clustering_result.n_clusters}
  - Clustering method: {clustering_result.method}
  - Silhouette score: {clustering_result.metrics.get('silhouette_score', 0):.3f}

Output files saved to: {output_dir}/
  - raw_data.csv: Original Google Trends data
  - processed_data.csv: Normalized data matrix
  - similarity_matrix.csv: Region similarity matrix
  - embedding.csv: Low-dimensional embedding coordinates
  - clusters.csv: Cluster assignments
  - *.png: Visualization figures

Key Findings:
  - Most typical region: {sim_stats['most_typical_region'][0]}
  - Most distinctive region: {sim_stats['least_typical_region'][0]}
  - Most similar pair: {sim_stats['most_similar_pair'][0]} & {sim_stats['most_similar_pair'][1]}
""")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Build configuration from arguments
    config = AnalysisConfig(
        use_mock_data=not args.real_data,
        geo_unit=args.geo_unit,
        similarity_metric=args.similarity,
        n_clusters=args.n_clusters,
        clustering_method=args.clustering,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     CULTURAL SIMILARITY ANALYSIS OF U.S. REGIONS                     ║
║     Using Google Trends Search Behavior Data                         ║
╚══════════════════════════════════════════════════════════════════════╝

Configuration:
  - Data source: {'Real API' if not config.use_mock_data else 'Mock data'}
  - Geographic unit: {config.geo_unit}
  - Similarity metric: {config.similarity_metric}
  - Clustering: {config.clustering_method}
  - Random seed: {config.random_seed}
""")

    try:
        results = run_analysis(config, args)
        logger.info("Analysis completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    sys.exit(main())
