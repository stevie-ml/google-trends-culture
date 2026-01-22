#!/usr/bin/env python3
"""
Interactive Cultural Similarity Analysis

This script provides an easy interface to explore cultural similarity
with different topic categories and cluster configurations.

Usage:
    # Run with music topics only
    python run_interactive.py --topics music

    # Run with political topics, 4 clusters
    python run_interactive.py --topics politics --clusters 4

    # Run with all topics and show metro areas
    python run_interactive.py --topics all --geo metro

    # Compare all topic presets
    python run_interactive.py --compare-all --clusters 5

    # Generate maps (requires geopandas)
    python run_interactive.py --topics all --show-map

Examples:
    python run_interactive.py --topics music --clusters 5
    python run_interactive.py --topics politics --geo state
    python run_interactive.py --compare-all
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import AnalysisConfig, TOPIC_PRESETS, STATE_NAMES
from src.interactive import (
    CulturalAnalysisRunner,
    create_embedding_plot_with_labels,
    create_comparison_figure,
    create_cluster_map_with_embedding,
    run_preset_comparison,
    CLUSTER_COLORS
)
from src.similarity import cultural_distance, get_most_similar_regions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_analysis_summary(results: dict):
    """Print a summary of the analysis results."""
    config = results['config']
    clustering = results['clustering_result']
    spectral = results['spectral_result']
    cluster_analysis = results['cluster_analysis']

    preset = config.topic_preset or 'custom'
    n_topics = results['processed_data'].shape[1]
    n_regions = results['processed_data'].shape[0]

    print_header(f"ANALYSIS: {preset.upper()} TOPICS")

    print(f"""
Configuration:
  Topic preset: {preset}
  Geographic unit: {config.geo_unit}
  Number of topics: {n_topics}
  Number of regions: {n_regions}

Latent Structure:
  Cultural dimensions found: {spectral.n_components_selected}
  Variance explained by top 2: {spectral.cumulative_variance[1]*100:.1f}%

Clustering:
  Number of clusters: {clustering.n_clusters}
  Silhouette score: {clustering.metrics.get('silhouette_score', 0):.3f}
""")

    # Print cluster composition
    print("Cluster Composition:")
    for k in range(clustering.n_clusters):
        info = cluster_analysis[f'cluster_{k}']
        members = info['members']
        color_idx = k % len(CLUSTER_COLORS)
        print(f"\n  Cluster {k + 1} ({info['size']} regions):")
        print(f"    Members: {', '.join(members[:10])}{'...' if len(members) > 10 else ''}")

        # Show distinctive topics
        high_topics = list(info['distinctive_topics_high'].keys())[:3]
        low_topics = list(info['distinctive_topics_low'].keys())[:3]
        print(f"    Higher interest: {', '.join(high_topics)}")
        print(f"    Lower interest: {', '.join(low_topics)}")

    # Axis interpretations
    print("\nCultural Axis Interpretations:")
    for i, label in enumerate(results['axis_labels'][:3]):
        print(f"  Axis {i + 1}: {label}")


def print_comparisons(results: dict):
    """Print cultural distance comparisons."""
    sim_matrix = results['similarity_matrix']
    available = set(sim_matrix.index)

    comparisons = [
        ('FL', 'NY', 'GA'),
        ('CA', 'WA', 'TX'),
        ('IL', 'MI', 'OH'),
        ('MA', 'CT', 'VT'),
    ]

    print("\nCultural Distance Comparisons:")
    for a, b, c in comparisons:
        if a in available and b in available and c in available:
            result = cultural_distance(sim_matrix, a, b, c)
            a_name = STATE_NAMES.get(a, a)
            print(f"  {result['interpretation']}")

    # Most similar to key states
    print("\nRegional Similarities:")
    key_states = ['NY', 'CA', 'TX', 'FL']
    for state in key_states:
        if state in available:
            similar = get_most_similar_regions(sim_matrix, state, n=3)
            name = STATE_NAMES.get(state, state)
            similar_list = [f"{s} ({v:.2f})" for s, v in similar.items()]
            print(f"  {name} most similar to: {', '.join(similar_list)}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Cultural Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--topics', '-t',
        choices=['music', 'politics', 'culture', 'all'],
        default='all',
        help='Topic preset to analyze (default: all)'
    )
    parser.add_argument(
        '--clusters', '-k',
        type=int,
        default=None,
        help='Number of clusters (default: auto-select)'
    )
    parser.add_argument(
        '--geo', '-g',
        choices=['state', 'metro'],
        default='state',
        help='Geographic unit (default: state)'
    )
    parser.add_argument(
        '--compare-all', '-c',
        action='store_true',
        help='Compare all topic presets'
    )
    parser.add_argument(
        '--show-map', '-m',
        action='store_true',
        help='Show geographic map (requires geopandas)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('outputs'),
        help='Output directory for figures'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Save figures but do not display'
    )
    parser.add_argument(
        '--highlight',
        nargs='+',
        default=['NY', 'CA', 'TX', 'FL', 'IL', 'GA', 'WA', 'MA'],
        help='Regions to highlight in plots'
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("""
╔═══════════════════════════════════════════════════════════════╗
║           INTERACTIVE CULTURAL SIMILARITY ANALYSIS             ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    runner = CulturalAnalysisRunner()

    if args.compare_all:
        # Compare all presets
        print("Comparing all topic presets...")
        presets = ['music', 'politics', 'culture', 'all']
        all_results = {}

        for preset in presets:
            print(f"\nAnalyzing {preset} topics...")
            all_results[preset] = runner.run_analysis(
                topic_preset=preset,
                geo_unit=args.geo,
                n_clusters=args.clusters
            )
            print_analysis_summary(all_results[preset])

        # Create comparison figure
        print("\nGenerating comparison figure...")
        fig = create_comparison_figure(all_results, figsize=(16, 12))
        fig.savefig(args.output_dir / 'topic_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output_dir / 'topic_comparison.png'}")

        if not args.no_show:
            plt.show()

    else:
        # Single analysis
        print(f"Analyzing with {args.topics} topics...")
        results = runner.run_analysis(
            topic_preset=args.topics,
            geo_unit=args.geo,
            n_clusters=args.clusters
        )

        # Print summary
        print_analysis_summary(results)
        print_comparisons(results)

        # Create embedding plot
        print("\nGenerating embedding plot...")
        fig1 = create_embedding_plot_with_labels(
            results,
            highlight_regions=args.highlight,
            show_all_labels=(args.geo == 'state')
        )
        filename1 = f'embedding_{args.topics}_{args.geo}.png'
        fig1.savefig(args.output_dir / filename1, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output_dir / filename1}")

        # Create map if requested
        if args.show_map and args.geo == 'state':
            try:
                print("\nGenerating geographic map...")
                fig2 = create_cluster_map_with_embedding(results, figsize=(18, 8))
                filename2 = f'map_{args.topics}_{args.geo}.png'
                fig2.savefig(args.output_dir / filename2, dpi=150, bbox_inches='tight')
                print(f"Saved: {args.output_dir / filename2}")
            except Exception as e:
                print(f"Could not generate map: {e}")
                print("Install geopandas for geographic visualization: pip install geopandas")

        if not args.no_show:
            plt.show()

    print("\n✓ Analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
