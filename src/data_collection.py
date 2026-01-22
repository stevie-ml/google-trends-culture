"""
Data Collection Module for Google Trends Cultural Analysis

This module handles fetching search interest data from Google Trends API
or generating realistic mock data for development and testing.

Mathematical Note:
------------------
Google Trends returns relative search interest normalized to [0, 100] where:
- 100 = peak popularity for that term in the given time/location
- 50 = half as popular as peak
- 0 = insufficient data

This relative scaling means we must be careful when comparing across
different queries. The pytrends library handles this by using a common
reference topic when comparing multiple terms.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    AnalysisConfig,
    US_STATES,
    STATE_NAMES,
    MAJOR_METROS,
    DEFAULT_TOPICS,
    default_config
)

logger = logging.getLogger(__name__)


class GoogleTrendsCollector:
    """
    Collector for Google Trends data with rate limiting and error handling.

    The collector supports both real API calls (via pytrends) and mock data
    generation for development/testing purposes.
    """

    def __init__(self, config: AnalysisConfig = default_config):
        self.config = config
        self._pytrends = None

    def _get_pytrends(self):
        """Lazy initialization of pytrends client."""
        if self._pytrends is None:
            try:
                from pytrends.request import TrendReq
                self._pytrends = TrendReq(hl='en-US', tz=360)
            except ImportError:
                raise ImportError(
                    "pytrends is required for real API calls. "
                    "Install with: pip install pytrends"
                )
        return self._pytrends

    def collect_data(self) -> pd.DataFrame:
        """
        Main entry point for data collection.

        Returns:
            DataFrame with shape (n_regions, n_topics) containing
            normalized search interest values.
        """
        if self.config.use_mock_data:
            logger.info("Using mock data for analysis")
            return self._generate_mock_data()
        else:
            logger.info("Fetching real data from Google Trends API")
            return self._fetch_real_data()

    def _fetch_real_data(self) -> pd.DataFrame:
        """
        Fetch real data from Google Trends API.

        Uses pytrends library to query interest by region for each topic.
        Implements rate limiting to avoid API throttling.

        Returns:
            DataFrame with regions as rows, topics as columns.
        """
        pytrends = self._get_pytrends()
        topics = self.config.get_topics()
        regions = self.config.get_regions()

        # Initialize result matrix
        data = np.zeros((len(regions), len(topics)))
        data[:] = np.nan

        # Create region to geo code mapping
        if self.config.geo_unit == 'state':
            geo_codes = {state: f'US-{state}' for state in regions}
        else:
            geo_codes = {metro: MAJOR_METROS[metro] for metro in regions}

        # Fetch data for each topic
        # Note: pytrends allows max 5 keywords per request
        for j, topic in enumerate(tqdm(topics, desc="Fetching topics")):
            try:
                # Build payload for this topic
                pytrends.build_payload(
                    kw_list=[topic],
                    timeframe=self.config.timeframe,
                    geo='US'
                )

                # Get interest by region
                interest_df = pytrends.interest_by_region(
                    resolution='REGION' if self.config.geo_unit == 'state' else 'DMA',
                    inc_low_vol=True
                )

                # Map results to our region ordering
                for i, region in enumerate(regions):
                    if self.config.geo_unit == 'state':
                        # Match by state name
                        state_name = STATE_NAMES.get(region, region)
                        if state_name in interest_df.index:
                            data[i, j] = interest_df.loc[state_name, topic]
                    else:
                        # Match by metro name (may need fuzzy matching)
                        if region in interest_df.index:
                            data[i, j] = interest_df.loc[region, topic]

                # Rate limiting
                time.sleep(self.config.api_delay)

            except Exception as e:
                logger.warning(f"Failed to fetch data for topic '{topic}': {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(data, index=regions, columns=topics)
        df.index.name = 'region'

        return df

    def _generate_mock_data(self) -> pd.DataFrame:
        """
        Generate realistic mock data for development and testing.

        The mock data simulates regional cultural variation using a
        latent factor model:

            X[i,j] = Σ_k (L[i,k] * F[k,j]) + ε[i,j]

        where:
            - L[i,k] = loading of region i on latent factor k
            - F[k,j] = factor score for topic j on factor k
            - ε[i,j] = noise term

        We use 4 latent factors roughly corresponding to:
            1. Urban/Rural dimension
            2. North/South dimension
            3. Coastal/Interior dimension
            4. Political leaning dimension

        This generative model allows us to verify that our analysis
        can recover known latent structure.
        """
        np.random.seed(self.config.random_seed)

        regions = self.config.get_regions()
        topics = self.config.get_topics()
        n_regions = len(regions)
        n_topics = len(topics)

        # Number of latent factors in the generative model
        n_factors = 4

        # Generate region loadings on latent factors
        # These are based on real geographic/demographic patterns
        region_loadings = self._generate_region_loadings(regions, n_factors)

        # Generate topic loadings on latent factors
        topic_loadings = self._generate_topic_loadings(topics, n_factors)

        # Generate data matrix: X = L @ F.T + noise
        signal = region_loadings @ topic_loadings.T
        noise = np.random.normal(0, 0.3, (n_regions, n_topics))
        data = signal + noise

        # Transform to [0, 100] scale (like Google Trends)
        # Apply sigmoid-like transformation to keep values bounded
        data = 100 * (1 / (1 + np.exp(-data)))

        # Add some sparsity (low search volume for some combinations)
        sparsity_mask = np.random.random((n_regions, n_topics)) < 0.05
        data[sparsity_mask] = 0

        # Create DataFrame
        df = pd.DataFrame(data, index=regions, columns=topics)
        df.index.name = 'region'

        return df

    def _generate_region_loadings(
        self,
        regions: List[str],
        n_factors: int
    ) -> np.ndarray:
        """
        Generate region loadings based on real geographic patterns.

        Factor interpretations:
        - Factor 0: Urban vs Rural (positive = more urban)
        - Factor 1: South vs North (positive = more southern)
        - Factor 2: Coastal vs Interior (positive = more coastal)
        - Factor 3: Political lean (positive = more conservative)

        These are approximations based on demographic data.
        """
        n_regions = len(regions)
        loadings = np.zeros((n_regions, n_factors))

        # Define factor values for states
        # (In production, these could come from Census data)
        state_factors = {
            # State: [urban, south, coastal, conservative]
            'AL': [-0.3, 1.0, 0.2, 0.8],
            'AK': [-0.5, -1.0, 1.0, 0.4],
            'AZ': [0.3, 0.5, -0.5, 0.3],
            'AR': [-0.6, 0.8, -0.8, 0.9],
            'CA': [0.9, 0.2, 1.0, -0.8],
            'CO': [0.5, -0.2, -0.8, -0.1],
            'CT': [0.8, -0.8, 1.0, -0.5],
            'DE': [0.4, 0.0, 1.0, -0.1],
            'FL': [0.6, 0.9, 1.0, 0.2],
            'GA': [0.4, 0.9, 0.3, 0.5],
            'HI': [0.5, 0.0, 1.0, -0.7],
            'ID': [-0.4, -0.5, -0.8, 0.9],
            'IL': [0.7, -0.3, -0.3, -0.2],
            'IN': [0.2, -0.1, -0.8, 0.6],
            'IA': [-0.2, -0.3, -1.0, 0.3],
            'KS': [-0.1, 0.2, -1.0, 0.7],
            'KY': [-0.2, 0.5, -0.8, 0.7],
            'LA': [0.1, 1.0, 0.5, 0.6],
            'ME': [-0.3, -1.0, 1.0, -0.1],
            'MD': [0.8, 0.1, 1.0, -0.4],
            'MA': [0.9, -0.9, 1.0, -0.8],
            'MI': [0.5, -0.5, 0.3, 0.1],
            'MN': [0.4, -0.7, -0.5, -0.2],
            'MS': [-0.5, 1.0, 0.2, 0.9],
            'MO': [0.2, 0.2, -0.8, 0.5],
            'MT': [-0.7, -0.6, -0.9, 0.6],
            'NE': [-0.2, -0.2, -1.0, 0.7],
            'NV': [0.6, 0.1, -0.3, 0.1],
            'NH': [0.3, -0.9, 0.5, 0.0],
            'NJ': [0.95, -0.5, 1.0, -0.3],
            'NM': [-0.1, 0.4, -0.7, -0.1],
            'NY': [0.9, -0.6, 0.8, -0.6],
            'NC': [0.3, 0.7, 0.5, 0.3],
            'ND': [-0.6, -0.8, -1.0, 0.8],
            'OH': [0.4, -0.2, -0.5, 0.3],
            'OK': [-0.1, 0.6, -0.8, 0.9],
            'OR': [0.4, -0.5, 1.0, -0.3],
            'PA': [0.5, -0.4, 0.3, 0.1],
            'RI': [0.8, -0.8, 1.0, -0.5],
            'SC': [0.1, 0.9, 0.8, 0.6],
            'SD': [-0.6, -0.5, -1.0, 0.8],
            'TN': [0.2, 0.8, -0.5, 0.7],
            'TX': [0.5, 0.8, 0.4, 0.5],
            'UT': [0.3, -0.1, -0.8, 0.8],
            'VT': [-0.4, -1.0, -0.3, -0.7],
            'VA': [0.5, 0.4, 0.6, 0.2],
            'WA': [0.6, -0.6, 1.0, -0.4],
            'WV': [-0.5, 0.3, -0.7, 0.8],
            'WI': [0.3, -0.5, 0.1, 0.1],
            'WY': [-0.8, -0.3, -1.0, 0.9],
            'DC': [1.0, 0.1, 0.5, -0.9],
        }

        # Metro factor values (subset of key cities)
        metro_factors = {
            'New York': [1.0, -0.5, 1.0, -0.7],
            'Los Angeles': [0.9, 0.3, 1.0, -0.6],
            'Chicago': [0.8, -0.2, 0.2, -0.3],
            'Houston': [0.7, 0.9, 0.6, 0.4],
            'Phoenix': [0.6, 0.5, -0.5, 0.3],
            'Philadelphia': [0.8, -0.3, 0.8, -0.4],
            'San Antonio': [0.5, 0.9, 0.3, 0.3],
            'San Diego': [0.7, 0.4, 1.0, -0.2],
            'Dallas': [0.7, 0.8, -0.3, 0.4],
            'Austin': [0.6, 0.8, -0.2, -0.4],
            'San Francisco': [1.0, 0.1, 1.0, -0.9],
            'Seattle': [0.8, -0.6, 1.0, -0.6],
            'Denver': [0.7, -0.1, -0.8, -0.1],
            'Boston': [0.9, -0.9, 1.0, -0.7],
            'Nashville': [0.5, 0.8, -0.4, 0.5],
            'Detroit': [0.6, -0.4, 0.3, 0.0],
            'Portland': [0.6, -0.5, 1.0, -0.7],
            'Las Vegas': [0.7, 0.2, -0.3, 0.1],
            'Atlanta': [0.7, 0.9, 0.2, 0.2],
            'Miami': [0.8, 1.0, 1.0, -0.1],
            'Minneapolis': [0.6, -0.7, -0.3, -0.3],
            'Cleveland': [0.5, -0.3, 0.2, 0.1],
            'New Orleans': [0.4, 1.0, 0.8, 0.3],
            'Tampa': [0.6, 0.9, 1.0, 0.3],
            'Pittsburgh': [0.5, -0.3, -0.2, 0.1],
            'St. Louis': [0.5, 0.2, -0.6, 0.2],
            'Cincinnati': [0.4, 0.0, -0.5, 0.3],
            'Orlando': [0.6, 0.9, 0.9, 0.2],
            'Kansas City': [0.4, 0.1, -0.8, 0.3],
            'Salt Lake City': [0.5, -0.2, -0.8, 0.6],
            'Birmingham': [0.3, 1.0, -0.3, 0.7],
            'Oklahoma City': [0.4, 0.6, -0.8, 0.8],
        }

        # Select appropriate factor dictionary
        if self.config.geo_unit == 'state':
            factor_dict = state_factors
        else:
            factor_dict = metro_factors

        # Assign loadings
        for i, region in enumerate(regions):
            if region in factor_dict:
                loadings[i, :] = factor_dict[region]
            else:
                # Random for unknown regions
                loadings[i, :] = np.random.normal(0, 0.5, n_factors)

        # Add some noise to loadings
        loadings += np.random.normal(0, 0.1, loadings.shape)

        return loadings

    def _generate_topic_loadings(
        self,
        topics: List[str],
        n_factors: int
    ) -> np.ndarray:
        """
        Generate topic loadings on latent factors.

        Maps each topic to expected factor associations.
        """
        n_topics = len(topics)
        loadings = np.zeros((n_topics, n_factors))

        # Topic to factor mapping (approximations)
        # Factor indices: [urban, south, coastal, conservative]
        topic_factors = {
            # Music
            'country music': [-0.3, 0.6, -0.3, 0.7],
            'hip hop': [0.7, 0.3, 0.3, -0.2],
            'rock music': [0.1, -0.1, 0.0, 0.1],
            'jazz': [0.5, 0.2, 0.4, -0.2],
            'reggaeton': [0.6, 0.5, 0.7, -0.3],
            'k-pop': [0.7, -0.1, 0.6, -0.3],
            'gospel music': [-0.1, 0.8, 0.0, 0.6],
            'electronic music': [0.8, -0.2, 0.5, -0.4],

            # Slang
            "y'all": [-0.2, 0.9, 0.1, 0.5],
            'hella': [0.3, -0.3, 0.8, -0.4],
            'wicked': [0.2, -0.9, 0.6, -0.3],
            'pop vs soda': [0.0, -0.3, 0.0, 0.1],
            'fixin to': [-0.3, 0.9, -0.2, 0.6],

            # Food
            'sweet tea': [-0.2, 0.9, 0.2, 0.5],
            'craft beer': [0.6, -0.3, 0.4, -0.4],
            'boba tea': [0.8, -0.1, 0.7, -0.5],
            'kolache': [0.0, 0.5, -0.2, 0.3],
            'barbecue': [0.0, 0.6, -0.2, 0.4],
            'vegan food': [0.8, -0.2, 0.6, -0.7],
            'soul food': [0.3, 0.7, 0.2, 0.1],

            # Sports
            'NFL football': [0.1, 0.2, 0.0, 0.2],
            'college football': [-0.1, 0.7, -0.1, 0.5],
            'NBA basketball': [0.5, 0.2, 0.2, -0.1],
            'hockey NHL': [0.3, -0.8, 0.2, 0.0],
            'soccer MLS': [0.6, 0.0, 0.5, -0.3],
            'rodeo': [-0.5, 0.5, -0.5, 0.8],
            'surfing': [0.4, 0.2, 1.0, -0.2],
            'skiing': [0.3, -0.6, 0.3, 0.0],

            # Lifestyle
            'church near me': [-0.2, 0.6, -0.1, 0.7],
            'yoga classes': [0.7, -0.1, 0.5, -0.6],
            'gun range': [-0.3, 0.4, -0.4, 0.8],
            'farmers market': [0.4, -0.1, 0.3, -0.3],
            'camping gear': [-0.1, -0.2, -0.2, 0.2],
            'beach vacation': [0.3, 0.3, 0.8, 0.0],
            'hunting license': [-0.6, 0.3, -0.5, 0.8],

            # Media
            'Fox News': [-0.2, 0.3, -0.2, 0.9],
            'NPR': [0.7, -0.3, 0.4, -0.8],
            'Joe Rogan': [0.2, 0.1, 0.1, 0.3],
            'true crime podcast': [0.4, -0.1, 0.2, -0.1],
            'reality TV': [0.1, 0.2, 0.1, 0.0],

            # Political
            'voter registration': [0.3, 0.1, 0.1, -0.1],
            'local news': [0.0, 0.0, 0.0, 0.0],
            'community events': [0.1, -0.1, 0.0, -0.1],
        }

        # Assign loadings
        for j, topic in enumerate(topics):
            if topic in topic_factors:
                loadings[j, :] = topic_factors[topic]
            else:
                # Random for unknown topics
                loadings[j, :] = np.random.normal(0, 0.3, n_factors)

        # Add noise
        loadings += np.random.normal(0, 0.1, loadings.shape)

        return loadings


def collect_trends_data(
    config: AnalysisConfig = default_config,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convenience function to collect Google Trends data.

    Args:
        config: Analysis configuration
        save_path: Optional path to save the raw data

    Returns:
        Region × topic DataFrame of search interest values
    """
    collector = GoogleTrendsCollector(config)
    data = collector.collect_data()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path)
        logger.info(f"Saved raw data to {save_path}")

    return data


def load_cached_data(path: Path) -> pd.DataFrame:
    """Load previously collected data from CSV."""
    return pd.read_csv(path, index_col=0)
