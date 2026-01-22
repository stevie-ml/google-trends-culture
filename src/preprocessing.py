"""
Preprocessing Module for Cultural Similarity Analysis

This module handles normalization and cleaning of the region × topic matrix.
Proper preprocessing is critical for valid downstream analysis.

Mathematical Background:
------------------------
The raw data matrix X_raw contains Google Trends values in [0, 100].
Different normalization strategies affect interpretation:

1. Z-score normalization (recommended):
   X[i,j] = (X_raw[i,j] - μ_j) / σ_j

   This standardizes each topic to mean=0, std=1, which is appropriate
   when computing correlation-based similarity. It ensures that all
   topics contribute equally regardless of their baseline popularity.

2. Min-max normalization:
   X[i,j] = (X_raw[i,j] - min_j) / (max_j - min_j)

   Scales to [0, 1]. Preserves the relative magnitudes within each topic
   but can be sensitive to outliers.

3. Robust normalization:
   X[i,j] = (X_raw[i,j] - median_j) / IQR_j

   Uses median and interquartile range. Robust to outliers but may
   compress the distribution for normally-distributed data.

For cultural analysis, z-score normalization is recommended because:
- Correlation between regions is scale-invariant
- Topics with different baseline popularity are equally weighted
- Standard PCA assumes centered data
"""

import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig, default_config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for the region × topic matrix.

    Handles missing values, normalization, and quality filtering.
    """

    def __init__(self, config: AnalysisConfig = default_config):
        self.config = config
        self._normalization_params = None

    def preprocess(
        self,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Main preprocessing pipeline.

        Args:
            data: Raw region × topic DataFrame
            verbose: Whether to log preprocessing statistics

        Returns:
            Tuple of (processed DataFrame, metadata dict)
        """
        metadata = {
            'original_shape': data.shape,
            'regions_dropped': [],
            'topics_dropped': [],
            'missing_before': data.isna().sum().sum(),
            'normalization': self.config.normalization,
        }

        if verbose:
            logger.info(f"Preprocessing data with shape {data.shape}")
            logger.info(f"Missing values: {metadata['missing_before']}")

        # Step 1: Handle low-volume topics
        data, dropped_topics = self._filter_low_volume(data)
        metadata['topics_dropped'] = dropped_topics

        # Step 2: Handle missing values
        data, _ = self._handle_missing(data)
        metadata['missing_after'] = data.isna().sum().sum()

        # Step 3: Normalize the data
        data, norm_params = self._normalize(data)
        self._normalization_params = norm_params
        metadata['normalization_params'] = norm_params

        # Step 4: Final quality checks
        data, dropped_regions = self._quality_filter(data)
        metadata['regions_dropped'] = dropped_regions

        metadata['final_shape'] = data.shape

        if verbose:
            logger.info(f"Final data shape: {data.shape}")
            logger.info(f"Topics dropped: {len(dropped_topics)}")
            logger.info(f"Regions dropped: {len(dropped_regions)}")

        return data, metadata

    def _filter_low_volume(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove topics with insufficient search volume.

        Topics where max(search_interest) < threshold across all regions
        are dropped, as they don't provide meaningful variation.

        Mathematical justification:
        - Topics with very low volume have high measurement error
        - Near-zero values may represent missing data disguised as zeros
        - Including them would add noise without cultural signal
        """
        threshold = self.config.min_volume_threshold
        max_values = data.max(axis=0)
        keep_mask = max_values >= threshold
        dropped = list(data.columns[~keep_mask])

        if dropped:
            logger.info(f"Dropping {len(dropped)} low-volume topics: {dropped[:5]}...")

        return data.loc[:, keep_mask], dropped

    def _handle_missing(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Handle missing values according to configuration.

        Strategies:
        - 'drop': Remove rows/columns with any missing values
        - 'impute_mean': Fill with column (topic) mean
        - 'impute_zero': Fill with zero

        Mathematical note:
        Mean imputation preserves column means but reduces variance
        and can distort correlations. For cultural analysis, this is
        usually acceptable as missing data typically indicates low
        search volume (which is itself informative).
        """
        strategy = self.config.missing_strategy
        stats = {
            'missing_by_region': data.isna().sum(axis=1).to_dict(),
            'missing_by_topic': data.isna().sum(axis=0).to_dict(),
        }

        if strategy == 'drop':
            # Drop regions with any missing data
            data = data.dropna(axis=0, how='any')
            # Drop topics with any missing data
            data = data.dropna(axis=1, how='any')

        elif strategy == 'impute_mean':
            # Impute with column (topic) means
            for col in data.columns:
                if data[col].isna().any():
                    col_mean = data[col].mean()
                    data[col] = data[col].fillna(col_mean)

        elif strategy == 'impute_zero':
            data = data.fillna(0)

        return data, stats

    def _normalize(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize the data matrix.

        Returns normalized data and parameters needed to inverse-transform.
        """
        method = self.config.normalization
        params = {'method': method}

        if method == 'zscore':
            # Z-score normalization: (x - μ) / σ
            means = data.mean(axis=0)
            stds = data.std(axis=0)

            # Avoid division by zero for constant columns
            stds = stds.replace(0, 1)

            normalized = (data - means) / stds
            params['means'] = means.to_dict()
            params['stds'] = stds.to_dict()

        elif method == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            ranges = maxs - mins

            # Avoid division by zero
            ranges = ranges.replace(0, 1)

            normalized = (data - mins) / ranges
            params['mins'] = mins.to_dict()
            params['maxs'] = maxs.to_dict()

        elif method == 'robust':
            # Robust normalization: (x - median) / IQR
            medians = data.median(axis=0)
            q1 = data.quantile(0.25, axis=0)
            q3 = data.quantile(0.75, axis=0)
            iqrs = q3 - q1

            # Avoid division by zero
            iqrs = iqrs.replace(0, 1)

            normalized = (data - medians) / iqrs
            params['medians'] = medians.to_dict()
            params['iqrs'] = iqrs.to_dict()

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, params

    def _quality_filter(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Final quality filtering.

        Remove regions that have extreme values after normalization,
        which may indicate data quality issues.
        """
        # Check for regions with extreme values (|z| > 5)
        extreme_mask = (data.abs() > 5).any(axis=1)
        dropped = list(data.index[extreme_mask])

        if dropped:
            logger.warning(f"Dropping {len(dropped)} regions with extreme values")
            data = data.loc[~extreme_mask]

        return data, dropped

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse the normalization to recover original scale.

        Useful for interpretation of specific values.
        """
        if self._normalization_params is None:
            raise ValueError("Must call preprocess() before inverse_transform()")

        params = self._normalization_params
        method = params['method']

        if method == 'zscore':
            means = pd.Series(params['means'])
            stds = pd.Series(params['stds'])
            return data * stds + means

        elif method == 'minmax':
            mins = pd.Series(params['mins'])
            maxs = pd.Series(params['maxs'])
            return data * (maxs - mins) + mins

        elif method == 'robust':
            medians = pd.Series(params['medians'])
            iqrs = pd.Series(params['iqrs'])
            return data * iqrs + medians

        else:
            raise ValueError(f"Unknown method: {method}")


def compute_data_quality_report(data: pd.DataFrame) -> dict:
    """
    Generate a quality report for the data matrix.

    Returns statistics useful for diagnosing data issues.
    """
    report = {
        'shape': data.shape,
        'n_regions': data.shape[0],
        'n_topics': data.shape[1],

        # Missing data
        'total_missing': data.isna().sum().sum(),
        'pct_missing': 100 * data.isna().sum().sum() / data.size,

        # Distribution statistics
        'global_mean': data.values.mean(),
        'global_std': data.values.std(),
        'global_min': data.values.min(),
        'global_max': data.values.max(),

        # Per-region statistics
        'region_means': data.mean(axis=1).describe().to_dict(),
        'region_stds': data.std(axis=1).describe().to_dict(),

        # Per-topic statistics
        'topic_means': data.mean(axis=0).describe().to_dict(),
        'topic_stds': data.std(axis=0).describe().to_dict(),

        # Sparsity (fraction of near-zero values)
        'sparsity': (data.abs() < 1e-6).sum().sum() / data.size,
    }

    # Check for potential issues
    report['issues'] = []

    if report['pct_missing'] > 10:
        report['issues'].append('High missing data rate (>10%)')

    if report['sparsity'] > 0.3:
        report['issues'].append('High sparsity (>30% near-zero values)')

    if report['global_std'] < 0.1:
        report['issues'].append('Low variance in data')

    return report


def preprocess_data(
    data: pd.DataFrame,
    config: AnalysisConfig = default_config
) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function for preprocessing.

    Args:
        data: Raw region × topic DataFrame
        config: Analysis configuration

    Returns:
        Tuple of (processed DataFrame, metadata dict)
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.preprocess(data)
