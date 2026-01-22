# CLAUDE.md - AI Assistant Guide for Cultural Similarity Analysis

## Project Overview

This repository implements an **empirical social science analysis** that uses Google Trends data to infer latent "cultural similarity" between U.S. regions. The goal is to recover a low-dimensional cultural space where regions can be compared and clustered.

**Research Question**: Can we use search behavior to identify distinct "cultural modes" across U.S. states/metros and make statements like "Miami is culturally closer to NYC than Atlanta"?

## Repository Structure

```
google-trends-culture/
├── src/                      # Core analysis modules
│   ├── __init__.py          # Package documentation with mathematical framework
│   ├── config.py            # Configuration and geographic data
│   ├── data_collection.py   # Google Trends API + mock data generation
│   ├── preprocessing.py     # Normalization and data cleaning
│   ├── similarity.py        # Similarity matrix computation
│   ├── spectral_analysis.py # PCA / eigenvalue decomposition
│   ├── clustering.py        # GMM, spectral, K-means clustering
│   └── visualization.py     # Plotting functions
├── run_analysis.py          # Main analysis pipeline
├── requirements.txt         # Python dependencies
├── data/                    # Data storage (created at runtime)
├── outputs/                 # Generated figures and results
└── notebooks/               # Jupyter notebooks (optional)
```

## Mathematical Framework

### Data Matrix
- **X** ∈ ℝ^(n×p): region × topic matrix
  - n = number of regions (states or metros)
  - p = number of search topics
  - X[i,j] = normalized search intensity for region i on topic j

### Similarity Matrix
- **S** ∈ ℝ^(n×n): pairwise regional similarity
  - S[i,j] = correlation(X[i,:], X[j,:]) or cosine similarity
  - Symmetric, positive semi-definite

### Spectral Decomposition
- **S = VΛV^T** where:
  - V = eigenvectors (cultural modes)
  - Λ = diagonal eigenvalues (variance explained)
- Each eigenvalue represents a "cultural dimension"
- The scree plot reveals how many dimensions matter

### Embedding
- Regions embedded as: E[i,:] = V[i,:] @ diag(√λ)
- Euclidean distance in embedding ≈ cultural dissimilarity

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with mock data (default)
python run_analysis.py

# Run with real Google Trends API
python run_analysis.py --real-data

# Specify clustering parameters
python run_analysis.py --n-clusters 5 --clustering gmm
```

## Key Modules

### 1. Data Collection (`src/data_collection.py`)

**Purpose**: Fetch Google Trends data or generate realistic mock data.

**Key Functions**:
- `collect_trends_data(config)` - Main entry point
- `GoogleTrendsCollector._generate_mock_data()` - Generates data using a latent factor model

**Mock Data Generation**:
The mock data uses a known generative model with 4 latent factors:
1. Urban/Rural dimension
2. North/South dimension
3. Coastal/Interior dimension
4. Political leaning dimension

This allows validation that the analysis recovers true structure.

### 2. Preprocessing (`src/preprocessing.py`)

**Purpose**: Normalize and clean the data matrix.

**Key Functions**:
- `preprocess_data(data, config)` - Main preprocessing pipeline
- Normalization options: `zscore` (recommended), `minmax`, `robust`
- Missing value handling: `drop`, `impute_mean`, `impute_zero`

**Recommendation**: Use z-score normalization for correlation-based similarity.

### 3. Similarity (`src/similarity.py`)

**Purpose**: Compute pairwise cultural similarity.

**Key Functions**:
- `compute_similarity_matrix(data, metric)` - Main computation
- `cultural_distance(S, a, b, c)` - Compare three regions
- `get_most_similar_regions(S, region, n)` - Find similar regions

**Metrics**:
- `correlation`: Pearson correlation (recommended)
- `cosine`: Cosine similarity
- `euclidean`: Negative Euclidean distance

### 4. Spectral Analysis (`src/spectral_analysis.py`)

**Purpose**: Eigenvalue decomposition to find latent cultural dimensions.

**Key Functions**:
- `spectral_decomposition(S, config)` - Returns `SpectralResult`
- `scree_analysis(result)` - Data for scree plot
- `interpret_cultural_modes(result, data)` - Explain what each mode means

**Component Selection**:
- Kaiser criterion: eigenvalue > 1
- Variance threshold: cumulative variance > X%
- Elbow detection in scree plot

### 5. Clustering (`src/clustering.py`)

**Purpose**: Identify distinct cultural groups.

**Key Functions**:
- `cluster_regions(spectral_result, similarity_matrix, config)` - Main clustering
- `analyze_clusters(result, spectral, data)` - Interpret clusters
- `compare_clusters(a, b, result, data)` - Contrast two clusters

**Algorithms**:
- `gmm`: Gaussian Mixture Model (soft clustering, recommended)
- `spectral`: Spectral clustering (graph-based)
- `kmeans`: K-means (baseline)

### 6. Visualization (`src/visualization.py`)

**Purpose**: Generate publication-quality figures.

**Key Functions**:
- `plot_scree(spectral_result)` - Eigenvalue decay plot
- `plot_embedding_2d(spectral_result, clustering)` - 2D cultural space
- `plot_similarity_heatmap(S, clustering)` - Similarity matrix
- `plot_mode_interpretation(spectral, data, mode_index)` - Mode meaning
- `create_all_visualizations(...)` - Generate all standard plots

## Configuration Options

Edit `src/config.py` or pass arguments to `run_analysis.py`:

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `use_mock_data` | bool | True | Use mock vs real API |
| `geo_unit` | state, metro | state | Geographic unit |
| `similarity_metric` | correlation, cosine, euclidean | correlation | Similarity measure |
| `normalization` | zscore, minmax, robust | zscore | Normalization method |
| `clustering_method` | gmm, spectral, kmeans | gmm | Clustering algorithm |
| `n_clusters` | int or None | None | Number of clusters (auto if None) |
| `component_criterion` | kaiser, variance, elbow | kaiser | How to select components |

## Common Tasks for AI Assistants

### 1. Run the Analysis
```python
python run_analysis.py --no-plots  # Skip visualization
```

### 2. Compare Specific Regions
```python
from src.similarity import cultural_distance, compute_similarity_matrix
from src.data_collection import collect_trends_data
from src.preprocessing import preprocess_data
from src.config import AnalysisConfig

config = AnalysisConfig()
raw = collect_trends_data(config)
data, _ = preprocess_data(raw, config)
S = compute_similarity_matrix(data)

# Is Florida closer to New York or Georgia?
result = cultural_distance(S, 'FL', 'NY', 'GA')
print(result['interpretation'])
```

### 3. Add New Topics
Edit `DEFAULT_TOPICS` in `src/config.py`:
```python
DEFAULT_TOPICS = {
    'music': [...],
    'your_category': [
        'new topic 1',
        'new topic 2',
    ],
    ...
}
```

### 4. Interpret Cultural Modes
```python
from src.spectral_analysis import interpret_cultural_modes

interpretations = interpret_cultural_modes(spectral_result, data, n_modes=4)
for mode, info in interpretations.items():
    print(f"{mode}: High={info['high_loading_regions'][:3]}")
```

### 5. Get Cluster Membership
```python
cluster_df = clustering_result.get_cluster_df()
# Includes hard labels and soft probabilities (for GMM)
```

## Output Files

After running `python run_analysis.py`:

| File | Description |
|------|-------------|
| `outputs/raw_data.csv` | Original Google Trends values |
| `outputs/processed_data.csv` | Normalized data matrix |
| `outputs/similarity_matrix.csv` | n×n similarity matrix |
| `outputs/embedding.csv` | Low-dimensional coordinates |
| `outputs/clusters.csv` | Cluster assignments |
| `outputs/scree_plot.png` | Eigenvalue decay |
| `outputs/embedding_2d.png` | 2D cultural space |
| `outputs/similarity_heatmap.png` | Similarity visualization |

## Testing and Validation

The mock data generator uses known latent factors, allowing validation:
- True clusters should be recoverable
- True dimensions should appear in top eigenvalues
- Known similar regions should cluster together

To verify:
```python
# Check if true factors are recovered
# States with similar factor profiles should have high similarity
```

## Extending the Analysis

### Add Geographic Visualization
```python
# Requires geopandas
import geopandas as gpd

# Load US states shapefile
# Merge with cluster assignments
# Create choropleth map
```

### Use Real Google Trends Data
```bash
pip install pytrends
python run_analysis.py --real-data
```
Note: Real API has rate limits (~1 request/second).

### Different Time Periods
```python
config = AnalysisConfig(timeframe='2020-01-01 2020-12-31')
```

## Code Conventions

- **Docstrings**: All functions have detailed docstrings with mathematical explanations
- **Type hints**: Function signatures include type annotations
- **Logging**: Use `logger.info()` for progress, `logger.warning()` for issues
- **Config**: All parameters are centralized in `AnalysisConfig`
- **DataFrames**: Prefer pandas DataFrames with meaningful indices/columns

## Interpreting Results

### Scree Plot
- Steep decline followed by flat tail = clear structure
- Eigenvalue > 1 = meaningful component (Kaiser)
- Cumulative variance tells how much is explained

### 2D Embedding
- Nearby points = culturally similar regions
- Clusters visible = distinct cultural groups
- Axes = first two cultural dimensions

### Similarity Matrix
- Red = high similarity, Blue = low similarity
- Block structure = cultural clusters
- Off-diagonal patterns = cross-cluster relationships

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pytrends` rate limit | Add delay: `config.api_delay = 2.0` |
| Too few topics | Lower `min_volume_threshold` |
| Singular matrix | Check for constant columns in data |
| Poor clustering | Try different `n_clusters` or method |
| Memory error | Reduce number of topics or regions |

## References

- Google Trends: https://trends.google.com
- pytrends: https://github.com/GeneralMills/pytrends
- Spectral methods: von Luxburg (2007) "A Tutorial on Spectral Clustering"
- Factor analysis: Costello & Osborne (2005) "Best Practices in Exploratory Factor Analysis"
