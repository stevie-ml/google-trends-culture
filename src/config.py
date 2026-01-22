"""
Configuration settings for the cultural similarity analysis.

This module defines all tunable parameters for the analysis pipeline,
organized by analysis stage. Defaults are chosen for interpretability
and robustness in social science applications.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from pathlib import Path

# =============================================================================
# Geographic Units
# =============================================================================

# U.S. State codes for analysis
US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

# Full state names for display
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'Washington DC'
}

# Major metro areas with their Google Trends DMA codes
# DMA = Designated Market Area (Nielsen)
MAJOR_METROS = {
    'New York': '501',
    'Los Angeles': '803',
    'Chicago': '602',
    'Houston': '618',
    'Phoenix': '753',
    'Philadelphia': '504',
    'San Antonio': '641',
    'San Diego': '825',
    'Dallas': '623',
    'San Jose': '807',
    'Austin': '635',
    'Jacksonville': '561',
    'San Francisco': '807',
    'Columbus': '535',
    'Indianapolis': '527',
    'Charlotte': '517',
    'Seattle': '819',
    'Denver': '751',
    'Boston': '506',
    'Nashville': '659',
    'Detroit': '505',
    'Portland': '820',
    'Las Vegas': '839',
    'Memphis': '640',
    'Louisville': '529',
    'Baltimore': '512',
    'Milwaukee': '617',
    'Albuquerque': '790',
    'Tucson': '789',
    'Fresno': '866',
    'Sacramento': '862',
    'Atlanta': '524',
    'Miami': '528',
    'Minneapolis': '613',
    'Cleveland': '510',
    'New Orleans': '622',
    'Tampa': '539',
    'Pittsburgh': '508',
    'St. Louis': '609',
    'Cincinnati': '515',
    'Orlando': '534',
    'Kansas City': '616',
    'Raleigh': '560',
    'Salt Lake City': '770',
    'Richmond': '556',
    'Hartford': '533',
    'Birmingham': '630',
    'Oklahoma City': '650'
}

# =============================================================================
# Search Topics for Cultural Analysis
# =============================================================================

# Topics chosen to span multiple cultural dimensions:
# - Music preferences (genre, artists)
# - Political/social attitudes
# - Regional slang and language
# - Pop culture consumption
# - Food and lifestyle
# - Sports preferences

DEFAULT_TOPICS = {
    # Music genres and artists (cultural taste)
    'music': [
        'country music',
        'hip hop',
        'rock music',
        'jazz',
        'reggaeton',
        'k-pop',
        'gospel music',
        'electronic music',
    ],

    # Regional slang and expressions
    'slang': [
        'y\'all',
        'hella',
        'wicked',
        'pop vs soda',
        'fixin to',
    ],

    # Food and drink preferences
    'food': [
        'sweet tea',
        'craft beer',
        'boba tea',
        'kolache',
        'barbecue',
        'vegan food',
        'soul food',
    ],

    # Sports and recreation
    'sports': [
        'NFL football',
        'college football',
        'NBA basketball',
        'hockey NHL',
        'soccer MLS',
        'rodeo',
        'surfing',
        'skiing',
    ],

    # Lifestyle and values
    'lifestyle': [
        'church near me',
        'yoga classes',
        'gun range',
        'farmers market',
        'camping gear',
        'beach vacation',
        'hunting license',
    ],

    # Pop culture and media
    'media': [
        'Fox News',
        'NPR',
        'Joe Rogan',
        'true crime podcast',
        'reality TV',
    ],

    # Political and social (handled carefully for research purposes)
    'political': [
        'voter registration',
        'local news',
        'community events',
    ],
}


@dataclass
class AnalysisConfig:
    """
    Main configuration class for cultural similarity analysis.

    All parameters can be adjusted for sensitivity analysis or
    to adapt the methodology to different research questions.
    """

    # -------------------------------------------------------------------------
    # Data Collection Settings
    # -------------------------------------------------------------------------

    # Use mock data (True) or real Google Trends API (False)
    use_mock_data: bool = True

    # Geographic unit: 'state' or 'metro'
    geo_unit: Literal['state', 'metro'] = 'state'

    # Time frame for Google Trends queries
    # Format: 'YYYY-MM-DD YYYY-MM-DD' or relative like 'today 12-m'
    timeframe: str = 'today 12-m'

    # Topic categories to include (keys from DEFAULT_TOPICS)
    topic_categories: List[str] = field(default_factory=lambda: [
        'music', 'slang', 'food', 'sports', 'lifestyle', 'media'
    ])

    # Rate limiting for API calls (seconds between requests)
    api_delay: float = 1.0

    # -------------------------------------------------------------------------
    # Preprocessing Settings
    # -------------------------------------------------------------------------

    # Normalization method for the region × topic matrix
    # 'zscore': standardize to mean=0, std=1 (recommended)
    # 'minmax': scale to [0, 1] range
    # 'robust': use median and IQR (robust to outliers)
    normalization: Literal['zscore', 'minmax', 'robust'] = 'zscore'

    # Handle missing values
    # 'drop': remove regions/topics with missing data
    # 'impute_mean': fill with column (topic) mean
    # 'impute_zero': fill with zero (implies no search interest)
    missing_strategy: Literal['drop', 'impute_mean', 'impute_zero'] = 'impute_mean'

    # Minimum search volume threshold (0-100 scale)
    # Topics with max < threshold across all regions are dropped
    min_volume_threshold: float = 5.0

    # -------------------------------------------------------------------------
    # Similarity Matrix Settings
    # -------------------------------------------------------------------------

    # Similarity metric for comparing regions
    # 'correlation': Pearson correlation (captures linear relationships)
    # 'cosine': Cosine similarity (captures directional similarity)
    # 'euclidean': Negative Euclidean distance (for completeness)
    similarity_metric: Literal['correlation', 'cosine', 'euclidean'] = 'correlation'

    # -------------------------------------------------------------------------
    # Spectral Analysis Settings
    # -------------------------------------------------------------------------

    # Number of principal components to retain for embedding
    # If None, determined by scree plot / Kaiser criterion
    n_components: Optional[int] = None

    # Criterion for automatic component selection
    # 'kaiser': eigenvalue > 1 (for correlation matrix)
    # 'variance': cumulative variance > variance_threshold
    # 'elbow': elbow detection in scree plot
    component_criterion: Literal['kaiser', 'variance', 'elbow'] = 'kaiser'

    # Cumulative variance threshold (if criterion='variance')
    variance_threshold: float = 0.80

    # -------------------------------------------------------------------------
    # Clustering Settings
    # -------------------------------------------------------------------------

    # Clustering algorithm
    # 'gmm': Gaussian Mixture Model (soft clustering, probabilistic)
    # 'spectral': Spectral clustering (graph-based)
    # 'kmeans': K-means (for comparison)
    clustering_method: Literal['gmm', 'spectral', 'kmeans'] = 'gmm'

    # Number of clusters (if None, selected by BIC/silhouette)
    n_clusters: Optional[int] = None

    # Range to search for optimal cluster count
    cluster_range: tuple = (2, 10)

    # -------------------------------------------------------------------------
    # Output Settings
    # -------------------------------------------------------------------------

    # Random seed for reproducibility
    random_seed: int = 42

    # Output directory
    output_dir: Path = Path('outputs')

    # Figure settings
    figure_dpi: int = 150
    figure_format: str = 'png'

    def get_topics(self) -> List[str]:
        """Return flat list of all topics from selected categories."""
        topics = []
        for category in self.topic_categories:
            if category in DEFAULT_TOPICS:
                topics.extend(DEFAULT_TOPICS[category])
        return topics

    def get_regions(self) -> List[str]:
        """Return list of regions based on geo_unit setting."""
        if self.geo_unit == 'state':
            return US_STATES
        else:
            return list(MAJOR_METROS.keys())


# Default configuration instance
default_config = AnalysisConfig()
