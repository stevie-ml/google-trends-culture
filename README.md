# google-trends-culture

Inferring latent cultural similarity between U.S. regions from Google Trends search behavior. The pipeline builds search-interest vectors by state/metro, computes similarity matrices, and uses PCA and spectral clustering to recover a low-dimensional "cultural space" — supporting claims like "Miami is culturally closer to NYC than to Atlanta."

Pipeline in `src/`: data collection (Trends API), normalization, similarity computation, spectral analysis, clustering. Run with `run_analysis.py`.
