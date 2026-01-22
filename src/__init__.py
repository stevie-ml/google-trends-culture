"""
Google Trends Cultural Similarity Analysis

A quantitative social science toolkit for inferring latent cultural similarity
between U.S. regions using Google search behavior data.

Mathematical Framework:
-----------------------
Let X ∈ ℝ^(n×p) be the region × topic matrix where:
  - n = number of regions (states or DMAs)
  - p = number of search topics
  - X[i,j] = normalized search intensity for region i on topic j

The cultural similarity matrix S ∈ ℝ^(n×n) is computed as:
  S[i,j] = sim(X[i,:], X[j,:])

where sim() is either Pearson correlation or cosine similarity.

Spectral decomposition of S reveals latent cultural dimensions:
  S = VΛV^T

where the columns of V are the eigenvectors (cultural modes) and
the diagonal of Λ contains eigenvalues (variance explained by each mode).
"""

__version__ = "0.1.0"
__author__ = "Cultural Analytics Research"
