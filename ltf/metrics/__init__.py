"""Metrics layer: the paper's fairness measures plus the Gap 1b / Gap 2 ones."""
from .fairness import (Metrics, compute, decompose_variance, gini, cv,
                       opportunity_normalised_utilisation, check_decomposition)

__all__ = ["Metrics", "compute", "decompose_variance", "gini", "cv",
           "opportunity_normalised_utilisation", "check_decomposition"]
