"""Long-term fairness in ride-hailing: gap implementation.

Extends Kang et al., "Long-term Fairness in Ride-Hailing Platform"
(ECML PKDD 2024, arXiv:2407.17839) with:

  Gap 1a  time/traffic-aware utility (their utility is distance-only and
          explicitly time-invariant, yet the same OD pair takes 2.42x longer
          at peak than off-peak in the data they use)
  Gap 1b  hourly-rate fairness with a within/between group decomposition
          (their Var(total earnings) is blind to hours worked)
  Gap 2   utilisation / access fairness (they never check whether an
          available driver was actually given work)
"""
from .config import Config, default_config

__all__ = ["Config", "default_config"]
__version__ = "0.1.0"
