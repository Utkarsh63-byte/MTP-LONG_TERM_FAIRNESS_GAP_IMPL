"""Allocation methods: the paper's baselines and the gap-aware variants."""
from .base import Policy, VarianceTracker, auto_omega
from .greedy import ScalarisedGreedy, make_paper_greedy, make_efficiency_only
from .classical import ReassignPolicy, LAFPolicy
from .momaql import (MOMAQL, make_balance_ride_pooling, make_paper_method,
                     make_gap_method)

__all__ = ["Policy", "VarianceTracker", "auto_omega", "ScalarisedGreedy",
           "make_paper_greedy", "make_efficiency_only", "ReassignPolicy",
           "LAFPolicy", "MOMAQL", "make_balance_ride_pooling",
           "make_paper_method", "make_gap_method"]
