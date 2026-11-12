"""Exact temporal-balance profiles for the fixed descriptive contrast design.

The public ``analyse`` function accepts outcome-free exact-weight records. It
returns deterministic basis responses, not estimated effects or confidence
intervals. The original design identifiers are treated as opaque labels.
"""
from .weights import analyse

__all__ = ["analyse"]
