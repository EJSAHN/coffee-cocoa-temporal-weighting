"""Exact temporal diagnostics for linear, observation-weighted contrasts."""

from .core import DiagnosticInputError, diagnose
from .report import write_report

__all__ = ["DiagnosticInputError", "diagnose", "write_report"]
__version__ = "1.0.0"
