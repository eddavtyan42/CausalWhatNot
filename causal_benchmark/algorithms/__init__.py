"""Convenience imports for algorithm modules."""

from . import pc
from . import ges
from . import cosmo

try:
    from . import notears  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    notears = None

__all__ = ["pc", "ges", "cosmo", "notears"]
