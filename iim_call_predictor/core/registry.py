"""Plug-in registry: colleges self-register with ``@register_college(...)``.

Adding a new IIM is: create ``colleges/<code>/`` with a ``config.yaml`` and a
``model.py`` that subclasses ``CollegeModel`` and decorates itself with
``@register_college("<code>")``, then import that module from
``colleges/__init__.py``. Nothing in ``core`` needs to change.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import CollegeModel

_REGISTRY: Dict[str, Type[CollegeModel]] = {}


def register_college(code: str):
    """Class decorator that registers a ``CollegeModel`` subclass under ``code``."""

    def decorator(cls: Type[CollegeModel]) -> Type[CollegeModel]:
        if code in _REGISTRY and _REGISTRY[code] is not cls:
            raise ValueError(f"A college is already registered under code '{code}'.")
        cls.name = code
        _REGISTRY[code] = cls
        return cls

    return decorator


def get_college(code: str) -> CollegeModel:
    """Instantiate and return the college model registered under ``code``."""
    try:
        cls = _REGISTRY[code]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown college code '{code}'. Available: {available}") from exc
    return cls()


def list_colleges() -> List[str]:
    """Return all registered college codes."""
    return sorted(_REGISTRY)
