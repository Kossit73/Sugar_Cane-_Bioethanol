"""Utility helpers for optional third-party dependencies.

The application prefers to use :mod:`matplotlib` (and optionally other plotting
libraries) when available, but the execution environment may not permit network
access to install them.  These helpers provide a tiny caching layer around
import attempts so callers can determine whether a dependency is present and, if
not, surface a clear message instructing the user to install it manually.

Unlike a traditional "auto-install" helper, this module intentionally **does not
attempt to invoke ``pip``**.  Repeated installation attempts can be slow and are
destined to fail in locked-down sandboxes, so we simply record the first import
failure and reuse that information for subsequent checks.
"""
from __future__ import annotations

import importlib
from typing import Dict, Mapping, Optional

# Cache import/installation attempts so we only try once per interpreter run.
_ATTEMPTS: Dict[str, bool] = {}
_ERRORS: Dict[str, str] = {}


def _try_import(module_name: str) -> bool:
    """Return ``True`` if *module_name* can be imported, ``False`` otherwise."""

    try:
        importlib.import_module(module_name)
        return True
    except Exception as exc:  # pragma: no cover - error captured by caller
        _ERRORS[module_name] = str(exc)
        return False


def ensure_package(module_name: str, package_name: Optional[str] = None) -> bool:
    """Return ``True`` if ``module_name`` can be imported.

    When the import fails we remember the error message and instruct callers to
    install the package manually.  Passing ``package_name`` allows the helper to
    mention a friendlier distribution name (for example ``"matplotlib"`` instead
    of ``"matplotlib.pyplot"``).
    """

    if module_name in _ATTEMPTS:
        return _ATTEMPTS[module_name]

    if _try_import(module_name):
        _ATTEMPTS[module_name] = True
        _ERRORS.pop(module_name, None)
        return True

    package = package_name or module_name.split(".")[0]
    _ATTEMPTS[module_name] = False
    _ERRORS[module_name] = (
        f"Optional dependency '{package}' is not installed. Install it via 'pip install {package}' "
        "and reload the application."
    )
    return False


def ensure_packages(mapping: Mapping[str, Optional[str]]) -> Dict[str, bool]:
    """Ensure multiple packages are available, returning a result dictionary."""

    return {module: ensure_package(module, pkg) for module, pkg in mapping.items()}


def get_package_error(module_name: str) -> Optional[str]:
    """Return the stored installation/import error for ``module_name``."""

    return _ERRORS.get(module_name)
