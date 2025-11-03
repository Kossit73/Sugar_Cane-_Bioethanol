"""Utility helpers for ensuring optional third-party packages are available.

The model prefers to use matplotlib and plotly for chart generation. These
packages may be absent in the execution environment, so this module provides
lightweight wrappers that try to import the requested modules and, if that
fails, attempt a best-effort installation via ``pip``. Results are cached to
avoid repeatedly invoking the package manager on subsequent imports.

The helpers are intentionally defensive: installation failures are captured and
stored so the caller can surface user-friendly diagnostics without raising
exceptions during normal execution.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
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
    """Ensure ``module_name`` can be imported, attempting ``pip install`` if
    necessary.

    Parameters
    ----------
    module_name:
        Fully-qualified module name to import (e.g. ``"matplotlib"`` or
        ``"matplotlib.pyplot"``).
    package_name:
        Optional pip package name to install if the import fails. Defaults to
        the top-level portion of ``module_name``.

    Returns
    -------
    bool
        ``True`` when the module is available after the call, otherwise
        ``False``. Installation errors are stored and retrievable via
        :func:`get_package_error`.
    """

    if module_name in _ATTEMPTS:
        if _ATTEMPTS[module_name]:
            return True
        if _try_import(module_name):  # Allow re-checks after manual installations.
            _ATTEMPTS[module_name] = True
            _ERRORS.pop(module_name, None)
            return True
        return False

    if _try_import(module_name):
        _ATTEMPTS[module_name] = True
        _ERRORS.pop(module_name, None)
        return True

    package = package_name or module_name.split(".")[0]
    cmd = [sys.executable, "-m", "pip", "install", package]
    try:  # pragma: no cover - relies on external tooling
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except Exception as exc:  # Capture the failure but do not raise.
        _ATTEMPTS[module_name] = False
        _ERRORS[module_name] = f"{cmd!r} failed: {exc}"
        return False

    if _try_import(module_name):
        _ATTEMPTS[module_name] = True
        _ERRORS.pop(module_name, None)
        return True

    _ATTEMPTS[module_name] = False
    _ERRORS[module_name] = (
        f"Package '{package}' was installed but importing '{module_name}' still failed"
    )
    return False


def ensure_packages(mapping: Mapping[str, Optional[str]]) -> Dict[str, bool]:
    """Ensure multiple packages are available, returning a result dictionary."""

    return {module: ensure_package(module, pkg) for module, pkg in mapping.items()}


def get_package_error(module_name: str) -> Optional[str]:
    """Return the stored installation/import error for ``module_name``."""

    return _ERRORS.get(module_name)
