"""API keys from the environment or a project .env file.

Only the variables a backend asks for are read; values are never printed or logged.
"""

from __future__ import annotations

import os
from pathlib import Path

_HARNESS = Path(__file__).resolve().parents[1]
# Searched in order: the working directory, the harness, the project root.
ENV_FILES = [Path.cwd() / ".env", _HARNESS / ".env", _HARNESS.parent / ".env"]


def _read(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key not in names:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        found[key] = value
    return found


def env_value(*names: str) -> str | None:
    """The first non-empty value among `names`, from the environment first, then .env files."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    for path in ENV_FILES:
        values = _read(path, names)
        for name in names:
            if values.get(name):
                return values[name]
    return None
