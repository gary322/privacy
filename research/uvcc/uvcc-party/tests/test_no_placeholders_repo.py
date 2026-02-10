from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple


_DENY_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".cursor",
    "__pycache__",
    "node_modules",
    "out",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    "lib",  # vendored deps
}

_ALLOW_EXT = {
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cu",
    ".cuh",
    ".sol",
}

# Intentionally strict: we do not allow temporary/incomplete markers in production code or tests.
# Note: tokens are assembled to avoid self-matching this very gate.
_TOKENS = [
    r"\b" + ("TO" + "DO") + r"\b",
    r"\b" + ("FIX" + "ME") + r"\b",
    r"\b" + ("TB" + "D") + r"\b",
    ("Not" + "ImplementedError"),
    r"raise\s+" + ("Not" + "Implemented"),
    r"\b" + ("pla" + "ceholder") + r"\b",
    r"\b" + ("st" + "ub") + r"\b",
    r"\b" + ("not " + "implemented") + r"\b",
    r"\b" + ("for " + "now") + r"\b",
]
_FORBIDDEN = re.compile("(" + "|".join(_TOKENS) + ")", re.IGNORECASE)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in _DENY_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in _ALLOW_EXT:
            continue
        yield p


def _scan_path(root: Path) -> List[Tuple[str, int, str]]:
    hits: List[Tuple[str, int, str]] = []
    for p in _iter_files(root):
        try:
            lines = p.read_text(encoding="utf-8", errors="strict").splitlines()
        except Exception:
            # If a file isn't strict UTF-8, it's not acceptable for deterministic builds anyway.
            hits.append((str(p), 0, "non-utf8-or-read-failed"))
            continue
        for i, line in enumerate(lines, start=1):
            if _FORBIDDEN.search(line):
                hits.append((str(p), int(i), line.strip()))
    return hits


def test_repo_has_no_placeholder_markers() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    targets = [
        repo_root / "research" / "uvcc" / "uvcc-party",
        repo_root / "research" / "uvcc" / "uvcc-relay",
        repo_root / "research" / "uvcc" / "uvcc-verifier",
        repo_root / "research" / "uvcc" / "uvcc-client",
        repo_root / "research" / "uvcc" / "uvcc-demo",
        repo_root / "research" / "uvcc" / "uvcc-contracts",
    ]
    all_hits: List[Tuple[str, int, str]] = []
    for t in targets:
        all_hits.extend(_scan_path(t))
    assert not all_hits, "Found disallowed markers:\n" + "\n".join(f"- {p}:{ln}: {txt}" for (p, ln, txt) in all_hits[:50])


