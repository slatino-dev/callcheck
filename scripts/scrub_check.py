#!/usr/bin/env python3
"""Generic secret scanner for CI and pre-commit use.

Fails (exit 1) if any tracked text file contains a likely secret (private keys,
cloud or API tokens) or a carrier-grade-NAT address. Ships no project- or
host-specific values.

Usage: ``python scripts/scrub_check.py [--path DIR]``
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")
_OPENAI_KEY = re.compile(r"\bsk-(?:ant-|proj-|or-v1-|live-)?[A-Za-z0-9]{32,}\b")
_SLACK_TOKEN = re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{12,}\b")
_CGNAT = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-key-block", _PRIVATE_KEY),
    ("aws-access-key-id", _AWS_KEY),
    ("github-token", _GITHUB_TOKEN),
    ("openai-style-key", _OPENAI_KEY),
    ("slack-token", _SLACK_TOKEN),
    ("cgnat-address", _CGNAT),
]

SKIP_SUFFIX: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".pdf",
        ".whl",
        ".pyc",
        ".so",
        ".dll",
        ".exe",
        ".lock",
        ".bin",
        ".gz",
        ".zip",
        ".tar",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
    }
)
SKIP_PARTS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "target",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".ruff_cache",
        ".astro",
        "vendor",
    }
)


def tracked_files(root: Path) -> list[Path]:
    """Return tracked files; fall back to a full walk outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return [p for p in root.rglob("*") if p.is_file()]
    return [root / line for line in result.stdout.splitlines() if line]


def is_skipped(path: Path) -> bool:
    """True if the path is a binary, lockfile, or lives in a vendored dir."""
    if path.suffix.lower() in SKIP_SUFFIX:
        return True
    if "min." in path.name:
        return True
    return any(part in SKIP_PARTS for part in path.parts)


def main() -> int:
    args = sys.argv[1:]
    if "--path" in args:
        root = Path(args[args.index("--path") + 1]).resolve()
    else:
        root = Path.cwd()
    self_path = Path(__file__).resolve()
    findings = 0
    for path in tracked_files(root):
        if path.resolve() == self_path or is_skipped(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    print(f"SCRUB FAIL [{label}] {path}:{lineno}")
                    findings += 1
    if findings:
        print(f"scrub_check: FAILED - {findings} match(es).")
        return 1
    print("scrub_check: OK - no secrets detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
