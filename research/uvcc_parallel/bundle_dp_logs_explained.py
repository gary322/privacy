from __future__ import annotations

"""
Bundle a UVCC DP (3×R pods) run output directory into a single, shareable Markdown file.

Safety:
- This script MUST NOT include secrets. In particular, it excludes:
  - out/private_keep/**   (party privkeys + checkpoints)
  - any */private/**      (per-party checkpoints/shares)
"""

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_hex(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def _fence_for(content: str) -> str:
    if "```" not in content:
        return "```"
    if "~~~~" not in content:
        return "~~~~"
    return "```"


def _iter_files(out_dir: Path) -> Iterable[Path]:
    # Allowlist-ish: include common textual artifacts; exclude secrets explicitly.
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(out_dir))
        # Exclude secrets.
        if rel.startswith("private_keep/"):
            continue
        if "/private/" in rel or rel.startswith("private/"):
            continue
        # Exclude big bundles and obvious binaries.
        if rel.endswith(".tgz") or rel.endswith(".tar") or rel.endswith(".tar.gz"):
            continue
        if rel.endswith(".png") or rel.endswith(".jpg") or rel.endswith(".jpeg") or rel.endswith(".pdf"):
            continue
        # Keep mostly-text files.
        if not any(rel.endswith(suf) for suf in (".log", ".jsonl", ".json", ".md", ".txt", ".csv", ".pid")):
            continue
        yield p


def _render_file_section(*, out_dir: Path, path: Path, max_embed_bytes: int) -> str:
    rel = str(path.relative_to(out_dir))
    size = int(path.stat().st_size)
    h = _sha256_hex(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    fence = _fence_for(text)

    lines: List[str] = []
    lines.append(f"## `{rel}`")
    lines.append("")
    lines.append(f"- sha256: `{h}`")
    lines.append(f"- bytes: `{size}`")
    lines.append("")
    if size <= int(max_embed_bytes):
        lines.append(fence)
        lines.append(text.rstrip("\n"))
        lines.append(fence)
    else:
        # Embed only a preview but keep hash for integrity.
        preview = text[: max(1, int(max_embed_bytes))]
        lines.append(f"> NOTE: file is larger than --max-embed-bytes ({max_embed_bytes}); embedded content is truncated.")
        lines.append("")
        lines.append(fence)
        lines.append(preview.rstrip("\n"))
        lines.append(fence)
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="DP run output directory")
    ap.add_argument("--output", default="all_logs_explained.md", help="Output markdown filename (written inside --out unless absolute)")
    ap.add_argument("--max-embed-bytes", default=2_000_000, type=int, help="Max bytes to embed per file (default: 2,000,000)")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out)).expanduser().resolve()
    if not out_dir.exists():
        raise FileNotFoundError(str(out_dir))

    out_path = Path(str(args.output)).expanduser()
    if not out_path.is_absolute():
        out_path = (out_dir / out_path).resolve()

    files = list(_iter_files(out_dir))

    lines: List[str] = []
    lines.append("# UVCC DP run — consolidated logs + explanations")
    lines.append("")
    lines.append(f"- Generated: `{_now_iso_utc()}`")
    lines.append(f"- Out dir: `{out_dir}`")
    lines.append("")
    lines.append("## What this file is")
    lines.append("")
    lines.append("This is a **single-file bundle** that embeds the key logs/artifacts from a UVCC **data-parallel (SR‑DP)** run.")
    lines.append("It is designed to be readable by someone who did not run the job, while preserving integrity via per-file SHA-256 hashes.")
    lines.append("")
    lines.append("## Safety / redaction policy")
    lines.append("")
    lines.append("- This bundle **excludes** `private_keep/**` and any `*/private/**` directories.")
    lines.append("- That means it does **not** include party signing private keys or checkpoint/share material.")
    lines.append("")
    lines.append("## Included files (index)")
    lines.append("")
    for p in files:
        rel = str(p.relative_to(out_dir))
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    for p in files:
        lines.append(_render_file_section(out_dir=out_dir, path=p, max_embed_bytes=int(args.max_embed_bytes)))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_path)}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


