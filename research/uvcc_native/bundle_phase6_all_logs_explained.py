#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_RE_PIT = re.compile(r"\bpit_[0-9a-fA-F]{16,}\b")
_RE_BEARER = re.compile(r"(Authorization:\s*Bearer\s+)(\S+)", re.IGNORECASE)


def _sanitize_line(s: str) -> str:
    # Redact Prime API keys if they ever appear in logs.
    s = _RE_PIT.sub("pit_REDACTED", s)
    # Redact bearer tokens (shouldn't appear, but protect anyway).
    s = _RE_BEARER.sub(r"\1REDACTED", s)
    return s


def _write_code_block_from_file(md, *, path: Path, fence: str = "```") -> None:
    md.write(f"{fence}\n")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                md.write(_sanitize_line(ln))
    except FileNotFoundError:
        md.write(f"[missing file: {path}]\n")
    md.write(f"{fence}\n")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return int(x)
        s = str(x).strip()
        if s == "":
            return None
        return int(s, 10)
    except Exception:
        return None


def _worker_key(r: Dict[str, Any]) -> Tuple[int, int, int, int]:
    # Order: replica -> stage -> tp -> party for readability.
    return (int(r.get("replica", 0)), int(r.get("stage", 0)), int(r.get("tp", 0)), int(r.get("party", 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Phase6 run output dir (contains runner_stdout.log, remote_logs/, audit_bundle.json)")
    ap.add_argument(
        "--compare-out-dir",
        default=None,
        help="Optional previous run dir to compare determinism (reads its audit_bundle.json global_root_hex)",
    )
    ap.add_argument(
        "--out-file",
        default=None,
        help="Output markdown path (default: <out-dir>/all_logs_explained.md)",
    )
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    logs_dir = out_dir / "remote_logs"
    runner_log = out_dir / "runner_stdout.log"
    done_txt = out_dir / "toy_open_matrix_done.txt"
    roots_json = out_dir / "roots_by_coord.json"
    audit_json = out_dir / "audit_bundle.json"

    if args.out_file:
        out_file = Path(str(args.out_file)).expanduser().resolve()
    else:
        out_file = out_dir / "all_logs_explained.md"

    # Load metadata if present.
    audit: Optional[Dict[str, Any]] = None
    if audit_json.exists():
        try:
            audit = _load_json(audit_json)
        except Exception:
            audit = None

    compare_root: Optional[str] = None
    compare_dir = Path(str(args.compare_out_dir)).expanduser().resolve() if args.compare_out_dir else None
    if compare_dir is not None:
        aj = compare_dir / "audit_bundle.json"
        if aj.exists():
            try:
                compare_root = str(_load_json(aj).get("global_root_hex") or "").strip() or None
            except Exception:
                compare_root = None

    # Load roots_by_coord (canonical list of 192 workers).
    roots: List[Dict[str, Any]] = []
    if roots_json.exists():
        try:
            roots_raw = _load_json(roots_json)
            if isinstance(roots_raw, list):
                roots = [r for r in roots_raw if isinstance(r, dict)]
        except Exception:
            roots = []

    roots_sorted = sorted(roots, key=_worker_key)

    # Start writing markdown.
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as md:
        md.write("## UVCC Phase 6 Native Parallelism — Complete Logs (Explained)\n\n")
        md.write("This is a **single, end-to-end log file** bundling the entire run output:\n")
        md.write("- Runner orchestration log (`runner_stdout.log`)\n")
        md.write("- All 192 worker logs (DP/PP/TP + OPEN) from `remote_logs/`\n")
        md.write("- Proof/audit metadata (`roots_by_coord.json`, `audit_bundle.json`)\n\n")

        md.write("### What this demonstrates (high-level)\n\n")
        md.write("- **Private / confidential compute**: parties operate on **secret shares** (RSS 3PC, honest-majority). No party sees full secrets.\n")
        md.write("- **Verifiable compute**: each worker emits a deterministic **`epoch_root`**; these are combined into per-replica roots and a single **`global_root`**.\n")
        md.write("- **Full parallelism (Phase 6 bring-up)**:\n")
        md.write("  - **DP (R replicas)**: NCCL allreduce across replicas after backward (`phase6_dp_ready` / `phase6_dp_after_bwd_ok`).\n")
        md.write("  - **PP (S stages)**: NCCL send/recv of activations and gradients across stages.\n")
        md.write("  - **TP (T ranks)**: NCCL allreduce sanity checks within each stage.\n\n")

        md.write("### Security / redaction note\n\n")
        md.write("This file is intended to be shareable with auditors/operators.\n")
        md.write("- It includes **full runtime logs**, but **redacts** any accidental bearer tokens / `pit_...` API keys if they appear.\n")
        md.write("- It does **not** include private key material.\n\n")

        md.write("### Run metadata\n\n")
        md.write(f"- **out_dir**: `{out_dir}`\n")
        if audit is not None:
            md.write(f"- **sid_job_hex**: `{audit.get('sid_job_hex')}`\n")
            topo = audit.get("topology") if isinstance(audit.get("topology"), dict) else {}
            md.write(f"- **topology**: R={topo.get('R')} S={topo.get('S')} T={topo.get('T')} M={topo.get('M')}\n")
            md.write(f"- **global_root_hex**: `{audit.get('global_root_hex')}`\n")
        if done_txt.exists():
            md.write(f"- **completion marker**: `{done_txt}`\n")
        if compare_root:
            md.write(f"- **determinism comparison global_root_hex**: `{compare_root}`\n")
            if audit is not None:
                cur = str(audit.get("global_root_hex") or "").strip()
                if cur and cur == compare_root:
                    md.write("  - **determinism result**: **MATCH**\n")
                elif cur and compare_root:
                    md.write("  - **determinism result**: **MISMATCH**\n")
        md.write("\n")

        md.write("### How to read the worker logs\n\n")
        md.write("- **`phase6_fwd mb=k`** / **`phase6_bwd mb=k`**: forward/backward microbatch progression.\n")
        md.write("- **`phase6_*_tp_ok`**: TP allreduce sanity check succeeded.\n")
        md.write("- **`phase6_*_open_ok`**: OPEN protocol for that microbatch completed (cross-party sync).\n")
        md.write("- **`phase6_dp_ready`**: DP communicator brought up (post-backward), followed by DP allreduce check.\n")
        md.write("- **`epoch_root=0x...`**: deterministic transcript root for this worker’s subsession. Used for audit bundling.\n")
        md.write("- **`NCCL INFO ...`**: NCCL transport details (multi-host socket selection, bootstrap timing, etc.).\n\n")

        md.write("## 1) Runner log (`runner_stdout.log`)\n\n")
        md.write("The runner provisions/attaches pods, bootstraps dependencies, starts the relay, launches workers, waits for `done.txt`, and downloads artifacts.\n\n")
        _write_code_block_from_file(md, path=runner_log)
        md.write("\n")

        md.write("## 2) Audit metadata (`audit_bundle.json`)\n\n")
        md.write("This is the transcript-of-transcripts summary: all per-worker `epoch_root_hex` values combine into per-replica roots and a single `global_root_hex`.\n\n")
        if audit_json.exists():
            _write_code_block_from_file(md, path=audit_json)
        else:
            md.write("```text\n[missing audit_bundle.json]\n```\n")
        md.write("\n")

        md.write("## 3) Per-worker roots index (`roots_by_coord.json`)\n\n")
        md.write("Index of all workers and the `epoch_root_hex` extracted from each worker log.\n\n")
        if roots_json.exists():
            _write_code_block_from_file(md, path=roots_json)
        else:
            md.write("```text\n[missing roots_by_coord.json]\n```\n")
        md.write("\n")

        md.write("## 4) All worker logs (192 workers) — DP/PP/TP + OPEN + transcript roots\n\n")
        md.write("Ordering: replica → stage → tp → party.\n\n")

        if not logs_dir.exists():
            md.write("```text\n[missing remote_logs directory]\n```\n")
        else:
            for r in roots_sorted:
                party = int(r.get("party", 0))
                replica = int(r.get("replica", 0))
                stage = int(r.get("stage", 0))
                tp = int(r.get("tp", 0))
                pod_id = str(r.get("pod_id") or "").strip()
                ssh_host = str(r.get("ssh_host") or "").strip()
                epoch_root_hex = str(r.get("epoch_root_hex") or "").strip()

                prefix = logs_dir / f"p{party}_r{replica}_s{stage}_t{tp}"
                p_cmd = Path(str(prefix) + ".cmd.txt")
                p_exit = Path(str(prefix) + ".exit_code.txt")
                p_done = Path(str(prefix) + ".done.txt")
                p_run = Path(str(prefix) + ".run.log")

                md.write(f"### Worker p{party} r{replica} s{stage} t{tp}\n\n")
                md.write(f"- **pod_id**: `{pod_id}`\n")
                md.write(f"- **ssh_host**: `{ssh_host}`\n")
                if epoch_root_hex:
                    md.write(f"- **epoch_root_hex**: `{epoch_root_hex}`\n")
                md.write("\n")

                md.write("#### cmd.txt (what was executed)\n\n")
                _write_code_block_from_file(md, path=p_cmd)
                md.write("\n")

                md.write("#### exit_code.txt\n\n")
                _write_code_block_from_file(md, path=p_exit)
                md.write("\n")

                md.write("#### done.txt\n\n")
                _write_code_block_from_file(md, path=p_done)
                md.write("\n")

                md.write("#### run.log (full worker stdout/stderr)\n\n")
                _write_code_block_from_file(md, path=p_run)
                md.write("\n")

        md.write("## 5) Completion marker\n\n")
        if done_txt.exists():
            _write_code_block_from_file(md, path=done_txt)
        else:
            md.write("```text\n[missing toy_open_matrix_done.txt]\n```\n")
        md.write("\n")

    print(str(out_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




