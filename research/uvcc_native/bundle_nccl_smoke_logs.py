#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WorkerLog:
    prefix: str
    run_log: Path
    exit_code: Optional[int]
    done: bool


def _safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _parse_exit_code(p: Path) -> Optional[int]:
    try:
        t = p.read_text(encoding="utf-8", errors="replace").strip()
        if not t:
            return None
        return int(t, 10)
    except Exception:
        return None


def _collect_workers(logs_dir: Path) -> List[WorkerLog]:
    out: List[WorkerLog] = []
    for run_log in sorted(logs_dir.glob("*.run.log")):
        prefix = run_log.name.removesuffix(".run.log")
        exit_p = logs_dir / f"{prefix}.exit_code.txt"
        done_p = logs_dir / f"{prefix}.done.txt"
        out.append(
            WorkerLog(
                prefix=prefix,
                run_log=run_log,
                exit_code=_parse_exit_code(exit_p) if exit_p.exists() else None,
                done=done_p.exists(),
            )
        )
    return out


def _find_nccl_warns(text: str, *, limit: int = 50) -> List[str]:
    out: List[str] = []
    for ln in text.splitlines():
        if "NCCL WARN" in ln:
            out.append(ln.strip())
            if len(out) >= limit:
                break
    return out


def _write_bundle(*, out_dir: Path, bundle_path: Path) -> None:
    # Bundle everything needed for offline review. Avoid embedding any private keys
    # (this repo does not write them into out_dir; still, keep a conservative filter).
    allow_rel_prefixes = {
        "runner_stdout.log",
        "runner.pid",
        "uvcc_native_bundle.tgz",
        "toy_open_matrix_done.txt",
        "nccl_smoke_done.json",
        "roots_by_coord.json",
        "remote_logs/",
    }

    def _should_include(rel: str) -> bool:
        rel = rel.replace("\\", "/")
        for pref in allow_rel_prefixes:
            if rel == pref or rel.startswith(pref):
                return True
        return False

    with tarfile.open(str(bundle_path), mode="w:gz") as tf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_dir():
                continue
            rel = str(p.relative_to(out_dir))
            if not _should_include(rel):
                continue
            tf.add(str(p), arcname=rel)


def _summary_counts(workers: List[WorkerLog]) -> Tuple[int, int, int]:
    total = len(workers)
    done = sum(1 for w in workers if w.done)
    ok = sum(1 for w in workers if w.exit_code == 0)
    return total, done, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Path to an out-prime-native-nccl-* directory")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    if not out_dir.exists():
        raise SystemExit(f"missing out-dir: {out_dir}")

    logs_dir = out_dir / "remote_logs"
    if not logs_dir.exists():
        raise SystemExit(f"missing remote_logs/ under {out_dir}")

    done_json = out_dir / "nccl_smoke_done.json"
    done_obj: Dict[str, object] = {}
    if done_json.exists():
        try:
            done_obj = json.loads(done_json.read_text(encoding="utf-8"))
        except Exception:
            done_obj = {}

    workers = _collect_workers(logs_dir)
    total, done_ct, ok_ct = _summary_counts(workers)

    # Collect a few high-signal warnings for quick diagnosis.
    warn_lines: List[str] = []
    for w in workers:
        warn_lines.extend(_find_nccl_warns(_safe_read_text(w.run_log), limit=10))
        if len(warn_lines) >= 30:
            break
    warn_lines = warn_lines[:30]

    bundle_path = out_dir / "nccl_smoke_logs_bundle.tgz"
    _write_bundle(out_dir=out_dir, bundle_path=bundle_path)

    md_path = out_dir / "nccl_smoke_all_logs_explained.md"
    sid_job = str(done_obj.get("sid_job_hex") or "").strip()
    topo = done_obj.get("topology") if isinstance(done_obj.get("topology"), dict) else {}
    step_id = done_obj.get("step_id")

    md = []
    md.append("# NCCL smoke run (bundle + explanations)\n")
    md.append(f"- out_dir: `{out_dir}`\n")
    md.append(f"- bundle: `{bundle_path}`\n")
    if sid_job:
        md.append(f"- sid_job: `{sid_job}`\n")
    if step_id is not None:
        md.append(f"- step_id: `{step_id}`\n")
    if topo:
        md.append(f"- topology: `{topo}`\n")

    md.append("\n## What this run is\n")
    md.append(
        "This is an **intra-party NCCL sanity run** launched across 3 parties. Each logical worker is addressed by "
        "coordinate `(party, replica, stage, tp)`.\n\n"
        "- **TP group**: fixed `(replica, stage)`; all-reduce over `tp` ranks\n"
        "- **PP group**: fixed `(replica, tp)`; send/recv across `stage` ranks\n"
        "- **DP group**: fixed `(stage, tp)`; all-reduce over `replica` ranks\n"
    )

    md.append("\n## Key files\n")
    md.append("- `runner_stdout.log`: orchestrator log (pod attach, relay start, launches, wait_done)\n")
    md.append("- `nccl_smoke_done.json`: machine-readable success marker\n")
    md.append("- `roots_by_coord.json`: per-worker metadata + remote paths\n")
    md.append("- `remote_logs/*.run.log`: per-worker stdout/stderr (includes NCCL INFO/WARN)\n")
    md.append("- `remote_logs/*.exit_code.txt`: per-worker exit code (0 == success)\n")
    md.append("- `remote_logs/*.done.txt`: per-worker done marker written by runner wrapper\n")
    md.append("- `nccl_smoke_logs_bundle.tgz`: portable tarball containing the above\n")

    md.append("\n## Completion summary\n")
    md.append(f"- workers_total: **{total}**\n")
    md.append(f"- workers_done_files: **{done_ct}**\n")
    md.append(f"- workers_exit_code_zero: **{ok_ct}**\n")
    if total and ok_ct != total:
        bad = [w for w in workers if w.exit_code not in (0, None)]
        md.append(f"- workers_nonzero_exit: **{len(bad)}**\n")
        for w in bad[:20]:
            md.append(f"  - `{w.prefix}` exit_code={w.exit_code}\n")

    if warn_lines:
        md.append("\n## NCCL warnings (first 30)\n")
        for ln in warn_lines:
            md.append(f"- `{ln}`\n")

    md.append("\n## How to inspect\n")
    md.append(
        "- **Per-worker log**: open any `remote_logs/pX_rY_sZ_tW.run.log`\n"
        "- **Quick scan for failures**:\n"
        "  - `grep -R \"error:\" -n remote_logs/`\n"
        "  - `grep -R \"NCCL WARN\" -n remote_logs/`\n"
        "- **Reproduce bundle**: rerun this script on the out_dir\n"
    )

    md_path.write_text("".join(md), encoding="utf-8")
    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


