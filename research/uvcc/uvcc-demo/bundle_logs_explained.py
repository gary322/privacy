from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-8", errors="replace")


def _sha256_hex(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_json_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("{") and s.endswith("}")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in _read_text(path).splitlines():
        if not _is_json_line(line):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _parse_run_summary(run_full_jsonl: Path) -> Dict[str, Any]:
    """
    Best-effort parse of key milestones from run_full.jsonl.
    """
    summary: Dict[str, Any] = {
        "provider_type": None,
        "cloud_id": None,
        "gpu_type": None,
        "image": None,
        "socket": None,
        "steps": None,
        "d": None,
        "sks_t_checks": None,
        "sks_sample_log2": None,
        "start_ts": None,
        "end_ts": None,
        "job_id32": None,
        "result_hash32_hex": None,
        "proof_bundle_hash32_hex": None,
        "epoch0_root_hex": None,
        "final_root_hex": None,
        "artifacts": {},
    }

    for ev in _iter_jsonl(run_full_jsonl):
        name = ev.get("event")
        ts = ev.get("ts")
        if summary["start_ts"] is None and isinstance(ts, str):
            summary["start_ts"] = ts

        fields = ev.get("fields") if isinstance(ev.get("fields"), dict) else {}
        if name == "config":
            summary["provider_type"] = fields.get("provider_type")
            summary["image"] = fields.get("image")
            summary["socket"] = fields.get("socket")
        elif name == "prime_offer_try":
            summary["cloud_id"] = fields.get("cloud_id") or summary["cloud_id"]
            summary["gpu_type"] = fields.get("gpu_type") or summary["gpu_type"]
        elif name == "job_config":
            summary["steps"] = fields.get("steps")
            summary["d"] = fields.get("d")
            summary["sks_t_checks"] = fields.get("sks_t_checks")
            summary["sks_sample_log2"] = fields.get("sks_sample_log2")
        elif name == "training_done":
            summary["result_hash32_hex"] = fields.get("result_hash32_hex") or summary["result_hash32_hex"]
            if isinstance(ts, str):
                summary["end_ts"] = ts
        elif name == "transcript_roots":
            summary["epoch0_root_hex"] = fields.get("epoch0_root_hex") or summary["epoch0_root_hex"]
            summary["final_root_hex"] = fields.get("final_root_hex") or summary["final_root_hex"]
        elif name == "verifier_ok":
            summary["proof_bundle_hash32_hex"] = fields.get("proof_bundle_hash32_hex") or summary["proof_bundle_hash32_hex"]
        elif name == "artifacts":
            if isinstance(fields, dict):
                summary["artifacts"] = fields

        # Some runs include the job id only in the human log; best-effort attempt to infer from artifacts/onchain logs
        # will be handled elsewhere.

    return summary


def _infer_job_id_from_run_full_log(run_full_log: Path) -> Optional[str]:
    """
    The job id appears in the final '=== Verifiability ===' section as:
      - job_id32: 0x....
    """
    text = _read_text(run_full_log)
    m = re.search(r"job_id32:\s*(0x[0-9a-fA-F]{64})", text)
    return m.group(1) if m else None


def _collect_event_names(party_log: Path) -> List[str]:
    names: set[str] = set()
    for line in _read_text(party_log).splitlines():
        if not _is_json_line(line):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("event")
        if isinstance(name, str):
            names.add(name)
    return sorted(names)


def _collect_run_events(run_full_jsonl: Path) -> List[str]:
    names: set[str] = set()
    for obj in _iter_jsonl(run_full_jsonl):
        name = obj.get("event")
        if isinstance(name, str):
            names.add(name)
    return sorted(names)


def _fence_for(content: str) -> str:
    # Avoid fence collisions (very unlikely for these logs, but cheap to harden).
    if "```" not in content:
        return "```"
    if "~~~~" not in content:
        return "~~~~"
    return "```"  # fallback


def _describe_file(rel: str) -> str:
    """
    Short human explanation per known artifact.
    """
    if rel == "run_full.log":
        return "Human-readable orchestrator log: provisioning → bootstrap/tests → relay → training → transcripts/proof → verifier → on-chain finalize."
    if rel == "run_full.jsonl":
        return "Machine-readable orchestrator events (one JSON object per line). Useful for programmatic parsing."
    if rel == "runner_stdout.log":
        return "Raw stdout/stderr from the runner process (warnings, tracebacks if any)."
    if rel.startswith("node_p") and rel.endswith("_bootstrap.log"):
        return "Per-node bootstrap log (OS/package setup, repo install, environment sanity checks)."
    if rel.startswith("node_p") and rel.endswith("_gpu_tests.log"):
        return "Per-node GPU validation log (torch CUDA check, simple kernels, sanity tests)."
    if rel.startswith("node_p") and rel.endswith("_nvidia_smi.txt"):
        return "Single-shot `nvidia-smi` snapshot captured during provisioning for that node."
    if rel.startswith("party_p") and rel.endswith("_run.log"):
        return "Per-party training log (JSONL). Includes trace events per step and cryptographic commitments/hashes."
    if rel.startswith("transcript_") and rel.endswith(".jsonl"):
        return "Transcript shard (JSONL) produced by a party or union transcript for verifiability. Lines are compact/base64 encoded transcript leaves."
    if rel == "transcript_v1.jsonl":
        return "Union transcript (JSONL). This is the primary transcript used for verification and proof generation."
    if rel == "proof_bundle.json":
        return "Final proof bundle (JSON). Contains what the verifier checks against the transcript roots/result hash."
    if rel == "onchain_createJob.log":
        return "On-chain createJob transaction receipt/logs for this run."
    if rel == "onchain_submitFinal.log":
        return "On-chain submitFinal transaction receipt/logs (finalization) for this run."
    if rel == "how_to_verify_public.md":
        return "Public verification instructions (how to re-run the verifier using transcript + proof bundle)."
    if rel.startswith("gpu_telemetry_p") and rel.endswith(".csv"):
        return "GPU telemetry captured during training (nvidia-smi sampled). Columns: timestamp,index,name,util.gpu,util.mem,mem.used,mem.total,power,temperature."
    if rel.startswith("live_keep/party_p") and rel.endswith("_run.log"):
        return "Append-only mirror of the per-party run log captured live during execution (redundant with top-level `party_p*_run.log`)."
    if rel.startswith("live_keep/party_p") and rel.endswith("_transcript.jsonl"):
        return "Append-only mirror of party transcript captured live (redundant with top-level `transcript_p*.jsonl`)."
    if rel.startswith("live_keep/gpu_telemetry_polled_p") and rel.endswith(".csv"):
        return "Append-only GPU telemetry polled over SSH (redundant/backstop for remote sampling). Same columns as other telemetry CSVs."
    if rel == "live_keep/recorder.log":
        return "Append-only recorder health log (ticks, SSH failures). Helps prove we were continuously capturing."
    if rel == "live_keep/recorder_stdout.log":
        return "Append-only recorder raw stdout/stderr (useful if it ever crashed)."
    if rel == "live_keep/watch_status.log":
        return "Watchdog status log (runner/recorder liveness + last step per party over time)."
    if rel == "live_keep/recorder_state.json":
        return "Recorder internal state (byte offsets/parts) used to ensure append-only mirroring."
    if rel.endswith(".pid"):
        return "Process id file (bookkeeping)."
    if rel.endswith(".json"):
        return "JSON artifact (machine-readable)."
    if rel.endswith(".txt"):
        return "Text snapshot."
    return "Run artifact / log."


def _render_intro(*, out_dir: Path, summary: Dict[str, Any], run_events: List[str], party_events: List[str]) -> str:
    job_id32 = summary.get("job_id32")
    lines = []
    lines.append(f"# UVCC Prime 3-node run — consolidated logs + explanations")
    lines.append("")
    lines.append(f"- Generated: `{_now_iso_utc()}`")
    lines.append(f"- Out dir: `{out_dir}`")
    lines.append("")
    lines.append("## What this file contains")
    lines.append("")
    lines.append(
        textwrap.dedent(
            """\
            This single document embeds **all logs/artifacts from the run output directory** and adds
            a guide explaining how to interpret them.

            There are two “layers” of logging:
            - **Orchestrator logs** (runner on your machine): provisioning + coordination + artifact collection.
            - **Party logs** (one per GPU node): the confidential training protocol execution (trace-level events).

            When someone is debugging or auditing a run, the usual reading order is:
            1) `run_full.log` (human narrative)
            2) `run_full.jsonl` (structured event timeline)
            3) `party_p*_run.log` (per-step protocol details)
            4) `transcript_v1.jsonl` + `proof_bundle.json` + on-chain receipts (verifiability/finalization)
            5) GPU telemetry CSVs (performance/health)
            """
        ).rstrip()
    )
    lines.append("")
    lines.append("## Quick run summary")
    lines.append("")
    def _kv(k: str, v: Any) -> str:
        return f"- **{k}**: `{v}`" if v is not None else f"- **{k}**: (unknown)"

    lines.append(_kv("provider_type", summary.get("provider_type")))
    lines.append(_kv("cloud_id", summary.get("cloud_id")))
    lines.append(_kv("gpu_type", summary.get("gpu_type")))
    lines.append(_kv("image", summary.get("image")))
    lines.append(_kv("socket", summary.get("socket")))
    lines.append(_kv("job_d", summary.get("d")))
    lines.append(_kv("steps", summary.get("steps")))
    lines.append(_kv("sks_t_checks", summary.get("sks_t_checks")))
    lines.append(_kv("sks_sample_log2", summary.get("sks_sample_log2")))
    lines.append(_kv("start_ts", summary.get("start_ts")))
    lines.append(_kv("end_ts", summary.get("end_ts")))
    lines.append(_kv("job_id32", job_id32))
    lines.append(_kv("result_hash32_hex", summary.get("result_hash32_hex")))
    lines.append(_kv("epoch0_root_hex", summary.get("epoch0_root_hex")))
    lines.append(_kv("final_root_hex", summary.get("final_root_hex")))
    lines.append(_kv("proof_bundle_hash32_hex", summary.get("proof_bundle_hash32_hex")))
    lines.append("")
    lines.append("## How to read the step numbers")
    lines.append("")
    lines.append(
        "Steps are **0-indexed**. For a 25-step job you will see `step=0` … `step=24`."
    )
    lines.append("")
    lines.append("## Log record formats (\"headings\")")
    lines.append("")
    lines.append(
        textwrap.dedent(
            """\
            ### `run_full.log` (human)
            Lines look like:
            - `[2025-...Z] EVENT <name> { ...json... }` for structured milestones
            - plain text for section headers and final summary

            The final summary section includes headings:
            - `=== Privacy ===`: relay URL(s) + TLS CA fingerprint used for secure node-to-node comms
            - `=== Verifiability ===`: job id, transcript roots, result hash, proof hash (what auditors verify)
            - `=== Efficiency ===`: cloud/gpu/job parameters and telemetry settings
            - `=== Artifacts ===`: where the key output files were written

            ### `run_full.jsonl` (structured)
            One JSON object per line:
            - `event`: short event name (string)
            - `fields`: event-specific payload (object)
            - `ts`: UTC timestamp (string)
            - `t_rel_s`: seconds since run start (float)

            ### `party_p*_run.log` (structured + occasional raw text)
            Mostly one JSON object per line:
            - `event`: protocol event name
            - `fields`: event-specific payload
            - `party_id`: 0/1/2
            - `ts`: UTC timestamp

            Some non-JSON lines may appear (e.g. `torch` warnings); those can be treated as normal stdout/stderr.

            ### `transcript*.jsonl`
            Transcript lines are stored in a compact format like `{"body_b64":"..."}`:
            - The `body_b64` is a base64 encoding of a binary transcript leaf.
            - The union transcript root hashes are printed in `run_full.log` / `run_full.jsonl` (`transcript_roots`).
            - The verifier uses these transcripts and `proof_bundle.json` to reproduce/check the same roots.
            """
        ).rstrip()
    )
    lines.append("")
    lines.append("## Event glossary (high-signal)")
    lines.append("")
    lines.append("### Orchestrator (`run_full.jsonl`) events you’ll care about")
    lines.append("")
    # Curated ordering of “most important” events, then the rest.
    curated_run = [
        "config",
        "prime_offer_try",
        "prime_pods_active",
        "node_bootstrap_done",
        "node_gpu_tests_done",
        "privacy_relay_started",
        "onchain_createJob",
        "training_launch",
        "party_done",
        "training_done",
        "transcript_roots",
        "verifier_ok",
        "onchain_submitFinal",
        "artifacts",
    ]
    run_desc = {
        "config": "Run configuration (provider, GPU telemetry settings, output dir).",
        "prime_offer_try": "Prime capacity/offer selection (cloud_id, gpu_type, region).",
        "prime_pods_active": "All 3 pods are ACTIVE and SSH hosts are known.",
        "node_bootstrap_done": "Node bootstrap complete; logs saved to `node_p*_bootstrap.log`.",
        "node_gpu_tests_done": "GPU sanity tests complete; logs saved to `node_p*_gpu_tests.log`.",
        "privacy_relay_started": "Secure relay started (TLS CA fingerprint is printed). Parties connect through this relay.",
        "onchain_createJob": "On-chain job creation submitted; receipt in `onchain_createJob.log`.",
        "training_launch": "The 3-party training protocol was launched (d/steps/SKS params).",
        "party_done": "A party finished and produced `result.json` + transcript shard (paths included).",
        "training_done": "All parties finished; `result_hash32_hex` is the public commitment to final output.",
        "transcript_roots": "Roots of the transcript Merkle trees (epoch0 + final + union transcript path).",
        "verifier_ok": "Local verifier succeeded; `proof_bundle.json` hash printed for auditing.",
        "onchain_submitFinal": "Final proof submitted on-chain; receipt in `onchain_submitFinal.log`.",
        "artifacts": "Summary of output paths written by the runner.",
    }
    remaining = [e for e in run_events if e not in curated_run]
    for e in curated_run + remaining:
        if e not in run_events:
            continue
        desc = run_desc.get(e, "See `fields` for details.")
        lines.append(f"- **{e}**: {desc}")
    lines.append("")
    lines.append("### Party (`party_p*_run.log`) events you’ll care about")
    lines.append("")
    curated_party = [
        "party_start",
        "inputs_commitments",
        "sks_epoch_setup_done",
        "step_start",
        "gemm1_done",
        "sks1_done",
        "gemm2_done",
        "sks2_done",
        "step_done",
        "open_final_done",
        "transcript_summary",
    ]
    party_desc = {
        "party_start": "Party process started; repeats key job params and confirms CUDA is available.",
        "inputs_commitments": "Hash commitments to private inputs (X/Y/W etc). Lets observers audit consistency without seeing the plaintext.",
        "sks_epoch_setup_done": "Secret-key-switching (SKS) epoch setup completed; commitment + randomness hash printed.",
        "step_start": "Step N started (0-indexed).",
        "gemm1_done": "First main GEMM for the step finished; public hashes of intermediate commitments printed + time.",
        "sks1_done": "First SKS check for the step passed (ok=true).",
        "gemm2_done": "Second main GEMM for the step finished; hashes + time.",
        "sks2_done": "Second SKS check for the step passed (ok=true).",
        "step_done": "Step N complete; prints public hashes of updated weights (W_hi/W_lo) + total step time.",
        "open_final_done": "Final output opened/published; prints `result_hash32_hex` (the public output commitment).",
        "transcript_summary": "Transcript shard summary: number of leaves and the shard hash.",
    }
    remaining_party = [e for e in party_events if e not in curated_party]
    for e in curated_party + remaining_party:
        if e not in party_events:
            continue
        desc = party_desc.get(e, "See `fields` for details.")
        lines.append(f"- **{e}**: {desc}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_file_section(*, out_dir: Path, rel: str, content: str) -> str:
    p = out_dir / rel
    size = p.stat().st_size if p.exists() else 0
    desc = _describe_file(rel)
    fence = _fence_for(content)
    lang = ""
    if rel.endswith(".jsonl"):
        lang = "jsonl"
    elif rel.endswith(".json"):
        lang = "json"
    elif rel.endswith(".md"):
        lang = "markdown"
    elif rel.endswith(".csv"):
        lang = "csv"
    else:
        lang = "text"
    hdr = []
    hdr.append(f"## `{rel}`")
    hdr.append("")
    hdr.append(f"- **Purpose**: {desc}")
    hdr.append(f"- **Bytes**: {size}")
    hdr.append(f"- **sha256**: `{_sha256_hex(p)}`" if p.exists() else "- **sha256**: (missing)")
    hdr.append("")
    if not content.strip():
        hdr.append("_File is empty._")
        hdr.append("")
        return "\n".join(hdr) + "\n"
    hdr.append(f"{fence}{lang}")
    hdr.append(content.rstrip("\n"))
    hdr.append(fence)
    hdr.append("")
    return "\n".join(hdr) + "\n"


def _list_files(out_dir: Path) -> List[str]:
    rels: List[str] = []

    # Top-level files first (stable, high-signal).
    top = sorted([p for p in out_dir.iterdir() if p.is_file()], key=lambda p: p.name)
    for p in top:
        rels.append(p.name)

    # Then live_keep folder (capture subsystem + append-only mirrors).
    live = out_dir / "live_keep"
    if live.exists() and live.is_dir():
        live_files = sorted([p for p in live.rglob("*") if p.is_file()], key=lambda p: str(p))
        for p in live_files:
            rels.append(str(p.relative_to(out_dir)))

    # Prefer a curated order rather than pure alphabetical.
    preferred = [
        "run_full.log",
        "run_full.jsonl",
        "runner_stdout.log",
        "node_p0_nvidia_smi.txt",
        "node_p0_bootstrap.log",
        "node_p0_gpu_tests.log",
        "node_p1_nvidia_smi.txt",
        "node_p1_bootstrap.log",
        "node_p1_gpu_tests.log",
        "node_p2_nvidia_smi.txt",
        "node_p2_bootstrap.log",
        "node_p2_gpu_tests.log",
        "onchain_createJob.log",
        "party_p0_run.log",
        "party_p1_run.log",
        "party_p2_run.log",
        "transcript_p0.jsonl",
        "transcript_p1.jsonl",
        "transcript_p2.jsonl",
        "transcript_v1.jsonl",
        "proof_bundle.json",
        "onchain_submitFinal.log",
        "how_to_verify_public.md",
        "gpu_telemetry_p0.csv",
        "gpu_telemetry_p1.csv",
        "gpu_telemetry_p2.csv",
        "relay_node0.log",
    ]

    def _rank(r: str) -> Tuple[int, str]:
        return (preferred.index(r), r) if r in preferred else (10_000, r)

    # Deduplicate while preserving final order.
    seen: set[str] = set()
    ordered = []
    for r in sorted(rels, key=_rank):
        if r in seen:
            continue
        seen.add(r)
        ordered.append(r)
    return ordered


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="bundle_logs_explained")
    ap.add_argument("--out", required=True, help="UVCC run out dir (contains run_full.log etc)")
    ap.add_argument(
        "--output",
        default="all_logs_explained.md",
        help="Output filename (written inside --out). Default: all_logs_explained.md",
    )
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out)).expanduser().resolve()
    if not out_dir.exists() or not out_dir.is_dir():
        raise SystemExit(f"--out dir does not exist: {out_dir}")

    run_full_jsonl = out_dir / "run_full.jsonl"
    run_full_log = out_dir / "run_full.log"
    if not run_full_jsonl.exists() or not run_full_log.exists():
        raise SystemExit("Expected run_full.jsonl and run_full.log in --out directory.")

    summary = _parse_run_summary(run_full_jsonl)
    summary["job_id32"] = _infer_job_id_from_run_full_log(run_full_log)

    run_events = _collect_run_events(run_full_jsonl)
    party_events: set[str] = set()
    for p in (out_dir / "party_p0_run.log", out_dir / "party_p1_run.log", out_dir / "party_p2_run.log"):
        if p.exists():
            party_events.update(_collect_event_names(p))

    intro = _render_intro(out_dir=out_dir, summary=summary, run_events=run_events, party_events=sorted(party_events))

    out_path = out_dir / str(args.output)
    files = _list_files(out_dir)

    parts: List[str] = []
    parts.append(intro)
    parts.append("# Embedded logs/artifacts (raw)\n\n")

    for rel in files:
        p = out_dir / rel
        if not p.exists() or not p.is_file():
            continue
        try:
            content = _read_text(p)
        except Exception as exc:
            content = f"<<failed to read: {exc}>>\n"
        parts.append(_render_file_section(out_dir=out_dir, rel=rel, content=content))

    out_path.write_text("".join(parts), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


