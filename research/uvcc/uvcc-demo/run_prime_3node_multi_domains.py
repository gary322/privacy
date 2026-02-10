from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _now_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_checked(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None, stdout_path: Optional[Path] = None) -> None:
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stdout_path, "a", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"command failed (exit={p.returncode}): {' '.join(cmd)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="run_prime_3node_multi_domains")
    ap.add_argument("--providers", default="datacrunch,hyperstack,crusoecloud", help="Comma-separated provider types to run")
    ap.add_argument("--out-base", default="research/uvcc/uvcc-demo/out-prime-3node-multi", help="Base output prefix directory")
    ap.add_argument("--prime-api-key-path", default="", help="Path to file containing Prime API key (recommended)")
    ap.add_argument("--ssh-key-path", default="", help="SSH key path (optional; runner auto-detects common keys)")
    ap.add_argument("--smoke-job-json", default="research/uvcc/uvcc-demo/job_smoke_2steps_trace.json", help="Job JSON for smoke run")
    ap.add_argument("--full-job-json", default="research/uvcc/uvcc-demo/job_big_4090_trace.json", help="Job JSON for full run")
    ap.add_argument("--run-gpu-tests", default="true", choices=["true", "false"])
    ap.add_argument("--keep-pods", default="false", choices=["true", "false"])
    ap.add_argument("--gpu-telemetry", default="true", choices=["true", "false"])
    ap.add_argument("--gpu-telemetry-interval-s", default="0.5")
    ap.add_argument("--party-log-level", default="trace", choices=["quiet", "info", "debug", "trace"])
    ap.add_argument("--smoke-only", default="false", choices=["true", "false"], help="If true, only run smoke tests")
    args = ap.parse_args(argv)

    repo = _repo_root()
    runner = repo / "research" / "uvcc" / "uvcc-demo" / "run_prime_3node.py"
    bundler = repo / "research" / "uvcc" / "uvcc-demo" / "bundle_logs_explained.py"

    out_base = (repo / str(args.out_base)).resolve() if not str(args.out_base).startswith("/") else Path(str(args.out_base)).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    providers = [p.strip() for p in str(args.providers).split(",") if p.strip()]
    if len(providers) < 1:
        raise SystemExit("must provide at least one provider in --providers")

    env = dict(os.environ)
    # Helpful on macOS (LibreSSL): reduce noise in logs.
    env.setdefault("PYTHONWARNINGS", "ignore:NotOpenSSLWarning")

    summary: list[dict] = []
    for prov in providers:
        for phase in ["smoke", "full"]:
            if phase == "full" and str(args.smoke_only).lower() == "true":
                continue

            job_json = str(args.smoke_job_json) if phase == "smoke" else str(args.full_job_json)
            ts = _now_ts()
            out_dir = out_base / f"{ts}-{prov}-{phase}"
            out_dir.mkdir(parents=True, exist_ok=True)

            runner_stdout = out_dir / "runner_stdout.log"
            cmd = [
                sys.executable,
                str(runner),
                "--out",
                str(out_dir),
                "--provider-type",
                str(prov),
                "--job-json",
                str((repo / job_json).resolve() if not job_json.startswith("/") else Path(job_json).expanduser().resolve()),
                "--party-log-level",
                str(args.party_log_level),
                "--run-gpu-tests",
                str(args.run_gpu_tests),
                "--keep-pods",
                str(args.keep_pods),
                "--gpu-telemetry",
                str(args.gpu_telemetry),
                "--gpu-telemetry-interval-s",
                str(args.gpu_telemetry_interval_s),
                "--failover-max-epochs",
                "5",
                "--live-recorder-interval-s",
                "1",
            ]
            if str(args.prime_api_key_path).strip():
                cmd += ["--prime-api-key-path", str(args.prime_api_key_path).strip()]
            if str(args.ssh_key_path).strip():
                cmd += ["--ssh-key-path", str(args.ssh_key_path).strip()]

            rec = {"provider": prov, "phase": phase, "out_dir": str(out_dir), "ok": False, "bundle_ok": False}
            try:
                _run_checked(cmd, cwd=repo, env=env, stdout_path=runner_stdout)
                rec["ok"] = True
            except Exception as exc:
                rec["error"] = str(exc)
            # Always attempt to generate a single explained bundle (even on failure) so auditing/debugging has one file.
            try:
                _run_checked(
                    [sys.executable, str(bundler), "--out", str(out_dir), "--output", "all_logs_explained.md"],
                    cwd=repo,
                    env=env,
                    stdout_path=runner_stdout,
                )
                rec["bundle_ok"] = True
            except Exception as exc:
                rec["bundle_error"] = str(exc)
            summary.append(rec)

            # Persist per-phase summary.
            (out_dir / "multi_domain_summary.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Global summary file.
    out_summary = out_base / f"multi_domain_summary_{_now_ts()}.json"
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


