from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import os
from pathlib import Path

from .demo_job import run_demo_job_v1
from .party_identity import load_or_create_party_privkey32_v1, party_identity_from_privkey_v1, party_sign_hash32_v1


def _cmd_run_demo(args: argparse.Namespace) -> int:
    out_dir = Path(str(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_demo_job_v1(out_dir=str(out_dir))
    return 0


def _parse_hex_bytes(s: str, *, n: int) -> bytes:
    t = str(s).strip()
    if t.startswith("0x"):
        t = t[2:]
    b = bytes.fromhex(t)
    if len(b) != int(n):
        raise ValueError(f"expected {n} bytes")
    return b


def _cmd_party_info(args: argparse.Namespace) -> int:
    priv = load_or_create_party_privkey32_v1(path=str(args.key_path))
    ident = party_identity_from_privkey_v1(party_id=int(args.party_id), privkey32=priv)
    print(json.dumps(ident.to_json_obj(), sort_keys=True, separators=(",", ":")))
    return 0


def _cmd_party_sign(args: argparse.Namespace) -> int:
    priv = load_or_create_party_privkey32_v1(path=str(args.key_path))
    ident = party_identity_from_privkey_v1(party_id=int(args.party_id), privkey32=priv)
    digest32 = _parse_hex_bytes(str(args.digest_hex), n=32)
    sig65 = party_sign_hash32_v1(privkey32=priv, digest32=digest32)
    out = dict(ident.to_json_obj())
    out["sig65_hex"] = "0x" + sig65.hex()
    print(json.dumps(out, sort_keys=True, separators=(",", ":")))
    return 0


def _cmd_run_party_train(args: argparse.Namespace) -> int:
    import torch
    from .party_train import PartyTrainInputsV1, run_party_train_v1

    device_s = str(args.device or "auto").strip().lower()
    if device_s == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_s == "cuda":
        dev = torch.device("cuda")
    elif device_s == "cpu":
        dev = torch.device("cpu")
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")

    out_dir = Path(str(args.out)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = PartyTrainInputsV1.from_json(str(args.inputs_json))
    relay_token = str(args.relay_token).strip() if args.relay_token is not None else ""
    if not relay_token and args.relay_token_file is not None:
        relay_token = Path(str(args.relay_token_file)).expanduser().read_text(encoding="utf-8").strip()
    if not relay_token:
        relay_token = str(os.environ.get("UVCC_RELAY_TOKEN", "")).strip()
    relay_token_opt = relay_token if relay_token else None
    run_party_train_v1(
        party_id=int(args.party_id),
        relay_base_url=str(args.relay_url),
        relay_group_id=str(args.group_id),
        relay_token=relay_token_opt,
        tls_ca_pem=str(args.tls_ca_pem) if args.tls_ca_pem is not None else None,
        job_id32=_parse_hex_bytes(str(args.job_id_hex), n=32),
        sid=str(args.sid).encode("utf-8"),
        inputs=inputs,
        out_dir=str(out_dir),
        device=dev,
        require_cuda=(str(args.require_cuda).lower() == "true"),
        steps=int(args.steps),
        epoch=int(args.epoch),
        step_offset=int(args.step_offset),
        epoch_setup_step=int(args.epoch_setup_step),
        checkpoint_enable=(str(args.checkpoint_enable).lower() == "true"),
        checkpoint_every=int(args.checkpoint_every),
        sks_t_checks=int(args.sks_t_checks),
        sks_sample_log2=int(args.sks_sample_log2),
        log_level=str(args.log_level),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="uvcc-client")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("run-demo", help="Run local MPC demo job and emit proof_bundle.json + transcript_v1.jsonl")
    d.add_argument("--out", required=True, help="Output directory")
    d.set_defaults(func=_cmd_run_demo)

    pi = sub.add_parser("party-info", help="Create/load a party identity key and print address/pubkey JSON")
    pi.add_argument("--party-id", required=True, type=int, choices=[0, 1, 2])
    pi.add_argument("--key-path", default="~/.uvcc/party_privkey.hex")
    pi.set_defaults(func=_cmd_party_info)

    ps = sub.add_parser("party-sign", help="Sign a 32-byte digest with the party identity key")
    ps.add_argument("--party-id", required=True, type=int, choices=[0, 1, 2])
    ps.add_argument("--key-path", default="~/.uvcc/party_privkey.hex")
    ps.add_argument("--digest-hex", required=True, help="0x + 64 hex digest")
    ps.set_defaults(func=_cmd_party_sign)

    rt = sub.add_parser("run-party-train", help="Run the UVCC 3PC training workload as one party (writes transcript + result)")
    rt.add_argument("--party-id", required=True, type=int, choices=[0, 1, 2])
    rt.add_argument("--relay-url", required=True, help="Relay base URL (http(s)://host:port)")
    rt.add_argument("--group-id", required=True, help="Relay group_id for this job")
    rt.add_argument("--relay-token", default=None, help="Relay bearer token (if relay require-token=true)")
    rt.add_argument("--relay-token-file", default=None, help="Path to file containing relay bearer token (avoids secrets in args)")
    rt.add_argument("--tls-ca-pem", default=None, help="CA cert PEM path to trust relay TLS")
    rt.add_argument("--job-id-hex", required=True, help="0x + 64 hex job_id32")
    rt.add_argument("--sid", required=True, help="sid bytes (utf-8) for v1 transcript/relay domain separation")
    rt.add_argument("--inputs-json", required=True, help="Path to party inputs JSON (shares + TCF key)")
    rt.add_argument("--out", required=True, help="Output directory for transcript_v1.jsonl and result.json")
    rt.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    rt.add_argument("--require-cuda", default="true", choices=["true", "false"])
    rt.add_argument("--steps", default=1, type=int)
    rt.add_argument("--epoch", default=0, type=int, help="Transcript/protocol epoch (default: 0)")
    rt.add_argument("--step-offset", default=0, type=int, help="Global step offset (so logs/transcript steps remain unique across restarts)")
    rt.add_argument("--epoch-setup-step", default=1000, type=int, help="Domain-separation step id used during SKS epoch setup (default: 1000)")
    rt.add_argument("--checkpoint-enable", default="false", choices=["true", "false"], help="If true, write private per-step checkpoints (default: false)")
    rt.add_argument("--checkpoint-every", default=1, type=int, help="Checkpoint every N local steps (default: 1)")
    rt.add_argument("--sks-t-checks", default=3, type=int, help="Number of Freivalds checks (t) when selected (default: 3)")
    rt.add_argument("--sks-sample-log2", default=0, type=int, help="Sampling log2 rate (0=always check; default: 0)")
    rt.add_argument("--log-level", default="info", choices=["quiet", "info", "debug", "trace"], help="Per-party log verbosity")
    rt.set_defaults(func=_cmd_run_party_train)

    args = p.parse_args(argv)
    return int(args.func(args))


