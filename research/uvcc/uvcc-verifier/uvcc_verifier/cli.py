from __future__ import annotations

import argparse
from pathlib import Path

from .proof_bundle_v1 import parse_proof_bundle_json_v1, verify_proof_bundle_v1
from .transcript_v1 import compute_epoch_roots_v1, compute_final_root_v1, parse_transcript_jsonl_v1, validate_transcript_leaves_v1


def _cmd_verify(args: argparse.Namespace) -> int:
    proof_bytes = Path(str(args.proof)).read_bytes()
    proof = parse_proof_bundle_json_v1(proof_bytes)

    leaves = parse_transcript_jsonl_v1(str(args.transcript))
    validate_transcript_leaves_v1(leaves, strict_unknown_msg_kind=False, strict_netframe_header_hash=True)
    roots_by_epoch = compute_epoch_roots_v1(leaves)
    epoch_roots = [roots_by_epoch.get(e, b"") for e in range(len(proof.epoch_roots))]
    if any(len(r) != 32 for r in epoch_roots):
        raise SystemExit("transcript is missing required epoch roots for proof bundle")
    final_root32 = compute_final_root_v1(epoch_roots=epoch_roots)

    verify_proof_bundle_v1(proof=proof, transcript_epoch_roots=epoch_roots, transcript_final_root32=final_root32)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="uvcc-verifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="Verify a UVCC v1 proof bundle against a transcript JSONL dump")
    v.add_argument("--proof", required=True, help="Path to proof_bundle.json")
    v.add_argument("--transcript", required=True, help="Path to transcript_v1.jsonl (leaf body dump)")
    v.set_defaults(func=_cmd_verify)

    args = p.parse_args(argv)
    return int(args.func(args))


