# UVCC Verifier (v1)

Deterministic verifier for UVCC v1 proof bundles and transcript roots, aligned to the canonical v1 runtime in `research/uvcc/uvcc-party/`.

## What it verifies (v1)
- Proof bundle JSON schema + hash checks
- Party identity + signatures over `(policy_hash32, final_root32, job_id32)`
- Transcript Merkle roots (epoch roots + final root) from a transcript leaf dump
- Basic structural validation of transcript leaves (sizes, reserved fields, leaf hash, ordering)
- Optional strict checks for NetFrame header hashing for SEND/RECV leaves

## CLI

```bash
python3 -m uvcc_verifier verify \
  --proof /path/to/proof_bundle.json \
  --transcript /path/to/transcript_v1.jsonl
```

Transcript JSONL format: one JSON object per line:

```json
{"body_b64":"...base64 leaf body bytes..."}
```

Optionally include `"leaf_hash_hex"` for debugging; the verifier will recompute and cross-check.


