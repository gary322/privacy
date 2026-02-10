# UVCC Prime Intellect 3-node runbook (v1)

This runbook provisions **3 GPU pods on Prime Intellect**, starts the **UVCC TLS relay on node0**, runs a **real 3-party MPC training job** across the 3 nodes, then produces:

- `transcript_v1.jsonl` (union transcript)
- `proof_bundle.json` (EIP-712 signatures)
- `run_full.log` + `run_full.jsonl` (aggregated privacy/verifiability/efficiency logs)
- on-chain receipts (Anvil / Foundry logs)

## Prereqs (local machine)
- Python 3.9+.
- Foundry installed: `anvil`, `forge`, `cast` available on PATH.
- A Prime Intellect API key available in env (do **not** paste it into logs/files).
- An SSH private key that Prime pods accept (ed25519 recommended).

## One-command run (auto-select GPU + cloud)

1) Export your Prime API key:

```bash
export UVCC_PRIME_API_KEY="(your Prime API key)"
```

2) Run the 3-node job:

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node
```

Notes:
- If you want to explicitly select your SSH key:

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node \
  --ssh-key-path ~/.ssh/vracu_prime_intellect_ed25519
```

- By default the runner will **best-effort delete pods** at the end to avoid runaway spend. To keep pods alive:

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node \
  --keep-pods true
```

## Job selection
The runner currently supports a built-in job kind:
- `train_v1`: a small MPC “training-like” workload (GEMM + SKS checks) that emits full transcripts + proof bundle.

You can override parameters via `--job-json`:

Example `job.json`:

```json
{
  "kind": "train_v1",
  "d": 64,
  "steps": 3,
  "seed": 424242,
  "require_cuda": true,
  "fxp_frac_bits": 0
}
```

Run:

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node \
  --job-json /absolute/path/to/job.json
```

## What gets produced
In `--out`:
- `run_full.log`: aggregated human-readable log, with explicit **Privacy / Verifiability / Efficiency** sections.
- `run_full.jsonl`: structured events (timestamps, durations, file paths).
- `transcript_v1.jsonl`: union transcript across all parties.
- `proof_bundle.json`: policy/job IDs, roots, signatures (EIP-712).
- `relay_node0.log`: relay server log from node0 (best-effort).
- `party_p0_run.log`, `party_p1_run.log`, `party_p2_run.log`: per-party runtime logs (best-effort).
- `node_p*_nvidia_smi.txt`: per-node GPU snapshots.
- `node_p*_bootstrap.log`: per-node bootstrap logs.
- `node_p*_gpu_tests.log`: optional CUDA parity test logs.
- `onchain_createJob.log`, `onchain_submitFinal.log`: Foundry outputs.

## Privacy guarantees in this run
- Relay uses **TLS** with a per-run CA; parties use **CA pinning** (`tls_ca_pem`) to prevent MITM.
- Relay requires a **bearer token**, stored in a file on each node (not in command args).
- Logs redact obvious secret keys and do not print the Prime API key or relay token value.

## Verifiability guarantees in this run
- Each party produces transcript leaves (SHA-256 leaf hashing + Merkle roots).
- The runner recomputes epoch roots + final root and verifies the proof bundle off-chain.
- The runner submits final commitments on-chain (Anvil) and confirms the stored `finalRoot/resultHash` match the proof.


