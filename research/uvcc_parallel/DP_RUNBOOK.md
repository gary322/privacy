### UVCC SR-DP runbook (Prime Intellect, 3 providers, R replicas)

This runbook covers the **staged rollout** recommended in `research/PARALLEL.txt`:

- Run a **small smoke** first (R=2 replicas → 6 pods total).
- If it passes, scale to the target (R=8 replicas → 24 pods total).

This runner provisions:
- party0 on provider A
- party1 on provider B
- party2 on provider C

DP reduction:
- within each party domain, uses **torch.distributed NCCL allreduce** across replicas.

---

### Prereqs (operator machine)

- Prime API key available in `~/.uvcc/prime_api_key.txt` (or set `UVCC_PRIME_API_KEY`).
- SSH key path that Prime pods accept (ed25519 recommended).

---

### Smoke (R=2, 2 steps)

```bash
OUT="research/uvcc_parallel/out-prime-dp-smoke-r2-$(date -u +%Y%m%dT%H%M%SZ)"

python3 research/uvcc_parallel/run_prime_dp_3xR.py \
  --out "$OUT" \
  --replicas 2 \
  --providers hyperstack,crusoecloud,lambdalabs \
  --image cuda_12_4_pytorch_2_4 \
  --job-json research/uvcc/uvcc-demo/job_smoke_2steps_trace.json \
  --ssh-key-path ~/.ssh/uvcc_prime_runner_ed25519 \
  --prime-api-key-path ~/.uvcc/prime_api_key.txt \
  --dp-preflight true \
  --keep-pods false \
  --party-log-level info

# Produce a single-file bundle (excludes secrets like party keys + checkpoints)
python3 research/uvcc_parallel/bundle_dp_logs_explained.py --out "$OUT" --output all_logs_explained.md
```

What to assert:
- `run_full.log` contains:
  - `EVENT dp_preflight_ok` for all 6 pods
  - `EVENT dp_result_hashes` with exactly 1 unique hash
  - `EVENT verifier_ok`
- `dp_layout.json`, `dp_roots.json`, `dp_matrix.md`, `proof_bundle.json` exist.

---

### Full (R=8, 25 steps)

```bash
OUT="research/uvcc_parallel/out-prime-dp-full-r8-$(date -u +%Y%m%dT%H%M%SZ)"

python3 research/uvcc_parallel/run_prime_dp_3xR.py \
  --out "$OUT" \
  --replicas 8 \
  --providers hyperstack,crusoecloud,lambdalabs \
  --image cuda_12_4_pytorch_2_4 \
  --job-json research/uvcc/uvcc-demo/job_big_4090_trace.json \
  --ssh-key-path ~/.ssh/uvcc_prime_runner_ed25519 \
  --prime-api-key-path ~/.uvcc/prime_api_key.txt \
  --dp-preflight true \
  --keep-pods false \
  --party-log-level info

python3 research/uvcc_parallel/bundle_dp_logs_explained.py --out "$OUT" --output all_logs_explained.md
```

Notes:
- Multi-host NCCL can be blocked by provider firewalls / port mappings. The runner’s `--dp-preflight true` is designed to fail fast before starting MPC training if NCCL rendezvous is not reachable.


