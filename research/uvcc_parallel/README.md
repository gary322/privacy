### UVCC Parallel Overlay (`research/uvcc_parallel/`)

This directory is an **additive overlay** for implementing the scaling design in `research/PARALLEL.txt` **on top of** the existing UVCC codebase in `research/uvcc/`.

Goals:
- Add **SR-DP (secure-replica data parallel)**: run **R independent 3-party triangles** concurrently.
- Preserve UVCC invariants:
  - no share leakage (logs contain hashes only)
  - deterministic IDs / transcripts
  - existing single-triangle runners continue to work unchanged

Important conventions (from `research/PARALLEL.txt`):
- Treat one UVCC 3PC job as a **triangle**.
- For DP, run triangles in parallel by giving each replica its own `sid_rep[r]`.
- Avoid collisions by keeping ID derivations unchanged and making `sid` unique per subgroup:
  - `sid_job` → `sid_rep[r]` → `sid_sub[r,s,t]`.

Files:
- `dp_ids.py`: deterministic sid derivations + per-replica relay group IDs.
- `dp_roots.py`: transcript-of-transcripts helpers (`replica_root`, `global_root`) per `PARALLEL.txt` §14.
- (to be added) `party_train_dp.py`: DP-aware worker (NCCL allreduce on share gradients).
- (to be added) `run_prime_dp_3xR.py`: Prime runner for 3×R pods.
- `bundle_dp_logs_explained.py`: DP-safe bundler (generates `all_logs_explained.md` while excluding secrets like `private_keep/**`).


