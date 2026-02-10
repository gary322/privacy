# UVCC (Universal Verifiable Confidential Computing)

UVCC is a practical system for **confidential** GPU computation across mutually distrustful domains and **third-party verifiability** via deterministic transcripts.

This repository is a standalone extraction of the UVCC code/spec from a larger workspace. It intentionally excludes any run outputs (`out-*`) and secret material (`private_keep`).

## Status / Disclaimer

This is research-grade software. It has not been independently audited. Do not treat it as production-ready cryptography.

## What “Verifiable Confidential Compute” Means Here

- **Confidentiality**: secrets (inputs, weights, activations, gradients, optimizer state) are protected using **3-party honest-majority MPC** with **replicated secret sharing (RSS)**.
- **Verifiability (receipt-style)**: the runtime emits deterministic **transcript leaves** for protocol events (transport + OPEN/LIFT/etc), computes **Merkle epoch roots** and a **final root**, and packages them into a **Proof Bundle** signed by the three parties (EIP-712). A verifier can recompute roots and validate signatures independently.

This is **not** a general-purpose zkSNARK/zkVM proof of arbitrary GPU execution. UVCC’s verifiability is a deterministic, auditor-friendly **execution receipt** bound to the MPC protocol events (plus optional protocol-level checks such as SKS/Freivalds where enabled).

## How People Use This Repo

- **Verify a UVCC run**: given `proof_bundle.json` + `transcript_v1.jsonl`, a third party can deterministically recompute transcript roots and validate party signatures.
- **Run a local 3-party demo**: run all three parties locally against a local relay and produce a proof bundle + transcript.
- **Run real GPUs**: use the Prime 3-node runner to provision 3 GPU nodes across providers, run the protocol, then verify and optionally anchor a receipt on-chain (demo contracts).
- **Hack on the protocol/runtime**: iterate on transcript rules, transport framing, OPEN batching, TCF triple derivations, SKS checks, and the native C++ runtime.

## Repo Layout

- `research/privacy_new.txt`
  - Normative UVCC v1 spec document (domain separation, hashing rules, transcript schema, etc).
- `research/uvcc/uvcc-spec/`
  - Canonical profile pinning a single interpretation: `profiles/uvcc_profile_v1.md`.
  - Coverage tooling: `coverage/` + `scripts/generate_coverage_matrix.py`.
- `research/uvcc/uvcc-party/`
  - Python v1 runtime building blocks:
    - transport framing: `uvcc_party/netframe.py`
    - transcript + Merkle roots: `uvcc_party/transcript.py`
    - OPEN (ring + bool): `uvcc_party/open.py`
    - TCF triples + GEMM Beaver: `uvcc_party/tcf.py`, `uvcc_party/gemm.py`
    - SKS (Freivalds-style checks): `uvcc_party/sks.py`
    - proof bundle + EIP-712: `uvcc_party/proof_bundle.py`, `uvcc_party/eip712.py`
- `research/uvcc/uvcc-relay/`
  - HTTP(S) relay with dedup, TTL, leased poll/ack: `relay_server.py`.
- `research/uvcc/uvcc-verifier/`
  - Deterministic verifier for transcript roots + proof bundles.
- `research/uvcc/uvcc-contracts/`
  - Solidity `UVCCJobLedger` verifying EIP-712 commits and recording `(finalRoot,resultHash)`.
- `research/uvcc/uvcc-demo/`
  - Operator runners; notably Prime 3-node runner: `run_prime_3node.py`.
- `research/uvcc_parallel/`
  - Data-parallel (DP) orchestration utilities aligned to `research/PARALLEL.txt`.
- `research/uvcc_native/`
  - Native C++ runtime implementing the PARALLEL phases (transport, transcript, DP/PP/TP scaffolding).
- `research/PARALLEL.txt`
  - Design notes for scaling UVCC using DP + PP + TP/MP.
- `docs/how_to_run.md`
  - Operator guide + postmortem for the Prime 3-node run.

## Quick Start: Verify Artifacts (No GPU Required)

If you only want to verify someone else’s run artifacts, you do **not** need CUDA or Torch.

```bash
uv venv
source .venv/bin/activate
uv pip install -r research/uvcc/requirements-uvcc-base.txt
```

Verify:

```bash
PYTHONPATH=research/uvcc/uvcc-verifier \
  python -m uvcc_verifier verify \
    --proof /path/to/proof_bundle.json \
    --transcript /path/to/transcript_v1.jsonl
```

## Quick Start: Local Demo (CPU)

### 1) Python environment

From repo root:

```bash
uv venv
source .venv/bin/activate
uv pip install -r research/uvcc/requirements-uvcc-base.txt
```

### 2) Install PyTorch

`research/uvcc/requirements-uvcc-base.txt` intentionally does **not** pin Torch (your Torch build must match your platform/CUDA). Install Torch separately, e.g.:

```bash
uv pip install torch
```

### 3) Run a deterministic local demo job

This starts a local relay, runs 3 parties locally, and writes `proof_bundle.json` + `transcript_v1.jsonl`:

```bash
PYTHONPATH=research/uvcc/uvcc-client \
  python -m uvcc_client run-demo --out ./out/uvcc-demo
```

### 4) Verify proof bundle vs transcript (third-party check)

```bash
PYTHONPATH=research/uvcc/uvcc-verifier \
  python -m uvcc_verifier verify \
    --proof ./out/uvcc-demo/proof_bundle.json \
    --transcript ./out/uvcc-demo/transcript_v1.jsonl
```

## Run the Relay (Standalone)

```bash
python research/uvcc/uvcc-relay/relay_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --db ./out/uvcc-relay.sqlite \
  --require-token false
```

## Prime 3-Node Runner (Real GPUs)

Start here:

- `docs/how_to_run.md` (deep operator guide + postmortem)
- `research/uvcc/uvcc-demo/PRIME_3NODE_RUNBOOK.md` (short runbook)
- `research/uvcc/uvcc-demo/run_prime_3node.py` (runner)

## Contracts (Optional On-Chain Anchoring)

UVCC’s demo contracts live in `research/uvcc/uvcc-contracts/` and are written for Foundry.

```bash
cd research/uvcc/uvcc-contracts
forge test
```

## Native Runtime (C++)

```bash
cmake -S research/uvcc_native -B research/uvcc_native/build -DCMAKE_BUILD_TYPE=Release
cmake --build research/uvcc_native/build -j
ctest --test-dir research/uvcc_native/build --output-on-failure
```

## Contact

Email: `gary@alien.international`

## License

See `LICENSE` and `NOTICE`.
