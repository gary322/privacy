# UVCC Demo (v1)

End-to-end deterministic demo that:
- runs a small UVCC 3PC job (TCF-v0a Beaver GEMM + SKS-Lite Freivalds),
- emits a proof bundle + transcript dump,
- verifies them with `uvcc-verifier`,
- deploys `uvcc-contracts` to a local Anvil chain,
- submits the job + final proof and checks on-chain state matches the bundle.

## Run

```bash
python3 run_demo.py --out ./out-demo
```

Outputs:
- `./out-demo/proof_bundle.json`
- `./out-demo/transcript_v1.jsonl`


