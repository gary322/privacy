# UVCC Client (v1)

`uvcc-client` is a deterministic local orchestrator for:
- running a small UVCC 3PC job (CPU reference),
- emitting a **proof bundle** (`proof_bundle.json`),
- dumping a **transcript leaf** JSONL (`transcript_v1.jsonl`),
- and (optionally) producing on-chain submission material for `uvcc-contracts`.

This is intended to be "plug-and-play" with the UVCC reference runtime in `research/uvcc/uvcc-party/`.

## Run the local MPC demo job

```bash
python3 -m uvcc_client run-demo --out ./out-demo
```

Outputs:
- `./out-demo/proof_bundle.json`
- `./out-demo/transcript_v1.jsonl`


