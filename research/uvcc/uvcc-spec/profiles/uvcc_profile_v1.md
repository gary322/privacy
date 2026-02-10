# UVCC Profile v1 (Canonical, Production-Strict)

This profile pins a **single deterministic interpretation** of `research/privacy_new.txt` for UVCC v1.

- **Source doc**: `research/privacy_new.txt`
- **Source SHA256**: `a1fa9c5f4f8f9cac5b36cace49532142995aba0cfd130aaae6955632c7214411`
- **Coverage gate**: `research/uvcc/uvcc-spec/coverage/privacy_new_coverage_matrix.md`
- **Coverage generator**: `research/uvcc/scripts/generate_coverage_matrix.py --check`

## Precedence rule (how we resolve overlaps)
- **Hard conventions** sections are always binding.
- When the document contains overlapping definitions, the **more detailed byte-exact spec** wins over earlier prose.
- When the document explicitly labels a choice as **MUST** (e.g., on-chain commitments), v1 follows it even if other sections show alternate hashes as examples.

## Execution mode (v1)
- **Default backend**: `CRYPTO_CC_3PC` (3-party honest-majority MPC, replicated secret sharing).
- **GPU execution**: CUDA-only (NVIDIA) acceleration is supported; CPU reference execution is required for deterministic cross-checks.
- **GPU-TEE backend**: treated as a future extension; v1 canonical demo/interop assumes 3PC RSS.

## Hashing policy (v1)
UVCC v1 uses **two distinct hash contexts** (both explicitly allowed by the doc, and required in practice due to EVM constraints):

### 1) On-chain commitments (EVM-native)
Used for:
- `policyHash` (on-chain)
- `sgir_hash32`, `runtime_hash32`, `fss_dir_hash32`, `preproc_hash32` (embedded in `uvcc_policy_wire_v1`)
- On-chain manifest hashes (`keyrec_hash`, etc.)

**Algorithm (v1)**: `keccak256`.
This follows `privacy_new.txt` §2.1 (“Policy hash used on-chain MUST be keccak256”) and §2.2/§2.3 (fixed digests and manifest hashing).

### 2) Transcript hashing (verifier-friendly, CPU/GPU ubiquitous)
Used for:
- Transcript leaf hashes
- Merkle roots (epoch roots)
- `finalRoot`
- Optional result/output roots when specified

**Algorithm (v1)**: `SHA256`.
Rationale: multiple later sections use `SHA256` for transcript roots and leaf hashing; SHA256 is universally available and easy to reproduce across languages/runtimes.

## Canonical policy wire format (v1)
UVCC v1 treats the **binary** policy wire format as canonical:

- **`policy_wire_v1_bytes`**: `uvcc_policy_wire_v1` as specified in `privacy_new.txt` §2.2
- **`policy_hash32` (on-chain)**:
  - `policy_hash32 = keccak256(policy_wire_v1_bytes)`

### Policy inputs (developer ergonomics)
Client tooling may accept JSON/CBOR policies, but they must compile into **exactly one** `policy_wire_v1_bytes`.

Canonical rules:
- `sid_hash32` in the policy header is **keccak256(sid_bytes)** (v1 fixed choice).
- Party set is exactly 3 parties, ids `0,1,2`, with **ECDSA secp256k1** addresses.

## Job id (v1)
The on-chain structs use `bytes32 jobId`. v1 sets:

`jobId = keccak256("UVCC.jobid.v1\\0" || policy_hash32 || client_nonce32)`

Notes:
- `client_nonce32` is a caller-provided 32-byte nonce.
- The optional `u64 job_id` field inside the policy header is set to `0` in v1 (reserved for sequential ids if desired).

## Commit signatures (v1)
v1 uses the **EIP-712** signatures defined in `privacy_new.txt` §3:
- `PolicyCommit` signed at job creation
- `FinalCommit` signed at job finalization

EIP-712 digests are Keccak-based by definition; the `finalRoot` field is a raw `bytes32` (v1: SHA256 transcript root) and is treated as opaque by EIP-712.

## Transcript + frame model (v1)
v1 uses the generic **NetFrame** model for relay transport:
- Canonical frame header + segment headers
- Strict ordering rules
- Alignment + zero-padding rules
- Retransmit/dedup behavior that does **not** affect transcript semantics

Transcript semantics (v1 canonical):
- **Order-independent**: leaves are keyed by `leaf_key` and epoch roots are computed by sorting by `leaf_key` then Merkle hashing.
- **Exactly-once**: receiver acceptance stores the first `(msg_id32 → frame_hash32)` binding; duplicates do not create new transcript entries.
- **SHA256 everywhere for transcript**: wherever `privacy_new.txt` uses `H256(...)` in transcript sections, v1 interprets this as **SHA256** with explicit domain-sep prefixes as written.

NCCL-specialized frame layouts in the doc are treated as a **future optional direct-NCCL transport**; the v1 canonical implementation uses NetFrames over TLS relay.

## Determinism rules (v1, enforced)
- OPEN batching/order is canonical (sorted by `(open_id, sub_id)` within a round).
- Segment ordering is canonical (sorted by segment descriptors).
- Padding bytes are always zero.
- Transcript leaves are computed from **semantic bytes only**; transport-layer retries/reordering do not affect epoch roots.

## Privacy / leakage model (v1)
- v1 ships **Class A SGIR** only: secret-dependent indexing/branching is rejected unless a future ORAM profile is explicitly enabled.
- Conditionals on secret bits lower to “both branches + mux” as specified.

## Proof bundle + verifier (v1)
Proof Bundles must allow a verifier to deterministically check:
- `policy_wire_v1_bytes` and `policy_hash32` (keccak)
- `jobId` derivation
- Transcript leaf hashing + Merkle roots (SHA256)
- `finalRoot` matches transcript
- `PolicyCommit` + `FinalCommit` EIP-712 signatures are valid for the three party addresses
- Optional Freivalds/SKS artifacts when policy requires them

## Chain config (v1)
- **Chain**: Arbitrum Sepolia `421614`
- **AVL token**: `0xf56E308F368e67D3Faa0509929348115ecCdF2BE`

