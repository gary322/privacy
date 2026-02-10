# `uvcc-spec/` (Frozen v1 Specs)
<!-- UVCC_REQ_GROUP: uvcc_group_4b3e1ae5d567ea44 -->

This directory is the **normative, byte-exact spec freeze** extracted from `research/privacy_new.txt` for the standalone UVCC system.

## Ground truth inputs
- **Coverage matrix**: `research/uvcc/uvcc-spec/coverage/privacy_new_coverage_matrix.md`
- **Canonical v1 profile**: `research/uvcc/uvcc-spec/profiles/uvcc_profile_v1.md`

## What “spec-freeze” means here
- Every packed header/container is represented as:
  - a **C header** in `uvcc/uvcc-spec/structs/` (packed structs + size asserts)
  - a **test** in `uvcc/uvcc-spec/tests/` validating size/layout invariants and hashing vectors
- Where the doc defines variable-length containers (TLV-like), we freeze:
  - the fixed headers as structs
  - the full layout as markdown + deterministic builders/verifier rules

## Contents (v1)
- `structs/uvcc_netframe_v1.h`: canonical NetFrame header + segment header (packed) and constants.
- `structs/uvcc_policy_wire_v1.h`: policy wire v1 fixed header + party records (packed).
- `tests/`: python tests for struct sizes and hashing vectors (keccak + sha256).


