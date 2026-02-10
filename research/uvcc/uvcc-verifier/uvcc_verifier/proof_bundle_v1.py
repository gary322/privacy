from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_ba7afac425406f12

import base64
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eth_keys import keys
from eth_utils.crypto import keccak

from .transcript_v1 import compute_final_root_v1, sha256


def keccak256(data: bytes) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    return bytes(keccak(bytes(data)))


def _abi_word_bytes32(x32: bytes) -> bytes:
    if not isinstance(x32, (bytes, bytearray)) or len(x32) != 32:
        raise ValueError("expected 32-byte value")
    return bytes(x32)


def _abi_word_uint(x: int) -> bytes:
    if int(x) < 0:
        raise ValueError("uint must be >= 0")
    return int(x).to_bytes(32, "big", signed=False)


def _abi_word_address(addr20: bytes) -> bytes:
    if not isinstance(addr20, (bytes, bytearray)) or len(addr20) != 20:
        raise ValueError("expected 20-byte address")
    return b"\x00" * 12 + bytes(addr20)


EIP712_DOMAIN_TYPEHASH = keccak256(b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)")
FINAL_COMMIT_TYPEHASH = keccak256(b"FinalCommit(bytes32 jobId,bytes32 policyHash,bytes32 finalRoot,bytes32 resultHash)")


def _hex32_to_bytes(x: str) -> bytes:
    if not isinstance(x, str) or not x.startswith("0x") or len(x) != 66:
        raise ValueError("expected 0x + 64 hex")
    return bytes.fromhex(x[2:])


def _b64_to_bytes(x: str) -> bytes:
    if not isinstance(x, str):
        raise ValueError("expected base64 string")
    return base64.b64decode(x.encode("ascii"))


def _hex20_to_bytes(x: str) -> bytes:
    if not isinstance(x, str) or not x.startswith("0x") or len(x) != 42:
        raise ValueError("expected 0x + 40 hex (address)")
    return bytes.fromhex(x[2:])


def _eip712_domain_separator_v1(*, chain_id: int, verifying_contract20: bytes) -> bytes:
    name_hash = keccak256(b"UVCC")
    version_hash = keccak256(b"1")
    enc = b"".join(
        [
            _abi_word_bytes32(EIP712_DOMAIN_TYPEHASH),
            _abi_word_bytes32(name_hash),
            _abi_word_bytes32(version_hash),
            _abi_word_uint(int(chain_id)),
            _abi_word_address(bytes(verifying_contract20)),
        ]
    )
    return keccak256(enc)


def _eip712_final_commit_digest_v1(*, chain_id: int, verifying_contract20: bytes, job_id32: bytes, policy_hash32: bytes, final_root32: bytes, result_hash32: bytes) -> bytes:
    for name, b in [
        ("job_id32", job_id32),
        ("policy_hash32", policy_hash32),
        ("final_root32", final_root32),
        ("result_hash32", result_hash32),
    ]:
        if not isinstance(b, (bytes, bytearray)) or len(b) != 32:
            raise ValueError(f"{name} must be 32 bytes")
    struct_hash = keccak256(
        b"".join(
            [
                _abi_word_bytes32(FINAL_COMMIT_TYPEHASH),
                _abi_word_bytes32(bytes(job_id32)),
                _abi_word_bytes32(bytes(policy_hash32)),
                _abi_word_bytes32(bytes(final_root32)),
                _abi_word_bytes32(bytes(result_hash32)),
            ]
        )
    )
    dom = _eip712_domain_separator_v1(chain_id=int(chain_id), verifying_contract20=verifying_contract20)
    return keccak256(b"\x19\x01" + dom + struct_hash)


@dataclass(frozen=True)
class ProofBundlePartyParsedV1:
    party_id: int
    pubkey64: bytes


@dataclass(frozen=True)
class ProofBundleSignatureParsedV1:
    party_id: int
    sig65: bytes


@dataclass(frozen=True)
class ProofBundleParsedV1:
    uvcc_version: str
    job_id32: bytes
    policy_hash32: bytes
    eip712_chain_id: int
    eip712_verifying_contract20: bytes
    sgir_hash32: bytes
    runtime_hash32: bytes
    backend: str
    parties: Tuple[ProofBundlePartyParsedV1, ...]
    epoch_roots: Tuple[bytes, ...]
    final_root32: bytes
    signatures: Tuple[ProofBundleSignatureParsedV1, ...]
    result_hash32: bytes
    status: str
    evidence: Optional[Dict[str, Any]] = None
    optional_proofs: Optional[Dict[str, Any]] = None


def parse_proof_bundle_json_v1(proof_json_bytes: bytes) -> ProofBundleParsedV1:
    obj = json.loads(proof_json_bytes.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("proof bundle must be a JSON object")
    uvcc_version = str(obj.get("uvcc_version", ""))
    job = obj.get("job", {})
    if not isinstance(job, dict):
        raise ValueError("job must be object")
    identity = obj.get("identity", {})
    if not isinstance(identity, dict):
        raise ValueError("identity must be object")
    transcript = obj.get("transcript", {})
    if not isinstance(transcript, dict):
        raise ValueError("transcript must be object")
    eip712 = obj.get("eip712", {})
    if not isinstance(eip712, dict):
        raise ValueError("eip712 must be object")
    verdict = obj.get("verdict", {})
    if not isinstance(verdict, dict):
        raise ValueError("verdict must be object")

    job_id32 = _hex32_to_bytes(str(job.get("job_id", "")))
    policy_hash32 = _hex32_to_bytes(str(job.get("policy_hash", "")))
    sgir_hash32 = _hex32_to_bytes(str(job.get("sgir_hash", "")))
    runtime_hash32 = _hex32_to_bytes(str(job.get("runtime_hash", "")))
    backend = str(job.get("backend", ""))

    name = str(eip712.get("name", ""))
    ver = str(eip712.get("version", ""))
    if name != "UVCC" or ver != "1":
        raise ValueError("bad eip712 name/version")
    chain_id = int(eip712.get("chain_id", -1))
    verifying_contract20 = _hex20_to_bytes(str(eip712.get("verifying_contract", "")))

    parties_in = identity.get("parties", [])
    if not isinstance(parties_in, list):
        raise ValueError("identity.parties must be array")
    parties: List[ProofBundlePartyParsedV1] = []
    for p in parties_in:
        if not isinstance(p, dict):
            raise ValueError("party entry must be object")
        pid_s = str(p.get("party_id", ""))
        if not pid_s.startswith("P"):
            raise ValueError("party_id must be like P0/P1/P2")
        pid = int(pid_s[1:])
        pubkey64 = _b64_to_bytes(str(p.get("pubkey_b64", "")))
        if len(pubkey64) != 64:
            raise ValueError("pubkey_b64 must decode to 64 bytes")
        parties.append(ProofBundlePartyParsedV1(party_id=pid, pubkey64=pubkey64))

    roots_in = transcript.get("epoch_roots", [])
    if not isinstance(roots_in, list):
        raise ValueError("transcript.epoch_roots must be array")
    epoch_roots = tuple(_hex32_to_bytes(str(r)) for r in roots_in)
    final_root32 = _hex32_to_bytes(str(transcript.get("final_root", "")))

    sigs_in = obj.get("signatures", [])
    if not isinstance(sigs_in, list):
        raise ValueError("signatures must be array")
    sigs: List[ProofBundleSignatureParsedV1] = []
    for s in sigs_in:
        if not isinstance(s, dict):
            raise ValueError("signature entry must be object")
        pid_s = str(s.get("party_id", ""))
        if not pid_s.startswith("P"):
            raise ValueError("signature.party_id must be like P0/P1/P2")
        pid = int(pid_s[1:])
        sig65 = _b64_to_bytes(str(s.get("sig65_b64", "")))
        if len(sig65) != 65:
            raise ValueError("sig65 must be 65 bytes")
        sigs.append(ProofBundleSignatureParsedV1(party_id=pid, sig65=sig65))

    result_hash32 = _hex32_to_bytes(str(verdict.get("result_hash", "")))
    status = str(verdict.get("status", ""))
    evidence = obj.get("evidence", None)
    optional_proofs = obj.get("optional_proofs", None)

    return ProofBundleParsedV1(
        uvcc_version=uvcc_version,
        job_id32=job_id32,
        policy_hash32=policy_hash32,
        eip712_chain_id=int(chain_id),
        eip712_verifying_contract20=verifying_contract20,
        sgir_hash32=sgir_hash32,
        runtime_hash32=runtime_hash32,
        backend=backend,
        parties=tuple(parties),
        epoch_roots=epoch_roots,
        final_root32=final_root32,
        signatures=tuple(sigs),
        result_hash32=result_hash32,
        status=status,
        evidence=evidence if isinstance(evidence, dict) else None,
        optional_proofs=optional_proofs if isinstance(optional_proofs, dict) else None,
    )


def verify_proof_bundle_v1(
    *,
    proof: ProofBundleParsedV1,
    transcript_epoch_roots: Optional[Sequence[bytes]] = None,
    transcript_final_root32: Optional[bytes] = None,
) -> None:
    # Verify transcript final root matches epoch roots.
    want_final = compute_final_root_v1(epoch_roots=list(proof.epoch_roots))
    if want_final != proof.final_root32:
        raise ValueError("proof.final_root does not match proof.epoch_roots")

    if transcript_epoch_roots is not None:
        if list(transcript_epoch_roots) != list(proof.epoch_roots):
            raise ValueError("transcript epoch_roots mismatch with proof bundle")
    if transcript_final_root32 is not None:
        if bytes(transcript_final_root32) != proof.final_root32:
            raise ValueError("transcript final_root mismatch with proof bundle")

    # Party set checks.
    if len(proof.parties) != 3:
        raise ValueError("proof bundle must include exactly 3 parties")
    if len(proof.signatures) != 3:
        raise ValueError("proof bundle must include exactly 3 signatures")

    parties_by_id = {int(p.party_id): p for p in proof.parties}
    sigs_by_id = {int(s.party_id): s for s in proof.signatures}
    if set(parties_by_id.keys()) != {0, 1, 2}:
        raise ValueError("party ids must be exactly {0,1,2}")
    if set(sigs_by_id.keys()) != {0, 1, 2}:
        raise ValueError("signature party ids must be exactly {0,1,2}")

    # Verify signatures over canonical hash.
    msg_hash32 = _eip712_final_commit_digest_v1(
        chain_id=int(proof.eip712_chain_id),
        verifying_contract20=bytes(proof.eip712_verifying_contract20),
        job_id32=proof.job_id32,
        policy_hash32=proof.policy_hash32,
        final_root32=proof.final_root32,
        result_hash32=proof.result_hash32,
    )
    for pid in (0, 1, 2):
        pub = parties_by_id[pid].pubkey64
        sig = sigs_by_id[pid].sig65
        try:
            pk = keys.PublicKey(pub)
            s = keys.Signature(sig)
            if not s.verify_msg_hash(msg_hash32, pk):
                raise ValueError(f"bad signature for party {pid}")
        except Exception as e:
            raise ValueError(f"signature verification failed for party {pid}: {e}") from e


