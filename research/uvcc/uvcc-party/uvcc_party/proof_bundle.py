from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .eip712 import EIP712DomainV1, FinalCommitV1
from .sig import secp256k1_pubkey_from_privkey, secp256k1_sign_hash


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _hex32(b: bytes) -> str:
    if len(b) != 32:
        raise ValueError("expected 32-byte value")
    return "0x" + b.hex()


def proof_bundle_hash32_v1(proof_bundle_json_bytes: bytes) -> bytes:
    # privacy_new.txt §8.4: SHA256("uvcc.proofbundle.v1" || proof_bundle_bytes)
    return hashlib.sha256(b"uvcc.proofbundle.v1" + bytes(proof_bundle_json_bytes)).digest()


@dataclass(frozen=True)
class ProofBundlePartyV1:
    party_id: int
    pubkey64: bytes

    def to_dict(self) -> Dict[str, Any]:
        return {"party_id": f"P{int(self.party_id)}", "pubkey_b64": _b64(self.pubkey64)}


@dataclass(frozen=True)
class ProofBundleSignatureV1:
    party_id: int
    sig65: bytes

    def to_dict(self) -> Dict[str, Any]:
        return {"party_id": f"P{int(self.party_id)}", "sig65_b64": _b64(self.sig65)}


@dataclass(frozen=True)
class ProofBundleV1:
    uvcc_version: str
    job_id32: bytes
    policy_hash32: bytes
    eip712_domain: EIP712DomainV1
    sgir_hash32: bytes
    runtime_hash32: bytes
    backend: str  # "GPU_TEE" | "CRYPTO_CC_3PC"
    parties: Sequence[ProofBundlePartyV1]
    epoch_roots: Sequence[bytes]
    final_root32: bytes
    signatures: Sequence[ProofBundleSignatureV1]
    result_hash32: bytes
    status: str
    evidence: Optional[Dict[str, Any]] = None
    optional_proofs: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        eip712 = {
            "name": str(self.eip712_domain.name),
            "version": str(self.eip712_domain.version),
            "chain_id": int(self.eip712_domain.chain_id),
            "verifying_contract": "0x" + bytes(self.eip712_domain.verifying_contract).hex(),
        }
        d: Dict[str, Any] = {
            "uvcc_version": self.uvcc_version,
            "job": {
                "job_id": _hex32(self.job_id32),
                "policy_hash": _hex32(self.policy_hash32),
                "sgir_hash": _hex32(self.sgir_hash32),
                "runtime_hash": _hex32(self.runtime_hash32),
                "backend": str(self.backend),
            },
            "eip712": eip712,
            "identity": {"parties": [p.to_dict() for p in self.parties]},
            "transcript": {
                "epoch_roots": [_hex32(r) for r in self.epoch_roots],
                "final_root": _hex32(self.final_root32),
            },
            "signatures": [s.to_dict() for s in self.signatures],
            "verdict": {"result_hash": _hex32(self.result_hash32), "status": str(self.status)},
        }
        if self.evidence is not None:
            d["evidence"] = self.evidence
        if self.optional_proofs is not None:
            d["optional_proofs"] = self.optional_proofs
        return d

    def to_json_bytes(self) -> bytes:
        # Deterministic JSON encoding for hashing.
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_final_root_v1(
    *,
    party_id: int,
    privkey32: bytes,
    policy_hash32: bytes,
    final_root32: bytes,
    result_hash32: bytes,
    job_id32: bytes,
    eip712_domain: EIP712DomainV1,
) -> ProofBundleSignatureV1:
    digest32 = FinalCommitV1(job_id32=job_id32, policy_hash32=policy_hash32, final_root32=final_root32, result_hash32=result_hash32).digest32(domain=eip712_domain)
    sig65 = secp256k1_sign_hash(privkey32, digest32)
    return ProofBundleSignatureV1(party_id=int(party_id), sig65=sig65)


def party_from_privkey(*, party_id: int, privkey32: bytes) -> ProofBundlePartyV1:
    pub = secp256k1_pubkey_from_privkey(privkey32)
    return ProofBundlePartyV1(party_id=int(party_id), pubkey64=pub)


