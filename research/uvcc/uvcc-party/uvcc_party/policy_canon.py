from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b2265a18869da013

import struct
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

from eth_utils.crypto import keccak

from .cbor_det import cbor_dumps_det_v1


def keccak256(b: bytes) -> bytes:
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError("expected bytes")
    return bytes(keccak(bytes(b)))


BackendV1 = Literal["GPU_TEE", "CRYPTO_CC_3PC"]


@dataclass(frozen=True)
class PolicyPartyV1:
    party_id: int
    addr20: bytes
    domain: str = ""

    def __post_init__(self) -> None:
        if int(self.party_id) not in (0, 1, 2):
            raise ValueError("party_id must be 0..2")
        if not isinstance(self.addr20, (bytes, bytearray)) or len(self.addr20) != 20:
            raise ValueError("addr20 must be 20 bytes")
        if not isinstance(self.domain, str):
            raise TypeError("domain must be str")


@dataclass(frozen=True)
class PolicyV1:
    uvcc_version: str
    backend: BackendV1
    sid: bytes
    flags_u32: int
    job_id_u64: int
    sgir_hash32: bytes
    runtime_hash32: bytes
    fss_dir_hash32: bytes
    preproc_hash32: bytes
    parties: Tuple[PolicyPartyV1, PolicyPartyV1, PolicyPartyV1]

    def __post_init__(self) -> None:
        if str(self.uvcc_version) != "1.0":
            raise ValueError("uvcc_version must be 1.0")
        if str(self.backend) not in ("GPU_TEE", "CRYPTO_CC_3PC"):
            raise ValueError("backend must be GPU_TEE or CRYPTO_CC_3PC")
        if not isinstance(self.sid, (bytes, bytearray)):
            raise TypeError("sid must be bytes")
        if int(self.flags_u32) < 0 or int(self.flags_u32) > 0xFFFFFFFF:
            raise ValueError("flags_u32 must be u32")
        if int(self.job_id_u64) < 0 or int(self.job_id_u64) > (2**64 - 1):
            raise ValueError("job_id_u64 must be u64")
        for name, b in [
            ("sgir_hash32", self.sgir_hash32),
            ("runtime_hash32", self.runtime_hash32),
            ("fss_dir_hash32", self.fss_dir_hash32),
            ("preproc_hash32", self.preproc_hash32),
        ]:
            if len(b) != 32:
                raise ValueError(f"{name} must be 32 bytes")
        if len(self.parties) != 3:
            raise ValueError("parties must have length 3")
        # Enforce canonical party_id set {0,1,2}
        ids = sorted(int(p.party_id) for p in self.parties)
        if ids != [0, 1, 2]:
            raise ValueError("parties.party_id must be 0,1,2 exactly once")

    def to_cbor_map(self) -> Dict[str, Any]:
        parties_sorted = sorted(self.parties, key=lambda p: int(p.party_id))
        return {
            "uvcc_version": self.uvcc_version,
            "backend": self.backend,
            "sid": bytes(self.sid),
            "flags_u32": int(self.flags_u32),
            "job_id_u64": int(self.job_id_u64),
            "sgir_hash32": bytes(self.sgir_hash32),
            "runtime_hash32": bytes(self.runtime_hash32),
            "fss_dir_hash32": bytes(self.fss_dir_hash32),
            "preproc_hash32": bytes(self.preproc_hash32),
            "parties": [
                {"party_id": int(p.party_id), "addr20": bytes(p.addr20), "domain": str(p.domain)}
                for p in parties_sorted
            ],
        }

    def sid_hash32(self) -> bytes:
        # v1 profile: sid_hash32 = keccak256(sid bytes)
        return keccak256(bytes(self.sid))

    def to_policy_wire_bytes_v1(self) -> bytes:
        """
        Canonical binary policy wire format v1 (fixed blocks only).

        Layout (in order):
          - uvcc_policy_hdr_v1 (64)
          - uvcc_policy_digests_v1 (128)
          - uvcc_policy_party_v1 * 3 (64*3)
        """
        backend_u8 = 0 if str(self.backend) == "CRYPTO_CC_3PC" else 1
        hdr = struct.pack(
            "<8sHBBI32sQQ",
            b"UVCCPOL1",
            1,  # version
            int(backend_u8) & 0xFF,
            3,  # party_count
            int(self.flags_u32) & 0xFFFFFFFF,
            self.sid_hash32(),
            int(self.job_id_u64) & 0xFFFFFFFFFFFFFFFF,
            0,
        )
        if len(hdr) != 64:
            raise AssertionError("policy hdr must be 64 bytes")
        dig = bytes(self.sgir_hash32) + bytes(self.runtime_hash32) + bytes(self.fss_dir_hash32) + bytes(self.preproc_hash32)
        if len(dig) != 128:
            raise AssertionError("policy digests must be 128 bytes")
        parties_sorted = sorted(self.parties, key=lambda p: int(p.party_id))
        body = bytearray(hdr)
        body += dig
        for p in parties_sorted:
            # uvcc_policy_party_v1 (64 bytes)
            rec = bytearray()
            rec += struct.pack("<BB", int(p.party_id) & 0xFF, 1)  # sig_scheme=1 (ECDSA_secp256k1)
            rec += bytes(p.addr20)
            rec += struct.pack("<B", 0)  # attn_type=0 (none)
            rec += b"\x00" * 32  # attn_policy_hash32
            rec += b"\x00" * 9  # reserved0
            if len(rec) != 64:
                raise AssertionError("policy party record must be 64 bytes")
            body += rec
        return bytes(body)


def policy_cbor_bytes_v1(policy: PolicyV1) -> bytes:
    return cbor_dumps_det_v1(policy.to_cbor_map())


def policy_hash32_v1(policy: PolicyV1) -> bytes:
    """
    policy_hash32 := keccak256(policy_wire_v1_bytes)
    """
    return keccak256(policy.to_policy_wire_bytes_v1())


