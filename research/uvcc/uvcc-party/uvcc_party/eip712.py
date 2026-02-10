from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass

from eth_utils.crypto import keccak


def keccak256(data: bytes) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    return bytes(keccak(bytes(data)))


def _abi_word_bytes32(x32: bytes) -> bytes:
    if not isinstance(x32, (bytes, bytearray)) or len(x32) != 32:
        raise ValueError("expected 32-byte value")
    return bytes(x32)


def _abi_word_uint(x: int) -> bytes:
    # ABI encodes all uint<M> as 32-byte big-endian words.
    if int(x) < 0:
        raise ValueError("uint must be >= 0")
    return int(x).to_bytes(32, "big", signed=False)


def _abi_word_address(addr20: bytes) -> bytes:
    if not isinstance(addr20, (bytes, bytearray)) or len(addr20) != 20:
        raise ValueError("expected 20-byte address")
    return b"\x00" * 12 + bytes(addr20)


EIP712_DOMAIN_TYPEHASH = keccak256(b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)")

# Must match privacy_new.txt exactly (no whitespace changes).
POLICY_COMMIT_TYPEHASH = keccak256(
    b"PolicyCommit(bytes32 jobId,bytes32 policyHash,bytes32 sidHash,bytes32 sgirHash,bytes32 runtimeHash,bytes32 fssDirHash,bytes32 preprocHash,uint8 backend,uint64 epoch)"
)
FINAL_COMMIT_TYPEHASH = keccak256(b"FinalCommit(bytes32 jobId,bytes32 policyHash,bytes32 finalRoot,bytes32 resultHash)")


@dataclass(frozen=True)
class EIP712DomainV1:
    """
    Canonical EIP-712 domain for UVCC v1 (privacy_new.txt §3.2 / profile v1).
    """

    chain_id: int
    verifying_contract: bytes  # 20 bytes address
    name: str = "UVCC"
    version: str = "1"

    def __post_init__(self) -> None:
        if int(self.chain_id) < 0:
            raise ValueError("chain_id must be >= 0")
        if not isinstance(self.verifying_contract, (bytes, bytearray)) or len(self.verifying_contract) != 20:
            raise ValueError("verifying_contract must be 20 bytes")
        if str(self.name) != "UVCC":
            raise ValueError("name must be UVCC")
        if str(self.version) != "1":
            raise ValueError("version must be 1")

    def separator32(self) -> bytes:
        name_hash = keccak256(self.name.encode("utf-8"))
        version_hash = keccak256(self.version.encode("utf-8"))
        enc = b"".join(
            [
                _abi_word_bytes32(EIP712_DOMAIN_TYPEHASH),
                _abi_word_bytes32(name_hash),
                _abi_word_bytes32(version_hash),
                _abi_word_uint(int(self.chain_id)),
                _abi_word_address(bytes(self.verifying_contract)),
            ]
        )
        return keccak256(enc)


def eip712_digest_v1(*, domain: EIP712DomainV1, struct_hash32: bytes) -> bytes:
    if not isinstance(struct_hash32, (bytes, bytearray)) or len(struct_hash32) != 32:
        raise ValueError("struct_hash32 must be 32 bytes")
    return keccak256(b"\x19\x01" + domain.separator32() + bytes(struct_hash32))


@dataclass(frozen=True)
class PolicyCommitV1:
    job_id32: bytes
    policy_hash32: bytes
    sid_hash32: bytes
    sgir_hash32: bytes
    runtime_hash32: bytes
    fss_dir_hash32: bytes
    preproc_hash32: bytes
    backend_u8: int
    epoch_u64: int

    def __post_init__(self) -> None:
        for name, b in [
            ("job_id32", self.job_id32),
            ("policy_hash32", self.policy_hash32),
            ("sid_hash32", self.sid_hash32),
            ("sgir_hash32", self.sgir_hash32),
            ("runtime_hash32", self.runtime_hash32),
            ("fss_dir_hash32", self.fss_dir_hash32),
            ("preproc_hash32", self.preproc_hash32),
        ]:
            if not isinstance(b, (bytes, bytearray)) or len(b) != 32:
                raise ValueError(f"{name} must be 32 bytes")
        if int(self.backend_u8) < 0 or int(self.backend_u8) > 255:
            raise ValueError("backend_u8 must be 0..255")
        if int(self.epoch_u64) < 0 or int(self.epoch_u64) > (2**64 - 1):
            raise ValueError("epoch_u64 must be u64")

    def struct_hash32(self) -> bytes:
        enc = b"".join(
            [
                _abi_word_bytes32(POLICY_COMMIT_TYPEHASH),
                _abi_word_bytes32(self.job_id32),
                _abi_word_bytes32(self.policy_hash32),
                _abi_word_bytes32(self.sid_hash32),
                _abi_word_bytes32(self.sgir_hash32),
                _abi_word_bytes32(self.runtime_hash32),
                _abi_word_bytes32(self.fss_dir_hash32),
                _abi_word_bytes32(self.preproc_hash32),
                _abi_word_uint(int(self.backend_u8)),
                _abi_word_uint(int(self.epoch_u64)),
            ]
        )
        return keccak256(enc)

    def digest32(self, *, domain: EIP712DomainV1) -> bytes:
        return eip712_digest_v1(domain=domain, struct_hash32=self.struct_hash32())


@dataclass(frozen=True)
class FinalCommitV1:
    job_id32: bytes
    policy_hash32: bytes
    final_root32: bytes
    result_hash32: bytes

    def __post_init__(self) -> None:
        for name, b in [
            ("job_id32", self.job_id32),
            ("policy_hash32", self.policy_hash32),
            ("final_root32", self.final_root32),
            ("result_hash32", self.result_hash32),
        ]:
            if not isinstance(b, (bytes, bytearray)) or len(b) != 32:
                raise ValueError(f"{name} must be 32 bytes")

    def struct_hash32(self) -> bytes:
        enc = b"".join(
            [
                _abi_word_bytes32(FINAL_COMMIT_TYPEHASH),
                _abi_word_bytes32(self.job_id32),
                _abi_word_bytes32(self.policy_hash32),
                _abi_word_bytes32(self.final_root32),
                _abi_word_bytes32(self.result_hash32),
            ]
        )
        return keccak256(enc)

    def digest32(self, *, domain: EIP712DomainV1) -> bytes:
        return eip712_digest_v1(domain=domain, struct_hash32=self.struct_hash32())


