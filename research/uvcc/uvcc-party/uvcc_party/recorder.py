from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .eip712 import EIP712DomainV1, FinalCommitV1
from .party import Party
from .proof_bundle import ProofBundlePartyV1, ProofBundleSignatureV1, ProofBundleV1, party_from_privkey
from .sig import secp256k1_sign_hash
from .transcript import TranscriptStoreV1


@dataclass
class PartyRecorderV1:
    """
    High-level recorder for a single party run.

    The Party already records per-frame transcript leaves. This recorder adds:
    - epoch finalization helpers
    - proof bundle assembly for that party
    """

    party: Party
    policy_hash32: bytes
    sgir_hash32: bytes
    runtime_hash32: bytes
    backend: str = "CRYPTO_CC_3PC"

    def __post_init__(self) -> None:
        if len(self.policy_hash32) != 32:
            raise ValueError("policy_hash32 must be 32 bytes")
        if len(self.sgir_hash32) != 32:
            raise ValueError("sgir_hash32 must be 32 bytes")
        if len(self.runtime_hash32) != 32:
            raise ValueError("runtime_hash32 must be 32 bytes")
        if self.party.transcript is None:
            raise ValueError("party.transcript is required")

    @property
    def transcript(self) -> TranscriptStoreV1:
        assert self.party.transcript is not None
        return self.party.transcript

    def epoch_root(self, *, epoch: int) -> bytes:
        return self.transcript.epoch_root(epoch=epoch)

    def final_root(self, *, epoch_count: int) -> bytes:
        return self.transcript.final_root(epoch_count=epoch_count)

    def build_proof_bundle(
        self,
        *,
        epoch_count: int,
        party_privkey32: bytes,
        result_hash32: bytes,
        eip712_domain: EIP712DomainV1,
        status: str = "OK",
        extra_parties: Optional[Sequence[ProofBundlePartyV1]] = None,
        extra_signatures: Optional[Sequence[ProofBundleSignatureV1]] = None,
    ) -> ProofBundleV1:
        if len(result_hash32) != 32:
            raise ValueError("result_hash32 must be 32 bytes")
        roots = [self.epoch_root(epoch=e) for e in range(int(epoch_count))]
        final_root32 = self.final_root(epoch_count=int(epoch_count))

        digest32 = FinalCommitV1(
            job_id32=self.party.job_id32,
            policy_hash32=self.policy_hash32,
            final_root32=final_root32,
            result_hash32=bytes(result_hash32),
        ).digest32(domain=eip712_domain)
        sig65 = secp256k1_sign_hash(party_privkey32, digest32)
        sig = ProofBundleSignatureV1(party_id=int(self.party.party_id), sig65=sig65)

        parties: List[ProofBundlePartyV1] = [party_from_privkey(party_id=int(self.party.party_id), privkey32=party_privkey32)]
        if extra_parties is not None:
            parties.extend(list(extra_parties))
        sigs: List[ProofBundleSignatureV1] = [sig]
        if extra_signatures is not None:
            sigs.extend(list(extra_signatures))

        return ProofBundleV1(
            uvcc_version="1.0",
            job_id32=self.party.job_id32,
            policy_hash32=self.policy_hash32,
            eip712_domain=eip712_domain,
            sgir_hash32=self.sgir_hash32,
            runtime_hash32=self.runtime_hash32,
            backend=self.backend,
            parties=parties,
            epoch_roots=roots,
            final_root32=final_root32,
            signatures=sigs,
            result_hash32=result_hash32,
            status=str(status),
        )


