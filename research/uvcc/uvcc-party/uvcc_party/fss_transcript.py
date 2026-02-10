from __future__ import annotations

import struct
from typing import Optional

from .fss_block import fssblock_hash32_v1
from .fss_plan import fsseval_hash32_v1
from .party import Party
from .transcript import sha256


# Local transcript-only msg kinds (do not go on the wire).
MSG_FSSBLOCK_COMMIT = 0x0001
MSG_FSSEVAL_COMMIT = 0x0002


def transcript_record_fssblock_commit_v1(*, party: Party, epoch: int, step: int, round: int, block_bytes: bytes) -> Optional[bytes]:
    if party.transcript is None:
        return None
    h_fssblock = fssblock_hash32_v1(block_bytes)
    hdr_hash32 = sha256(b"UVCC.fssblock.commit.hdr.v1\0" + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF))
    party.transcript.record_frame(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_FSSBLOCK_COMMIT,
        sender=int(party.party_id),
        receiver=int(party.party_id),
        dir=0,
        seq_no=0,
        payload_bytes=len(block_bytes),
        payload_hash32=h_fssblock,
        header_hash32=hdr_hash32,
        segments=[],
    )
    return h_fssblock


def transcript_record_fsseval_commit_v1(
    *,
    party: Party,
    epoch: int,
    step: int,
    round: int,
    plan_prime_bytes: bytes,
    tasks_bytes: bytes,
    fssblock_hash32: bytes,
) -> Optional[bytes]:
    if party.transcript is None:
        return None
    h_fsseval = fsseval_hash32_v1(plan_prime_bytes=plan_prime_bytes, tasks_bytes=tasks_bytes, fssblock_hash32=fssblock_hash32)
    hdr_hash32 = sha256(
        b"UVCC.fsseval.commit.hdr.v1\0"
        + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
        + struct.pack("<I", (len(tasks_bytes) // 76) & 0xFFFFFFFF)
    )
    party.transcript.record_frame(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_FSSEVAL_COMMIT,
        sender=int(party.party_id),
        receiver=int(party.party_id),
        dir=0,
        seq_no=0,
        payload_bytes=len(plan_prime_bytes) + len(tasks_bytes) + 32,
        payload_hash32=h_fsseval,
        header_hash32=hdr_hash32,
        segments=[],
    )
    return h_fsseval


