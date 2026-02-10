from __future__ import annotations

"""
Deterministic session-id derivations for PARALLEL DP/PP/TP on top of UVCC.

This file intentionally does NOT change any existing UVCC message-id derivations.
Instead, it follows the rule in `research/PARALLEL.txt`:

  "Do not change your existing derivations. Instead, derive a unique sid per parallel subgroup."

We expose:
- sid_job: top-level 32B session id for a DP job
- sid_rep[r]: replica sid for SR-DP replica r
- sid_sub[r,s,t]: subgroup sid for (replica r, pipeline stage s, tensor rank t)

All are bytes (32 bytes).
"""

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Optional, Tuple


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _le32(x: int) -> bytes:
    return struct.pack("<I", int(x) & 0xFFFFFFFF)


def _le16(x: int) -> bytes:
    return struct.pack("<H", int(x) & 0xFFFF)


def _u8(x: int) -> bytes:
    return bytes([int(x) & 0xFF])


def new_sid_job(seed32: Optional[bytes] = None) -> bytes:
    """
    Create a job-level sid (32 bytes).

    - If seed32 is provided: deterministic sid_job = SHA256("UVCC_SID_JOB_V1" || seed32)
    - Else: sid_job = SHA256("UVCC_SID_JOB_V1" || os.urandom(32))
    """
    if seed32 is None:
        seed32 = os.urandom(32)
    if not isinstance(seed32, (bytes, bytearray)) or len(seed32) != 32:
        raise ValueError("seed32 must be 32 bytes")
    return sha256(b"UVCC_SID_JOB_V1\0" + bytes(seed32))


def sid_replica_v1(*, sid_job: bytes, replica_id: int) -> bytes:
    """
    sid_rep[r] = SHA256("UVCC_SID_REPLICA_V1" || sid_job || LE32(r))
    """
    if not isinstance(sid_job, (bytes, bytearray)) or len(sid_job) != 32:
        raise ValueError("sid_job must be 32 bytes")
    r = int(replica_id)
    if r < 0:
        raise ValueError("replica_id must be >= 0")
    return sha256(b"UVCC_SID_REPLICA_V1\0" + bytes(sid_job) + _le32(r))


def sid_subgroup_v1(*, sid_rep: bytes, stage: int, tp_rank: int) -> bytes:
    """
    sid_sub[r,s,t] = SHA256("UVCC_SID_SUB_V1" || sid_rep[r] || U8(s) || LE16(t))
    """
    if not isinstance(sid_rep, (bytes, bytearray)) or len(sid_rep) != 32:
        raise ValueError("sid_rep must be 32 bytes")
    s = int(stage)
    t = int(tp_rank)
    if s < 0 or s > 255:
        raise ValueError("stage must be in [0,255]")
    if t < 0 or t > 65535:
        raise ValueError("tp_rank must be in [0,65535]")
    return sha256(b"UVCC_SID_SUB_V1\0" + bytes(sid_rep) + _u8(s) + _le16(t))


def relay_group_id_for_replica(*, sid_rep: bytes, replica_id: int, prefix: str = "uvcc-dp") -> str:
    """
    A stable relay group_id for a given replica.

    Notes:
    - Relay dedup key is (group_id, msg_id), so group_id separation is a nice safety belt.
    - msg_id already incorporates sid_hash32, so sid separation is the primary collision-avoidance.
    """
    if not isinstance(sid_rep, (bytes, bytearray)) or len(sid_rep) != 32:
        raise ValueError("sid_rep must be 32 bytes")
    r = int(replica_id)
    if r < 0:
        raise ValueError("replica_id must be >= 0")
    p = str(prefix).strip() or "uvcc-dp"
    tag = bytes(sid_rep)[:4].hex()
    return f"{p}-r{r}-{tag}"


@dataclass(frozen=True)
class SubgroupCoord:
    replica_id: int
    stage: int = 0
    tp_rank: int = 0

    def __post_init__(self) -> None:
        if int(self.replica_id) < 0:
            raise ValueError("replica_id must be >= 0")
        if int(self.stage) < 0 or int(self.stage) > 255:
            raise ValueError("stage must be in [0,255]")
        if int(self.tp_rank) < 0 or int(self.tp_rank) > 65535:
            raise ValueError("tp_rank must be in [0,65535]")


def derive_all_v1(*, sid_job: bytes, coord: SubgroupCoord) -> Tuple[bytes, bytes]:
    """
    Convenience helper:
    - returns (sid_rep, sid_sub) for a subgroup coordinate.
    """
    sid_rep = sid_replica_v1(sid_job=bytes(sid_job), replica_id=int(coord.replica_id))
    sid_sub = sid_subgroup_v1(sid_rep=sid_rep, stage=int(coord.stage), tp_rank=int(coord.tp_rank))
    return sid_rep, sid_sub


