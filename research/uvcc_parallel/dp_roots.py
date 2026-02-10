from __future__ import annotations

"""
Transcript-of-transcripts root helpers for PARALLEL DP/PP/TP.

Implements the hashing scheme described in `research/PARALLEL.txt` §14:

Replica root (per replica r, per epoch):
  replica_root[r] = SHA256(
    "UVCC_REPLICA_ROOT_V1" ||
    sid_rep[r] ||
    LE32(epoch) ||
    concat_{(s,t) in lex order} epoch_root[r,s,t]
  )

Global root (per epoch):
  global_root = SHA256(
    "UVCC_GLOBAL_ROOT_V1" ||
    sid_job ||
    LE32(epoch) ||
    concat_{r=0..R-1} replica_root[r]
  )

Notes:
- These roots are *higher-level commitments* over many subgroup transcript roots.
- v1 UVCC verifier does not need to understand the internal structure if the orchestrator
  produces correct epoch roots and the proof bundle commits to them.
"""

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _le32(x: int) -> bytes:
    return struct.pack("<I", int(x) & 0xFFFFFFFF)


@dataclass(frozen=True)
class SubgroupEpochRoot:
    stage: int
    tp_rank: int
    epoch_root32: bytes

    def __post_init__(self) -> None:
        if int(self.stage) < 0 or int(self.stage) > 255:
            raise ValueError("stage must be in [0,255]")
        if int(self.tp_rank) < 0 or int(self.tp_rank) > 65535:
            raise ValueError("tp_rank must be in [0,65535]")
        if not isinstance(self.epoch_root32, (bytes, bytearray)) or len(self.epoch_root32) != 32:
            raise ValueError("epoch_root32 must be 32 bytes")


def replica_root_v1(*, sid_rep: bytes, epoch: int, subgroup_epoch_roots: Sequence[SubgroupEpochRoot]) -> bytes:
    if not isinstance(sid_rep, (bytes, bytearray)) or len(sid_rep) != 32:
        raise ValueError("sid_rep must be 32 bytes")
    e = int(epoch)
    if e < 0:
        raise ValueError("epoch must be >= 0")

    subs = list(subgroup_epoch_roots)
    subs_sorted = sorted(subs, key=lambda x: (int(x.stage), int(x.tp_rank)))
    concat = b"".join(bytes(x.epoch_root32) for x in subs_sorted)
    return sha256(b"UVCC_REPLICA_ROOT_V1\0" + bytes(sid_rep) + _le32(e) + concat)


def global_root_v1(*, sid_job: bytes, epoch: int, replica_roots: Sequence[bytes]) -> bytes:
    if not isinstance(sid_job, (bytes, bytearray)) or len(sid_job) != 32:
        raise ValueError("sid_job must be 32 bytes")
    e = int(epoch)
    if e < 0:
        raise ValueError("epoch must be >= 0")
    rs = [bytes(r) for r in replica_roots]
    if any(len(r) != 32 for r in rs):
        raise ValueError("each replica_root must be 32 bytes")
    return sha256(b"UVCC_GLOBAL_ROOT_V1\0" + bytes(sid_job) + _le32(e) + b"".join(rs))


def replica_root_from_map_v1(*, sid_rep: bytes, epoch: int, roots_by_sub: Dict[Tuple[int, int], bytes]) -> bytes:
    subs: List[SubgroupEpochRoot] = []
    for (s, t), r32 in roots_by_sub.items():
        subs.append(SubgroupEpochRoot(stage=int(s), tp_rank=int(t), epoch_root32=bytes(r32)))
    return replica_root_v1(sid_rep=bytes(sid_rep), epoch=int(epoch), subgroup_epoch_roots=subs)


def global_root_from_replica_maps_v1(
    *, sid_job: bytes, epoch: int, replica_entries: Iterable[Tuple[int, bytes, Dict[Tuple[int, int], bytes]]]
) -> Tuple[bytes, List[bytes]]:
    """
    Compute the global_root and also return the per-replica roots (ordered by replica_id).

    replica_entries yields tuples: (replica_id, sid_rep, roots_by_sub[(s,t)] = epoch_root32).
    """
    entries = [(int(rid), bytes(sid_rep), dict(m)) for (rid, sid_rep, m) in replica_entries]
    if not entries:
        raise ValueError("replica_entries must be non-empty")
    entries_sorted = sorted(entries, key=lambda x: x[0])
    roots: List[bytes] = []
    for rid, sid_rep, roots_by_sub in entries_sorted:
        _ = rid
        roots.append(replica_root_from_map_v1(sid_rep=sid_rep, epoch=int(epoch), roots_by_sub=roots_by_sub))
    glob = global_root_v1(sid_job=bytes(sid_job), epoch=int(epoch), replica_roots=roots)
    return glob, roots


