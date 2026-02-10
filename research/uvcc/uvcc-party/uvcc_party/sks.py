from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_ba7afac425406f12

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .dpf_dcf import _chacha12_block_u32  # reuse canonical ChaCha12 core
from .gemm import _matmul_u64 as _matmul_u64  # u64 ring matmul (CUDA kernel when on GPU)
from .netframe import DT_U64, DT_U8, SegmentPayloadV1, build_netframe_v1
from .party import DEFAULT_NET_TIMEOUT_S, DEFAULT_RELAY_TTL_S, Party
from .rss import RSSArithU64
from .transcript import sha256


# SKS transcript leaf types (privacy_new.txt §4.3).
LEAF_SKS_EPOCH_COMMIT_V1 = 0x70
LEAF_SKS_EPOCH_REVEAL_V1 = 0x71
LEAF_SKS_CHECK_META_V1 = 0x72
LEAF_SKS_OPEN_COMMIT_V1 = 0x73
LEAF_SKS_OPEN_RESULT_V1 = 0x74


# Domain separation strings (ASCII, no NUL) per privacy_new.txt §4.1.
DS_EPOCH_COMMIT = b"UVCC.SKS.EPOCH_COMMIT.v1"
DS_EPOCH_RAND = b"UVCC.SKS.EPOCH_RAND.v1"
DS_CHK_SEED = b"UVCC.SKS.CHK_SEED.v1"
DS_PRGSPLIT = b"UVCC.SKS.PRGSPLIT.v1"


def _u64_to_i64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    return x if x < (1 << 63) else x - (1 << 64)


def _i64_to_u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _matvec_u64(A_u64_i64: torch.Tensor, v_u64_i64: torch.Tensor) -> torch.Tensor:
    """
    u64 ring matvec (mod 2^64) where tensors are int64 carrying u64 bit-patterns.
    Uses the same CUDA u64 matmul kernel by treating v as a column matrix.
    """
    if A_u64_i64.dtype != torch.int64 or v_u64_i64.dtype != torch.int64:
        raise TypeError("matvec_u64 expects int64 tensors (u64 bit-patterns)")
    if A_u64_i64.ndim != 2:
        raise ValueError("matvec_u64 expects A to be 2D")
    v = v_u64_i64.view(-1)
    if int(A_u64_i64.shape[1]) != int(v.shape[0]):
        raise ValueError("matvec_u64 shape mismatch")
    out = _matmul_u64(A_u64_i64, v.view(-1, 1))
    return out.view(-1)


def _u64_tensor_to_le_bytes(x_u64_i64: torch.Tensor) -> bytes:
    x = x_u64_i64.contiguous().view(-1).cpu().tolist()
    out = bytearray()
    for v in x:
        out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _u8_tensor_to_bytes(x_u8: torch.Tensor) -> bytes:
    return bytes(bytearray(int(v) & 0xFF for v in x_u8.contiguous().view(-1).cpu().tolist()))


def _bytes_xor(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("xor length mismatch")
    return bytes(x ^ y for x, y in zip(a, b))


def sks_epoch_commit_hash_v1(*, sid: bytes, epoch: int, pid: int, nonce32: bytes) -> bytes:
    if len(nonce32) != 32:
        raise ValueError("nonce32 must be 32 bytes")
    h = hashlib.sha256()
    h.update(DS_EPOCH_COMMIT)
    h.update(bytes(sid))
    h.update(struct.pack("<I", int(epoch) & 0xFFFFFFFF))
    h.update(struct.pack("<B", int(pid) & 0xFF))
    h.update(bytes(nonce32))
    return h.digest()


def sks_epoch_rand_v1(*, sid: bytes, epoch: int, nonce0: bytes, nonce1: bytes, nonce2: bytes) -> bytes:
    if len(nonce0) != 32 or len(nonce1) != 32 or len(nonce2) != 32:
        raise ValueError("nonces must be 32 bytes")
    nonce_xor = _bytes_xor(_bytes_xor(nonce0, nonce1), nonce2)
    h = hashlib.sha256()
    h.update(DS_EPOCH_RAND)
    h.update(bytes(sid))
    h.update(struct.pack("<I", int(epoch) & 0xFFFFFFFF))
    h.update(nonce_xor)
    return h.digest()


def sks_check_seed_v1(*, epoch_rand32: bytes, sgir_op_id: int, kernel_instance_id: int) -> bytes:
    if len(epoch_rand32) != 32:
        raise ValueError("epoch_rand32 must be 32 bytes")
    h = hashlib.sha256()
    h.update(DS_CHK_SEED)
    h.update(epoch_rand32)
    h.update(struct.pack("<I", int(sgir_op_id) & 0xFFFFFFFF))
    h.update(struct.pack("<I", int(kernel_instance_id) & 0xFFFFFFFF))
    return h.digest()


def sks_is_selected_v1(*, check_seed32: bytes, sks_sample_log2: int) -> bool:
    if len(check_seed32) != 32:
        raise ValueError("check_seed32 must be 32 bytes")
    s = int(sks_sample_log2)
    if s < 0 or s > 32:
        raise ValueError("sks_sample_log2 must be 0..32")
    u = int.from_bytes(check_seed32[0:4], "little", signed=False)
    if s == 0:
        return True
    return (u & ((1 << s) - 1)) == 0


def _chacha12_stream_bytes(*, key32: bytes, n_bytes: int) -> bytes:
    if len(key32) != 32:
        raise ValueError("key32 must be 32 bytes")
    if n_bytes < 0:
        raise ValueError("n_bytes must be >= 0")
    key_words = [int.from_bytes(key32[4 * i : 4 * i + 4], "little", signed=False) for i in range(8)]
    key8 = torch.tensor([key_words], dtype=torch.int64)
    nonce3 = torch.zeros((1, 3), dtype=torch.int64)
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        out16 = _chacha12_block_u32(key8, nonce3, counter32=int(counter))[0].tolist()
        for w in out16:
            out += int(w & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        counter += 1
    return bytes(out[:n_bytes])


def _sks_prg_key_v1(*, check_seed32: bytes) -> bytes:
    if len(check_seed32) != 32:
        raise ValueError("check_seed32 must be 32 bytes")
    return hashlib.sha256(DS_PRGSPLIT + check_seed32).digest()


def sks_challenge_vectors_ring_u64_v1(*, check_seed32: bytes, m: int, n: int, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ring-mode (Z2^64) challenges: r in u64^n, s in u64^{t x m}.
    """
    m = int(m)
    n = int(n)
    t = int(t)
    if m <= 0 or n <= 0 or t <= 0:
        raise ValueError("m,n,t must be > 0")
    prg_key = _sks_prg_key_v1(check_seed32=check_seed32)
    need_u64 = n + t * m
    bs = _chacha12_stream_bytes(key32=prg_key, n_bytes=need_u64 * 8)
    vals = [int.from_bytes(bs[8 * i : 8 * i + 8], "little", signed=False) for i in range(need_u64)]
    r = torch.tensor([_u64_to_i64(v) for v in vals[:n]], dtype=torch.int64)
    s = torch.tensor([_u64_to_i64(v) for v in vals[n:]], dtype=torch.int64).view(t, m)
    return r, s


def _rss_add_public_vec_into_share0(party_id: int, x: RSSArithU64, pub_u64_i64: torch.Tensor) -> RSSArithU64:
    if pub_u64_i64.dtype != torch.int64 or pub_u64_i64.shape != x.lo.shape:
        raise TypeError("pub_u64_i64 must match RSS shape/dtype")
    pid = int(party_id)
    lo = x.lo
    hi = x.hi
    if pid == 0:
        lo = lo + pub_u64_i64
    elif pid == 2:
        hi = hi + pub_u64_i64
    return RSSArithU64(lo=lo, hi=hi, fxp_frac_bits=x.fxp_frac_bits)


def sks_freivalds_residual_tile_u64_v1(
    party: Party,
    *,
    Z: RSSArithU64,
    triple_A: RSSArithU64,
    triple_B: RSSArithU64,
    triple_C: RSSArithU64,
    E_pub: torch.Tensor,
    F_pub: torch.Tensor,
    r_u64_i64: torch.Tensor,
) -> RSSArithU64:
    """
    Compute tvec = X(Yr) - Zr for a tile GEMM instance using Beaver artifacts:
      X = A + E_pub,  Y = B + F_pub,  A*B = C.
    """
    r = r_u64_i64.view(-1)
    # B*r (secret vector)
    br = RSSArithU64(lo=_matvec_u64(triple_B.lo, r), hi=_matvec_u64(triple_B.hi, r))
    # F*r (public vector)
    fr_pub = _matvec_u64(F_pub, r)
    # E*(B*r) (public matrix times secret vector)
    ebr = RSSArithU64(lo=_matvec_u64(E_pub, br.lo), hi=_matvec_u64(E_pub, br.hi))
    # A*(F*r) (secret matrix times public vector)
    afr = RSSArithU64(lo=_matvec_u64(triple_A.lo, fr_pub), hi=_matvec_u64(triple_A.hi, fr_pub))
    # E*(F*r) (public vector)
    efr_pub = _matvec_u64(E_pub, fr_pub)
    # (A*B)*r = C*r (secret vector)
    cr = RSSArithU64(lo=_matvec_u64(triple_C.lo, r), hi=_matvec_u64(triple_C.hi, r))
    # Z*r (secret vector)
    zr = RSSArithU64(lo=_matvec_u64(Z.lo, r), hi=_matvec_u64(Z.hi, r))

    tvec = cr.add(ebr).add(afr)
    tvec = _rss_add_public_vec_into_share0(int(party.party_id), tvec, efr_pub)
    tvec = tvec.sub(zr)
    return tvec


def sks_dot_compress_v1(*, tvec: RSSArithU64, s_u64_i64: torch.Tensor) -> RSSArithU64:
    """
    z_j = <tvec, s_j> for each row s_j (public), returning RSS shares of z vector (len t).
    """
    if s_u64_i64.dtype != torch.int64 or s_u64_i64.ndim != 2:
        raise TypeError("s_u64_i64 must be int64 matrix [t,m]")
    # (t,m) * (m,) -> (t,)
    z_lo = (s_u64_i64 * tvec.lo.view(1, -1)).sum(dim=1).to(torch.int64)
    z_hi = (s_u64_i64 * tvec.hi.view(1, -1)).sum(dim=1).to(torch.int64)
    return RSSArithU64(lo=z_lo, hi=z_hi, fxp_frac_bits=0)


def _sks_seqno_v1(*, sgir_op_id: int, kernel_instance_id: int) -> int:
    # Stable u32 for SKS transport messages per (op,instance).
    return ((int(sgir_op_id) & 0xFFFF) << 16) ^ (int(kernel_instance_id) & 0xFFFF)


def _record_sks_leaf_v1(party: Party, *, msg_kind: int, epoch: int, step: int, round: int, payload: bytes) -> None:
    if party.transcript is None:
        return
    party.transcript.record_frame(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=int(msg_kind),
        sender=int(party.party_id),
        receiver=int(party.party_id),
        dir=0,
        seq_no=0,
        payload_bytes=len(payload),
        payload_hash32=sha256(payload),
        header_hash32=sha256(b"UVCC.sks.leaf.v1\0" + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF) + struct.pack("<H", int(msg_kind) & 0xFFFF)),
        segments=[],
    )


@dataclass
class SKSEpochStateV1:
    epoch: int
    nonce32: bytes
    commit32: bytes
    commits: Dict[int, bytes]
    reveals: Dict[int, bytes]
    epoch_rand32: Optional[bytes] = None


def sks_epoch_setup_v1(
    party: Party,
    *,
    sid: bytes,
    epoch: int,
    step: int,
    relay_domain: bytes = b"uvcc.sks.epoch.v1",
    nonce32: Optional[bytes] = None,
) -> SKSEpochStateV1:
    """
    One-shot commit+reveal exchange to derive epoch_rand (privacy_new.txt §6).
    """
    pid = int(party.party_id)
    if nonce32 is None:
        nonce32 = os.urandom(32)
    if len(nonce32) != 32:
        raise ValueError("nonce32 must be 32 bytes")
    commit32 = sks_epoch_commit_hash_v1(sid=sid, epoch=int(epoch), pid=int(pid), nonce32=nonce32)

    # Record local commit leaf (0x70).
    body_commit = struct.pack("<BI32s", int(pid) & 0xFF, int(epoch) & 0xFFFFFFFF, commit32)
    _record_sks_leaf_v1(party, msg_kind=LEAF_SKS_EPOCH_COMMIT_V1, epoch=int(epoch), step=int(step), round=0, payload=body_commit)

    # Send commit to both other parties.
    for dst in (0, 1, 2):
        if dst == pid:
            continue
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=0,
            msg_kind=int(LEAF_SKS_EPOCH_COMMIT_V1),
            flags=0,
            sender=int(pid),
            receiver=int(dst),
            seq_no=0,
            segments=[SegmentPayloadV1(seg_kind=1_000_070, object_id=int(epoch) & 0xFFFFFFFF, sub_id=0, dtype=DT_U8, fxp_frac_bits=0, payload=commit32)],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=relay_domain)

    commits: Dict[int, bytes] = {pid: commit32}
    for src in (0, 1, 2):
        if src == pid:
            continue
        f_in = party.recv_netframe_expect(
            epoch=int(epoch),
            step=int(step),
            round=0,
            msg_kind=int(LEAF_SKS_EPOCH_COMMIT_V1),
            sender=int(src),
            receiver=int(pid),
            seq_no=0,
            timeout_s=float(DEFAULT_NET_TIMEOUT_S),
            relay_domain=relay_domain,
        )
        seg = next((s for s in f_in.segments if int(s.seg_kind) != 1), None)
        if seg is None:
            raise ValueError("missing SKS epoch commit segment")
        commits[int(src)] = bytes(f_in.payload[int(seg.offset) : int(seg.offset) + int(seg.length)])

    # Record local reveal leaf (0x71).
    body_reveal = struct.pack("<BI32s", int(pid) & 0xFF, int(epoch) & 0xFFFFFFFF, nonce32)
    _record_sks_leaf_v1(party, msg_kind=LEAF_SKS_EPOCH_REVEAL_V1, epoch=int(epoch), step=int(step), round=1, payload=body_reveal)

    # Send reveal to both other parties.
    for dst in (0, 1, 2):
        if dst == pid:
            continue
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=1,
            msg_kind=int(LEAF_SKS_EPOCH_REVEAL_V1),
            flags=0,
            sender=int(pid),
            receiver=int(dst),
            seq_no=0,
            segments=[SegmentPayloadV1(seg_kind=1_000_071, object_id=int(epoch) & 0xFFFFFFFF, sub_id=0, dtype=DT_U8, fxp_frac_bits=0, payload=nonce32)],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=relay_domain)

    reveals: Dict[int, bytes] = {pid: nonce32}
    for src in (0, 1, 2):
        if src == pid:
            continue
        f_in = party.recv_netframe_expect(
            epoch=int(epoch),
            step=int(step),
            round=1,
            msg_kind=int(LEAF_SKS_EPOCH_REVEAL_V1),
            sender=int(src),
            receiver=int(pid),
            seq_no=0,
            timeout_s=float(DEFAULT_NET_TIMEOUT_S),
            relay_domain=relay_domain,
        )
        seg = next((s for s in f_in.segments if int(s.seg_kind) != 1), None)
        if seg is None:
            raise ValueError("missing SKS epoch reveal segment")
        reveals[int(src)] = bytes(f_in.payload[int(seg.offset) : int(seg.offset) + int(seg.length)])

    # Verify commits and derive epoch_rand.
    for spid in (0, 1, 2):
        exp = sks_epoch_commit_hash_v1(sid=sid, epoch=int(epoch), pid=int(spid), nonce32=reveals[int(spid)])
        if exp != commits[int(spid)]:
            raise ValueError("SKS epoch commit mismatch")
    epoch_rand32 = sks_epoch_rand_v1(sid=sid, epoch=int(epoch), nonce0=reveals[0], nonce1=reveals[1], nonce2=reveals[2])
    return SKSEpochStateV1(epoch=int(epoch), nonce32=nonce32, commit32=commit32, commits=commits, reveals=reveals, epoch_rand32=epoch_rand32)


def sks_open_commit_then_open_u64_v1(
    party: Party,
    *,
    z: RSSArithU64,  # shape (t,)
    sid: bytes,
    epoch: int,
    step: int,
    sgir_op_id: int,
    kernel_instance_id: int,
    field_id: int,
    relay_domain: bytes = b"uvcc.sks.open.v1",
) -> Tuple[bool, List[int], Tuple[bytes, bytes, bytes]]:
    """
    Commit-then-open protocol for z scalars (privacy_new.txt §10).
    Returns (pass, z_pub_u64_list, (c0,c1,c2) open_commit hashes).
    """
    pid = int(party.party_id)
    z_lo = z.lo.contiguous().view(-1)
    z_hi = z.hi.contiguous().view(-1)
    t = int(z_lo.numel())
    if int(z_hi.numel()) != t:
        raise ValueError("z lo/hi length mismatch")

    # Commit to contrib_pid := z_pid (which is the lo component at each party).
    payload = _u64_tensor_to_le_bytes(z_lo)
    open_commit_pid = sha256(payload)

    body_commit = struct.pack(
        "<BIII BBH 32s",
        int(pid) & 0xFF,
        int(epoch) & 0xFFFFFFFF,
        int(sgir_op_id) & 0xFFFFFFFF,
        int(kernel_instance_id) & 0xFFFFFFFF,
        int(t) & 0xFF,
        int(field_id) & 0xFF,
        0,
        open_commit_pid,
    )
    _record_sks_leaf_v1(party, msg_kind=LEAF_SKS_OPEN_COMMIT_V1, epoch=int(epoch), step=int(step), round=0, payload=body_commit)

    seq_no = _sks_seqno_v1(sgir_op_id=int(sgir_op_id), kernel_instance_id=int(kernel_instance_id))

    # Exchange commit hashes with both other parties.
    for dst in (0, 1, 2):
        if dst == pid:
            continue
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=0,
            msg_kind=int(LEAF_SKS_OPEN_COMMIT_V1),
            flags=0,
            sender=int(pid),
            receiver=int(dst),
            seq_no=int(seq_no),
            segments=[SegmentPayloadV1(seg_kind=1_000_073, object_id=int(sgir_op_id) & 0xFFFFFFFF, sub_id=int(kernel_instance_id) & 0xFFFFFFFF, dtype=DT_U8, fxp_frac_bits=0, payload=open_commit_pid)],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=relay_domain)

    commits: Dict[int, bytes] = {pid: open_commit_pid}
    for src in (0, 1, 2):
        if src == pid:
            continue
        f_in = party.recv_netframe_expect(
            epoch=int(epoch),
            step=int(step),
            round=0,
            msg_kind=int(LEAF_SKS_OPEN_COMMIT_V1),
            sender=int(src),
            receiver=int(pid),
            seq_no=int(seq_no),
            timeout_s=float(DEFAULT_NET_TIMEOUT_S),
            relay_domain=relay_domain,
        )
        seg = next((s for s in f_in.segments if int(s.seg_kind) != 1), None)
        if seg is None:
            raise ValueError("missing SKS open-commit segment")
        commits[int(src)] = bytes(f_in.payload[int(seg.offset) : int(seg.offset) + int(seg.length)])

    # Verification status (on mismatch, still complete and emit FAIL leaf to avoid deadlocks).
    ok_commit = True
    if commits[pid] != open_commit_pid:
        ok_commit = False
    next_pid = (pid + 1) % 3
    hi_commit = sha256(_u64_tensor_to_le_bytes(z_hi))
    if commits[next_pid] != hi_commit:
        ok_commit = False

    # Send payload (contrib_pid values) to next party; receive prev payload.
    frame = build_netframe_v1(
        job_id32=party.job_id32,
        epoch=int(epoch),
        step=int(step),
        round=1,
        msg_kind=int(LEAF_SKS_OPEN_COMMIT_V1),
        flags=0,
        sender=int(pid),
        receiver=int((pid + 1) % 3),
        seq_no=int(seq_no),
        segments=[SegmentPayloadV1(seg_kind=1_000_073 + 1, object_id=int(sgir_op_id) & 0xFFFFFFFF, sub_id=int(kernel_instance_id) & 0xFFFFFFFF, dtype=DT_U64, fxp_frac_bits=0, payload=payload)],
    )
    party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=relay_domain)

    prev_pid = (pid + 2) % 3
    f_prev = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=1,
        msg_kind=int(LEAF_SKS_OPEN_COMMIT_V1),
        sender=int(prev_pid),
        receiver=int(pid),
        seq_no=int(seq_no),
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
        relay_domain=relay_domain,
    )
    seg = next((s for s in f_prev.segments if int(s.seg_kind) != 1), None)
    if seg is None:
        raise ValueError("missing SKS open payload segment")
    payload_prev = bytes(f_prev.payload[int(seg.offset) : int(seg.offset) + int(seg.length)])
    if sha256(payload_prev) != commits[prev_pid]:
        ok_commit = False

    # Parse prev contrib (t u64s).
    if len(payload_prev) != 8 * t:
        raise ValueError("bad SKS payload length")
    # Keep all arithmetic on the same device as z_lo/z_hi (CUDA in GPU runs).
    dev = z_lo.device
    vals_prev = [int.from_bytes(payload_prev[8 * j : 8 * j + 8], "little", signed=True) for j in range(t)]
    z_prev = torch.tensor(vals_prev, dtype=torch.int64, device=dev)

    z_pub = (z_prev + z_lo + z_hi).contiguous().view(-1)
    z_pub_u64 = [_i64_to_u64(int(v.item())) for v in z_pub]
    ok = bool(ok_commit) and all(int(v) == 0 for v in z_pub_u64)

    c0 = commits.get(0, b"\x00" * 32)
    c1 = commits.get(1, b"\x00" * 32)
    c2 = commits.get(2, b"\x00" * 32)
    flags = 1 if ok else 0
    body_res = struct.pack("<III BBH", int(epoch) & 0xFFFFFFFF, int(sgir_op_id) & 0xFFFFFFFF, int(kernel_instance_id) & 0xFFFFFFFF, int(t) & 0xFF, int(field_id) & 0xFF, int(flags) & 0xFFFF)
    body_res += c0 + c1 + c2
    for v in z_pub_u64:
        body_res += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    _record_sks_leaf_v1(party, msg_kind=LEAF_SKS_OPEN_RESULT_V1, epoch=int(epoch), step=int(step), round=2, payload=body_res)
    return ok, z_pub_u64, (c0, c1, c2)


def sks_record_check_meta_v1(
    party: Party,
    *,
    epoch: int,
    step: int,
    sgir_op_id: int,
    kernel_instance_id: int,
    selected: bool,
    t: int,
    field_id: int,
    sks_sample_log2: int,
    check_seed32: bytes,
) -> None:
    if len(check_seed32) != 32:
        raise ValueError("check_seed32 must be 32 bytes")
    body = struct.pack(
        "<IIIBBBB32s",
        int(epoch) & 0xFFFFFFFF,
        int(sgir_op_id) & 0xFFFFFFFF,
        int(kernel_instance_id) & 0xFFFFFFFF,
        1 if bool(selected) else 0,
        int(t) & 0xFF,
        int(field_id) & 0xFF,
        int(sks_sample_log2) & 0xFF,
        bytes(check_seed32),
    )
    _record_sks_leaf_v1(party, msg_kind=LEAF_SKS_CHECK_META_V1, epoch=int(epoch), step=int(step), round=0, payload=body)


def sks_freivalds_check_tile_gemm_u64_v1(
    party: Party,
    *,
    sid: bytes,
    epoch_rand32: bytes,
    epoch: int,
    step: int,
    sgir_op_id: int,
    kernel_instance_id: int,
    sks_sample_log2: int,
    t_checks: int,
    field_id: int,
    Z: RSSArithU64,
    triple_A: RSSArithU64,
    triple_B: RSSArithU64,
    triple_C: RSSArithU64,
    E_pub: torch.Tensor,
    F_pub: torch.Tensor,
) -> Optional[bool]:
    """
    Run SKS-Lite Freivalds checks for one GEMM tile instance (ring mode), emitting SKS leaves + open protocol.

    Returns:
      - None if not selected by sampling
      - True/False if selected and the opened z scalars are all zero / nonzero.
    """
    check_seed32 = sks_check_seed_v1(epoch_rand32=epoch_rand32, sgir_op_id=int(sgir_op_id), kernel_instance_id=int(kernel_instance_id))
    selected = sks_is_selected_v1(check_seed32=check_seed32, sks_sample_log2=int(sks_sample_log2))
    if not selected:
        return None

    # Record meta leaf (helpful for auditors/debug).
    sks_record_check_meta_v1(
        party,
        epoch=int(epoch),
        step=int(step),
        sgir_op_id=int(sgir_op_id),
        kernel_instance_id=int(kernel_instance_id),
        selected=True,
        t=int(t_checks),
        field_id=int(field_id),
        sks_sample_log2=int(sks_sample_log2),
        check_seed32=check_seed32,
    )

    # For tile GEMM, m=n=d.
    d = int(Z.lo.shape[0])
    r_u64_i64, s_u64_i64 = sks_challenge_vectors_ring_u64_v1(check_seed32=check_seed32, m=d, n=d, t=int(t_checks))
    # Ensure challenge vectors live on the same device as the checked computation.
    dev = Z.lo.device
    if r_u64_i64.device != dev:
        r_u64_i64 = r_u64_i64.to(dev)
    if s_u64_i64.device != dev:
        s_u64_i64 = s_u64_i64.to(dev)

    tvec = sks_freivalds_residual_tile_u64_v1(
        party,
        Z=Z,
        triple_A=triple_A,
        triple_B=triple_B,
        triple_C=triple_C,
        E_pub=E_pub,
        F_pub=F_pub,
        r_u64_i64=r_u64_i64,
    )
    z = sks_dot_compress_v1(tvec=tvec, s_u64_i64=s_u64_i64)
    ok, _zpub, _commits = sks_open_commit_then_open_u64_v1(
        party,
        z=z,
        sid=sid,
        epoch=int(epoch),
        step=int(step),
        sgir_op_id=int(sgir_op_id),
        kernel_instance_id=int(kernel_instance_id),
        field_id=int(field_id),
    )
    return bool(ok)


