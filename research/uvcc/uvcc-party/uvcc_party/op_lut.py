from __future__ import annotations

# pyright: reportMissingImports=false

# UVCC_REQ_GROUP: uvcc_group_48a6f9c1656f1342,uvcc_group_e6b8d87d4097bede,uvcc_group_509a5eba42fcc1ce,uvcc_group_b69668db9263f95f

import struct
from typing import Optional, Tuple

import torch

from .dpf_aes128 import eval_dpf_point_aes128_ar64_v1
from .dpf_dcf import chacha20_block_bytes_v1
from .netframe import DT_U64, SegmentPayloadV1, build_netframe_v1
from .open import OpenArithItemU64, open_arith_u64_round_v1
from .op_lut_blob import EDGE_01, EDGE_12, EDGE_20, OPLUTRecordV1, parse_oplut_record_v1
from .party import DEFAULT_NET_TIMEOUT_S, DEFAULT_RELAY_TTL_S, Party
from .rss import RSSArithU64
from .op_lut_plan import DS_OP_LUT, OPLUTPlanPrimeV1, OPLUTTaskV1, oplut_tasks_bytes_v1
from .transcript import sha256


# NetFrame msg_kind for OP_LUT REPL share exchange (local v1).
MSG_OPLUT_REPL = 0x0310

# Local transcript-only commit leaf.
MSG_OPLUT_COMMIT = 0x0003


def _u64_tensor_to_le_bytes(x: torch.Tensor) -> bytes:
    if x.dtype != torch.int64:
        raise TypeError("expected int64 tensor of u64 bit-patterns")
    xs = x.contiguous().view(-1).tolist()
    out = bytearray()
    for v in xs:
        out += (int(v) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _le_bytes_to_u64_tensor(b: bytes, n: int, *, device: torch.device) -> torch.Tensor:
    if len(b) != 8 * int(n):
        raise ValueError("bad u64 byte length")
    vals = [int.from_bytes(b[i * 8 : (i + 1) * 8], "little", signed=False) for i in range(int(n))]
    # Store as int64 u64 bit-patterns.
    vals_i64 = [v if v < (1 << 63) else v - (1 << 64) for v in vals]
    return torch.tensor(vals_i64, dtype=torch.int64, device=device)


def _derive_edge_u64_stream(
    *,
    seed32: bytes,
    nonce12: bytes,
    ctr0: int,
    lane_base: int,
    lanes: int,
    lane_stride: int,
) -> torch.Tensor:
    # NOTE: chacha20_block_bytes_v1 accepts counter32; we truncate ctr to low 32.
    out = torch.empty((int(lanes),), dtype=torch.int64, device=torch.device("cpu"))
    for ell in range(int(lanes)):
        ctr = int(ctr0) + (int(lane_base) + ell) * int(lane_stride)
        blk = chacha20_block_bytes_v1(key32=seed32, nonce12=nonce12, counter32=int(ctr) & 0xFFFFFFFF)
        u = int.from_bytes(blk[0:8], "little", signed=False)
        out[ell] = int(u if u < (1 << 63) else u - (1 << 64))
    return out


def _derive_r_pair_v1(*, rec: OPLUTRecordV1) -> RSSArithU64:
    meta = rec.meta
    mask = rec.mask
    if int(meta.prg_id) != 1:
        raise ValueError("only PRG_CHACHA20 supported in v1 runtime")
    W = int(meta.domain_w)
    if W not in (8, 16):
        raise ValueError("domain_w must be 8 or 16")
    maskN = (1 << W) - 1

    def comp_from_seed(seed32: bytes) -> torch.Tensor:
        # Extract low w bits from the ChaCha stream.
        out = torch.empty((int(meta.lanes),), dtype=torch.int64, device=torch.device("cpu"))
        for ell in range(int(meta.lanes)):
            ctr = int(mask.counter0) + (int(meta.lane_base) + ell) * int(mask.lane_stride)
            blk = chacha20_block_bytes_v1(key32=seed32, nonce12=mask.nonce_r12, counter32=int(ctr) & 0xFFFFFFFF)
            if W == 8:
                v = int(blk[0])
            else:
                v = int.from_bytes(blk[0:2], "little", signed=False)
            v &= maskN
            out[ell] = int(v)
        return out

    # Map edge -> component
    comp_a = comp_from_seed(mask.seed_a32)
    comp_b = comp_from_seed(mask.seed_b32)
    # Determine which component corresponds to which RSS share index (share0/1/2).
    # EDGE_20 -> share0, EDGE_01 -> share1, EDGE_12 -> share2.
    # Party0 holds (share0,share1); Party1 (share1,share2); Party2 (share2,share0).
    def edge_to_share_index(e: int) -> int:
        if int(e) == EDGE_20:
            return 0
        if int(e) == EDGE_01:
            return 1
        if int(e) == EDGE_12:
            return 2
        raise ValueError("bad edge id")

    share_map: dict[int, torch.Tensor] = {}
    share_map[edge_to_share_index(int(mask.edge_a))] = comp_a
    share_map[edge_to_share_index(int(mask.edge_b))] = comp_b

    pid = int(meta.party_id)
    if pid == 0:
        lo = share_map[0]
        hi = share_map[1]
    elif pid == 1:
        lo = share_map[1]
        hi = share_map[2]
    elif pid == 2:
        lo = share_map[2]
        hi = share_map[0]
    else:
        raise ValueError("party_id must be 0..2")
    # Lift into ring element (already small); store as u64 bit-patterns in int64.
    return RSSArithU64(lo=lo.to(torch.int64), hi=hi.to(torch.int64))


def _derive_refresh_masks_v1(*, rec: OPLUTRecordV1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    meta = rec.meta
    mask = rec.mask
    ref = rec.refresh
    if int(meta.prg_id) != 1:
        raise ValueError("only PRG_CHACHA20 supported in v1 runtime")
    # Determine which seeds correspond to edge20 and edge12 for this party.
    seed_edge20 = None
    seed_edge12 = None
    if int(mask.edge_a) == EDGE_20:
        seed_edge20 = mask.seed_a32
    if int(mask.edge_b) == EDGE_20:
        seed_edge20 = mask.seed_b32
    if int(mask.edge_a) == EDGE_12:
        seed_edge12 = mask.seed_a32
    if int(mask.edge_b) == EDGE_12:
        seed_edge12 = mask.seed_b32

    m0 = None
    m1 = None
    if int(ref.uses_edge20) & 1:
        if seed_edge20 is None:
            raise ValueError("refresh requires edge20 seed but not present")
        m0 = _derive_edge_u64_stream(seed32=seed_edge20, nonce12=ref.nonce_m12, ctr0=int(ref.counter0), lane_base=int(meta.lane_base), lanes=int(meta.lanes), lane_stride=int(ref.lane_stride))
    if int(ref.uses_edge12) & 1:
        if seed_edge12 is None:
            raise ValueError("refresh requires edge12 seed but not present")
        m1 = _derive_edge_u64_stream(seed32=seed_edge12, nonce12=ref.nonce_m12, ctr0=int(ref.counter0), lane_base=int(meta.lane_base), lanes=int(meta.lanes), lane_stride=int(ref.lane_stride))
    return m0, m1


def _eval_oplut_cpu_lane(*, key_bytes: bytes, w: int, u_pub: int, table_u64: torch.Tensor, party_b: int) -> int:
    W = int(w)
    maskN = (1 << W) - 1
    acc = 0
    # Table is public.
    tab = table_u64.contiguous().view(-1).tolist()
    if len(tab) != (1 << W):
        raise ValueError("table length mismatch")
    for j, t in enumerate(tab):
        v = (int(u_pub) - int(j)) & maskN
        s = eval_dpf_point_aes128_ar64_v1(key_bytes=key_bytes, w=W, x=v, party_b=int(party_b))
        acc = (acc + ((s * (int(t) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return acc


def op_lut_phase2_local_cpu_v1(
    *,
    fss_blob: bytes,
    u_pub_u8: torch.Tensor,
    table_u64: torch.Tensor,
) -> torch.Tensor:
    """
    CPU reference for OP_LUT Phase-2 (local, no network):
    - Evaluates the embedded 2PC arithmetic DPF keys for P0/P1 (if present)
    - Applies refresh masks to produce the additive share y' for this party

    Returns:
      y_prime_u64_i64[lanes] as torch.int64 bit-patterns (Z2^64).
    """
    rec = parse_oplut_record_v1(fss_blob)
    meta = rec.meta
    W = int(meta.domain_w)
    if W not in (8, 16):
        raise ValueError("domain_w must be 8 or 16")
    lanes = int(meta.lanes)
    if not isinstance(u_pub_u8, torch.Tensor) or int(u_pub_u8.numel()) != lanes:
        raise ValueError("u_pub_u8 length mismatch")
    if u_pub_u8.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
        raise TypeError("u_pub_u8 must be an integer tensor")
    # canonical: interpret u as low w bits.
    u_vals = u_pub_u8.to(torch.int64).view(-1)

    y_share_u64 = torch.zeros((lanes,), dtype=torch.int64, device=torch.device("cpu"))
    if int(meta.dpf_role) in (1, 2):
        if rec.dpf2pc is None:
            raise ValueError("missing DPF_2PC section")
        party_b = 0 if int(meta.dpf_role) == 1 else 1
        for ell in range(lanes):
            key_lane = rec.dpf2pc.lane_key(ell)
            y = _eval_oplut_cpu_lane(key_bytes=key_lane, w=W, u_pub=int(u_vals[ell].item()) & ((1 << W) - 1), table_u64=table_u64, party_b=party_b)
            y_share_u64[ell] = int(y if y < (1 << 63) else y - (1 << 64))

    m0, m1 = _derive_refresh_masks_v1(rec=rec)
    pid = int(meta.party_id)
    if pid == 0:
        if m0 is None:
            raise ValueError("P0 refresh missing m0")
        return (y_share_u64 + m0.to(torch.int64)).to(torch.int64)
    if pid == 1:
        if m1 is None:
            raise ValueError("P1 refresh missing m1")
        return (y_share_u64 + m1.to(torch.int64)).to(torch.int64)
    if pid == 2:
        if m0 is None or m1 is None:
            raise ValueError("P2 refresh missing m0/m1")
        return (-(m0.to(torch.int64) + m1.to(torch.int64))).to(torch.int64)
    raise ValueError("party_id must be 0..2")


def _u16_tensor_to_i16_bitpattern(u16: torch.Tensor) -> torch.Tensor:
    x = u16.to(torch.int64).contiguous()
    x = x.clone()
    x[x >= 32768] -= 65536
    return x.to(torch.int16)


def op_lut_phase2_local_v1(
    *,
    fss_blob: bytes,
    u_pub: torch.Tensor,
    table_u64: torch.Tensor,
    prefer_cuda: bool,
) -> torch.Tensor:
    """
    Phase-2 local evaluator selector (CPU vs CUDA).
    Returns y_prime_i64[LANES] on CPU.
    """
    rec = parse_oplut_record_v1(fss_blob)
    W = int(rec.meta.domain_w)
    if W == 8:
        u_pub_u8 = u_pub.to(torch.uint8).contiguous()
    else:
        u_pub_u8 = _u16_tensor_to_i16_bitpattern(u_pub)

    if bool(prefer_cuda) and torch.cuda.is_available() and (table_u64.is_cuda or u_pub.is_cuda):
        try:
            from .cuda_ext import oplut_phase2_w8_record, oplut_phase2_w16_record

            rec_cuda = torch.tensor(list(fss_blob), dtype=torch.uint8, device="cuda").contiguous()
            if W == 8:
                u_cuda = u_pub_u8.to(device="cuda", dtype=torch.uint8).contiguous()
                table_cuda = table_u64.to(device="cuda", dtype=torch.int64).contiguous()
                y_cuda = oplut_phase2_w8_record(rec_cuda, u_cuda, table_cuda)
            else:
                u_cuda = u_pub_u8.to(device="cuda", dtype=torch.int16).contiguous()
                table_cuda = table_u64.to(device="cuda", dtype=torch.int64).contiguous()
                y_cuda = oplut_phase2_w16_record(rec_cuda, u_cuda, table_cuda)
            return y_cuda.cpu()
        except Exception:
            # Deterministic fallback to CPU reference if CUDA extension is unavailable/unbuildable.
            return op_lut_phase2_local_cpu_v1(fss_blob=fss_blob, u_pub_u8=u_pub_u8, table_u64=table_u64)

    return op_lut_phase2_local_cpu_v1(fss_blob=fss_blob, u_pub_u8=u_pub_u8, table_u64=table_u64)


def op_lut_public_v1(
    party: Party,
    *,
    x: RSSArithU64,
    table_u64: torch.Tensor,
    fss_blob: bytes,
    epoch: int,
    step: int,
    sgir_op_id: int,
) -> RSSArithU64:
    """
    OP_LUT_PUBLIC_V1 (CPU reference): OPEN(u=x+r) then 2PC arithmetic DPF dot product + refresh + REPL -> RSS.

    NOTE: This python implementation is intended as a correctness reference; for w=16 it is slow.
    """
    rec = parse_oplut_record_v1(fss_blob)
    meta = rec.meta
    if int(meta.domain_w) not in (8, 16):
        raise ValueError("domain_w must be 8 or 16")
    W = int(meta.domain_w)
    maskN = (1 << W) - 1
    lanes = int(meta.lanes)
    if x.lo.numel() != lanes:
        raise ValueError("x lanes mismatch with record meta")
    if int(meta.ring_id) != 1:
        raise ValueError("only Z2^64 ring supported")

    # --- Phase 1: compute u_share = (x + r) mod 2^w and OPEN u ---
    r_pair = _derive_r_pair_v1(rec=rec)
    u_pair = x.add(r_pair)
    u_pair = RSSArithU64(lo=(u_pair.lo & maskN), hi=(u_pair.hi & maskN), fxp_frac_bits=0)

    pub_u = open_arith_u64_round_v1(
        party,
        items=[OpenArithItemU64(open_id=int(sgir_op_id) & 0xFFFFFFFFFFFFFFFF, sub_id=0, x=u_pair)],
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
    )[(int(sgir_op_id) & 0xFFFFFFFFFFFFFFFF, 0)]
    pub_u = (pub_u & maskN).contiguous().to(torch.int64)

    # --- Phase 2: compute additive y share for P0/P1 via DPF dot-product ---
    y_prime = op_lut_phase2_local_v1(fss_blob=fss_blob, u_pub=pub_u, table_u64=table_u64, prefer_cuda=True)

    # --- OP_A2R-style replicate (additive -> RSS) ---
    # Canonical RSS pair is (share_i, share_{i+1}) at party Pi (see `uvcc_rss_u64_pair_v1`).
    # Therefore each party sends its additive share to the previous party and receives the next share
    # from the next party:
    #   P0 -> P2, P1 -> P0, P2 -> P1.
    pid = int(meta.party_id)
    send_to = (pid + 2) % 3
    recv_from = (pid + 1) % 3

    payload = _u64_tensor_to_le_bytes(y_prime)
    frame = build_netframe_v1(
        job_id32=party.job_id32,
        epoch=int(epoch),
        step=int(step),
        round=1,
        msg_kind=MSG_OPLUT_REPL,
        flags=0,
        sender=int(pid),
        receiver=int(send_to),
        seq_no=int(sgir_op_id) & 0xFFFFFFFF,
        segments=[
            SegmentPayloadV1(seg_kind=10, object_id=int(sgir_op_id) & 0xFFFFFFFF, sub_id=0, dtype=DT_U64, fxp_frac_bits=0, payload=payload),
        ],
    )
    party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=b"uvcc.oplut.repl.v1")

    got = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=1,
        msg_kind=MSG_OPLUT_REPL,
        sender=int(recv_from),
        receiver=int(pid),
        seq_no=int(sgir_op_id) & 0xFFFFFFFF,
        relay_domain=b"uvcc.oplut.repl.v1",
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
    )
    # Extract first non-PAD segment payload; that's our vector bytes.
    seg0 = next((s for s in got.segments if int(s.seg_kind) != 1), None)
    if seg0 is None:
        raise ValueError("missing segment in REPL frame")
    recv_bytes = got.payload[int(seg0.offset) : int(seg0.offset) + int(seg0.length)]
    recv_vec = _le_bytes_to_u64_tensor(recv_bytes, lanes, device=torch.device("cpu"))

    # Construct RSS pair (share_i, share_{i+1}) with share_i = y_prime (this party's additive share),
    # share_{i+1} received from next party.
    lo = y_prime
    hi = recv_vec

    # Transcript commit for the OP_LUT batch (hash only, deterministic).
    if party.transcript is not None:
        # h_T commits to table bytes, and h_u commits to opened u_pub bytes (safe).
        h_T = sha256(_u64_tensor_to_le_bytes(table_u64.to(torch.int64)))
        # Canonical u_pub bytes are u8/u16 per domain.
        if W == 8:
            u_pub_bytes = bytes(int(x.item()) & 0xFF for x in pub_u.view(-1))
        else:
            u_pub_bytes = b"".join((int(x.item()) & 0xFFFF).to_bytes(2, "little", signed=False) for x in pub_u.view(-1))
        h_u = sha256(u_pub_bytes)

        # Build plan' and one-task table for transcript binding (privacy_new.txt §12).
        plan_prime = OPLUTPlanPrimeV1(
            task_count=1,
            key_arena_bytes=len(fss_blob),
            const_arena_bytes=int(table_u64.numel()) * 8,
            u_pub_bytes=len(u_pub_bytes),
            out_arena_bytes=int(lanes) * 2 * 8,
            scratch_bytes=0,
        ).to_bytes()
        task = OPLUTTaskV1(
            fss_id=int(rec.header.fss_id),
            sgir_op_id=int(sgir_op_id),
            domain_w=W,
            elem_fmt=7,  # UVCC_LUT_ELEM_R64
            dpf_mode=1,  # UVCC_LUT_DPF_ARITH_R64
            flags=0,
            lanes=int(lanes),
            u_pub_offset=0,
            u_pub_stride=1 if W == 8 else 2,
            table_offset=0,
            table_bytes=int(table_u64.numel()) * 8,
            out_offset=0,
            out_stride=8,
            key_offset=0,
            key_bytes=len(fss_blob),
        )
        tasks_bytes = oplut_tasks_bytes_v1([task])
        h_oplut = sha256(DS_OP_LUT + plan_prime + tasks_bytes + h_T + h_u)
        hdr_hash32 = sha256(b"UVCC.oplut.commit.hdr.v1\0" + struct.pack("<II", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF) + struct.pack("<I", int(sgir_op_id) & 0xFFFFFFFF))
        party.transcript.record_frame(
            epoch=int(epoch),
            step=int(step),
            round=2,
            msg_kind=MSG_OPLUT_COMMIT,
            sender=int(pid),
            receiver=int(pid),
            dir=0,
            seq_no=0,
            payload_bytes=len(plan_prime) + len(tasks_bytes) + 64,
            payload_hash32=h_oplut,
            header_hash32=hdr_hash32,
            segments=[],
        )

    return RSSArithU64(lo=lo.to(torch.int64), hi=hi.to(torch.int64), fxp_frac_bits=0)


