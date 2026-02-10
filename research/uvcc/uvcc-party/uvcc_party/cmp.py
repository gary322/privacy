from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_0287a0390f86b1c6,uvcc_group_d4a68ca07fd9802a,uvcc_group_740cc94010953fd5,uvcc_group_5a83f07bbef68c40,uvcc_group_2f305dfbd543379d

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from .edabit import A2BPackV1
from .gf2_and import GF2AndGate, gf2_and_batch_v1
from .gf2_triples import GF2TriplesPackV1
from .netframe import DT_U64, SegmentPayloadV1, build_netframe_v1
from .open import OpenArithItemU64, open_arith_u64_round_v1
from .party import DEFAULT_NET_TIMEOUT_S, DEFAULT_RELAY_TTL_S, Party
from .rss import RSSArithU64, RSSBoolU64Words

from .dpf_dcf import dcf_full_w8_v1, dpf_stage1_w16_v1, dcf_stage2_w16_v1


PRED_LT = 0
PRED_LE = 1
PRED_GT = 2
PRED_GE = 3
PRED_EQ = 4
PRED_NE = 5


def _mask_words_u64(n_bits: int, device: torch.device) -> torch.Tensor:
    n = int(n_bits)
    if n <= 0:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    n_words = (n + 63) // 64
    if n % 64 == 0:
        return torch.full((n_words,), -1, dtype=torch.int64, device=device)
    rem = n % 64
    mask_last = (1 << rem) - 1
    mask = torch.full((n_words,), -1, dtype=torch.int64, device=device)
    mask[-1] = int(mask_last)
    return mask


def _xor(a: RSSBoolU64Words, b: RSSBoolU64Words) -> RSSBoolU64Words:
    if a.n_bits != b.n_bits:
        raise ValueError("xor n_bits mismatch")
    return RSSBoolU64Words(lo_words=(a.lo_words ^ b.lo_words), hi_words=(a.hi_words ^ b.hi_words), n_bits=a.n_bits)


def _xor_public_into_share0(party_id: int, s: RSSBoolU64Words, pub_words: torch.Tensor) -> RSSBoolU64Words:
    lo = s.lo_words
    hi = s.hi_words
    if int(party_id) == 0:
        lo = lo ^ pub_words
    if int(party_id) == 2:
        hi = hi ^ pub_words
    return RSSBoolU64Words(lo_words=lo, hi_words=hi, n_bits=s.n_bits)


def _not(party_id: int, s: RSSBoolU64Words) -> RSSBoolU64Words:
    mask = _mask_words_u64(s.n_bits, device=s.lo_words.device)
    ones = torch.full(mask.shape, -1, dtype=torch.int64, device=s.lo_words.device) & mask
    return _xor_public_into_share0(party_id, s, ones)


def _pack_bits01_to_u64_words(bits01: torch.Tensor, n_bits: int) -> torch.Tensor:
    # bits01: int64 shape (n_bits,) 0/1
    if bits01.dtype != torch.int64 or bits01.ndim != 1:
        raise TypeError("bits01 must be 1-D int64")
    if int(bits01.shape[0]) != int(n_bits):
        raise ValueError("length mismatch")
    n_words = (int(n_bits) + 63) // 64
    if n_words == 0:
        return torch.zeros((0,), dtype=torch.int64, device=bits01.device)
    pow2 = torch.tensor([(1 << k) if k < 63 else -(1 << 63) for k in range(64)], dtype=torch.int64, device=bits01.device)
    words = torch.empty((n_words,), dtype=torch.int64, device=bits01.device)
    for wi in range(n_words):
        start = wi * 64
        end = min(int(n_bits), start + 64)
        chunk = bits01[start:end]
        words[wi] = (chunk * pow2[: (end - start)]).sum(dtype=torch.int64)
    return words


def _public_cbit_words(c_pub: torch.Tensor, *, bit: int, n_bits: int) -> torch.Tensor:
    # c_pub: int64 u64 bit-patterns, shape (N,)
    v = ((c_pub >> int(bit)) & 1).to(torch.int64).contiguous()
    return _pack_bits01_to_u64_words(v, n_bits)


def _and_public(s: RSSBoolU64Words, pub_words: torch.Tensor) -> RSSBoolU64Words:
    if pub_words.dtype != torch.int64 or pub_words.ndim != 1:
        raise TypeError("pub_words must be 1-D int64")
    return RSSBoolU64Words(lo_words=(s.lo_words & pub_words), hi_words=(s.hi_words & pub_words), n_bits=s.n_bits)


def _const_one(party_id: int, n_bits: int, device: torch.device) -> RSSBoolU64Words:
    mask = _mask_words_u64(n_bits, device=device)
    z = torch.zeros_like(mask)
    one = RSSBoolU64Words(lo_words=z, hi_words=z, n_bits=int(n_bits))
    return _xor_public_into_share0(int(party_id), one, mask)


@dataclass
class _GF2TripleCursor:
    triples: GF2TriplesPackV1
    cursor: int
    n_bits: int

    def take(self) -> Tuple[RSSBoolU64Words, RSSBoolU64Words, RSSBoolU64Words]:
        a, b, c = self.triples.vector_at(triple_id_start=int(self.cursor), n_bits=int(self.n_bits))
        self.cursor += int(self.n_bits)
        return a, b, c


def _gf2_and_layer(
    party: Party,
    *,
    open_id: int,
    xs: Sequence[RSSBoolU64Words],
    ys: Sequence[RSSBoolU64Words],
    cursor: _GF2TripleCursor,
    gate_index_base: int,
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
) -> Tuple[List[RSSBoolU64Words], int]:
    if len(xs) != len(ys):
        raise ValueError("xs/ys length mismatch")
    m = len(xs)
    if m == 0:
        return [], int(gate_index_base)
    gates: List[GF2AndGate] = []
    gid = int(gate_index_base)
    for i in range(m):
        ta, tb, tc = cursor.take()
        gates.append(GF2AndGate(gate_index=gid, x=xs[i], y=ys[i], triple_a=ta, triple_b=tb, triple_c=tc))
        gid += 1
    out_map = gf2_and_batch_v1(
        party,
        open_id=int(open_id),
        gates=gates,
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        sub_id_base=0x1000,  # canonical: sub_id = 0x1000 + 2*gate_index + {0,1}
        sgir_op_id=int(sgir_op_id),
    )
    return [out_map[int(g.gate_index)] for g in gates], gid


def _a2b_two_operands_lookahead_v1(
    party: Party,
    *,
    open_id: int,
    x: RSSArithU64,
    y: RSSArithU64,
    w: int,
    edabit_x: A2BPackV1,
    edabit_y: A2BPackV1,
    triples: GF2TriplesPackV1,
    triple_cursor_base: int,
    epoch: int,
    step: int,
    sgir_op_id: int,
) -> Tuple[List[RSSBoolU64Words], List[RSSBoolU64Words], int]:
    """
    Convert both operands to secret bit-planes using one OPEN_ARITH (mx,my) and a log-depth borrow-lookahead scan.

    Returns (xbits[0..w-1], ybits[0..w-1], triple_cursor_end).
    """
    if int(edabit_x.count_vals) != int(edabit_y.count_vals):
        raise ValueError("edabit count mismatch")
    n = int(edabit_x.count_vals)
    if n <= 0:
        raise ValueError("bad edabit count")
    if x.lo.numel() != n or y.lo.numel() != n:
        raise ValueError("x/y length mismatch")
    if edabit_x.sid_hash32 != party.sid_hash32() or edabit_y.sid_hash32 != party.sid_hash32():
        raise ValueError("sid_hash32 mismatch")

    w = int(w)
    if w <= 0 or w > 64:
        raise ValueError("bitwidth must be 1..64")

    # Round R0: OPEN mx=x+r_x and my=y+r_y (macro sub_id=0 and 1).
    rx = RSSArithU64(lo=edabit_x.r_arith_lo, hi=edabit_x.r_arith_hi, fxp_frac_bits=0)
    ry = RSSArithU64(lo=edabit_y.r_arith_lo, hi=edabit_y.r_arith_hi, fxp_frac_bits=0)
    mx = x.add(rx)
    my = y.add(ry)
    pub = open_arith_u64_round_v1(
        party,
        items=[
            OpenArithItemU64(open_id=int(open_id), sub_id=0, x=mx),
            OpenArithItemU64(open_id=int(open_id), sub_id=1, x=my),
        ],
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
    )
    mx_pub = pub[(int(open_id), 0)].contiguous()
    my_pub = pub[(int(open_id), 1)].contiguous()
    if w != 64:
        mx_pub = mx_pub & int((1 << w) - 1)
        my_pub = my_pub & int((1 << w) - 1)

    # Secret r bit-planes from EDABIT (packed to u64 words).
    rbx: List[RSSBoolU64Words] = []
    rby: List[RSSBoolU64Words] = []
    for j in range(w):
        lo_bits_x = edabit_x.r_bits_lo[:, j].to(torch.int64).contiguous()
        hi_bits_x = edabit_x.r_bits_hi[:, j].to(torch.int64).contiguous()
        lo_bits_y = edabit_y.r_bits_lo[:, j].to(torch.int64).contiguous()
        hi_bits_y = edabit_y.r_bits_hi[:, j].to(torch.int64).contiguous()
        rbx.append(RSSBoolU64Words(lo_words=_pack_bits01_to_u64_words(lo_bits_x, n), hi_words=_pack_bits01_to_u64_words(hi_bits_x, n), n_bits=n))
        rby.append(RSSBoolU64Words(lo_words=_pack_bits01_to_u64_words(lo_bits_y, n), hi_words=_pack_bits01_to_u64_words(hi_bits_y, n), n_bits=n))

    device = rbx[0].lo_words.device
    mask_words = _mask_words_u64(n, device=device)
    cursor = _GF2TripleCursor(triples=triples, cursor=int(triple_cursor_base), n_bits=n)
    gate_base = 0

    # Initialize per-bit (g,p) for x and y.
    gx: List[RSSBoolU64Words] = []
    px: List[RSSBoolU64Words] = []
    gy: List[RSSBoolU64Words] = []
    py: List[RSSBoolU64Words] = []
    for j in range(w):
        mxj = _public_cbit_words(mx_pub, bit=j, n_bits=n).to(device=device) & mask_words
        myj = _public_cbit_words(my_pub, bit=j, n_bits=n).to(device=device) & mask_words
        not_mxj = (~mxj) & mask_words
        not_myj = (~myj) & mask_words
        gx.append(_and_public(rbx[j], not_mxj))
        gy.append(_and_public(rby[j], not_myj))
        # Canonical propagate choice for log-depth scan: p = (~m) XOR r
        px.append(_xor_public_into_share0(int(party.party_id), rbx[j], not_mxj))
        py.append(_xor_public_into_share0(int(party.party_id), rby[j], not_myj))

    # Borrow lookahead scan (in-place inclusive scan on (g,p) pairs), L rounds.
    L = int((w - 1).bit_length())
    for s in range(L):
        dist = 1 << s
        xs_layer: List[RSSBoolU64Words] = []
        ys_layer: List[RSSBoolU64Words] = []
        updates: List[Tuple[str, int, int]] = []
        # x operand
        for j in range(dist, w):
            # p[j] & p[j-dist]
            xs_layer.append(px[j])
            ys_layer.append(px[j - dist])
            updates.append(("px", j, -1))
            # p[j] & g[j-dist]
            xs_layer.append(px[j])
            ys_layer.append(gx[j - dist])
            updates.append(("gx", j, -2))
        # y operand
        for j in range(dist, w):
            xs_layer.append(py[j])
            ys_layer.append(py[j - dist])
            updates.append(("py", j, -1))
            xs_layer.append(py[j])
            ys_layer.append(gy[j - dist])
            updates.append(("gy", j, -2))

        outs, gate_base = _gf2_and_layer(
            party,
            open_id=open_id,
            xs=xs_layer,
            ys=ys_layer,
            cursor=cursor,
            gate_index_base=gate_base,
            epoch=epoch,
            step=step,
            round=1 + s,
            sgir_op_id=sgir_op_id,
        )
        if len(outs) != len(updates):
            raise RuntimeError("internal gate count mismatch")
        for out, (which, j, kind) in zip(outs, updates):
            if which == "px":
                px[j] = out
            elif which == "py":
                py[j] = out
            elif which == "gx":
                # g_new = g_old XOR (p & g_prev)
                gx[j] = _xor(gx[j], out)
            elif which == "gy":
                gy[j] = _xor(gy[j], out)
            else:
                raise RuntimeError("bad update kind")

    # Borrow-in bits: b0=0, b[i]=G[i-1]
    zero = RSSBoolU64Words(lo_words=torch.zeros_like(mask_words), hi_words=torch.zeros_like(mask_words), n_bits=n)
    bx = [zero] + gx[:-1]
    by = [zero] + gy[:-1]

    # Output bits: x_i = m_i XOR r_i XOR b_i (public xor injected into share0)
    xbits: List[RSSBoolU64Words] = []
    ybits: List[RSSBoolU64Words] = []
    for j in range(w):
        mxj = _public_cbit_words(mx_pub, bit=j, n_bits=n).to(device=device) & mask_words
        myj = _public_cbit_words(my_pub, bit=j, n_bits=n).to(device=device) & mask_words
        tx = _xor(rbx[j], bx[j])
        ty = _xor(rby[j], by[j])
        xbits.append(_xor_public_into_share0(int(party.party_id), tx, mxj))
        ybits.append(_xor_public_into_share0(int(party.party_id), ty, myj))

    return xbits, ybits, int(cursor.cursor)

def _a2b_from_pack_v1(
    party: Party,
    *,
    open_id: int,
    sub_id: int,
    x: RSSArithU64,
    pack: A2BPackV1,
    triples: GF2TriplesPackV1,
    epoch: int,
    step: int,
    round0: int,
) -> List[RSSBoolU64Words]:
    """A2B (w=8/16) using mask-open and ripple borrow with 1 triple per bit."""

    w = int(pack.w_bits)
    n = int(pack.count_vals)
    if int(pack.count_vals) <= 0:
        raise ValueError("bad pack count_vals")
    if x.lo.numel() != n:
        raise ValueError("x length mismatch")
    if pack.sid_hash32 != party.sid_hash32():
        raise ValueError("sid_hash32 mismatch")
    if triples.sid_hash32 != party.sid_hash32():
        raise ValueError("triples sid_hash32 mismatch")

    r_share = RSSArithU64(lo=pack.r_arith_lo, hi=pack.r_arith_hi, fxp_frac_bits=0)
    s = x.add(r_share)

    pub = open_arith_u64_round_v1(
        party,
        items=[OpenArithItemU64(open_id=int(open_id), sub_id=int(sub_id), x=s)],
        epoch=int(epoch),
        step=int(step),
        round=int(round0),
        sgir_op_id=int(open_id & 0xFFFFFFFF),
    )
    c_pub = pub[(int(open_id), int(sub_id))].contiguous()
    if w != 64:
        c_pub = c_pub & int((1 << w) - 1)

    # Pack r bits into bit-plane words.
    r_planes: List[RSSBoolU64Words] = []
    for j in range(w):
        lo_bits = pack.r_bits_lo[:, j].to(torch.int64).contiguous()
        hi_bits = pack.r_bits_hi[:, j].to(torch.int64).contiguous()
        lo_w = _pack_bits01_to_u64_words(lo_bits, n)
        hi_w = _pack_bits01_to_u64_words(hi_bits, n)
        r_planes.append(RSSBoolU64Words(lo_words=lo_w, hi_words=hi_w, n_bits=n))

    n_words = (n + 63) // 64
    zero = torch.zeros((n_words,), dtype=torch.int64)
    b = RSSBoolU64Words(lo_words=zero, hi_words=zero, n_bits=n)
    out_bits: List[RSSBoolU64Words] = []

    for j in range(w):
        # g_j = r_j & b_j
        tid_start = int(pack.base_triple_id) + j * n
        ta, tb, tc = triples.vector_at(triple_id_start=tid_start, n_bits=n)
        g_map = gf2_and_batch_v1(
            party,
            open_id=int(open_id),
            gates=[GF2AndGate(gate_index=0, x=r_planes[j], y=b, triple_a=ta, triple_b=tb, triple_c=tc)],
            epoch=int(epoch),
            step=int(step),
            round=int(round0 + 1 + j),
            sub_id_base=0x1000 + j * 16,
            sgir_op_id=int(open_id & 0xFFFFFFFF),
        )
        g = g_map[0]

        t = _xor(r_planes[j], b)  # r xor borrow
        c_j_words = _public_cbit_words(c_pub, bit=j, n_bits=n).to(device=t.lo_words.device)
        xj = _xor_public_into_share0(int(party.party_id), t, c_j_words)
        out_bits.append(xj)

        # bnext = g if c_j==1 else g xor t  -> g xor (~c_j & t)
        nc_words = (~c_j_words) & _mask_words_u64(n, device=c_j_words.device)
        t_mask = RSSBoolU64Words(lo_words=(t.lo_words & nc_words), hi_words=(t.hi_words & nc_words), n_bits=n)
        b = _xor(g, t_mask)

    return out_bits


def op_cmp_v1(
    party: Party,
    *,
    cmp_uid: int,
    x: RSSArithU64,
    y: RSSArithU64,
    bitwidth: int,
    pred: int,
    signedness: int,
    edabit_x: A2BPackV1,
    edabit_y: A2BPackV1,
    triples: GF2TriplesPackV1,
    cmp_triple_cursor_base: int,
    epoch: int,
    step: int,
) -> RSSBoolU64Words:
    """
    EDABIT-based secret-secret compare (3PC RSS over GF(2) bitshares).

    Implements the canonical log-depth structure from `privacy_new.txt`:
    - one OPEN_ARITH round for (x+r_x, y+r_y)
    - borrow-lookahead scan (log-depth)
    - suffixEq scan (log-depth) + lt/eq synthesis
    """

    w = int(bitwidth)
    if w <= 0 or w > 64:
        raise ValueError("bitwidth must be 1..64")
    if int(pred) not in (PRED_LT, PRED_LE, PRED_GT, PRED_GE, PRED_EQ, PRED_NE):
        raise ValueError("bad pred")
    if int(signedness) not in (0, 1):
        raise ValueError("bad signedness")

    n = int(edabit_x.count_vals)
    if int(edabit_y.count_vals) != n:
        raise ValueError("edabit count mismatch")

    open_id = int(cmp_uid) & 0xFFFFFFFFFFFFFFFF

    xb, yb, cursor_end = _a2b_two_operands_lookahead_v1(
        party,
        open_id=open_id,
        x=x,
        y=y,
        w=w,
        edabit_x=edabit_x,
        edabit_y=edabit_y,
        triples=triples,
        triple_cursor_base=int(cmp_triple_cursor_base),
        epoch=int(epoch),
        step=int(step),
        sgir_op_id=int(open_id & 0xFFFFFFFF),
    )

    # Signedness: flip MSB
    if int(signedness) == 1:
        mask = _mask_words_u64(n, device=xb[w - 1].lo_words.device)
        ones = torch.full(mask.shape, -1, dtype=torch.int64, device=mask.device) & mask
        xb[w - 1] = _xor_public_into_share0(int(party.party_id), xb[w - 1], ones)
        yb[w - 1] = _xor_public_into_share0(int(party.party_id), yb[w - 1], ones)

    # Signedness: flip MSB (bias trick) then do unsigned compare.
    if int(signedness) == 1:
        xb[w - 1] = _not(int(party.party_id), xb[w - 1])
        yb[w - 1] = _not(int(party.party_id), yb[w - 1])

    n = int(edabit_x.count_vals)
    device = xb[0].lo_words.device

    # eq_i = NOT(xor(x_i,y_i))
    eq = [_not(int(party.party_id), _xor(xb[i], yb[i])) for i in range(w)]

    # SuffixEq via prefix AND scan on reversed eq.
    cursor = _GF2TripleCursor(triples=triples, cursor=int(cursor_end), n_bits=n)
    gate_base = 0
    eq_rev = [eq[w - 1 - i] for i in range(w)]
    L = int((w - 1).bit_length())
    for s in range(L):
        dist = 1 << s
        xs_layer = [eq_rev[i] for i in range(dist, w)]
        ys_layer = [eq_rev[i - dist] for i in range(dist, w)]
        outs, gate_base = _gf2_and_layer(
            party,
            open_id=open_id,
            xs=xs_layer,
            ys=ys_layer,
            cursor=cursor,
            gate_index_base=gate_base,
            epoch=epoch,
            step=step,
            round=1 + L + s,
            sgir_op_id=int(open_id & 0xFFFFFFFF),
        )
        for i, out in enumerate(outs, start=dist):
            eq_rev[i] = out

    eq_all = eq_rev[w - 1]
    one = _const_one(int(party.party_id), n, device=device)
    suffix = [one] * w
    suffix[w - 1] = one
    for i in range(0, w - 1):
        suffix[i] = eq_rev[w - 2 - i]

    # u_i = (1-x_i) & y_i
    not_x = [_not(int(party.party_id), xb[i]) for i in range(w)]
    u, gate_base = _gf2_and_layer(
        party,
        open_id=open_id,
        xs=not_x,
        ys=yb,
        cursor=cursor,
        gate_index_base=gate_base,
        epoch=epoch,
        step=step,
        round=1 + 2 * L,
        sgir_op_id=int(open_id & 0xFFFFFFFF),
    )

    # t_i = suffix[i] & u_i
    t, gate_base = _gf2_and_layer(
        party,
        open_id=open_id,
        xs=suffix,
        ys=u,
        cursor=cursor,
        gate_index_base=gate_base,
        epoch=epoch,
        step=step,
        round=2 + 2 * L,
        sgir_op_id=int(open_id & 0xFFFFFFFF),
    )

    lt = t[0]
    for i in range(1, w):
        lt = _xor(lt, t[i])

    # Map predicates from lt/eq in a total order.
    if int(pred) == PRED_LT:
        out = lt
    elif int(pred) == PRED_LE:
        out = _xor(lt, eq_all)  # disjoint
    elif int(pred) == PRED_EQ:
        out = eq_all
    elif int(pred) == PRED_NE:
        out = _not(int(party.party_id), eq_all)
    elif int(pred) == PRED_GT:
        out = _not(int(party.party_id), _xor(lt, eq_all))  # NOT(le)
    else:  # GE
        out = _not(int(party.party_id), lt)
    return out


def _rss_arith_add_public_into_share0(party_id: int, x: RSSArithU64, c_pub: int) -> RSSArithU64:
    # Add a public constant into the replicated share-0 component only.
    lo = x.lo
    hi = x.hi
    if int(party_id) == 0:
        lo = lo + int(c_pub)
    if int(party_id) == 2:
        hi = hi + int(c_pub)
    return RSSArithU64(lo=lo, hi=hi, fxp_frac_bits=x.fxp_frac_bits)


def _prg_shared_bits_v1(*, key32: bytes, job_id32: bytes, cmp_uid: int, label: bytes, n_bits: int) -> torch.Tensor:
    """
    Expand a 32-byte secret into a deterministic shared bitstream (0/1 int64).

    Security note: `key32` MUST be secret to the two parties that share it; the third party must not have it.
    """
    if len(key32) != 32:
        raise ValueError("key32 must be 32 bytes")
    if len(job_id32) != 32:
        raise ValueError("job_id32 must be 32 bytes")
    if not isinstance(label, (bytes, bytearray)):
        raise TypeError("label must be bytes")
    n = int(n_bits)
    if n < 0:
        raise ValueError("n_bits must be >= 0")
    out = torch.empty((n,), dtype=torch.int64)
    ctr = 0
    i = 0
    dom = b"UVCC.CMP.REPLBITS.v1\0"
    while i < n:
        h = hashlib.sha256(dom + key32 + job_id32 + struct.pack("<Q", int(cmp_uid) & 0xFFFFFFFFFFFFFFFF) + bytes(label) + struct.pack("<I", ctr)).digest()
        ctr += 1
        for byte in h:
            for k in range(8):
                if i >= n:
                    break
                out[i] = (byte >> k) & 1
                i += 1
            if i >= n:
                break
    return out


def _u64_words_to_bytes(words: torch.Tensor) -> bytes:
    if words.dtype != torch.int64 or words.ndim != 1:
        raise TypeError("words must be 1-D int64")
    out = bytearray()
    for v in words.tolist():
        out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _bytes_to_u64_words(buf: bytes, n_words: int, device: torch.device) -> torch.Tensor:
    if len(buf) != 8 * int(n_words):
        raise ValueError("bad words byte length")
    out = torch.empty((int(n_words),), dtype=torch.int64, device=device)
    for i in range(int(n_words)):
        out[i] = int.from_bytes(buf[8 * i : 8 * i + 8], "little", signed=True)
    return out


def op_cmp_lt_fast_dcf_v1(
    party: Party,
    *,
    cmp_uid: int,
    x: RSSArithU64,
    y: RSSArithU64,
    bitwidth: int,
    signedness: int,
    r_mask: RSSArithU64,
    dcf_keyrecs_edge01: List[bytes],
    edge01_key32: bytes | None,
    epoch: int,
    step: int,
) -> RSSBoolU64Words:
    """
    Fast-path LT for w<=16 using one shared mask r and 2-party DCF eval on edge 01, embedded into 3PC RSS.

    This follows the identity in `privacy_new.txt` §8.2:
      lt = (a<b) XOR DCF_r(a) XOR DCF_r(b)
    where a = (x+r) mod 2^w, b = (y+r) mod 2^w (opened).

    Notes:
    - This helper uses only edge 01 keys; party2 receives replicated components from P0 and P1.
    - The caller must provide per-element DCF keyrecs for this party if party_id in {0,1}; party2 passes an empty list.
    """
    w = int(bitwidth)
    if w not in (8, 16):
        raise ValueError("fast DCF compare supports w=8 or w=16")
    if int(signedness) not in (0, 1):
        raise ValueError("signedness must be 0/1")
    n = int(x.lo.numel())
    if n <= 0 or y.lo.numel() != n or r_mask.lo.numel() != n:
        raise ValueError("x/y/r length mismatch")

    open_id = int(cmp_uid) & 0xFFFFFFFFFFFFFFFF

    # Optional signed bias: x' = x + 2^(w-1) mod 2^w
    if int(signedness) == 1:
        bias = 1 << (w - 1)
        x = _rss_arith_add_public_into_share0(int(party.party_id), x, bias)
        y = _rss_arith_add_public_into_share0(int(party.party_id), y, bias)

    # OPEN a=x+r and b=y+r (macro sub_id=0,1).
    a_share = x.add(r_mask)
    b_share = y.add(r_mask)
    pub = open_arith_u64_round_v1(
        party,
        items=[
            OpenArithItemU64(open_id=open_id, sub_id=0, x=a_share),
            OpenArithItemU64(open_id=open_id, sub_id=1, x=b_share),
        ],
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(open_id & 0xFFFFFFFF),
    )
    a_pub = (pub[(open_id, 0)] & int((1 << w) - 1)).to(torch.int64)
    b_pub = (pub[(open_id, 1)] & int((1 << w) - 1)).to(torch.int64)
    c_pub = (a_pub < b_pub).to(torch.int64)

    # Evaluate DCF shares (only parties 0 and 1 hold keys for edge01 in this helper).
    if int(party.party_id) in (0, 1):
        if len(dcf_keyrecs_edge01) != n:
            raise ValueError("dcf_keyrecs_edge01 length mismatch")
        if edge01_key32 is None or len(edge01_key32) != 32:
            raise ValueError("edge01_key32 (32 bytes) is required for party 0/1 to keep REPL secrecy from P2")

        share_a_bits = torch.empty((n,), dtype=torch.int64)
        share_b_bits = torch.empty((n,), dtype=torch.int64)
        for i in range(n):
            keyrec = dcf_keyrecs_edge01[i]
            u_a = int(a_pub[i].item()) & ((1 << w) - 1)
            u_b = int(b_pub[i].item()) & ((1 << w) - 1)
            if w == 8:
                vec = dcf_full_w8_v1(keyrec, device=torch.device("cpu"))
                share_a_bits[i] = int(vec[u_a].item()) & 1
                share_b_bits[i] = int(vec[u_b].item()) & 1
            else:
                front = dpf_stage1_w16_v1(keyrec, device=torch.device("cpu"))
                vec = dcf_stage2_w16_v1(keyrec, frontier_seed_lo=front[0], frontier_seed_hi=front[1], frontier_t=front[2], frontier_acc=front[3], device=torch.device("cpu"))
                share_a_bits[i] = int(vec[u_a].item()) & 1
                share_b_bits[i] = int(vec[u_b].item()) & 1

        s0_a = _pack_bits01_to_u64_words(share_a_bits, n)
        s0_b = _pack_bits01_to_u64_words(share_b_bits, n)
        n_words = int(s0_a.numel())

        # Shared secret "b1" component between P0 and P1 derived from the edge secret (K_01).
        b1_a_bits = _prg_shared_bits_v1(key32=edge01_key32, job_id32=party.job_id32, cmp_uid=open_id, label=b"wx", n_bits=n)
        b1_b_bits = _prg_shared_bits_v1(key32=edge01_key32, job_id32=party.job_id32, cmp_uid=open_id, label=b"wy", n_bits=n)
        b1_a = _pack_bits01_to_u64_words(b1_a_bits, n)
        b1_b = _pack_bits01_to_u64_words(b1_b_bits, n)

        if int(party.party_id) == 0:
            wx = RSSBoolU64Words(lo_words=s0_a, hi_words=b1_a, n_bits=n)
            wy = RSSBoolU64Words(lo_words=s0_b, hi_words=b1_b, n_bits=n)
            # Send b0 components to P2 as a NetFrame (transcripted).
            segs = [
                SegmentPayloadV1(seg_kind=20, object_id=int(open_id), sub_id=0, dtype=DT_U64, fxp_frac_bits=0, payload=_u64_words_to_bytes(s0_a)),
                SegmentPayloadV1(seg_kind=20, object_id=int(open_id), sub_id=1, dtype=DT_U64, fxp_frac_bits=0, payload=_u64_words_to_bytes(s0_b)),
            ]
            frame = build_netframe_v1(
                job_id32=party.job_id32,
                epoch=int(epoch),
                step=int(step),
                round=1,
                msg_kind=0x0201,
                flags=0,
                sender=0,
                receiver=2,
                seq_no=0,
                segments=segs,
            )
            party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=b"uvcc.cmp.repl.v1")
        else:
            # P1 computes b2 = s1 XOR b1 and sends to P2.
            b2_a = s0_a ^ b1_a
            b2_b = s0_b ^ b1_b
            wx = RSSBoolU64Words(lo_words=b1_a, hi_words=b2_a, n_bits=n)
            wy = RSSBoolU64Words(lo_words=b1_b, hi_words=b2_b, n_bits=n)
            segs = [
                SegmentPayloadV1(seg_kind=20, object_id=int(open_id), sub_id=2, dtype=DT_U64, fxp_frac_bits=0, payload=_u64_words_to_bytes(b2_a)),
                SegmentPayloadV1(seg_kind=20, object_id=int(open_id), sub_id=3, dtype=DT_U64, fxp_frac_bits=0, payload=_u64_words_to_bytes(b2_b)),
            ]
            frame = build_netframe_v1(
                job_id32=party.job_id32,
                epoch=int(epoch),
                step=int(step),
                round=1,
                msg_kind=0x0201,
                flags=0,
                sender=1,
                receiver=2,
                seq_no=0,
                segments=segs,
            )
            party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=b"uvcc.cmp.repl.v1")

        # lt = c XOR wx XOR wy
        c_words = _pack_bits01_to_u64_words(c_pub, n)
        s = _xor(wx, wy)
        return _xor_public_into_share0(int(party.party_id), s, c_words)

    # Party 2: receive b0 from P0 and b2 from P1 as NetFrames; derive its (b2,b0) pair.
    n_words = (n + 63) // 64
    f0 = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=1,
        msg_kind=0x0201,
        sender=0,
        receiver=2,
        seq_no=0,
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
        relay_domain=b"uvcc.cmp.repl.v1",
    )
    f1 = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=1,
        msg_kind=0x0201,
        sender=1,
        receiver=2,
        seq_no=0,
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
        relay_domain=b"uvcc.cmp.repl.v1",
    )
    # Pull segments by sub_id (fixed mapping above).
    seg_map0 = {(int(s.object_id), int(s.sub_id)): f0.payload[int(s.offset) : int(s.offset) + int(s.length)] for s in f0.segments}
    seg_map1 = {(int(s.object_id), int(s.sub_id)): f1.payload[int(s.offset) : int(s.offset) + int(s.length)] for s in f1.segments}
    buf0_a = seg_map0.get((int(open_id), 0), None)
    buf0_b = seg_map0.get((int(open_id), 1), None)
    buf2_a = seg_map1.get((int(open_id), 2), None)
    buf2_b = seg_map1.get((int(open_id), 3), None)
    if buf0_a is None or buf0_b is None or buf2_a is None or buf2_b is None:
        raise ValueError("missing REPL segments")
    if len(buf0_a) != 8 * n_words or len(buf0_b) != 8 * n_words or len(buf2_a) != 8 * n_words or len(buf2_b) != 8 * n_words:
        raise ValueError("bad REPL segment sizes")
    s0_a = _bytes_to_u64_words(buf0_a, n_words, device=torch.device("cpu"))
    s0_b = _bytes_to_u64_words(buf0_b, n_words, device=torch.device("cpu"))
    b2_a = _bytes_to_u64_words(buf2_a, n_words, device=torch.device("cpu"))
    b2_b = _bytes_to_u64_words(buf2_b, n_words, device=torch.device("cpu"))
    wx = RSSBoolU64Words(lo_words=b2_a, hi_words=s0_a, n_bits=n)
    wy = RSSBoolU64Words(lo_words=b2_b, hi_words=s0_b, n_bits=n)
    c_words = _pack_bits01_to_u64_words(c_pub, n)
    s = _xor(wx, wy)
    return _xor_public_into_share0(int(party.party_id), s, c_words)


