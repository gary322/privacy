from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_cb38774080e4a4d3

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from .open import OpenBoolItemWords, open_bool_words_round_v1
from .party import Party
from .rss import RSSBoolU64Words


@dataclass(frozen=True)
class GF2AndGate:
    gate_index: int
    x: RSSBoolU64Words
    y: RSSBoolU64Words
    triple_a: RSSBoolU64Words
    triple_b: RSSBoolU64Words
    triple_c: RSSBoolU64Words


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


def gf2_and_batch_v1(
    party: Party,
    *,
    open_id: int,
    gates: Sequence[GF2AndGate],
    epoch: int,
    step: int,
    round: int,
    sub_id_base: int,
    sgir_op_id: int,
) -> Dict[int, RSSBoolU64Words]:
    """Compute batch of GF(2) AND gates using Beaver triples with exactly one OPEN_BOOL round."""

    if not gates:
        return {}

    items: List[OpenBoolItemWords] = []
    for g in gates:
        if g.x.n_bits != g.y.n_bits:
            raise ValueError("gate x/y n_bits mismatch")
        if g.x.n_bits != g.triple_a.n_bits or g.x.n_bits != g.triple_b.n_bits or g.x.n_bits != g.triple_c.n_bits:
            raise ValueError("gate triple n_bits mismatch")
        e = RSSBoolU64Words(lo_words=(g.x.lo_words ^ g.triple_a.lo_words), hi_words=(g.x.hi_words ^ g.triple_a.hi_words), n_bits=g.x.n_bits)
        f = RSSBoolU64Words(lo_words=(g.y.lo_words ^ g.triple_b.lo_words), hi_words=(g.y.hi_words ^ g.triple_b.hi_words), n_bits=g.y.n_bits)
        sub_e = int(sub_id_base) + (2 * int(g.gate_index) + 0)
        sub_f = int(sub_id_base) + (2 * int(g.gate_index) + 1)
        items.append(OpenBoolItemWords(open_id=int(open_id), sub_id=sub_e, x=e))
        items.append(OpenBoolItemWords(open_id=int(open_id), sub_id=sub_f, x=f))

    pub = open_bool_words_round_v1(party, items=items, epoch=int(epoch), step=int(step), round=int(round), sgir_op_id=int(sgir_op_id))

    out: Dict[int, RSSBoolU64Words] = {}
    for g in gates:
        sub_e = int(sub_id_base) + (2 * int(g.gate_index) + 0)
        sub_f = int(sub_id_base) + (2 * int(g.gate_index) + 1)
        e_pub = pub[(int(open_id), sub_e)]
        f_pub = pub[(int(open_id), sub_f)]
        mask = _mask_words_u64(g.x.n_bits, device=e_pub.device)
        if mask.numel() > 0:
            e_pub = e_pub & mask
            f_pub = f_pub & mask

        # z = c ^ (e&b) ^ (f&a) ^ (e&f) with public term placed into share-0.
        z_lo = g.triple_c.lo_words ^ (g.triple_b.lo_words & e_pub) ^ (g.triple_a.lo_words & f_pub)
        z_hi = g.triple_c.hi_words ^ (g.triple_b.hi_words & e_pub) ^ (g.triple_a.hi_words & f_pub)
        ef = e_pub & f_pub
        if int(party.party_id) == 0:
            z_lo = z_lo ^ ef
        if int(party.party_id) == 2:
            z_hi = z_hi ^ ef
        out[int(g.gate_index)] = RSSBoolU64Words(lo_words=z_lo, hi_words=z_hi, n_bits=g.x.n_bits)
    return out


