from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_fef1ba5ecf03ce43

import torch

from uvcc_party.rss import EDGE_01, EDGE_12, EDGE_20, make_rss_arith_u64_triple, rss_pair_share_indices_for_party_v1, rss_share_index_for_edge_v1


def test_rss_edge_to_share_mapping_is_canonical() -> None:
    assert rss_share_index_for_edge_v1(EDGE_20) == 0
    assert rss_share_index_for_edge_v1(EDGE_01) == 1
    assert rss_share_index_for_edge_v1(EDGE_12) == 2

    # Each share is held by exactly two parties.
    holders = {}
    for share in (0, 1, 2):
        hs = []
        for pid in (0, 1, 2):
            a, b = rss_pair_share_indices_for_party_v1(pid)
            if share in (a, b):
                hs.append(pid)
        holders[share] = tuple(hs)

    assert holders[0] == (0, 2)
    assert holders[1] == (0, 1)
    assert holders[2] == (1, 2)


def test_rss_reconstruction_matches_public_value() -> None:
    gen = torch.Generator(device="cpu").manual_seed(999)
    x = torch.randint(0, 2**32, (128,), dtype=torch.int64, generator=gen)
    x = (x << 32) | torch.randint(0, 2**32, (128,), dtype=torch.int64, generator=gen)

    p0, p1, _p2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))
    # share0=a == p0.lo, share1=b == p0.hi, share2=c == p1.hi
    x_rec = (p0.lo + p0.hi + p1.hi).to(torch.int64)
    assert torch.equal(x_rec, x)


