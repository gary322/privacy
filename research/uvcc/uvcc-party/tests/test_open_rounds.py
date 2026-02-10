from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_516fb18a3bfe557c,uvcc_group_df382033ede3f858,uvcc_group_20bb5431060336ca

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.open import (
    MSG_OPEN_ARITH_RESULT,
    MSG_OPEN_BOOL_RESULT,
    OpenArithItemU64,
    OpenBoolItemWords,
    open_arith_u64_round_v1,
    open_bool_words_round_v1,
)
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.rss import RSSArithU64, RSSBoolU64Words, make_rss_arith_u64_triple


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _start_relay(port: int, db_path: str) -> subprocess.Popen:
    repo_root = Path(__file__).resolve().parents[4]
    relay_py = repo_root / "research" / "uvcc" / "uvcc-relay" / "relay_server.py"
    assert relay_py.exists()
    return subprocess.Popen(
        [
            sys.executable,
            str(relay_py),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--db",
            db_path,
            "--require-token",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_health(base_url: str) -> None:
    rc = RelayClient(base_url=base_url, group_id="health", token=None, timeout_s=2.0)
    for _ in range(200):
        try:
            rc.healthz()
            return
        except Exception:
            time.sleep(0.02)
    raise RuntimeError("relay never became healthy")


def test_open_arith_and_bool_rounds_record_transcript_leaves() -> None:
    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            job_id32 = b"\x33" * 32
            sid = b"sid-open-rounds"
            gen = torch.Generator(device="cpu").manual_seed(1234)

            # Arithmetic OPEN round: open one vector.
            n = 17
            x = torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) | (torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) << 32)
            x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))

            # Boolean OPEN round: open one packed bitvector (same N bits for all parties).
            n_bits = 65
            n_words64 = (n_bits + 63) // 64
            lo0 = torch.randint(0, 2**32, (n_words64,), dtype=torch.int64, generator=gen) | (torch.randint(0, 2**32, (n_words64,), dtype=torch.int64, generator=gen) << 32)
            hi0 = torch.randint(0, 2**32, (n_words64,), dtype=torch.int64, generator=gen) | (torch.randint(0, 2**32, (n_words64,), dtype=torch.int64, generator=gen) << 32)
            # Choose a public bitvector so reconstruction is well-defined: b_pub = lo_prev ^ lo ^ hi (for party0 view).
            b_pub = lo0 ^ hi0 ^ torch.zeros_like(lo0)
            # Mask unused bits so we match OPEN_BOOL canonical packing (high bits in last word are zero).
            rem = n_bits % 64
            if rem != 0 and n_words64 > 0:
                mask = (1 << rem) - 1
                lo0[-1] = int(lo0[-1].item()) & mask
                hi0[-1] = int(hi0[-1].item()) & mask
                b_pub[-1] = int(b_pub[-1].item()) & mask
            # Create replicated boolean shares consistent with b_pub.
            # Pick b0,b1 random; b2 = b_pub ^ b0 ^ b1.
            b0 = lo0
            b1 = hi0
            b2 = b_pub ^ b0 ^ b1
            b_p0 = RSSBoolU64Words(lo_words=b0, hi_words=b1, n_bits=n_bits)
            b_p1 = RSSBoolU64Words(lo_words=b1, hi_words=b2, n_bits=n_bits)
            b_p2 = RSSBoolU64Words(lo_words=b2, hi_words=b0, n_bits=n_bits)

            def run_party(party_id: int, xs: RSSArithU64, bs: RSSBoolU64Words):
                relay = RelayClient(base_url=base, group_id="g-open", token=None, timeout_s=10.0)
                p = Party(party_id=party_id, job_id32=job_id32, sid=sid, relay=relay)

                pub_x = open_arith_u64_round_v1(
                    p,
                    items=[OpenArithItemU64(open_id=10, sub_id=0, x=xs)],
                    epoch=0,
                    step=0,
                    round=0,
                    sgir_op_id=0,
                )[(10, 0)]
                pub_b = open_bool_words_round_v1(
                    p,
                    items=[OpenBoolItemWords(open_id=20, sub_id=0, x=bs)],
                    epoch=0,
                    step=1,
                    round=0,
                    sgir_op_id=0,
                )[(20, 0)]

                # Each OPEN call records 2 leaves (SEND+RECV) for this party.
                leaves = p.transcript.leaves() if p.transcript is not None else []
                return pub_x, pub_b, leaves

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, x0, b_p0)
                f1 = ex.submit(run_party, 1, x1, b_p1)
                f2 = ex.submit(run_party, 2, x2, b_p2)
                o0 = f0.result(timeout=60)
                o1 = f1.result(timeout=60)
                o2 = f2.result(timeout=60)

            # OPEN_ARITH reconstruction should match plaintext x for all parties.
            for pub_x, _, _ in (o0, o1, o2):
                assert torch.equal(pub_x, x)

            # OPEN_BOOL reconstruction should match public b_pub (packed u64 words) for all parties.
            for _, pub_b, _ in (o0, o1, o2):
                assert torch.equal(pub_b, b_pub)

            # Transcript leaves:
            # - OPEN_ARITH: SEND + RECV + OPEN_ARITH_RESULT => 3
            # - OPEN_BOOL:  SEND + RECV + OPEN_BOOL_RESULT  => 3
            # Total: 6 leaves per party.
            for _, _, leaves in (o0, o1, o2):
                assert len(leaves) == 6
                # Presence checks for result leaves.
                kinds = [int(l.prefix.msg_kind) for l in leaves]
                assert MSG_OPEN_ARITH_RESULT in kinds
                assert MSG_OPEN_BOOL_RESULT in kinds
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


