from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_1957bee5c681ad25

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.interpreter import SGIRInterpreterV1
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.rss import make_rss_arith_u64_triple
from uvcc_party.sgir import parse_sgir_module_v1

import struct
from uvcc_party.sgir import (
    DIM_LIT,
    DT_U64,
    ENDIAN_LITTLE,
    HASH_SHA256,
    LYT_ROW_MAJOR,
    SEC_PUB,
    SEC_SEC,
    SECT_CODE,
    SECT_FUNTAB,
    SECT_TYPETAB,
    SECT_VALTAB,
)


def _align8(x: int) -> int:
    r = x % 8
    return x if r == 0 else (x + (8 - r))


def _build_min_sgir_add_open_module(*, n: int) -> bytes:
    hdr_size = 76
    # TYPETAB: two types (sec u64[N], pub u64[N])
    typ = bytearray()
    typ += struct.pack("<II", 2, 0)
    for secrecy in (SEC_SEC, SEC_PUB):
        typ += struct.pack("<BBHHhI", secrecy, DT_U64, LYT_ROW_MAJOR, 1, 0, 0)
        typ += struct.pack("<BBHIQ", DIM_LIT, 0, 0, 0, int(n))
    typ_pl = bytes(typ)

    # VALTAB: 4 values: x,y,z,z_pub
    val = bytearray()
    val += struct.pack("<II", 4, 0)
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 1, 0)  # x
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 1, 0)  # y
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 0, 0)  # z
    val += struct.pack("<IIII", 1, 0xFFFFFFFF, 2, 0)  # z_pub
    val_pl = bytes(val)

    # CODE
    inst = bytearray()
    inst += struct.pack("<HHHHHH", 10, 0, 1, 2, 0, 0)  # OP_ADD
    inst += struct.pack("<I", 2)
    inst += struct.pack("<II", 0, 1)
    inst += struct.pack("<HHHHHH", 60, 0, 1, 1, 0, 0)  # OP_OPEN
    inst += struct.pack("<I", 3)
    inst += struct.pack("<I", 2)
    inst += struct.pack("<HHHHHH", 52, 0, 0, 0, 0, 0)  # OP_RET
    code_pl = bytes(inst)

    # FUNTAB
    fun = bytearray()
    fun += struct.pack("<II", 1, 0)
    blocks_off = 8 + 44
    fun += struct.pack("<IIIIIQQQ", 0, 0, 1, 2, 1, blocks_off, 0, len(code_pl))
    fun += struct.pack("<IIQQ", 0, 0, 0, len(code_pl))
    fun_pl = bytes(fun)

    payloads = [
        (SECT_TYPETAB, typ_pl),
        (SECT_VALTAB, val_pl),
        (SECT_FUNTAB, fun_pl),
        (SECT_CODE, code_pl),
    ]

    cur = _align8(hdr_size)
    sect_meta = []
    blob = bytearray(b"\x00" * hdr_size)
    for kind, pl in payloads:
        cur = _align8(cur)
        if len(blob) < cur:
            blob += b"\x00" * (cur - len(blob))
        off = cur
        blob += pl
        cur = off + len(pl)
        sect_meta.append((kind, off, len(pl)))

    section_table_off = _align8(cur)
    if len(blob) < section_table_off:
        blob += b"\x00" * (section_table_off - len(blob))

    for kind, off, ln in sect_meta:
        blob += struct.pack("<IIQQ32s", int(kind), 0, int(off), int(ln), b"\x00" * 32)

    file_bytes = len(blob)
    hdr = struct.pack(
        "<4sHHIIIIBBHQ16s16s8s",
        b"SGIR",
        0,
        1,
        hdr_size,
        0,
        len(sect_meta),
        section_table_off,
        HASH_SHA256,
        ENDIAN_LITTLE,
        0,
        file_bytes,
        b"\x00" * 16,
        b"\x00" * 16,
        b"\x00" * 8,
    )
    blob[0:hdr_size] = hdr
    return bytes(blob)


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


def test_interpreter_add_then_open_u64() -> None:
    n = 17
    mod = parse_sgir_module_v1(_build_min_sgir_add_open_module(n=n))

    gen = torch.Generator(device="cpu").manual_seed(999)
    x = torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) | (torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) << 32)
    y = torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) | (torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) << 32)
    z_expect = x + y

    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))
    y0, y1, y2 = make_rss_arith_u64_triple(x_pub=y, generator=gen, device=torch.device("cpu"))

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)
            job_id32 = b"\x44" * 32
            sid = b"sid-sgir-int"

            def run_party(pid: int, xs, ys):
                relay = RelayClient(base_url=base, group_id="g-sgir", token=None, timeout_s=10.0)
                p = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)
                itp = SGIRInterpreterV1(party=p, module=mod)
                # value_id 0 and 1 are x and y, type_id 0 is SEC u64[N].
                itp.set_value(value_id=0, type_id=0, value=xs)
                itp.set_value(value_id=1, type_id=0, value=ys)
                itp.run_function0(epoch=0, step_base=0, sgir_fun_id=0)
                out = itp.get_value(3).value
                assert isinstance(out, torch.Tensor)
                return out

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, x0, y0)
                f1 = ex.submit(run_party, 1, x1, y1)
                f2 = ex.submit(run_party, 2, x2, y2)
                o0 = f0.result(timeout=60)
                o1 = f1.result(timeout=60)
                o2 = f2.result(timeout=60)

            assert torch.equal(o0, z_expect)
            assert torch.equal(o1, z_expect)
            assert torch.equal(o2, z_expect)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


