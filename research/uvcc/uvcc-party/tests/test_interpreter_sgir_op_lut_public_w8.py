from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_1957bee5c681ad25

import concurrent.futures as cf
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.interpreter import SGIRInterpreterV1
from uvcc_party.op_lut_blob import build_oplut_record_blobs_v1
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.rss import make_rss_arith_u64_triple
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
    parse_sgir_module_v1,
)


def _align8(x: int) -> int:
    r = x % 8
    return x if r == 0 else (x + (8 - r))


def _build_min_sgir_lut_open_module(*, n: int) -> bytes:
    hdr_size = 76
    # TYPETAB: two types (sec u64[N], pub u64[N])
    typ = bytearray()
    typ += struct.pack("<II", 2, 0)
    for secrecy in (SEC_SEC, SEC_PUB):
        typ += struct.pack("<BBHHhI", secrecy, DT_U64, LYT_ROW_MAJOR, 1, 0, 0)
        typ += struct.pack("<BBHIQ", DIM_LIT, 0, 0, 0, int(n))
    typ_pl = bytes(typ)

    # VALTAB: 3 values: x (input SEC), y (tmp SEC), y_pub (output PUB)
    val = bytearray()
    val += struct.pack("<II", 3, 0)
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 1, 0)  # x
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 0, 0)  # y
    val += struct.pack("<IIII", 1, 0xFFFFFFFF, 2, 0)  # y_pub
    val_pl = bytes(val)

    # CODE: OP_LUT dst=1 src=0; OP_OPEN dst=2 src=1; OP_RET
    inst = bytearray()
    inst += struct.pack("<HHHHHH", 23, 0, 1, 1, 0, 0)  # OP_LUT
    inst += struct.pack("<I", 1)
    inst += struct.pack("<I", 0)
    inst += struct.pack("<HHHHHH", 60, 0, 1, 1, 0, 0)  # OP_OPEN
    inst += struct.pack("<I", 2)
    inst += struct.pack("<I", 1)
    inst += struct.pack("<HHHHHH", 52, 0, 0, 0, 0, 0)  # OP_RET
    code_pl = bytes(inst)

    # FUNTAB: one function, one block.
    fun = bytearray()
    fun += struct.pack("<II", 1, 0)
    blocks_off = 8 + 44  # header + one function record
    # name_str_id=0, entry_block_id=0, block_count=1, param_count=1, result_count=1
    fun += struct.pack("<IIIIIQQQ", 0, 0, 1, 1, 1, blocks_off, 0, len(code_pl))
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


def test_interpreter_op_lut_public_v1_then_open_w8() -> None:
    lanes = 2
    mod = parse_sgir_module_v1(_build_min_sgir_lut_open_module(n=lanes))

    gen = torch.Generator(device="cpu").manual_seed(2033)
    x_pub = torch.randint(0, 256, (lanes,), dtype=torch.int64, generator=gen)
    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x_pub, generator=gen, device=torch.device("cpu"))

    # Table: T[j] = j^2 (lifted into ring).
    T = (torch.arange(256, dtype=torch.int64) ** 2).to(torch.int64)
    y_expect = T[x_pub]

    # OP_LUT sgir_op_id32 = inst_index = 0 in this module.
    sgir_op_id = 0

    sid = b"sid-sgir-oplut"
    fss_id = 0x1111222233334444
    p0_blob, p1_blob, p2_blob = build_oplut_record_blobs_v1(
        sid=sid,
        fss_id=fss_id,
        sgir_op_id=sgir_op_id,
        domain_w=8,
        lanes=lanes,
        lane_base=0,
        K_master32=b"\x11" * 32,
        seed_edge01_32=b"\x22" * 32,
        seed_edge12_32=b"\x33" * 32,
        seed_edge20_32=b"\x44" * 32,
    )
    blobs = {0: p0_blob, 1: p1_blob, 2: p2_blob}
    xs = {0: x0, 1: x1, 2: x2}

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)
            job_id32 = b"\x77" * 32

            def run_party(pid: int):
                relay = RelayClient(base_url=base, group_id="g-sgir-oplut", token=None, timeout_s=20.0)
                p = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)
                itp = SGIRInterpreterV1(party=p, module=mod)
                itp.set_value(value_id=0, type_id=0, value=xs[pid])
                itp.register_op_lut_public_v1(sgir_op_id=sgir_op_id, table_u64=T, fss_blob=blobs[pid])
                itp.run_function0(epoch=0, step_base=0, sgir_fun_id=0)
                out = itp.get_value(2).value
                assert isinstance(out, torch.Tensor)
                return out

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0)
                f1 = ex.submit(run_party, 1)
                f2 = ex.submit(run_party, 2)
                o0 = f0.result(timeout=120)
                o1 = f1.result(timeout=120)
                o2 = f2.result(timeout=120)

            assert torch.equal(o0, y_expect)
            assert torch.equal(o1, y_expect)
            assert torch.equal(o2, y_expect)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


