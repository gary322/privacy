from __future__ import annotations

"""
DP-aware party training worker (SR-DP).

This is an additive overlay for `uvcc_client.party_train.run_party_train_v1` that adds:
- optional torch.distributed NCCL allreduce across replicas *within a party domain*
  (i.e. DP reduction on secret-share gradients).

It does NOT modify the MPC plane. Each replica still runs the same UVCC 3PC protocol
over the relay, isolated by its own sid/group_id.
"""

# pyright: reportMissingImports=false

import argparse
import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _repo_root() -> Path:
    # .../research/uvcc_parallel/party_train_dp.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _add_paths() -> None:
    import sys

    root = _repo_root()
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-client"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-party"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-verifier"))


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(str(s).encode("ascii"), validate=True)


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _hex32(b: bytes) -> str:
    if len(b) != 32:
        raise ValueError("expected 32-byte value")
    return "0x" + b.hex()


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _lvl_rank(level: str) -> int:
    lv = str(level or "").strip().lower()
    if lv in ("", "info"):
        return 20
    if lv in ("debug",):
        return 10
    if lv in ("trace",):
        return 0
    if lv in ("quiet", "off", "none"):
        return 100
    return 20


def _log_json(*, level: str, want: str, party_id: int, event: str, **fields: Any) -> None:
    """
    Emit a single JSON log line to stdout (captured in per-party run.log).
    This is meant to be readable by humans and machine-parsable.
    """
    if _lvl_rank(level) > _lvl_rank(want):
        return
    rec = {
        "ts": _now_iso_utc(),
        "party_id": int(party_id),
        "event": str(event),
        "fields": {str(k): v for k, v in fields.items()},
    }
    print(json.dumps(rec, sort_keys=True, separators=(",", ":")), flush=True)


def _u64_matrix_to_le_bytes(x_u64_i64) -> bytes:
    # x_u64_i64 is torch.Tensor but we avoid importing torch at file import time.
    if x_u64_i64.dtype != x_u64_i64.new_empty((), dtype=x_u64_i64.dtype).dtype:  # type: ignore[attr-defined]
        # dead code path (kept for type checkers); real dtype check below
        pass
    import torch

    if not isinstance(x_u64_i64, torch.Tensor) or x_u64_i64.dtype != torch.int64:
        raise TypeError("expected int64 tensor of u64 bit-patterns")
    xs = x_u64_i64.contiguous().view(-1).cpu().tolist()
    out = bytearray()
    for v in xs:
        out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _le_bytes_to_u64_matrix(buf: bytes, *, d: int, device) -> Any:
    import torch

    d = int(d)
    n = d * d
    if len(buf) != 8 * n:
        raise ValueError("bad u64 matrix byte length")
    out = torch.empty((n,), dtype=torch.int64, device=device)
    for i in range(n):
        out[i] = int.from_bytes(buf[8 * i : 8 * i + 8], "little", signed=True)
    return out.view(d, d)


def _tensor_hash32_u64_i64(x) -> str:
    return _hex32(_sha256(_u64_matrix_to_le_bytes(x)))


def _parse_hex_bytes(s: str, *, n: int) -> bytes:
    t = str(s).strip()
    if t.startswith("0x"):
        t = t[2:]
    b = bytes.fromhex(t)
    if len(b) != int(n):
        raise ValueError(f"expected {n} bytes")
    return b


@dataclass(frozen=True)
class DPConfigV1:
    enable: bool
    world_size: int
    rank: int
    master_addr: str
    master_port: int
    timeout_s: int = 180

    def __post_init__(self) -> None:
        if bool(self.enable) is False:
            return
        if int(self.world_size) <= 0:
            raise ValueError("dp.world_size must be > 0")
        if int(self.rank) < 0 or int(self.rank) >= int(self.world_size):
            raise ValueError("dp.rank must be in [0, world_size)")
        if not str(self.master_addr).strip():
            raise ValueError("dp.master_addr required")
        if int(self.master_port) <= 0 or int(self.master_port) > 65535:
            raise ValueError("dp.master_port invalid")
        if int(self.timeout_s) <= 0:
            raise ValueError("dp.timeout_s must be > 0")


def _dp_init(*, cfg: DPConfigV1, device, log_level: str, party_id: int) -> Optional[Any]:
    """
    Initialize torch.distributed process group for DP (NCCL).
    Returns torch.distributed module if enabled; else None.
    """
    if not bool(cfg.enable) or int(cfg.world_size) <= 1:
        return None
    import datetime
    import torch
    import torch.distributed as dist

    if device.type != "cuda":
        raise RuntimeError("DP NCCL requires cuda device")

    # On 1-GPU pods, we always use cuda:0.
    torch.cuda.set_device(int(device.index or 0))

    # NCCL defaults for heterogeneous cloud providers:
    # - Force IPv4 sockets
    # - Prefer common public NIC names (avoid docker0)
    # - Disable IB paths on networks that don't support it
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    # Exclude-mode is robust across heterogeneous providers with different NIC naming.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker0")

    init_method = f"tcp://{cfg.master_addr}:{int(cfg.master_port)}"
    _log_json(
        level=str(log_level),
        want="info",
        party_id=int(party_id),
        event="dp_init_start",
        backend="nccl",
        init_method=str(init_method),
        world_size=int(cfg.world_size),
        rank=int(cfg.rank),
        timeout_s=int(cfg.timeout_s),
    )
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=int(cfg.world_size),
        rank=int(cfg.rank),
        timeout=datetime.timedelta(seconds=int(cfg.timeout_s)),
    )
    # Use an actual CUDA collective as a readiness check (more representative than barrier()).
    t = torch.tensor([int(cfg.rank) + 1], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    _log_json(level=str(log_level), want="info", party_id=int(party_id), event="dp_init_done")
    return dist


def run_party_train_dp_v1(
    *,
    party_id: int,
    replica_id: int,
    relay_base_url: str,
    relay_group_id: str,
    relay_token: Optional[str],
    tls_ca_pem: Optional[str],
    job_id32: bytes,
    sid: bytes,
    inputs_json_path: str,
    out_dir: str,
    device,
    require_cuda: bool,
    steps: int = 1,
    epoch: int = 0,
    step_offset: int = 0,
    epoch_setup_step: int = 1000,
    checkpoint_enable: bool = False,
    checkpoint_every: int = 1,
    sks_t_checks: int = 3,
    sks_sample_log2: int = 0,
    log_level: str = "info",
    dp: Optional[DPConfigV1] = None,
) -> Tuple[bytes, bytes]:
    """
    DP-aware training loop:
    - Same MPC compute as uvcc_client.party_train.run_party_train_v1
    - After computing gradient shares, do DP allreduce (SUM) across replicas within a party domain
    """
    _add_paths()
    import torch

    from uvcc_client.party_train import PartyTrainInputsV1
    from uvcc_party.gemm import op_gemm_tile_beaver_tcf_v0a_u64_v1
    from uvcc_party.open import OpenArithItemU64, open_arith_u64_round_v1
    from uvcc_party.party import Party
    from uvcc_party.relay_client import RelayClient
    from uvcc_party.rss import RSSArithU64
    from uvcc_party.sks import sks_epoch_setup_v1, sks_freivalds_check_tile_gemm_u64_v1
    from uvcc_party.tcf import TCFKeyV1

    if require_cuda and device.type != "cuda":
        raise RuntimeError("require_cuda=true but selected device is not cuda")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("require_cuda=true but torch.cuda.is_available() is false")

    dpcfg = dp if dp is not None else DPConfigV1(enable=False, world_size=1, rank=0, master_addr="", master_port=0)
    dist = _dp_init(cfg=dpcfg, device=device, log_level=str(log_level), party_id=int(party_id))

    inputs = PartyTrainInputsV1.from_json(str(inputs_json_path))
    d = int(inputs.d)
    fxp = int(inputs.fxp_frac_bits)

    def hex32(s: str) -> bytes:
        t = str(s)
        if not t.startswith("0x") or len(t) != 66:
            raise ValueError("expected 0x + 64 hex")
        return bytes.fromhex(t[2:])

    def hex_any(s: str, n: int) -> bytes:
        t = str(s)
        if t.startswith("0x"):
            t = t[2:]
        b = bytes.fromhex(t)
        if len(b) != n:
            raise ValueError(f"expected {n} bytes")
        return b

    tcf = inputs.tcf_key_dict
    tcf_key = TCFKeyV1(
        sid_hash32=hex32(tcf["sid_hash32_hex"]),
        s01=hex_any(tcf["s01_hex"], 32),
        s02=hex_any(tcf["s02_hex"], 32),
        s12=hex_any(tcf["s12_hex"], 32),
    )

    X = RSSArithU64(lo=_le_bytes_to_u64_matrix(_b64d(inputs.X_lo_b64), d=d, device=device), hi=_le_bytes_to_u64_matrix(_b64d(inputs.X_hi_b64), d=d, device=device), fxp_frac_bits=fxp)
    Y = RSSArithU64(lo=_le_bytes_to_u64_matrix(_b64d(inputs.Y_lo_b64), d=d, device=device), hi=_le_bytes_to_u64_matrix(_b64d(inputs.Y_hi_b64), d=d, device=device), fxp_frac_bits=fxp)
    W = RSSArithU64(lo=_le_bytes_to_u64_matrix(_b64d(inputs.W_lo_b64), d=d, device=device), hi=_le_bytes_to_u64_matrix(_b64d(inputs.W_hi_b64), d=d, device=device), fxp_frac_bits=fxp)

    outp = Path(str(out_dir)).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)

    # Private artifacts (DO NOT SHARE): checkpoint shares for mid-run recovery.
    priv_dir = outp / "private"
    ckpt_path = priv_dir / "checkpoints_W.jsonl"
    if bool(checkpoint_enable):
        priv_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(priv_dir), 0o700)
        except Exception:
            pass

    # Transcript live dump (append-only) so partial progress is verifiable even if the process crashes.
    transcript_live = outp / "transcript_v1_live.jsonl"
    last_flushed = 0

    def flush_transcript_live(party_obj: Party) -> None:
        nonlocal last_flushed
        ts = party_obj.transcript
        if ts is None:
            return
        leaves = ts.leaves()
        if len(leaves) <= last_flushed:
            return
        with open(transcript_live, "a", encoding="utf-8") as f:
            for lf in leaves[last_flushed:]:
                f.write(json.dumps({"body_b64": _b64e(lf.body_bytes)}, sort_keys=True, separators=(",", ":")) + "\n")
        last_flushed = len(leaves)

    # Start logging.
    _log_json(
        level=str(log_level),
        want="info",
        party_id=int(party_id),
        event="party_start",
        replica_id=int(replica_id),
        device=str(device),
        require_cuda=bool(require_cuda),
        torch_version=str(torch.__version__),
        torch_cuda_available=bool(torch.cuda.is_available()),
        d=int(d),
        steps=int(steps),
        sks_t_checks=int(sks_t_checks),
        sks_sample_log2=int(sks_sample_log2),
        dp_enable=bool(dpcfg.enable),
        dp_world_size=int(dpcfg.world_size) if bool(dpcfg.enable) else 1,
        dp_rank=int(dpcfg.rank) if bool(dpcfg.enable) else 0,
    )
    if device.type == "cuda":
        try:
            _log_json(
                level=str(log_level),
                want="debug",
                party_id=int(party_id),
                event="cuda_device_info",
                device_index=int(device.index or 0),
                device_name=str(torch.cuda.get_device_name(device)),
                capability=str(torch.cuda.get_device_capability(device)),
            )
        except Exception:
            pass

    # Trace-level: record safe commitments to secret inputs (hashes only; never print shares).
    _log_json(
        level=str(log_level),
        want="debug",
        party_id=int(party_id),
        event="inputs_commitments",
        X_lo_hash32=_tensor_hash32_u64_i64(X.lo),
        X_hi_hash32=_tensor_hash32_u64_i64(X.hi),
        Y_lo_hash32=_tensor_hash32_u64_i64(Y.lo),
        Y_hi_hash32=_tensor_hash32_u64_i64(Y.hi),
        W_lo_hash32=_tensor_hash32_u64_i64(W.lo),
        W_hi_hash32=_tensor_hash32_u64_i64(W.hi),
        fxp_frac_bits=int(fxp),
    )

    relay = RelayClient(base_url=str(relay_base_url), group_id=str(relay_group_id), token=relay_token, timeout_s=240.0, tls_ca_pem=tls_ca_pem)
    party = Party(party_id=int(party_id), job_id32=bytes(job_id32), sid=bytes(sid), relay=relay)

    # SKS epoch setup.
    st = sks_epoch_setup_v1(party, sid=bytes(sid), epoch=int(epoch), step=int(epoch_setup_step))
    if st.epoch_rand32 is None or len(st.epoch_rand32) != 32:
        raise RuntimeError("missing epoch_rand32")
    _log_json(
        level=str(log_level),
        want="debug",
        party_id=int(party_id),
        event="sks_epoch_setup_done",
        epoch=int(st.epoch),
        commit32_hex=_hex32(bytes(st.commit32)),
        epoch_rand32_hash32=_hex32(_sha256(bytes(st.epoch_rand32))),
    )
    flush_transcript_live(party)

    for step_i_local in range(int(steps)):
        step_i = int(step_offset) + int(step_i_local)
        t_step0 = time.monotonic()
        _log_json(level=str(log_level), want="info", party_id=int(party_id), event="step_start", step=int(step_i))

        # "Training" step: P = X@W, E = P - Y, G = X^T@E, W <- W - DP_ALLREDUCE(G)
        t0 = time.monotonic()
        res1 = op_gemm_tile_beaver_tcf_v0a_u64_v1(
            party,
            X=X,
            Y=W,
            tcf_key=tcf_key,
            op_id=100 + int(step_i),
            tile_i=0,
            tile_j=0,
            tile_p=0,
            epoch=int(epoch),
            step=10_000 + 100 * int(step_i),
            sgir_op_id=10_000 + int(step_i),
            fxp_frac_bits=int(fxp),
            d=d,
        )
        _log_json(
            level=str(log_level),
            want="debug",
            party_id=int(party_id),
            event="gemm1_done",
            step=int(step_i),
            t_s=round(time.monotonic() - t0, 6),
            E_pub_hash32=_tensor_hash32_u64_i64(res1.E_pub),
            F_pub_hash32=_tensor_hash32_u64_i64(res1.F_pub),
        )
        t0 = time.monotonic()
        ok1 = sks_freivalds_check_tile_gemm_u64_v1(
            party,
            sid=bytes(sid),
            epoch_rand32=bytes(st.epoch_rand32),
            epoch=int(epoch),
            step=20_000 + 100 * int(step_i),
            sgir_op_id=10_000 + int(step_i),
            kernel_instance_id=int(step_i),
            sks_sample_log2=int(sks_sample_log2),
            t_checks=int(sks_t_checks),
            field_id=0,
            Z=res1.Z,
            triple_A=res1.triple_A,
            triple_B=res1.triple_B,
            triple_C=res1.triple_C,
            E_pub=res1.E_pub,
            F_pub=res1.F_pub,
        )
        _log_json(level=str(log_level), want="info", party_id=int(party_id), event="sks1_done", step=int(step_i), ok=bool(ok1), t_s=round(time.monotonic() - t0, 6))
        if ok1 is not True:
            raise RuntimeError("SKS check failed for X@W")

        E = res1.Z.sub(Y)

        t0 = time.monotonic()
        res2 = op_gemm_tile_beaver_tcf_v0a_u64_v1(
            party,
            X=X,  # X^T == X for the default identity input.
            Y=E,
            tcf_key=tcf_key,
            op_id=200 + int(step_i),
            tile_i=0,
            tile_j=0,
            tile_p=0,
            epoch=int(epoch),
            step=30_000 + 100 * int(step_i),
            sgir_op_id=20_000 + int(step_i),
            fxp_frac_bits=int(fxp),
            d=d,
        )
        _log_json(
            level=str(log_level),
            want="debug",
            party_id=int(party_id),
            event="gemm2_done",
            step=int(step_i),
            t_s=round(time.monotonic() - t0, 6),
            E_pub_hash32=_tensor_hash32_u64_i64(res2.E_pub),
            F_pub_hash32=_tensor_hash32_u64_i64(res2.F_pub),
        )
        t0 = time.monotonic()
        ok2 = sks_freivalds_check_tile_gemm_u64_v1(
            party,
            sid=bytes(sid),
            epoch_rand32=bytes(st.epoch_rand32),
            epoch=int(epoch),
            step=40_000 + 100 * int(step_i),
            sgir_op_id=20_000 + int(step_i),
            kernel_instance_id=1000 + int(step_i),
            sks_sample_log2=int(sks_sample_log2),
            t_checks=int(sks_t_checks),
            field_id=0,
            Z=res2.Z,
            triple_A=res2.triple_A,
            triple_B=res2.triple_B,
            triple_C=res2.triple_C,
            E_pub=res2.E_pub,
            F_pub=res2.F_pub,
        )
        _log_json(level=str(log_level), want="info", party_id=int(party_id), event="sks2_done", step=int(step_i), ok=bool(ok2), t_s=round(time.monotonic() - t0, 6))
        if ok2 is not True:
            raise RuntimeError("SKS check failed for X^T@E")

        # DP reduce on the gradient shares (within each party domain).
        G = res2.Z
        if dist is not None:
            _log_json(level=str(log_level), want="debug", party_id=int(party_id), event="dp_allreduce_start", step=int(step_i))
            t_dp0 = time.monotonic()
            dist.all_reduce(G.lo, op=dist.ReduceOp.SUM)
            dist.all_reduce(G.hi, op=dist.ReduceOp.SUM)
            dist.barrier()
            _log_json(
                level=str(log_level),
                want="debug",
                party_id=int(party_id),
                event="dp_allreduce_done",
                step=int(step_i),
                t_s=round(time.monotonic() - t_dp0, 6),
                G_lo_hash32=_tensor_hash32_u64_i64(G.lo),
                G_hi_hash32=_tensor_hash32_u64_i64(G.hi),
            )

        W = W.sub(G)

        w_lo_h = _tensor_hash32_u64_i64(W.lo)
        w_hi_h = _tensor_hash32_u64_i64(W.hi)
        _log_json(
            level=str(log_level),
            want="debug",
            party_id=int(party_id),
            event="step_done",
            step=int(step_i),
            t_s=round(time.monotonic() - t_step0, 6),
            W_lo_hash32=str(w_lo_h),
            W_hi_hash32=str(w_hi_h),
        )
        flush_transcript_live(party)

        if bool(checkpoint_enable) and ((int(step_i_local) + 1) % int(checkpoint_every) == 0):
            priv_dir.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": _now_iso_utc(),
                "party_id": int(party_id),
                "replica_id": int(replica_id),
                "epoch": int(epoch),
                "step": int(step_i),
                "d": int(d),
                "fxp_frac_bits": int(fxp),
                "W_lo_hash32": str(w_lo_h),
                "W_hi_hash32": str(w_hi_h),
                "W_lo_b64": _b64e(_u64_matrix_to_le_bytes(W.lo)),
                "W_hi_b64": _b64e(_u64_matrix_to_le_bytes(W.hi)),
            }
            with open(ckpt_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
            try:
                os.chmod(str(ckpt_path), 0o600)
            except Exception:
                pass
            _log_json(
                level=str(log_level),
                want="info",
                party_id=int(party_id),
                event="checkpoint_written",
                epoch=int(epoch),
                step=int(step_i),
                path=str(ckpt_path),
                W_lo_hash32=str(w_lo_h),
                W_hi_hash32=str(w_hi_h),
            )

    # OPEN final weights and compute stable result hash.
    open_id = 0xC0FFEE01
    t_open0 = time.monotonic()
    out = open_arith_u64_round_v1(
        party,
        items=[OpenArithItemU64(open_id=open_id, sub_id=0, x=W)],
        epoch=int(epoch),
        step=50_000,
        round=0,
        sgir_op_id=0x51515151,
    )
    W_pub = out[(open_id, 0)].view(d, d).contiguous()
    result_hash32 = _sha256(_u64_matrix_to_le_bytes(W_pub))
    _log_json(
        level=str(log_level),
        want="info",
        party_id=int(party_id),
        event="open_final_done",
        t_s=round(time.monotonic() - t_open0, 6),
        result_hash32_hex=_hex32(bytes(result_hash32)),
        W_pub_hash32=_tensor_hash32_u64_i64(W_pub),
    )
    flush_transcript_live(party)

    leaves = party.transcript.leaves() if party.transcript is not None else []
    transcript_jsonl = "\n".join(json.dumps({"body_b64": _b64e(lf.body_bytes)}, sort_keys=True, separators=(",", ":")) for lf in leaves) + "\n"
    transcript_bytes = transcript_jsonl.encode("utf-8")
    _log_json(
        level=str(log_level),
        want="debug",
        party_id=int(party_id),
        event="transcript_summary",
        leaf_count=int(len(leaves)),
        transcript_hash32_hex=_hex32(_sha256(bytes(transcript_bytes))),
    )

    (outp / "transcript_v1.jsonl").write_bytes(transcript_bytes)
    (outp / "result.json").write_text(
        json.dumps(
            {
                "party_id": int(party_id),
                "replica_id": int(replica_id),
                "result_hash32_hex": _hex32(result_hash32),
                "device": str(device),
                "sks_t_checks": int(sks_t_checks),
                "sks_sample_log2": int(sks_sample_log2),
                "dp_enable": bool(dpcfg.enable),
                "dp_world_size": int(dpcfg.world_size) if bool(dpcfg.enable) else 1,
                "dp_rank": int(dpcfg.rank) if bool(dpcfg.enable) else 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    # Best-effort shutdown.
    if dist is not None:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    return result_hash32, transcript_bytes


def main(argv: Optional[list[str]] = None) -> int:
    _add_paths()
    import torch

    ap = argparse.ArgumentParser(prog="party_train_dp.py")
    ap.add_argument("--party-id", required=True, type=int, choices=[0, 1, 2])
    ap.add_argument("--replica-id", required=True, type=int, help="Replica id r (also DP rank by default)")

    ap.add_argument("--relay-url", required=True)
    ap.add_argument("--relay-group-id", required=True)
    ap.add_argument("--relay-token", default=None)
    ap.add_argument("--relay-token-file", default=None)
    ap.add_argument("--tls-ca-pem", default=None)

    ap.add_argument("--job-id-hex", required=True, help="0x + 64 hex job_id32")
    ap.add_argument("--sid-hex", required=True, help="0x + 64 hex sid (32 bytes) for this replica/subgroup")

    ap.add_argument("--inputs-json", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--require-cuda", default="true", choices=["true", "false"])
    ap.add_argument("--steps", default=1, type=int)
    ap.add_argument("--epoch", default=0, type=int)
    ap.add_argument("--step-offset", default=0, type=int)
    ap.add_argument("--epoch-setup-step", default=1000, type=int)
    ap.add_argument("--checkpoint-enable", default="false", choices=["true", "false"])
    ap.add_argument("--checkpoint-every", default=1, type=int)
    ap.add_argument("--sks-t-checks", default=3, type=int)
    ap.add_argument("--sks-sample-log2", default=0, type=int)
    ap.add_argument("--log-level", default="info", choices=["quiet", "info", "debug", "trace"])

    # DP options
    ap.add_argument("--dp-enable", default="false", choices=["true", "false"])
    ap.add_argument("--dp-world-size", default=1, type=int)
    ap.add_argument("--dp-rank", default=None, type=int, help="If omitted, defaults to --replica-id")
    ap.add_argument("--dp-master-addr", default=None, help="Master addr for torch.distributed init_method")
    ap.add_argument("--dp-master-port", default=None, type=int, help="Master port for torch.distributed init_method")
    ap.add_argument("--dp-timeout-s", default=180, type=int)

    args = ap.parse_args(argv)

    device_s = str(args.device or "auto").strip().lower()
    if device_s == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_s == "cuda":
        dev = torch.device("cuda")
    elif device_s == "cpu":
        dev = torch.device("cpu")
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")

    relay_token = str(args.relay_token).strip() if args.relay_token is not None else ""
    if not relay_token and args.relay_token_file is not None:
        relay_token = Path(str(args.relay_token_file)).expanduser().read_text(encoding="utf-8").strip()
    if not relay_token:
        relay_token = str(os.environ.get("UVCC_RELAY_TOKEN", "")).strip()
    relay_token_opt = relay_token if relay_token else None

    dp_enable = str(args.dp_enable).lower() == "true"
    dp_rank = int(args.dp_rank) if args.dp_rank is not None else int(args.replica_id)
    dp_master_addr = str(args.dp_master_addr).strip() if args.dp_master_addr is not None else ""
    dp_master_port = int(args.dp_master_port) if args.dp_master_port is not None else 0
    dpcfg = DPConfigV1(
        enable=bool(dp_enable),
        world_size=int(args.dp_world_size),
        rank=int(dp_rank),
        master_addr=dp_master_addr,
        master_port=int(dp_master_port),
        timeout_s=int(args.dp_timeout_s),
    )

    run_party_train_dp_v1(
        party_id=int(args.party_id),
        replica_id=int(args.replica_id),
        relay_base_url=str(args.relay_url),
        relay_group_id=str(args.relay_group_id),
        relay_token=relay_token_opt,
        tls_ca_pem=str(args.tls_ca_pem) if args.tls_ca_pem else None,
        job_id32=_parse_hex_bytes(str(args.job_id_hex), n=32),
        sid=_parse_hex_bytes(str(args.sid_hex), n=32),
        inputs_json_path=str(args.inputs_json),
        out_dir=str(args.out),
        device=dev,
        require_cuda=(str(args.require_cuda).lower() == "true"),
        steps=int(args.steps),
        epoch=int(args.epoch),
        step_offset=int(args.step_offset),
        epoch_setup_step=int(args.epoch_setup_step),
        checkpoint_enable=(str(args.checkpoint_enable).lower() == "true"),
        checkpoint_every=int(args.checkpoint_every),
        sks_t_checks=int(args.sks_t_checks),
        sks_sample_log2=int(args.sks_sample_log2),
        log_level=str(args.log_level),
        dp=dpcfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


