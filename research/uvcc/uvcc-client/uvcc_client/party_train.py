from __future__ import annotations

# pyright: reportMissingImports=false

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


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


def _u64_matrix_to_le_bytes(x_u64_i64: torch.Tensor) -> bytes:
    if x_u64_i64.dtype != torch.int64:
        raise TypeError("expected int64 tensor of u64 bit-patterns")
    xs = x_u64_i64.contiguous().view(-1).cpu().tolist()
    out = bytearray()
    for v in xs:
        out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _le_bytes_to_u64_matrix(buf: bytes, *, d: int, device: torch.device) -> torch.Tensor:
    d = int(d)
    n = d * d
    if len(buf) != 8 * n:
        raise ValueError("bad u64 matrix byte length")
    out = torch.empty((n,), dtype=torch.int64, device=device)
    for i in range(n):
        out[i] = int.from_bytes(buf[8 * i : 8 * i + 8], "little", signed=True)
    return out.view(d, d)


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


def _tensor_hash32_u64_i64(x: torch.Tensor) -> str:
    """
    sha256 of u64 LE bytes for int64 tensors that carry u64 bit-patterns.
    """
    return _hex32(_sha256(_u64_matrix_to_le_bytes(x)))


@dataclass(frozen=True)
class PartyTrainInputsV1:
    d: int
    fxp_frac_bits: int
    tcf_key_dict: Dict[str, str]
    X_lo_b64: str
    X_hi_b64: str
    Y_lo_b64: str
    Y_hi_b64: str
    W_lo_b64: str
    W_hi_b64: str

    @staticmethod
    def from_json(path: str) -> "PartyTrainInputsV1":
        p = Path(str(path)).expanduser().resolve()
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("inputs must be a json object")
        d = int(obj.get("d"))
        fxp = int(obj.get("fxp_frac_bits", 0))
        tcf = obj.get("tcf_key")
        if not isinstance(tcf, dict):
            raise ValueError("missing tcf_key")
        for k in ("sid_hash32_hex", "s01_hex", "s02_hex", "s12_hex"):
            if not isinstance(tcf.get(k), str):
                raise ValueError(f"tcf_key.{k} missing")
        def need_s(name: str) -> str:
            v = obj.get(name)
            if not isinstance(v, str):
                raise ValueError(f"missing {name}")
            return v
        return PartyTrainInputsV1(
            d=d,
            fxp_frac_bits=fxp,
            tcf_key_dict={str(k): str(v) for k, v in tcf.items()},
            X_lo_b64=need_s("X_lo_b64"),
            X_hi_b64=need_s("X_hi_b64"),
            Y_lo_b64=need_s("Y_lo_b64"),
            Y_hi_b64=need_s("Y_hi_b64"),
            W_lo_b64=need_s("W_lo_b64"),
            W_hi_b64=need_s("W_hi_b64"),
        )


def run_party_train_v1(
    *,
    party_id: int,
    relay_base_url: str,
    relay_group_id: str,
    relay_token: Optional[str],
    tls_ca_pem: Optional[str],
    job_id32: bytes,
    sid: bytes,
    inputs: PartyTrainInputsV1,
    out_dir: str,
    device: torch.device,
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
) -> Tuple[bytes, bytes]:
    if require_cuda and device.type != "cuda":
        raise RuntimeError("require_cuda=true but selected device is not cuda")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("require_cuda=true but torch.cuda.is_available() is false")

    from uvcc_party.gemm import op_gemm_tile_beaver_tcf_v0a_u64_v1
    from uvcc_party.open import OpenArithItemU64, open_arith_u64_round_v1
    from uvcc_party.party import Party
    from uvcc_party.relay_client import RelayClient
    from uvcc_party.rss import RSSArithU64
    from uvcc_party.sks import sks_epoch_setup_v1, sks_freivalds_check_tile_gemm_u64_v1
    from uvcc_party.tcf import TCFKeyV1

    d = int(inputs.d)
    if d <= 0:
        raise ValueError("d must be > 0")
    if int(epoch) < 0:
        raise ValueError("epoch must be >= 0")
    if int(step_offset) < 0:
        raise ValueError("step_offset must be >= 0")
    if int(epoch_setup_step) < 0:
        raise ValueError("epoch_setup_step must be >= 0")
    if int(steps) < 0:
        raise ValueError("steps must be >= 0")
    if int(checkpoint_every) <= 0:
        raise ValueError("checkpoint_every must be >= 1")

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

    X = RSSArithU64(
        lo=_le_bytes_to_u64_matrix(_b64d(inputs.X_lo_b64), d=d, device=device),
        hi=_le_bytes_to_u64_matrix(_b64d(inputs.X_hi_b64), d=d, device=device),
        fxp_frac_bits=int(inputs.fxp_frac_bits),
    )
    Y = RSSArithU64(
        lo=_le_bytes_to_u64_matrix(_b64d(inputs.Y_lo_b64), d=d, device=device),
        hi=_le_bytes_to_u64_matrix(_b64d(inputs.Y_hi_b64), d=d, device=device),
        fxp_frac_bits=int(inputs.fxp_frac_bits),
    )
    W = RSSArithU64(
        lo=_le_bytes_to_u64_matrix(_b64d(inputs.W_lo_b64), d=d, device=device),
        hi=_le_bytes_to_u64_matrix(_b64d(inputs.W_hi_b64), d=d, device=device),
        fxp_frac_bits=int(inputs.fxp_frac_bits),
    )

    # Trace-level: record safe commitments to secret inputs (hashes only; never print shares).
    _log_json(
        level=str(log_level),
        want="info",
        party_id=int(party_id),
        event="party_start",
        device=str(device),
        require_cuda=bool(require_cuda),
        torch_version=str(torch.__version__),
        torch_cuda_available=bool(torch.cuda.is_available()),
        d=int(d),
        steps=int(steps),
        sks_t_checks=int(sks_t_checks),
        sks_sample_log2=int(sks_sample_log2),
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
        fxp_frac_bits=int(inputs.fxp_frac_bits),
    )

    outp = Path(str(out_dir)).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)

    # Private artifacts (DO NOT SHARE): checkpoint shares for mid-run recovery.
    # These are never printed in logs; only hashes are logged.
    priv_dir = outp / "private"
    ckpt_path = priv_dir / "checkpoints_W.jsonl"
    if bool(checkpoint_enable):
        priv_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(priv_dir), 0o700)
        except Exception:
            pass

    # Transcript live dump (append-only) so partial progress is verifiable even if the process crashes.
    # This is SAFE to share (contains hashes/headers only, not payloads).
    transcript_live = outp / "transcript_v1_live.jsonl"
    last_flushed = 0

    relay = RelayClient(base_url=str(relay_base_url), group_id=str(relay_group_id), token=relay_token, timeout_s=240.0, tls_ca_pem=tls_ca_pem)
    party = Party(party_id=int(party_id), job_id32=bytes(job_id32), sid=bytes(sid), relay=relay)

    def flush_transcript_live() -> None:
        nonlocal last_flushed
        ts = party.transcript
        if ts is None:
            return
        leaves = ts.leaves()
        if len(leaves) <= last_flushed:
            return
        with open(transcript_live, "a", encoding="utf-8") as f:
            for lf in leaves[last_flushed:]:
                f.write(json.dumps({"body_b64": _b64e(lf.body_bytes)}, sort_keys=True, separators=(",", ":")) + "\n")
        last_flushed = len(leaves)

    # SKS epoch setup (commit/reveal) for deterministic, transcripted checks.
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
    flush_transcript_live()

    for step_i_local in range(int(steps)):
        step_i = int(step_offset) + int(step_i_local)
        t_step0 = time.monotonic()
        _log_json(level=str(log_level), want="info", party_id=int(party_id), event="step_start", step=int(step_i))
        # "Training" step: P = X@W, E = P - Y, G = X^T@E (with X=I in the default demo config), W <- W - G
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
            fxp_frac_bits=int(inputs.fxp_frac_bits),
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
        _log_json(
            level=str(log_level),
            want="info",
            party_id=int(party_id),
            event="sks1_done",
            step=int(step_i),
            ok=bool(ok1),
            t_s=round(time.monotonic() - t0, 6),
        )
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
            fxp_frac_bits=int(inputs.fxp_frac_bits),
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
        _log_json(
            level=str(log_level),
            want="info",
            party_id=int(party_id),
            event="sks2_done",
            step=int(step_i),
            ok=bool(ok2),
            t_s=round(time.monotonic() - t0, 6),
        )
        if ok2 is not True:
            raise RuntimeError("SKS check failed for X^T@E")

        W = W.sub(res2.Z)
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
        flush_transcript_live()

        if bool(checkpoint_enable) and ((int(step_i_local) + 1) % int(checkpoint_every) == 0):
            # Append checkpoint shares (private) for step boundary recovery.
            priv_dir.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": _now_iso_utc(),
                "party_id": int(party_id),
                "epoch": int(epoch),
                "step": int(step_i),
                "d": int(d),
                "fxp_frac_bits": int(inputs.fxp_frac_bits),
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

    # OPEN final weights (model output) and compute a stable result_hash32.
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
    flush_transcript_live()

    leaves = party.transcript.leaves() if party.transcript is not None else []
    transcript_jsonl = "\n".join(
        json.dumps({"body_b64": _b64e(lf.body_bytes)}, sort_keys=True, separators=(",", ":")) for lf in leaves
    ) + "\n"
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
                "result_hash32_hex": _hex32(result_hash32),
                "device": str(device),
                "sks_t_checks": int(sks_t_checks),
                "sks_sample_log2": int(sks_sample_log2),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return result_hash32, transcript_bytes


