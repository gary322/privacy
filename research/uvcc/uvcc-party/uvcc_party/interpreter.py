from __future__ import annotations

# pyright: reportMissingImports=false

# UVCC_REQ_GROUP: uvcc_group_1957bee5c681ad25

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .op_lut import op_lut_public_v1
from .trunc import op_trunc_exact_v1, op_trunc_prob_v1, parse_trunc_pack_v1, TruncFSSKeysV1
from .open import OpenArithItemU64, open_arith_u64_round_v1
from .party import Party
from .rss import RSSArithU64
from .sgir import (
    SEC_PUB,
    SEC_SEC,
    DT_U64,
    SGIRInstructionV1,
    SGIRModuleV1,
    SGIRTypeRecordV1,
)


# SGIR opcodes (privacy_new.txt §A.10.1).
OP_ADD = 10
OP_SUB = 11
OP_TRUNC = 13
OP_LUT = 23
OP_OPEN = 60
OP_RET = 52


@dataclass(frozen=True)
class ValueHandle:
    type_id: int
    value: object  # RSSArithU64 or torch.Tensor


class SGIRInterpreterV1:
    """
    Minimal SGIR interpreter for Class A arithmetic + OPEN over RSS.

    This is intentionally strict and deterministic:
    - Only supports DT_U64 values for v1 tests.
    - Only supports OP_ADD/OP_SUB/OP_OPEN/OP_RET.
    """

    def __init__(self, *, party: Party, module: SGIRModuleV1):
        self.party = party
        self.module = module
        self._vals: Dict[int, ValueHandle] = {}
        # Optional per-op assets (used by macro-ops like OP_LUT_PUBLIC_V1).
        self._op_lut_assets: Dict[int, Tuple[torch.Tensor, bytes]] = {}
        # Optional per-op assets (used by OP_TRUNC).
        self._op_trunc_assets: Dict[int, Tuple[bytes, int, int, Optional[TruncFSSKeysV1], Optional[bytes], Optional[bytes], Optional[bytes], Optional[bytes]]] = {}

    def register_op_lut_public_v1(self, *, sgir_op_id: int, table_u64: torch.Tensor, fss_blob: bytes) -> None:
        """
        Register assets for OP_LUT_PUBLIC_V1 execution in this interpreter instance.

        Args:
          sgir_op_id: the SGIR op id (we use the instruction index in v1 tests).
          table_u64: public table as torch.int64 u64 ring elements (length 2^w).
          fss_blob: per-party OP_LUT record blob bytes for this op instance.
        """
        if not isinstance(table_u64, torch.Tensor) or table_u64.dtype != torch.int64:
            raise TypeError("table_u64 must be torch.int64 tensor")
        if not isinstance(fss_blob, (bytes, bytearray)):
            raise TypeError("fss_blob must be bytes")
        self._op_lut_assets[int(sgir_op_id) & 0xFFFFFFFF] = (table_u64, bytes(fss_blob))

    def register_op_trunc_prob_v1(self, *, sgir_op_id: int, trunc_pack_blob: bytes, f_bits: int, signedness: int) -> None:
        self._op_trunc_assets[int(sgir_op_id) & 0xFFFFFFFF] = (bytes(trunc_pack_blob), int(f_bits), int(signedness), None, None, None, None, None)

    def register_op_trunc_exact_v1(
        self,
        *,
        sgir_op_id: int,
        trunc_pack_blob: bytes,
        f_bits: int,
        signedness: int,
        fss_keys: TruncFSSKeysV1,
        gf2_triples_blob: bytes,
        b2a_carry_blob: bytes,
        b2a_ov_blob: bytes,
        edge_key32: Optional[bytes],
    ) -> None:
        self._op_trunc_assets[int(sgir_op_id) & 0xFFFFFFFF] = (
            bytes(trunc_pack_blob),
            int(f_bits),
            int(signedness),
            fss_keys,
            bytes(gf2_triples_blob),
            bytes(b2a_carry_blob),
            bytes(b2a_ov_blob),
            bytes(edge_key32) if edge_key32 is not None else None,
        )

    def set_value(self, *, value_id: int, type_id: int, value: object) -> None:
        self._vals[int(value_id)] = ValueHandle(type_id=int(type_id), value=value)

    def get_value(self, value_id: int) -> ValueHandle:
        if int(value_id) not in self._vals:
            raise KeyError(f"value_id {value_id} not set")
        return self._vals[int(value_id)]

    def _type(self, type_id: int) -> SGIRTypeRecordV1:
        return self.module.types[int(type_id)]

    def _assert_u64(self, t: SGIRTypeRecordV1) -> None:
        if int(t.dtype) != int(DT_U64):
            raise ValueError("only DT_U64 supported by v1 interpreter")

    def run_function0(
        self,
        *,
        epoch: int,
        step_base: int,
        sgir_fun_id: int = 0,
    ) -> Dict[int, object]:
        insts = self.module.iter_function_instructions(0)
        step = int(step_base)
        for inst_index, inst in enumerate(insts):
            if int(inst.opcode) == OP_RET:
                break
            if int(inst.opcode) == OP_ADD or int(inst.opcode) == OP_SUB:
                self._exec_addsub(inst, is_sub=(int(inst.opcode) == OP_SUB))
                continue
            if int(inst.opcode) == OP_TRUNC:
                sgir_op_id32 = int(inst_index) & 0xFFFFFFFF
                self._exec_op_trunc_v1(inst, epoch=int(epoch), step=step, sgir_op_id=sgir_op_id32)
                step += 1
                continue
            if int(inst.opcode) == OP_LUT:
                # v1 test convention: sgir_op_id = inst_index (u32).
                sgir_op_id32 = int(inst_index) & 0xFFFFFFFF
                self._exec_op_lut_public_v1(inst, epoch=int(epoch), step=step, sgir_op_id=sgir_op_id32)
                step += 1
                continue
            if int(inst.opcode) == OP_OPEN:
                self._exec_open_u64(inst, epoch=int(epoch), step=step, round=0, open_id=((int(sgir_fun_id) & 0xFFFFFFFF) << 32) | (int(inst_index) & 0xFFFFFFFF))
                step += 1
                continue
            raise ValueError(f"unsupported opcode {inst.opcode}")
        return {vid: vh.value for vid, vh in self._vals.items()}

    def _exec_addsub(self, inst: SGIRInstructionV1, *, is_sub: bool) -> None:
        if len(inst.dst_ids) != 1 or len(inst.src_ids) != 2:
            raise ValueError("OP_ADD/OP_SUB require dst_count=1 src_count=2")
        dst = int(inst.dst_ids[0])
        a = self.get_value(int(inst.src_ids[0]))
        b = self.get_value(int(inst.src_ids[1]))
        t_dst = self._type(a.type_id)
        self._assert_u64(t_dst)
        if int(t_dst.secrecy) == SEC_SEC:
            if not isinstance(a.value, RSSArithU64) or not isinstance(b.value, RSSArithU64):
                raise TypeError("secret add/sub expects RSSArithU64 operands")
            out = a.value.sub(b.value) if is_sub else a.value.add(b.value)
            self._vals[dst] = ValueHandle(type_id=a.type_id, value=out)
        elif int(t_dst.secrecy) == SEC_PUB:
            if not isinstance(a.value, torch.Tensor) or not isinstance(b.value, torch.Tensor):
                raise TypeError("public add/sub expects torch.Tensor operands")
            self._vals[dst] = ValueHandle(type_id=a.type_id, value=(a.value - b.value if is_sub else a.value + b.value))
        else:
            raise ValueError("bad secrecy")

    def _exec_open_u64(self, inst: SGIRInstructionV1, *, epoch: int, step: int, round: int, open_id: int) -> None:
        if len(inst.dst_ids) != 1 or len(inst.src_ids) != 1:
            raise ValueError("OP_OPEN requires dst_count=1 src_count=1")
        dst = int(inst.dst_ids[0])
        src = self.get_value(int(inst.src_ids[0]))
        t_src = self._type(src.type_id)
        self._assert_u64(t_src)
        if int(t_src.secrecy) != SEC_SEC:
            raise ValueError("OP_OPEN requires SEC src")
        if not isinstance(src.value, RSSArithU64):
            raise TypeError("OP_OPEN expects RSSArithU64 src")

        pub = open_arith_u64_round_v1(
            self.party,
            items=[OpenArithItemU64(open_id=int(open_id), sub_id=0, x=src.value)],
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            sgir_op_id=int(open_id & 0xFFFFFFFF),
        )[(int(open_id), 0)]

        # Destination type must be PUB u64 in v1.
        # We trust the module's dst type_id (validated below).
        dst_type_id = self.module.values[dst].type_id
        t_dst = self._type(dst_type_id)
        self._assert_u64(t_dst)
        if int(t_dst.secrecy) != SEC_PUB:
            raise ValueError("OP_OPEN dst must be PUB")
        self._vals[dst] = ValueHandle(type_id=dst_type_id, value=pub)

    def _exec_op_lut_public_v1(self, inst: SGIRInstructionV1, *, epoch: int, step: int, sgir_op_id: int) -> None:
        if len(inst.dst_ids) != 1 or len(inst.src_ids) != 1:
            raise ValueError("OP_LUT requires dst_count=1 src_count=1")
        dst = int(inst.dst_ids[0])
        src = self.get_value(int(inst.src_ids[0]))
        t_src = self._type(src.type_id)
        self._assert_u64(t_src)
        if int(t_src.secrecy) != SEC_SEC:
            raise ValueError("OP_LUT requires SEC src")
        if not isinstance(src.value, RSSArithU64):
            raise TypeError("OP_LUT expects RSSArithU64 src")

        assets = self._op_lut_assets.get(int(sgir_op_id) & 0xFFFFFFFF)
        if assets is None:
            raise KeyError(f"missing OP_LUT assets for sgir_op_id={sgir_op_id}")
        table_u64, fss_blob = assets

        y = op_lut_public_v1(
            self.party,
            x=src.value,
            table_u64=table_u64,
            fss_blob=fss_blob,
            epoch=int(epoch),
            step=int(step),
            sgir_op_id=int(sgir_op_id),
        )

        # Destination type must be SEC u64 in v1.
        dst_type_id = self.module.values[dst].type_id
        t_dst = self._type(dst_type_id)
        self._assert_u64(t_dst)
        if int(t_dst.secrecy) != SEC_SEC:
            raise ValueError("OP_LUT dst must be SEC")
        self._vals[dst] = ValueHandle(type_id=dst_type_id, value=y)

    def _exec_op_trunc_v1(self, inst: SGIRInstructionV1, *, epoch: int, step: int, sgir_op_id: int) -> None:
        if len(inst.dst_ids) != 1 or len(inst.src_ids) != 1:
            raise ValueError("OP_TRUNC requires dst_count=1 src_count=1")
        dst = int(inst.dst_ids[0])
        src = self.get_value(int(inst.src_ids[0]))
        t_src = self._type(src.type_id)
        self._assert_u64(t_src)
        if int(t_src.secrecy) != SEC_SEC:
            raise ValueError("OP_TRUNC requires SEC src")
        if not isinstance(src.value, RSSArithU64):
            raise TypeError("OP_TRUNC expects RSSArithU64 src")

        assets = self._op_trunc_assets.get(int(sgir_op_id) & 0xFFFFFFFF)
        if assets is None:
            raise KeyError(f"missing OP_TRUNC assets for sgir_op_id={sgir_op_id}")
        trunc_pack_blob, f_bits, signedness, fss_keys, gf2_triples_blob, b2a_carry_blob, b2a_ov_blob, edge_key32 = assets

        # Dispatch: exact if fss_keys present; else probabilistic.
        if fss_keys is None:
            y = op_trunc_prob_v1(
                self.party,
                x=src.value,
                trunc_pack_blob=trunc_pack_blob,
                epoch=int(epoch),
                step=int(step),
                sgir_op_id=int(sgir_op_id),
                f_bits=int(f_bits),
                signedness=int(signedness),
            )
        else:
            if gf2_triples_blob is None or b2a_carry_blob is None or b2a_ov_blob is None:
                raise ValueError("missing TRUNC exact artifacts")
            y = op_trunc_exact_v1(
                self.party,
                x=src.value,
                trunc_pack_blob=trunc_pack_blob,
                fss_keys=fss_keys,
                gf2_triples_blob=gf2_triples_blob,
                b2a_carry_blob=b2a_carry_blob,
                b2a_ov_blob=b2a_ov_blob,
                edge_key32=edge_key32,
                epoch=int(epoch),
                step=int(step),
                sgir_op_id=int(sgir_op_id),
                f_bits=int(f_bits),
                signedness=int(signedness),
            )

        dst_type_id = self.module.values[dst].type_id
        t_dst = self._type(dst_type_id)
        self._assert_u64(t_dst)
        if int(t_dst.secrecy) != SEC_SEC:
            raise ValueError("OP_TRUNC dst must be SEC")
        self._vals[dst] = ValueHandle(type_id=dst_type_id, value=y)


