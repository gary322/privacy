from __future__ import annotations

from uvcc_party.fss_block import FSSRecordV1, build_fss_block_v1
from uvcc_party.fss_plan import FSSExecTaskV1, FSSPlanPrimeV1, fss_tasks_bytes_v1
from uvcc_party.fss_transcript import MSG_FSSBLOCK_COMMIT, MSG_FSSEVAL_COMMIT, transcript_record_fssblock_commit_v1, transcript_record_fsseval_commit_v1
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient


def test_fssblock_and_fsseval_commits_recorded() -> None:
    relay = RelayClient(base_url="http://127.0.0.1:1", group_id="noop", token=None)
    p = Party(party_id=0, job_id32=b"\x55" * 32, sid=b"sid-fss-transcript", relay=relay)

    rec = FSSRecordV1(
        fss_id=123,
        sgir_op_id=7,
        lane=0xFFFF,
        fss_kind=2,
        domain_bits=8,
        range_bits=1,
        share_mode=1,
        edge_mode=1,
        edge_id=0,
        flags=1,
        key_bytes=0,
        key_offset=0,
        key_hash_hi=0,
        key_hash_lo=0,
    )
    block = build_fss_block_v1(epoch=0, step=1, records=[rec], key_blobs=[b"keyblob"])
    h_fssblock = transcript_record_fssblock_commit_v1(party=p, epoch=0, step=1, round=0, block_bytes=block)
    assert h_fssblock is not None and len(h_fssblock) == 32

    plan = FSSPlanPrimeV1(task_count=1, key_arena_bytes=16, in_arena_bytes=16, out_arena_bytes=16, scratch_bytes=0)
    task = FSSExecTaskV1(
        fss_id=123,
        sgir_op_id=7,
        lane=0xFFFF,
        kind=2,
        domain_bits=8,
        range_bits=1,
        in_type=1,
        out_type=1,
        flags=0,
        lanes=1,
        in_offset=0,
        in_stride=8,
        out_offset=0,
        out_stride=4,
        key_offset=0,
        key_bytes=7,
    )
    plan_b = plan.to_bytes()
    tasks_b = fss_tasks_bytes_v1([task])
    h_fsseval = transcript_record_fsseval_commit_v1(
        party=p,
        epoch=0,
        step=1,
        round=0,
        plan_prime_bytes=plan_b,
        tasks_bytes=tasks_b,
        fssblock_hash32=h_fssblock,
    )
    assert h_fsseval is not None and len(h_fsseval) == 32

    kinds = [int(l.prefix.msg_kind) for l in p.transcript.leaves()]
    assert MSG_FSSBLOCK_COMMIT in kinds
    assert MSG_FSSEVAL_COMMIT in kinds


