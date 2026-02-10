from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b2265a18869da013

from uvcc_party.cbor_det import cbor_dumps_det_v1
from uvcc_party.policy_canon import PolicyPartyV1, PolicyV1, policy_cbor_bytes_v1, policy_hash32_v1


def test_cbor_det_orders_map_keys_by_utf8_bytes() -> None:
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert cbor_dumps_det_v1(a) == cbor_dumps_det_v1(b)


def test_policy_cbor_and_hash_are_deterministic() -> None:
    p = PolicyV1(
        uvcc_version="1.0",
        backend="CRYPTO_CC_3PC",
        sid=b"sid-demo",
        flags_u32=0,
        job_id_u64=0,
        sgir_hash32=b"\x11" * 32,
        runtime_hash32=b"\x22" * 32,
        fss_dir_hash32=b"\x33" * 32,
        preproc_hash32=b"\x44" * 32,
        parties=(
            PolicyPartyV1(party_id=2, addr20=b"\x22" * 20, domain="p2"),
            PolicyPartyV1(party_id=0, addr20=b"\x00" * 20, domain="p0"),
            PolicyPartyV1(party_id=1, addr20=b"\x11" * 20, domain="p1"),
        ),
    )
    c1 = policy_cbor_bytes_v1(p)
    h1 = policy_hash32_v1(p)

    # Changing party tuple order must not change the canonical CBOR bytes.
    p2 = PolicyV1(
        uvcc_version="1.0",
        backend="CRYPTO_CC_3PC",
        sid=b"sid-demo",
        flags_u32=0,
        job_id_u64=0,
        sgir_hash32=b"\x11" * 32,
        runtime_hash32=b"\x22" * 32,
        fss_dir_hash32=b"\x33" * 32,
        preproc_hash32=b"\x44" * 32,
        parties=(
            PolicyPartyV1(party_id=0, addr20=b"\x00" * 20, domain="p0"),
            PolicyPartyV1(party_id=1, addr20=b"\x11" * 20, domain="p1"),
            PolicyPartyV1(party_id=2, addr20=b"\x22" * 20, domain="p2"),
        ),
    )
    c2 = policy_cbor_bytes_v1(p2)
    h2 = policy_hash32_v1(p2)
    assert c2 == c1
    assert h2 == h1


def test_cbor_rejects_float() -> None:
    try:
        cbor_dumps_det_v1(1.5)  # type: ignore[arg-type]
        raise AssertionError("expected float to be rejected")
    except TypeError:
        pass


