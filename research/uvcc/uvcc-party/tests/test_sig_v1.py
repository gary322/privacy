from __future__ import annotations

from uvcc_party.sig import (
    secp256k1_eth_address_from_pubkey,
    secp256k1_pubkey_from_privkey,
    secp256k1_recover_pubkey_from_hash,
    secp256k1_sign_hash,
    secp256k1_verify_hash,
)


def test_sig_roundtrip_recover() -> None:
    priv = b"\x01" * 32
    msg = b"\xAB" * 32
    pub = secp256k1_pubkey_from_privkey(priv)
    sig = secp256k1_sign_hash(priv, msg)
    assert len(pub) == 64
    assert len(sig) == 65
    assert secp256k1_verify_hash(pub, msg, sig)
    rec = secp256k1_recover_pubkey_from_hash(msg, sig)
    assert rec == pub
    addr = secp256k1_eth_address_from_pubkey(pub)
    assert len(addr) == 20


