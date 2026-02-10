from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .relay_client import RelayClient
from .netframe import NetFrameV1, parse_netframe_v1, payload_hash32_v1, relay_msg_id_v1
from .transcript import SegmentDescV1, TranscriptStoreV1


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


# Relay/transport timing defaults.
#
# NOTE: Distributed GPU kernels can take minutes to compile on first use, and heterogeneous providers/GPU types
# can create multi-minute skew between parties reaching the same protocol round. The relay TTL and recv timeouts
# must exceed that skew, or parties will deadlock/time out even when nothing is actually wrong.
DEFAULT_RELAY_TTL_S = int(os.environ.get("UVCC_RELAY_TTL_S", "3600"))
DEFAULT_NET_TIMEOUT_S = float(os.environ.get("UVCC_NET_TIMEOUT_S", "900"))


@dataclass
class Party:
    party_id: int
    job_id32: bytes
    sid: bytes
    relay: RelayClient
    transcript: Optional[TranscriptStoreV1] = None

    def __post_init__(self) -> None:
        if int(self.party_id) not in (0, 1, 2):
            raise ValueError("party_id must be 0..2")
        if len(self.job_id32) != 32:
            raise ValueError("job_id32 must be 32 bytes")
        if not isinstance(self.sid, (bytes, bytearray)):
            raise TypeError("sid must be bytes")
        self._sid_hash32 = sha256(bytes(self.sid))
        self._inbox: Dict[str, Tuple[int, bytes]] = {}  # msg_id -> (sender, payload)
        if self.transcript is None:
            self.transcript = TranscriptStoreV1(job_id32=self.job_id32)

    def sid_hash32(self) -> bytes:
        return self._sid_hash32

    def next_party(self) -> int:
        return (int(self.party_id) + 1) % 3

    def prev_party(self) -> int:
        return (int(self.party_id) + 2) % 3

    def send(self, *, msg_id: str, receiver: int, payload: bytes, ttl_s: Optional[int] = None) -> None:
        self.relay.enqueue(msg_id=msg_id, sender=int(self.party_id), receiver=int(receiver), payload=payload, ttl_s=ttl_s)

    def recv_expect(self, *, msg_id: str, sender: int, timeout_s: float = DEFAULT_NET_TIMEOUT_S) -> bytes:
        # Fast path: already stashed
        v = self._inbox.pop(msg_id, None)
        if v is not None:
            got_sender, payload = v
            if int(got_sender) != int(sender):
                raise RuntimeError("stashed sender mismatch")
            return payload

        deadline = time.time() + float(timeout_s)
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"timeout waiting for msg_id={msg_id}")
            m = self.relay.poll(receiver=int(self.party_id), deadline_s=min(deadline, time.time() + 2.0))
            if m is None:
                continue
            mid = str(m["msg_id"])
            msender = int(m["sender"])
            lease_token = str(m["lease_token"])
            payload = base64.b64decode(str(m["payload_b64"]).encode("ascii"))
            # Ack immediately to avoid lease stalls.
            self.relay.ack(receiver=int(self.party_id), msg_id=mid, lease_token=lease_token)
            if mid == msg_id:
                if msender != int(sender):
                    raise RuntimeError(f"unexpected sender: got={msender} want={sender}")
                return payload
            # stash for later
            self._inbox[mid] = (msender, payload)

    def send_netframe(self, *, frame: NetFrameV1, ttl_s: Optional[int] = None, relay_domain: bytes = b"uvcc.netframe.relay.v1") -> str:
        """Send a NetFrame over the relay and record a SEND transcript leaf (dir=0)."""
        hdr = frame.header
        msg_id = relay_msg_id_v1(
            domain=relay_domain,
            sid_hash32=self.sid_hash32(),
            job_id32=self.job_id32,
            epoch=int(hdr.epoch),
            step=int(hdr.step),
            round=int(hdr.round),
            msg_kind=int(hdr.msg_kind),
            sender=int(hdr.sender),
            receiver=int(hdr.receiver),
            seq_no=int(hdr.seq_no),
            frame_no=int(hdr.frame_no),
        )
        raw = frame.to_bytes()
        self.send(msg_id=msg_id, receiver=int(hdr.receiver), payload=raw, ttl_s=ttl_s)

        if self.transcript is not None:
            segs = tuple(
                SegmentDescV1(
                    seg_kind=int(s.seg_kind),
                    object_id=int(s.object_id),
                    sub_id=int(s.sub_id),
                    dtype=int(s.dtype),
                    offset=int(s.offset),
                    length=int(s.length),
                    fxp_frac_bits=int(s.fxp_frac_bits),
                )
                for s in frame.segments
            )
            self.transcript.record_frame(
                epoch=int(hdr.epoch),
                step=int(hdr.step),
                round=int(hdr.round),
                msg_kind=int(hdr.msg_kind),
                sender=int(hdr.sender),
                receiver=int(hdr.receiver),
                dir=0,
                seq_no=int(hdr.seq_no),
                payload_bytes=int(hdr.payload_bytes),
                payload_hash32=payload_hash32_v1(frame),
                header_hash32=bytes(hdr.header_hash32),
                segments=segs,
            )
        return msg_id

    def recv_netframe_expect(
        self,
        *,
        epoch: int,
        step: int,
        round: int,
        msg_kind: int,
        sender: int,
        receiver: int,
        seq_no: int,
        frame_no: int = 0,
        timeout_s: float = DEFAULT_NET_TIMEOUT_S,
        relay_domain: bytes = b"uvcc.netframe.relay.v1",
    ) -> NetFrameV1:
        """Receive a specific NetFrame (by deterministic relay msg_id) and record a RECV transcript leaf (dir=1)."""
        msg_id = relay_msg_id_v1(
            domain=relay_domain,
            sid_hash32=self.sid_hash32(),
            job_id32=self.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(msg_kind),
            sender=int(sender),
            receiver=int(receiver),
            seq_no=int(seq_no),
            frame_no=int(frame_no),
        )
        raw = self.recv_expect(msg_id=msg_id, sender=int(sender), timeout_s=float(timeout_s))
        frame = parse_netframe_v1(raw)
        hdr = frame.header
        if int(hdr.epoch) != int(epoch) or int(hdr.step) != int(step) or int(hdr.round) != int(round):
            raise ValueError("received frame epoch/step/round mismatch")
        if int(hdr.msg_kind) != int(msg_kind):
            raise ValueError("received frame msg_kind mismatch")
        if int(hdr.sender) != int(sender) or int(hdr.receiver) != int(receiver):
            raise ValueError("received frame sender/receiver mismatch")
        if int(hdr.seq_no) != int(seq_no) or int(hdr.frame_no) != int(frame_no):
            raise ValueError("received frame seq/frame_no mismatch")

        if self.transcript is not None:
            segs = tuple(
                SegmentDescV1(
                    seg_kind=int(s.seg_kind),
                    object_id=int(s.object_id),
                    sub_id=int(s.sub_id),
                    dtype=int(s.dtype),
                    offset=int(s.offset),
                    length=int(s.length),
                    fxp_frac_bits=int(s.fxp_frac_bits),
                )
                for s in frame.segments
            )
            self.transcript.record_frame(
                epoch=int(hdr.epoch),
                step=int(hdr.step),
                round=int(hdr.round),
                msg_kind=int(hdr.msg_kind),
                sender=int(hdr.sender),
                receiver=int(hdr.receiver),
                dir=1,
                seq_no=int(hdr.seq_no),
                payload_bytes=int(hdr.payload_bytes),
                payload_hash32=payload_hash32_v1(frame),
                header_hash32=bytes(hdr.header_hash32),
                segments=segs,
            )
        return frame


