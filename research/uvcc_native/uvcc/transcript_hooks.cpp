#include "uvcc/transcript_hooks.h"

#include "uvcc/transport.h"
#include "uvcc/sha256.h"

namespace uvcc {

TransportCallbacksV1 make_lift_transcript_callbacks_v1(TranscriptStoreV1* ts, const Sid32& sid32) {
    TransportCallbacksV1 cbs;
    if (ts == nullptr) return cbs;

    cbs.on_send_first = [ts](const FrameHdrV1& hdr, const Hash32& frame_hash32) {
        LeafV1 l;
        l.leaf_type = LEAF_LIFT_DATA_SEND;
        l.epoch_id32 = hdr.epoch_id32;
        l.stream_id64 = hdr.stream_id64;
        l.msg_id32 = hdr.msg_id32;
        l.op_id32 = hdr.op_id32;
        l.src_party = hdr.src_party;
        l.dst_party = hdr.dst_party;
        l.msg_class = hdr.msg_class;
        l.payload_kind = hdr.payload_kind;
        l.chunk_idx = hdr.chunk_idx;
        l.chunk_count = hdr.chunk_count;
        l.payload_bytes = hdr.payload_bytes;
        l.sid_hash64 = hdr.sid_hash64;
        l.frame_hash32 = frame_hash32;
        ts->record_leaf(l);
    };

    cbs.on_accept = [ts](const FrameHdrV1& hdr, const Hash32& frame_hash32) {
        LeafV1 l;
        l.leaf_type = LEAF_LIFT_DATA_ACCEPT;
        l.epoch_id32 = hdr.epoch_id32;
        l.stream_id64 = hdr.stream_id64;
        l.msg_id32 = hdr.msg_id32;
        l.op_id32 = hdr.op_id32;
        l.src_party = hdr.src_party;
        l.dst_party = hdr.dst_party;
        l.msg_class = hdr.msg_class;
        l.payload_kind = hdr.payload_kind;
        l.chunk_idx = hdr.chunk_idx;
        l.chunk_count = hdr.chunk_count;
        l.payload_bytes = hdr.payload_bytes;
        l.sid_hash64 = hdr.sid_hash64;
        l.frame_hash32 = frame_hash32;
        ts->record_leaf(l);
    };

    cbs.on_ack_send = [ts, sid32](const AckMsgV1& ack) {
        LeafV1 l;
        l.leaf_type = LEAF_LIFT_ACK_SEND;
        l.epoch_id32 = ack.epoch_id32;
        l.stream_id64 = ack.stream_id64;
        l.msg_id32 = ack.msg_id32;
        l.op_id32 = ack.op_id32;
        l.src_party = ack.src_party;
        l.dst_party = ack.dst_party;
        l.msg_class = 0x22;
        l.payload_kind = 0;
        l.chunk_idx = 0;
        l.chunk_count = 1;
        l.payload_bytes = 0;
        l.sid_hash64 = ack.sid_hash64;
        // control_hash32 binds to control message bytes deterministically.
        const auto raw = encode_ack_v1(ack, sid32);
        l.control_hash32 = sha256(raw.data(), raw.size());
        ts->record_leaf(l);
    };

    cbs.on_ack_accept = [ts, sid32](const AckMsgV1& ack) {
        LeafV1 l;
        l.leaf_type = LEAF_LIFT_ACK_ACCEPT;
        l.epoch_id32 = ack.epoch_id32;
        l.stream_id64 = ack.stream_id64;
        l.msg_id32 = ack.msg_id32;
        l.op_id32 = ack.op_id32;
        l.src_party = ack.src_party;
        l.dst_party = ack.dst_party;
        l.msg_class = 0x22;
        l.payload_kind = 0;
        l.chunk_idx = 0;
        l.chunk_count = 1;
        l.payload_bytes = 0;
        l.sid_hash64 = ack.sid_hash64;
        const auto raw = encode_ack_v1(ack, sid32);
        l.control_hash32 = sha256(raw.data(), raw.size());
        ts->record_leaf(l);
    };

    (void)sid32;
    return cbs;
}

}  // namespace uvcc


