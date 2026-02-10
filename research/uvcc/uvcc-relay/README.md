# `uvcc-relay` (UVCC Relay Hub)

Production-grade **HTTP(S) relay** for UVCC v1. The relay carries **opaque payload bytes** between parties and provides:
- **Idempotent enqueue** (dedup by `(group_id, msg_id)`)
- **Leased delivery** (poll grants a lease; ack finalizes)
- **Bounded storage** (TTL + garbage collection)

This component is intentionally dependency-free (stdlib only).

## Endpoints

All POST bodies are JSON, all responses are JSON.

### `GET /healthz`
Returns `{ "ok": true }`.

### `POST /enqueue`
Request:
- `group_id` (string)
- `msg_id` (string; client-chosen deterministic id)
- `sender` (int 0..2)
- `receiver` (int 0..2)
- `payload_b64` (base64 string)
- `ttl_s` (optional int)

Response:
- `{ "ok": true, "status": "enqueued" | "dedup" }`

### `POST /poll`
Request:
- `group_id` (string)
- `receiver` (int 0..2)
- `deadline_s` (optional float unix timestamp; server long-polls until then)

Response:
- `{ "ok": true, "msg": null | { msg_id, sender, receiver, lease_token, lease_until, payload_b64, payload_hash32_b64 } }`

### `POST /ack`
Request:
- `group_id` (string)
- `receiver` (int 0..2)
- `msg_id` (string)
- `lease_token` (string)

Response:
- `{ "ok": true, "status": "acked" | "already_acked" }` or `{ "ok": false, "error": ... }`

## Running

HTTP (local):

```bash
python3 research/uvcc/uvcc-relay/relay_server.py --host 127.0.0.1 --port 8080 --db /tmp/uvcc-relay.sqlite --require-token false
```

HTTPS (production):

```bash
python3 research/uvcc/uvcc-relay/relay_server.py --host 0.0.0.0 --port 443 --db /var/lib/uvcc-relay/relay.sqlite --require-token true --token "$UVCC_RELAY_TOKEN" --tls-cert /etc/uvcc/tls.crt --tls-key /etc/uvcc/tls.key
```


