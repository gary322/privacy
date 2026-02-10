### UVCC `how_to_run` (production 3-node GPU run: confidential + verifiable + efficient)

This document is a **comprehensive operator guide + postmortem** for running UVCC end-to-end on **3 Prime Intellect GPU nodes** with:

- **Confidentiality**: 3-party honest-majority MPC (RSS 3PC) + TLS relay + token auth
- **Verifiability**: transcript leaves → Merkle roots → proof bundle → verifier pass → on-chain receipt
- **GPU execution**: CUDA-only compute on each party GPU (kernels + Torch CUDA)

If you only want the short runbook, see `research/uvcc/uvcc-demo/PRIME_3NODE_RUNBOOK.md`. This file goes deeper and documents **every issue we hit and how we fixed it**, plus the exact checklist to repeat runs smoothly.

---

### 0) What “done” means (acceptance criteria)

A run is considered successful only if **all** of the following are true:

- **3 pods** provisioned and SSH-able
- **GPU present** on all nodes (`nvidia-smi` snapshot captured)
- Relay started on node0:
  - TLS enabled
  - CA pinned by parties
  - bearer token required (token value never printed)
- Parties run the MPC workload on CUDA:
  - `require_cuda=true` and the run completes
  - all parties produce `result.json` and `transcript_v1.jsonl`
- Orchestrator:
  - merges transcript
  - recomputes epoch roots + final root
  - creates proof bundle
  - verifier passes (`verifier_ok`)
- On-chain receipts exist:
  - `createJob`
  - `submitFinal`
- Cleanup succeeds (unless `--keep-pods true`)

---

### 1) One-command run (the “smooth path”)

#### Prereqs on your laptop / operator box

- **Python**: 3.9+ (local). The remote pods use their own Python (typically 3.10+).
- **Foundry**: `anvil`, `forge`, `cast` on PATH.
- **Prime Intellect API key**: never hardcode; use env.
- **SSH private key** that Prime pods accept (ed25519 recommended).

#### Required secrets

- **Prime API key** in env:

```bash
export UVCC_PRIME_API_KEY="(your Prime API key)"
```

- Optional (hides the noisy LibreSSL warning on macOS python builds):

```bash
export PYTHONWARNINGS='ignore:NotOpenSSLWarning'
```

#### Run

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node-real \
  --provider-type auto \
  --run-gpu-tests true \
  --keep-pods false
```

Notes:
- `--run-gpu-tests true` compiles CUDA extensions and runs parity tests on each node; it’s slower but catches GPU/kernel issues early.
- For debugging, keep pods:

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node-real \
  --keep-pods true
```

---

### 2) What to look at after a run (artifacts)

In the chosen `--out` directory you get:

- **Aggregated logs**
  - `run_full.log`: human readable; includes explicit **Privacy / Verifiability / Efficiency** sections
  - `run_full.jsonl`: structured event stream (timestamps, durations, file paths)
- **GPU + node evidence**
  - `node_p*_nvidia_smi.txt`: GPU model/VRAM/driver
  - `node_p*_bootstrap.log`: remote bootstrap output (includes torch/cuda check)
  - `node_p*_gpu_tests.log`: CUDA parity tests (if enabled)
- **Per-party runtime logs**
  - `party_p0_run.log`, `party_p1_run.log`, `party_p2_run.log`
- **Verifiability bundle**
  - `transcript_v1.jsonl`: union transcript
  - `proof_bundle.json`: roots + signatures (EIP-712)
- **Relay evidence**
  - `relay_node0.log`: best-effort fetch of relay log
- **On-chain receipts (Anvil / local chain)**
  - `onchain_createJob.log`
  - `onchain_submitFinal.log`

---

### 3) How the end-to-end pipeline works (what happens in order)

This is the operational sequence `run_prime_3node.py` performs:

1. **Start local chain** (Anvil) + deploy contracts (Foundry).
2. **Select Prime offer** (cloud/GPU/image) and provision 3 pods (or reuse existing pods).
3. **SSH-gate**: verify each pod is actually reachable over SSH (not just “ACTIVE”).
4. **Bootstrap each node**:
   - upload `uvcc_bundle.tgz`
   - `apt-get` deps
   - create venv
   - `pip install -r research/uvcc/requirements-uvcc-base.txt`
   - sanity-check torch + cuda
5. **Optional GPU conformance tests** (parity tests + extension compilation).
6. **Relay on node0**:
   - generate per-run CA + server cert
   - upload CA to all nodes
   - start relay with TLS + token-file
   - health-check from node0 (loopback) and node1 (public)
7. **Party identity**: fetch each party’s pubkey/address.
8. **Policy commit**: EIP-712 digests + signatures from all parties.
9. **On-chain createJob** (Anvil).
10. **Upload secret-shared inputs** to each party.
11. **Launch party runtime** on each node (CUDA-only).
12. **Poll for completion**: fetch `result.json`, `transcript_v1.jsonl`, `run.log`.
13. **Union transcript** and compute Merkle roots.
14. **Create proof bundle**, run verifier self-check.
15. **On-chain submitFinal** (Anvil) and confirm stored roots/hashes.
16. **Cleanup** pods unless `--keep-pods true`.

---

### 4) What we learned (deep postmortem of every issue + fix)

Below is the full issue ledger. Each entry includes: **symptom → root cause → fix → where**.

#### 4.1 Local machine / operator environment issues

- **LibreSSL + urllib3 warning**
  - **symptom**: `NotOpenSSLWarning` spam in logs
  - **root cause**: macOS python linked against LibreSSL, urllib3 v2 warns
  - **fix**: set `PYTHONWARNINGS='ignore:NotOpenSSLWarning'` (runner also tolerates it)
  - **where**: operator env, not UVCC logic

- **Manual ssh went to github.com (debugging hazard)**
  - **symptom**: `ssh root@<pod_ip>` unexpectedly connects to `github.com`
  - **root cause**: `~/.ssh/config` had a `Host github.com` block without a `Host *` header, causing global match
  - **fix**: for manual debugging use `ssh -F /dev/null ...` or fix your ssh config
  - **where**: operator machine; UVCC runner uses Paramiko (ignores local ssh config)

#### 4.2 Prime API / provisioning issues

- **GPU availability endpoint mismatch**
  - **symptom**: `prime gpu availability failed (404)`
  - **root cause**: Prime endpoint changed; `/availability` wasn’t stable for our needs
  - **fix**: probe + adopt `/availability/multi-node` parsing + normalize to offers
  - **where**: `research/uvcc/uvcc-client/uvcc_client/prime_api.py`

- **Image not supported by provider**
  - **symptom**: `Provider Runpod is not supported for image ...`
  - **root cause**: images differ per provider/cloud offering
  - **fix**: `_pick_prime_image()` picks a compatible image from offer.images
  - **where**: `uvcc_client/prime_api.py` + `uvcc-demo/run_prime_3node.py`

- **Provider-specific datacenter routing**
  - **symptom A**: `Missing field: data_center_id not specified`
  - **symptom B**: `Extra inputs are not permitted`
  - **root cause**: some providers expect datacenter routing in different request fields
  - **fix**: provider-specific mapping:
    - **datacrunch**: `pod.dataCenterId`
    - **most others**: `provider.data_center_id`
  - **where**: `uvcc_client/prime_api.py` (`PrimePodSpecV1.to_create_body`)

- **“ACTIVE but not SSHable” pods**
  - **symptom**: pod status says ACTIVE, but SSH auth fails or never connects
  - **root cause**: some offers are misconfigured / key injection broken / network delayed
  - **fix**: immediate SSH probe after ACTIVE; if probe fails, delete and retry next offer
  - **where**: `uvcc-demo/run_prime_3node.py`

- **Datacrunch: ACTIVE pods with `sshConnection=None` (or missing host/port)**
  - **symptom**: Prime reports a pod `ACTIVE`, but `sshConnection` is `None` (or an empty/placeholder string). The runner either stalls in `wait_active()` or tries to SSH to an invalid host.
  - **root cause**: Datacrunch sometimes does not populate `sshConnection` reliably/quickly; the SSH host/port must be derived from the pod’s `ip` + `primePortMapping`.
  - **fix**: In `PrimeClientV1.wait_active()`, if provider is datacrunch and `sshConnection` is missing/invalid, fetch the full pod record and derive:
    - `ssh_host`: pod `ip`
    - `ssh_port`: `primePortMapping` entry with `usedBy=SSH` (or description containing `SSH`), else fallback `22`
    - `ssh_user`: `root` (datacrunch images use `root`)
  - **where**: `research/uvcc/uvcc-client/uvcc_client/prime_api.py` (`wait_active()`)

- **sshConnection placeholder like `ubuntu@ -p 22`**
  - **symptom**: runner tries `ubuntu@:22`, effectively localhost/invalid host
  - **root cause**: provider transiently emitted an empty host
  - **fix**: `parse_prime_ssh_connection()` rejects empty host/user (treat as not-ready)
  - **where**: `uvcc_client/prime_api.py`

- **Prime instance limit reached**
  - **symptom**: `Maximum number of instances reached. Limit: 4`
  - **root cause**: leaked pods (failed runs / stuck provisioning)
  - **fix**:
    - default `--keep-pods false` + best-effort cleanup
    - add deletion helper + retries (some deletions return 500 transiently)
    - operator fallback: list `/pods/` and delete leftovers
  - **where**: runner cleanup + `PrimeClientV1.delete_pod`

#### 4.3 Relay issues (TLS + reachability + safety)

- **Token leaked via argv**
  - **symptom**: relay bearer token visible in process list / shell history
  - **fix**: relay reads token from `--token-file`; client uses `--relay-token-file`
  - **where**: `uvcc-relay/relay_server.py`, `uvcc-client/cli.py`

- **Relay reachable only on some providers**
  - **symptom**: `relay public health failed ... timed out` (node1 couldn’t reach node0 public port)
  - **root cause**: hosted NAT / hairpin / provider port mapping constraints
  - **fix**:
    - robust relay candidate selection + multiple health-check attempts
    - ability to switch provider (we stabilized on datacrunch for this run)
  - **where**: `uvcc-demo/run_prime_3node.py`

- **Accidentally attempted to run relay on SSH port**
  - **symptom**: `OSError: [Errno 98] Address already in use` when trying relay port 1234
  - **root cause**: datacrunch `primePortMapping` only advertised SSH mapping; runner mistakenly treated it as a candidate relay port
  - **fix**: explicitly reject SSH/Jupyter mappings and never select `nodes[0].ssh_port` as relay port
  - **where**: `uvcc-demo/run_prime_3node.py` relay candidate logic

- **SQLite relay concurrency crash**
  - **symptom**: `RemoteDisconnected` / relay crash under concurrent polls/enqueues
  - **root cause**: sqlite connection used concurrently without lock
  - **fix**: serialize sqlite ops with a mutex
  - **where**: `uvcc-relay/relay_server.py`

#### 4.4 CUDA / MPC runtime issues (the big ones)

- **SKS Freivalds matvec failing on CUDA**
  - **symptom**: `RuntimeError: "addmv_impl_cuda" not implemented for 'Long'`
  - **root cause**: PyTorch doesn’t implement some `int64` matvec kernels on CUDA; Freivalds used `A @ r` on int64
  - **fix**: implement ring-u64 matvec via the u64 matmul CUDA kernel (`_matmul_u64`) by treating vectors as `(n,1)`
  - **where**: `uvcc_party/sks.py` (Freivalds residual path)

- **SKS open protocol mixed CPU+CUDA tensors**
  - **symptom**: `Expected all tensors to be on the same device, cuda:0 and cpu`
  - **root cause**: `z_prev` parsed from bytes on CPU, added to CUDA `z_lo/z_hi`
  - **fix**: place `z_prev` on `z_lo.device` before arithmetic
  - **where**: `uvcc_party/sks.py` (`sks_open_commit_then_open_u64_v1`)

- **u64 ring matmul on GPU needed**
  - **symptom**: standard torch ops don’t give u64 wraparound semantics / missing kernels
  - **fix**: custom CUDA extension `matmul_u64` (int64 carrying u64 bit patterns)
  - **where**: `uvcc_party/cuda_ext/*` + tests

#### 4.5 Logging / operator visibility issues

- **`run_full.log` initially only had one line**
  - **symptom**: JSONL updated but human log stayed at “run starting”
  - **root cause**: events only wrote JSONL
  - **fix**: also mirror events into `run_full.log` as single-line JSON
  - **where**: `uvcc-demo/run_prime_3node.py` (`RunLoggerV1.event`)

#### 4.6 Long-run completion + timeouts (25 steps)

- **Runner failed with `missing result.json` on long runs**
  - **symptom**: the orchestrator errors out with “missing `result.json`” while parties are still legitimately running (often near steps 24/25).
  - **root cause**: `party_timeout_s` was too small for the job length (25 steps is ~2+ hours; each step is ~5 minutes).
  - **fix**: auto-bump an **effective** timeout based on steps (roughly `steps * 300s + 600s`, with a minimum floor), and use that for the `result.json` polling loop. The runner emits `EVENT party_timeout_effective` so the effective timeout is visible in logs.
  - **where**: `research/uvcc/uvcc-demo/run_prime_3node.py`

#### 4.7 Log + telemetry preservation (don’t lose history mid-run)

- **Remote GPU telemetry processes can die**
  - **symptom**: remote `nvidia-smi -lms` telemetry stops mid-run; gaps appear if you rely only on the remote background process.
  - **root cause**: remote background telemetry can be killed (SSH session cleanup / cgroup limits / provider policies).
  - **fix**: use a local backstop that **polls `nvidia-smi` over SSH** and appends locally:
    - `live_keep/gpu_telemetry_polled_p*.csv` (append-only, redundant)
    - the runner also produces `gpu_telemetry_p*.csv` in the final artifact bundle
  - **where**: `research/uvcc/uvcc-demo/record_live_logs_append.py` + `run_prime_3node.py`

- **Append-only recorder crashed with `Too many open files`**
  - **symptom**: `OSError: [Errno 23] Too many open files in system`
  - **root cause**: too many SFTP sessions/channels opened per tick (FD pressure on the operator machine).
  - **fix**:
    - reuse **one SFTP session per node per tick** (don’t open/close SFTP for every file)
    - close SFTP deterministically
    - make recorder state writes best-effort (log warning instead of crashing)
    - run a watchdog that records step progress and restarts the recorder if it dies
  - **where**:
    - `research/uvcc/uvcc-demo/record_live_logs_append.py`
    - `research/uvcc/uvcc-demo/watch_prime_3node_status.py`

- **One-file audit bundle for “anyone can read it”**
  - **need**: a single file that contains *all artifacts* + explains formats/events for auditors.
  - **fix**: generate `all_logs_explained.md` from the run output directory.
  - **where**: `research/uvcc/uvcc-demo/bundle_logs_explained.py`

---

### 5) How to run repeatedly without issues (checklist)

#### Before running

- **Secrets**
  - `UVCC_PRIME_API_KEY` exported
  - SSH key exists and is added to Prime
- **Local tooling**
  - `anvil`, `forge`, `cast` available
  - Python 3.9+ working
- **Prime hygiene**
  - ensure you don’t have leaked pods (instance limit); if you do:
    - run with `--keep-pods false` (default)
    - or manually delete pods via Prime UI / API

#### Provider recommendation

- For stability, prefer `--provider-type datacrunch` (worked reliably in the final run).
- If using `runpod`, relay reachability can fail depending on port mappings; the runner now retries candidate mappings, but provider differences exist.

#### During a run

- Watch:
  - `tail -f <out>/run_full.log`
  - or consume `<out>/run_full.jsonl`
- (Recommended) If you want “never lose logs even if something truncates/rotates”, run the **append-only recorder** + **watchdog**:

```bash
# Start append-only recorder (after you know the SSH hosts/ports from `EVENT prime_pods_active`)
python3 research/uvcc/uvcc-demo/record_live_logs_append.py \
  --out <out_dir> \
  --ssh-key-path ~/.ssh/uvcc_prime_runner_ed25519 \
  --interval-s 10 \
  --node 0,<host0>,<ssh_port0>,root \
  --node 1,<host1>,<ssh_port1>,root \
  --node 2,<host2>,<ssh_port2>,root

# Start watchdog (writes `live_keep/watch_status.log` and restarts the recorder if needed)
python3 research/uvcc/uvcc-demo/watch_prime_3node_status.py \
  --out <out_dir> \
  --ssh-key-path ~/.ssh/uvcc_prime_runner_ed25519 \
  --interval-s 30
```

Notes:
- For **datacrunch** runs, the SSH port is commonly **1234**, but always trust the runner logs (`prime_pods_active` / `sshConnection` parsing).
- After pods are deleted at the end of the run, the recorder will start logging SSH timeouts; that’s expected. You can stop it.
- If it hangs on provisioning:
  - check instance limit
  - try a different provider (`--provider-type datacrunch` / `--provider-type hyperstack`)
- If it fails at relay health:
  - it’s almost always provider network/port mapping; switch provider

#### After a run

- Confirm:
  - `EVENT verifier_ok`
  - `EVENT onchain_submitFinal`
  - transcript + proof bundle files exist
- If you need to reproduce the same pods for faster iteration:
  - run with `--keep-pods true`
  - then re-run with `--reuse-pod-id` (3 times)

---

### 6) The reference “known-good” successful run (example)

We have a completed run with full artifacts under:

- `research/uvcc/uvcc-demo/out-prime-3node-real-20251218T185121Z-datacrunch-reuse2/`
- `research/uvcc/uvcc-demo/out-prime-3node-datacrunch-25steps-20251218T235212Z/` (25 steps, full verifier + on-chain finalize, includes `all_logs_explained.md`)

It shows:

- GPUs: L40S on all nodes (`node_p*_nvidia_smi.txt`)
- CUDA available in bootstrap: torch prints `... True` (`node_p0_bootstrap.log`)
- Relay TLS CA hash + URLs in `run_full.log` Privacy section
- Transcript roots + proof bundle hash
- `verifier_ok`
- `onchain_createJob.log` and `onchain_submitFinal.log`

---

### 7) Troubleshooting quick table

- **stuck at provisioning**
  - check Prime instance limit; delete leaked pods
  - try a different provider
- **pods ACTIVE but no SSH**
  - wait; sshConnection can be delayed
  - runner SSH-gate should auto-retry or delete + retry
- **relay health fails**
  - provider networking; switch provider (datacrunch recommended)
  - ensure relay doesn’t bind on SSH port (runner now prevents this)
- **CUDA dtype / device errors**
  - check per-party logs (`party_p*_run.log`)
  - most are fixed now (SKS matvec + device placement)

---

### 8) Lessons learned from the full 25-step Datacrunch run (practical runbook)

This section is the “do exactly this” playbook derived from the successful long run.

#### 8.1 Always do a smoke run first (3 steps)

Before spending ~2–3 hours on a full run, do a 3-step job to validate:
- provisioning + SSH + relay reachability
- GPU health (torch CUDA)
- parties actually step and write logs
- transcripts + proof bundle + on-chain receipts land at the end

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node-smoke-3steps \
  --provider-type datacrunch \
  --job-json research/uvcc/uvcc-demo/job_smoke_3steps_trace.json \
  --party-log-level trace \
  --gpu-telemetry true \
  --gpu-telemetry-interval-s 0.5 \
  --keep-pods false
```

#### 8.2 Then do the full run (25 steps)

```bash
python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out research/uvcc/uvcc-demo/out-prime-3node-datacrunch-25steps \
  --provider-type datacrunch \
  --job-json research/uvcc/uvcc-demo/job_big_4090_trace.json \
  --party-log-level trace \
  --gpu-telemetry true \
  --gpu-telemetry-interval-s 0.5 \
  --keep-pods false
```

Timing expectations (based on the successful run):
- provisioning + bootstrap + GPU tests: ~30–40 minutes total (varies by provider/load)
- training: ~2h 10m (25 steps × ~315–320 seconds/step)

#### 8.3 Monitoring “what step are we at?”

The two simplest sources of truth:
- `run_full.log`: orchestration milestones (relay started, training launch, party_done, verifier_ok, submitFinal)
- `party_p*_run.log`: step-by-step truth

Examples:

```bash
# Orchestrator timeline
tail -n 200 <out>/run_full.log

# Latest step per party
for p in 0 1 2; do
  echo "party $p"
  grep '\"event\":\"step_done\"' <out>/party_p${p}_run.log | tail -n 1
done
```

If you are running the watchdog:

```bash
tail -f <out>/live_keep/watch_status.log
```

#### 8.4 Don’t lose logs mid-run (recommended redundancy)

If you care about “never lose logs even if something truncates/rotates”, run:
- append-only recorder: `record_live_logs_append.py` → writes `live_keep/*`
- watchdog: `watch_prime_3node_status.py` → logs progress and restarts recorder if needed

These are **local** processes (on your operator machine). They do not require modifying the pods.

#### 8.5 Completion checklist (what to confirm before declaring success)

A run is complete only if:
- `run_full.log` ends with `UVCC Prime 3-node run complete.`
- it contains:
  - `EVENT verifier_ok`
  - `EVENT onchain_submitFinal`
- these files exist in `<out>/`:
  - `transcript_v1.jsonl`
  - `proof_bundle.json`
  - `onchain_createJob.log`
  - `onchain_submitFinal.log`
  - `how_to_verify_public.md`

#### 8.6 Create a single-file “audit bundle” for sharing

This creates a single Markdown file that:
- explains log formats + high-signal events
- embeds every run artifact (logs/telemetry/transcripts/proof/on-chain receipts)
- includes SHA-256 of each embedded file for integrity

```bash
python3 research/uvcc/uvcc-demo/bundle_logs_explained.py \
  --out <out_dir> \
  --output all_logs_explained.md
```


---

### 9) Addendum (extreme detail): running on different Prime providers + “3 different providers in one go”

This addendum is a **complete runbook + postmortem** for:

- **Multi-provider** operation: running the same UVCC workload across different Prime providers (separate runs).
- **Mixed-provider** operation: running a single UVCC 3-node job where **party0/party1/party2 are each on a different provider**.

It includes:
- Every issue we hit
- Why it happened
- How we diagnosed it
- What we changed (code + run procedure) to prevent recurrence
- Copy/paste commands and checklists to repeat the “3 different providers in one go” run reliably

This section is **append-only** and does not modify earlier guidance.

---

#### 9.1 Two modes you must not confuse

##### 9.1.1 “Multi-provider” (separate runs per provider)

You run the job multiple times, one provider at a time:

- Run 3 nodes on provider A (party0/1/2 all from A)
- Run 3 nodes on provider B (party0/1/2 all from B)
- …

This is useful for:
- comparing cost/perf across providers
- validating portability of the workflow
- building confidence before mixing providers in one run

##### 9.1.2 “Mixed providers” (single run with 3 different providers)

You run a single job where each party is provisioned from a different provider:

- party0 provider != party1 provider != party2 provider

This is what you requested (“3 GPUs but all 3 from different providers on Prime Intellect”).

This is harder because:
- providers expose different **images**
- providers have different **networking** behaviors (relay reachability)
- providers have different **SSH user/port** rules
- providers require different **datacenter routing fields** in the create-pod payload

---

#### 9.2 What we implemented to make “mixed providers” safe

We added mixed-provider support in `research/uvcc/uvcc-demo/run_prime_3node.py`.

##### 9.2.1 New CLI flag: `--providers`

`--providers` is a comma-separated list of exactly 3 provider types:

- order is **party0,party1,party2**
- they must be **distinct**
- if set, it overrides `--provider-type` for provisioning

Example:

```bash
--providers hyperstack,crusoecloud,lambdalabs
```

Important:
- party0 hosts the **relay**. Choose a provider for party0 that you trust most.
- current failover mode does **not** automatically replace party0 (relay host).

##### 9.2.2 Hard requirement: all parties must run the same Prime image

The biggest mixed-provider pitfall we discovered:

- Different providers offer different image catalogs per GPU offer.
- If the runner picks different images for different parties, you risk:
  - different Python versions
  - different Torch versions
  - different CUDA minor versions
  - different compiled extension ABI expectations

We saw this concretely:
- Hyperstack + Crusoecloud offered `cuda_12_6_pytorch_2_7`
- Runpod did **not** offer that image and would fall back to `cuda_12_4_pytorch_2_4`

This “image skew” is unacceptable for a robust operator runbook.

**Fix we implemented**:
- In mixed-provider mode, the runner computes the **intersection of available images** across the three chosen providers and selects one common image.
- It logs:
  - `EVENT prime_common_image_selected {...}`
  - plus the per-party chosen offer lines:
    - `EVENT prime_offer_selected {... "party_id": N, "provider_type": "...", "image": "..." }`

Operator implication:
- You can (and should) set `--image <preferred_image>` to bias the selection.
- In practice, we found `cuda_12_4_pytorch_2_4` was widely supported across providers and was a stable choice.

##### 9.2.3 Provider recorded per node (audit + failover correctness)

We made the runner record provider information for auditing and for failover:
- `prime_pods_active` now logs `provider_types` (one per pod).
- each `RemoteNodeV1` stores `provider_type`.

##### 9.2.4 Failover semantics in mixed-provider mode

Failover is for long runs and flaky providers; it is enabled by default (`--enable-failover true`).

In mixed-provider mode, failover replacement prefers:
- the failed party’s original provider (to preserve “three different providers” semantics as much as possible)
- then falls back if necessary (capacity can be 0)

Note:
- party0 replacement is intentionally not supported in this mode (relay host).

---

#### 9.3 Provider quirks discovered while running across providers

This is the “what broke and why” list you asked for.

##### 9.3.1 Runpod: image catalog lag (forces older common image)

Symptom:
- Requested image `cuda_12_6_pytorch_2_7` exists on some providers, but runpod did not offer it on common GPU offers.

Root cause:
- Prime image availability is offer/provider-specific.
- Runpod’s offer catalog often includes:
  - `ubuntu_22_cuda_12`
  - `cuda_12_4_pytorch_2_4`
  - but not necessarily newer ones like `cuda_12_6_pytorch_2_7`

Fix:
- enforce common image intersection in mixed-provider mode
- in practice, pin `--image cuda_12_4_pytorch_2_4` for broad compatibility

##### 9.3.2 LambdaLabs: datacenter field placement mismatch (`data_center_id` error)

Symptom:
- Creating a lambdalabs pod failed with:
  - `prime create pod failed (400): {"detail":"Missing field: data_center_id not specified"}`

Root cause:
- LambdaLabs expects datacenter routing in `pod.dataCenterId`
- We were sending it as `provider.data_center_id`
- The error message is misleading: it says `data_center_id`, but the correct placement is `pod.dataCenterId` for this provider.

How we diagnosed:
- Inspect the offer raw payload: it contains `dataCenter: us-east-1` / `us-west-3`
- Reproduce a minimal create request and observe that `provider.data_center_id` does not satisfy the API

Fix:
- Treat `lambdalabs` like `datacrunch/hyperstack/crusoecloud` in `PrimePodSpecV1.to_create_body`:
  - set `pod["dataCenterId"] = ...`
- Where:
  - `research/uvcc/uvcc-client/uvcc_client/prime_api.py`

##### 9.3.3 Crusoecloud: SSH user differs (`ubuntu`, not `root`)

Symptom:
- SSH authentication failures when connecting as `root`

Root cause:
- Crusoecloud images often default to `ubuntu` as the SSH user.

Fix:
- ensure `wait_active()` / provisioning sets the correct SSH user for crusoecloud
- where:
  - `research/uvcc/uvcc-client/uvcc_client/prime_api.py`
  - `research/uvcc/uvcc-demo/run_prime_3node.py` bootstrap logic (provider-specific user override)

##### 9.3.4 Datacrunch: ACTIVE pods with `sshConnection=None`

Symptom:
- Pod shows `ACTIVE`, but `sshConnection` is missing; runner can stall or fail to connect.

Root cause:
- Datacrunch sometimes does not populate `sshConnection` reliably/quickly.

Fix:
- Derive SSH host/port from pod IP + `primePortMapping`.
- Where:
  - `research/uvcc/uvcc-client/uvcc_client/prime_api.py` (`wait_active`)

##### 9.3.5 “Provider has many offers” does not mean “provider has usable offers”

We repeatedly observed:
- `/availability` can show large counts for a provider, but the provider may have **zero offers** that satisfy:
  - `gpu_count=1`
  - `socket=PCIe`
  - `require_cuda=true` (exclude CPU_NODE)
  - a **common image intersection** with two other providers (mixed mode)

This is why mixed-provider runs must be treated as:
- a constraint satisfaction problem: (provider, gpu offer, image) must align across all 3 parties.

---

#### 9.4 What happened during the actual “3 different providers in one run” rollout

This section is the literal incident ledger you requested.

##### 9.4.1 Attempt: mixed providers with default image (dangerous skew)

We first tried mixed providers with default image selection. The runner would pick:
- hyperstack: `cuda_12_6_pytorch_2_7`
- crusoecloud: `cuda_12_6_pytorch_2_7`
- runpod: `cuda_12_4_pytorch_2_4` (because runpod didn’t have 12.6)

Why that is bad:
- Version skew across parties is a major risk for correctness and reproducibility.

What we did:
- We aborted that attempt and deleted the pods.
- Then we implemented common-image enforcement.

##### 9.4.2 Attempt: mixed-provider smoke (2 steps) with common image → success

We ran:
- providers: `hyperstack,crusoecloud,runpod`
- image: `cuda_12_4_pytorch_2_4`
- steps: 2

Result:
- completed end-to-end (`training_done`, `verifier_ok`, `onchain_submitFinal`)
- pods deleted
- generated `all_logs_explained.md`

This validated:
- cross-provider relay reachability
- per-party stepping and checkpointing
- append-only recorder + watchdog working

##### 9.4.3 Attempt: mixed-provider full (25 steps) with runpod as 3rd provider → failed early (no offers)

We attempted:
- providers: `hyperstack,crusoecloud,runpod`
- steps: 25

Failure:
- `no Prime availability offers for requested provider_type=runpod (party_id=2)`

Why:
- provider catalog + constraints + common image intersection can make “usable offers” temporarily 0.

What we did:
- switched the third provider to a different one that had usable GPU offers at that time.

##### 9.4.4 Attempt: mixed-provider full with lambdalabs as 3rd provider → provisioning failed (datacenter field)

We attempted:
- providers: `hyperstack,crusoecloud,lambdalabs`

Failure:
- `Missing field: data_center_id not specified`

Root cause and fix:
- lambdalabs expects `pod.dataCenterId` (see §9.3.2)
- patched `PrimePodSpecV1.to_create_body`
- verified by creating+deleting a single lambdalabs pod

Also:
- we manually deleted leaked pods from the failed provisioning attempt (Prime `/pods/` endpoint).

##### 9.4.5 Final: mixed-provider full (25 steps) on 3 different providers → success

We ran:
- providers: `hyperstack,crusoecloud,lambdalabs`
- image: `cuda_12_4_pytorch_2_4`
- steps: 25

Result:
- `training_done`
- `verifier_ok`
- `onchain_submitFinal`
- `UVCC Prime 3-node run complete.`
- pods deleted
- generated `all_logs_explained.md` in the run output directory

---

#### 9.5 Mixed-provider runbook (copy/paste)

##### 9.5.1 Smoke (2 steps)

```bash
export PYTHONWARNINGS='ignore:NotOpenSSLWarning'

OUT="research/uvcc/uvcc-demo/out-prime-3node-mixed-3providers-smoke-2steps-$(date -u +%Y%m%dT%H%M%SZ)"

python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out "$OUT" \
  --providers hyperstack,crusoecloud,runpod \
  --image cuda_12_4_pytorch_2_4 \
  --job-json research/uvcc/uvcc-demo/job_smoke_2steps_trace.json \
  --party-log-level trace \
  --run-gpu-tests true \
  --gpu-telemetry true \
  --gpu-telemetry-interval-s 0.5 \
  --enable-failover true \
  --failover-max-epochs 5 \
  --live-recorder-interval-s 1 \
  --keep-pods false \
  --prime-api-key-path ~/.uvcc/prime_api_key.txt

python3 research/uvcc/uvcc-demo/bundle_logs_explained.py --out "$OUT" --output all_logs_explained.md
```

If runpod is temporarily not usable, replace it with lambdalabs:

```bash
--providers hyperstack,crusoecloud,lambdalabs
```

##### 9.5.2 Full (25 steps)

```bash
OUT="research/uvcc/uvcc-demo/out-prime-3node-mixed-3providers-full-25steps-$(date -u +%Y%m%dT%H%M%SZ)"

python3 research/uvcc/uvcc-demo/run_prime_3node.py \
  --out "$OUT" \
  --providers hyperstack,crusoecloud,lambdalabs \
  --image cuda_12_4_pytorch_2_4 \
  --job-json research/uvcc/uvcc-demo/job_big_4090_trace.json \
  --party-log-level trace \
  --run-gpu-tests true \
  --gpu-telemetry true \
  --gpu-telemetry-interval-s 0.5 \
  --enable-failover true \
  --failover-max-epochs 5 \
  --live-recorder-interval-s 1 \
  --keep-pods false \
  --prime-api-key-path ~/.uvcc/prime_api_key.txt

python3 research/uvcc/uvcc-demo/bundle_logs_explained.py --out "$OUT" --output all_logs_explained.md
```

---

#### 9.6 Monitoring: “is it still going on? which step?”

The best operator UX we found is:

- **watchdog heartbeat** (step progress + recorder liveness):
  - `tail -f <out>/live_keep/watch_status.log`
  - It prints fields like:
    - `p0_start`, `p0_done`, … (step numbers + timestamps)
    - `runner_alive`, `recorder_alive`

- **per-party step truth**:
  - `tail -f <out>/live_keep/party_p0_run.log`
  - `tail -f <out>/live_keep/party_p1_run.log`
  - `tail -f <out>/live_keep/party_p2_run.log`
  - Each step ends with `checkpoint_written` (so failover can resume).

---

#### 9.7 Cleanup and “don’t leak pods” procedure (especially after aborts)

If you kill a run mid-provision or mid-training, you can leak pods and hit Prime instance limits.

We used Prime’s pods listing endpoint to clean up:
- `GET /api/v1/pods/` lists your pods
- delete each leaked pod with `DELETE /api/v1/pods/<id>/`

Operational guidance:
- If you abort a mixed-provider attempt, immediately delete pods whose names match your run prefix (e.g. `uvcc-3pc-mixed-*`).

---

#### 9.8 Extending to new providers (what to expect to break)

When adding a new provider to mixed-provider mode, expect issues in three categories:

1. **Images**: provider may not offer the image you want
   - solution: rely on `prime_common_image_selected` intersection logic

2. **Datacenter routing fields**: provider may require routing fields in a different JSON location
   - solution: inspect offer.raw for `dataCenter*` fields and adjust `PrimePodSpecV1.to_create_body`

3. **SSH**: provider may use a different default user or report incomplete sshConnection fields
   - solution: add provider-specific fallbacks in `wait_active` (derive from pod IP + primePortMapping) and ensure correct ssh_user


---

### 10) Addendum (native parallelism, Phase 6): PP/TP/DP bring-up in `uvcc_native` + the PP deadlock incident + determinism proof

This section documents the **native (C++) parallel runtime bring-up** in `research/uvcc_native/` and the key incident we hit (a PP deadlock), how we diagnosed it, the fix, and how to run/verify determinism.

#### 10.1 What this is (and what it is not)

- **What it is**:
  - A native C++ worker (`research/uvcc_native/tools/uvcc_worker_main.cpp` → `uvcc_worker`) that exercises:
    - **Cross-party OPEN** messaging (via relay HTTP transport + exactly-once semantics)
    - **Transcript leaf recording** + deterministic epoch roots
    - **Intra-party NCCL groups** for:
      - **TP** (tensor-parallel allreduce sanity)
      - **PP** (pipeline send/recv of a 1-u64 “activation/grad” token per microbatch)
      - **DP** (replica allreduce sanity within a party domain)
  - A Prime runner/orchestrator (`research/uvcc_native/run_prime_native_toy_open.py`) that:
    - attaches to existing pods or provisions pods
    - builds `uvcc_worker` on the remote machines
    - starts the relay
    - launches a whole `W[p,r,s,t]` matrix of workers
    - waits for done markers and collects logs
    - computes `roots_by_coord.json` + `audit_bundle.json` (global root)

- **What it is not (yet)**:
  - Not the full transformer SGIR runtime from `research/PARALLEL.txt`.
  - Not a full multi-step training loop with optimizer state + checkpointing.
  - Not failover / resume for the native runtime (we prioritize correctness + determinism first).

#### 10.2 Topology and mapping (what ran on GPUs)

We validated the “small-but-real” parallel topology:

- **R=2 replicas**
- **S=2 pipeline stages**
- **T=2 tensor ranks**
- **M=8 microbatches**

Each party launches `R*S*T = 8` workers, so total workers across 3 parties is `3*8 = 24`:

- **Worker coordinates**: `W[p, r, s, t]`
- **DP group**: fixed `(p, s, t)`, ranks are `r in [0..R-1]`
- **PP group**: fixed `(p, r, t)`, ranks are `s in [0..S-1]`
- **TP group**: fixed `(p, r, s)`, ranks are `t in [0..T-1]`

Each worker runs `uvcc_worker --mode phase6_step` and prints progress like:

- `phase6_tp_ready`, `phase6_pp_ready`, `phase6_dp_ready` (if DP enabled)
- `phase6_fwd mb=0..M-1`
- `phase6_bwd mb=0..M-1`
- `epoch_root=0x...`

#### 10.3 The PP deadlock incident (“stuck after phase6_pp_ready”)

**Symptom**:

- Workers reached `phase6_pp_ready` and then appeared hung.
- GPUs were mostly idle, while CPU was high (tight polling loops).
- TP ranks sometimes looked like they were “spinning”, but the real block was inside PP ordering.

**Root cause** (classic CUDA stream ordering trap):

- The initial Phase 6 bring-up **pre-posted both directions of PP receives on the same PP CUDA stream**:
  - activation receives (needed for forward on stage>0)
  - gradient receives (needed for backward on stage<S-1)
- With a single CUDA stream, operations are **issued and executed in FIFO order**.
- Posting a backward `recv` early can block later forward `send`s (which depend on matching peer `recv`s) because the stream won’t progress past the earlier queued operation.
- Result: **forward sends never run**, so the peer never gets the activations, so backward never truly becomes reachable → deadlock.

This was particularly visible when `stage0` needed to `send` activations to `stage1`, but its stream was already “occupied” by a queued gradient `recv`.

#### 10.4 The fix (deterministic lockstep PP schedule)

We rewrote the Phase 6 execution schedule inside `uvcc_worker --mode phase6_step` to be **simple and deterministic**:

- **Post only activation recvs upfront** (only for stage>0)
- Run **forward** `mb=0..M-1`:
  - enqueue OPEN (cross-party)
  - TP allreduce sanity check
  - (optional) DP allreduce sanity check
  - PP `send` activation to next stage (if not last stage)
- `cudaStreamSynchronize(pp_stream)` to ensure all forward PP ops are drained
- **Post grad recvs only after forward finishes** (only for stage<S-1)
- Run **backward** `mb=0..M-1`:
  - wait grad recv event (if stage<S-1)
  - enqueue OPEN (cross-party)
  - TP allreduce sanity check
  - (optional) DP allreduce sanity check
  - PP `send` grad to prev stage (if not first stage)

Additional correctness hardening:

- Use **separate send/recv buffers** (don’t reuse pointers across roles).
- Allocate **per-microbatch send buffers** to avoid reuse while NCCL might still be using them.
- TP/DP allreduce checks use **pinned host memory + the communicator’s CUDA stream** (avoid default-stream global sync).

Files changed:

- `research/uvcc_native/tools/uvcc_worker_main.cpp`

#### 10.5 Enabling DP in Phase 6 (and why it used to be skipped)

Historically, `run_prime_native_toy_open.py` forced `--phase6-skip-dp` because DP NCCL init can hang on some provider networks.

We changed this to be **operator-controlled**:

- Default remains “skip DP” (safer)
- Add `--phase6-enable-dp` to turn on DP NCCL groups + DP allreduce sanity checks

Files changed:

- `research/uvcc_native/run_prime_native_toy_open.py`
  - added `--phase6-enable-dp`
  - relay startup now clears the relay sqlite DB (`relay_db.sqlite` + `-wal/-shm`) to make deterministic reruns safe

#### 10.6 Determinism proof (same roots across reruns)

To test determinism, we run the exact same job twice with a fixed `sid_job_hex`:

- `--sid-job-hex 0x...` (32 bytes)
- `--phase6-enable-dp`
- same topology `R=2,S=2,T=2,M=8`

The operator/runner produces:

- `roots_by_coord.json`: per `(p,r,s,t)` `epoch_root_hex`
- `audit_bundle.json`: per-replica roots and a final `global_root_hex`

We verified:

- **0 mismatches** across all 24 coordinates
- **identical** `replica_root_hex` for replica 0 and 1
- **identical** `global_root_hex`

This is enabled by a key design choice in the transcript:

- Transcript epoch roots sort leaves by `leaf_key` (not “arrival order”), so the Merkle root is stable under benign nondeterminism in network timing.

#### 10.7 Copy/paste run commands (attach mode, DP enabled, deterministic)

You can run Phase 6 on already-running pods (attach mode) with DP enabled:

```bash
OUT="research/uvcc_native/out-prime-native-phase6-r2s2t2-m8-dp-$(date -u +%Y%m%dT%H%M%SZ)"
SID_JOB="0x$(python3 - <<'PY'
import os
print(os.urandom(32).hex())
PY
)"

python3 research/uvcc_native/run_prime_native_toy_open.py \
  --out "$OUT" \
  --attach-name-prefix "uvcc-native-toy" \
  --pods-per-party 8 \
  --gpus-per-pod 1 \
  --worker-mode phase6_step \
  --with-nccl \
  --replicas 2 --stages 2 --tp-ranks 2 \
  --microbatches 8 \
  --step-id 0 \
  --sid-job-hex "$SID_JOB" \
  --phase6-enable-dp \
  --launch-mode bg \
  --relay-lease-s 240 \
  --keep-pods true
```

Determinism rerun (same sid):

```bash
OUT2="research/uvcc_native/out-prime-native-phase6-r2s2t2-m8-dp-rerun-$(date -u +%Y%m%dT%H%M%SZ)"

python3 research/uvcc_native/run_prime_native_toy_open.py \
  --out "$OUT2" \
  --attach-name-prefix "uvcc-native-toy" \
  --pods-per-party 8 \
  --gpus-per-pod 1 \
  --worker-mode phase6_step \
  --with-nccl \
  --replicas 2 --stages 2 --tp-ranks 2 \
  --microbatches 8 \
  --step-id 0 \
  --sid-job-hex "$SID_JOB" \
  --phase6-enable-dp \
  --launch-mode bg \
  --relay-lease-s 240 \
  --keep-pods true
```

Compare roots (should be identical):

```bash
python3 - <<'PY'
import json
from pathlib import Path
A=Path("OUT_DIR_1")  # replace
B=Path("OUT_DIR_2")  # replace
ab1=json.loads((A/"audit_bundle.json").read_text())
ab2=json.loads((B/"audit_bundle.json").read_text())
assert ab1["global_root_hex"]==ab2["global_root_hex"]
r1=json.loads((A/"roots_by_coord.json").read_text())
r2=json.loads((B/"roots_by_coord.json").read_text())
key=lambda d:(d["party"],d["replica"],d["stage"],d["tp"])
m1={key(x):x["epoch_root_hex"] for x in r1}
m2={key(x):x["epoch_root_hex"] for x in r2}
assert m1==m2
print("determinism_ok", ab1["global_root_hex"])
PY
```

