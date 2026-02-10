from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


class PrimeAPIError(RuntimeError):
    pass


PRIME_API_BASE_DEFAULT = "https://api.primeintellect.ai/api/v1"


def _prime_api_base() -> str:
    return str(os.environ.get("PRIME_API_BASE", PRIME_API_BASE_DEFAULT)).rstrip("/")


def _prime_headers(api_key: str) -> Dict[str, str]:
    token = str(api_key or "").strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token:
        raise PrimeAPIError("prime api_key is empty")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def parse_prime_ssh_connection(value: Any) -> Tuple[str, str, int]:
    """
    Parse Prime's sshConnection into (user, host, port).
    Example payload: "ubuntu@135.23.125.123 -p 22"
    """
    raw = value
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, str) or not raw.strip():
        raise PrimeAPIError("sshConnection missing")
    text = raw.strip()
    parts = text.split()
    if not parts:
        raise PrimeAPIError("sshConnection missing")
    user_host = parts[0]
    user, host = user_host.split("@", 1) if "@" in user_host else ("ubuntu", user_host)
    user = str(user).strip()
    host = str(host).strip()
    # Some providers transiently return placeholders like "ubuntu@" before host assignment.
    # Treat these as not-ready so callers (e.g., wait_active) can continue polling.
    if not user:
        raise PrimeAPIError("sshConnection user missing")
    if not host:
        raise PrimeAPIError("sshConnection host missing")
    port = 22
    if "-p" in parts:
        idx = parts.index("-p")
        if idx + 1 < len(parts):
            try:
                port = int(str(parts[idx + 1]))
            except Exception:
                port = 22
    return str(user), str(host), int(port)


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        t = x.strip()
        if not t:
            return None
        try:
            return int(t, 10)
        except Exception:
            try:
                return int(float(t))
            except Exception:
                return None
    return None


def _extract_list_payload(payload: Any) -> List[Dict[str, Any]]:
    """
    Best-effort extraction for Prime list-like responses across endpoints.
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        raise PrimeAPIError("unexpected response shape (expected dict/list)")
    for k in ("data", "item", "items", "results", "result"):
        v = payload.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            for kk in ("data", "item", "items", "results", "result"):
                vv = v.get(kk)
                if isinstance(vv, list):
                    return [x for x in vv if isinstance(x, dict)]
    # Some APIs nest under {"item": {"item": [...]}} etc; we keep it strict after a few layers.
    raise PrimeAPIError(f"could not find list payload keys in response: {sorted(payload.keys())}")


@dataclass(frozen=True)
class PrimeGPUAvailV1:
    cloud_id: str
    gpu_type: str
    provider: Optional[str]
    region: Optional[str]
    # Provider routing field used for pod creation on many Prime providers.
    # This is NOT always equal to `region` (often a human-readable grouping).
    data_center_id: Optional[str]
    available: Optional[int]
    gpu_count: Optional[int]
    socket: Optional[str]
    images: Optional[List[str]]
    raw: Dict[str, Any]


def _parse_gpu_avail_row(row: Dict[str, Any]) -> Optional[PrimeGPUAvailV1]:
    cloud_id = str(row.get("cloudId") or row.get("cloud_id") or row.get("cloudID") or "").strip()
    gpu_type = str(row.get("gpuType") or row.get("gpu_type") or row.get("gpu") or row.get("gpuName") or "").strip()
    if not cloud_id or not gpu_type:
        return None

    provider = row.get("provider") or row.get("providerType") or row.get("provider_type")
    provider_s = str(provider).strip() if provider is not None else None
    if provider_s == "":
        provider_s = None

    region = row.get("region") or row.get("location") or row.get("geo")
    region_s = str(region).strip() if region is not None else None
    if region_s == "":
        region_s = None

    # Many providers expose a specific "data center id" routing token distinct from region.
    dc = (
        row.get("dataCenterId")
        or row.get("data_center_id")
        or row.get("data_center")
        or row.get("dataCenter")
        or row.get("datacenterId")
        or row.get("datacenter")
    )
    dc_s = str(dc).strip() if dc is not None else None
    if dc_s == "":
        dc_s = None

    # Many availability APIs return a numeric availability/count field, but key varies.
    avail = (
        _as_int(row.get("available"))
        or _as_int(row.get("availableCount"))
        or _as_int(row.get("available_count"))
        or _as_int(row.get("count"))
        or _as_int(row.get("numAvailable"))
        or _as_int(row.get("nAvailable"))
    )
    gpu_count = _as_int(row.get("gpuCount")) or _as_int(row.get("gpu_count"))
    socket = str(row.get("socket") or "").strip() or None
    images = row.get("images")
    images_list: Optional[List[str]] = None
    if isinstance(images, list):
        images_list = [str(x) for x in images if str(x).strip()]
    return PrimeGPUAvailV1(
        cloud_id=cloud_id,
        gpu_type=gpu_type,
        provider=provider_s,
        region=region_s,
        data_center_id=dc_s,
        available=avail,
        gpu_count=gpu_count,
        socket=socket,
        images=images_list,
        raw=dict(row),
    )


@dataclass(frozen=True)
class PrimePodSpecV1:
    cloud_id: str
    gpu_type: str
    gpu_count: int
    socket: str
    image: str
    name: str
    provider_type: str = "runpod"
    data_center_id: Optional[str] = None
    max_price: Optional[float] = None

    def to_create_body(self) -> Dict[str, Any]:
        pod: Dict[str, Any] = {
            "cloudId": str(self.cloud_id),
            "gpuType": str(self.gpu_type),
            "socket": str(self.socket),
            "gpuCount": int(self.gpu_count),
            "name": str(self.name),
            "image": str(self.image),
        }
        dc = str(self.data_center_id).strip() if self.data_center_id is not None else ""
        if self.max_price is not None:
            pod["maxPrice"] = float(self.max_price)
        provider: Dict[str, Any] = {"type": str(self.provider_type)}
        if dc:
            # Provider routing fields vary across Prime providers.
            # - datacrunch expects pod.dataCenterId
            # - hyperstack expects pod.dataCenterId (despite the error message mentioning data_center_id)
            # - crusoecloud also expects pod.dataCenterId (error messages still mention data_center_id)
            # - lambdalabs also expects pod.dataCenterId (error messages still mention data_center_id)
            # - nebius also expects pod.dataCenterId (error messages may be generic)
            # - vultr also expects pod.dataCenterId (error messages still mention data_center_id)
            # - dc_wildebeest also expects pod.dataCenterId (error messages may be generic)
            # - oblivus expects pod.dataCenterId (error messages mention data_center_id)
            # - massedcompute expects pod.dataCenterId (error messages mention data_center_id)
            # - others often accept provider.data_center_id
            ptype = str(self.provider_type or "").strip().lower()
            if ptype in ("datacrunch", "hyperstack", "crusoecloud", "lambdalabs", "nebius", "vultr", "dc_wildebeest", "oblivus", "massedcompute"):
                pod["dataCenterId"] = dc
            else:
                provider["data_center_id"] = dc
        return {"pod": pod, "provider": provider}


@dataclass(frozen=True)
class PrimePodV1:
    pod_id: str
    status_row: Dict[str, Any]
    ssh_user: str
    ssh_host: str
    ssh_port: int


@dataclass
class PrimeClientV1:
    api_key: str
    api_base: str = PRIME_API_BASE_DEFAULT
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        base = str(self.api_base or "").strip() or PRIME_API_BASE_DEFAULT
        self.api_base = base.rstrip("/")

    def create_pod(self, spec: PrimePodSpecV1) -> str:
        url = f"{self.api_base}/pods/"
        resp = requests.post(url, headers=_prime_headers(self.api_key), json=spec.to_create_body(), timeout=float(self.timeout_s))
        if resp.status_code >= 400:
            raise PrimeAPIError(f"prime create pod failed ({resp.status_code}): {resp.text}")
        try:
            data = resp.json()
        except Exception as e:
            raise PrimeAPIError(f"prime create pod returned non-json: {e}") from e
        if not isinstance(data, dict):
            raise PrimeAPIError("prime create pod returned non-object json")
        pod_id = data.get("id") or data.get("podId") or data.get("pod_id")
        if not pod_id:
            raise PrimeAPIError(f"prime create pod response missing pod id: {data}")
        return str(pod_id)

    def delete_pod(self, pod_id: str) -> None:
        """
        Best-effort pod deletion. Endpoint shape varies across Prime API versions, so we try a small set.
        """
        headers = _prime_headers(self.api_key)
        last_err = ""
        for path in (f"/pods/{pod_id}/", f"/pods/{pod_id}"):
            url = f"{self.api_base}{path}"
            try:
                resp = requests.delete(url, headers=headers, timeout=max(5.0, float(self.timeout_s) / 2.0))
            except Exception as e:
                last_err = str(e)
                continue
            if int(resp.status_code) in (200, 202, 204):
                return
            last_err = f"{resp.status_code}: {resp.text}"
        raise PrimeAPIError(f"prime delete pod failed: {last_err}")

    def pod_status(self, pod_id: str) -> Dict[str, Any]:
        url = f"{self.api_base}/pods/status/"
        resp = requests.get(
            url,
            headers=_prime_headers(self.api_key),
            params={"pod_ids": str(pod_id)},
            timeout=max(5.0, float(self.timeout_s) / 2.0),
        )
        if resp.status_code >= 400:
            raise PrimeAPIError(f"prime pod status failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        if not isinstance(data, dict):
            raise PrimeAPIError("prime pod status returned non-object json")
        return data

    def pod_get(self, pod_id: str) -> Dict[str, Any]:
        """
        Fetch a single pod's full details (includes primePortMapping on hosted providers).
        """
        url = f"{self.api_base}/pods/{str(pod_id).strip()}/"
        resp = requests.get(url, headers=_prime_headers(self.api_key), timeout=max(5.0, float(self.timeout_s) / 2.0))
        if resp.status_code >= 400:
            raise PrimeAPIError(f"prime pod get failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        if not isinstance(data, dict):
            raise PrimeAPIError("prime pod get returned non-object json")
        return data

    def gpu_availability(self, *, regions: Optional[Iterable[str]] = None, cloud_ids: Optional[Iterable[str]] = None) -> List[PrimeGPUAvailV1]:
        """
        Fetch GPU availability list.

        Docs: GET /api/v1/availability  (returns a dict keyed by gpuType; values are lists of offers)
        """
        url = f"{self.api_base}/availability"
        params: Dict[str, Any] = {}
        # Prime's /availability endpoint currently returns the full table. We filter client-side
        # to keep behavior stable across minor API changes.

        resp = requests.get(url, headers=_prime_headers(self.api_key), params=params, timeout=max(5.0, float(self.timeout_s)))
        if resp.status_code >= 400:
            raise PrimeAPIError(f"prime gpu availability failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        # /availability returns {gpuType: [offer,...], ...}. Older/alternate endpoints may return lists.
        rows: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            # If it looks like keyed-by-gpuType, flatten.
            for v in data.values():
                if isinstance(v, list):
                    rows.extend([x for x in v if isinstance(x, dict)])
        else:
            rows = _extract_list_payload(data)

        out_all: List[PrimeGPUAvailV1] = []
        for r in rows:
            parsed = _parse_gpu_avail_row(r)
            if parsed is not None:
                out_all.append(parsed)
        if not out_all:
            raise PrimeAPIError("prime gpu availability returned zero usable rows (missing cloudId/gpuType)")

        rs_set = {str(r).strip() for r in regions} if regions is not None else set()
        cs_set = {str(c).strip() for c in cloud_ids} if cloud_ids is not None else set()
        out: List[PrimeGPUAvailV1] = []
        for it in out_all:
            if rs_set and (it.region is None or str(it.region) not in rs_set):
                continue
            if cs_set and str(it.cloud_id) not in cs_set:
                continue
            out.append(it)
        if not out:
            # Fall back to the full list if filters eliminated everything.
            out = out_all
        return out

    def pick_cloud_and_gpu(
        self,
        *,
        nodes: int,
        gpu_count_per_node: int,
        prefer_gpu_types: Optional[Iterable[str]] = None,
        prefer_regions: Optional[Iterable[str]] = None,
    ) -> Tuple[str, str]:
        """
        Deterministically choose a (cloud_id, gpu_type) pair to use for provisioning.

        Strategy:
        - Pull /availability/gpu
        - Filter by prefer_regions if provided
        - Sort by preference list (consumer-first) then by higher availability if present
        """
        need = max(1, int(nodes)) * max(1, int(gpu_count_per_node))

        prefs = [str(x).strip() for x in (prefer_gpu_types or []) if str(x).strip()]
        pref_index = {p: i for i, p in enumerate(prefs)}

        av = self.gpu_availability(regions=list(prefer_regions) if prefer_regions is not None else None)

        # Filter out entries that explicitly report insufficient availability.
        candidates: List[PrimeGPUAvailV1] = []
        for it in av:
            if it.available is not None and int(it.available) > 0 and int(it.available) < need:
                continue
            candidates.append(it)
        if not candidates:
            candidates = av

        def score(it: PrimeGPUAvailV1) -> Tuple[int, int, str, str]:
            # Lower is better.
            pidx = pref_index.get(it.gpu_type, 1_000_000)
            # Prefer higher availability when present (negated for ascending sort).
            avail = int(it.available) if it.available is not None else 0
            return (int(pidx), -avail, it.cloud_id, it.gpu_type)

        best = sorted(candidates, key=score)[0]
        return best.cloud_id, best.gpu_type

    def pick_offer_v1(
        self,
        *,
        nodes: int,
        gpu_count_per_node: int,
        provider_type: Optional[str],
        socket: Optional[str],
        prefer_gpu_types: Optional[Iterable[str]] = None,
        prefer_regions: Optional[Iterable[str]] = None,
    ) -> PrimeGPUAvailV1:
        """
        Pick a concrete availability offer (cloudId+gpuType+provider+images) suitable for provisioning.
        """
        offers = self.candidate_offers_v1(
            nodes=nodes,
            gpu_count_per_node=gpu_count_per_node,
            provider_type=provider_type,
            socket=socket,
            prefer_gpu_types=prefer_gpu_types,
            prefer_regions=prefer_regions,
        )
        return offers[0]

    def candidate_offers_v1(
        self,
        *,
        nodes: int,
        gpu_count_per_node: int,
        provider_type: Optional[str],
        socket: Optional[str],
        prefer_gpu_types: Optional[Iterable[str]] = None,
        prefer_regions: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> List[PrimeGPUAvailV1]:
        """
        Return a sorted list of candidate offers to try provisioning with.
        """
        want_provider = str(provider_type).strip() if provider_type is not None else ""
        want_socket = str(socket).strip() if socket is not None else ""
        # "auto" means "do not filter"; most availability rows use concrete sockets like PCIe/SXM*.
        if want_socket.lower() == "auto":
            want_socket = ""
        if want_provider.lower() == "auto":
            want_provider = ""

        prefs = [str(x).strip() for x in (prefer_gpu_types or []) if str(x).strip()]
        pref_index = {p: i for i, p in enumerate(prefs)}

        av = self.gpu_availability(regions=list(prefer_regions) if prefer_regions is not None else None)
        candidates: List[PrimeGPUAvailV1] = []
        for it in av:
            if want_provider and (it.provider is None or str(it.provider).strip().lower() != want_provider.lower()):
                continue
            if want_socket and (it.socket is None or str(it.socket).strip().lower() != want_socket.lower()):
                continue
            if it.gpu_count is not None and int(it.gpu_count) != int(gpu_count_per_node):
                continue
            candidates.append(it)
        if not candidates:
            # If provider/socket filtering is too strict, fall back to raw list (but still keep gpu_count filter).
            for it in av:
                if it.gpu_count is not None and int(it.gpu_count) != int(gpu_count_per_node):
                    continue
                candidates.append(it)
        if not candidates:
            raise PrimeAPIError("no Prime availability offers match requested constraints")

        def score(it: PrimeGPUAvailV1) -> Tuple[int, int, str, str]:
            pidx = pref_index.get(it.gpu_type, 1_000_000)
            avail = int(it.available) if it.available is not None else 0
            return (int(pidx), -avail, it.cloud_id, it.gpu_type)

        ordered = sorted(candidates, key=score)
        if limit is not None:
            ordered = ordered[: max(1, int(limit))]
        return ordered

    def wait_active(self, pod_id: str, *, timeout_s: int = 1800, poll_s: float = 10.0) -> PrimePodV1:
        deadline = time.time() + max(10, int(timeout_s))
        last_state = ""
        while time.time() < deadline:
            try:
                payload = self.pod_status(pod_id)
            except Exception:
                # Transient API/network errors do happen (read timeouts, intermittent 5xx).
                # Treat them as "not ready yet" and continue polling until the overall timeout.
                time.sleep(max(2.0, float(poll_s)))
                continue
            items = payload.get("data") if isinstance(payload, dict) else None
            row = items[0] if isinstance(items, list) and items else {}
            if not isinstance(row, dict):
                row = {}
            state = str(row.get("status") or "")
            install_state = str(row.get("installationStatus") or "")
            if state != last_state:
                last_state = state
            if state.upper() == "ACTIVE":
                # Some providers briefly report ACTIVE before sshConnection is populated.
                # Treat missing sshConnection as "not ready" and continue polling until the overall timeout.
                # Provider-specific override: some providers (notably crusoecloud) can return an sshConnection
                # that is missing host/port details or otherwise unreliable. In those cases, prefer deriving
                # SSH connectivity from the full pod details + primePortMapping.
                prov = str(row.get("providerType") or row.get("provider") or "").strip().lower()
                if prov == "crusoecloud":
                    try:
                        full = self.pod_get(str(pod_id))
                        full_row = full.get("data") if isinstance(full, dict) and isinstance(full.get("data"), dict) else full
                        ip2 = str(full_row.get("ip") or full_row.get("podIp") or full_row.get("pod_ip") or row.get("ip") or "").strip()
                        pm = full_row.get("primePortMapping") or full_row.get("portMappings") or []
                        ssh_port_guess = None
                        if isinstance(pm, list):
                            for m in pm:
                                if not isinstance(m, dict):
                                    continue
                                used_by = str(m.get("usedBy") or "").strip().upper()
                                desc = str(m.get("description") or "").strip().upper()
                                if used_by == "SSH" or "SSH" in desc:
                                    ssh_port_guess = _as_int(m.get("external")) or _as_int(m.get("internal"))
                                    break
                        if ssh_port_guess is None:
                            ssh_port_guess = 22
                        if ip2:
                            return PrimePodV1(
                                pod_id=str(pod_id),
                                status_row=full_row if isinstance(full_row, dict) else row,
                                ssh_user="root",
                                ssh_host=str(ip2),
                                ssh_port=int(ssh_port_guess),
                            )
                    except Exception:
                        time.sleep(max(2.0, float(poll_s)))
                        continue
                try:
                    ssh_user, ssh_host, ssh_port = parse_prime_ssh_connection(row.get("sshConnection"))
                except PrimeAPIError:
                    # Provider-specific fallback: some hosted providers (notably datacrunch)
                    # can return ACTIVE with sshConnection still unset for extended periods, and sometimes even
                    # return ip=null on /pods/status. In those cases, we can often derive the SSH host/port from
                    # the full pod details endpoint.
                    prov = str(row.get("providerType") or row.get("provider") or "").strip().lower()
                    ip = str(row.get("ip") or row.get("podIp") or row.get("pod_ip") or "").strip()
                    if prov in ("datacrunch", "hyperstack", "crusoecloud", "lambdalabs", "vultr", "nebius", "dc_wildebeest"):
                        try:
                            full = self.pod_get(str(pod_id))
                            full_row = full.get("data") if isinstance(full, dict) and isinstance(full.get("data"), dict) else full
                            ip2 = str(full_row.get("ip") or full_row.get("podIp") or full_row.get("pod_ip") or ip).strip()
                            pm = full_row.get("primePortMapping") or full_row.get("portMappings") or []
                            ssh_port_guess = None
                            if isinstance(pm, list):
                                for m in pm:
                                    if not isinstance(m, dict):
                                        continue
                                    used_by = str(m.get("usedBy") or "").strip().upper()
                                    desc = str(m.get("description") or "").strip().upper()
                                    if used_by == "SSH" or "SSH" in desc:
                                        # Prefer the external port (what clients use).
                                        ssh_port_guess = _as_int(m.get("external")) or _as_int(m.get("internal"))
                                        break
                            if ssh_port_guess is None:
                                ssh_port_guess = 22
                            if ip2:
                                return PrimePodV1(
                                    pod_id=str(pod_id),
                                    status_row=full_row if isinstance(full_row, dict) else row,
                                    ssh_user="root",
                                    ssh_host=str(ip2),
                                    ssh_port=int(ssh_port_guess),
                                )
                        except Exception:
                            # Fall back to polling; transient API inconsistencies happen during provisioning.
                            time.sleep(max(2.0, float(poll_s)))
                            continue
                    time.sleep(max(2.0, float(poll_s)))
                    continue
                return PrimePodV1(
                    pod_id=str(pod_id),
                    status_row=row,
                    ssh_user=ssh_user,
                    ssh_host=ssh_host,
                    ssh_port=int(ssh_port),
                )
            if state.upper() in {"ERROR", "TERMINATED"}:
                raise PrimeAPIError(f"prime pod entered terminal state: {state} (install={install_state})")
            time.sleep(max(2.0, float(poll_s)))
        raise PrimeAPIError("timed out waiting for prime pod to become ACTIVE")


def prime_client_from_env_v1(*, api_key_env: str = "UVCC_PRIME_API_KEY") -> PrimeClientV1:
    api_key = str(os.environ.get(api_key_env, "")).strip()
    if not api_key:
        raise PrimeAPIError(f"missing env {api_key_env}")
    return PrimeClientV1(api_key=api_key, api_base=_prime_api_base())


