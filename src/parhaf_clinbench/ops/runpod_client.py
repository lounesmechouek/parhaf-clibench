"""Minimal HTTP client for RunPod API interactions with retries."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class RunpodClient:
    """RunPod API client with bounded retry/backoff behavior."""

    api_base: str
    api_key: str
    timeout_seconds: int = 20
    max_retries: int = 3
    backoff_seconds: float = 1.5

    def _headers(self) -> dict[str, str]:
        """Return authenticated JSON headers for RunPod requests."""

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _response_excerpt(response: requests.Response) -> str:
        """Return a bounded string excerpt from an HTTP response body."""

        try:
            payload = response.json()
            rendered = json.dumps(payload, ensure_ascii=False)
        except Exception:
            rendered = response.text.strip()
        if not rendered:
            return "<empty>"
        if len(rendered) > 800:
            return rendered[:800] + "..."
        return rendered

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one HTTP request with retry/backoff and error normalization."""

        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise RuntimeError(f"Réponse API inattendue pour {url}: {type(body)!r}")
                return body
            except requests.HTTPError as exc:
                err_response = exc.response
                if err_response is not None and 400 <= err_response.status_code < 500 and err_response.status_code != 429:
                    # NOTE: Deterministic client error (4xx except 429): do not retry.
                    raise RuntimeError(
                        f"HTTP {err_response.status_code} {method} {url} - body: {self._response_excerpt(err_response)}"
                    ) from exc
                if err_response is None:
                    last_error = exc
                else:
                    last_error = RuntimeError(
                        f"HTTP {err_response.status_code} {method} {url} - body: {self._response_excerpt(err_response)}"
                    )
            except Exception as exc:
                last_error = exc
            if attempt >= self.max_retries:
                break
            time.sleep(self.backoff_seconds * attempt)
        raise RuntimeError(f"Échec requête RunPod {method} {url}: {last_error}") from last_error

    def launch_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a pod using `POST /pods`."""

        return self._request("POST", "/pods", payload)

    def start_pod(self, pod_id: str) -> dict[str, Any]:
        """Start an existing pod."""

        return self._request("POST", f"/pods/{pod_id}/start")

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        """Fetch the latest pod state."""

        return self._request("GET", f"/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        """Stop a pod without deleting it."""

        return self._request("POST", f"/pods/{pod_id}/stop")

    def terminate_pod(self, pod_id: str) -> dict[str, Any]:
        """Terminate and delete a pod."""

        return self._request("DELETE", f"/pods/{pod_id}")

    def wait_pod(
        self,
        *,
        pod_id: str,
        target_status: str = "RUNNING",
        timeout_seconds: int = 1800,
        poll_interval_seconds: int = 10,
    ) -> dict[str, Any]:
        """Poll a pod until a target status or timeout is reached."""

        # NOTE: EXITED can represent normal completion (exit code 0).
        terminal_failure_statuses = {"FAILED", "ERROR", "TERMINATED", "DELETED"}
        start = time.monotonic()
        while True:
            pod = self.get_pod(pod_id)
            status = str(pod.get("status", "")).upper()
            if status == target_status.upper():
                return pod
            if status in terminal_failure_statuses:
                raise RuntimeError(
                    f"Pod {pod_id} est dans un statut terminal `{status}` "
                    f"(attendu `{target_status}`)."
                )
            if status == "EXITED" and target_status.upper() != "EXITED":
                # NOTE: Benchmark completed before target window was reached.
                # NOTE: Return pod state so caller can collect produced artifacts.
                return pod
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(
                    f"Timeout en attente pod {pod_id}: statut courant={status}, attendu={target_status}"
                )
            time.sleep(poll_interval_seconds)
