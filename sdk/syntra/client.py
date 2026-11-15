"""HTTP client for Syntra training sessions. Only token ids and numbers cross the network; your reward code stays local."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

RETRY_STATUSES = (502, 503, 504)


class SyntraError(RuntimeError):
    """Base class for errors from the Syntra API."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class BadRequest(SyntraError):
    """The request was invalid (shapes, lengths, values). Nothing was changed on the server."""


class NotSignedIn(SyntraError):
    """Missing, wrong or revoked API key."""


class SessionNotRunning(SyntraError):
    """The session is still starting, or it was closed (by you, or automatically after being idle)."""


@dataclass
class Sample:
    """One sampled completion. `tokens` = prompt + completion; `logprobs` are per completion token under the policy that
    sampled it (use them as `old_logprobs` for PPO)."""
    prompt_index: int
    tokens: list[int]
    prompt_len: int
    completion: list[int]
    logprobs: list[float]
    text: str
    finished: bool
    extra: dict = field(default_factory=dict)

    def as_sequence(self) -> dict:
        return {"tokens": self.tokens, "prompt_len": self.prompt_len}


def _error(r: httpx.Response) -> SyntraError:
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    detail = detail if isinstance(detail, str) else str(detail)
    cls = {401: NotSignedIn, 409: SessionNotRunning, 422: BadRequest, 404: BadRequest}.get(r.status_code, SyntraError)
    return cls(f"HTTP {r.status_code}: {detail}", r.status_code)


class Client:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 900.0, retries: int = 3):
        self.api_key = api_key or os.environ.get("SYNTRA_API_KEY", "")
        self.base_url = (base_url or os.environ.get("SYNTRA_BASE_URL", "http://localhost:8000")).rstrip("/")
        if not self.api_key:
            raise NotSignedIn("No API key: pass api_key= or set SYNTRA_API_KEY (create one on the website)")
        self.retries = retries
        self._http = httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- low level -------------------------------------------------------------------------------
    def request(self, method: str, path: str, json: Any = None, idempotency_key: str | None = None) -> Any:
        """Retries network errors and 502/503/504 with backoff. Calls that change state carry an idempotency key, so a retry
        of a call that did reach the server returns the first result instead of applying it twice."""
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                r = self._http.request(method, path, json=json)
            except httpx.TransportError as exc:
                if attempt == self.retries or (method != "GET" and idempotency_key is None):
                    raise SyntraError(f"network error: {exc}") from exc
            else:
                if r.status_code < 400:
                    return r.json() if r.content else None
                if r.status_code not in RETRY_STATUSES or attempt == self.retries or (method != "GET" and idempotency_key is None):
                    raise _error(r)
            time.sleep(delay)
            delay = min(delay * 2, 10)
        raise SyntraError("unreachable")

    # ---- account ---------------------------------------------------------------------------------
    def base_models(self) -> list[dict]:
        return self.request("GET", "/base-models")

    def models(self) -> list[dict]:
        return self.request("GET", "/models")

    def _base_model_id(self, base_model: str | int) -> int:
        if isinstance(base_model, int):
            return base_model
        for m in self.base_models():
            if base_model in (m["name"], m.get("hf_id")):
                return m["id"]
        raise BadRequest(f"unknown base model {base_model!r}; see client.base_models()")

    # ---- sessions --------------------------------------------------------------------------------
    def training_session(self, base_model: str | int, *, init_model: int | None = None, rank: int = 8, lora_alpha: float = 16,
                         weight_bits: int = 16, gpus: int | None = None, trusted_only: bool = False, max_seq_len: int = 1024,
                         name: str = "", wait: bool = True, timeout: float = 1800.0) -> "TrainingSession":
        """Lease GPUs and start a LoRA (weight_bits=16) or QLoRA (weight_bits=4) training session.
        init_model: continue from one of your fine-tuned models (it is also the reference policy for KL and DPO)."""
        body = {"base_model_id": self._base_model_id(base_model), "init_model_id": init_model, "rank": rank,
                "lora_alpha": lora_alpha, "weight_bits": weight_bits, "requested_gpus": gpus, "trusted_only": trusted_only,
                "max_seq_len": max_seq_len, "name": name}
        info = self.request("POST", "/sessions", json=body, idempotency_key=None)
        session = TrainingSession(self, info["id"])
        if wait:
            try:
                session.wait_ready(timeout)
            except BaseException:  # timed out / interrupted: do not leave a queued session holding a place (or GPUs)
                try:
                    session.close()
                except SyntraError:
                    pass
                raise
        return session

    def session(self, session_id: int) -> "TrainingSession":
        return TrainingSession(self, session_id)


class TrainingSession:
    def __init__(self, client: Client, session_id: int):
        self.client, self.id = client, session_id

    def __repr__(self):
        return f"TrainingSession(id={self.id})"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except SyntraError:
            pass

    def info(self) -> dict:
        return self.client.request("GET", f"/sessions/{self.id}")

    def wait_ready(self, timeout: float = 1800.0, poll: float = 2.0) -> dict:
        """Waits while the session queues for GPUs and the workers load the model."""
        end = time.monotonic() + timeout
        while True:
            info = self.info()
            if info["ready"]:
                return info
            if info["status"] in ("stopped", "failed"):
                raise SessionNotRunning(f"session {self.id} is {info['status']}: {info['job']['result'].get('error') or ''}")
            if time.monotonic() > end:
                raise SyntraError(f"session {self.id} not ready after {timeout:.0f} s (status {info['status']})")
            time.sleep(poll)

    def call(self, op: str, **args) -> dict:
        key = uuid.uuid4().hex  # one key per logical call, reused by retries
        return self.client.request("POST", f"/sessions/{self.id}/call", json={"op": op, "args": args, "idempotency_key": key},
                                   idempotency_key=key)["data"]

    # ---- primitives ------------------------------------------------------------------------------
    def tokenize(self, texts: Iterable[str], add_special_tokens: bool = True) -> list[list[int]]:
        return self.call("tokenize", texts=list(texts), add_special_tokens=add_special_tokens)["tokens"]

    def detokenize(self, tokens: Iterable[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return self.call("detokenize", tokens=list(tokens), skip_special_tokens=skip_special_tokens)["texts"]

    def sample(self, prompts: Iterable, n: int = 1, max_new_tokens: int = 32, temperature: float = 1.0, top_p: float = 1.0,
               top_k: int = 0, seed: int | None = None, stop_at_eos: bool = True) -> list[Sample]:
        """n completions for each prompt (text or token ids). Results are grouped by prompt: n consecutive samples each."""
        data = self.call("sample", prompts=list(prompts), n=n, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, top_k=top_k, seed=seed, stop_at_eos=stop_at_eos)
        keys = Sample.__dataclass_fields__.keys() - {"extra"}
        return [Sample(**{k: s[k] for k in keys}, extra={k: v for k, v in s.items() if k not in keys}) for s in data["samples"]]

    def logprobs(self, sequences: Iterable, reference: bool = False) -> list[list[float]]:
        """Per completion token log-probs; reference=True uses the session's starting adapter (the reference policy)."""
        seqs = [s.as_sequence() if isinstance(s, Sample) else s for s in sequences]
        return self.call("logprobs", sequences=seqs, reference=reference)["logprobs"]

    def forward_backward(self, batch: list[dict], loss: str = "cross_entropy", **loss_args) -> dict:
        """Accumulate gradients. loss: cross_entropy | dpo | ppo | importance_sampling. Rows:
        cross_entropy {tokens, prompt_len}; ppo / importance_sampling {tokens, prompt_len, advantage (or advantages per token),
        old_logprobs}; dpo {chosen: {tokens, prompt_len}, rejected: {...}, optional ref_chosen / ref_rejected}."""
        return self.call("forward_backward", loss=loss, batch=batch, **loss_args)

    def optim_step(self, lr: float = 1e-5, weight_decay: float = 0.0, max_grad_norm: float = 1.0) -> dict:
        return self.call("optim_step", lr=lr, weight_decay=weight_decay, max_grad_norm=max_grad_norm)

    def state(self) -> dict:
        return self.call("state")

    def save(self, name: str) -> int:
        """Store the adapter as one of your models (deployable, comparable). Returns the model id."""
        # not retried automatically: a retry after the server already saved would store a second copy
        return self.client.request("POST", f"/sessions/{self.id}/save", json={"name": name})["model_id"]

    def close(self, save_as: str | None = None) -> int | None:
        """Release the GPUs (billing stops). save_as: save the adapter first under this name."""
        return self.client.request("POST", f"/sessions/{self.id}/close", json={"save_as": save_as})["model_id"]
