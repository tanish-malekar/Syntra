"""A DiLoCo replica: holds the whole model on one GPU, trains `inner_steps` on its data shard, uploads theta_start - theta_end.

    python -m syntra_agent.diloco_worker --server URL --job J --lease L --model PATH_OR_HF_ID --device mps|cuda|cpu
    (token in SYNTRA_AGENT_TOKEN)

Started by the agent like a BloomBee worker (same "Running a server" / "Started" log lines, so the agent's online and
crash detection work unchanged). Needs torch + transformers + safetensors + httpx (the BloomBee environment has them).
The inner AdamW state is kept across rounds; on (re)joining, the replica starts from the current global weights.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time

import httpx


def log(msg: str) -> None:
    print(time.strftime("%b %d %H:%M:%S") + f".000 [INFO] {msg}", flush=True)


class Platform:
    def __init__(self, server: str, token: str, job: int, lease: int):
        self.http = httpx.Client(base_url=server, headers={"Authorization": f"Bearer {token}"}, timeout=120)
        self.job, self.lease = job, lease

    def get(self, path: str, **params) -> httpx.Response:
        delay = 1.0
        while True:
            try:
                r = self.http.get(path, params={"lease_id": self.lease, **params})
                if r.status_code < 500:
                    return r
            except httpx.TransportError as exc:
                log(f"platform not reachable ({exc}); retrying")
            time.sleep(delay)
            delay = min(delay * 2, 15)

    def state(self) -> dict | None:
        r = self.get(f"/agent/diloco/{self.job}/state")
        if r.status_code == 404:
            log("this replica's lease is no longer active: exiting")
            sys.exit(0)
        return r.json() if r.status_code == 200 else None

    def upload(self, round_no: int, body: bytes, tokens: int, loss: float) -> str:
        sha = hashlib.sha256(body).hexdigest()
        for attempt in range(5):
            try:
                r = self.http.put(f"/agent/diloco/{self.job}/delta/{round_no}", content=body,
                                  params={"lease_id": self.lease, "sha256": sha, "tokens": tokens, "loss": loss})
            except httpx.TransportError as exc:
                log(f"upload failed ({exc}); retrying")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return "ok"
            if r.status_code == 409:
                return "closed"  # the round closed without us; our work for it is dropped
            if r.status_code == 404:
                sys.exit(0)
            log(f"upload refused: {r.status_code} {r.text[:200]}")
            time.sleep(2 ** attempt)
        return "failed"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--job", type=int, required=True)
    p.add_argument("--lease", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--work-dir", default=".")
    a = p.parse_args(argv)
    token = os.environ.get("SYNTRA_AGENT_TOKEN", "")
    log(f"Running a server on ['/diloco/job{a.job}/p2p/diloco{a.lease}']")  # the agent's peer-id line

    import torch
    from safetensors.torch import load, save
    from transformers import AutoModelForCausalLM, AutoTokenizer

    plat = Platform(a.server, token, a.job, a.lease)
    st = None
    while st is None:
        st = plat.state()
        if st is None:
            time.sleep(2)
    device = a.device if a.device != "mps" or torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(device).train()
    params = dict(model.named_parameters())

    # this replica's shard of the data: every replicas-th chunk starting at its slot
    rows = [json.loads(l) for l in plat.get(f"/agent/diloco/{a.job}/data").text.splitlines() if l.strip()]
    ids = []
    for r in rows:
        ids.extend(tok(r["text"])["input_ids"] + ([tok.eos_token_id] if tok.eos_token_id is not None else []))
    seq = int(st["seq_len"])
    chunks = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)] or [ids[:seq]]
    shard = chunks[st["slot"] % max(1, st["replicas"])::max(1, st["replicas"])] or chunks
    rng = random.Random(1000 + st["slot"])
    inner = torch.optim.AdamW(model.parameters(), lr=float(st["lr"]), weight_decay=0.0)
    log("Started")  # the agent reports this replica online

    done_round = -1
    while True:
        st = plat.state()
        if st is None:
            time.sleep(2)
            continue
        if st["status"] == "done":
            log("training finished")
            return 0
        r = int(st["round"])
        if st["status"] != "running" or r <= done_round:
            time.sleep(1)
            continue
        resp = plat.get(f"/agent/diloco/{a.job}/weights/{r}")
        if resp.status_code != 200:
            time.sleep(2)
            continue
        if resp.headers.get("X-Sha256") and hashlib.sha256(resp.content).hexdigest() != resp.headers["X-Sha256"]:
            log("global weights checksum mismatch; downloading again")
            continue
        global_w = load(resp.content)
        with torch.no_grad():
            for n, prm in params.items():
                prm.copy_(global_w[n].to(device))
        start = {n: global_w[n].float() for n in params}  # theta_start on the CPU
        del global_w
        losses, tokens = [], 0
        for _ in range(int(st["inner_steps"])):
            batch = torch.tensor([rng.choice(shard) for _ in range(int(st["batch_size"]))], device=device)
            out = model(input_ids=batch, labels=batch)
            if not torch.isfinite(out.loss):
                losses.append(float("nan"))
                break
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner.step()
            inner.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach()))
            tokens += batch.numel()
        loss = sum(losses) / len(losses) if losses else float("nan")
        with torch.no_grad():
            delta = {n: (start[n] - prm.detach().float().cpu()).to(torch.bfloat16).contiguous() for n, prm in params.items()}
        body = save(delta)
        verdict = plat.upload(r, body, tokens, loss)  # a NaN loss is sent as nan: the coordinator drops this update
        log(f"round {r + 1}/{st['rounds']}: loss {loss:.4f}, {tokens} tokens, upload {verdict}")
        done_round = r


if __name__ == "__main__":
    sys.exit(main())
