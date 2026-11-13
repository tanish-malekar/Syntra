"""Policy worker for "Model plays live": loads the base model once and switches LoRA checkpoints on request.

    mario/.venv/bin/python -m mario.train.serve --runs mario/runs --port 8011         (on this Mac: MPS or CPU)

The demo server (mario/server.py) POSTs {"text", "checkpoint"} to /act and gets back {"probs": [6 floats]}.
The same worker can run on context3 behind an SSH tunnel (mario/remote.py serve) if the Mac is too slow.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from mario.train.policy import DEFAULT_MODEL, Policy


class Worker:
    def __init__(self, runs: str, model: str):
        self.runs = runs
        self.pol = Policy(model, trainable=False).eval()   # fresh (untrained) adapter named "default"
        self.loaded: dict[str, str] = {}
        self.lock = threading.Lock()

    def checkpoints(self) -> list[str]:
        out = ["untrained"]
        for p in sorted(glob.glob(os.path.join(self.runs, "*", "ckpt", "*", "adapter_config.json"))):
            d = os.path.dirname(p)
            out.append(os.path.relpath(d, self.runs))
        return out

    def text_version(self, ck: str) -> int:
        from mario.game.textstate import TEXT_VERSION
        if not ck or ck == "untrained":
            return TEXT_VERSION
        p = os.path.join(self.runs, ck, "meta.json")
        return json.load(open(p)).get("text_version", 1) if os.path.exists(p) else 1

    def _select(self, ck: str):
        m = self.pol.model
        if not ck or ck == "untrained":
            return False
        if ck not in self.loaded:
            name = "a%d" % len(self.loaded)
            m.load_adapter(os.path.join(self.runs, ck), adapter_name=name)
            self.loaded[ck] = name
        m.set_adapter(self.loaded[ck])
        return True

    @torch.no_grad()
    def act(self, text: str, ck: str) -> list[float]:
        with self.lock:
            if self._select(ck):
                logits, _ = self.pol.act_texts([text])
            else:
                with self.pol.model.disable_adapter():
                    logits, _ = self.pol.act_texts([text])
            return torch.softmax(logits[0], -1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8011)
    a = ap.parse_args()
    w = Worker(a.runs, a.model)
    print(f"policy worker on 127.0.0.1:{a.port} device={w.pol.device_name}", flush=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/status"):
                self._send({"ok": True, "device": w.pol.device_name, "model": a.model, "checkpoints": w.checkpoints(),
                        "text_versions": {c: w.text_version(c) for c in w.checkpoints()}})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if not self.path.startswith("/act"):
                return self._send({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            t = time.time()
            try:
                probs = w.act(body["text"], body.get("checkpoint", ""))
            except Exception as e:  # noqa: BLE001
                return self._send({"error": str(e)[:300]}, 500)
            self._send({"probs": probs, "ms": round((time.time() - t) * 1000, 1)})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
