"""Demo page server: runs the Python sim at 60 frames/s and streams frames to the browser over a WebSocket.

    backend/.venv/bin/python -m uvicorn mario.server:app --port 8010      (from the repo root)

Modes: play (keyboard), bot (scripted), model (asks a policy worker over HTTP, see mario/train/serve.py), replay.
Replays need no GPU and no model: a replay is a level + seed + list of moves, re-run by the same sim.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import time
import urllib.request

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mario.game.bot import bot_action
from mario.game.level import get_level
from mario.game.sim import FRAMES_PER_DECISION, MOVES, N_MOVES
from mario.game.textstate import render_text
from mario.game.vec import EVAL_SEED_BASE, make_sim

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")          # fetched from context3: runs/<run>/{metrics.jsonl,replays/<ckpt>/*.json}
BUNDLED = os.path.join(HERE, "replays")    # replays committed with the repo (backup demo)
POLICY_URL = os.environ.get("MARIO_POLICY_URL", "http://127.0.0.1:8011")
FPS = 60

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "web")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "web", "index.html"))


FEATURED_RUNS = ("main", "scratch", "r128", "fe")   # replay picker: bundled demo replays + these runs


def _replay_files():
    out = []
    run_files = []
    for r in FEATURED_RUNS:
        run_files += sorted(glob.glob(os.path.join(RUNS, r, "replays", "**", "*.json"), recursive=True))
    for p in sorted(glob.glob(os.path.join(BUNDLED, "**", "*.json"), recursive=True)) + run_files:
        if not p.endswith("summary.json"):
            out.append(os.path.relpath(p, HERE))
    return out


@app.get("/api/replays")
def replays():
    items = []
    for rel in _replay_files():
        try:
            with open(os.path.join(HERE, rel)) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        items.append({"id": rel, "label": r.get("label") or rel, "checkpoint": r.get("checkpoint", ""),
                      "won": r.get("won"), "dist": r.get("dist"), "level": r.get("level"), "seed": r.get("seed")})
    return items


@app.get("/api/runs")
def runs():
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "metrics.jsonl"))):
        name = os.path.basename(os.path.dirname(p))
        rows = []
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        meta = {}
        mp = os.path.join(os.path.dirname(p), "run.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
        out.append({"name": name, "meta": meta, "metrics": rows})
    return out


@app.get("/api/policy")
def policy_status():
    try:
        with urllib.request.urlopen(POLICY_URL + "/status", timeout=2) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001 - shown on the page
        return JSONResponse({"ok": False, "error": str(e)[:200]})


def _ask_policy(text: str, checkpoint: str) -> list[float]:
    req = urllib.request.Request(POLICY_URL + "/act", data=json.dumps({"text": text, "checkpoint": checkpoint}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["probs"]


class Game:
    def __init__(self, cfg: dict):
        self.mode = cfg.get("mode", "play")
        self.replay = None
        self.checkpoint = cfg.get("checkpoint", "")
        self.text_version = 2
        if self.mode == "model":
            try:
                with urllib.request.urlopen(POLICY_URL + "/status", timeout=3) as r:
                    self.text_version = json.loads(r.read()).get("text_versions", {}).get(self.checkpoint, 2)
            except Exception:  # noqa: BLE001 - the decide() call reports an unreachable worker
                pass
        if self.mode == "replay":
            with open(os.path.join(HERE, cfg["replay"])) as f:
                self.replay = json.load(f)
            level, seed, diff = self.replay["level"], self.replay["seed"], self.replay.get("difficulty", 1)
        else:
            level = cfg.get("level", "level1")
            seed = int(cfg.get("seed", EVAL_SEED_BASE + int(time.time()) % 1000))
            diff = int(cfg.get("difficulty", 1))
        self.level, self.seed, self.diff = level, seed, diff
        self.sim = make_sim(level, seed, diff)
        if self.mode == "play":
            self.sim.stuck_frames = None
        self.keys = {"left": False, "right": False, "jump": False}
        self.prev_jump = False
        self.k = 0               # frames into the current decision
        self.action = 5
        self.jumped = False
        self.decisions = 0
        self.post = 0
        self.ret = 0.0

    def decide(self) -> dict:
        s = self.sim
        txt = render_text(s, self.text_version)
        if self.mode == "bot":
            a = bot_action(s)
            probs = [1.0 if i == a else 0.0 for i in range(N_MOVES)]
        elif self.mode == "replay":
            acts = self.replay["actions"]
            a = int(acts[self.decisions]) if self.decisions < len(acts) else 5
            pl = self.replay.get("probs")
            probs = pl[self.decisions] if pl and self.decisions < len(pl) else [1.0 if i == a else 0.0 for i in range(N_MOVES)]
        else:  # model
            probs = _ask_policy(txt, self.checkpoint)
            a = max(range(N_MOVES), key=lambda i: probs[i])
        self.action, self.jumped, self.k = a, False, 0
        self.decisions += 1
        return {"txt": txt, "p": [round(p, 3) for p in probs], "a": a}

    def frame(self, extra: dict | None = None) -> dict:
        s = self.sim
        if self.mode == "play":
            mx = (1 if self.keys["right"] else 0) - (1 if self.keys["left"] else 0)
            j = 0
            if self.keys["jump"] and not self.prev_jump:
                j = 2
            elif not self.keys["jump"] and self.prev_jump:
                j = 3
            self.prev_jump = self.keys["jump"]
            s.step_frame(mx, j)
        else:
            _, mx, j = MOVES[self.action]
            self.jumped |= s.step_frame(mx, 0 if self.jumped else j)
            self.k += 1
        snap = s.snapshot()
        s.events = []
        if extra:
            snap.update(extra)
        return snap


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    game: Game | None = None
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()

    async def reader():
        try:
            while True:
                inbox.put_nowait(json.loads(await sock.receive_text()))
        except (WebSocketDisconnect, RuntimeError):
            inbox.put_nowait({"cmd": "_closed"})

    rtask = asyncio.create_task(reader())
    speed = 1.0
    try:
        next_t = loop.time()
        while True:
            while not inbox.empty():
                msg = inbox.get_nowait()
                cmd = msg.get("cmd")
                if cmd == "_closed":
                    return
                if cmd == "start":
                    try:
                        game = Game(msg)
                    except Exception as e:  # noqa: BLE001
                        await sock.send_json({"t": "error", "msg": str(e)[:300]})
                        game = None
                        continue
                    lv = get_level(game.level).to_json()
                    await sock.send_json({"t": "level", "mode": game.mode, "seed": game.seed, "level": lv,
                                          "label": (game.replay or {}).get("label", "")})
                    next_t = loop.time()
                elif cmd == "keys" and game:
                    game.keys.update({k: bool(msg.get(k)) for k in ("left", "right", "jump")})
                elif cmd == "speed":
                    speed = max(0.25, min(4.0, float(msg.get("value", 1))))
                elif cmd == "stop":
                    game = None
            if game is None:
                await asyncio.sleep(0.05)
                continue
            extra = None
            if game.mode != "play" and not game.sim.done and (game.k == 0 or game.k >= FRAMES_PER_DECISION):
                try:
                    extra = await asyncio.to_thread(game.decide) if game.mode == "model" else game.decide()
                except Exception as e:  # noqa: BLE001
                    await sock.send_json({"t": "error", "msg": f"policy worker not reachable: {e}"[:300]})
                    game = None
                    continue
                next_t = max(next_t, loop.time() - 0.05)  # a slow model pauses the game instead of skipping frames
            await sock.send_json({"t": "f", **game.frame(extra)})
            if game.sim.done:
                game.post += 1
                if game.post > 150:
                    s = game.sim
                    await sock.send_json({"t": "end", "won": int(s.won), "reason": s.reason,
                                          "dist": round(s.max_x / 16, 1), "coins": s.coins, "stomps": s.stomps,
                                          "time": round(s.frame / 60, 1)})
                    game = None
                    continue
            next_t += 1.0 / (FPS * speed)
            delay = next_t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
    finally:
        rtask.cancel()
