// Pip's Run page: draws what the Python sim sends; never simulates anything itself.
const T = 16, VW = 416, VH = 240;
const MOVES = ["run right", "run right + long jump", "run right + short hop", "left", "jump in place", "wait"];
const cv = document.getElementById("game"), g = cv.getContext("2d");
g.imageSmoothingEnabled = false;
const $ = id => document.getElementById(id);

let ws = null, level = null, tiles = null, coins = new Set(), mode = "play";
let frame = null, cam = 0, tick = 0, particles = [], bumps = {}, label = "", ended = null, sinceDecision = 0;
let keys = { left: false, right: false, jump: false };

// ---------- background (parallax, generated once) ----------
const bg = (() => {
  const mk = (w, h, f) => { const c = document.createElement("canvas"); c.width = w; c.height = h; f(c.getContext("2d")); return c; };
  const sky = mk(VW, VH, s => {
    const gr = s.createLinearGradient(0, 0, 0, VH);
    gr.addColorStop(0, "#4f7dff"); gr.addColorStop(0.55, "#8cc8ff"); gr.addColorStop(1, "#d8f0ff");
    s.fillStyle = gr; s.fillRect(0, 0, VW, VH);
    s.fillStyle = "#fff6c8"; s.beginPath(); s.arc(340, 46, 16, 0, 7); s.fill();
    s.fillStyle = "#fffbe8"; s.beginPath(); s.arc(340, 46, 12, 0, 7); s.fill();
  });
  const ridge = (w, base, amp, f1, f2, col, col2, seed) => mk(w, VH, s => {
    for (let x = 0; x < w; x++) {
      const t = (x / w) * Math.PI * 2;
      const h = base - amp * (0.55 * Math.sin(t * f1 + seed) + 0.3 * Math.sin(t * f2 + seed * 2) + 0.15 * Math.sin(t * 13 + seed));
      s.fillStyle = col; s.fillRect(x, Math.round(h), 1, VH);
      if (col2 && (x % 7 < 3)) { s.fillStyle = col2; s.fillRect(x, Math.round(h) + 3 + (x % 5), 1, 2); }
    }
  });
  const mountains = ridge(832, 140, 40, 3, 7, "#8aa6e8", "#9fb8f0", 1.3);
  const snow = ridge(832, 140, 40, 3, 7, null, null, 1.3);
  const hills = ridge(832, 178, 22, 5, 11, "#4fae6a", "#62c27c", 0.4);
  const cloud = mk(48, 20, s => {
    s.fillStyle = "#ffffff";
    for (const [x, y, r] of [[12, 12, 7], [22, 8, 9], [33, 11, 7], [40, 14, 5], [6, 15, 4]]) { s.beginPath(); s.arc(x, y, r, 0, 7); s.fill(); }
    s.fillStyle = "#dbe9ff"; s.fillRect(4, 16, 40, 3);
  });
  return { sky, mountains, hills, cloud };
})();

function drawBackground() {
  g.drawImage(bg.sky, 0, 0);
  const rep = (img, f, y = 0) => { const w = img.width, o = -((cam * f) % w); g.drawImage(img, o, y); g.drawImage(img, o + w, y); };
  for (let i = 0; i < 6; i++) {
    const x = ((i * 157 - cam * 0.15 - tick * 0.05) % 640 + 640) % 640 - 60;
    g.globalAlpha = 0.9; g.drawImage(bg.cloud, Math.round(x), 18 + (i * 37) % 60); g.globalAlpha = 1;
  }
  rep(bg.mountains, 0.2);
  rep(bg.hills, 0.45);
}

// ---------- tiles ----------
const SOLID = new Set(["#", "B", "?", "U", "S", "P"]);
function tileAt(c, r) { if (!tiles || r < 0 || r >= tiles.length || c < 0 || c >= tiles[0].length) return "."; return tiles[r][c]; }
function drawTiles() {
  const c0 = Math.floor(cam / T), c1 = c0 + Math.ceil(VW / T) + 1;
  for (let r = 0; r < tiles.length; r++) for (let c = c0; c <= c1; c++) {
    const ch = tileAt(c, r);
    if (ch === ".") continue;
    let img = null;
    if (ch === "#") img = SOLID.has(tileAt(c, r - 1)) ? TILES.dirt : TILES.grass;
    else if (ch === "B") img = TILES.brick;
    else if (ch === "?") img = (Math.floor(tick / 20) % 3 === 0) ? TILES.q1 : TILES.q0;
    else if (ch === "U") img = TILES.used;
    else if (ch === "S") img = TILES.stone;
    else if (ch === "P") {
      const isLeft = pipeLeft(c, r);
      const top = tileAt(c, r - 1) !== "P";
      img = top ? (isLeft ? TILES.pipeTopL : TILES.pipeTopR) : (isLeft ? TILES.pipeL : TILES.pipeR);
    }
    const b = bumps[c + "," + r];
    const dy = b ? -Math.sin((b / 10) * Math.PI) * 5 : 0;
    if (img) g.drawImage(img, Math.round(c * T - cam), Math.round(r * T + dy));
  }
  for (const k in bumps) { bumps[k]++; if (bumps[k] > 10) delete bumps[k]; }
}
function pipeLeft(c, r) { let n = 0; while (tileAt(c - 1 - n, r) === "P") n++; return n % 2 === 0; }

function drawCoins() {
  const ph = Math.floor(tick / 6) % 4, widths = [8, 6, 2, 6];
  for (const k of coins) {
    const [c, r] = k.split(",").map(Number), x = c * T - cam;
    if (x < -16 || x > VW) continue;
    const w = widths[ph], cx = Math.round(x + 8 - w / 2), y = r * T + 2 + Math.round(Math.sin((tick + c * 9) / 14));
    g.fillStyle = "#8a5a00"; g.fillRect(cx - 1, y, w + 2, 12);
    g.fillStyle = "#ffd23f"; g.fillRect(cx, y + 1, w, 10);
    if (w > 2) { g.fillStyle = "#fff3b0"; g.fillRect(cx + 1, y + 2, 1, 7); }
  }
}

function drawFlag() {
  const fx = level.flag * T - cam;
  if (fx < -40 || fx > VW + 40) return;
  const top = 2 * T, base = 12 * T;
  g.fillStyle = "#3a3f55"; g.fillRect(fx + 6, top, 4, base - top);
  g.fillStyle = "#e8ecff"; g.fillRect(fx + 7, top, 2, base - top);
  g.fillStyle = "#ffd23f"; g.beginPath(); g.arc(fx + 8, top - 3, 4, 0, 7); g.fill();
  for (let i = 0; i < 18; i++) {
    const wave = Math.round(Math.sin(tick / 8 + i / 3) * 1.5);
    const h = Math.max(1, 14 - Math.round(i * 0.75));
    g.fillStyle = i % 6 < 3 ? "#2bb3a3" : "#35cbb9";
    g.fillRect(fx + 10 + i, Math.round(flagY) + wave + (14 - h) / 2, 1, h);
  }
  g.fillStyle = "#fff"; g.fillRect(fx + 15, Math.round(flagY) + 6 + Math.round(Math.sin(tick / 8 + 2) * 1.5), 3, 3);
  // little hut behind the flag
  const hx = fx + 4 * T;
  g.fillStyle = "#7a4a2a"; g.fillRect(hx, base - 40, 56, 40);
  g.fillStyle = "#c9541f"; g.beginPath(); g.moveTo(hx - 8, base - 40); g.lineTo(hx + 28, base - 66); g.lineTo(hx + 64, base - 40); g.fill();
  g.fillStyle = "#1b1430"; g.fillRect(hx + 22, base - 22, 12, 22);
  g.fillStyle = "#ffd27a"; g.fillRect(hx + 6, base - 32, 9, 8); g.fillRect(hx + 41, base - 32, 9, 8);
}
let flagY = 2 * T + 4;

function drawEnemies() {
  for (const [x, y, dir, sq] of frame.e) {
    const sx = Math.round(x - 1 - cam), sy = Math.round(y - 2);
    if (sx < -20 || sx > VW + 20) continue;
    if (sq > 0) { g.globalAlpha = sq > 20 ? (30 - sq) / 10 : 1; g.drawImage(SPR.beetleFlat, sx, sy); g.globalAlpha = 1; continue; }
    const f = Math.floor(tick / 10) % 2;
    g.drawImage(SPR[(f ? "beetle2" : "beetle1") + (dir > 0 ? "f" : "")], sx, sy);
  }
}

function drawHero() {
  const [x, y, vx, vy, face, ground, dead, won] = frame.m;
  let k = "idle";
  if (dead) k = "dead";
  else if (!ground) k = "jump";
  else if (Math.abs(vx) > 0.1) k = Math.floor(tick / 5) % 2 ? "run1" : "run2";
  const img = SPR["hero_" + k + "_" + (face < 0 ? "l" : "r")];
  g.drawImage(img, Math.round(x - 2 - cam), Math.round(y - 2));
}

// ---------- particles ----------
function spawn(kind, x, y) {
  if (kind === "coin") {
    particles.push({ t: "coin", x, y, vy: -3.2, life: 30 });
    for (let i = 0; i < 6; i++) particles.push({ t: "spark", x: x + 8, y: y + 4, vx: Math.cos(i) * 1.5, vy: Math.sin(i) * 1.5 - 1, life: 18 });
  } else if (kind === "dust") {
    for (let i = 0; i < 8; i++) particles.push({ t: "dust", x: x + 7, y: y + 12, vx: (i - 3.5) * 0.4, vy: -0.6 - (i % 3) * 0.3, life: 22 });
  } else if (kind === "pickup") {
    for (let i = 0; i < 8; i++) particles.push({ t: "spark", x: x + 8, y: y + 8, vx: Math.cos(i * 0.8) * 1.2, vy: Math.sin(i * 0.8) * 1.2, life: 16 });
  }
}
function drawParticles() {
  particles = particles.filter(p => p.life-- > 0);
  for (const p of particles) {
    p.x += p.vx || 0; p.y += p.vy || 0;
    const sx = Math.round(p.x - cam), sy = Math.round(p.y);
    if (p.t === "coin") { p.vy += 0.2; g.fillStyle = "#8a5a00"; g.fillRect(sx + 3, sy - 1, 10, 13); g.fillStyle = "#ffd23f"; g.fillRect(sx + 4, sy, 8, 11); }
    else if (p.t === "spark") { g.fillStyle = p.life % 4 < 2 ? "#fff6c7" : "#ffd23f"; g.fillRect(sx, sy, 2, 2); }
    else if (p.t === "dust") { p.vy += 0.03; g.fillStyle = `rgba(255,255,255,${p.life / 22})`; g.fillRect(sx, sy, 3, 3); }
  }
}

// ---------- HUD ----------
function drawHUD() {
  g.font = "8px 'Press Start 2P', monospace"; g.textBaseline = "top";
  const sh = (t, x, y, c = "#fff") => { g.fillStyle = "#000"; g.fillText(t, x + 1, y + 1); g.fillStyle = c; g.fillText(t, x, y); };
  const h = frame ? frame.hud : { coins: 0, dist: 0, time: 0 };
  g.fillStyle = "#ffd23f"; g.fillRect(10, 9, 6, 8); g.fillStyle = "#8a5a00"; g.fillRect(10, 16, 6, 1);
  sh("x" + String(h.coins).padStart(2, "0"), 20, 9);
  sh("DIST " + Math.floor(h.dist) + "/" + (level ? level.flag : 0), 92, 9);
  sh("TIME " + h.time.toFixed(1), 236, 9);
  const tag = { play: "YOU", bot: "BOT", model: "LLM", replay: "REPLAY" }[mode] || "";
  sh(tag, VW - 10 - g.measureText(tag).width, 9, mode === "model" ? "#7ff3ff" : "#ffd27a");
  if (level) {  // progress bar
    const p = Math.min(1, h.dist / level.flag);
    g.fillStyle = "#0006"; g.fillRect(92, 21, 120, 3); g.fillStyle = "#7ff3ff"; g.fillRect(92, 21, Math.round(120 * p), 3);
  }
}

function render() {
  tick++;
  if (level && frame) {
    const target = Math.max(0, Math.min(level.w * T - VW, frame.m[0] - VW * 0.38));
    cam += (target - cam) * 0.14;
    if (Math.abs(target - cam) > VW) cam = target;
    if (frame.m[7]) flagY = Math.min(12 * T - 18, flagY + 2.5); else flagY = 2 * T + 4;
    drawBackground(); drawTiles(); drawCoins(); drawFlag(); drawEnemies(); drawHero(); drawParticles(); drawHUD();
  } else if (level) {
    drawBackground(); drawTiles(); drawCoins(); drawFlag(); drawHUD();
  } else {
    drawBackground();
  }
  requestAnimationFrame(render);
}

// ---------- side panel ----------
const bars = $("bars");
MOVES.forEach((m, i) => {
  const d = document.createElement("div"); d.className = "bar"; d.id = "bar" + i;
  d.innerHTML = `<div class="lab"><b>${i + 1}</b>${m}</div><div class="track"><div class="fill"></div></div><div class="val">–</div>`;
  bars.appendChild(d);
});
function showDecision(f) {
  $("txt").textContent = f.txt;
  f.p.forEach((p, i) => {
    const b = $("bar" + i); b.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%";
    b.querySelector(".val").textContent = (p * 100).toFixed(0) + "%"; b.classList.toggle("pick", i === f.a);
  });
}
function showStats(h) {
  $("stats").innerHTML = [["dist", Math.floor(h.dist)], ["coins", h.coins], ["stomps", h.stomps], ["time", h.time.toFixed(1)]]
    .map(([k, v]) => `<div><b>${v}</b><span>${k}</span></div>`).join("");
}
showStats({ dist: 0, coins: 0, stomps: 0, time: 0 });

function overlay(html) { const o = $("overlay"); o.innerHTML = html || ""; o.classList.toggle("show", !!html); }

// ---------- connection ----------
function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.t === "level") {
      level = m.level; tiles = level.tiles.map(r => r.split("")); coins = new Set(level.coins.map(([c, r]) => c + "," + r));
      particles = []; bumps = {}; frame = null; ended = null; label = m.label; overlay("");
      cam = 0;
    } else if (m.t === "f") {
      frame = m;
      for (const e of m.ev) {
        if (e[0] === "coin") { coins.delete(e[1] + "," + e[2]); spawn("pickup", e[1] * T, e[2] * T); }
        else if (e[0] === "bump") { bumps[e[1] + "," + e[2]] = 1; if (e[3]) { tiles[e[2]][e[1]] = "U"; spawn("coin", e[1] * T, e[2] * T - 14); } }
        else if (e[0] === "stomp") spawn("dust", e[1], e[2]);
        else if (e[0] === "win") overlay(`<div class="big" style="color:#7ff3ff">CLEARED!</div>`);
        else if (e[0] === "die") overlay(`<div class="big" style="color:#ff8a8a">${e[1] === "fell" ? "FELL IN!" : "OUCH!"}</div>`);
      }
      if (m.hud.done && m.hud.reason === "stuck" && !ended) overlay(`<div class="big">STUCK</div>`);
      if (m.txt) showDecision(m);
      showStats(m.hud);
    } else if (m.t === "end") {
      ended = m;
      overlay(`<div class="big" style="color:${m.won ? "#7ff3ff" : "#ff8a8a"}">${m.won ? "CLEARED!" : m.reason.toUpperCase()}</div>
        <div class="small">distance ${m.dist} · coins ${m.coins} · stomps ${m.stomps} · ${m.time}s<br>press Start or R to go again</div>`);
    } else if (m.t === "error") {
      overlay(`<div class="small" style="max-width:80%">${m.msg}</div>`);
    }
  };
  ws.onclose = () => setTimeout(connect, 1000);
  ws.onopen = () => { if (!level) start(); };
}

function start() {
  if (!ws || ws.readyState !== 1) return;
  const msg = { cmd: "start", mode, level: $("level").value, seed: Number($("seed").value) };
  if (mode === "model") msg.checkpoint = $("ckpt").value;
  if (mode === "replay") { msg.replay = $("replay").value; if (!msg.replay) { overlay(`<div class="small">no replays yet</div>`); return; } }
  ws.send(JSON.stringify({ cmd: "speed", value: Number($("speed").value) }));
  ws.send(JSON.stringify(msg));
  if (mode === "play") { $("txt").textContent = "You are playing: the model's view appears in bot, model and replay modes."; cv.focus(); }
}

document.querySelectorAll("#modes button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#modes button").forEach(x => x.classList.toggle("on", x === b));
  mode = b.dataset.mode; syncControls(); start();
});
function syncControls() {
  $("ckptWrap").style.display = mode === "model" ? "" : "none";
  $("replayWrap").style.display = mode === "replay" ? "" : "none";
  $("hint").textContent = mode === "play" ? "Arrow keys / A D to move, Space or ↑ to jump (hold for higher). R restarts."
    : mode === "replay" ? "A replay is a level, a seed and a list of moves, re-run by the same simulation: no GPU needed."
    : mode === "model" ? "The model reads the text on the right and picks one of 6 moves every 6 frames (0.1 s)."
    : "A simple scripted player: run right, long-jump at walls, pits and enemies. Its mistakes are the point.";
}
$("go").onclick = start;
$("speed").onchange = () => ws && ws.send(JSON.stringify({ cmd: "speed", value: Number($("speed").value) }));

const KEYMAP = { ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right", Space: "jump", ArrowUp: "jump", KeyW: "jump" };
function sendKeys() { if (ws && ws.readyState === 1 && mode === "play") ws.send(JSON.stringify({ cmd: "keys", ...keys })); }
addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "KeyR") { start(); return; }
  const k = KEYMAP[e.code]; if (!k) return; e.preventDefault();
  if (!keys[k]) { keys[k] = true; sendKeys(); }
});
addEventListener("keyup", e => { const k = KEYMAP[e.code]; if (!k) return; keys[k] = false; sendKeys(); });

// ---------- checkpoints, replays, curves ----------
async function loadLists() {
  try {
    const reps = await (await fetch("/api/replays")).json();
    const keep = $("replay").value;
    $("replay").innerHTML = reps.map(r => `<option value="${r.id}">${r.label}${r.won ? " ✓" : ""}</option>`).join("");
    if (keep) $("replay").value = keep;
  } catch (e) { }
  try {
    const st = await (await fetch("/api/policy")).json();
    const cks = st.checkpoints || [];
    const keepC = $("ckpt").value;
    $("ckpt").innerHTML = cks.length ? cks.map(c => `<option>${c}</option>`).join("") : `<option value="">(policy worker offline)</option>`;
    if (keepC && cks.includes(keepC)) $("ckpt").value = keepC;
  } catch (e) { }
  try { runsData = await (await fetch("/api/runs")).json(); drawCurves(); } catch (e) { }
}
let runsData = [];
const COLORS = ["#ffb547", "#4fd1c5", "#a78bfa", "#ff6b6b"];
function drawCurves() {
  const c = $("curve"), x = c.getContext("2d"), key = $("metric").value, dpr = window.devicePixelRatio || 1;
  const W = Math.max(280, c.clientWidth || 520), H = 210, L = 40, B = 24;
  c.width = W * dpr; c.height = H * dpr; c.style.height = H + "px";
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, W, H);
  x.font = "11px Inter, sans-serif"; x.fillStyle = "#8b93b8"; x.strokeStyle = "#2a3358";
  // the runs that tell the story (all others stay in mario/runs for reference)
  const FEATURED = { scratch: "PPO from scratch (rank 16)", main: "bot warm-up + PPO (rank 16)",
    r128: "final: expert corrections (rank 128) + PPO" };
  const shown = runsData.filter(r => FEATURED[r.name]).sort((a, b) => Object.keys(FEATURED).indexOf(a.name) - Object.keys(FEATURED).indexOf(b.name));
  const series = (shown.length ? shown : runsData).map(r => ({ name: FEATURED[r.name] || r.meta.label || r.name, pts: r.metrics.filter(m => m[key] !== undefined).map(m => [m.step || m.iter || 0, m[key]]) }))
    .filter(s => s.pts.length);
  if (!series.length) { x.fillText("No training runs fetched yet (mario/runs/<run>/metrics.jsonl).", 10, 24); $("legend").innerHTML = ""; return; }
  const xmax = Math.max(...series.flatMap(s => s.pts.map(p => p[0]))) || 1;
  const vals = series.flatMap(s => s.pts.map(p => p[1]));
  const isFrac = ["eval_clear", "clear", "frac"].includes(key);
  const ymin = isFrac ? 0 : Math.min(...vals), ymax = isFrac ? 1 : Math.max(...vals) || 1;
  const px = v => L + (v / xmax) * (W - L - 10), py = v => H - B - ((v - ymin) / (ymax - ymin || 1)) * (H - B - 10);
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (ymax - ymin) * i / 4, y = py(v);
    x.beginPath(); x.moveTo(L, y); x.lineTo(W - 10, y); x.stroke();
    x.fillText(isFrac ? Math.round(v * 100) + "%" : v.toFixed(1), 4, y + 4);
  }
  x.fillText("PPO iteration", W / 2 - 30, H - 6);
  x.fillText(String(xmax), W - 30, H - 6);
  series.forEach((s, i) => {
    x.strokeStyle = COLORS[i % 4]; x.lineWidth = 2; x.beginPath();
    s.pts.forEach(([a, b], j) => j ? x.lineTo(px(a), py(b)) : x.moveTo(px(a), py(b))); x.stroke(); x.lineWidth = 1; x.strokeStyle = "#2a3358";
  });
  if (key === "eval_clear") { x.strokeStyle = "#6ee7a0"; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(L, py(0.9)); x.lineTo(W - 10, py(0.9)); x.stroke(); x.setLineDash([]); x.strokeStyle = "#2a3358"; }
  $("legend").innerHTML = series.map((s, i) => `<span><i style="background:${COLORS[i % 4]}"></i>${s.name}</span>`).join("") +
    (key === "eval_clear" ? `<span><i style="background:#6ee7a0"></i>target 90%</span>` : "");
}
$("metric").onchange = drawCurves;

syncControls(); loadLists(); connect(); render();
setInterval(loadLists, 30000);
addEventListener("resize", () => drawCurves());
