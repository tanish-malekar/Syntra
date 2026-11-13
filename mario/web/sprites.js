// Original pixel art for Pip's Run, drawn in code (no external assets).
// Each sprite is a list of rows; each character is a palette key ('.' = transparent).
const PAL = {
  o: "#ff8a3d", O: "#c9541f", y: "#ffd27a",           // beanie + pom
  s: "#ffd0a8", S: "#e8a77c", k: "#1b1430",           // skin, shade, ink
  g: "#7ff3ff", G: "#2b8fb0",                         // goggles
  t: "#2bb3a3", T: "#17716a", w: "#f4f7ff",           // jacket
  b: "#7a4a2a", B: "#4d2b17",                         // boots
  p: "#8b5cf6", P: "#5b34c4", q: "#c4b0ff",           // beetle shell
  r: "#ffcf3f", R: "#c98a00", e: "#fff6c7",           // beetle legs / coin
};

const HERO_TOP = [
  "......yy........",
  ".....oooooo.....",
  "....oooooooo....",
  "...oooooooooo...",
  "...OOOOOOOOOO...",
  "...sGGGsGGGss...",
  "...sGggsGggss...",
  "...sssssssssS...",
  "....ssskksss....",
];
const HERO_IDLE = HERO_TOP.concat([
  "...tttttttttt...",
  "..sttttwwtttts..",
  "..s.ttttwttt.s..",
  "....TTTTTTTT....",
  "....TTT..TTT....",
  "...bbbb..bbbb...",
  "...BBBB..BBBB...",
]);
const HERO_RUN1 = HERO_TOP.concat([
  "...tttttttttts..",
  "..sttttwwttttt..",
  "....ttttwttt....",
  "....TTTTTTTT....",
  "...TTT....TTT...",
  "..bbbb.....bbb..",
  "..BBB......BBB..",
]);
const HERO_RUN2 = HERO_TOP.concat([
  "...tttttttttt...",
  "..sttttwwtttts..",
  "...sttttwttt....",
  "....TTTTTTTT....",
  ".....TTTTTT.....",
  ".....bbbbbb.....",
  ".....BBBBBB.....",
]);
const HERO_JUMP = [
  "......yy......s.",
  ".....oooooo...s.",
  "....oooooooo..s.",
  "...oooooooooo.t.",
  "...OOOOOOOOOO.t.",
  "...sGGGsGGGsstt.",
  "...sGggsGggsst..",
  "...sssssssssSt..",
  "....sskkkkss.t..",
  "...ttttttttttt..",
  "..sttttwwttt....",
  "..s.ttttwttt....",
  "....TTTTTTTTT...",
  "....TTT...TTTb..",
  "...bbbb....BBB..",
  "...BBBB.........",
];
const HERO_DEAD = [
  "......yy........",
  ".....oooooo.....",
  "....oooooooo....",
  "...oooooooooo...",
  "...OOOOOOOOOO...",
  "...skkksskkks...",
  "...sskssssksS...",
  "...sssssssssS...",
  "....ssskksss....",
  "s..tttttttttt..s",
  "stttttwwttttttts",
  "....ttttwttt....",
  "....TTTTTTTT....",
  "....TTT..TTT....",
  "...bbbb..bbbb...",
  "...BBBB..BBBB...",
];

const BEETLE1 = [
  "................",
  "................",
  "......k..k......",
  ".....qk..kq.....",
  "....pppppppp....",
  "...pqqppppppp...",
  "..pqpppPPppppp..",
  "..pwwkpppPwwkp..",
  "..pwkkppppwkkp..",
  "..pppppppppppp..",
  "..pPpPpPpPpPpP..",
  "...PPPPPPPPPP...",
  "....r.r..r.r....",
  "...rr.r..r.rr...",
  "..RR..........R.",
  "................",
];
const BEETLE2 = BEETLE1.slice(0, 12).concat([
  "....r..rr..r....",
  "....rr.rr.rr....",
  "....RR....RR....",
  "................",
]);
const BEETLE_FLAT = [
  "................", "................", "................", "................", "................",
  "................", "................", "................", "................", "................",
  "...pqqpppppp....",
  "..pwwkppppwwkp..",
  "..pPpPpPpPpPpP..",
  "..PPPPPPPPPPPP..",
  "...r.r....r.r...",
  "................",
];

function makeSprite(rows, flip = false) {
  const c = document.createElement("canvas");
  c.width = rows[0].length; c.height = rows.length;
  const g = c.getContext("2d");
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === "." || !PAL[ch]) continue;
      g.fillStyle = PAL[ch];
      g.fillRect(flip ? row.length - 1 - x : x, y, 1, 1);
    }
  });
  return c;
}

const SPR = {};
for (const [k, rows] of Object.entries({ idle: HERO_IDLE, run1: HERO_RUN1, run2: HERO_RUN2, jump: HERO_JUMP, dead: HERO_DEAD })) {
  SPR["hero_" + k + "_r"] = makeSprite(rows);
  SPR["hero_" + k + "_l"] = makeSprite(rows, true);
}
SPR.beetle1 = makeSprite(BEETLE1); SPR.beetle2 = makeSprite(BEETLE2);
SPR.beetle1f = makeSprite(BEETLE1, true); SPR.beetle2f = makeSprite(BEETLE2, true);
SPR.beetleFlat = makeSprite(BEETLE_FLAT);

// ---------- tiles (16x16), drawn procedurally ----------
function hash(x, y) { let h = (x * 374761393 + y * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }

function tileCanvas(draw) { const c = document.createElement("canvas"); c.width = c.height = 16; draw(c.getContext("2d")); return c; }
function px(g, col, x, y, w = 1, h = 1) { g.fillStyle = col; g.fillRect(x, y, w, h); }

const TILES = {};
TILES.dirt = tileCanvas(g => {
  px(g, "#9b5a2e", 0, 0, 16, 16);
  for (let i = 0; i < 18; i++) { const x = (hash(i, 3) * 16) | 0, y = (hash(i, 9) * 16) | 0; px(g, i % 3 ? "#7d4521" : "#b8723f", x, y, 2, 1); }
  px(g, "#6a3a1b", 0, 15, 16, 1);
});
TILES.grass = tileCanvas(g => {
  g.drawImage(TILES.dirt, 0, 0);
  px(g, "#56c94a", 0, 0, 16, 5); px(g, "#9be86a", 0, 0, 16, 1);
  for (let x = 0; x < 16; x += 2) px(g, "#3c9a35", x, 5 + ((x * 7) % 3 === 0 ? 1 : 0), 2, 1);
  px(g, "#3c9a35", 3, 4, 1, 1); px(g, "#3c9a35", 11, 4, 1, 1);
});
TILES.brick = tileCanvas(g => {
  px(g, "#c8623a", 0, 0, 16, 16);
  px(g, "#e98a5c", 0, 0, 16, 1);
  for (const y of [7, 15]) px(g, "#6b2a15", 0, y, 16, 1);
  for (const [x, y0] of [[7, 0], [15, 0], [3, 8], [11, 8]]) px(g, "#6b2a15", x, y0, 1, 7);
  px(g, "#f0a07a", 1, 1, 3, 1); px(g, "#f0a07a", 9, 9, 3, 1);
});
function qblock(phase) {
  return tileCanvas(g => {
    px(g, "#2a1a05", 0, 0, 16, 16);
    px(g, phase ? "#ffc23d" : "#ffd25c", 1, 1, 14, 14);
    px(g, "#fff0a8", 1, 1, 14, 1); px(g, "#fff0a8", 1, 1, 1, 14);
    px(g, "#c27a00", 1, 14, 14, 1); px(g, "#c27a00", 14, 1, 1, 14);
    // a little four-point star glyph
    const c = phase ? "#fff7d6" : "#a35d00";
    px(g, c, 7, 4, 2, 8); px(g, c, 4, 7, 8, 2); px(g, c, 6, 6, 4, 4);
    for (const [x, y] of [[2, 2], [13, 2], [2, 13], [13, 13]]) px(g, "#7a4a00", x, y, 1, 1);
  });
}
TILES.q0 = qblock(0); TILES.q1 = qblock(1);
TILES.used = tileCanvas(g => {
  px(g, "#2a1a05", 0, 0, 16, 16); px(g, "#8a6040", 1, 1, 14, 14); px(g, "#a87a55", 1, 1, 14, 1);
  for (const [x, y] of [[3, 3], [12, 3], [3, 12], [12, 12]]) px(g, "#4a3020", x, y, 1, 1);
});
TILES.stone = tileCanvas(g => {
  px(g, "#3a3f55", 0, 0, 16, 16); px(g, "#8e96b8", 1, 1, 14, 14);
  px(g, "#c4cbe6", 1, 1, 14, 2); px(g, "#c4cbe6", 1, 1, 2, 14);
  px(g, "#5d6585", 1, 13, 14, 2); px(g, "#5d6585", 13, 1, 2, 14);
  px(g, "#a6aecf", 5, 5, 6, 6);
});
// copper pipes: body column (left/right halves) and a riveted rim on top
function pipe(part) {
  return tileCanvas(g => {
    const top = part.startsWith("top"), left = part.endsWith("L");
    const cols = ["#3d1d0c", "#b8642c", "#e08a4a", "#ffbe82", "#d07a3c", "#a2551f", "#7a3d14", "#3d1d0c"];
    for (let x = 0; x < 16; x++) {
      const gx = left ? x : x + 16; // 32 px wide pipe: gradient across both halves
      let i = Math.min(7, Math.floor((gx / 32) * 8));
      if (!top && (gx < 2 || gx > 29)) continue;
      px(g, cols[i], x, 0, 1, 16);
    }
    if (top) {
      px(g, "#3d1d0c", 0, 0, 16, 1); px(g, "#3d1d0c", 0, 7, 16, 1);
      for (let x = left ? 4 : 2; x < 16; x += 6) px(g, "#ffe1bd", x, 3, 2, 2);
    } else {
      if (left) px(g, "#3d1d0c", 2, 0, 1, 16); else px(g, "#3d1d0c", 13, 0, 1, 16);
    }
  });
}
TILES.pipeTopL = pipe("topL"); TILES.pipeTopR = pipe("topR"); TILES.pipeL = pipe("bodyL"); TILES.pipeR = pipe("bodyR");
