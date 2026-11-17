// PRO Worlds assembly 6.0: the tile library. A site is one 16x16 tile of colour
// indices (0 = white ground, 1 = black ink, 2-10 = named colours). Tiles are large
// shapes only (fills, thick lines, dots, letters, simple icons) so that the
// untrained model can read each piece. No tile draws a digit.
export const T = 16;
export const COLORS = ['#ffffff', '#111111', '#d62728', '#1f5fd6', '#2ca02c', '#f2c200', '#ff7f0e', '#8c3fbf', '#ff6fb1', '#8c5a2b', '#8a8a8a'];
export const COLOUR_NAMES = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'grey'];
export const C = Object.fromEntries(COLOUR_NAMES.map((n, i) => [n, i]));

const TILES = new Map();
export const blankTile = () => new Uint8Array(T * T);
export function defineTile(name, pixels) {
  if (TILES.has(name)) throw Error(`Tile defined twice: ${name}`);
  if (!(pixels instanceof Uint8Array) || pixels.length !== T * T) throw Error(`Tile ${name}: 256 colour indices expected`);
  TILES.set(name, pixels);
  return name;
}
export function tile(name) { const t = TILES.get(name); if (!t) throw Error(`Unknown tile: ${name}`); return t; }
export const tileNames = () => [...TILES.keys()];

// Drawing helpers on one 16x16 tile.
export function paint(fn) { const p = blankTile(); fn((x, y, c) => { if (x >= 0 && y >= 0 && x < T && y < T) p[y * T + x] = c; }); return p; }
export const rect = (set, x0, y0, x1, y1, c) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) set(x, y, c); };
export const disc = (set, cx, cy, r, c) => { for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) set(x, y, c); };
export const ring = (set, cx, cy, r0, r1, c) => { for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) { const d = (x - cx) ** 2 + (y - cy) ** 2; if (d >= r0 * r0 && d <= r1 * r1) set(x, y, c); } };
export function line(set, x0, y0, x1, y1, w, c) { const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1); for (let i = 0; i <= n; i++) { const x = Math.round(x0 + (x1 - x0) * i / n), y = Math.round(y0 + (y1 - y0) * i / n); rect(set, x - (w >> 1), y - (w >> 1), x - (w >> 1) + w - 1, y - (w >> 1) + w - 1, c); } }

// Basic tiles every game may use.
defineTile('blank', blankTile());
for (const [name, c] of Object.entries(C)) if (c > 1) defineTile(`fill:${name}`, paint(s => rect(s, 0, 0, T - 1, T - 1, c)));
defineTile('fill:black', paint(s => rect(s, 0, 0, T - 1, T - 1, C.black)));
defineTile('block', paint(s => { rect(s, 1, 1, 14, 14, C.black); }));
defineTile('dot', paint(s => disc(s, 7.5, 7.5, 4, C.black)));
// Pipe / path pieces: thick black lines between the tile edge centres (n, e, s, w).
const EDGE = { n: [7.5, 0], e: [15, 7.5], s: [7.5, 15], w: [0, 7.5] };
export function pathTile(edges, c = C.black) {
  const key = `path:${[...edges].sort().join('')}${c === C.black ? '' : ':' + COLOUR_NAMES[c]}`;
  if (TILES.has(key)) return key;
  return defineTile(key, paint(s => { for (const e of edges) line(s, 7.5, 7.5, EDGE[e][0], EDGE[e][1], 4, c); rect(s, 6, 6, 9, 9, c); }));
}
// Arrow tiles (a thick shaft and a head), pointing n / e / s / w.
for (const [d, [dx, dy]] of Object.entries({ n: [0, -1], e: [1, 0], s: [0, 1], w: [-1, 0] })) {
  defineTile(`arrow:${d}`, paint(s => {
    line(s, 7.5 - dx * 5, 7.5 - dy * 5, 7.5 + dx * 4, 7.5 + dy * 4, 3, C.black);
    for (let k = 0; k < 5; k++) { const bx = 7.5 + dx * (5 - k), by = 7.5 + dy * (5 - k); line(s, bx - dy * k, by - dx * k, bx + dy * k, by + dx * k, 1, C.black); }
  }));
}

// A 5x7 capital-letter font, drawn x2 (10x14) and centred. No digits are defined.
const FONT = {
  A: '01110 10001 10001 11111 10001 10001 10001', B: '11110 10001 10001 11110 10001 10001 11110', C: '01110 10001 10000 10000 10000 10001 01110',
  D: '11110 10001 10001 10001 10001 10001 11110', E: '11111 10000 10000 11110 10000 10000 11111', F: '11111 10000 10000 11110 10000 10000 10000',
  G: '01110 10001 10000 10111 10001 10001 01111', H: '10001 10001 10001 11111 10001 10001 10001', I: '01110 00100 00100 00100 00100 00100 01110',
  J: '00111 00010 00010 00010 00010 10010 01100', K: '10001 10010 10100 11000 10100 10010 10001', L: '10000 10000 10000 10000 10000 10000 11111',
  M: '10001 11011 10101 10101 10001 10001 10001', N: '10001 11001 10101 10011 10001 10001 10001', O: '01110 10001 10001 10001 10001 10001 01110',
  P: '11110 10001 10001 11110 10000 10000 10000', Q: '01110 10001 10001 10001 10101 10010 01101', R: '11110 10001 10001 11110 10100 10010 10001',
  S: '01111 10000 10000 01110 00001 00001 11110', T: '11111 00100 00100 00100 00100 00100 00100', U: '10001 10001 10001 10001 10001 10001 01110',
  V: '10001 10001 10001 10001 10001 01010 00100', W: '10001 10001 10001 10101 10101 10101 01010', X: '10001 10001 01010 00100 01010 10001 10001',
  Y: '10001 10001 01010 00100 00100 00100 00100', Z: '11111 00001 00010 00100 01000 10000 11111',
};
// O, I, S, Z, B, G look like the digits 0, 1, 5, 2, 8, 6: they get no tile (rule A6).
export const DIGIT_LIKE = ['O', 'I', 'S', 'Z', 'B', 'G'];
for (const [ch, rows] of Object.entries(FONT)) if (!DIGIT_LIKE.includes(ch)) defineTile(`letter:${ch}`, paint(s => {
  rows.split(' ').forEach((row, y) => [...row].forEach((b, x) => { if (b === '1') rect(s, 3 + 2 * x, 1 + 2 * y, 4 + 2 * x, 2 + 2 * y, C.black); }));
}));
export const letterTile = ch => { const n = `letter:${ch.toUpperCase()}`; tile(n); return n; };
