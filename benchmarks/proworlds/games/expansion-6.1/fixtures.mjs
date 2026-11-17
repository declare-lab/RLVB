// PRO Worlds assembly 6.0: test fixtures (not real games). Three worlds break one
// rule each and MUST fail test.mjs; one clean world MUST pass. Run:
// node games/assembly/test.mjs --fixtures
import { registerGame } from './games.js';
import { defineTile, paint, rect, C } from './tiles.js';
const NAMES = ['ash', 'elm', 'fir', 'oak', 'yew', 'lime', 'pine', 'teak'];
const marks = NAMES.map((n, i) => defineTile(`fx:${n}`, paint(s => rect(s, 1 + (i % 4) * 3, 1 + Math.floor(i / 4) * 7, 3 + (i % 4) * 3, 6 + Math.floor(i / 4) * 7, C.black))));
defineTile('fx:on', paint(s => rect(s, 2, 2, 13, 13, C.red)));
defineTile('fx:off', paint(s => rect(s, 2, 2, 13, 13, C.blue)));
const B = (cols, rows, budget) => ({ cols, rows, budget });
const base = { kind: 'join', picture: false, members: NAMES, ask: 'Which tree is it?', setText: 'one of: ash, elm, fir, oak, yew, lime, pine, teak.', intro: 'Fixture.' };
// 1. COUNTING (A5): the member = how many of 8 fixed sites are red (1-8); places vary.
registerGame('fxcount', { ...base, boards: { normal: B(5, 5, 48), long: B(9, 9, 150) },
  nuisances: () => Array.from({ length: 24 }, (_, k) => k),
  layout(m, nu, board) { const n = NAMES.indexOf(m) + 1, cols = board === 'normal' ? 5 : 9, sites = Array(cols * cols).fill('blank'), slots = [0, 2, 4, 10, 14, 20, 22, 24].map(x => board === 'normal' ? x : x * 2 + Math.floor(x / 5) * 8); const order = slots.map((s, i) => [s, (i * 7 + nu * 13) % 17]).sort((a, b) => a[1] - b[1]).map(x => x[0]); order.forEach((s, i) => { sites[s] = i < n ? 'fx:on' : 'fx:off'; }); return sites; } });
// 2. NEEDLE (B7): one fixed site shows the member's own mark; nothing else varies.
registerGame('fxneedle', { ...base, boards: { normal: B(5, 5, 48), long: B(9, 9, 150) },
  nuisances: () => [0],
  layout(m, nu, board) { const cols = board === 'normal' ? 5 : 9, sites = Array(cols * cols).fill('blank'); sites[cols * cols - 1] = marks[NAMES.indexOf(m)]; return sites; } });
// 3. PAIR-ONLY (C5): 3 bits, each = whether two far sites show the SAME colour; one site alone tells nothing.
registerGame('fxpair', { ...base, boards: { normal: B(7, 7, 48), long: B(11, 11, 150) },
  nuisances: () => [0, 1, 2, 3, 4, 5, 6, 7],
  layout(m, nu, board) { const cols = board === 'normal' ? 7 : 11, k = NAMES.indexOf(m), sites = Array(cols * cols).fill('blank'), pairs = [[0, cols - 1], [cols * (cols - 1), cols * cols - 1], [Math.floor(cols / 2), cols * Math.floor(cols / 2)]];
    pairs.forEach(([a, b], j) => { const first = (nu >> j) & 1, same = (k >> j) & 1; sites[a] = first ? 'fx:on' : 'fx:off'; sites[b] = (same ? first : 1 - first) ? 'fx:on' : 'fx:off'; }); return sites; } });
// 4. CLEAN: 3 independent features, each read at its own far site, each splitting the set in half.
registerGame('fxclean', { ...base, boards: { normal: B(7, 7, 48), long: B(13, 13, 150) },
  nuisances: () => [0],
  layout(m, nu, board) { const cols = board === 'normal' ? 7 : 13, k = NAMES.indexOf(m), sites = Array(cols * cols).fill('blank'), at = [0, cols - 1, cols * cols - 1 - Math.floor(cols / 2)];
    at.forEach((s, j) => { sites[s] = ((k >> j) & 1) ? 'fx:on' : 'fx:off'; }); return sites; } });
// 5. ADAPTIVE (B3): the centre shows bit 0; bit 1 shows on the LEFT site if bit 0 is off, else on the RIGHT
// site (the other side stays empty); bit 2 likewise one level further. The next best look depends on what
// was seen. fxclean (fixed places) must be FLAGGED as not adaptive; fxadapt must be adaptive.
registerGame('fxadapt', { ...base, boards: { normal: B(9, 7, 48), long: B(17, 13, 150) },
  nuisances: () => [0],
  layout(m, nu, board) { const cols = board === 'normal' ? 9 : 17, rows = board === 'normal' ? 7 : 13, k = NAMES.indexOf(m), sites = Array(cols * rows).fill('blank'), mid = Math.floor(rows / 2) * cols, c = Math.floor(cols / 2), q = Math.floor(cols / 4);
    const b0 = k & 1, b1 = (k >> 1) & 1, b2 = (k >> 2) & 1, x1 = b0 ? c + q : c - q, x2 = b1 ? x1 + Math.floor(q / 2) + 1 : x1 - Math.floor(q / 2) - 1;
    sites[mid + c] = b0 ? 'fx:on' : 'fx:off'; sites[mid + x1] = b1 ? 'fx:on' : 'fx:off'; sites[(rows - 1) * cols + x2] = b2 ? 'fx:on' : 'fx:off'; return sites; } });
