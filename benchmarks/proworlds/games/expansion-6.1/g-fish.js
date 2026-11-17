// Fish (assemble): one large fish fills the board on plain blue water. Its name follows from three parts, each a
// SHAPE difference in the same colours: the back fin (pointed or rounded), the tail (forked or fan-shaped) and the side of the body
// (stripes or spots). The whole fish may sit one site further left or right (hidden). Each part sits at one site; every other
// site is plain water in every world, so only part sites vary (C5).
import { defineTile, paint, rect, disc, line, C } from './tiles.js';
const MEMBERS = { pip: '111', rio: '110', coco: '101', milo: '100', kiki: '011', zara: '010', toto: '001', bobo: '000' };   // pointed fin? forked tail? stripes?
const water = s => rect(s, 0, 0, 15, 15, C.blue);
defineTile('fish:water', paint(s => water(s)));
// v1.1 (P2 .66: the model read the orange fin / tail sites as plain body corners): every fin and tail tile is now a large BLACK shape
// on blue water, so a part site never looks like the body; the text names each shape.
defineTile('fish:fin_pointed', paint(s => { water(s); for (let y = 2; y <= 15; y++) { const w = Math.round((y - 2) * 0.55); rect(s, 7 - w, y, 8 + w, y, C.black); } }));
defineTile('fish:fin_round', paint(s => { water(s); disc(s, 7.5, 15, 7.5, C.black); }));
defineTile('fish:tail_forked', paint(s => { water(s); line(s, 1, 7, 14, 1, 4, C.black); line(s, 1, 8, 14, 14, 4, C.black); }));
defineTile('fish:tail_fan', paint(s => { water(s); for (let x = 1; x <= 14; x++) { const h = Math.round(1 + x * 0.45); rect(s, x, 7 - h, x, 8 + h, C.black); } }));
defineTile('fish:side_stripes', paint(s => { rect(s, 0, 0, 15, 15, C.orange); for (const x of [2, 7, 12]) rect(s, x, 0, x + 1, 15, C.black); }));
defineTile('fish:side_spots', paint(s => { rect(s, 0, 0, 15, 15, C.orange); for (const [x, y] of [[4, 4], [11, 4], [7, 8], [4, 12], [11, 12]]) disc(s, x, y, 1.6, C.black); }));
const PARTS = [['fin_pointed', 'fin_round', [3, 2]], ['tail_forked', 'tail_fan', [7, 5]], ['side_stripes', 'side_spots', [3, 6]]];
const BOARDS = { normal: { cols: 9, rows: 9, budget: 35, shift: 1, stride: 1, scale: 1 }, long: { cols: 32, rows: 32, budget: 65, shift: 1, stride: 2, scale: 3.5 } };
const WORD = [['a pointed back fin (a black triangle)', 'a rounded back fin (a black half-disc)'], ['a forked tail (a black V)', 'a fan-shaped tail (a solid black fan)'], ['black stripes on its orange side', 'black spots on its orange side']];
const says = Object.entries(MEMBERS).map(([m, b]) => `${m} = ${[...b].map((v, k) => WORD[k][v === '1' ? 0 : 1]).join(', ')}`).join('; ');
export const FISH = {
  name: 'Fish', kind: 'assemble', picture: true,
  members: Object.keys(MEMBERS),
  intro: {
    normal: 'One large orange fish fills the board on plain blue water, facing left. Three parts show which fish it is, each drawn in black on the blue water or on its orange side: the back fin near the top left, the tail at the middle right and one patch of its orange side at the lower left. Every other site shows plain water. The whole fish may sit one site further left or right.',
    long: 'One very large orange fish fills the board on plain blue water, facing left, drawn wide: the back fin near the top left, the tail at the middle right and one patch of its side at the lower left, with wide plain water between them. Every other site shows plain water. The whole fish may sit two sites further left or right.',
  },
  ask: 'Which fish is it?',
  setText: `one of eight fish, each with its own back fin, tail and side: ${says}.`,
  boards: Object.fromEntries(Object.entries(BOARDS).map(([k, b]) => [k, { cols: b.cols, rows: b.rows, budget: b.budget }])),
  nuisances(board) { const b = BOARDS[board], out = []; for (let dy = 0; dy <= 0; dy++) for (let dx = -b.shift; dx <= b.shift; dx++) out.push({ dx: dx * b.stride, dy: dy * b.stride }); return out; },
  layout(member, nu, board) {
    const b = BOARDS[board], sites = Array(b.cols * b.rows).fill('fish:water'), bits = MEMBERS[member];
    for (let k = 0; k < PARTS.length; k++) {
      const [one, zero, [x0, y0]] = PARTS[k], x = (board === 'normal' ? x0 : Math.round(x0 * b.scale) + 1) + nu.dx, y = (board === 'normal' ? y0 : Math.round(y0 * b.scale) + 1) + nu.dy;
      if (x < 0 || y < 0 || x >= b.cols || y >= b.rows) return null;
      sites[y * b.cols + x] = `fish:${bits[k] === '1' ? one : zero}`;
    }
    return sites;
  },
};
