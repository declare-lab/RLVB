// Bird v1.4 (after probe19 P1 .80: fan read as plain feathers, round head read as crest -> fix: a smooth arc with an eye dot + a fan of 5 separate rays).
// Bird v1.3 (beak + tail tiles redrawn and 'each part filling exactly one site', after probe15 P2 .94: long beak read as thick,
// the beak judged by its span, the fan read as forked). Bird (assemble; picture PARTS games, shape parts, hidden placement): one large bird
// fills the board, seen from the side, facing left. Its name follows from three SHAPE parts at fixed places on the bird: the head top
// (a crest or a smooth round head), the beak (long and thin or short and thick) and the tail (forked or fan-shaped). The two values of
// a part have the same colours. The names match their real look: hoopoe, cardinal, heron, swallow, robin, terns, the crested treeswift.
// a part have the same colours. The rules name the parts of every bird. The whole bird may sit one site further left or right.
import { defineTile, paint, rect, disc, line, C } from './tiles.js';
import { partsGame, partList } from './rb8-parts.js';
const BG = C.brown;
const bg = s => rect(s, 0, 0, 15, 15, BG);
defineTile('bird:feathers', paint(bg));
defineTile('bird:crest', paint(s => { bg(s); for (const x of [3, 7, 11]) for (let y = 2; y <= 13; y++) { const h = Math.round((y - 2) / 11 * 2); rect(s, x - h, y, x + h, y, C.black); } rect(s, 1, 13, 14, 15, C.black); }));
defineTile('bird:round', paint(s => { bg(s); for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const d = (x - 7.5) ** 2 + (y - 15) ** 2; if (d >= 8.5 * 8.5 && d <= 11.5 * 11.5) s(x, y, C.black); } disc(s, 7.5, 11, 1.8, C.black); }));   // v1.4: a thick smooth black arc with a black eye dot, no points
defineTile('bird:beak_long', paint(s => { bg(s); rect(s, 2, 7, 15, 8, C.orange); }));   // v1.3: a thin 2-px orange spike over most of the site (a gap at the left edge)
defineTile('bird:beak_short', paint(s => { bg(s); for (let x = 8; x < 16; x++) { const h = Math.round((x - 8) / 7 * 6); rect(s, x, 7 - h, x, 8 + h, C.orange); } }));   // v1.3: a short, very thick orange wedge in the right half only
defineTile('bird:tail_fork', paint(s => { bg(s); line(s, 1, 7, 14, 0, 3, C.black); line(s, 1, 8, 14, 15, 3, C.black); }));   // v1.3: two long thin points with a wide notch
defineTile('bird:tail_fan', paint(s => { bg(s); for (const a of [-1.1, -0.55, 0, 0.55, 1.1]) line(s, 1, 7.5, Math.round(1 + 13.5 * Math.cos(a)), Math.round(7.5 + 13.5 * Math.sin(a)), 2, C.black); disc(s, 1, 7.5, 2.5, C.black); }));   // v1.4: a spread of 5 separate black feather rays from one base point
const MEMBERS = { 'crested tern': '111', hoopoe: '110', treeswift: '101', cardinal: '100', tern: '011', heron: '010', swallow: '001', robin: '000' };
const WORDS = [['a crest on its head', 'a smooth round head'], ['a long thin beak', 'a short thick beak'], ['a forked tail', 'a fan-shaped tail']];
export const BIRD = partsGame({
  background: 'bird:feathers', name: 'Bird', kind: 'assemble', picture: true, members: MEMBERS,
  intro: {
    normal: 'One large bird fills the board, seen from the side and facing left. Three parts sit in their places on the bird, each part filling exactly one site: the top of its head near the top left (a crest of three black spikes, or a smooth thick black arc with an eye dot and no points), its beak at the left edge below the head (an orange beak: a long thin spike, or a short thick wedge) and its tail near the lower right (a black tail: two long points with a wide notch between them, or a fan of five separate black feather rays spread from one point). Every other site shows plain brown feathers. The whole bird may sit one site further left or right.',
    long: 'One very large bird fills the board, seen from the side and facing left, with wide plain feathers between its parts, each part filling exactly one site: the top of its head near the top left (a crest of three black spikes, or a smooth thick black arc with an eye dot and no points), its beak at the left below the head (an orange beak: a long thin spike, or a short thick wedge) and its tail near the lower right (a black tail: two long points with a wide notch between them, or a fan of five separate black feather rays spread from one point). Every other site shows plain brown feathers. The whole bird may sit two sites further left or right.',
  },
  ask: 'Which bird is it?',
  setText: `one of eight birds, each with its own head, beak and tail: ${partList(MEMBERS, WORDS)}.`,
  parts: [
    { one: 'bird:crest', zero: 'bird:round', at: { normal: [[3, 1]] } },
    { one: 'bird:beak_long', zero: 'bird:beak_short', at: { normal: [[1, 4]] } },
    { one: 'bird:tail_fork', zero: 'bird:tail_fan', at: { normal: [[7, 7]] } },
  ],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shiftX: 1, shiftY: 0, stride: 1 }, long: { cols: 24, rows: 24, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 2.5 } },
});
