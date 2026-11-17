// Bee v2 (tile rule: v1 had 2-px lines + object words): every part tile = ONE proven black shape (turtle / shield / bicycle paint code), words = the shape names, the intro names each part's two shapes. Members, slots, boards = v1.
// Bee (relative; batch 3, on the FACE template: 3 parts at 2 mirrored slots, all 9 normal shifts valid, long shift left-right only): one large bee on a plain yellow board. Four binary
// SHAPE parts (no counts), each a big BLACK shape on the plain field; the eight members use the eight even-parity part sets (as Face), so every informative
// look halves what is left. VARIETY: the legs part has THREE slots in a row. The whole picture may sit one site higher, lower, left or right (hidden placement). Neutral names.
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame } from './w3-parts.js';
import { partList } from './w3-text.js';
const F = s => rect(s, 0, 0, 15, 15, C.yellow);
defineTile('bee:field', paint(s => F(s)));
defineTile('bee:ant_a', paint(s => { F(s); rect(s, 6, 1, 9, 14, C.black); rect(s, 1, 6, 14, 9, C.black); }));   // a black plus sign
defineTile('bee:ant_b', paint(s => { F(s); for (let y = 1; y < 15; y++) { const h = 7 - Math.abs(y - 7.5); rect(s, Math.round(7.5 - h), y, Math.round(7.5 + h), y, C.black); } }));   // a black diamond
defineTile('bee:wing_a', paint(s => { F(s); rect(s, 6, 1, 9, 14, C.black); rect(s, 1, 6, 14, 9, C.black); line(s, 3, 3, 12, 12, 3, C.black); line(s, 12, 3, 3, 12, 3, C.black); }));   // a black eight-point star
defineTile('bee:wing_b', paint(s => { F(s); ring(s, 7.5, 7.5, 3.5, 6.5, C.black); }));   // a black ring
defineTile('bee:legs_a', paint(s => { F(s); rect(s, 0, 5, 15, 10, C.black); }));   // a wide flat black bar
defineTile('bee:legs_b', paint(s => { F(s); line(s, 1, 3, 7, 12, 3, C.black); line(s, 14, 3, 8, 12, 3, C.black); }));   // a black V shape
defineTile('bee:sting_a', paint(s => { F(s); rect(s, 2, 2, 13, 13, C.black); }));   // a black square
defineTile('bee:sting_b', paint(s => { F(s); rect(s, 5, 0, 10, 15, C.black); }));   // a tall thin black bar
const MEMBERS = { cobalt: '1111', indigo: '1100', ochre: '1010', sienna: '1001', cerise: '0110', teal: '0101', vermilion: '0011', sepia: '0000' };
const WORDS = [["a black plus sign", "a black diamond"], ["a black eight-point star", "a black ring"], ["a wide flat black bar", "a black V shape"], ["a black square", "a tall thin black bar"]];
export const BEE = partsGame({
  prefix: 'bee', background: 'field', name: 'Bee', kind: 'relative', picture: true, members: MEMBERS,
  intro: { normal: "One large bee is on a plain yellow board. Four parts sit in their places, each drawn as one big black shape filling one site (a part that sits at two or more sites shows the same shape at each of them): its two antennae at the top: a black plus sign or a black diamond; its two wings at the far left and far right: a black eight-point star or a black ring; its legs in a row of three low on the board: a wide flat black bar or a black V shape; its stinger at the bottom: a black square or a tall thin black bar. Every other site shows plain yellow. The whole bee may sit one site higher, lower, left or right, or one site diagonally.", long: "One very large bee is on a plain yellow board, drawn wide with wide plain yellow between the parts. Four parts sit in their places, each drawn as one big black shape filling one site (a part that sits at two or more sites shows the same shape at each of them): its two antennae at the top: a black plus sign or a black diamond; its two wings at the far left and far right: a black eight-point star or a black ring; its legs in a row of three low on the board: a wide flat black bar or a black V shape; its stinger at the bottom: a black square or a tall thin black bar. Every other site shows plain yellow. The whole bee may sit two sites further left or right." },
  ask: 'Which bee is it?',
  setText: `one of eight bees, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [{ one: 'ant_a', zero: 'ant_b', at: { normal: [[3, 1], [5, 1, true]] } }, { one: 'wing_a', zero: 'wing_b', at: { normal: [[1, 4], [7, 4, true]] } }, { one: 'legs_a', zero: 'legs_b', at: { normal: [[2, 7], [4, 7], [6, 7]] } }, { one: 'sting_a', zero: 'sting_b', at: { normal: [[4, 8]] } }],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 41, rows: 41, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 4.3 } },
});
