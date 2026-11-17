// Rocket v2 (probe73 fix round, tile rule): every part tile = ONE proven black shape (paint code of turtle / shield / bicycle), words = the shape names; long board = longfix-wiv-v1. (v1:) Rocket (relative; batch screening on the FACE template): one large rocket on a plain white board. Four binary
// SHAPE parts (no counts), each a big BLACK shape on the plain field; the eight members use the eight even-parity part sets (as Face), so every informative
// look halves what is left. LURE: the window part has TWO slots side by side near the centre (big, easy to find, 1 bit); the three telling parts sit at far corners. The whole picture may sit one site higher, lower, left or right (hidden placement). Neutral names.
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame } from './w3-parts.js';
import { partList } from './w3-text.js';
const F = s => rect(s, 0, 0, 15, 15, C.white);
defineTile('rocket:field', paint(s => F(s)));
defineTile('rocket:nose_a', paint(s => { F(s); disc(s, 7.5, 7.5, 6, C.black); }));   // a black disc
defineTile('rocket:nose_b', paint(s => { F(s); for (let y = 1; y < 15; y++) { const h = Math.round((y - 1) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, C.black); } }));   // a black triangle pointing up
defineTile('rocket:windows_a', paint(s => { F(s); rect(s, 0, 5, 15, 10, C.black); }));   // a wide flat black bar
defineTile('rocket:windows_b', paint(s => { F(s); line(s, 1, 3, 7, 12, 3, C.black); line(s, 14, 3, 8, 12, 3, C.black); }));   // a black V shape
defineTile('rocket:fins_a', paint(s => { F(s); rect(s, 6, 1, 9, 14, C.black); rect(s, 1, 6, 14, 9, C.black); line(s, 3, 3, 12, 12, 3, C.black); line(s, 12, 3, 3, 12, 3, C.black); }));   // a black eight-point star
defineTile('rocket:fins_b', paint(s => { F(s); ring(s, 7.5, 7.5, 3.5, 6.5, C.black); }));   // a black ring
defineTile('rocket:flame_a', paint(s => { F(s); rect(s, 5, 0, 10, 15, C.black); }));   // a tall thin black bar
defineTile('rocket:flame_b', paint(s => { F(s); for (const y of [1, 6, 11]) rect(s, 0, y, 15, y + 3, C.black); }));   // three flat black stripes
const MEMBERS = { arrow: '1111', comet: '1100', dart: '1010', falcon: '1001', hawk: '0110', lance: '0101', spark: '0011', swift: '0000' };
const WORDS = [["a black disc", "a black triangle pointing up"], ["a wide flat black bar", "a black V shape"], ["a black eight-point star", "a black ring"], ["a tall thin black bar", "three flat black stripes"]];
export const ROCKET = partsGame({
  prefix: 'rocket', background: 'field', name: 'Rocket', kind: 'relative', picture: true, members: MEMBERS,
  intro: { normal: "One large rocket lies across a plain white board. Four parts sit in their places, each drawn as one big black shape filling one site: its nose at the top right corner: a black disc or a black triangle pointing up; its two windows side by side in the middle: a wide flat black bar or a black V shape; its fins at the bottom left corner: a black eight-point star or a black ring; its flame at the bottom right corner: a tall thin black bar or three flat black stripes. Every other site shows plain white. The whole rocket may sit one site higher, lower, left or right.", long: "One very large rocket lies across a plain white board, drawn wide with wide plain white between the parts. Four parts sit in their places, each drawn as one big black shape filling one site: its nose at the top right corner: a black disc or a black triangle pointing up; its two windows side by side in the middle: a wide flat black bar or a black V shape; its fins at the bottom left corner: a black eight-point star or a black ring; its flame at the bottom right corner: a tall thin black bar or three flat black stripes. Every other site shows plain white. The whole rocket may sit two sites further left or right." },
  ask: 'Which rocket is it?',
  setText: `one of eight rockets, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [{ one: 'nose_a', zero: 'nose_b', at: { normal: [[8, 1]] } }, { one: 'windows_a', zero: 'windows_b', at: { normal: [[4, 4], [5, 4]] } }, { one: 'fins_a', zero: 'fins_b', at: { normal: [[1, 8]] } }, { one: 'flame_a', zero: 'flame_b', at: { normal: [[8, 8]] } }],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 26, rows: 26, budget: 65, shift: 1, shiftY: 0, stride: 2, scale: 2.6 } },
});
