// Candle v2 (tile rule: v1 had 2-px lines + object words): every part tile = ONE proven black shape (turtle / shield / bicycle paint code), words = the shape names, the intro names each part's two shapes. Members, slots, boards = v1.
// Candle (relative; batch 3, on the FACE template: 3 parts at 2 mirrored slots, all 9 normal shifts valid, long shift left-right only): one large candle on a plain pink board. Four binary
// SHAPE parts (no counts), each a big BLACK shape on the plain field; the eight members use the eight even-parity part sets (as Face), so every informative
// look halves what is left. LURE: the wax-drip part has TWO slots side by side in the middle (big, 1 bit); the three telling parts sit far away. The whole picture may sit one site higher, lower, left or right (hidden placement). Neutral names.
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame } from './w3-parts.js';
import { partList } from './w3-text.js';
const F = s => rect(s, 0, 0, 15, 15, C.pink);
defineTile('candle:field', paint(s => F(s)));
defineTile('candle:flame_a', paint(s => { F(s); rect(s, 6, 1, 9, 14, C.black); rect(s, 1, 6, 14, 9, C.black); line(s, 3, 3, 12, 12, 3, C.black); line(s, 12, 3, 3, 12, 3, C.black); }));   // a black eight-point star
defineTile('candle:flame_b', paint(s => { F(s); rect(s, 2, 2, 13, 13, C.black); }));   // a black square
defineTile('candle:drip_a', paint(s => { F(s); ring(s, 7.5, 7.5, 3.5, 6.5, C.black); }));   // a black ring
defineTile('candle:drip_b', paint(s => { F(s); for (let y = 1; y < 15; y++) { const h = 7 - Math.abs(y - 7.5); rect(s, Math.round(7.5 - h), y, Math.round(7.5 + h), y, C.black); } }));   // a black diamond
defineTile('candle:handle_a', paint(s => { F(s); line(s, 1, 3, 7, 12, 3, C.black); line(s, 14, 3, 8, 12, 3, C.black); }));   // a black V shape
defineTile('candle:handle_b', paint(s => { F(s); disc(s, 7.5, 7.5, 6, C.black); }));   // a black disc
defineTile('candle:feet_a', paint(s => { F(s); for (let y = 1; y < 15; y++) { const h = Math.round((y - 1) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, C.black); } }));   // a black triangle pointing up
defineTile('candle:feet_b', paint(s => { F(s); rect(s, 0, 5, 15, 10, C.black); }));   // a wide flat black bar
const MEMBERS = { boreas: '1111', eurus: '1100', notus: '1010', zephyr: '1001', sirocco: '0110', mistral: '0101', chinook: '0011', foehn: '0000' };
const WORDS = [["a black eight-point star", "a black square"], ["a black ring", "a black diamond"], ["a black V shape", "a black disc"], ["a black triangle pointing up", "a wide flat black bar"]];
export const CANDLE = partsGame({
  prefix: 'candle', background: 'field', name: 'Candle', kind: 'relative', picture: true, members: MEMBERS,
  intro: { normal: "One large candle in a holder stands on a plain pink board. Four parts sit in their places, each drawn as one big black shape filling one site (a part that sits at two or more sites shows the same shape at each of them): its flame at the top: a black eight-point star or a black square; two wax drips side by side in the middle: a black ring or a black diamond; the two holder handles at the far left and far right: a black V shape or a black disc; the two holder feet at the bottom: a black triangle pointing up or a wide flat black bar. Every other site shows plain pink. The whole candle may sit one site higher, lower, left or right, or one site diagonally.", long: "One very large candle in a holder stands on a plain pink board, drawn wide with wide plain pink between the parts. Four parts sit in their places, each drawn as one big black shape filling one site (a part that sits at two or more sites shows the same shape at each of them): its flame at the top: a black eight-point star or a black square; two wax drips side by side in the middle: a black ring or a black diamond; the two holder handles at the far left and far right: a black V shape or a black disc; the two holder feet at the bottom: a black triangle pointing up or a wide flat black bar. Every other site shows plain pink. The whole candle may sit two sites further left or right." },
  ask: 'Which candle is it?',
  setText: `one of eight candles, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [{ one: 'flame_a', zero: 'flame_b', at: { normal: [[4, 1]] } }, { one: 'drip_a', zero: 'drip_b', at: { normal: [[4, 4], [5, 4, true]] } }, { one: 'handle_a', zero: 'handle_b', at: { normal: [[1, 6], [8, 6, true]] } }, { one: 'feet_a', zero: 'feet_b', at: { normal: [[2, 8], [7, 8, true]] } }],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 40, rows: 40, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 4.1 } },
});
