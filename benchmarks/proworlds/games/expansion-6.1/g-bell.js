// Bell v2 (tile rule: v1 had 2-px lines + object words): every part tile = ONE proven black shape (turtle / shield / bicycle paint code), words = the shape names, the intro names each part's two shapes. Members, slots, boards = v1.
// Bell (relative; batch 3, on the FACE template: 3 parts at 2 mirrored slots, all 9 normal shifts valid, long shift left-right only): one large bell on a plain green board. Four binary
// SHAPE parts (no counts), each a big BLACK shape on the plain field; the eight members use the eight even-parity part sets (as Face), so every informative
// look halves what is left. The whole picture may sit one site higher, lower, left or right (hidden placement). Neutral names.
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame } from './w3-parts.js';
import { partList } from './w3-text.js';
const F = s => rect(s, 0, 0, 15, 15, C.green);
defineTile('bell:field', paint(s => F(s)));
defineTile('bell:loop_a', paint(s => { F(s); disc(s, 7.5, 7.5, 6, C.black); }));   // a black disc
defineTile('bell:loop_b', paint(s => { F(s); for (let y = 1; y < 15; y++) { const h = Math.round((y - 1) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, C.black); } }));   // a black triangle pointing up
defineTile('bell:rim_a', paint(s => { F(s); for (let y = 1; y < 15; y++) { const h = Math.round((14 - y) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, C.black); } }));   // a black triangle pointing down
defineTile('bell:rim_b', paint(s => { F(s); rect(s, 2, 2, 13, 13, C.black); }));   // a black square
defineTile('bell:band_a', paint(s => { F(s); rect(s, 5, 0, 10, 15, C.black); }));   // a tall thin black bar
defineTile('bell:band_b', paint(s => { F(s); for (const y of [1, 6, 11]) rect(s, 0, y, 15, y + 3, C.black); }));   // three flat black stripes
defineTile('bell:clap_a', paint(s => { F(s); rect(s, 6, 1, 9, 14, C.black); rect(s, 1, 6, 14, 9, C.black); }));   // a black plus sign
defineTile('bell:clap_b', paint(s => { F(s); for (let y = 1; y < 15; y++) { const h = 7 - Math.abs(y - 7.5); rect(s, Math.round(7.5 - h), y, Math.round(7.5 + h), y, C.black); } }));   // a black diamond
const MEMBERS = { aria: '1111', cadenza: '1100', coda: '1010', etude: '1001', fermata: '0110', largo: '0101', nocturne: '0011', sonata: '0000' };
const WORDS = [["a black disc", "a black triangle pointing up"], ["a black triangle pointing down", "a black square"], ["a tall thin black bar", "three flat black stripes"], ["a black plus sign", "a black diamond"]];
export const BELL = partsGame({
  prefix: 'bell', background: 'field', name: 'Bell', kind: 'relative', picture: true, members: MEMBERS,
  intro: { normal: "One large bell hangs on a plain green board. Four parts sit in their places, each drawn as one big black shape filling one site (a part that sits at two or more sites shows the same shape at each of them): its hanging loop at the top: a black disc or a black triangle pointing up; the two ends of its rim at the lower left and lower right: a black triangle pointing down or a black square; two decorated bands side by side in the middle: a tall thin black bar or three flat black stripes; its clapper at the bottom: a black plus sign or a black diamond. Every other site shows plain green. The whole bell may sit one site higher, lower, left or right, or one site diagonally.", long: "One very large bell hangs on a plain green board, drawn wide with wide plain green between the parts. Four parts sit in their places, each drawn as one big black shape filling one site (a part that sits at two or more sites shows the same shape at each of them): its hanging loop at the top: a black disc or a black triangle pointing up; the two ends of its rim at the lower left and lower right: a black triangle pointing down or a black square; two decorated bands side by side in the middle: a tall thin black bar or three flat black stripes; its clapper at the bottom: a black plus sign or a black diamond. Every other site shows plain green. The whole bell may sit two sites further left or right." },
  ask: 'Which bell is it?',
  setText: `one of eight bells, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [{ one: 'loop_a', zero: 'loop_b', at: { normal: [[4, 1]] } }, { one: 'rim_a', zero: 'rim_b', at: { normal: [[2, 7], [6, 7, true]] } }, { one: 'band_a', zero: 'band_b', at: { normal: [[3, 4], [5, 4, true]] } }, { one: 'clap_a', zero: 'clap_b', at: { normal: [[4, 8]] } }],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 40, rows: 40, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 4.2 } },
});
