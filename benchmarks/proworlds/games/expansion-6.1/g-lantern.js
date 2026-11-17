// Lantern (assemble; BATCH 2, variant of the face template: 5 parts).
// Its name follows from 5 SHAPE parts (black on a plain field). The rules name every part, both shapes and the hidden shift.
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame, partList } from './rb8-parts.js';
const K = C.black, FIELD = C.orange;
const bg = s => rect(s, 0, 0, 15, 15, FIELD);
defineTile('lantern:field', paint(bg));
defineTile('lantern:handle_a', paint(s => { bg(s); rect(s, 2, 2, 13, 13, K); }));   // a black square
defineTile('lantern:handle_b', paint(s => { bg(s); for (let y = 1; y < 15; y++) { const h = Math.round((y - 1) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, K); } }));   // a black triangle pointing up
defineTile('lantern:hooks_a', paint(s => { bg(s); rect(s, 5, 0, 10, 15, K); }));   // a tall thin black bar
defineTile('lantern:hooks_b', paint(s => { bg(s); for (const y of [1, 6, 11]) rect(s, 0, y, 15, y + 3, K); }));   // three flat black stripes
defineTile('lantern:flame_a', paint(s => { bg(s); ring(s, 7.5, 7.5, 3.5, 6.5, K); }));   // a black ring
defineTile('lantern:flame_b', paint(s => { bg(s); rect(s, 6, 1, 9, 14, K); rect(s, 1, 6, 14, 9, K); line(s, 3, 3, 12, 12, 3, K); line(s, 12, 3, 3, 12, 3, K); }));   // a black eight-point star
defineTile('lantern:sides_a', paint(s => { bg(s); for (let y = 3; y < 15; y++) for (let x = 0; x < 16; x++) if ((x - 7.5) ** 2 + (y - 14) ** 2 <= 56) s(x, y, K); }));   // a black dome (flat side down)
defineTile('lantern:sides_b', paint(s => { bg(s); for (let y = 1; y < 13; y++) for (let x = 0; x < 16; x++) if ((x - 7.5) ** 2 + (y - 1) ** 2 <= 56) s(x, y, K); }));   // a black bowl (flat side up)
defineTile('lantern:base_a', paint(s => { bg(s); rect(s, 0, 5, 15, 10, K); }));   // a wide flat black bar
defineTile('lantern:base_b', paint(s => { bg(s); line(s, 1, 3, 7, 12, 3, K); line(s, 14, 3, 8, 12, 3, K); }));   // a black V shape
const MEMBERS = {"rello": "00000", "sanvi": "00101", "tobin": "01010", "ulmar": "01111", "vesna": "10011", "wilko": "10110", "yaren": "11001", "zorla": "11100"};
const WORDS = [["a black square", "a black triangle pointing up"], ["a tall thin black bar", "three flat black stripes"], ["a black ring", "a black eight-point star"], ["a black dome (flat side down)", "a black bowl (flat side up)"], ["a wide flat black bar", "a black V shape"]];
export const LANTERN = partsGame({
  background: 'lantern:field', name: 'Lantern', kind: 'assemble', picture: true, members: MEMBERS,
  intro: { normal: "One large lantern hangs on an orange wall, with its handle at the top. Five parts sit in their places, each drawn in black; a part that sits at two or more sites shows the same shape at each of them: its handle at the top middle: a black square or a black triangle pointing up; its two hooks on the left and right below the handle: a tall thin black bar or three flat black stripes; its flame window at the centre: a black ring or a black eight-point star; its two side panes on the lower left and lower right: a black dome (flat side down) or a black bowl (flat side up); its base at the bottom middle: a wide flat black bar or a black V shape. Every other site shows plain orange wall. The whole lantern may sit one site higher, lower, left or right.", long: "One large lantern hangs on an orange wall, with its handle at the top, drawn very large with wide plain space between its parts. Five parts sit in their places, each drawn in black; a part that sits at two or more sites shows the same shape at each of them: its handle at the top middle: a black square or a black triangle pointing up; its two hooks on the left and right below the handle: a tall thin black bar or three flat black stripes; its flame window at the centre: a black ring or a black eight-point star; its two side panes on the lower left and lower right: a black dome (flat side down) or a black bowl (flat side up); its base at the bottom middle: a wide flat black bar or a black V shape. Every other site shows plain orange wall. The whole lantern may sit two sites further left or right." },
  ask: 'Which lantern is it?',
  setText: `one of eight lanterns, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [
    { one: 'lantern:handle_a', zero: 'lantern:handle_b', at: { normal: [[5, 1]] } },
    { one: 'lantern:hooks_a', zero: 'lantern:hooks_b', at: { normal: [[3, 3], [7, 3]] } },
    { one: 'lantern:flame_a', zero: 'lantern:flame_b', at: { normal: [[5, 5]] } },
    { one: 'lantern:sides_a', zero: 'lantern:sides_b', at: { normal: [[1, 6], [8, 6]] } },
    { one: 'lantern:base_a', zero: 'lantern:base_b', at: { normal: [[4, 8]] } },
  ],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 38, rows: 37, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 4.2 } },
});
