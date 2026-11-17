// Turtle (relative; BATCH 2, variant of the face template: 5 parts).
// Its name follows from 5 SHAPE parts (black on a plain field). The rules name every part, both shapes and the hidden shift.
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame, partList } from './rb8-parts.js';
const K = C.black, FIELD = C.blue;
const bg = s => rect(s, 0, 0, 15, 15, FIELD);
defineTile('turtle:field', paint(bg));
defineTile('turtle:head_a', paint(s => { bg(s); disc(s, 7.5, 7.5, 6, K); }));   // a black disc
defineTile('turtle:head_b', paint(s => { bg(s); for (let y = 1; y < 15; y++) { const h = Math.round((y - 1) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, K); } }));   // a black triangle pointing up
defineTile('turtle:front_a', paint(s => { bg(s); rect(s, 0, 5, 15, 10, K); }));   // a wide flat black bar
defineTile('turtle:front_b', paint(s => { bg(s); line(s, 1, 3, 7, 12, 3, K); line(s, 14, 3, 8, 12, 3, K); }));   // a black V shape
defineTile('turtle:shell_a', paint(s => { bg(s); ring(s, 7.5, 7.5, 3.5, 6.5, K); }));   // a black ring
defineTile('turtle:shell_b', paint(s => { bg(s); for (let y = 1; y < 15; y++) { const h = 7 - Math.abs(y - 7.5); rect(s, Math.round(7.5 - h), y, Math.round(7.5 + h), y, K); } }));   // a black diamond
defineTile('turtle:back_a', paint(s => { bg(s); rect(s, 2, 2, 13, 13, K); }));   // a black square
defineTile('turtle:back_b', paint(s => { bg(s); rect(s, 5, 0, 10, 15, K); }));   // a tall thin black bar
defineTile('turtle:tail_a', paint(s => { bg(s); for (let y = 1; y < 15; y++) { const h = Math.round((14 - y) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, K); } }));   // a black triangle pointing down
defineTile('turtle:tail_b', paint(s => { bg(s); line(s, 1, 1, 14, 14, 3, K); line(s, 14, 1, 1, 14, 3, K); }));   // a black X
const MEMBERS = {"brisa": "00000", "calder": "00101", "dovan": "01010", "eskel": "01111", "fyra": "10011", "gorm": "10110", "halla": "11001", "imbra": "11100"};
const WORDS = [["a black disc", "a black triangle pointing up"], ["a wide flat black bar", "a black V shape"], ["a black ring", "a black diamond"], ["a black square", "a tall thin black bar"], ["a black triangle pointing down", "a black X"]];
export const TURTLE = partsGame({
  background: 'turtle:field', name: 'Turtle', kind: 'relative', picture: true, members: MEMBERS,
  intro: { normal: "One large turtle fills the board, seen from above with its head at the top. Five parts sit in their places, each drawn in black; a part that sits at two or more sites shows the same shape at each of them: its head at the top middle: a black disc or a black triangle pointing up; its two front legs on the upper left and upper right: a wide flat black bar or a black V shape; the middle of its shell at the centre: a black ring or a black diamond; its two back legs on the lower left and lower right: a black square or a tall thin black bar; its tail at the bottom middle: a black triangle pointing down or a black X. Every other site shows plain blue water. The whole turtle may sit one site higher, lower, left or right.", long: "One large turtle fills the board, seen from above with its head at the top, drawn very large with wide plain space between its parts. Five parts sit in their places, each drawn in black; a part that sits at two or more sites shows the same shape at each of them: its head at the top middle: a black disc or a black triangle pointing up; its two front legs on the upper left and upper right: a wide flat black bar or a black V shape; the middle of its shell at the centre: a black ring or a black diamond; its two back legs on the lower left and lower right: a black square or a tall thin black bar; its tail at the bottom middle: a black triangle pointing down or a black X. Every other site shows plain blue water. The whole turtle may sit two sites further left or right." },
  ask: 'Which turtle is it?',
  setText: `one of eight turtles, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [
    { one: 'turtle:head_a', zero: 'turtle:head_b', at: { normal: [[4, 1]] } },
    { one: 'turtle:front_a', zero: 'turtle:front_b', at: { normal: [[1, 3], [8, 3]] } },
    { one: 'turtle:shell_a', zero: 'turtle:shell_b', at: { normal: [[4, 5]] } },
    { one: 'turtle:back_a', zero: 'turtle:back_b', at: { normal: [[1, 7], [8, 7]] } },
    { one: 'turtle:tail_a', zero: 'turtle:tail_b', at: { normal: [[5, 8]] } },
  ],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 36, rows: 35, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 4.0 } },
});
