// Hot-air balloon (assemble; batch screening: the FACE template): one large hot-air balloon fills the board.
// Its name follows from four SHAPE parts (black on a plain field; the two values of a part have the same colours), most at two symmetric
// slots. The rules name every part, both shapes and the hidden shift. Codes = the face code set (any 3 parts decide). Budget 35 / 65.
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame, partList } from './rb8-parts.js';
const K = C.black, FIELD = C.blue;
const bg = s => rect(s, 0, 0, 15, 15, FIELD);
defineTile('balloon:field', paint(bg));
defineTile('balloon:vents_a', paint(s => { bg(s); (s => { for (let y = 1; y < 16; y++) { const h = Math.round((y - 1) / 14 * 7); rect(s, 7 - h, y, 8 + h, y, K); } })(s); }));   // a pointed flap (a black triangle)
defineTile('balloon:vents_b', paint(s => { bg(s); (s => { disc(s, 7.5, 7.5, 6.5, K); })(s); }));   // a round flap (a black disc)
defineTile('balloon:panels_a', paint(s => { bg(s); (s => { ring(s, 7.5, 7.5, 3.5, 6.5, K); })(s); }));   // a black ring emblem
defineTile('balloon:panels_b', paint(s => { bg(s); (s => { rect(s, 6, 1, 9, 14, K); rect(s, 1, 6, 14, 9, K); })(s); }));   // a black plus emblem
defineTile('balloon:ropes_a', paint(s => { bg(s); (s => { rect(s, 3, 0, 5, 15, K); rect(s, 10, 0, 12, 15, K); })(s); }));   // two straight ropes
defineTile('balloon:ropes_b', paint(s => { bg(s); (s => { line(s, 1, 0, 14, 15, 3, K); line(s, 14, 0, 1, 15, 3, K); })(s); }));   // two crossed ropes (an X)
defineTile('balloon:basket_a', paint(s => { bg(s); (s => { rect(s, 2, 3, 13, 14, K); })(s); }));   // a square basket
defineTile('balloon:basket_b', paint(s => { bg(s); (s => { for (let y = 1; y < 15; y++) { const h = Math.round((14 - y) / 13 * 6.5); rect(s, 7 - h, y, 8 + h, y, K); } })(s); }));   // a cone basket (a black triangle pointing down)
const MEMBERS = {"aurora": "0110", "breeze": "1100", "cirrus": "0101", "drift": "1111", "ember": "0000", "flare": "1001", "glide": "1010", "haze": "0011"};
const WORDS = [["a pointed flap (a black triangle)", "a round flap (a black disc)"], ["a black ring emblem", "a black plus emblem"], ["two straight ropes", "two crossed ropes (an X)"], ["a square basket", "a cone basket (a black triangle pointing down)"]];
export const BALLOON = partsGame({
  background: 'balloon:field', name: 'Hot-air balloon', kind: 'assemble', picture: true, members: MEMBERS,
  intro: { normal: "One large hot-air balloon fills the board on a blue sky. Four parts sit in their places, each drawn in black and each filling exactly one site: its two upper corners on the left and right: a pointed flap (a black triangle) or a round flap (a black disc); its two middle panels side by side at the centre: a black ring emblem or a black plus emblem; its ropes lower on the left and right: two straight ropes or two crossed ropes (an X); its basket at the bottom middle: a square basket or a cone basket (a black triangle pointing down). Every other site shows plain blue sky. The whole hot-air balloon may sit one site higher, lower, left or right.", long: "One large hot-air balloon fills the board on a blue sky, drawn very large with wide plain space between its parts. Four parts sit in their places, each drawn in black and each filling exactly one site: its two upper corners on the left and right: a pointed flap (a black triangle) or a round flap (a black disc); its two middle panels side by side at the centre: a black ring emblem or a black plus emblem; its ropes lower on the left and right: two straight ropes or two crossed ropes (an X); its basket at the bottom middle: a square basket or a cone basket (a black triangle pointing down). Every other site shows plain blue sky. The whole hot-air balloon may sit two sites further left or right." },
  ask: 'Which hot-air balloon is it?',
  setText: `one of eight hot-air balloons, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [
    { one: 'balloon:vents_a', zero: 'balloon:vents_b', at: { normal: [[2, 1], [7, 1]] } },
    { one: 'balloon:panels_a', zero: 'balloon:panels_b', at: { normal: [[4, 4], [5, 4]] } },
    { one: 'balloon:ropes_a', zero: 'balloon:ropes_b', at: { normal: [[2, 6], [7, 6]] } },
    { one: 'balloon:basket_a', zero: 'balloon:basket_b', at: { normal: [[4, 8]] } },
  ],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 36, rows: 35, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 4.0 } },
});
