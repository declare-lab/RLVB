// Barn (assemble, batch screening on the FACE template): 4 binary picture parts, 3 of them at 2 symmetric slots,
// each part a large black SHAPE on a plain yellow field; hidden shift +-1 on both axes; 8 members use 8 of the 16 part sets (as face).
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame, partList } from './rb8-parts.js';
const FIELD = C.yellow;
defineTile('barn:field', paint(s => rect(s, 0, 0, 15, 15, FIELD)));
defineTile('barn:roof_tri_up', paint(s => { rect(s, 0, 0, 15, 15, FIELD); for(let y=2;y<=13;y++){const w=Math.round((y-2)*0.5);rect(s,7-w,y,8+w,y,C.black);}; }));
defineTile('barn:roof_hbar', paint(s => { rect(s, 0, 0, 15, 15, FIELD); rect(s,1,5,14,10,C.black); }));   // v1.1 (tile rule): the thin arch (19% ink) -> the proven wide flat bar (bicycle)
defineTile('barn:loft_cross', paint(s => { rect(s, 0, 0, 15, 15, FIELD); rect(s,6,1,9,14,C.black);rect(s,1,6,14,9,C.black); }));
defineTile('barn:loft_diamond', paint(s => { rect(s, 0, 0, 15, 15, FIELD); for(let y=1;y<=14;y++){const w=Math.round(6.5-Math.abs(y-7.5));rect(s,7-w,y,8+w,y,C.black);}; }));   // v1.1: plus vs X differed only in angle -> the proven plus vs diamond (bicycle)
defineTile('barn:door_square', paint(s => { rect(s, 0, 0, 15, 15, FIELD); rect(s,2,2,13,13,C.black); }));
defineTile('barn:door_tri_down', paint(s => { rect(s, 0, 0, 15, 15, FIELD); for(let y=2;y<=13;y++){const w=Math.round((13-y)*0.5);rect(s,7-w,y,8+w,y,C.black);}; }));   // v1.2: the ladder had 2-px lines -> square vs triangle down (proven pair, shield)
defineTile('barn:weather_vbar', paint(s => { rect(s, 0, 0, 15, 15, FIELD); rect(s,5,0,10,15,C.black); }));   // v1.3: probe80 misread the vane star / disc 6 of 12 -> the proven tall thin bar vs three flat stripes (shield code)
defineTile('barn:weather_stripes', paint(s => { rect(s, 0, 0, 15, 15, FIELD); for(const y of [1,6,11])rect(s,0,y,15,y+3,C.black); }));
const MEMBERS = {"jace": "0110", "kira": "1100", "lyle": "0101", "maeve": "1111", "nils": "0000", "opal": "1001", "penn": "1010", "rhys": "0011"};
const WORDS = [["the roof marks show a black triangle pointing up", "the roof marks show a wide flat black bar"], ["the loft marks show a black plus sign", "the loft marks show a black diamond"], ["the door marks show a black square", "the door marks show a black triangle pointing down"], ["the weather vane mark shows a tall thin black bar", "the weather vane mark shows three flat black stripes"]];
export const GAME = partsGame({
  background: 'barn:field', name: 'Barn', kind: 'assemble', picture: true, members: MEMBERS,
  intro: {
    normal: 'One large barn fills the board on a plain yellow field. Its parts are large black shapes: two roof marks in the top row at the left, two loft marks one above the other at the right edge, two door marks in the lower half at the left and middle, and one weather vane mark near the bottom right. Every other site is the plain yellow field. The whole picture may sit one site higher, lower, left or right.',
    long: 'One very large barn fills the board on a plain yellow field, drawn wide with wide plain field between its parts. Its parts are large black shapes: two roof marks in the top row at the left, two loft marks one above the other at the right edge, two door marks in the lower half at the left and middle, and one weather vane mark near the bottom right. Every other site is the plain yellow field. The whole picture may sit two sites further left or right.',
  },
  ask: 'Which barn is it?',
  setText: `one of eight barns, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [{ one: 'barn:roof_tri_up', zero: 'barn:roof_hbar', at: { normal: [[1, 1], [4, 1]] } }, { one: 'barn:loft_cross', zero: 'barn:loft_diamond', at: { normal: [[8, 2], [8, 4]] } }, { one: 'barn:door_square', zero: 'barn:door_tri_down', at: { normal: [[2, 6], [5, 6]] } }, { one: 'barn:weather_vbar', zero: 'barn:weather_stripes', at: { normal: [[7, 8]] } }],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 33, rows: 33, budget: 65, shift: 1, shiftY: 0, stride: 2, scale: 3.4 } },
});
