// Sailboat (assemble, insurance batch 4): 4 binary picture parts (3 at 2 slots, 1 at 1 slot) on the face template,
// built ONLY from proven tile pairs (exact paint code of turtle / shield / bicycle; tile rule).
import { defineTile, paint, rect, disc, ring, line, C } from './tiles.js';
import { partsGame, partList } from './rb8-parts.js';
const K = C.black, FIELD = C.blue;
const bg = s => rect(s, 0, 0, 15, 15, FIELD);
defineTile('sailboat:field', paint(bg));
defineTile('sailboat:mast_star', paint(s => { bg(s); rect(s,6,1,9,14,K);rect(s,1,6,14,9,K);line(s,3,3,12,12,3,K);line(s,12,3,3,12,3,K); }));   // a black eight-point star
defineTile('sailboat:mast_square', paint(s => { bg(s); rect(s,2,2,13,13,K); }));   // a black square
defineTile('sailboat:sail_tri_down', paint(s => { bg(s); for(let y=1;y<15;y++){const h=Math.round((14-y)/13*6.5);rect(s,7-h,y,8+h,y,K);}; }));   // a black triangle pointing down
defineTile('sailboat:sail_xmark', paint(s => { bg(s); line(s,1,1,14,14,3,K);line(s,14,1,1,14,3,K); }));   // a black X
defineTile('sailboat:hull_disc', paint(s => { bg(s); disc(s,7.5,7.5,6,K); }));   // a black disc
defineTile('sailboat:hull_tri_up', paint(s => { bg(s); for(let y=1;y<15;y++){const h=Math.round((y-1)/13*6.5);rect(s,7-h,y,8+h,y,K);}; }));   // a black triangle pointing up
defineTile('sailboat:flag_plus', paint(s => { bg(s); rect(s,6,1,9,14,K);rect(s,1,6,14,9,K); }));   // a black plus sign
defineTile('sailboat:flag_diamond', paint(s => { bg(s); for(let y=1;y<15;y++){const h=7-Math.abs(y-7.5);rect(s,Math.round(7.5-h),y,Math.round(7.5+h),y,K);}; }));   // a black diamond
const MEMBERS = {"nimbo": "0110", "pharo": "1100", "quilla": "0101", "rennet": "1111", "sabla": "0000", "torvi": "1001", "ulmo": "1010", "vessa": "0011"};
const WORDS = [["the mast marks show a black eight-point star", "the mast marks show a black square"], ["the sail marks show a black triangle pointing down", "the sail marks show a black X"], ["the hull marks show a black disc", "the hull marks show a black triangle pointing up"], ["the flag shows a black plus sign", "the flag shows a black diamond"]];
export const GAME = partsGame({
  background: 'sailboat:field', name: 'Sailboat', kind: 'assemble', picture: true, members: MEMBERS,
  intro: {
    normal: 'One large sailboat fills the board on a plain blue field. Its parts are large black shapes: two mast marks in the middle column, one near the top and one a little above the middle; two sail marks at the left and right, below the middle; two hull marks near the bottom, left and right of the middle; one flag at the right edge near the top. Every other site is the plain blue field. The whole picture may sit one site higher, lower, left or right.',
    long: 'One very large sailboat fills the board on a plain blue field, drawn wide with wide plain field between its parts. Its parts are large black shapes: two mast marks in the middle column, one near the top and one a little above the middle; two sail marks at the left and right, below the middle; two hull marks near the bottom, left and right of the middle; one flag at the right edge near the top. Every other site is the plain blue field. The whole picture may sit one site further left or right.',
  },
  ask: 'Which sailboat is it?',
  setText: `one of eight sailboats, each with its own parts: ${partList(MEMBERS, WORDS)}.`,
  parts: [{ one: 'sailboat:mast_star', zero: 'sailboat:mast_square', at: { normal: [[4, 1], [4, 4]] } }, { one: 'sailboat:sail_tri_down', zero: 'sailboat:sail_xmark', at: { normal: [[1, 6], [7, 6]] } }, { one: 'sailboat:hull_disc', zero: 'sailboat:hull_tri_up', at: { normal: [[2, 8], [6, 8]] } }, { one: 'sailboat:flag_plus', zero: 'sailboat:flag_diamond', at: { normal: [[8, 2]] } }],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 36, rows: 35, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 4.0 } },
});
