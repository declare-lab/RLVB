// Recipe (join: "if a repair fails, build animals or recipe (picture) instead"): one plate of food fills the board. Its dish
// follows from three ingredient places on the plate: the cheese place, the tomato place and the bread place. Each place shows its ingredient, or an empty grey ring when the
// dish has none of it. The 8 dishes are the 8 ways to join the three ingredients, and every name carries its meaning (pizza = cheese + tomato + bread; toast = bread only ...).
// The plate may sit one site higher, lower, left or right (hidden). Each ingredient look halves the dishes that are left: 3 informative looks to certainty, no big look.
// v7 (TABLE complete .55, a little harder): LURE = a side plate in the bottom left corner of the board shows one of the three
// ingredients at random (it does not move with the plate and does not count); long 25x25 k 3 -> 27x27 k 4 (B8 long 40-50). Tiles, names and code unchanged (the P1 holds).
// (A shift +-2 draft failed the NOTE one-step bar .84 and put the normal reference at 19.9: recipe-v7-shift2-draft.js.txt.)
// v7.1 (after probe22 P2 .82: the lure made the model doubt every reading): the side plate has its OWN tiles
// (recipe:side_cheese / side_tomato / side_bread: a thick black rim + a small yellow / red / brown piece), and the rules say the ingredient places never have a black rim.
// Everything else = v7.0 (code, names, slots, boards, nuisances).
import { defineTile, paint, rect, disc, line, C } from './tiles.js';
const MEMBERS = {   // cheese, tomato, bread
  'steamed rice': '000', 'cheese platter': '100', 'tomato salad': '010', toast: '001',
  'caprese salad': '110', 'grilled cheese': '101', bruschetta: '011', pizza: '111',
};
const WORD = [['cheese', 'no cheese'], ['tomato', 'no tomato'], ['bread', 'no bread']];   // the code is stated in the rules (as face), not left to food knowledge
const says = Object.entries(MEMBERS).map(([m, b]) => `${m} = ${[...b].map((v, k) => WORD[k][v === '1' ? 0 : 1]).join(', ')}`).join('; ');
// Tiles: the plate (plain white ground), the empty place (a grey ring), and the three ingredients (large shapes).
defineTile('recipe:plate', paint(s => { rect(s, 0, 0, 15, 15, C.white); }));   // v6.1.1: plain white, no centre dot (probe3: the dot read as an empty ring)
defineTile('recipe:empty', paint(s => { for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const d = Math.hypot(x - 7.5, y - 7.5); if (d >= 5 && d <= 6.5) s(x, y, C.grey); } }));
defineTile('recipe:cheese', paint(s => { for (let y = 3; y <= 12; y++) for (let x = 2; x <= 13; x++) if (x - 2 <= (y - 3) * 1.2) s(x, y, C.yellow); disc(s, 6, 10, 1.2, C.orange); disc(s, 4, 7, 0.9, C.orange); }));
defineTile('recipe:tomato', paint(s => { disc(s, 7.5, 8, 6, C.red); disc(s, 5.5, 7, 1.2, C.pink); disc(s, 9.5, 9, 1.2, C.pink); rect(s, 7, 1, 8, 3, C.green); }));
// v7.1 side plate tiles: a thick black rim and a small ingredient piece (yellow cheese, red tomato, brown bread), so a side-plate site never looks like a place.
for (const [n, c] of [['cheese', C.yellow], ['tomato', C.red], ['bread', C.brown]]) defineTile(`recipe:side_${n}`, paint(s => { for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const d = Math.hypot(x - 7.5, y - 7.5); if (d >= 5 && d <= 7.5) s(x, y, C.black); } disc(s, 7.5, 7.5, 3, c); }));
defineTile('recipe:bread', paint(s => { rect(s, 2, 4, 13, 14, C.brown); disc(s, 5, 4.5, 3, C.brown); disc(s, 10, 4.5, 3, C.brown); for (const x of [5, 8, 11]) line(s, x - 1, 5, x + 1, 12, 1, C.black); }));   // v6.1.2: a brown loaf with black score lines, NO yellow (probe s11: the yellow centre read as cheese / plate)
const PARTS = [['recipe:cheese', [2, 3]], ['recipe:tomato', [6, 3]], ['recipe:bread', [4, 6]]];   // normal slots (x, y); v7: cheese / tomato one row lower (C4: v6.1.2 had r .80 with fish)
const BOARDS = { normal: { cols: 9, rows: 9, budget: 35, shift: 1, k: 1, o: 0 }, long: { cols: 27, rows: 27, budget: 65, shift: 0, k: 4, o: 2 } };
export const RECIPE = {
  name: 'Recipe', kind: 'join', picture: true,
  members: Object.keys(MEMBERS),
  intro: {
    normal: 'One plate of food fills the board. Three ingredient places sit on the plate: the cheese place at the upper left, the tomato place at the upper right and the bread place near the bottom middle. Each place shows its ingredient, or an empty grey ring when the dish has none of it. A small side plate with a thick black rim sits in the bottom left corner of the board; it shows a small piece of cheese (yellow), tomato (red) or bread (brown), at random, and it does not count. The ingredient places never have a black rim. Every other site shows plain plate. The plate may sit one site higher, lower, left or right; the side plate does not move.',
    long: 'One large plate of food fills the board, drawn wide: the cheese place at the upper left, the tomato place at the upper right and the bread place near the bottom middle, with wide plain plate between them. Each place shows its ingredient, or an empty grey ring when the dish has none of it. A small side plate with a thick black rim sits in the bottom left corner of the board; it shows a small piece of cheese (yellow), tomato (red) or bread (brown), at random, and it does not count. The ingredient places never have a black rim. The large plate always sits in the middle of the board.',
  },
  ask: 'Which dish is on the plate?',
  setText: `one of eight dishes, each with its own ingredients: ${says}.`,
  boards: Object.fromEntries(Object.entries(BOARDS).map(([k, b]) => [k, { cols: b.cols, rows: b.rows, budget: b.budget }])),
  nuisances(board) { const b = BOARDS[board], out = []; for (let dy = -b.shift; dy <= b.shift; dy++) for (let dx = -b.shift; dx <= b.shift; dx++) for (const side of [0, 1, 2]) out.push({ dx: dx * b.k, dy: dy * b.k, side }); return out; },   // side: the side-plate ingredient (the lure)
  layout(member, nu, board) {
    const b = BOARDS[board], bits = MEMBERS[member], sites = Array(b.cols * b.rows).fill('recipe:plate');
    sites[(b.rows - 1) * b.cols] = ['recipe:side_cheese', 'recipe:side_tomato', 'recipe:side_bread'][nu.side];   // the side plate (bottom left corner, fixed; v7.1: its own black-rimmed tiles)
    for (let j = 0; j < 3; j++) {
      const [x0, y0] = PARTS[j][1], X = x0 * b.k + b.o + nu.dx, Y = y0 * b.k + b.o + nu.dy;
      if (X < 0 || Y < 0 || X >= b.cols || Y >= b.rows) return null;
      sites[Y * b.cols + X] = bits[j] === '1' ? PARTS[j][0] : 'recipe:empty';
    }
    return sites;
  },
};
