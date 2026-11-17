# PRO Worlds assembly 6.0 — how to write one game

One generic engine (`engine.js`) runs every game. A game is a small spec file `g-<id>.js` that exports one object. `games.js` loads every `g-*.js` file by itself: do not edit it. Look at `g-block.js` (Block piece) as the worked example.

## The spec

```js
export const MY_GAME = {
  name: 'Circuit', // shown in the rules
  kind: 'path', // assemble | order | symmetry | relative | path | join | clues
  picture: false, // true when the hidden whole is one picture
  members: ['red', 'blue', ...], // 8-12 NAMES; the answer is one name; meaning the model already knows
  intro: 'One sentence: what the board shows.',
  ask: 'Which lamp does the wire light?',
  setText: 'one of the lamp colours: red, blue, ... .', // names the set; no pictures, no numbers
  boards: { normal: { cols, rows, budget }, long: { cols, rows, budget } },
  nuisances(board) { return [...]; }, // a FINITE list (placements, small variations)
  layout(member, nuisance, board) { return [...]; } // cols*rows tile names, or null if this pair is impossible
};
```

- **World = (member, nuisance).** The engine enumerates all of them once per board. Keep the number of worlds small enough that a state takes <= 1 s (the test prints `ms`; a few thousand worlds is fine).
- **Each member gets the same prior**, whatever its number of nuisance worlds.
- **The board is a grid of sites; each site is one 16x16 tile.** The window shows one site; one move = one site.
- **Normal and long use the SAME tiles.** Long = the same whole drawn larger or spread wider, so the travel grows (B8) while every site inside the whole still tells about the member. Keep the placement nuisance small (about +-1 site) on both boards.

## Tiles (`tiles.js`)

- Palette: `C.white` (ground), `C.black` (ink), and named colours `C.red, C.blue, C.green, C.yellow, C.orange, C.purple, C.pink, C.brown, C.grey`.
- Ready tiles: `'blank'`, `'block'`, `'dot'`, `'fill:<colour>'`, `'arrow:n|e|s|w'`, `'letter:A'`-`'letter:Z'`, `pathTile(['n','e'], C.red)` (thick line segments between edge centres).
- New tile: `defineTile('my:name', paint(s => { rect(s, x0, y0, x1, y1, C.red); disc(s, cx, cy, r, C.blue); line(s, x0, y0, x1, y1, width, C.black); }))` at module load. Use a `prefix:` for your game's tiles. Large shapes only. **No digits.**

## The rules every game must pass (`node games/assembly/test.mjs <id>`)

| Rule | Check |
|---|---|
| A10 / A3 / A6 | 8-12 members, unique names, no digits, no word shared with another game's set |
| A5 | no counting: the member must not depend on HOW MANY of a tile appear |
| B4 | the first view never decides |
| B5 | the exact belief equals an independent recount |
| B7 | >= 3 informative looks to 95% in every seed; <= 5% of looks give > 1.6 bits; no "needle" look |
| B8 | the reference policy needs 10-20 moves (normal) / 30-50 (long); it is right >= .85 |
| C5 | no pair-only evidence: every site whose tile varies must change the member belief on its own |
| speed | <= 1 s per state |

Run the test after each game. It writes `out/assembly/checks.json` and prints `ALL PASS` or the failing rules. A game counts as done only on `ALL PASS` for both boards.

## Parts of the set

- **Engine and shared parts:** engine, tests, block piece, word tiles, flag stripes, turned emblem, circuit, maze doors, animal parts, recipe, playing card, light ray.
- **Assembly games:** train, bead necklace, folded paper cut, kaleidoscope, sun + shadows, star map, jigsaw scene, face, footprints, scent trail.
- **Name sets must not share words** (the test checks it): beads and kaleidoscope use disjoint pattern names; fruits, flowers, landmarks and animals are all distinct.
