# PRO Worlds assembly 6.1: 15 visual parts games (frozen set)

This folder holds the 15 games of the frozen v6.1 set . Every game is visual: the model sees the world only in the picture of one site at a time. It moves the window, collects evidence, and then names the member (for example "Which crab is it?").

## Files

| File | What it is |
|---|---|
| `g-<game>.js` | The 15 frozen game files. `FREEZE.tsv` gives the sha256 (first 8 characters) of each file as it was frozen and measured, and of this copy. |
| `engine.js`, `games.js`, `tiles.js`, `rb8-parts.js`, `w3-*.js`, `API.md` | The shared engine and helper modules that the games import. `games.js` loads every `g-*.js` file in this folder. |
| `test.mjs`, `fixtures.mjs` | The design checks for each game and board, and the rule fixtures. |
| `slack.mjs`, `b7many.mjs`, `perworld.mjs` | Extra checks: budget slack, B7 over many seeds, B7 per public world. |
| `FREEZE.tsv` | The 15 games in pass order (k), file, sha8, and the root of the cell that passed. |
| `TEST-ORDER.txt` | The pass order. Line k = game k. Test seeds = 50,101,300 + 10,000 (k-1) + 0..19. |
| `UNTRAINED.md`, `UNTRAINED.json` | The untrained model results per game (normal and long boards). |
| `SHA256SUMS` | The sha256 of every file in this folder (this copy). |

## This copy

For the public release, the comments and the docs of this folder were cleaned of internal notes, and the run paths were
removed from `UNTRAINED.json`. The program code is unchanged: every `.js` / `.mjs` file parses to the same syntax tree as the
frozen file, and `node test.mjs --fixtures` and the design checks give the same output. The game versions and sha8 values in
`UNTRAINED.*` and in the `sha8` column of `FREEZE.tsv` name the frozen files that were measured.

## The 15 games

| k | game | kind |
|---|---|---|
| 1 | face | relative |
| 2 | jigsaw | assemble |
| 3 | kaleido | symmetry |
| 4 | bird v1.4 | assemble |
| 5 | fish v1.1 | assemble |
| 6 | recipe v7.1 | join |
| 7 | turtle | relative |
| 8 | balloon v1.2 | assemble |
| 9 | rocket v2 | relative |
| 10 | bell v2 | relative |
| 11 | bee v2 | relative |
| 12 | candle v2 | relative |
| 13 | sailboat | assemble |
| 14 | barn v1.3 | assemble |
| 15 | lantern v1.2 | assemble |

## How a game got into the set

A game is in the set only if it passed all of these:

1. **Design checks** (`test.mjs`), normal board: every rule passes (named members, no counting, B7 at least 3 informative looks in every seed, reference right at least .85, the note rule and the two-step rule right at least .85).
2. **Probes** on the untrained model: P1 (b), reading one part at its site, at least .95; P2, combining one reading with the rules, at least .95.
3. **40-play cell** on the untrained model, normal board (budget 35): right 30-79 %, and the complete-evidence share (B9, the exact belief top reaches .95 at any turn) at most .50. A B9 share of .40-.60 gets a third run of 20 plays, and the verdict is on the pool of 60 (fish and barn).

## Run the checks

From this folder (Node 22):

```
node test.mjs face jigsaw kaleido bird fish recipe turtle balloon rocket bell bee candle sailboat barn lantern
node test.mjs --fixtures
node slack.mjs lantern normal 35 200
node b7many.mjs lantern normal 72100000 200
```

Result of a fresh copy of this folder:

- Per game: all normal-board rules pass for all 15 games. One per-game failure: `sailboat/long: B7 min informative looks 2`. This is why sailboat has no long result (see below).
- `--fixtures`: PASS.
- `test.mjs --set` (and `test.mjs` with no game names) also applies the set variety rules. These fail, and we report this: the kinds are assemble 7, relative 6, symmetry 1, join 1 (the rule wants at most 3 of each kind), and the kinds order, path and clues are not in the set.

## Long board

The long board (budget 65) is descriptive only. Training uses the normal board only. These games have no long result, because the long board does not give at least 3 informative looks in every seed: bell, candle, sailboat, barn. The long result of face comes from a long-board copy of face (root L, 8d63f53d) with the same normal worlds. `UNTRAINED.md` marks the long cells that are still running.
