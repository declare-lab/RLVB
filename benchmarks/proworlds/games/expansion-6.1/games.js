// PRO Worlds assembly 6.0: the game registry. Every file g-<id>.js in this
// folder exports ONE game spec (see API.md); this module loads them all, so two
// writers never edit the same registry file. The key of a game is its <id>.
// name, kind ('assemble' | 'order' | 'symmetry' | 'relative' | 'path' | 'join' | 'clues'),
// picture, members (8-12 names), intro, ask, setText,
// boards: {normal: {cols, rows, budget}, long: {cols, rows, budget}},
// nuisances(board) -> a finite list, layout(member, nuisance, board) -> cols*rows tile names.
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
export const GAMES = {};
for (const f of readdirSync(here).filter(f => /^g-[a-z0-9]+\.js$/.test(f)).sort()) {
  const mod = await import(join(here, f)), specs = Object.values(mod);
  if (specs.length !== 1) throw Error(`${f}: export exactly one game spec`);
  GAMES[f.slice(2, -3)] = specs[0];
}
export function registerGame(id, spec) { if (GAMES[id]) throw Error(`Game ${id} exists`); GAMES[id] = spec; }
