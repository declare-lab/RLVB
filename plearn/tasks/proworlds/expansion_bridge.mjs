// Pro-Worlds (benchmarks/proworlds/games/expansion-6.1): one episode of the frozen engine in each process,
// for expansion_env.py. Run from any folder; one JSON request per line, one JSON reply per line:
// {"op": "new", "id": GAME, "seed": S, "geometry": "normal"} | {"op": "move", "direction": "up"|"down"|"left"|"right"}
// | {"op": "commit", "name": NAME} | {"op": "palette"} | {"op": "close"}. The replies hold only public fields (task, observation)
// and, after commit, the engine's metrics. For a training teacher only (the test never sends them):
// {"op": "worlds", "id": GAME, "geometry": "normal"} gives every world of the game as the engine holds it (member, weight w, public
// class pub, the interned tile of each site); {"op": "state"} gives the episode's private state (the live worlds, the true world
// and member, the site, the moves, the engine's exact belief); {"op": "cases", "id": GAME, "seeds": [...], "geometry": "normal"}
// gives the true world and the start site of each seed; {"op": "keys", "id": GAME, "geometry": "normal"} gives the board-free key
// [member index, shift] of every world.
import readline from 'node:readline';
import * as E from '../../../benchmarks/proworlds/games/expansion-6.1/engine.js';
import {COLORS} from '../../../benchmarks/proworlds/games/expansion-6.1/tiles.js';

let e = null;
const out = o => process.stdout.write(JSON.stringify(o) + '\n');
for await (const line of readline.createInterface({input: process.stdin})) {
  try {
    const a = JSON.parse(line);
    if (a.op === 'close') break;
    if (a.op === 'palette') out({ok: true, result: {colors: COLORS, version: E.VERSION}});
    else if (a.op === 'new') { e = new E.Episode(a.id, a.seed, {geometry: a.geometry || 'normal', budget: a.budget ?? undefined}); out({ok: true, result: {task: e.publicTask(), observation: e.observe()}}); }
    else if (a.op === 'move') out({ok: true, result: {observation: e.move(a.direction)}});
    else if (a.op === 'commit') out({ok: true, result: {metrics: e.commit(a.name)}});
    else if (a.op === 'worlds') { const w = E.worldsOf(a.id, a.geometry || 'normal'); out({ok: true, result: {members: [...E.GAMES[a.id].members], cols: w.cols, rows: w.rows, budget: w.budget, worlds: w.worlds.map(x => ({m: x.member, w: x.w, pub: x.pub, s: Array.from(x.sites)}))}}); }
    else if (a.op === 'state') out({ok: true, result: {live: [...e.live], world: e.case.world, member: e.case.member, site: e.site(), steps: e.steps, belief: e.belief().candidates}});
    else if (a.op === 'keys') {                     // the board-free key of every world, in worldsOf order: [member index, shift]
      const G = E.GAMES[a.id], b = a.geometry || 'normal', nus = G.nuisances(b), ks = [];
      G.members.forEach((m, mi) => nus.forEach(nu => { if (G.layout(m, nu, b)) ks.push([mi, nu]); }));
      if (ks.length !== E.worldsOf(a.id, b).worlds.length) throw Error(`${a.id}/${b}: keys ${ks.length} != worlds`);
      out({ok: true, result: ks});
    }
    else if (a.op === 'cases') out({ok: true, result: a.seeds.map(s => { const c = E.makeCase(a.id, s, a.geometry || 'normal'); return [c.world, c.start]; })});
    else throw Error('Unknown operation');
  } catch (err) { out({ok: false, error: String(err && err.stack || err)}); }
}
process.exit(0);   // the open stdin keeps node alive after "close": end the process (the caller waits for it)
