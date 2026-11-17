// per-world B7 report (the test.mjs counting, grouped by the episode's public world): node perworld.mjs GAME BOARD NSEEDS
const { Episode, beliefOf, playReference } = await import('./engine.js');
const [id, board, ns] = [process.argv[2], process.argv[3], +process.argv[4]];
const by = new Map();
for (let s = 0; s < ns; s++) {
  const seed = 20100000 + s, e = new Episode(id, seed, { geometry: board }); const nu = e.W[e.case.world].pub ?? 0;
  let prev = beliefOf(id, board, e.W.map((_, i) => i).filter(i => e.W[i].pub === e.W[e.case.world].pub)), inf = 0, big = 0, nd = 0, looks = 0;
  const m = playReference(e, { onStep(ep) { const b = ep.belief(); if (b.entropy < prev.entropy - 1e-9) { const d = prev.entropy - b.entropy; inf++; looks++; if (d > 1.6) { big++; if (prev.entropy > 2) nd++; } } prev = b; } });
  const r = by.get(nu) ?? { n: 0, minLooks: 99, under: 0, big: 0, nd: 0, looks: 0, moves: 0, right: 0 }; by.set(nu, r);
  r.n++; r.minLooks = Math.min(r.minLooks, inf); r.under += inf < 3; r.big += big; r.nd += nd; r.looks += looks; r.moves += m.steps; r.right += m.right ? 1 : 0;
}
const rows = [...by.entries()].map(([nu, r]) => ({ nu, n: r.n, minLooks: r.minLooks, shareUnder3: +(r.under / r.n).toFixed(3), big: +(r.big / r.looks).toFixed(3), needles: +(r.nd / r.looks).toFixed(3), moves: +(r.moves / r.n).toFixed(1), right: +(r.right / r.n).toFixed(2) }));
const tot = rows.reduce((a, r) => a + r.n, 0), under = [...by.values()].reduce((a, r) => a + r.under, 0);
const worlds0 = rows.filter(r => r.minLooks >= 3).length;
console.log(JSON.stringify({ game: id, board, seeds: ns, worlds: rows.length, worlds_min3: worlds0, under3_share_all: +(under / tot).toFixed(3), worst_under3: [...rows].sort((a, b) => b.shareUnder3 - a.shareUnder3).slice(0, 3), worst_big: [...rows].sort((a, b) => b.big - a.big)[0], worst_needles: [...rows].sort((a, b) => b.needles - a.needles)[0], moves_range: [Math.min(...rows.map(r => r.moves)), Math.max(...rows.map(r => r.moves))] }));
