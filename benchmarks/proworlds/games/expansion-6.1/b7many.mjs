// B7 over many seeds (rule): min informative looks (member-entropy drops, the free first view counted from the prior) >= 3 in EVERY seed; big (> 1.6 bits) share; needles.
const [id, board, s0, n] = [process.argv[2], process.argv[3], +process.argv[4], +process.argv[5]];
const { Episode, beliefOf, playReference } = await import('./engine.js');
let minInf = 99, under3 = 0, looks = 0, big = 0, needles = 0, moves = 0, right = 0;
for (let s = s0; s < s0 + n; s++) {
  const e = new Episode(id, s, { geometry: board }); let prev = beliefOf(id, board, e.W.map((_, i) => i)), inf = 0;
  const m = playReference(e, { onStep(ep) { const b = ep.belief(); if (b.entropy < prev.entropy - 1e-9) { inf++; looks++; const d = prev.entropy - b.entropy; if (d > 1.6) { big++; if (prev.entropy > 2) needles++; } } prev = b; } });
  minInf = Math.min(minInf, inf); if (inf < 3) under3++; moves += m.steps; right += m.right ? 1 : 0;
}
console.log(JSON.stringify({ id, board, seeds: `${s0}-${s0 + n - 1}`, minInf, under3, bigShare: +(big / looks).toFixed(4), needles, moves: +(moves / n).toFixed(1), right: right / n }));
