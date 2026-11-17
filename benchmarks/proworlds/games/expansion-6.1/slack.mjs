// Slack check for a tighter budget: with budget B, the reference (to 95%), the note one-step rule and the two-step rule
// must stay right >= .85, and the reference's 95th-percentile moves must be <= B. Fresh seeds 4,000,000+.
const [id, board, B, n] = [process.argv[2], process.argv[3], +process.argv[4], +process.argv[5]];
const { Episode, playReference, playNoteRule, playTwoStepRule } = await import('./engine.js');
let rr = 0, nr = 0, tr = 0; const mv = [];
for (let s = 0; s < n; s++) {
  const seed = 4_000_000 + s;
  const r = playReference(new Episode(id, seed, { geometry: board, budget: B })); mv.push(r.steps); if (r.right) rr++;
  if (playNoteRule(new Episode(id, seed, { geometry: board, budget: B })).right) nr++;
  if (playTwoStepRule(new Episode(id, seed, { geometry: board, budget: B })).right) tr++;
}
mv.sort((a, b) => a - b);
const ok = mv[Math.floor(.95 * (n - 1))] <= B && rr / n >= .85 && Math.min(nr, tr) / n >= .85;
console.log([id, board, 'B' + B, 'p95', mv[Math.floor(.95 * (n - 1))], 'ref', (rr / n).toFixed(2), 'note', (nr / n).toFixed(2), 'two', (tr / n).toFixed(2), ok ? 'OK' : 'FAIL'].join(' '));
