// PRO Worlds assembly 6.0: the design checks for every game x board, with the
// reference policy over SEEDS seeds. Rules:
// A3/A10 8-12 named members; A6 no digits in names and no digit-like tiles; set words not shared;
// A5 no counting (the member must not need tile COUNTS beyond which tiles appear);
// B4 the first view never decides; B5 the belief equals an independent recount;
// B7 >= 3 informative looks to STOP in every seed; <= 5% of informative looks give > 1.6 bits;
// needle looks (> 1.6 bits removed while the entropy before is > 2 bits) <= 1%;
// B8 reference moves to STOP: normal 10-20, long 30-50; reference right >= .85;
// C4 no two games share a looking strategy (first-10-look visit maps, r > .8 on the same board);
// B3 adaptivity (reported): share of seeds where the 2nd reference look depends on what the 1st look showed;
// C5 no pair-only evidence: no site with 0 bits alone that raises the bits of a look at another site;
// NOTE the note's one-step rule as a player (normal): right >= .85, moves 10-20;
// speed: exact belief + one-step look-ahead <= 1 s per state.
// Usage: node games/assembly/test.mjs [game ...] [--set] or --fixtures (--set: apply the set rules to the listed games) (the rule fixtures must fail / pass)
import { mkdir, writeFile } from 'node:fs/promises';
const FIXTURES = process.argv.includes('--fixtures');
if (FIXTURES) await import('./fixtures.mjs');
const { GAMES, BOARDS, Episode, worldsOf, beliefOf, lookahead, playReference, playNoteRule, playTwoStepRule, informationTarget, STOP, VERSION } = await import('./engine.js');
const { DIGIT_LIKE } = await import('./tiles.js');
const SEEDS = 32, MOVES = { normal: [10, 20], long: [30, 50] };
const args = process.argv.slice(2).filter(a => !a.startsWith('--'));
const ids = FIXTURES ? Object.keys(GAMES).filter(k => k.startsWith('fx')) : (args.length ? args : Object.keys(GAMES).filter(k => !k.startsWith('fx')));
const fails = [], rows = [], visits = {};
const fail = (where, msg) => fails.push(`${where}: ${msg}`);
const mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);

// Set checks across games (not for the fixtures, which share one name set on purpose).
if (!FIXTURES) {
  const words = new Map();
  for (const id of Object.keys(GAMES).filter(k => !k.startsWith('fx'))) {
    const g = GAMES[id];
    if (g.members.length < 8 || g.members.length > 12) fail(id, `A10 ${g.members.length} members (need 8-12)`);
    if (new Set(g.members.map(m => m.toLowerCase())).size !== g.members.length) fail(id, 'A3 duplicate member names');
    for (const m of g.members) { if (/\d/.test(m)) fail(id, `A6 digit in member "${m}"`); for (const w of m.toLowerCase().split(/\s+/)) { if (words.has(w) && words.get(w) !== id) fail(id, `set word "${w}" also in ${words.get(w)}`); words.set(w, id); } }
  }
}
function counting(id, board) {   // A5, by world share: the tiles present leave the member open, the tile counts decide it
  const { worlds } = worldsOf(id, board), byPresence = new Map(), byHist = new Map(), keys = [];
  for (const w of worlds) {
    const h = new Map(); for (const t of w.sites) h.set(t, (h.get(t) ?? 0) + 1);
    const pres = [...h.keys()].sort((a, b) => a - b).join(','), hist = [...h.entries()].sort((a, b) => a[0] - b[0]).map(e => e.join(':')).join(',');
    (byPresence.get(pres) ?? byPresence.set(pres, new Set()).get(pres)).add(w.member);
    (byHist.get(hist) ?? byHist.set(hist, new Set()).get(hist)).add(w.member); keys.push([pres, hist]);
  }
  const share = (m, j) => keys.filter(k => m.get(k[j]).size > 1).length / keys.length;
  return share(byPresence, 0) > .5 && share(byHist, 1) < .1;
}
// C5 at the start state: pair-only evidence = a site whose one look gives 0 member bits, but which makes the
// look at some other site more informative (the bits of "s then t" exceed the bits of t alone). A varying site
// that tells nothing alone AND nothing together with any other site is irrelevant scenery, not pair-only.
function entropyLive(e, live) { const k = GAMES[e.id].members.length, p = new Float64Array(k); let z = 0; for (const i of live) { p[e.W[i].member] += e.W[i].w; z += e.W[i].w; } let H = 0; for (const x of p) if (x > 0) H -= x / z * Math.log2(x / z); return H; }
function groupsAt(e, live, site) { const g = new Map(); for (const i of live) { const t = e.W[i].sites[site]; (g.get(t) ?? g.set(t, []).get(t)).push(i); } return [...g.values()]; }
function expBits(e, live, site) { const H = entropyLive(e, live), z = live.reduce((a, i) => a + e.W[i].w, 0); let h = 0; for (const g of groupsAt(e, live, site)) h += g.reduce((a, i) => a + e.W[i].w, 0) / z * entropyLive(e, g); return H - h; }
// B3 adaptivity (the original Integration counterfactual test): fix the start and the first reference
// look; for each tile that this first look can show, find the next reference look. Adaptive = the
// next look differs between the possible first-look results.
function adaptive(e) {
  const t1 = informationTarget(e); if (!t1) return false;
  const nexts = new Set();
  for (const g of groupsAt(e, e.live, t1.site)) {
    const c = Object.assign(Object.create(Object.getPrototypeOf(e)), e, { sx: t1.site % e.cols, sy: Math.floor(t1.site / e.cols), seenSites: [...e.seenSites, t1.site], live: g });
    const t2 = informationTarget(c); nexts.add(t2 ? t2.site : -1);
  }
  return nexts.size > 1;
}
function pairOnly(e) {
  const rows = lookahead(e), varying = rows.filter(r => new Set(e.live.map(i => e.W[i].sites[r.site])).size >= 2);
  for (const r of varying) {
    if (r.bits > 1e-9) continue;
    const gs = groupsAt(e, e.live, r.site), z = e.live.reduce((a, i) => a + e.W[i].w, 0);
    for (const t of varying) { if (t.site === r.site) continue; let both = 0; for (const g of gs) both += g.reduce((a, i) => a + e.W[i].w, 0) / z * expBits(e, g, t.site); if (both > t.bits + 1e-9) return r.site; }
  }
  return null;
}

for (const id of ids) for (const board of BOARDS) {
  const where = `${id}/${board}`, t0 = Date.now(); let W;
  try { W = worldsOf(id, board); } catch (err) { fail(where, err.message); continue; }
  if (W.names.some(n => DIGIT_LIKE.some(ch => n === `letter:${ch}`))) fail(where, 'A6 a digit-like letter tile is used');
  if (counting(id, board)) fail(where, 'A5 the answer depends on tile counts');
  const r = { moves: [], right: 0, informative: [], bitsPerLook: [], jumps: 0, big: 0, firstViewCertain: 0, blindMoves: [], beliefMismatch: 0, maxStateMs: 0, pairOnly: 0, noteMoves: [], noteRight: 0, twoMoves: [], twoRight: 0, adaptive: 0 };
  const vmap = new Float64Array(25);
  for (let s = 0; s < SEEDS; s++) {
    const seed = 600000 + s, e = new Episode(id, seed, { geometry: board });
    if (e.belief().top >= STOP) r.firstViewCertain++;
    if (pairOnly(e) !== null) r.pairOnly++;
    if (adaptive(e)) r.adaptive++;
    let prev = beliefOf(id, board, e.W.map((_, i) => i)), inf = 0;   // the prior: the free first view counts as a look
    const m = playReference(e, { onStep(ep) {
      const ts = Date.now(), b = ep.belief(); lookahead(ep); r.maxStateMs = Math.max(r.maxStateMs, Date.now() - ts);
      const ind = beliefOf(id, board, ep.W.map((_, i) => i).filter(i => ep.W[i].pub === ep.W[ep.case.world].pub && ep.seenSites.every(k => ep.W[i].sites[k] === ep.W[ep.case.world].sites[k])));   // B5: an independent recount (v6.1: inside the public class)
      if (ind.candidates.some((p, j) => Math.abs(p - b.candidates[j]) > 1e-9)) r.beliefMismatch++;
      if (b.entropy < prev.entropy - 1e-9) { inf++; r.bitsPerLook.push(prev.entropy - b.entropy); if (prev.entropy - b.entropy > 1.6) { r.big++; if (prev.entropy > 2) r.jumps++; } }
      prev = b;
    } });
    r.moves.push(m.steps); if (m.right) r.right++; r.informative.push(inf);
    for (const k of e.seenSites.slice(1, 11)) vmap[Math.floor((Math.floor(k / e.cols) + .5) * 5 / e.rows) * 5 + Math.floor((k % e.cols + .5) * 5 / e.cols)]++;   // C4: first-10 looks
    r.blindMoves.push(playReference(new Episode(id, seed, { geometry: board }), { blind: true }).steps);
    const n = playNoteRule(new Episode(id, seed, { geometry: board })); r.noteMoves.push(n.steps); if (n.right) r.noteRight++;
    const n2 = playTwoStepRule(new Episode(id, seed, { geometry: board })); r.twoMoves.push(n2.steps); if (n2.right) r.twoRight++;
  }
  visits[where] = { board, map: [...vmap].map(v => v / Math.max(1, vmap.reduce((a, b) => a + b, 0))) };
  const out = { game: id, board, worlds: W.worlds.length, sites: W.cols * W.rows, members: GAMES[id].members.length, kind: GAMES[id].kind, picture: !!GAMES[id].picture,
    moves: +mean(r.moves).toFixed(1), right: r.right / SEEDS, informativeLooks: +mean(r.informative).toFixed(2), minInformativeLooks: Math.min(...r.informative),
    bitsPerLook: +mean(r.bitsPerLook).toFixed(2), maxBitsPerLook: +Math.max(0, ...r.bitsPerLook).toFixed(2), bigLookShare: +(r.big / Math.max(1, r.bitsPerLook.length)).toFixed(3), needleLooks: r.jumps,
    firstViewCertain: r.firstViewCertain, blindMoves: +mean(r.blindMoves).toFixed(1), whereSaves: +(1 - mean(r.moves) / Math.max(1e-9, mean(r.blindMoves))).toFixed(2),
    adaptiveShare: +(r.adaptive / SEEDS).toFixed(2), noteRuleMoves: +mean(r.noteMoves).toFixed(1), noteRuleRight: r.noteRight / SEEDS, twoStepMoves: +mean(r.twoMoves).toFixed(1), twoStepRight: r.twoRight / SEEDS, beliefMismatch: r.beliefMismatch, pairOnlyStarts: r.pairOnly, maxStateMs: r.maxStateMs, seconds: (Date.now() - t0) / 1000 };
  rows.push(out);
  const [lo, hi] = MOVES[board];
  if (out.moves < lo || out.moves > hi) fail(where, `B8 reference moves ${out.moves} (need ${lo}-${hi})`);
  if (out.right < .85) fail(where, `reference right ${out.right} (need >= .85)`);
  if (out.minInformativeLooks < 3) fail(where, `B7 min informative looks ${out.minInformativeLooks} (need >= 3)`);
  if (out.bigLookShare > .05) fail(where, `B7 ${out.bigLookShare} of informative looks give > 1.6 bits (need <= .05)`);
  if (out.needleLooks > .01 * r.bitsPerLook.length) fail(where, `B7 needle: ${out.needleLooks} of ${r.bitsPerLook.length} looks remove > 1.6 bits from an entropy > 2 bits (need <= 1%)`);
  if (out.firstViewCertain > 0) fail(where, `B4 first view certain in ${out.firstViewCertain} seeds`);
  if (out.beliefMismatch > 0) fail(where, `B5 belief differs from the recount in ${out.beliefMismatch} states`);
  if (out.pairOnlyStarts > 0) fail(where, `C5 pair-only evidence at the start in ${out.pairOnlyStarts} seeds`);
  if (out.maxStateMs > 1000) fail(where, `speed ${out.maxStateMs} ms per state (need <= 1000)`);
  if (board === 'normal' && (out.noteRuleRight < .85 || out.noteRuleMoves < lo || out.noteRuleMoves > hi)) fail(where, `NOTE one-step rule right ${out.noteRuleRight}, moves ${out.noteRuleMoves} (need >= .85, ${lo}-${hi})`);
}
// Set-level rules, on the full set only: >= 60% adaptive games (normal adaptiveShare >= .30),
// >= 60% picture games, all 7 kinds present, <= 3 games per kind.
const normalRows = rows.filter(r => r.board === 'normal'), isAdaptive = r => r.adaptiveShare >= .30;
// How a game is adaptive: 'chain' = the next informative place depends on what was seen
// (decoy chains: circuit, maze, vanes); 'placement' = only finding WHERE the parts are (the +-1 shift of
// the parts games); 'fixed' = not adaptive.
const ADAPT_BY = { circuit: 'chain', maze: 'chain', vanes: 'chain', face: 'placement', sun: 'placement', jigsaw: 'placement', kaleido: 'placement', papercut: 'placement' };
for (const r of rows) r.adaptiveBy = isAdaptive(normalRows.find(x => x.game === r.game) ?? r) ? (ADAPT_BY[r.game] ?? 'unclassified') : 'fixed';
if (!FIXTURES && (!args.length || process.argv.includes('--set'))) {
  const n = normalRows.length, kinds = {}; for (const r of normalRows) kinds[r.kind] = (kinds[r.kind] ?? 0) + 1;
  const ad = normalRows.filter(isAdaptive).length, pic = normalRows.filter(r => r.picture).length;
  if (ad < .6 * n) fail('SET', `B3 ${ad} of ${n} games adaptive (need >= 60%)`);
  if (pic < .6 * n) fail('SET', `${pic} of ${n} picture games (need >= 60%)`);
  for (const k of ['assemble', 'order', 'symmetry', 'relative', 'path', 'join', 'clues']) if (!kinds[k]) fail('SET', `kind ${k} missing`);
  for (const [k, c] of Object.entries(kinds)) if (c > 3) fail('SET', `kind ${k} has ${c} games (need <= 3)`);
  console.log(`SET: ${n} games, ${ad} adaptive (by chain ${normalRows.filter(r => r.adaptiveBy === 'chain').length}, by placement ${normalRows.filter(r => r.adaptiveBy === 'placement').length}), ${pic} picture, kinds ${JSON.stringify(kinds)}`);
}
// C4: visit-map correlation between games on the same board.
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let n = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; } return da && db ? n / Math.sqrt(da * db) : 0; };
const vk = Object.keys(visits), c4 = [];
for (let i = 0; i < vk.length; i++) for (let j = i + 1; j < vk.length; j++) if (visits[vk[i]].board === visits[vk[j]].board) { const r = corr(visits[vk[i]].map, visits[vk[j]].map); c4.push({ a: vk[i], b: vk[j], r: +r.toFixed(2) }); if (r > .8 && !FIXTURES) fail(`${vk[i]} + ${vk[j]}`, `C4 visit maps correlate r = ${r.toFixed(2)} (> .8)`); }

console.table(rows.map(r => ({ game: r.game, board: r.board, worlds: r.worlds, moves: r.moves, right: r.right, looks: r.informativeLooks, minLooks: r.minInformativeLooks, bits: r.bitsPerLook, maxBits: r.maxBitsPerLook, big: r.bigLookShare, needles: r.needleLooks, saves: r.whereSaves, adapt: r.adaptiveShare, noteMoves: r.noteRuleMoves, noteRight: r.noteRuleRight, twoMoves: r.twoStepMoves, twoRight: r.twoStepRight, ms: r.maxStateMs })));
if (FIXTURES) {   // each broken fixture must fail its rule; the clean one must pass every rule
  const has = (id, tag) => fails.some(f => f.startsWith(id + '/') && f.includes(tag));
  const expect = [['fxcount', 'A5', true], ['fxneedle', 'needle', true], ['fxpair', 'C5', true], ['fxclean', '', false]];
  const bad = expect.filter(([id, tag, must]) => must ? !has(id, tag) : fails.some(f => f.startsWith(id + '/')));
  const ad = id => isAdaptive(normalRows.find(r => r.game === id));
  if (ad('fxclean')) bad.push(['fxclean is adaptive (must be flagged as fixed-place)']); if (!ad('fxadapt')) bad.push(['fxadapt is not adaptive']);
  console.log(fails.join('\n'));
  if (bad.length) { console.log('FIXTURES FAIL: ' + bad.map(b => b[0]).join(', ')); process.exitCode = 1; } else console.log('FIXTURES PASS: fxcount fails A5, fxneedle fails B7 needle, fxpair fails C5, fxclean passes (flagged fixed-place), fxadapt is adaptive');
} else {
  await mkdir('out/assembly-61', { recursive: true });
  await writeFile('out/assembly-61/checks.json', JSON.stringify({ version: VERSION, passed: fails.length === 0, seeds: SEEDS, games: rows, c4, fails }, null, 1) + '\n');
  if (fails.length) { console.log(`FAIL (${fails.length}):\n` + fails.join('\n')); process.exitCode = 1; } else console.log('ALL PASS');
}
