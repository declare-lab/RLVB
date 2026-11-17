// PRO Worlds assembly 6.0: one generic engine for "put the pieces together"
// games. A game names a closed set of members and a finite set of nuisances
// (placements, small variations). A world = (member, nuisance); the game draws
// it as a grid of 16x16 site tiles. The window shows one site and each move goes
// to the next site. The exact belief over the members is the weighted share of
// worlds that match every site seen so far (each member has the same prior).
// Normal and long boards use the same tiles; long spreads the evidence wider.
// The answer is one member's name. Score (MNIST-PRO return form): 20 if right,
// minus 10, minus 0.1 per move.
import { T, COLORS, tile } from './tiles.js';
import { GAMES } from './games.js';
export { GAMES, COLORS };
export const VERSION = 'assembly-6.1';
export const WINDOW = T, STEP = T, STOP = .95;
export const RETURN = { right: 20, base: -10, move: .1 };
export const returnOf = (right, steps) => RETURN.right * (right ? 1 : 0) + RETURN.base - RETURN.move * steps;
export const DIRECTIONS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
export const BOARDS = ['normal', 'long'];
export function rng(seed) { let a = seed >>> 0; return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const norm = s => String(s).trim().toLowerCase().replace(/\s+/g, ' ');

// All worlds of a game x board, with tile names interned as small integers.
const CACHE = new Map();
export function worldsOf(id, board) {
  const key = id + '|' + board; if (CACHE.has(key)) return CACHE.get(key);
  const g = GAMES[id]; if (!g) throw Error(`Unknown game: ${id}`);
  const b = g.boards[board]; if (!b) throw Error(`Unknown board: ${board}`);
  const ids = new Map(), names = [], worlds = [], pubs = [], pubIds = new Map();
  const intern = n => { if (!ids.has(n)) { tile(n); ids.set(n, names.length); names.push(n); } return ids.get(n); };
  const nus = g.nuisances(board);
  // v6.1 (option A): a game may give introOf(nu, board), a PUBLIC rules text that depends on the world (a map, the
  // pans). Worlds with the same text form one public class (pub); an episode's belief starts inside its own class. No introOf = v6.0.
  const pubOf = nu => { if (!g.introOf) return 0; const s = g.introOf(nu, board); if (!pubIds.has(s)) { pubIds.set(s, pubs.length); pubs.push(s); } return pubIds.get(s); };
  g.members.forEach((m, mi) => {
    const own = nus.map((nu, ni) => [g.layout(m, nu, board), ni]).filter(x => x[0]);
    if (!own.length) throw Error(`${id}/${board}: member ${m} has no world`);
    for (const [sites, ni] of own) {
      if (sites.length !== b.cols * b.rows) throw Error(`${id}/${board}: layout gives ${sites.length} sites, not ${b.cols * b.rows}`);
      worlds.push({ member: mi, w: 1 / own.length, sites: Int16Array.from(sites, intern), pub: pubOf(nus[ni]) });
    }
  });
  const out = { worlds, names, pubs, cols: b.cols, rows: b.rows, budget: b.budget };
  CACHE.set(key, out); return out;
}

// Exact belief from a list of consistent world indices.
export function beliefOf(id, board, live) {
  const W = worldsOf(id, board).worlds, k = GAMES[id].members.length, p = new Float64Array(k);
  let z = 0; for (const i of live) { p[W[i].member] += W[i].w; z += W[i].w; }
  for (let m = 0; m < k; m++) p[m] /= z;
  let top = 0, arg = 0, H = 0; for (let m = 0; m < k; m++) { if (p[m] > top) { top = p[m]; arg = m; } if (p[m] > 0) H -= p[m] * Math.log2(p[m]); }
  return { candidates: [...p], top, topMember: GAMES[id].members[arg], entropy: H, certain: top >= 1 - 1e-12 };
}
const filterLive = (W, live, site, t) => live.filter(i => W[i].sites[site] === t);
const classOf = (W, world) => W.map((_, i) => i).filter(i => W[i].pub === W[world].pub);   // v6.1: the public class (all worlds when no introOf)

export function makeCase(id, seed, board = 'normal') {
  const g = GAMES[id], { worlds, cols, rows, budget } = worldsOf(id, board);
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('Seed must be uint32');
  let salt = 2166136261; for (const ch of id + '|' + board + '|6.0') salt = Math.imul(salt ^ ch.charCodeAt(0), 16777619);
  const r = rng((seed ^ salt) >>> 0), k = g.members.length;
  for (let tries = 0; tries < 1000; tries++) {
    const member = Math.floor(r() * k), own = worlds.map((w, i) => w.member === member ? i : -1).filter(i => i >= 0);
    const world = own[Math.floor(r() * own.length)], start = Math.floor(r() * cols * rows);
    const live = filterLive(worlds, classOf(worlds, world), start, worlds[world].sites[start]);
    if (beliefOf(id, board, live).top < STOP) return { id, seed, board, member, world, start, cols, rows, budget };
  }
  throw Error(`${id}/${board}: no start where the first view leaves doubt`);
}

export class Episode {
  constructor(id, seed = 0, { geometry = 'normal', budget } = {}) {
    if (!BOARDS.includes(geometry)) throw Error('Unknown geometry: use normal or long');
    this.id = id; this.seed = seed; this.geometry = geometry; this.case = makeCase(id, seed, geometry);
    this.cols = this.case.cols; this.rows = this.case.rows; this.budget = budget ?? this.case.budget;
    if (!Number.isInteger(this.budget) || this.budget < 1 || this.budget > 4096) throw Error('Invalid movement budget');
    const W = worldsOf(id, geometry); this.W = W.worlds; this.names = W.names;
    this.sx = this.case.start % this.cols; this.sy = Math.floor(this.case.start / this.cols);
    this.steps = 0; this.done = false; this.answer = null; this.seenSites = []; this.history = []; this.predictions = [];
    this.live = classOf(this.W, this.case.world); this.reveal(); this.initial = this.observe();
  }
  site() { return this.sy * this.cols + this.sx; }
  tileAt(k) { return this.names[this.W[this.case.world].sites[k]]; }
  reveal() { const k = this.site(); if (!this.seenSites.includes(k)) { this.seenSites.push(k); this.live = filterLive(this.W, this.live, k, this.W[this.case.world].sites[k]); } }
  observe() { return { bounds: { x: this.sx * T, y: this.sy * T, width: T, height: T }, site: { column: this.sx + 1, row: this.sy + 1 }, window: [...tile(this.tileAt(this.site()))], steps: this.steps, remaining: this.budget - this.steps, done: this.done }; }
  publicTask() { const g = GAMES[this.id]; return { id: this.id, geometry: this.geometry, name: g.name, rules: publicRules(this.id, this.geometry, this.budget, GAMES[this.id].introOf ? worldsOf(this.id, this.geometry).pubs[this.W[this.case.world].pub] : undefined), question: g.ask, setText: g.setText, members: [...g.members], cols: this.cols, rows: this.rows, width: this.cols * T, height: this.rows * T, window: T, step: STEP, budget: this.budget }; }
  move(action) {
    if (!Object.hasOwn(DIRECTIONS, action)) throw Error('Invalid action');
    if (this.done || this.steps >= this.budget) throw Error('No moves remaining');
    const [dx, dy] = DIRECTIONS[action]; this.sx = Math.max(0, Math.min(this.cols - 1, this.sx + dx)); this.sy = Math.max(0, Math.min(this.rows - 1, this.sy + dy));
    this.steps++; this.reveal(); const observation = this.observe(); this.history.push({ action, observation }); return observation;
  }
  belief() { return beliefOf(this.id, this.geometry, this.live); }
  predict(probabilities) { const k = GAMES[this.id].members.length; if (this.done) throw Error('Round ended'); if (!Array.isArray(probabilities) || probabilities.length !== k || Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) > 1e-6) throw Error(`Supply ${k} probabilities that sum to 1`); this.predictions.push({ step: this.steps, probabilities: [...probabilities] }); return { recorded: true, step: this.steps }; }
  commit(value) { if (this.done) throw Error('Already committed'); if (typeof value !== 'string') throw Error('Supply the answer name as a string'); this.done = true; this.answer = value; return this.metrics(); }
  truthName() { return GAMES[this.id].members[this.case.member]; }
  metrics() { const b = this.belief(), right = this.answer === null ? null : norm(this.answer) === norm(this.truthName()); return { right, exact: right, steps: this.steps, sitesSeen: this.seenSites.length, sites: this.cols * this.rows, siteShare: this.seenSites.length / (this.cols * this.rows), topChance: b.top, certain: b.certain, entropy: b.entropy, score: this.answer === null ? null : returnOf(right, this.steps) }; }
  record() { return { version: VERSION, id: this.id, seed: this.seed, geometry: this.geometry, ...(this.budget === this.case.budget ? {} : { budget: this.budget }), initial: this.initial, history: structuredClone(this.history), predictions: structuredClone(this.predictions), answer: this.answer, private: { truth: this.truthName(), world: this.case.world, metrics: this.metrics() } }; }
}
export function replay(r) {
  if (r.version !== VERSION) throw Error('Version mismatch');
  const e = new Episode(r.id, r.seed, { geometry: r.geometry ?? 'normal', budget: r.budget });
  if (JSON.stringify(e.initial) !== JSON.stringify(r.initial)) throw Error('Initial mismatch');
  const probes = () => { for (const p of r.predictions) if (p.step === e.steps) e.predict(p.probabilities); };
  probes(); for (const row of r.history) { if (JSON.stringify(e.move(row.action)) !== JSON.stringify(row.observation)) throw Error('Observation mismatch'); probes(); }
  if (r.answer !== null) e.commit(r.answer);
  if (JSON.stringify(e.record()) !== JSON.stringify(r)) throw Error('Record mismatch'); return e;
}

export function publicRules(id, board, budget, pubText) {
  const g = GAMES[id], b = g.boards[board];
  const intro = pubText !== undefined ? pubText : (typeof g.intro === 'string' ? g.intro : g.intro[board]);   // v6.1: a per-world intro
  return `${g.name}. The board has ${b.cols}x${b.rows} sites (${b.cols * T}x${b.rows * T} cells). The window shows one site; each move goes to the next site (up, down, left or right). ${intro} Question: ${g.ask} The answer is ${g.setText} Answer with its name. You may answer at any time; you do not need to see every site. Score: ${RETURN.right} if the answer is right, minus ${-RETURN.base}, minus ${RETURN.move} per move (movement budget ${budget ?? b.budget}).`;
}

// Look-ahead. distance = moves from the current site; bits = expected entropy
// reduction of the member belief; topAfter = expected top chance after the look.
const dist = (e, k) => Math.abs(k % e.cols - e.sx) + Math.abs(Math.floor(k / e.cols) - e.sy);
function split(e, live, site) {
  const k = GAMES[e.id].members.length, groups = new Map();
  for (const i of live) { const t = e.W[i].sites[site]; let g = groups.get(t); if (!g) { g = { z: 0, p: new Float64Array(k), live: [] }; groups.set(t, g); } g.z += e.W[i].w; g.p[e.W[i].member] += e.W[i].w; g.live.push(i); }
  return [...groups.values()];
}
const entropyOf = (p, z) => { let H = 0; for (const x of p) if (x > 0) { const q = x / z; H -= q * Math.log2(q); } return H; };
export function lookahead(e, live = e.live) {
  const b = beliefOf(e.id, e.geometry, live), rows = [];
  for (let s = 0; s < e.cols * e.rows; s++) {
    if (e.seenSites.includes(s)) continue;
    const gs = split(e, live, s), z = gs.reduce((a, g) => a + g.z, 0);
    let top = 0, H = 0; for (const g of gs) { top += Math.max(...g.p); H += g.z / z * entropyOf(g.p, g.z); }
    rows.push({ site: s, column: s % e.cols + 1, row: Math.floor(s / e.cols) + 1, distance: dist(e, s), bits: Math.max(0, b.entropy - H), topAfter: top / z, gain: top / z - b.top });
  }
  return rows;
}
// Two looks in a row: the first at a, then (after seeing a) the best second look c
// among the best `width` single looks, OR no second look. Each branch pays its own
// walk: value = E[max(top after a, top after a and c - lam x moves a->c)] - lam x moves to a,
// with lam = 0.1 / 20 (the move cost in units of the top chance). moves = the expected
// moves of the chosen plan (rounded for the note); topAfter2 = the expected top chance.
export function lookahead2(e, width = 12) {
  const lam = RETURN.move / RETURN.right, d2 = (a, c) => Math.abs(c.column - a.column) + Math.abs(c.row - a.row);
  const one = lookahead(e).filter(r => r.distance > 0).sort((a, b) => b.bits / b.distance - a.bits / a.distance).slice(0, width);
  let best = null;
  for (const a of one) {
    const gs = split(e, e.live, a.site), z = gs.reduce((s, g) => s + g.z, 0); let value = 0, top = 0, moves = a.distance;
    for (const g of gs) {
      let bv = Math.max(...g.p) / g.z, bt = bv, bd = 0;
      for (const c of one) { if (c.site === a.site) continue; let t = 0; for (const h of split(e, g.live, c.site)) t += Math.max(...h.p); t /= g.z; const v = t - lam * d2(a, c); if (v > bv) { bv = v; bt = t; bd = d2(a, c); } }
      value += g.z / z * bv; top += g.z / z * bt; moves += g.z / z * bd;
    }
    value -= lam * a.distance;
    if (!best || value > best.value) best = { first: a, topAfter2: top, moves, movesShown: Math.round(moves), value };
  }
  return best;
}
// The 2-step note rule: move toward the first look of the best 2-look plan while its
// gain = 20 x (topAfter2 - top) - 0.1 x moves is above 0, else answer.
export function playTwoStepRule(e, { onStep } = {}) {
  let target = null;
  for (;;) {
    onStep?.(e); const b = e.belief();
    if (e.steps >= e.budget) return e.commit(b.topMember);
    if (!target || e.seenSites.includes(target)) { const t = lookahead2(e); if (!t || RETURN.right * (t.topAfter2 - b.top) - RETURN.move * t.moves <= 0) return e.commit(b.topMember); target = t.first.site; }
    e.move(toward(e, target));
  }
}
export function toward(e, site) { const x = site % e.cols, y = Math.floor(site / e.cols); if (e.sx < x) return 'right'; if (e.sx > x) return 'left'; if (e.sy < y) return 'down'; if (e.sy > y) return 'up'; return null; }
export function informationTarget(e) { let best = null; for (const r of lookahead(e)) { if (r.bits <= 1e-9 || r.distance === 0) continue; const v = r.bits / r.distance; if (!best || v > best.v) best = { ...r, v }; } return best; }
export function nearestUnseen(e) { let best = null; for (let s = 0; s < e.cols * e.rows; s++) { if (e.seenSites.includes(s)) continue; const d = dist(e, s); if (!best || d < best.d) best = { site: s, d }; } return best; }
// The reference policy: walk to the unseen site with the most expected bits per
// move; answer the most likely member once its chance reaches STOP (or at the
// budget, or when no unseen site can change the belief).
export function playReference(e, { onStep, stop = STOP, blind = false } = {}) {
  let target = null;
  for (;;) {
    onStep?.(e); const b = e.belief();
    if (b.top >= stop || e.steps >= e.budget) return e.commit(b.topMember);
    if (!target || e.seenSites.includes(target)) { const t = blind ? nearestUnseen(e) : informationTarget(e); if (!t) return e.commit(b.topMember); target = t.site; }
    e.move(toward(e, target));
  }
}

// The note's one-step rule (the teacher's Decision): look at the unseen site with
// the largest gain = 20 x (topAfter - top) - 0.1 x distance; move toward it while
// that gain is above 0, else answer the most likely member.
export function bestNoteLook(e) { const b = e.belief(); let best = null; for (const r of lookahead(e)) { if (r.distance === 0) continue; const gain = RETURN.right * (r.topAfter - b.top) - RETURN.move * r.distance; if (!best || gain > best.gain) best = { ...r, gain }; } return best; }
export function playNoteRule(e, { onStep } = {}) {
  let target = null;
  for (;;) {
    onStep?.(e); const b = e.belief();
    if (e.steps >= e.budget) return e.commit(b.topMember);
    if (!target || e.seenSites.includes(target)) { const t = bestNoteLook(e); if (!t || t.gain <= 0) return e.commit(b.topMember); target = t.site; }
    e.move(toward(e, target));
  }
}
