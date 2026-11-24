// "The same game under three rewards": one cups game, animated under the outcome reward, a judged process reward and the
// posterior reward of RLVB. Each chapter is a scene on one SVG stage with a timeline of tweens; the scene is drawn from the
// time alone, so it plays, loops and can be scrubbed. The game and its numbers are those of the first two figures of the paper.
(() => {
  const root = document.getElementById("credit");
  if (!root) return;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const C = { cup: "#2f3340", cupTop: "#555a6c", green: "#2e9e5b", greenTop: "#62c48b", red: "#d0402f", redTop: "#ea8172",
    teal: "#12805c", tealSoft: "#e7f4ee", tealLine: "#a9d3c6", bad: "#d43c3c", badSoft: "#fbeaea", orange: "#c9460c",
    ink: "#0c0c0e", ink2: "#46484f", ink3: "#6b6d76", line: "#e6e6e9", line2: "#cfcfd4", page: "#f5f5f6", soft: "#ececef", card: "#ffffff" };
  const E = { lin: p => p, out: p => 1 - Math.pow(1 - p, 3), io: p => p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2,
    back: p => 1 + 2.70158 * Math.pow(p - 1, 3) + 1.70158 * Math.pow(p - 1, 2) };
  const lerp = (a, b, p) => a + (b - a) * p, clamp = p => Math.max(0, Math.min(1, p));
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, p) => { const x = hex(a), y = hex(b); return `rgb(${x.map((v, i) => Math.round(lerp(v, y[i], p))).join(",")})`; };
  const mk = (tag, attrs, parent, text) => { const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; if (parent) parent.appendChild(n); return n; };
  const txt = (parent, x, y, s, cls, anchor = "middle") => mk("text", { x, y, class: cls, "text-anchor": anchor, "dominant-baseline": "central" }, parent, s);

  // ---------- a scene: actors (groups with x, y, scale, rotation, opacity), tweens on them, and free callbacks
  class Scene {
    constructor(svg) { this.g = mk("g", { class: "scene" }, svg); this.nodes = []; this.tw = []; this.calls = []; this.caps = []; this.end = 0; }
    actor(bx, by, init, parent) { const n = mk("g", {}, parent || this.g); n._b = [bx, by]; n._i = Object.assign({ x: 0, y: 0, s: 1, r: 0, o: 1 }, init); this.nodes.push(n); return n; }
    to(n, props, t0, dur = .4, ease = E.out) { this.tw.push({ n, props, t0, dur, ease }); this.end = Math.max(this.end, t0 + dur); }
    call(t0, dur, fn, ease = E.out) { this.calls.push({ t0, dur, fn, ease }); this.end = Math.max(this.end, t0 + dur); }
    cap(t, s) { this.caps.push([t, s]); }
    done(hold = 2.2) { this.tw.sort((a, b) => a.t0 - b.t0); this.dur = this.end + hold; return this; }
    caption(t) { let s = ""; for (const [t0, c] of this.caps) if (t >= t0) s = c; return s; }
    render(t) {
      for (const n of this.nodes) n._s = Object.assign({}, n._i);
      for (const w of this.tw) { if (t < w.t0) break; const p = w.ease(clamp((t - w.t0) / w.dur)); for (const k in w.props) w.n._s[k] = lerp(w.n._s[k], w.props[k], p); }
      for (const n of this.nodes) { const s = n._s;
        n.setAttribute("transform", `translate(${(n._b[0] + s.x).toFixed(2)} ${(n._b[1] + s.y).toFixed(2)}) rotate(${s.r.toFixed(2)}) scale(${Math.max(0, s.s).toFixed(3)})`);
        n.setAttribute("opacity", clamp(s.o).toFixed(3)); }
      for (const c of this.calls) c.fn(c.ease(clamp((t - c.t0) / c.dur)));
    }
  }
  const pop = (S, n, t, d = .38) => S.to(n, { s: 1, o: 1 }, t, d, E.back);
  const fade = (S, n, t, o = 1, d = .3) => S.to(n, { o }, t, d);
  const pulse = (S, n, t, k = 1.28) => { S.to(n, { s: k }, t, .16); S.to(n, { s: 1 }, t + .16, .3, E.io); };

  // ---------- the things on stage
  // the table with four cups; the green cylinder is under cup 1 and the red one under cup 4; the pips below show the cups still possible
  function table(S, par) {
    const g = par || S.g, X = [100, 190, 280, 370], T = { X, cups: [], pips: [] };
    mk("rect", { x: 44, y: 250, width: 382, height: 9, rx: 4.5, fill: "#e2ddd3" }, g);
    const cyl = (x, col, top) => { mk("rect", { x: x - 11, y: 226, width: 22, height: 24, fill: col }, g);
      mk("ellipse", { cx: x, cy: 226, rx: 11, ry: 3.5, fill: top }, g); };
    cyl(X[0], C.green, C.greenTop); cyl(X[3], C.red, C.redTop);
    X.forEach((x, k) => {
      const c = S.actor(x, 250, null, g);
      mk("path", { d: "M-29 0L-22 -72H22L29 0Z", fill: C.cup }, c);
      mk("path", { d: "M-29 0L-22 -72H-11L-16 0Z", fill: "#fff", opacity: .12 }, c);
      mk("ellipse", { cx: 0, cy: -72, rx: 22, ry: 5, fill: C.cupTop }, c);
      T.cups.push(c);
      const p = S.actor(x, 296, null, g);
      mk("circle", { r: 13, fill: "none", stroke: C.line2, "stroke-dasharray": "3 3" }, p); txt(p, 0, .5, k + 1, "s-pip off"); mk("line", { x1: -8, y1: 8, x2: 8, y2: -8, stroke: C.ink3, "stroke-width": 1.3, "stroke-linecap": "round" }, p);
      const on = S.actor(0, 0, {}, p); mk("circle", { r: 13, fill: C.orange }, on); txt(on, 0, .5, k + 1, "s-pip");
      T.pips.push(on);
    });
    txt(g, 235, 332, "cups that are still possible", "s-lab");
    return T;
  }
  const lift = (S, T, k, t) => { S.to(T.cups[k], { r: 5 }, t, .12, E.io); S.to(T.cups[k], { y: -64, r: -8 }, t + .12, .45, E.back); };
  const drop = (S, T, k, t) => S.to(T.cups[k], { y: 0, r: 0 }, t, .4, E.io);
  const rule = (S, T, k, t) => S.to(T.pips[k], { o: 0, s: .3 }, t, .3);

  function chip(S, x, y, w, label, init, parent) {
    const a = S.actor(x, y, init, parent);
    a.rect = mk("rect", { x: -w / 2, y: -15, width: w, height: 30, rx: 8, fill: C.page, stroke: C.line2 }, a);
    a.txt = txt(a, 0, .5, label, "s-mono");
    return a;
  }
  const tint = (S, rect, kind, t, base = C.page) => S.call(t, .3, p => {
    rect.setAttribute("fill", mix(base, kind === "ok" ? C.tealSoft : C.badSoft, p)); rect.setAttribute("stroke", mix(C.line2, kind === "ok" ? C.teal : C.bad, p)); });
  function badge(S, x, y, kind, parent) {
    const a = S.actor(x, y, { s: 0, o: 0 }, parent), bad = kind === "minus" || kind === "no";
    mk("circle", { r: 11, fill: bad ? C.bad : C.teal, stroke: C.card, "stroke-width": 2.5 }, a);
    txt(a, 0, .5, { plus: "+", minus: "−", ok: "✓", no: "✕" }[kind], "s-badge");
    return a;
  }
  function coin(S, x, y, v) {
    const a = S.actor(x, y, { s: 0, o: 0 });
    mk("circle", { r: 21, fill: v ? C.teal : C.bad }, a); txt(a, 0, 1, v, "s-coin"); txt(a, 0, -33, "reward", "s-lab");
    return a;
  }

  // ---------- chapter 1: one reward after the final answer
  function sceneOutcome(svg) {
    const S = new Scene(svg), T = table(S), X = T.X, RY = [122, 262], SX = [560, 680, 806], CX = 938;
    const row = (e, labels) => {
      const lab = S.actor(0, 0, { o: 0 }); txt(lab, 506, RY[e] - 38, "Episode " + "AB"[e], "s-lab", "start");
      const rail = mk("path", { fill: "none", stroke: e ? C.bad : C.teal, "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", opacity: 0 }, S.g);
      const ch = labels.map((l, j) => chip(S, SX[j], RY[e], 108, l, { x: X[+l.slice(-1) - 1] - SX[j], y: 200 - RY[e], s: .5, o: 0 }));
      return { lab, rail, ch, cn: coin(S, CX, RY[e], e ? 0 : 1), bd: labels.map((l, j) => badge(S, SX[j], RY[e] + 38, e ? "minus" : "plus")) };
    };
    const A = row(0, ["LIFT 1", "LIFT 3", "SUBMIT 4"]), B = row(1, ["LIFT 1", "LIFT 3", "SUBMIT 2"]);
    const yes = badge(S, X[3], 94, "ok"), no = badge(S, X[1], 94, "no");
    const act = (R, j, k, t) => { lift(S, T, k, t); S.to(R.ch[j], { x: 0, y: 0, s: 1, o: 1 }, t + .25, .55, E.io); };
    // the reward travels back under the steps, and each step it passes takes the credit of the whole episode
    const credit = (R, e, t) => {
      const y = RY[e] + 38;
      S.call(t, 1, p => { R.rail.setAttribute("opacity", p > 0 ? 1 : 0); R.rail.setAttribute("d", `M${CX} ${RY[e] + 21}V${y}H${lerp(CX, SX[0], p).toFixed(1)}`); }, E.lin);
      R.ch.forEach((c, j) => { const tj = t + (CX - SX[j]) / (CX - SX[0]); pop(S, R.bd[j], tj, .3); tint(S, c.rect, e ? "bad" : "ok", tj); });
    };
    S.cap(0, "GRPO plays whole episodes, and the only reward comes after the final answer.");
    fade(S, A.lab, .2);
    act(A, 0, 0, .5); rule(S, T, 0, .95);
    act(A, 1, 2, 1.6); rule(S, T, 2, 2.05);
    S.cap(2.7, "Cups 2 and 4 are both still possible, and the agent guesses.");
    pulse(S, T.pips[1], 2.8); pulse(S, T.pips[3], 2.95);
    act(A, 2, 3, 3.9); pop(S, yes, 4.35);
    pop(S, A.cn, 4.8); S.cap(4.8, "The guess happens to be right, and every step of the episode gets the same credit.");
    credit(A, 0, 5.3);
    // a second episode of the group
    fade(S, yes, 7, 0, .2); [0, 2, 3].forEach(k => drop(S, T, k, 7)); [0, 2].forEach(k => S.to(T.pips[k], { o: 1, s: 1 }, 7.1, .3));
    S.cap(7, "In another episode of the same group, the same two lifts end in a wrong guess.");
    fade(S, B.lab, 7.1);
    act(B, 0, 0, 7.5); rule(S, T, 0, 7.9);
    act(B, 1, 2, 8.3); rule(S, T, 2, 8.7);
    act(B, 2, 1, 9.4); pop(S, no, 9.85);
    pop(S, B.cn, 10.2);
    credit(B, 1, 10.6);
    S.cap(11.8, "The same lifts get opposite credit, and the lucky guess is reinforced.");
    [0, 1].forEach(j => { const l = mk("line", { x1: SX[j], x2: SX[j], y1: RY[0] + 54, y2: RY[1] - 20, stroke: C.ink, "stroke-width": 1.5, "stroke-dasharray": "4 4", opacity: 0 }, S.g);
      S.call(11.8 + j * .15, .5, p => { l.setAttribute("opacity", p); l.setAttribute("y2", lerp(RY[0] + 54, RY[1] - 20, p)); });
      [A, B].forEach((R, e) => pulse(S, R.ch[j], 12.2 + j * .15 + e * .08, 1.12)); });
    return S.done(2.6);
  }

  // ---------- chapter 2: PPO with a process reward from a judge that sees the answer
  function sceneJudge(svg) {
    const S = new Scene(svg), T = table(S), X = T.X, g = S.g, RY = [198, 258, 318];
    const BEL = [[1, 2, 3, 4], [2, 3, 4], [4]], SHARE = ["1/4", "1/3", "1/1"], ACT = [["LIFT 1", 0], ["LIFT 3", 2], ["SUBMIT 4", 3]];
    txt(g, 235, 34, "the agent's belief", "s-lab");
    const bub = BEL.map(b => { const a = S.actor(235, 68, { s: 0, o: 0 });
      mk("rect", { x: -124, y: -21, width: 248, height: 42, rx: 21, fill: C.card, stroke: C.line2 }, a); txt(a, -108, .5, "Possible:", "s-mono", "start");
      b.forEach((k, i) => { mk("circle", { cx: -14 + i * 28, cy: 0, r: 11, fill: C.card, stroke: C.orange, "stroke-width": 1.5 }, a); txt(a, -14 + i * 28, .5, k, "s-pip agent"); });
      return a; });
    // the judge and what it sees
    mk("rect", { x: 610, y: 26, width: 396, height: 96, rx: 12, fill: C.page, stroke: C.line2, "stroke-dasharray": "5 4" }, g);
    txt(g, 632, 60, "LLM judge", "s-title", "start"); txt(g, 632, 86, "sees the hidden answer", "s-sub", "start");
    [0, 1, 2, 3].forEach(k => { const x = 866 + k * 34;
      if (k === 3) mk("rect", { x: x - 4.5, y: 86, width: 9, height: 14, rx: 2, fill: C.red }, g);
      mk("path", { d: `M${x - 12} 100L${x - 9} 66H${x + 9}L${x + 12} 100Z`, fill: C.cup, "fill-opacity": .16, stroke: C.cup, "stroke-width": 1.2 }, g); });
    txt(g, 540, 160, "action", "s-lab"); txt(g, 688, 160, "claims in the belief", "s-lab"); txt(g, 880, 160, "score of the judge", "s-lab");
    const rows = BEL.map((b, r) => {
      const cl = S.actor(640, RY[r], { x: 221 - 640, y: 68 - RY[r], s: .85, o: 0 });
      const sq = b.map((k, i) => { const q = mk("rect", { x: i * 32 - 13, y: -13, width: 26, height: 26, rx: 6, fill: C.page, stroke: C.line2, "stroke-width": 1.5 }, cl);
        txt(cl, i * 32, .5, k, "s-pip dark"); return [q, k === 4]; });
      const sc = S.actor(812, RY[r], { s: 0, o: 0 }); txt(sc, 0, 1, SHARE[r], "s-serif");
      const mt = S.actor(0, 0, { o: 0 }); mk("rect", { x: 850, y: RY[r] - 4, width: 132, height: 8, rx: 4, fill: C.soft }, mt);
      const fill = mk("rect", { x: 850, y: RY[r] - 4, width: 0, height: 8, rx: 4, fill: C.ink }, mt);
      const ac = chip(S, 540, RY[r], 100, ACT[r][0], { x: X[ACT[r][1]] - 540, y: 200 - RY[r], s: .5, o: 0 });
      return { cl, sq, sc, mt, fill, ac, share: [1 / 4, 1 / 3, 1][r], bd: badge(S, 476, RY[r], "plus") };
    });
    const yes = badge(S, X[3], 94, "ok");
    const note = S.actor(0, 0, { o: 0 }); txt(note, 190, 358, "cup 2 is still possible", "s-note");
    const judge = (r, t) => {                                        // the belief goes to the judge, which marks each claim
      const R = rows[r];
      S.to(R.cl, { x: 0, y: 0, s: 1, o: 1 }, t, .6, E.io);
      R.sq.forEach(([q, ok], i) => tint(S, q, ok ? "ok" : "bad", t + .7 + i * .14));
      const t2 = t + .8 + R.sq.length * .14;
      pop(S, R.sc, t2); fade(S, R.mt, t2); S.call(t2 + .1, .5, p => R.fill.setAttribute("width", (132 * R.share * p).toFixed(1)));
      return t2;
    };
    const act = (r, t) => { lift(S, T, ACT[r][1], t); S.to(rows[r].ac, { x: 0, y: 0, s: 1, o: 1 }, t + .25, .55, E.io); };
    S.cap(0, "With a process reward, PPO scores every step. Here an LLM judge scores each belief of the agent.");
    pop(S, bub[0], .5); judge(0, 1.1);
    act(0, 2.9); rule(S, T, 0, 3.35);
    S.to(bub[0], { s: .8, o: 0 }, 3.6, .2); pop(S, bub[1], 3.75); judge(1, 4.3);
    act(1, 5.9); rule(S, T, 2, 6.35);
    S.to(bub[1], { s: .8, o: 0 }, 6.6, .2); pop(S, bub[2], 6.75);
    S.cap(6.7, "After two lifts the agent claims cup 4, although cup 2 is still possible.");
    pulse(S, T.pips[1], 7.1, 1.4); pulse(S, T.pips[1], 7.7, 1.4); fade(S, note, 7.1);
    const t3 = judge(2, 8.4);
    S.cap(t3, "The judge sees the answer, and this guess gets the full score.");
    pulse(S, rows[2].sc, t3 + .6, 1.4);
    S.to(bub[2], { s: .8, o: 0 }, 10.6, .2); fade(S, note, 10.6, 0);
    act(2, 10.7); pop(S, yes, 11.15);
    // the outcome reaches the actions through the critic
    const rail = mk("path", { d: "M945 370H476V198", fill: "none", stroke: C.teal, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": "641 641", "stroke-dashoffset": 641 }, g);
    const cn = coin(S, 966, 370, 1), pill = S.actor(740, 370, { s: 0, o: 0 });
    mk("rect", { x: -34, y: -13, width: 68, height: 26, rx: 13, fill: C.card, stroke: C.teal }, pill); txt(pill, 0, .5, "critic", "s-mono small");
    pop(S, cn, 11.6); pop(S, pill, 11.8);
    S.cap(11.6, "The actions are credited through a critic that is fitted to the outcome of the episode.");
    S.call(12, 1.3, p => rail.setAttribute("stroke-dashoffset", (641 * (1 - p)).toFixed(1)), E.io);
    [2, 1, 0].forEach((r, i) => { pop(S, rows[r].bd, 13 + i * .12, .3); tint(S, rows[r].ac.rect, "ok", 13 + i * .12); });
    return S.done(2.6);
  }

  // ---------- chapter 3: RLVB
  const TURNS = [
    { poss: [2, 4], a: "LIFT 2", r: 0, w: .8 },
    { poss: [1, 2, 3, 4], a: "LIFT 2", r: 0, w: .8 },
    { poss: [2, 4], a: "SUBMIT 4", r: -7.42, w: .12 },
    { poss: [4], a: "SUBMIT 4", r: -7.42, w: .12 },
  ];
  const num = v => (v < -.004 ? "−" : "") + Math.abs(v).toFixed(2);
  function sceneRLVB(svg) {
    const S = new Scene(svg), CY = [108, 172, 236, 300];
    // a few games, drawn as rows of steps; every step is a history. The strip starts large, moves to the corner while one
    // history is opened, and comes back for the count at the end.
    const GAMES = [["Lamps", 5], ["MineSweeper", 4], ["Cups", 4]], NX = j => 104 + j * 30, total = GAMES.reduce((s, x) => s + x[1], 0);
    const strip = S.actor(290, 92, { s: 1.9 }), nodes = [], marks = [];
    GAMES.forEach(([name, n], r) => {
      const y = r * 40, t0 = .5 + r * .45;
      txt(strip, 0, y, name, "s-lab", "start");
      const ln = mk("line", { x1: NX(0), x2: NX(0), y1: y, y2: y, stroke: C.line2, "stroke-width": 2, "stroke-linecap": "round" }, strip);
      S.call(t0, (n - 1) * .28 + .1, p => ln.setAttribute("x2", lerp(NX(0), NX(n - 1), p).toFixed(1)), E.lin);
      for (let j = 0; j < n; j++) {
        const idx = nodes.length;
        marks.push([0, 1, 2, 3].map(m => mk("rect", { x: NX(j) - 5, y: y + 10 + m * 4.6, width: 10, height: 3.2, rx: 1.6, opacity: 0,
          fill: (idx * 7 + m * 3) % 5 < 3 ? C.teal : "#d9897c" }, strip)));
        const a = S.actor(NX(j), y, { s: 0, o: 0 }, strip); mk("circle", { r: 6.5, fill: C.ink }, a);
        pop(S, a, t0 + j * .28, .3); nodes.push(a);
      }
    });
    const ring = S.actor(NX(2), 80, { s: 0, o: 0 }, strip); mk("circle", { r: 11.5, fill: "none", stroke: C.orange, "stroke-width": 2.5 }, ring);
    const count = (x, s, cls) => { const a = S.actor(x, 334, { s: .6, o: 0 }); a.t = txt(a, 0, 0, s, "s-count " + cls); return a; };
    const c1 = count(352, GAMES.length + " games", ""), c2 = count(522, "", ""), c3 = count(738, "", "teal");
    const tally = (a, n, label, t, d) => { pop(S, a, t, .3); S.call(t, d, p => { a.t.textContent = "→ " + Math.round(n * p) + " " + label; }, E.lin); };
    const last = S.actor(0, 0, { o: 0 }); txt(last, 520, 372, "With an outcome reward, these " + GAMES.length + " games would give " + GAMES.length + " rewards.", "s-sub", "middle");

    // one history, opened: the table as it stands after two lifts
    const left = S.actor(0, 0, { o: 0 }), T = table(S, left);
    [0, 2].forEach(k => { Object.assign(T.cups[k]._i, { y: -64, r: -8 }); Object.assign(T.pips[k]._i, { o: 0, s: .3 }); });
    mk("rect", { x: 52, y: 98, width: 366, height: 250, rx: 16, fill: "none", stroke: C.ink, "stroke-width": 1.5, "stroke-dasharray": "5 5" }, left);
    mk("line", { x1: 162.8, x2: 162.8, y1: 86, y2: 98, stroke: C.ink, "stroke-width": 1.5, "stroke-dasharray": "3 3" }, left);
    txt(left, 404, 114, "this history", "s-lab dark", "end");
    const right = S.actor(0, 0);
    const head = S.actor(0, 0, { o: 0 }, right); txt(head, 592, 66, "next turns written by the agent", "s-lab");
    const head2 = S.actor(0, 0, { o: 0 }, right); txt(head2, 852, 66, "reward for the action", "s-lab");
    txt(head2, 852, 40, "rewarded from the known posterior", "s-lab blue");
    const scan = S.actor(852, CY[0], { o: 0 }, right); mk("rect", { x: -132, y: -24, width: 264, height: 48, rx: 10, fill: C.page, stroke: C.line2 }, scan);
    const rows = TURNS.map((t, i) => {
      const bar = S.actor(0, 0, { o: 0 }, right); mk("rect", { x: 470, y: CY[i] + 28, width: 244, height: 5, rx: 2.5, fill: C.soft }, bar);
      const fill = mk("rect", { x: 470, y: CY[i] + 28, width: 98, height: 5, rx: 2.5, fill: C.line2 }, bar);
      const c = S.actor(592, CY[i], { x: 235 - 592, y: 200 - CY[i], s: .3, o: 0 }, right);
      c.rect = mk("rect", { x: -122, y: -22, width: 244, height: 44, rx: 10, fill: C.card, stroke: C.line2 }, c);
      [1, 2, 3, 4].forEach((k, j) => { const x = -76 + j * 26;
        if (t.poss.includes(k)) { mk("circle", { cx: x, cy: 0, r: 10, fill: C.card, stroke: C.orange, "stroke-width": 1.5 }, c); txt(c, x, .5, k, "s-pip agent"); }
        else txt(c, x, .5, k, "s-pip faint"); });
      txt(c, 74, .5, t.a, "s-mono");
      const cell = (x, v, cls) => { const a = S.actor(x, CY[i], { o: 0, s: .7 }, right); a.t = txt(a, 0, .5, num(v), "s-num " + cls + (v < -1 ? " low" : "")); a.v = v; return a; };
      return { c, bar, fill, w: t.w, r: cell(852, t.r, "tot") };
    });
    const tick = (a, t) => { pop(S, a, t, .3); S.call(t, .3, p => { a.t.textContent = num(a.v * p); }); };

    // 1. a few games
    S.cap(0, "RLVB starts from games that the current agent plays.");
    pop(S, c1, 2.5);
    // 2. their steps are the histories
    S.cap(3.2, "Every step of these games is a history, and the simulator knows which answers are still possible at each one.");
    nodes.forEach((a, i) => pulse(S, a, 3.3 + i * .07, 1.5));
    tally(c2, total, "histories", 3.4, .9);
    // 3. open one history
    S.cap(5.8, "Take one history of the Cups game. After two lifts, cups 2 and 4 are still possible.");
    [c1, c2].forEach(a => fade(S, a, 5.6, 0, .25));
    S.to(strip, { x: 48 - 290, y: 20 - 92, s: .7 }, 5.8, .8, E.io);
    pop(S, ring, 6.5); fade(S, left, 6.5, 1, .5);
    pulse(S, T.pips[1], 7.1, 1.35); pulse(S, T.pips[3], 7.25, 1.35);
    // 4. the agent writes its next turn several times
    S.cap(8, "At this history the agent writes its next turn several times.");
    fade(S, head, 8);
    rows.forEach((R, i) => S.to(R.c, { x: 0, y: 0, s: 1, o: 1 }, 8.1 + i * .16, .6, E.io));
    // 5. each turn is rewarded at once
    S.cap(9.7, "Each turn is rewarded at once from the known posterior, without playing the game to its end.");
    fade(S, nodes[total - 1], 9.7, .22); fade(S, head2, 9.7); fade(S, scan, 9.8);
    rows.forEach((R, i) => { const t = 10 + i * .65;
      if (i) S.to(scan, { y: CY[i] - CY[0] }, t - .28, .28, E.io);
      tick(R.r, t + .1); });
    fade(S, scan, 12.7, 0);
    S.cap(12.7, "A guess gets a low reward, whether or not it would have been right.");
    [2, 3].forEach((i, j) => { tint(S, rows[i].c.rect, "bad", 12.7 + j * .12, C.card); pulse(S, rows[i].r, 12.8 + j * .12, 1.3); });
    // 6. the update
    S.cap(14.6, "The turns with a higher reward become more probable.");
    rows.forEach((R, i) => { fade(S, R.bar, 14.6);
      S.call(14.8 + i * .08, 1, p => { R.fill.setAttribute("width", (244 * lerp(.4, R.w, p)).toFixed(1)); R.fill.setAttribute("fill", mix(C.line2, i < 2 ? C.teal : "#d9897c", p)); }, E.io);
      if (i > 1) fade(S, R.c, 15.1, .45, .5); });
    [0, 1].forEach(i => tint(S, rows[i].c.rect, "ok", 15, C.card));
    // 7. the same at every history
    S.cap(17, "The same is done at every history of every game, and each one gives its own learning signal.");
    fade(S, left, 17, 0, .4); fade(S, right, 17, 0, .4); fade(S, ring, 17.2, 0); fade(S, nodes[total - 1], 17.2, 1);
    S.to(strip, { x: 0, y: 0, s: 1.9 }, 17.3, .8, E.io);
    [c1, c2].forEach(a => fade(S, a, 18, 1));
    nodes.forEach((a, i) => { const t = 18.3 + i * .14; pulse(S, a, t, 1.4);
      marks[i].forEach((m, k) => S.call(t + .08 + k * .05, .2, p => m.setAttribute("opacity", p))); });
    tally(c3, total * TURNS.length, "rewarded turns", 18.3, total * .14 + .3);
    fade(S, last, 20.8, 1, .5);
    return S.done(3.2);
  }

  // ---------- the player
  const svg = mk("svg", { viewBox: "0 0 1040 400", class: "cr-svg", role: "img", "aria-label": "Animation of one cups game under the three rewards" }, root.querySelector("#cr-canvas"));
  const SC = [sceneOutcome(svg), sceneJudge(svg), sceneRLVB(svg)];
  const tabs = [...root.querySelectorAll(".way")], cap = root.querySelector("#cr-cap"), seek = root.querySelector("#cr-seek"), btn = root.querySelector("#cr-play");
  const bars = tabs.map(t => { t.querySelector(".way-bar").innerHTML = "<i></i>"; return t.querySelector(".way-bar i"); });
  const PLAY = '<svg viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"/></svg>', PAUSE = '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>';
  let ch = 0, t = 0, playing = !still, inView = false, last = 0, raf = 0, shown = "", drawn = null, held = false;

  function draw() {
    const S = SC[ch];
    SC.forEach((x, i) => { x.g.style.display = i === ch ? "" : "none"; });
    S.render(Math.min(t, S.dur));
    S.g.setAttribute("opacity", clamp(Math.min(t / .3, (S.dur - t) / .3)).toFixed(3));
    const c = S.caption(t);
    if (c !== shown) { shown = c; cap.textContent = c; cap.classList.remove("in"); void cap.offsetWidth; cap.classList.add("in"); }
    tabs.forEach((x, i) => { x.classList.toggle("on", i === ch); x.setAttribute("aria-selected", i === ch); bars[i].style.transform = `scaleX(${i === ch ? (t / S.dur).toFixed(4) : 0})`; });
    if (!held) seek.value = Math.round(t / S.dur * 1000);
    if (drawn !== playing) { drawn = playing; btn.innerHTML = playing ? PAUSE : PLAY; btn.setAttribute("aria-label", playing ? "Pause" : "Play"); btn.title = playing ? "Pause" : "Play"; }
  }
  function advance(now) {
    if (!held) t += Math.min(.25, (now - last) / 1000);
    last = now;
    if (t >= SC[ch].dur) { ch = (ch + 1) % SC.length; t = 0; }
    draw();
  }
  function frame(now) {
    raf = 0;
    if (!(playing && inView)) return;
    advance(now); raf = requestAnimationFrame(frame);
  }
  const run = () => { draw(); last = performance.now(); if (playing && inView && !raf) raf = requestAnimationFrame(frame); };
  // the stage plays only while it is on screen; where the browser holds back animation frames (an embedded or covered
  // view), this timer also keeps the clock moving
  const stage = root.querySelector(".cr-stage");
  const look = () => { const r = stage.getBoundingClientRect(), was = inView;
    inView = r.bottom > innerHeight * .25 && r.top < innerHeight * .75; if (inView !== was) run(); };
  setInterval(() => { look(); const now = performance.now(); if (playing && inView && !document.hidden && now - last > 180) advance(now); }, 120);
  addEventListener("scroll", look, { passive: true });
  tabs.forEach((x, i) => x.onclick = () => { ch = i; t = 0; playing = true; run(); });
  btn.onclick = () => { playing = !playing; run(); };
  seek.oninput = () => { t = seek.value / 1000 * SC[ch].dur * .999; run(); };
  seek.addEventListener("pointerdown", () => { held = true; });
  ["pointerup", "pointercancel", "change", "blur"].forEach(e => seek.addEventListener(e, () => { held = false; }));
  if (still) { ch = 2; t = SC[2].dur - 1; }                          // without motion: the end of the RLVB chapter, and no autoplay
  draw();
  look();
  document.addEventListener("visibilitychange", () => { if (!document.hidden) run(); });
})();
