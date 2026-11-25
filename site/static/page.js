// The parts of the page around the episode explorer: the hero animation, the turn example, the charts and the benchmark cards.
(() => {
  const q = id => document.getElementById(id);
  const h = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PLAY = '<svg viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"/></svg>', PAUSE = '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>';

  // ---------- numbers of the paper (accuracy in %, mean of three seeds; columns HiddenGames, Pro-Worlds, RoboQuest)
  const MODELS = { vl8: "Qwen3-VL-8B-Thinking", vl4: "Qwen3-VL-4B-Thinking", q35: "Qwen3.5-4B" };
  const BENCH = [["HiddenGames", "unseen games of the training tasks"], ["Pro-Worlds", "never seen in training"], ["RoboQuest", "never seen in training"]];
  const ACC = [
    ["Base", "", { q35: [26.5, 20.2, 39.6], vl4: [24.6, 17.2, 28.4], vl8: [22.9, 14.8, 36.2] }],
    ["Base (w/ belief)", "", { q35: [38.1, 23.9, 41.3], vl4: [24.5, 18.2, 34.0], vl8: [23.9, 20.5, 39.8] }],
    ["SFT (action-only)", "", { q35: [26.9, 13.6, 38.3], vl4: [31.2, 18.4, 34.5], vl8: [27.0, 19.3, 42.5] }],
    ["PPO (action-only)", "", { q35: [22.9, 11.6, 12.6], vl4: [26.5, 14.4, 20.9], vl8: [23.7, 12.6, 39.2] }],
    ["GRPO (action-only)", "", { q35: [32.5, 15.1, 36.9], vl4: [24.9, 15.2, 24.3], vl8: [23.1, 14.9, 34.7] }],
    ["PPO (belief + action)", "", { q35: [40.5, 29.1, 57.1], vl4: [25.9, 15.1, 38.7], vl8: [26.4, 16.9, 51.4] }],
    ["RLVB (EM)", "ours", { q35: [42.9, 28.1, 47.3], vl4: [31.5, 23.1, 42.6], vl8: [29.0, 32.4, 47.2] }],
    ["RLVB", "ours main", { q35: [48.5, 32.1, 54.0], vl4: [36.2, 28.7, 61.2], vl8: [31.1, 29.9, 62.7] }],
    ["RLVB, without the belief line", "ours", { q35: [38.9, 19.3, 57.9], vl4: [36.1, 27.7, 46.7], vl8: [32.2, 25.4, 56.2] }],
  ];
  // how episodes end, in % of all episodes: right with enough evidence, right too early, wrong too early, wrong with enough, no answer
  const ENDS = {
    q35: [["Base", [17.8, 9.9, 59.1, 3.5, 9.6]], ["GRPO (action-only)", [21.6, 9.9, 62.1, 5.6, 0.7]],
          ["Base (w/ belief)", [26.8, 11.0, 54.4, 6.6, 1.3]], ["PPO (belief + action)", [34.8, 8.8, 45.4, 7.8, 3.2]],
          ["RLVB (EM)", [33.4, 8.5, 42.4, 8.1, 7.7]], ["RLVB", [40.2, 8.3, 31.0, 9.8, 10.6]]],
    vl4: [["Base", [11.0, 11.5, 74.8, 1.7, 0.9]], ["GRPO (action-only)", [9.3, 11.5, 77.1, 1.5, 0.6]],
          ["Base (w/ belief)", [10.1, 13.4, 72.2, 2.4, 1.9]], ["PPO (belief + action)", [15.0, 12.7, 68.4, 1.8, 2.1]],
          ["RLVB (EM)", [19.0, 10.9, 60.8, 7.1, 2.3]], ["RLVB", [36.1, 8.0, 32.4, 14.1, 9.4]]],
    vl8: [["Base", [11.8, 11.7, 72.8, 2.6, 1.2]], ["GRPO (action-only)", [11.2, 11.6, 73.1, 2.9, 1.3]],
          ["Base (w/ belief)", [15.4, 11.4, 67.4, 4.8, 1.0]], ["PPO (belief + action)", [23.3, 10.2, 58.2, 6.6, 1.7]],
          ["RLVB (EM)", [20.2, 9.8, 62.6, 6.2, 1.2]], ["RLVB", [34.2, 8.1, 33.9, 14.6, 9.2]]],
  };
  const END_KEYS = [["re", "Right, with enough evidence"], ["rt", "Right, too early"], ["wt", "Wrong, too early"],
                    ["we", "Wrong, with enough evidence"], ["na", "No answer"]];
  // answers given too early (%) and steps per episode: HiddenGames, Pro-Worlds, RoboQuest
  const PROC = {
    q35: [["Base", [80.8, 1.9, 98.0, 12.4, 52.8, 5.5]], ["Base (w/ belief)", [64.2, 3.8, 95.8, 18.0, 48.5, 6.0]],
          ["PPO (belief + action)", [58.9, 4.5, 91.0, 24.1, 29.5, 7.8]], ["RLVB (EM)", [48.7, 7.2, 88.2, 25.4, 40.9, 6.6]],
          ["RLVB", [34.7, 8.2, 83.7, 29.3, 28.9, 8.0]], ["RLVB, without the belief line", [18.2, 14.7, 83.4, 29.2, 27.4, 8.2]]],
    vl4: [["Base", [93.1, 3.3, 99.9, 10.6, 72.1, 3.1]], ["Base (w/ belief)", [95.1, 3.6, 100.0, 9.8, 61.8, 4.2]],
          ["PPO (belief + action)", [94.3, 3.8, 100.0, 3.9, 59.3, 4.7]], ["RLVB (EM)", [77.4, 4.3, 98.2, 17.3, 52.4, 5.0]],
          ["RLVB", [34.5, 8.3, 90.4, 31.7, 21.8, 10.0]], ["RLVB, without the belief line", [48.7, 6.4, 91.5, 30.2, 38.3, 6.8]]],
    vl8: [["Base", [97.4, 3.1, 100.0, 5.3, 62.9, 4.7]], ["Base (w/ belief)", [88.9, 3.5, 100.0, 10.6, 56.0, 5.0]],
          ["PPO (belief + action)", [81.4, 3.8, 99.0, 10.0, 37.6, 7.4]], ["RLVB (EM)", [81.3, 4.1, 98.8, 16.6, 47.3, 5.9]],
          ["RLVB", [43.1, 8.4, 88.5, 33.9, 20.0, 10.1]], ["RLVB, without the belief line", [43.4, 6.1, 90.6, 29.9, 28.8, 8.4]]],
  };
  const COST = [["PPO (belief + action)", 120.1, ""], ["RLVB", 74.2, "ours main"], ["RLVB (EM)", 29.7, "ours"]];
  // commits (% of turns) with insufficient evidence while the belief line names several answers: Base (w/ belief), RLVB
  const COMMIT = [
    ["Tabletop", [["Qwen3.5-4B", 7.9, 0.1], ["Qwen3-VL-4B", 67.9, 0.6], ["Qwen3-VL-8B", 29.2, 0.1]]],
    ["MiniGrid, MineSweeper", [["Qwen3.5-4B", 15.7, 0.1], ["Qwen3-VL-4B", 28.9, 1.3], ["Qwen3-VL-8B", 21.3, 0.1]]],
    ["Pro-Worlds", [["Qwen3.5-4B", 3.2, 0.9], ["Qwen3-VL-4B", 8.5, 2.0], ["Qwen3-VL-8B", 7.5, 2.2]]],
  ];
  // episodes answered with sufficient evidence (%): Base, PPO (belief + action), RLVB
  const SUFF = [["HiddenGames", 8, 20, 51], ["Pro-Worlds", 0.4, 3, 12], ["RoboQuest", 36, 57, 74]];
  // gain of RLVB over Base in points: from more evidence, at the same evidence
  const SPLIT = [["HiddenGames", 26.7, -8.5], ["Pro-Worlds", 15.8, -2.8], ["RoboQuest", 31.8, 0.3]];

  let model = "vl4";
  const sign = d => (d >= 0 ? "+" : "−") + Math.abs(d).toFixed(1);

  function accCharts() {
    // one column of method names, then one column of bars per benchmark, with the rows on shared lines
    const max = 66, rowsData = ACC.filter(r => r[2][model]);
    const names = `<div class="chart names" aria-hidden="true"><div class="chart-head"></div><div class="bars">${rowsData.map(([n, cls]) =>
      `<div class="bar-row ${cls}"><span class="bar-name">${h(n)}</span></div>`).join("")}</div></div>`;
    q("acc-charts").innerHTML = names + BENCH.map(([name, sub], bi) => {
      const base = ACC[0][2][model][bi], best = Math.max(...rowsData.map(r => r[2][model][bi]));
      const rows = rowsData.map(([n, cls, v]) => {
        const x = v[model][bi], d = x - base;
        const delta = cls.includes("ours") ? `<i class="${d >= 0 ? "up" : "down"}">${sign(d)}</i>` : "";
        return `<div class="bar-row ${cls} ${x === best ? "best" : ""}" role="img" aria-label="${h(n)}: ${x.toFixed(1)}"><span class="bar-name">${h(n)}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${(x / max * 100).toFixed(2)}%"></span>
          <span class="bar-val" style="left:${(x / max * 100).toFixed(2)}%">${x.toFixed(1)}${delta}</span></span></div>`;
      }).join("");
      return `<div class="chart"><div class="chart-head"><h3>${name}</h3><span>${sub}</span></div>
        <div class="bars" style="--base:${(base / max * 100).toFixed(2)}%">${rows}</div></div>`;
    }).join("");
  }

  function endings() {
    q("end-legend").innerHTML = END_KEYS.map(([k, t]) => `<span><i class="sw ${k}"></i>${t}</span>`).join("");
    q("endings").innerHTML = ENDS[model].map(([name, v]) => {
      const segs = v.map((x, i) => `<span class="seg-${END_KEYS[i][0]} sg" style="flex:${x} 0 0" title="${END_KEYS[i][1]}: ${x.toFixed(1)}%">${x >= 7 ? x.toFixed(0) : ""}</span>`).join("");
      return `<div class="end-row ${name === "RLVB" ? "ours main" : name.startsWith("RLVB") ? "ours" : ""}"><span class="bar-name">${h(name)}</span><span class="stack">${segs}</span></div>`;
    }).join("");
  }

  function process() {
    const rows = PROC[model], best = [0, 2, 4].map(i => Math.min(...rows.map(r => r[1][i])));
    const head = `<tr><th rowspan="2">Method</th>${BENCH.map(b => `<th colspan="2" class="grp">${b[0]}</th>`).join("")}</tr>
      <tr>${BENCH.map(() => `<th class="first">early %</th><th>steps</th>`).join("")}</tr>`;
    const body = rows.map(([name, v]) => `<tr class="${name === "RLVB" ? "ours main" : name.startsWith("RLVB") ? "ours" : ""}"><th>${h(name)}</th>${[0, 2, 4].map((i, k) =>
      `<td class="first ${v[i] === best[k] ? "best" : ""}">${v[i].toFixed(1)}</td><td>${v[i + 1].toFixed(1)}</td>`).join("")}</tr>`).join("");
    q("process").innerHTML = `<table class="tbl num proc"><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }

  function cost() {
    const max = 130;
    q("cost").innerHTML = COST.map(([name, x, cls]) => `<div class="bar-row ${cls}"><span class="bar-name">${h(name)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${x / max * 100}%"></span>
      <span class="bar-val" style="left:${x / max * 100}%">${x.toFixed(1)} h</span></span></div>`).join("");
  }

  function fullTable() {
    const ks = ["vl8", "vl4", "q35"], best = {};
    ks.forEach(k => [0, 1, 2].forEach(i => best[k + i] = Math.max(...ACC.filter(r => r[2][k]).map(r => r[2][k][i]))));
    const head = `<tr><th rowspan="2">Method</th>${ks.map(k => `<th colspan="3" class="grp">${MODELS[k]}</th>`).join("")}</tr>
      <tr>${ks.map(() => `<th class="first">HiddenGames</th><th>Pro-Worlds</th><th>RoboQuest</th>`).join("")}</tr>`;
    const body = ACC.map(([name, cls, v]) => `<tr class="${cls}"><th>${h(name)}</th>${ks.map(k => (v[k] || [null, null, null]).map((x, i) =>
      `<td class="${i === 0 ? "first " : ""}${x !== null && x === best[k + i] ? "best" : ""}">${x === null ? "–" : x.toFixed(1)}</td>`).join("")).join("")}</tr>`).join("");
    q("full-table").innerHTML = `<table class="tbl num"><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }

  function commitChart() {
    const max = 70, x = v => (v / max * 100).toFixed(2);
    const axis = `<div class="dumb-axis"><span></span><div>${[0, 20, 40, 60].map(t => `<i style="left:${x(t)}%">${t}%</i>`).join("")}</div></div>`;
    q("commit-chart").innerHTML = axis + COMMIT.map(([grp, rows]) => `<div class="dumb-grp">${h(grp)}</div>` + rows.map(([m, b, r]) =>
      `<div class="dumb-row"><span class="dumb-name">${h(m)}</span><div class="dumb-track">
        <span class="dumb-line" style="left:${x(r)}%;width:${x(b - r)}%"></span>
        <span class="dumb-dot ours" style="left:${x(r)}%"></span><span class="dumb-dot base" style="left:${x(b)}%"></span>
        <span class="dumb-val" style="left:${x(b)}%">${b.toFixed(1)} <em>&rarr; ${r.toFixed(1)}</em></span></div></div>`).join("")).join("");
  }

  function suff() {
    const who = ["Base", "PPO (belief + action)", "RLVB"];
    q("suff").innerHTML = SUFF.map(([name, ...v]) => `<div class="suff-card"><h4>${name}</h4>
      <div class="suff-big"><b>${v[2]}%</b><span>against ${v[0]}% for Base</span></div>
      ${v.map((x, i) => `<div class="bar-row ${i === 2 ? "ours main" : ""}"><span class="bar-name">${who[i]}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${x / .8}%"></span><span class="bar-val" style="left:${x / .8}%">${x}</span></span></div>`).join("")}</div>`).join("");
  }

  function splitBars() {
    const max = 34, neg = 10, w = v => (Math.abs(v) / (max + neg) * 100).toFixed(2), zero = (neg / (max + neg) * 100).toFixed(2);
    q("splitbars").innerHTML = SPLIT.map(([name, more, same]) => `<div class="split-row"><span class="bar-name">${name}</span>
      <div class="split-track" style="--zero:${zero}%">
        <div class="split-line"><span class="sb more" style="left:${zero}%;width:${w(more)}%"></span><em style="left:calc(${zero}% + ${w(more)}% + 8px)">${sign(more)}</em></div>
        <div class="split-line"><span class="sb same" style="${same < 0 ? `right:${100 - zero}%` : `left:${zero}%`};width:${w(same)}%"></span>
          <em style="${same < 0 ? `right:calc(${100 - zero}% + ${w(same)}% + 8px)` : `left:calc(${zero}% + ${w(same)}% + 8px)`}">${sign(same)}</em></div>
      </div></div>`).join("");
  }

  function results() {
    accCharts(); endings(); process(); cost(); fullTable(); commitChart(); suff(); splitBars();
    q("model-seg").querySelectorAll("button").forEach(b => b.onclick = () => {
      model = b.dataset.k;
      q("model-seg").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
      accCharts(); endings(); process();
    });
  }

  // ---------- the turn example (the first figure of the paper). The reward of RLVB is the action term.
  const TURNS = [
    { b: "cups 2, 4", a: "LIFT 2", sb: "−0.07", sa: "0.00", judge: "1/2", out: "none yet",
      note: "The action lifts another cup while two cups are still possible, which is what the teacher does at this history. The reward is zero, the highest possible." },
    { b: "all", a: "LIFT 2", sb: "−2.64", sa: "0.00", judge: "1/4", out: "none yet",
      note: "The belief line still lists cups 1 and 3, which the evidence has ruled out. The reward does not read the line, and the action is the same as in the first turn." },
    { b: "cups 2, 4", a: "SUBMIT 4", sb: "−0.07", sa: "−7.42", judge: "1/2", out: "1 or 0",
      note: "The answer is a guess between two cups. RLVB gives the same low reward whether or not the cylinder is under cup 4, while the outcome reward pays 1 when the guess happens to be right." },
    { b: "cup 4", a: "SUBMIT 4", sb: "−9.92", sa: "−7.42", judge: "1 or 0", out: "1 or 0",
      note: "The belief line drops cup 2, which is still possible, and the answer is a guess. A judge that sees the hidden answer and the outcome reward both pay in full when the guess happens to be right." },
  ];
  function turnLab() {
    const opts = q("turn-opts"), out = q("turn-scores");
    opts.innerHTML = TURNS.map((t, i) => `<button role="radio" aria-checked="${i === 0}" class="turn-opt ${i === 0 ? "on" : ""}" data-i="${i}">
      <code>Possible: ${t.b}</code><code class="act">ACTION: ${t.a}</code></button>`).join("");
    const show = i => {
      const t = TURNS[i];
      opts.querySelectorAll("button").forEach((b, j) => { b.classList.toggle("on", i === j); b.setAttribute("aria-checked", i === j); });
      out.innerHTML = `<div class="score ours"><span class="score-k">RLVB, exact posterior</span><b class="${t.sa === "0.00" ? "" : "low"}">${t.sa}</b>
          <span class="score-sub">action against the teacher</span></div>
        <div class="score"><span class="score-k">Judge that sees the answer</span><b>${t.judge}</b><span class="score-sub">share of true claims in the belief</span></div>
        <div class="score"><span class="score-k">Outcome reward</span><b>${t.out}</b><span class="score-sub">${t.out === "none yet" ? "it comes at the end of the episode" : "depends on where the cylinder is"}</span></div>
        <p class="score-note">${t.note}</p>`;
    };
    opts.querySelectorAll("button").forEach(b => b.onclick = () => show(+b.dataset.i));
    show(0);
  }

  // ---------- hero: the idea as one continuous animation in four chapters. The cups, the belief line and the action are
  // drawn from the time alone; the posterior bars and the reward ease towards the value that the state of the game gives.
  function film() {
    const box = q("film-canvas"); if (!box) return;
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg"); svg.setAttribute("class", "hero-svg"); svg.setAttribute("aria-hidden", "true"); box.appendChild(svg);
    const root = document.createElementNS(NS, "g"); svg.appendChild(root);
    const game = document.createElementNS(NS, "g"); root.appendChild(game);
    const mk = (tag, a, par, text) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); if (text != null) n.textContent = text; (par || game).appendChild(n); return n; };
    const cl = p => Math.max(0, Math.min(1, p)), E = p => 1 - Math.pow(1 - cl(p), 3), IO = p => { p = cl(p); return p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; };
    const BK = p => { p = cl(p); return 1 + 2.70158 * Math.pow(p - 1, 3) + 1.70158 * Math.pow(p - 1, 2); }, sg = (t, a, d) => E((t - a) / d);
    const X = [120, 250, 380, 510], TY = 300, RY = k => 142 + k * 42;
    const mq = matchMedia("(max-width: 900px)"); let portrait = mq.matches;
    const setBox = () => { svg.setAttribute("viewBox", portrait ? "30 20 580 1100" : "0 0 1040 514"); svg.classList.toggle("portrait", portrait); };
    setBox();
    // the table, the two hidden objects and the four cups
    mk("ellipse", { cx: 315, cy: TY + 16, rx: 268, ry: 15, class: "h-shadow" });
    mk("rect", { x: 52, y: TY, width: 526, height: 12, rx: 6, class: "h-table" });
    const shades = X.map(x => mk("ellipse", { cx: x, cy: TY + 2, rx: 44, ry: 6, class: "h-cshadow" }));
    const cyl = (x, cls) => { mk("rect", { x: x - 14, y: TY - 36, width: 28, height: 36, class: cls }); mk("ellipse", { cx: x, cy: TY - 36, rx: 14, ry: 5, class: cls + " top" }); };
    cyl(X[0], "h-green"); cyl(X[3], "h-red");
    const ring = mk("ellipse", { cx: X[3], cy: TY - 50, rx: 60, ry: 72, class: "h-ring" }), ring2 = mk("ellipse", { cx: X[1], cy: TY - 50, rx: 60, ry: 72, class: "h-ring ok" });
    const cups = X.map((x, k) => { const g = mk("g", {});
      mk("path", { d: "M-42 0L-32 -104H32L42 0Z", class: "h-cup" }, g); mk("path", { d: "M-42 0L-32 -104H-17L-23 0Z", class: "h-cup-hi" }, g);
      mk("ellipse", { cx: 0, cy: -104, rx: 32, ry: 8, class: "h-cup-top" }, g); mk("text", { x: 0, y: -42, class: "h-cupno" }, g, k + 1); return g; });
    const pips = X.map((x, k) => { const g = mk("g", { transform: `translate(${x} ${TY + 52})` });
      mk("circle", { r: 16, class: "h-pip" }, g); mk("text", { y: 1, class: "h-pipno" }, g, k + 1); mk("line", { x1: -10, y1: 10, x2: 10, y2: -10, class: "h-strike" }, g); return g; });
    const act = mk("g", {}); mk("rect", { x: -108, y: -21, width: 216, height: 42, rx: 21, class: "h-act" }, act);
    const actOld = mk("text", { class: "h-acttxt" }, act), actNew = mk("text", { class: "h-acttxt" }, act);
    mk("text", { x: 315, y: TY + 108, class: "h-lab" }, game, "BELIEF LINE WRITTEN BY THE AGENT");
    const bel = mk("text", { class: "h-belief" });
    const ver = mk("g", { transform: `translate(315 ${TY + 182})` }); mk("rect", { x: -140, y: -15, width: 280, height: 30, rx: 15 }, ver); const verT = mk("text", { y: 1 }, ver);
    // chapter 4: the belief line leads to the action
    const arrow = mk("g", {}), arrowP = mk("path", { pathLength: 1, class: "f-arrow" }, arrow), arrowH = mk("path", { class: "f-arrowhead" }, arrow);
    // in the portrait view the belief line and the pill are wider, so the arrow starts and ends further left
    const setArrow = () => { arrowP.setAttribute("d", portrait ? "M128 438C60 438 40 380 40 250C40 120 60 66 156 66" : "M188 438C64 438 36 380 36 250C36 120 64 66 198 66");
      arrowH.setAttribute("d", portrait ? "M168 66l-13-6.5v13z" : "M207 66l-13-6.5v13z"); };
    setArrow();
    // chapter 2: what a lifted cup shows goes to the simulator
    const links = [[104, "M104 112C104 20 560 20 650 " + RY(0), 10.4], [364, "M364 112C364 50 560 50 650 " + RY(2), 12]]
      .map(([x, d, t0]) => ({ t0, el: mk("path", { d, pathLength: 1, class: "f-link" }, root), dot: mk("circle", { cx: x, cy: 112, r: 3, class: "f-linkdot" }, root) }));
    // what the simulator knows: the exact posterior, and the reward it gives to the next action
    const pan = mk("g", {}, root); mk("rect", { x: 640, y: 34, width: 372, height: 432, rx: 18, class: "f-panel" }, pan);
    mk("text", { x: 668, y: 76, class: "f-lab" }, pan, "EXACT POSTERIOR"); mk("text", { x: 668, y: 98, class: "f-sub" }, pan, "computed by the simulator from the evidence");
    const rows = [0, 1, 2, 3].map(k => { const y = RY(k);
      const hl = mk("rect", { x: 656, y: y - 17, width: 340, height: 34, rx: 8, class: "f-hl" }, pan);
      mk("text", { x: 668, y: y + 1, class: "f-row" }, pan, "cup " + (k + 1)); mk("rect", { x: 730, y: y - 8, width: 200, height: 16, rx: 4, class: "f-track" }, pan);
      return { hl, fill: mk("rect", { x: 730, y: y - 8, width: 0, height: 16, rx: 4, class: "f-fill" }, pan), pct: mk("text", { x: 944, y: y + 1, class: "f-pct" }, pan) }; });
    const rew = mk("g", {}, pan); mk("line", { x1: 668, x2: 984, y1: 318, y2: 318, class: "f-rule" }, rew);
    mk("text", { x: 668, y: 350, class: "f-lab" }, rew, "REWARD FOR THE NEXT ACTION");
    mk("text", { x: 770, y: 374, class: "f-tick" }, rew, "−10"); mk("text", { x: 920, y: 374, class: "f-tick" }, rew, "0");
    const rrow = (y, label) => { const g = mk("g", {}, rew); mk("text", { x: 668, y: y + 1, class: "f-row dark" }, g, label);
      mk("rect", { x: 770, y: y - 2, width: 150, height: 4, rx: 2, class: "f-track" }, g);
      return { g, bar: mk("rect", { x: 920, y: y - 2, width: 0, height: 4, rx: 2, class: "f-neg" }, g), dot: mk("circle", { cx: 920, cy: y, r: 6 }, g), num: mk("text", { x: 990, y: y + 1, class: "f-rnum" }, g) }; };
    const rA = rrow(398, "SUBMIT 4"), rB = rrow(436, "LIFT 2");
    // each cup: the times at which it is lifted (1) and put down (0)
    const EV = [[[.95, 1], [9.6, 0], [10.4, 1], [22, 0], [22.85, 1], [30.2, 0]], [[25.3, 1], [30.2, 0]],
                [[2.65, 1], [9.6, 0], [12, 1], [22, 0], [23.65, 1], [30.2, 0]], [[26.75, 1], [30.2, 0]]];
    const up = (k, t) => { let v = 0; for (const [te, to] of EV[k]) if (t >= te) v += (to - v) * (to ? BK((t - te) / .5) : IO((t - te) / .5)); return v; };
    const ruled = (k, t) => { if (k === 3) return false; let on = false; for (const [te, to] of EV[k]) if (t >= te + (to ? .4 : 0)) on = !!to; return on; };
    // the action line: two stretches of play, [time, action, kind]
    const PLAY_ = [{ a: .8, b: 8.6, acts: [[.8, "LIFT 1", ""], [2.5, "LIFT 3", ""], [6.4, "SUBMIT 4", "bad"]] },
                   { a: 22.7, b: 30.3, acts: [[22.7, "LIFT 1", ""], [23.5, "LIFT 3", ""], [24.85, "LIFT 2", ""], [26.6, "SUBMIT 4", "ok"]] }];
    const CAPS = [[0, "An agent must find the red cylinder under one of four cups."], [4.2, "Its belief line says that two cups are still possible."], [6.4, "It answers anyway, although two cups are still possible."],
      [9, "During training, the simulator knows the exact posterior after every step."], [13.4, "This posterior is the belief supported by the evidence."],
      [15.7, "RLVB gives a guess a low reward, whether or not it turns out right."], [18.6, "A further step gets the highest reward while two answers remain."],
      [22, "After training, the agent plays the same game again."], [24.3, "Two cups are still possible, and the agent acts on its belief line."], [26.6, "It commits once the evidence identifies the answer."]];
    const CH = [[0, "The problem"], [9, "The posterior"], [15.7, "The reward"], [22, "After training"]], DUR = 31;
    const chBox = q("film-ch"), btn = q("film-play"), cap = q("film-cap");
    chBox.innerHTML = CH.map((c, k) => `<button data-k="${k}"><i></i><span>${k + 1}</span>${c[1]}</button>`).join("");
    const chs = [...chBox.querySelectorAll("button")], bump = (t, a, d) => t > a && t < a + d ? Math.sin(Math.PI * (t - a) / d) : 0;
    const fmt = v => (v < -.004 ? "−" : "") + Math.abs(v).toFixed(2);
    let t = 0, playing = !still, inView = false, last = performance.now(), shown = null, P = [.25, .25, .25, .25], vA = 0;
    function draw(dt) {
      const out = [0, 1, 2, 3].filter(k => ruled(k, t)), poss = [0, 1, 2, 3].filter(k => !out.includes(k)), sure = poss.length === 1;
      cups.forEach((g, k) => { const u = up(k, t);
        g.setAttribute("transform", `translate(${X[k]} ${(TY - 76 * u).toFixed(1)}) rotate(${(-9 * u).toFixed(1)})`);
        shades[k].setAttribute("transform", `translate(${X[k]} ${TY + 2}) scale(${(1 - .4 * cl(u)).toFixed(3)}) translate(${-X[k]} ${-TY - 2})`); });
      pips.forEach((g, k) => g.setAttribute("class", "h-pipg" + (out.includes(k) ? " out" : sure ? " sure" : "")));
      bel.textContent = "Possible: " + (poss.length === 4 ? "all" : (sure ? "cup " : "cups ") + poss.map(k => k + 1).join(", "));
      const pulse = 1 + .07 * (bump(t, 4.2, .7) + bump(t, 4.9, .7) + bump(t, 13.6, .8) + bump(t, 24.3, .8));
      bel.setAttribute("class", "h-belief" + (sure ? " ok" : "")); bel.setAttribute("transform", `translate(315 ${TY + 138}) scale(${pulse.toFixed(3)})`);
      // the action pill stays up while the agent plays; only its text changes
      const W = PLAY_.find(w => t >= w.a && t < w.b + .4);
      if (W) { const i = W.acts.reduce((n, a, k) => t >= a[0] ? k : n, 0), a = W.acts[i], p = i ? sg(t, a[0], .25) : 1, vis = Math.min(sg(t, W.a, .3), 1 - sg(t, W.b, .4));
        act.setAttribute("opacity", vis.toFixed(2)); act.setAttribute("transform", `translate(315 ${(66 - 10 * (1 - sg(t, W.a, .3))).toFixed(1)})`);
        act.setAttribute("class", "h-actg " + (p > .5 ? a[2] : i ? W.acts[i - 1][2] : a[2]));
        actNew.textContent = "ACTION: " + a[1]; actNew.setAttribute("y", (1 + 9 * (1 - p)).toFixed(1)); actNew.setAttribute("opacity", p.toFixed(2));
        actOld.textContent = i ? "ACTION: " + W.acts[i - 1][1] : ""; actOld.setAttribute("y", (1 - 9 * p).toFixed(1)); actOld.setAttribute("opacity", (1 - p).toFixed(2));
      } else act.setAttribute("opacity", 0);
      ring.setAttribute("opacity", (sg(t, 6.4, .3) * (1 - sg(t, 8.6, .4)) + sg(t, 15.9, .3) * (1 - sg(t, 18.6, .3))).toFixed(2));
      ring2.setAttribute("opacity", (sg(t, 18.6, .3) * (1 - sg(t, 21.1, .3))).toFixed(2));
      const late = t >= 20;
      ver.setAttribute("opacity", (late ? sg(t, 27.4, .35) * (1 - sg(t, 30.2, .4)) : sg(t, 6.9, .35) * (1 - sg(t, 8.6, .4))).toFixed(2));
      ver.setAttribute("class", "h-ver " + (late ? "ok" : "bad")); verT.textContent = late ? "correct, and backed by the evidence" : "a guess, right half of the time";
      const ar = sg(t, 24.3, .5); arrow.setAttribute("opacity", (t >= 24.3 ? 1 - sg(t, 26.3, .4) : 0).toFixed(2));
      arrowP.setAttribute("stroke-dasharray", "1 1"); arrowP.setAttribute("stroke-dashoffset", (1 - ar).toFixed(3)); arrowH.setAttribute("opacity", ar > .95 ? 1 : 0);
      links.forEach(l => { const p = sg(t, l.t0, .45); l.el.setAttribute("stroke-dasharray", "1 1"); l.el.setAttribute("stroke-dashoffset", (1 - p).toFixed(3));
        const o = portrait || t < l.t0 ? 0 : (1 - sg(t, 14, .5)).toFixed(2); l.el.setAttribute("opacity", o); l.dot.setAttribute("opacity", o); });
      // the panel slides in at the second chapter; its bars follow what the lifted cups have shown
      const pin = sg(t, 8.8, .8) * (1 - sg(t, 21.2, .6)), k7 = dt == null ? 1 : 1 - Math.exp(-dt * 7);
      pan.setAttribute("opacity", pin.toFixed(2)); pan.setAttribute("transform", portrait ? `translate(-844 ${(452 + 40 * (1 - pin)).toFixed(1)}) scale(1.4)` : `translate(${(40 * (1 - pin)).toFixed(1)} 0)`);
      game.setAttribute("transform", portrait ? `translate(0 ${(300 * (1 - pin)).toFixed(1)})` : `translate(${(205 * (1 - pin)).toFixed(1)} 0)`);
      rows.forEach((r, k) => { const target = poss.includes(k) ? 1 / poss.length : 0; P[k] += (target - P[k]) * k7;
        r.fill.setAttribute("width", (200 * P[k]).toFixed(1)); r.fill.setAttribute("class", "f-fill" + (sure && k === poss[0] ? " ok" : "")); r.pct.textContent = Math.round(P[k] * 100) + "%";
        r.hl.setAttribute("opacity", (k === 1 || k === 3 ? bump(t, 13.6, .9) : 0).toFixed(2)); });
      // the reward of each candidate action. Submitting cup 4 is a guess while two cups remain, and the best action once cup 4 is the only one left
      rew.setAttribute("opacity", sg(t, 15.7, .4).toFixed(2));
      vA += ((t >= 16.2 ? -7.42 : 0) - vA) * k7;
      rA.g.setAttribute("opacity", sg(t, 15.9, .3).toFixed(2)); rB.g.setAttribute("opacity", sg(t, 18.6, .3).toFixed(2)); rB.dot.setAttribute("r", (6 * BK((t - 18.6) / .35)).toFixed(2));
      [[rA, vA], [rB, 0]].forEach(([r, v]) => { const x = 920 + v * 15; r.dot.setAttribute("cx", x.toFixed(1)); r.bar.setAttribute("x", x.toFixed(1)); r.bar.setAttribute("width", (920 - x).toFixed(1));
        r.num.textContent = fmt(v); const c = v < -.05 ? "bad" : "ok"; r.num.setAttribute("class", "f-rnum " + c); r.dot.setAttribute("class", "f-dot " + c); });
      root.setAttribute("opacity", Math.min(sg(t, 0, .4), 1 - sg(t, 30.65, .3)).toFixed(2));
      let c = ""; for (const [t0, x] of CAPS) if (t >= t0) c = x;
      if (c !== shown) { shown = c; cap.textContent = c; cap.classList.remove("in"); void cap.offsetWidth; cap.classList.add("in"); }
      chs.forEach((b, k) => { const a0 = CH[k][0], a1 = k < CH.length - 1 ? CH[k + 1][0] : DUR;
        b.classList.toggle("on", t >= a0 && t < a1); b.firstChild.style.transform = `scaleX(${cl((t - a0) / (a1 - a0)).toFixed(4)})`; });
      btn.innerHTML = playing ? PAUSE : PLAY; btn.setAttribute("aria-label", playing ? "Pause" : "Play");
    }
    function step(now) { const dt = Math.max(0, Math.min(.25, (now - last) / 1000)); last = now; if (playing && inView) { t += dt; if (t >= DUR) { t = 0; draw(); } else draw(dt); } }
    function frame(now) { step(now); requestAnimationFrame(frame); }
    chs.forEach((b, k) => b.onclick = () => { t = still ? [8.5, 15.5, 21, DUR - 1.3][k] : CH[k][0]; playing = !still; draw(); });
    btn.onclick = () => { playing = !playing; draw(); };
    mq.addEventListener("change", e => { portrait = e.matches; setBox(); setArrow(); draw(); });
    if (still) { t = DUR - 1.3; btn.hidden = true; return draw(); }
    draw();
    new IntersectionObserver(es => { inView = es[0].isIntersecting; }).observe(q("film"));
    // where the browser holds back animation frames (an embedded or covered view), this timer keeps the clock moving
    setInterval(() => { const now = performance.now(); if (!document.hidden && now - last > 180) step(now); }, 120);
    requestAnimationFrame(frame);
  }

  // ---------- parts of the page rise into view, and the bars of a chart grow when the chart is first seen
  function reveal() {
    if (still || !("IntersectionObserver" in window)) return;
    const els = document.querySelectorAll("[data-rv]");
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { rootMargin: "0px 0px -7% 0px" });
    els.forEach(e => { e.classList.add("rv"); io.observe(e); });
  }

  // ---------- benchmark cards
  async function benches() {
    let g;
    try { g = await (await fetch("data/games.json")).json(); } catch (e) { return; }
    const extra = { hidden: "756 test games", proworlds: "300 test games", roboquest: "500 test games" };
    const role = { hidden: "train and test", proworlds: "test only · new", roboquest: "test only" };
    q("bench-cards").innerHTML = g.benches.map(b => {
      const thumbs = b.games.slice(0, 16).map(t => { const x = g.games[t];
        return `<button class="bthumb ${x.pixel ? "pixel" : ""}" data-b="${b.id}" data-t="${t}" title="Open ${h(x.title)}"><img src="${x.thumb}" alt="" loading="lazy"><span>${h(x.title)}</span></button>`; }).join("");
      return `<article class="bcard"><div class="bcard-head"><span class="tag">${role[b.id] || ""}</span><h3>${h(b.name)}</h3>
        <span class="count">${b.count} ${b.id === "roboquest" ? "tasks" : "games"} &middot; ${extra[b.id] || ""}</span></div>
        <p>${h(b.blurb)}</p><div class="bthumbs">${thumbs}</div></article>`;
    }).join("");
    q("bench-cards").querySelectorAll(".bthumb").forEach(el => el.onclick = async () => {
      if (typeof pickBench === "function") await pickBench(el.dataset.b, el.dataset.t, 0);
      q("cases").scrollIntoView({ behavior: still ? "auto" : "smooth" });
    });
  }

  // ---------- citation, navigation
  function misc() {
    const b = q("copy-bib");
    b.onclick = async () => { try { await navigator.clipboard.writeText(q("bib-code").textContent); b.textContent = "Copied"; } catch (e) { b.textContent = "Select and copy"; }
      setTimeout(() => b.textContent = "Copy", 1600); };
    const links = [...document.querySelectorAll(".topnav nav a")];
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting)
      links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)); }), { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll("main > section").forEach(s => io.observe(s));
  }

  results(); turnLab(); film(); benches(); misc(); reveal();
})();
