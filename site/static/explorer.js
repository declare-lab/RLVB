// Episode explorer: pick a game, read how it is played, and step through each method's decisions.
const METHODS = [
  ["base", "Base", "untrained"], ["base_b", "Base (w/ belief)", "untrained, states a belief"],
  ["sft", "SFT", "imitates winning games"], ["grpo", "GRPO", "outcome reward"], ["ppo", "PPO", "outcome reward, critic"],
  ["ppo_ba", "PPO (belief + action)", "outcome + judge reward"], ["rlvb_em", "RLVB (EM)", "posterior reward, ours"],
  ["rlvb_grpo", "RLVB (GRPO)", "posterior reward, ours"],
];
const NAME = Object.fromEntries(METHODS.map(m => [m[0], m[1]]));
const S = { games: null, bench: "hidden", task: null, method: "rlvb_grpo", step: 0, cache: {}, timer: null, ms: 800, reveal: false };
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const ours = m => m.startsWith("rlvb");
const clean = a => String(a || "").replace(/^\s*ACTION:\s*/i, "").trim();

async function init() {
  S.games = await (await fetch("data/games.json")).json();
  tabs(); controls(); nav(); teaser();
  const q = new URLSearchParams(location.search);
  const task = q.get("case");
  if (task && S.games.games[task]) {
    S.method = q.get("m") && NAME[q.get("m")] ? q.get("m") : S.method;
    await pickBench(S.games.games[task].bench, task, +q.get("s") || 0);
  } else {
    await pickBench("hidden", "tabletop_cups", 0);
  }
  if (location.hash) {                                            // a link to a section: scroll once the episode has loaded
    const el = document.querySelector(location.hash);
    if (el) el.scrollIntoView();
  }
}

function frameUrl(task, m, f) {
  const r = S.cache[task].runs[m];
  return `frames/${task}/${m}/${f}.${r.ext}`;
}

// ---------- benchmark cards (section "Benchmarks")
function teaser() {                                              // a picture of the hero opens its game in the explorer
  document.querySelectorAll(".teaser figure[data-task]").forEach(f => f.onclick = async () => {
    const t = f.dataset.task, g = S.games.games[t]; if (!g) return;
    await pickBench(g.bench, t, 0); $("explorer").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function benchCards() {
  if (!$("bench-cards")) return;                                // the page may leave out the benchmark cards
  $("bench-cards").innerHTML = S.games.benches.map(b => {
    const thumbs = (b.covers || []).map(src =>
      `<img src="${src}" alt="" loading="lazy" style="${src.endsWith(".png") ? "image-rendering:pixelated" : ""}">`).join("");
    return `<div class="bcard" data-b="${b.id}"><div class="bcard-thumbs">${thumbs}</div><div class="bcard-body">
      <div class="count">${b.count} ${b.id === "roboquest" ? "tasks" : "games"}</div><h3>${esc(b.name)}</h3><p>${esc(b.blurb)}</p></div></div>`;
  }).join("");
  document.querySelectorAll(".bcard").forEach(el => el.onclick = async () => {
    await pickBench(el.dataset.b); $("cases").scrollIntoView({ behavior: "smooth" });
  });
}

// ---------- tabs and game chips
function tabs() {
  $("bench-tabs").innerHTML = S.games.benches.map(b =>
    `<button class="tab" role="tab" data-b="${b.id}">${esc(b.name)} <span style="opacity:.6;font-weight:500">${b.count}</span></button>`).join("");
  document.querySelectorAll(".tab").forEach(el => el.onclick = () => pickBench(el.dataset.b));
}

async function pickBench(bench, task, step) {
  S.bench = bench;
  document.querySelectorAll(".tab").forEach(el => el.classList.toggle("on", el.dataset.b === bench));
  const b = S.games.benches.find(x => x.id === bench);
  $("game-strip").innerHTML = b.games.map(t => {
    const g = S.games.games[t];
    return `<button class="gchip ${g.pixel ? "pixel" : ""}" data-t="${t}"><img src="${g.thumb}" alt="" loading="lazy">${esc(g.title)}</button>`;
  }).join("");
  document.querySelectorAll(".gchip").forEach(el => el.onclick = () => pickGame(el.dataset.t, 0));
  let rules = $("bench-rules");
  if (!rules) { rules = document.createElement("div"); rules.id = "bench-rules"; rules.className = "bench-rules"; $("explorer").before(rules); }
  rules.innerHTML = `<b>${esc(b.name)}.</b> ${esc(b.rules)}`;
  await pickGame(task && b.games.includes(task) ? task : b.games[0], step || 0);
}

async function pickGame(task, step) {
  stop();
  if (!S.cache[task]) S.cache[task] = await (await fetch(`data/${task}.json`)).json();
  S.task = task; S.reveal = false;
  const rb = $("reveal-board"); rb.classList.remove("on"); rb.textContent = "Show the whole board";
  document.querySelectorAll(".gchip").forEach(el => el.classList.toggle("on", el.dataset.t === task));
  const chip = document.querySelector(`.gchip[data-t="${task}"]`);
  const strip = $("game-strip");                                // scroll the strip only, never the page
  if (chip) strip.scrollLeft = Math.max(0, chip.offsetLeft - strip.clientWidth / 2 + chip.clientWidth / 2);
  if (!S.cache[task].runs[S.method]) S.method = "rlvb_grpo";
  howto(); methodChips(); filmstrips(); preload(task, S.method);
  setStep(Math.min(step || 0, run().steps.length - 1));
}

const run = () => S.cache[S.task].runs[S.method];

// ---------- how to play
function howto() {
  const g = S.games.games[S.task], d = S.cache[S.task];
  const bench = S.games.benches.find(b => b.id === g.bench).name;
  const own = g.bench === "roboquest" ? (d.text || "").split(/(?<=Submit\.)/)[0].trim() : "";
  const rows = [["Goal", g.goal], ["This episode", own], ["The agent sees", g.see], ["Actions", g.act], ["Answer", g.answer],
                ["Budget", g.budget && `at most ${g.budget}`], ["What to watch", g.tip]]
    .filter(r => r[1]).map(r => `<dt>${r[0]}</dt><dd>${fmt(r[1])}</dd>`).join("");
  let cands = "";
  if (g.members && g.members.length) {
    cands = `<dt>Candidates</dt><dd><div class="cands">${g.members.map((m, i) =>
      `<span class="cand ${m === g.truth ? "truth" : ""}">${i + 1}. ${esc(m)}</span>`).join("")}</div></dd>`;
  }
  const reveal = g.truth ? `<div class="reveal" id="reveal">Hidden answer of this episode:
      <button id="reveal-btn">Show</button></div>` : "";
  let prompt = d.text || "";
  if (d.intro) prompt += "\n\n" + d.intro;
  if (d.help) prompt += "\n\nActions:\n" + d.help;
  $("howto").innerHTML = `<div class="gbench">${esc(bench)}</div><h3 class="gtitle">${esc(g.title)}</h3>
    <p class="gq">${esc(g.short)}</p><dl>${rows}${cands}</dl>${reveal}
    <details class="prompt"><summary>The full prompt the agent reads</summary><pre>${esc(prompt.trim())}</pre></details>`;
  const btn = $("reveal-btn");
  if (btn) btn.onclick = () => {
    $("reveal").innerHTML = `Hidden answer of this episode: <span class="ans">${esc(g.truth)}</span>`;
    document.querySelectorAll(".cand.truth").forEach(el => el.classList.add("revealed"));
  };
}

// command words in the rules as code
function fmt(s) {
  return esc(s).replace(/\b((?:SUBMIT|LIFT|PRESS|PUSH|PULL|TURN LEFT|TURN RIGHT|FORWARD|TOGGLE|LEFT|RIGHT|UP|DOWN|MOVE UP|MOVE DOWN|MOVE LEFT|MOVE RIGHT|FIRE|OPEN|LOOK AROUND THE KITCHEN|LOOK AT COUNTER|LIFT THE CLOCHE ON COUNTER|PUT THE|WEIGH|LIFT CUBE|LOOK AT CUBE|LIFT BOWL|LIFT THE PLATE|PULL SLIDER|PUSH SLIDER|OPEN THE LID|PRINT|WATCH THE BALL|PUT PLATE|REMOVE PLATE|INSPECT THE LEGS)(?:[A-Z0-9 :,&lt;&gt;]*[A-Z0-9&gt;])?)/g,
    m => `<code>${m}</code>`);
}

// ---------- methods
function methodChips() {
  const g = S.games.games[S.task];
  $("method-chips").innerHTML = METHODS.map(([m, name]) => {
    const o = g.outcomes[m];
    if (!o) return `<button class="mchip off" disabled title="${esc(name)} has not been run on this episode yet">
      <span class="mname">${esc(name)}</span><span class="mres">pending</span></button>`;
    return `<button class="mchip ${ours(m) ? "ours" : ""}" data-m="${m}" title="${esc(name)}">
      <span class="mname">${esc(name)}</span><span class="mres ${o.ok ? "ok" : "bad"}">${o.ok ? "&#10003;" : "&#10007;"} ${o.n}</span></button>`;
  }).join("");
  document.querySelectorAll(".mchip:not(.off)").forEach(el => el.onclick = () => {
    const playing = !!S.timer;                                    // a new method starts from its first step
    stop(); pickMethod(el.dataset.m, 0);
    if (playing) play();                                          // and keeps playing if the last one was playing
  });
}

function pickMethod(m, step) {
  S.method = m; preload(S.task, m);
  setStep(Math.min(step, run().steps.length - 1));
}

// ---------- one step
function setStep(i) {
  const r = run(), n = r.steps.length;
  S.step = Math.max(0, Math.min(i, n - 1));
  const st = r.steps[S.step], last = S.step === n - 1, g = S.games.games[S.task];
  document.querySelectorAll(".mchip").forEach(el => el.classList.toggle("on", el.dataset.m === S.method));
  const box = $("frame-box");
  box.classList.toggle("pixel", !!g.pixel);
  box.classList.toggle("final-ok", last && r.ok);
  box.classList.toggle("final-bad", last && !r.ok);
  const url = frameUrl(S.task, S.method, st.f), board = S.cache[S.task].board;
  box.classList.toggle("has-board", !!board);
  if (board) showBoard(board, r, st.f, url); else showFrame(url, `${NAME[S.method]}, step ${S.step + 1}`);
  $("step-badge").textContent = `${NAME[S.method]} · step ${S.step + 1} of ${n}`;
  const ob = $("outcome-badge");
  ob.className = "outcome-badge" + (last ? (r.ok ? " ok" : " bad") : "");
  ob.textContent = last ? (r.ok ? "Correct" : "Wrong") : "";
  if (!board) minimap(r, st.f); else minimap({}, 0);
  panel(st, last, r);
  $("slider").max = n - 1; $("slider").value = S.step;
  $("step-count").textContent = `step ${S.step + 1} / ${n}`;
  document.querySelectorAll(".frow").forEach(el => el.classList.toggle("on", el.dataset.m === S.method));
  document.querySelectorAll(".tile").forEach(el => el.classList.toggle("cur", el.dataset.m === S.method && +el.dataset.i === S.step));
  const cur = document.querySelector(".tile.cur");
  if (cur) cur.parentElement.scrollLeft = Math.max(0, cur.offsetLeft - cur.parentElement.clientWidth / 2);
  const q = new URLSearchParams({ case: S.task, m: S.method, s: S.step });
  history.replaceState(null, "", `?${q}${location.hash}`);
}

// two stacked images: the next frame loads behind the current one and fades in, so playback does not flicker
let FRONT = null, BACK = null;
function showFrame(src, alt) {
  if (!FRONT) { FRONT = $("frame-a"); BACK = $("frame-b"); }
  if (FRONT.dataset.src === src) return;
  const swap = () => { if (BACK.dataset.src !== src) return; BACK.classList.add("on"); FRONT.classList.remove("on"); [FRONT, BACK] = [BACK, FRONT]; };
  BACK.onload = swap; BACK.alt = alt || ""; BACK.dataset.src = src; BACK.src = src;
  if (BACK.complete && BACK.naturalWidth) swap();
}
const PRE = new Set();
function preload(task, m) {
  const r = S.cache[task].runs[m]; if (!r) return;
  r.steps.forEach(st => { const u = frameUrl(task, m, st.f); if (!PRE.has(u)) { PRE.add(u); const im = new Image(); im.src = u; } });
}

// Pro-Worlds: the whole board under fog, the sites seen so far uncovered, and the window gliding from site to site
function showBoard(b, r, f, siteUrl) {
  const img = $("board-img"), fog = $("fog"), win = $("win");
  if (img.dataset.src !== b.img) { img.dataset.src = b.img; img.src = b.img; fog.innerHTML = ""; }
  fog.setAttribute("viewBox", `0 0 ${b.cols} ${b.rows}`);
  if (!fog.childElementCount) {
    let h = "";
    for (let y = 1; y <= b.rows; y++) for (let x = 1; x <= b.cols; x++)
      h += `<rect data-k="${x},${y}" x="${x - 1}" y="${y - 1}" width="1" height="1" fill="#eef0f3" stroke="#d6dae0" stroke-width=".035"/>`;
    fog.innerHTML = h;
  }
  const seen = new Set((r.sites || []).slice(0, f + 1).map(s => s.join(",")));
  fog.querySelectorAll("rect").forEach(el => { el.style.opacity = S.reveal || seen.has(el.dataset.k) ? 0 : 1; });
  const [cx, cy] = (r.sites || [[1, 1]])[Math.min(f, (r.sites || [[1, 1]]).length - 1)];
  Object.assign(win.style, { left: `${(cx - 1) / b.cols * 100}%`, top: `${(cy - 1) / b.rows * 100}%`,
                             width: `${100 / b.cols}%`, height: `${100 / b.rows}%` });
  S.siteUrl = siteUrl;
}

function norm(s) { return String(s || "").toLowerCase().replace(/^\s*possible:\s*/, "").replace(/[\s.]+/g, ""); }

function panel(st, last, r) {
  const m = S.method, blocks = [];
  if (S.cache[S.task].board) blocks.push(`<div class="blk view"><div class="lab"><span>What the agent sees</span><span>site ${(r.sites || [])[st.f] ? r.sites[st.f].join(", ") : ""}</span></div>
    <img src="${frameUrl(S.task, m, st.f)}" alt="The window of the agent"></div>`);
  if (st.b) {
    blocks.push(`<div class="blk belief ${ours(m) ? "ours" : ""}"><div class="lab"><span>${m === "ppo_ba" ? "Belief (claims for the judge)" : "Belief"}</span>
      ${st.x && norm(st.b) === norm(st.x) ? '<span class="match yes">matches the exact posterior</span>' : ""}</div>
      <div class="val">${esc(st.b.trim())}</div></div>`);
  } else {
    blocks.push(`<div class="blk belief"><div class="lab"><span>Belief</span></div><div class="none">This method writes only an action.</div></div>`);
  }
  if (st.x) {
    const tag = st.id === true ? '<span class="match yes">answer identified</span>'
              : (last && st.id === false ? '<span class="match no">several answers still possible</span>' : "");
    blocks.push(`<div class="blk exact"><div class="lab"><span>Exact posterior, from the simulator</span>${tag}</div><div class="val">${esc(st.x)}</div></div>`);
  }
  const act = clean(st.a) || "(no action)";
  const tag = st.v === false ? '<span class="noeffect">no effect</span>' : "";
  blocks.push(`<div class="blk action ${last ? "final " + (r.ok ? "ok" : "bad") : ""}"><div class="lab"><span>${last ? "Final action" : "Action"}</span>
    ${last ? `<span>${r.ok ? "correct" : "wrong"}</span>` : ""}</div><div class="val">${esc(act)}${tag}</div></div>`);
  if (st.fb) blocks.push(`<div class="blk reply"><div class="lab"><span>Reply</span></div><div class="val">${esc(st.fb)}</div></div>`);
  $("step-panel").innerHTML = blocks.join("");
}

function minimap(r, f) {
  const svg = $("minimap");
  if (!r.sites || !r.board) { svg.classList.remove("show"); svg.innerHTML = ""; return; }
  const [cols, rows] = r.board, c = 10;
  const seen = new Set(r.sites.slice(0, f).map(s => s.join(",")));
  const [cc, cr] = r.sites[Math.min(f, r.sites.length - 1)];
  let h = `<rect x="0" y="0" width="${cols * c}" height="${rows * c}" fill="#fff" fill-opacity=".93" stroke="#2f4f8f" stroke-width="1"/>`;
  seen.forEach(k => { const [x, y] = k.split(",").map(Number); h += `<rect x="${(x - 1) * c}" y="${(y - 1) * c}" width="${c}" height="${c}" fill="#afc3e6"/>`; });
  h += `<rect x="${(cc - 1) * c}" y="${(cr - 1) * c}" width="${c}" height="${c}" fill="#2f4f8f"/>`;
  svg.setAttribute("viewBox", `-1 -1 ${cols * c + 2} ${rows * c + 2}`);
  svg.innerHTML = h; svg.classList.add("show");
}

// ---------- all methods
function filmstrips() {
  const g = S.games.games[S.task];
  $("filmstrips").innerHTML = METHODS.filter(([m]) => S.cache[S.task].runs[m]).map(([m, name, sub]) => {
    const r = S.cache[S.task].runs[m], n = r.steps.length;
    const tiles = r.steps.map((st, i) => `<button class="tile ${g.pixel ? "pixel" : ""} ${i === n - 1 ? "final " + (r.ok ? "ok" : "bad") : ""}"
        data-m="${m}" data-i="${i}" title="${esc(name)}, step ${i + 1}: ${esc(clean(st.a))}">
        <img src="${frameUrl(S.task, m, st.f)}" alt="" loading="lazy"><span>${i + 1}</span></button>`).join("");
    return `<div class="frow ${ours(m) ? "ours" : ""}" data-m="${m}"><div><div class="fname">${esc(name)}</div>
      <div class="fsub">${esc(sub)} · ${r.ok ? "correct" : "wrong"} in ${n} step${n > 1 ? "s" : ""}</div></div><div class="ftiles">${tiles}</div></div>`;
  }).join("");
  document.querySelectorAll(".tile").forEach(el => el.onclick = () => {
    S.method = el.dataset.m; setStep(+el.dataset.i);
    $("explorer").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

// ---------- playback and keys
function stop() { clearInterval(S.timer); S.timer = null; $("btn-play").innerHTML = "&#x25B6;"; }
function play() {
  if (S.timer) return stop();
  if (S.step >= run().steps.length - 1) setStep(0);
  $("btn-play").innerHTML = "&#x275A;&#x275A;";
  S.timer = setInterval(() => { if (S.step >= run().steps.length - 1) stop(); else setStep(S.step + 1); }, S.ms);
}
function controls() {
  $("btn-first").onclick = () => { stop(); setStep(0); };
  $("btn-prev").onclick = () => { stop(); setStep(S.step - 1); };
  $("btn-next").onclick = () => { stop(); setStep(S.step + 1); };
  $("btn-last").onclick = () => { stop(); setStep(run().steps.length - 1); };
  $("btn-play").onclick = play;
  document.querySelectorAll("#speed button").forEach(b => b.onclick = () => {
    S.ms = +b.dataset.ms; document.querySelectorAll("#speed button").forEach(x => x.classList.toggle("on", x === b));
    if (S.timer) { stop(); play(); }
  });
  $("reveal-board").onclick = () => { S.reveal = !S.reveal; $("reveal-board").classList.toggle("on", S.reveal);
    $("reveal-board").textContent = S.reveal ? "Hide the unseen sites" : "Show the whole board"; setStep(S.step); };
  $("slider").oninput = e => { stop(); setStep(+e.target.value); };
  document.addEventListener("keydown", e => {
    if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName) && document.activeElement.type !== "range") return;
    const r = $("explorer").getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;              // only while the explorer is on screen
    if (e.key === "ArrowRight") { stop(); setStep(S.step + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { stop(); setStep(S.step - 1); e.preventDefault(); }
    else if (e.key === " ") { play(); e.preventDefault(); }
  });
}
function nav() {
  const bar = $("topnav"), hero = document.querySelector(".hero");
  const f = () => bar.classList.toggle("show", hero.getBoundingClientRect().bottom < 60);
  addEventListener("scroll", f, { passive: true }); f();
}

init();
