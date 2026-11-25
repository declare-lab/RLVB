// Episode explorer: one game at a time. The rows show what every method did on that game, one cell per step.
// A click on a row plays that method, and a click on a cell shows that turn.
const METHODS = [["base", "Base"], ["base_b", "Base (w/ belief)"], ["sft", "SFT (action-only)"], ["grpo", "GRPO (action-only)"],
  ["ppo", "PPO (action-only)"], ["ppo_ba", "PPO (belief + action)"], ["rlvb_em", "RLVB (EM)"], ["rlvb_grpo", "RLVB (+ belief term)"]];
const NAME = Object.fromEntries(METHODS);
const S = { games: null, task: null, method: "rlvb_grpo", step: 0, cache: {}, timer: null, ms: 900, reveal: false };
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const ours = m => m.startsWith("rlvb");
const clean = a => String(a || "").replace(/^\s*ACTION:\s*/i, "").trim();
const norm = s => String(s || "").toLowerCase().replace(/^\s*possible:\s*/, "").replace(/[\s.]+/g, "");
const ICON = { play: '<svg viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"/></svg>', pause: '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>' };
const run = () => S.cache[S.task].runs[S.method];
const frameUrl = (task, m, f) => `frames/${task}/${m}/${f}.${S.cache[task].runs[m].ext}`;

async function init() {
  S.games = await (await fetch("data/games.json")).json();
  $("game-select").innerHTML = S.games.benches.map(b => `<optgroup label="${esc(b.name)}">${b.games.map(t =>
    `<option value="${t}">${esc(S.games.games[t].title)}</option>`).join("")}</optgroup>`).join("");
  $("game-select").onchange = e => pickGame(e.target.value, 0);
  controls();
  const q = new URLSearchParams(location.search), task = q.get("case");
  if (task && S.games.games[task]) { if (NAME[q.get("m")]) S.method = q.get("m"); await pickGame(task, +q.get("s") || 0); }
  else await pickGame("tabletop_cups", 0);
  if (/^#[\w-]+$/.test(location.hash)) { const el = document.querySelector(location.hash); if (el) el.scrollIntoView(); }   // a link to a section
}

// the benchmark cards of the page open a game through this
async function pickBench(bench, task, step) {
  const b = S.games.benches.find(x => x.id === bench);
  await pickGame(task && b.games.includes(task) ? task : b.games[0], step || 0);
}

async function pickGame(task, step) {
  stop();
  if (!S.cache[task]) S.cache[task] = await (await fetch(`data/${task}.json`)).json();
  S.task = task; S.reveal = false; $("game-select").value = task;
  const rb = $("reveal-board"); rb.classList.remove("on"); rb.textContent = "Show the whole board";
  if (!S.cache[task].runs[S.method]) S.method = "rlvb_grpo";
  about(); rows(); preload(task, S.method);
  setStep(Math.min(step || 0, run().steps.length - 1));
}

// ---------- the game in one sentence, and its rules behind a fold
function about() {
  const g = S.games.games[S.task], d = S.cache[S.task];
  $("ep-task").innerHTML = esc(g.short) + (g.truth ? ` <span class="ep-ans" id="reveal">Hidden answer <button id="reveal-btn">show</button></span>` : "");
  const own = g.bench === "roboquest" && (d.text || "").includes("Submit.") ? d.text.split("Submit.")[0].trim() + " Submit." : "";
  const items = [["Goal", g.goal], ["This episode", own], ["The agent sees", g.see], ["Actions", g.act], ["Answer", g.answer],
                 ["Budget", g.budget && `at most ${g.budget}`], ["What to watch", g.tip]]
    .filter(r => r[1]).map(r => `<dt>${r[0]}</dt><dd>${fmt(r[1])}</dd>`).join("");
  const cands = g.members && g.members.length ? `<dt>Candidates</dt><dd><div class="cands">${g.members.map((m, i) =>
    `<span class="cand ${m === g.truth ? "truth" : ""}">${i + 1}. ${esc(m)}</span>`).join("")}</div></dd>` : "";
  let prompt = d.text || "";
  if (d.intro) prompt += "\n\n" + d.intro;
  if (d.help) prompt += "\n\nActions:\n" + d.help;
  $("howto").innerHTML = `<summary>How this game works</summary><dl>${items}${cands}</dl>
    <details class="prompt"><summary>The full prompt the agent reads</summary><pre>${esc(prompt.trim())}</pre></details>`;
  const btn = $("reveal-btn");
  if (btn) btn.onclick = () => {
    $("reveal").innerHTML = `Hidden answer <b>${esc(g.truth)}</b>`;
    document.querySelectorAll(".cand.truth").forEach(el => el.classList.add("revealed"));
  };
}

// command words in the rules as code
function fmt(s) {
  return esc(s).replace(/\b((?:SUBMIT|LIFT|PRESS|PUSH|PULL|TURN LEFT|TURN RIGHT|FORWARD|TOGGLE|LEFT|RIGHT|UP|DOWN|MOVE UP|MOVE DOWN|MOVE LEFT|MOVE RIGHT|FIRE|OPEN|LOOK AROUND THE KITCHEN|LOOK AT COUNTER|LIFT THE CLOCHE ON COUNTER|PUT THE|WEIGH|LIFT CUBE|LOOK AT CUBE|LIFT BOWL|LIFT THE PLATE|PULL SLIDER|PUSH SLIDER|OPEN THE LID|PRINT|WATCH THE BALL|PUT PLATE|REMOVE PLATE|INSPECT THE LEGS)(?:(?:[A-Z0-9 :,]|&lt;|&gt;)*(?:[A-Z0-9]|&gt;))?)/g,
    m => `<code>${m}</code>`);
}

// ---------- one row per method: its steps on a shared axis, the last one green or red, and the outcome in words
function rows() {
  const runs = S.cache[S.task].runs, ms = METHODS.filter(([m]) => runs[m]);
  const box = $("filmstrips");
  box.style.setProperty("--n", Math.max(...ms.map(([m]) => runs[m].steps.length)));
  box.innerHTML = ms.map(([m, name]) => {
    const r = runs[m], n = r.steps.length, last = r.steps[n - 1], answered = last.k === "answer";
    const first = r.steps.findIndex(st => st.id === true);
    const cells = r.steps.map((st, i) => {
      const cls = i === n - 1 ? (r.ok ? "ok" : "bad") : (st.v === false || st.k === "invalid" || st.k === "unreadable" ? "void" : "");
      return `<button class="tile ${cls}${i === first ? " enough" : ""}" data-m="${m}" data-i="${i}" aria-label="${esc(name)}, step ${i + 1}: ${esc(clean(st.a))}"></button>`;
    }).join("");
    const steps = `${n} step${n > 1 ? "s" : ""}`;
    const res = r.ok ? `Correct in ${steps}` : answered ? `Wrong after ${steps}${last.id === false ? ", too early" : ""}` : `No answer in ${steps}`;
    return `<div class="frow ${ours(m) ? "ours" : ""}" data-m="${m}"><button class="fname" data-m="${m}">${esc(name)}</button>
      <div class="ftiles">${cells}</div><div class="fres ${r.ok ? "ok" : "bad"}">${res}</div></div>`;
  }).join("");
  box.querySelectorAll(".frow").forEach(el => el.onclick = e => {
    const tile = e.target.closest(".tile"), playing = !!S.timer;
    stop(); S.method = el.dataset.m; preload(S.task, S.method);
    setStep(tile ? +tile.dataset.i : 0);
    if (playing && !tile) play();                                   // a new method keeps playing if the last one was playing
  });
}

// ---------- one step
function setStep(i) {
  const r = run(), n = r.steps.length;
  S.step = Math.max(0, Math.min(i, n - 1));
  const st = r.steps[S.step], last = S.step === n - 1, g = S.games.games[S.task], board = S.cache[S.task].board;
  const box = $("frame-box"), url = frameUrl(S.task, S.method, st.f);
  box.classList.toggle("pixel", !!g.pixel);
  box.classList.toggle("final-ok", last && r.ok);
  box.classList.toggle("final-bad", last && !r.ok);
  box.classList.toggle("has-board", !!board);
  if (board) showBoard(board, r, st.f, url); else showFrame(url, `${NAME[S.method]}, step ${S.step + 1}`);
  const ob = $("outcome-badge");
  ob.className = "outcome-badge" + (last ? (r.ok ? " ok" : " bad") : "");
  ob.textContent = last ? (r.ok ? "Correct" : "Wrong") : "";
  turn(st, last, r);
  $("step-count").innerHTML = `<b>${esc(NAME[S.method])}</b> · step ${S.step + 1} of ${n}`;
  document.querySelectorAll(".frow").forEach(el => el.classList.toggle("on", el.dataset.m === S.method));
  document.querySelectorAll(".tile").forEach(el => el.classList.toggle("cur", el.dataset.m === S.method && +el.dataset.i === S.step));
  history.replaceState(null, "", `?${new URLSearchParams({ case: S.task, m: S.method, s: S.step })}${location.hash}`);
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
  S.cache[task].runs[m].steps.forEach(st => { const u = frameUrl(task, m, st.f); if (!PRE.has(u)) { PRE.add(u); new Image().src = u; } });
}

// Pro-Worlds: the whole board under fog, the sites seen so far uncovered, the window moving from site to site,
// and the one site the agent sees in the corner
function showBoard(b, r, f, siteUrl) {
  const img = $("board-img"), fog = $("fog"), win = $("win");
  if (img.dataset.src !== b.img) { img.dataset.src = b.img; img.src = b.img; fog.innerHTML = ""; }
  fog.setAttribute("viewBox", `0 0 ${b.cols} ${b.rows}`);
  if (!fog.childElementCount) {
    let h = "";
    for (let y = 1; y <= b.rows; y++) for (let x = 1; x <= b.cols; x++)
      h += `<rect data-k="${x},${y}" x="${x - 1}" y="${y - 1}" width="1" height="1" fill="#f0f0f2" stroke="#dcdce0" stroke-width=".035"/>`;
    fog.innerHTML = h;
  }
  const sites = r.sites || [[1, 1]], seen = new Set(sites.slice(0, f + 1).map(s => s.join(",")));
  fog.querySelectorAll("rect").forEach(el => { el.style.opacity = S.reveal || seen.has(el.dataset.k) ? 0 : 1; });
  const [cx, cy] = sites[Math.min(f, sites.length - 1)];
  Object.assign(win.style, { left: `${(cx - 1) / b.cols * 100}%`, top: `${(cy - 1) / b.rows * 100}%`, width: `${100 / b.cols}%`, height: `${100 / b.rows}%` });
  $("inset-img").src = siteUrl;
}

// what the agent wrote at this step, next to what the evidence supports
function turn(st, last, r) {
  const m = S.method, out = [];
  out.push(`<div class="trow belief"><span class="tk">${m === "ppo_ba" ? "Agent's belief, as claims" : "Agent's belief"}</span>` +
    (st.b ? `<span class="tv">${esc(st.b.trim())}</span>` : `<span class="tv none">This method writes only an action.</span>`) + `</div>`);
  if (st.x) {
    const tag = last && st.id === false ? '<i class="pill bad">several answers still possible</i>' : st.id === true ? '<i class="pill ok">answer identified</i>'
              : st.b && norm(st.b) === norm(st.x) ? '<i class="pill">the belief matches</i>' : "";
    out.push(`<div class="trow exact"><span class="tk">Supported by the evidence</span><span class="tv">${esc(st.x)}${tag}</span></div>`);
  }
  const tag = (st.v === false ? '<i class="pill">no effect</i>' : "") + (last ? `<i class="pill ${r.ok ? "ok" : "bad"}">${r.ok ? "correct" : "wrong"}</i>` : "");
  out.push(`<div class="trow action"><span class="tk">${last ? "Final action" : "Action"}</span><span class="tv">${esc(clean(st.a) || "(no action)")}${tag}</span></div>`);
  if (st.fb) out.push(`<div class="trow reply"><span class="tk">Reply</span><span class="tv">${esc(st.fb)}</span></div>`);
  $("step-panel").innerHTML = out.join("");
}

// ---------- playback and keys
function setPlay(on) { const b = $("btn-play"); b.innerHTML = on ? ICON.pause : ICON.play; b.setAttribute("aria-label", on ? "Pause" : "Play"); b.title = on ? "Pause (space)" : "Play (space)"; }
function stop() { clearInterval(S.timer); S.timer = null; setPlay(false); }
function play() {
  if (S.timer) return stop();
  if (S.step >= run().steps.length - 1) setStep(0);
  setPlay(true);
  S.timer = setInterval(() => { if (S.step >= run().steps.length - 1) stop(); else setStep(S.step + 1); }, S.ms);
}
function controls() {
  $("btn-prev").onclick = () => { stop(); setStep(S.step - 1); };
  $("btn-next").onclick = () => { stop(); setStep(S.step + 1); };
  $("btn-play").onclick = play;
  $("reveal-board").onclick = () => { S.reveal = !S.reveal; $("reveal-board").classList.toggle("on", S.reveal);
    $("reveal-board").textContent = S.reveal ? "Hide the unseen sites" : "Show the whole board"; setStep(S.step); };
  document.addEventListener("keydown", e => {
    const el = document.activeElement, r = $("explorer").getBoundingClientRect();
    if (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || r.bottom < 0 || r.top > innerHeight) return;   // only while the explorer is on screen
    if (e.key === "ArrowRight") { stop(); setStep(S.step + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { stop(); setStep(S.step - 1); e.preventDefault(); }
    else if (e.key === " " && !["BUTTON", "SUMMARY", "A"].includes(el.tagName)) { play(); e.preventDefault(); }
  });
}

init();
