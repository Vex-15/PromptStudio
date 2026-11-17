/* Prompt Studio frontend: talks to the FastAPI JSON API, renders with textContent
   (no innerHTML for model/user strings except through esc()). */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));
const pct = (v) => `${Math.round(v * 100)}%`;

const state = { result: null, rows: null, examples: [], voted: null };

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

function showError(message) {
  const box = $("error");
  if (!message) { box.hidden = true; box.textContent = ""; return; }
  box.hidden = false;
  box.textContent = message;
}

/* --------------------------------------------------------------- examples */
async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    state.examples = await res.json();
  } catch {
    state.examples = [];
  }
  const wrap = $("examples");
  wrap.innerHTML = "";
  for (const ex of state.examples) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = ex.name;
    b.addEventListener("click", () => {
      $("prompt").value = ex.text;
      analyze();
    });
    wrap.appendChild(b);
  }
}

/* ---------------------------------------------------------------- analyze */
async function analyze({ scroll = true } = {}) {
  const prompt = $("prompt").value.trim();
  if (!prompt) { showError("Type a prompt first — even three words are enough."); return; }
  showError("");
  const btn = $("run");
  btn.disabled = true;
  $("run-status").textContent = "running both engines…";
  const started = performance.now();
  try {
    const result = await api("/api/analyze", {
      prompt,
      mode: $("mode").value,
      level: $("level").value,
    });
    state.result = result;
    render(result);
    const results = $("results");
    results.hidden = false;
    playReveal(results, "is-fresh");
    $("run-status").textContent = `${Math.round(performance.now() - started)} ms`;
    if (scroll) results.scrollIntoView({ behavior: motionOn() ? "smooth" : "auto", block: "start" });
  } catch (err) {
    showError(`Analysis failed: ${err.message}`);
    $("run-status").textContent = "";
  } finally {
    btn.disabled = false;
  }
}

/* ---------------------------------------------------------------- render */
function render(r) {
  renderScore(r);
  renderSignals(r);
  renderDiagnosis(r);
  renderOutput(r);
  renderCompare(r);
}

function renderScore(r) {
  $("score-before").textContent = r.score.before;
  countUp($("score-after"), r.score.before, r.score.after, 420);

  const delta = r.score.delta;
  const chip = $("score-delta");
  chip.textContent = `${delta >= 0 ? "+" : ""}${delta}`;
  chip.className = "delta" + (delta === 0 ? " delta--flat" : delta < 0 ? " delta--neg" : "");

  $("meta-task").innerHTML = `${esc(r.neural.task)}<small>${pct(r.neural.confidence)} confidence</small>`;
  $("meta-intent").innerHTML = `${esc(r.rubric.intent_name)}<small>${esc(r.rubric.strategy)} strategy</small>`;
  $("meta-engine").innerHTML = `${esc(r.combined.engine === "rubric" ? "rule rubric" : r.combined.engine)}<small>driving the rewrite</small>`;
  $("meta-tokens").innerHTML = `${r.meta.tokens_before} → ${r.meta.tokens_after}<small>${Number(r.rubric.tokens.ratio).toFixed(1)}× tokens</small>`;
}

function renderSignals(r) {
  $("task-note").innerHTML =
    `Detected <b>${esc(r.neural.task)}</b> at ${pct(r.neural.confidence)} confidence · ` +
    `template <b>${esc(r.neural.template)}</b>.`;

  const gaps = Object.entries(r.neural.gaps);
  $("gaps").innerHTML = "";
  for (const [label, value] of gaps) {
    const missing = r.neural.missing.includes(label);
    const row = document.createElement("div");
    row.className = "gap" + (missing ? " is-missing" : "");
    row.innerHTML = `
      <span class="gap__name">${esc(label)}${missing ? " — missing" : ""}</span>
      <span class="gap__val">${pct(value)}</span>
      <span class="gap__track"><span class="gap__fill" style="width:${Math.min(100, Math.max(0, value * 100))}%"></span></span>`;
    $("gaps").appendChild(row);
  }

  const overrides = $("rule-overrides");
  if (r.neural.rule_overrides.length) {
    overrides.hidden = false;
    overrides.textContent = "Rules proved already present: " +
      r.neural.rule_overrides.map((g) => g).join(", ");
  } else {
    overrides.hidden = true;
  }

  paintHeat(r.neural.tokens, r.neural.attention);

  $("gaps-closed").textContent =
    `${r.combined.addressed.length} of ${gaps.length} gaps addressed`;
}

function renderDiagnosis(r) {
  const issues = $("issues");
  issues.innerHTML = "";
  if (!r.rubric.issues.length) {
    const good = document.createElement("div");
    good.className = "issue issue--good";
    good.textContent = "No major issues found by the rubric.";
    issues.appendChild(good);
  }
  for (const [severity, message] of r.rubric.issues) {
    const row = document.createElement("div");
    row.className = `issue issue--${severity}`;
    row.textContent = message;
    issues.appendChild(row);
  }

  const questions = [];
  for (const q of r.rubric.questions) questions.push(["rule", q]);
  for (const q of r.neural.questions) questions.push(["neural", q]);

  const box = $("questions");
  if (!questions.length) {
    box.innerHTML = "";
  } else {
    box.innerHTML = `<h4>only you can answer these</h4><ul>${questions.map(
      ([src, q]) => `<li><b>${src}</b>${esc(q)}</li>`
    ).join("")}</ul>`;
  }

  const d = r.rubric.drift;
  const t = r.rubric.tokens;
  const afterGapCount = r.combined.remaining.length;
  $("drift").innerHTML = `
    <div>intent ${d.intent_same ? '<span class="ok">unchanged</span>' : '<span class="warn">changed</span>'}
      · key terms kept <b>${Math.round(d.coverage * 100)}%</b></div>
    <div>original tokens <b>${t.before}</b> → merged <b>${t.after}</b> (${Number(t.ratio).toFixed(1)}×)</div>
    <div>gaps still open after fusion: <b>${afterGapCount}</b></div>
    <div>${d.ok ? '<span class="ok">drift guard passed</span>' : '<span class="warn">review the intent — the rewrite may have moved it</span>'}</div>`;
}

function renderOutput(r) {
  $("out-combined").textContent = r.combined.text;
  $("out-rubric").textContent = r.rubric.optimized;
  $("out-neural").textContent = r.neural.enhanced;

  const remaining = $("remaining");
  remaining.innerHTML = r.combined.remaining.length
    ? `Still weak after fusion: ${r.combined.remaining
        .map((g) => `<span class="pill">${esc(g)}</span>`).join("")}`
    : `<span class="pill pill--none">no gaps left — every part is covered</span>`;

  const sections = $("sections");
  sections.innerHTML = "";
  for (const section of r.combined.sections) {
    const card = document.createElement("div");
    card.className = "section-card";
    card.innerHTML = `
      <h5>${esc(section.heading)}<em>${esc(section.source)}</em></h5>
      <p>${esc(section.lines.join(" · "))}</p>`;
    sections.appendChild(card);
  }

  const changes = $("changes");
  changes.innerHTML = (r.combined.engine === "compressed"
    ? r.rubric.changes
    : r.rubric.changes.concat(["Gap rewrite merged from the neural engine"])
  ).map((c) => `<li>${esc(c)}</li>`).join("");

  const versions = $("versions");
  versions.innerHTML = (r.rubric.versions || []).map(
    ([name, text]) => `<div><h4>${esc(name)}</h4><pre class="code">${esc(text)}</pre></div>`
  ).join("");
}

/* -------------------------------------------------------------- compare */
function renderCompare(r) {
  $("cmp-original-text").textContent = r.prompt;
  $("cmp-merged-text").textContent = r.combined.text;
  $("cmp-original-score").textContent = `${r.score.before}/100`;
  $("cmp-merged-score").textContent = `${r.score.after}/100 (${r.score.delta >= 0 ? "+" : ""}${r.score.delta})`;
  const gapWord = (n) => `${n} of 5 gap${n === 1 ? "" : "s"} open`;
  $("cmp-original-meta").textContent =
    `${r.meta.tokens_before} est. tokens · ${gapWord(r.neural.missing.length)}`;
  $("cmp-merged-meta").textContent =
    `${r.meta.tokens_after} est. tokens · ${gapWord(r.combined.remaining.length)} · ` +
    `${r.combined.addressed.length} of 5 addressed`;

  const changes = r.combined.engine === "compressed"
    ? r.rubric.changes
    : r.rubric.changes.concat([
        `Neural gap rewrite merged (${r.combined.addressed.length} parts addressed)`,
      ]);
  $("cmp-changes").innerHTML = changes.map((c) => `<li>${esc(c)}</li>`).join("");

  state.voted = null;
  markChoice(null, false);
  $("vote-status").textContent = "";
}

function markChoice(choice, locked) {
  for (const btn of document.querySelectorAll(".prefer")) {
    const chosen = btn.dataset.choice === choice;
    btn.closest(".compare-card").classList.toggle("is-chosen", chosen);
    btn.setAttribute("aria-pressed", String(chosen));
    btn.textContent = chosen ? "Your pick ✓" : "I'd use this";
    btn.disabled = locked;
  }
}

function updateTally(data) {
  const t = data && data.tally;
  if (!t) return;
  $("vote-tally").textContent = t.total
    ? `Saved on this machine: merged ${t.merged} · original ${t.original} ` +
      `(${t.merged_share}% merged, ${t.total} vote${t.total === 1 ? "" : "s"})`
    : "No votes yet — pick one and it is saved locally.";
}

async function loadTally() {
  try {
    const res = await fetch("/api/preferences");
    if (res.ok) updateTally(await res.json());
  } catch { /* the comparison still works without a saved tally */ }
}

async function vote(choice) {
  if (!state.result || state.voted) return;
  const r = state.result;
  try {
    const data = await api("/api/preference", {
      prompt: r.prompt,
      merged: r.combined.text,
      choice,
      score_before: r.score.before,
      score_after: r.score.after,
      engine: r.combined.engine,
    });
    state.voted = choice;
    markChoice(choice, true);
    updateTally(data);
    $("vote-status").textContent = `Saved — you picked the ${choice} prompt. Run a new analysis to vote again.`;
  } catch (err) {
    $("vote-status").textContent = `Could not save the vote: ${err.message}`;
  }
}

/* ------------------------------------------------------------------ tabs */
function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      for (const other of tabs) {
        const active = other === tab;
        other.classList.toggle("is-active", active);
        other.setAttribute("aria-selected", String(active));
      }
      for (const panel of document.querySelectorAll(".tabpanel")) {
        panel.hidden = panel.id !== `panel-${tab.dataset.tab}`;
      }
    });
  }
}

/* ---------------------------------------------------------------- signals */
function paintHeat(tokens, weights) {
  const dark = document.documentElement.dataset.theme === "dark";
  const rgb = dark ? "226, 102, 74" : "196, 68, 43";
  const max = Math.max(...weights, 0.0001);
  const heat = $("heat");
  heat.innerHTML = "";
  tokens.forEach((tok, i) => {
    const w = weights[i] ?? 0;
    const alpha = 0.06 + (dark ? 0.72 : 0.82) * (w / max);
    const span = document.createElement("span");
    span.textContent = tok;
    span.style.background = `rgba(${rgb}, ${alpha.toFixed(3)})`;
    span.style.color = dark ? "#f3eee1" : (alpha > 0.55 ? "#fdf8ec" : "#6e2015");
    span.title = `${w.toFixed(3)} attention`;
    heat.appendChild(span);
  });
}

/* theme + motion live in theme.js; app.js reacts to their events */
document.addEventListener("themechange", () => {
  if (state.result) paintHeat(state.result.neural.tokens, state.result.neural.attention);
});

document.addEventListener("motionchange", (event) => {
  // stop a count-up mid-flight so the visible score is the real one
  if (!event.detail.on && state.result) $("score-after").textContent = state.result.score.after;
});

function playReveal(el, className) {
  if (!el || !motionOn()) return;
  el.classList.remove(className);
  void el.offsetWidth;   // force a reflow so the animation restarts
  el.classList.add(className);
}

function countUp(el, from, to, duration) {
  if (!el) return;
  if (!motionOn() || from === to) { el.textContent = to; return; }
  const started = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = to;
  };
  requestAnimationFrame(step);
}

/* ------------------------------------------------------------------ copy */
async function copyMerged() {
  if (!state.result) return;
  const text = state.result.combined.text;
  const btn = $("copy");
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied";
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    btn.textContent = "Copied";
  }
  setTimeout(() => { btn.textContent = "Copy"; }, 1600);
}

/* ----------------------------------------------------------------- batch */
async function runBatch() {
  const prompts = $("batch-input").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!prompts.length) { $("batch-status").textContent = "add one prompt per line first"; return; }
  const btn = $("batch-run");
  btn.disabled = true;
  $("batch-status").textContent = `scoring ${prompts.length} prompts…`;
  try {
    const data = await api("/api/batch", { prompts, mode: $("mode").value, level: $("level").value });
    state.rows = data.rows;
    renderBatch(data);
    playReveal(document.querySelector(".batch"), "is-fresh");
    $("batch-status").textContent = `${data.rows.length} rows`;
    $("batch-download").disabled = false;
  } catch (err) {
    $("batch-status").textContent = `failed: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
}

function renderBatch(data) {
  const s = data.summary;
  const summary = $("batch-summary");
  summary.hidden = false;
  summary.innerHTML = [
    ["avg before", s.avg_before],
    ["avg after", s.avg_after],
    ["avg gain", `+${s.avg_gain}`],
    ["intent preserved", `${s.intent_preserved}%`],
  ].map(([label, value]) => `<div class="metric"><b>${value}</b><span>${label}</span></div>`).join("");

  const head = `<tr>
    <th>#</th><th>prompt</th><th>task</th><th>intent</th>
    <th>before</th><th>after</th><th>gain</th><th>engine</th>
    <th>gaps left</th><th>intent kept</th></tr>`;
  const rows = data.rows.map((row, i) => `<tr>
    <td class="num">${i + 1}</td>
    <td class="prompt">${esc(row.prompt)}</td>
    <td>${esc(row.task)}</td>
    <td>${esc(row.intent)}</td>
    <td class="num">${row.before}</td>
    <td class="num">${row.after}</td>
    <td class="num ${row.gain > 0 ? "gain-up" : "gain-flat"}">${row.gain > 0 ? "+" : ""}${row.gain}</td>
    <td>${esc(row.engine)}</td>
    <td class="num">${row.gaps_left}</td>
    <td class="${row.intent_preserved ? "yes" : "no"}">${row.intent_preserved ? "yes" : "no"}</td>
  </tr>`).join("");
  $("batch-table").innerHTML = `<table><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

function downloadCsv() {
  if (!state.rows) return;
  const cols = ["prompt", "task", "intent", "before", "after", "gain",
    "engine", "gaps_left", "intent_preserved", "token_ratio"];
  const quote = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const csv = [cols.join(",")]
    .concat(state.rows.map((row) => cols.map((c) => quote(row[c])).join(",")))
    .join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "prompt_studio_benchmark.csv";
  a.click();
  URL.revokeObjectURL(url);
}

/* ---------------------------------------------------------------- ollama */
async function runOllama() {
  if (!state.result) { $("ollama-status").textContent = "analyze a prompt first"; return; }
  const btn = $("ollama-run");
  btn.disabled = true;
  $("ollama-status").textContent = "generating with your local model…";
  try {
    const data = await api("/api/ollama", {
      prompt: state.result.prompt,
      optimized: state.result.combined.text,
      model: $("ollama-model").value.trim() || "llama3.2",
      base_url: $("ollama-url").value.trim() || "http://localhost:11434",
    });
    $("ollama-out").innerHTML = `
      <div><h5>original response</h5><pre class="code">${esc(data.original)}</pre></div>
      <div><h5>merged prompt response</h5><pre class="code">${esc(data.optimized)}</pre></div>`;
    $("ollama-status").textContent = "done";
  } catch (err) {
    $("ollama-status").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

/* ------------------------------------------------------------------ init */
document.addEventListener("DOMContentLoaded", async () => {
  // a refresh always starts at the top; the demo analysis must not scroll.
  // scroll-behavior is smooth in CSS, so override it to jump without animating
  const nav = performance.getEntriesByType?.("navigation")?.[0];
  if (nav?.type === "reload") {
    const toTop = () => {
      const root = document.documentElement;
      const prev = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      window.scrollTo(0, 0);
      root.style.scrollBehavior = prev;
    };
    toTop();
    window.addEventListener("load", toTop, { once: true });
  }

  setupTabs();
  $("run").addEventListener("click", () => analyze());
  $("copy").addEventListener("click", copyMerged);
  $("batch-run").addEventListener("click", runBatch);
  $("batch-download").addEventListener("click", downloadCsv);
  $("ollama-run").addEventListener("click", runOllama);
  for (const btn of document.querySelectorAll(".prefer")) {
    btn.addEventListener("click", () => vote(btn.dataset.choice));
  }
  $("prompt").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); analyze(); }
  });

  await loadExamples();
  await loadTally();
  if (state.examples.length) {
    $("prompt").value = state.examples[0].text;
    $("batch-input").value = state.examples.slice(0, 4).map((e) => e.text).join("\n");
    analyze({ scroll: false });
  }
});
