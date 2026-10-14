// Wiring: the URL picks an exercise, the worker runs it, the player replays it. All the DOM lives here.
import { Player } from "./player.js";
import { describeRun, drawStrips, fitCanvas, readoutText } from "./view.js";
import { markPassed } from "./progress.js";

const FRAME_INTERVAL_MS = 100; // camera-frame requests are throttled to one per interval
// What each exercise kind does and shows. The kind comes from exercises.json; no exercise id appears in this file.
// `shows` lists the page groups that kind uses; every other group is hidden (hidden, never removed).
const KINDS = {
  drive: { message: "run", shows: ["road-label", "drive-view", "player-view", "dashboard-view"] },
  stats: { message: "stats", shows: [] },
  image: { message: "image", shows: ["drive-view", "mask"] },
};
const GROUPS = [...new Set(Object.values(KINDS).flatMap((kind) => kind.shows))];
const RUN_SEED = 0; // the seed run() uses, so the frame shown is the frame the car saw
const EDITOR_MODULES = ["view", "state", "lang-python", "commands", "language", "theme-one-dark"].map((name) => `@codemirror/${name}`);

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

let data; // { records, roads, palette }, loaded once
let app; // the page's long-lived pieces, built once: { record, roads, palette, player, strips, editor, notice, autoRun }
let worker;
let workerReady = false;
let busy = null; // the worker message in flight: "run" | "stats" | "image" | "check" | "dashboard"
let loadNote = ""; // "Stopped." while the runtime restarts after Stop
let hasDashboard = false; // true only after a completed run: the worker keeps that run for the Dashboard
let driveRoad = ""; // the road of the drive the player holds
let dashboardUrl = "";
const frames = { wanted: null, inFlight: false, timer: 0, sentAt: 0 };

// ---- status and controls ----------------------------------------------------------------------

function setStatus(state, text) {
  $("status").dataset.state = state;
  $("status").textContent = text;
  $("run-hint").hidden = state !== "ready" || Boolean(app?.player.length); // the empty boxes are explained only before there is anything to show
}

function controls() {
  const idle = workerReady && !busy;
  $("run").disabled = $("check").disabled = !idle;
  $("stop").disabled = !busy;
  $("road").disabled = Boolean(busy);
  $("dashboard").disabled = !(idle && hasDashboard);
}

/** The worker cannot be used any more (it failed to start, or sent something the page cannot use): stop it and say why, with Retry. */
function failWorker(message) {
  worker?.terminate();
  workerReady = false;
  busy = null;
  showLoadFailure(message);
}

function showLoadFailure(message, { retry = true } = {}) {
  setStatus("error", message);
  $("retry").hidden = !retry;
  if (app) {
    $("colab-fallback").href = app.record.colab_url;
    $("colab-fallback").hidden = false;
  }
  controls();
}

function clearFailure() {
  $("retry").hidden = true;
  $("colab-fallback").hidden = true;
}

function clearOutputs() {
  $("stdout").textContent = "";
  $("verdict").hidden = true;
  $("stats").hidden = true;
  $("traceback").hidden = true;
  $("dashboard-img").hidden = true; // it drew the previous run
  $("dashboard-error").hidden = true;
}

// ---- data and page setup -----------------------------------------------------------------------

async function fetchJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (cause) {
    throw new Error(`Could not load ${url}: ${cause.message}`);
  }
}

async function loadData() {
  const [records, roads, palette] = await Promise.all(["exercises.json", "roads.json", "palette.json"].map(fetchJson));
  return { records, roads, palette };
}

/**
 * The exercise and preset the URL asks for. An unknown exercise falls back to the first record and an unknown
 * preset to the starter, each with a notice. `autoRun` is true only when the URL named both and both exist.
 */
function choose({ records }) {
  const id = params.get("exercise");
  const known = records.find((r) => r.id === id);
  const record = known ?? records[0];
  const notices = [];
  if (!known && id) notices.push(`Unknown exercise ${id}; showing ${record.label}.`);
  const asked = params.get("preset");
  const preset = asked !== null && Object.hasOwn(record.presets, asked) ? asked : "starter";
  if (asked !== null && preset !== asked) notices.push(`Unknown preset ${asked}; showing the starter.`);
  return { record, code: record.presets[preset], notice: notices.join(" "), autoRun: Boolean(known) && asked === preset };
}

/** CodeMirror 6 from the import map in index.html; any failure leaves a plain textarea. Returns { get, set } for the code. */
async function mountEditor(parent, doc) {
  try {
    const [view, state, python, commands, language, dark] = await Promise.all(EDITOR_MODULES.map((name) => import(name)));
    const editor = new view.EditorView({
      parent,
      state: state.EditorState.create({
        doc,
        extensions: [
          view.lineNumbers(), view.drawSelection(), view.EditorView.lineWrapping, language.indentOnInput(), language.bracketMatching(), commands.history(),
          view.keymap.of([...commands.defaultKeymap, ...commands.historyKeymap, commands.indentWithTab]),
          python.python(), dark.oneDark,
        ],
      }),
    });
    return {
      get: () => editor.state.doc.toString(),
      set: (text) => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: text } }),
    };
  } catch (error) {
    console.warn("CodeMirror did not load; using a plain textarea.", error);
    const textarea = Object.assign(document.createElement("textarea"), { id: "editor-fallback", value: doc, spellcheck: false });
    const note = Object.assign(document.createElement("p"), {
      id: "editor-note",
      textContent: "The code editor did not load; this is a plain text box. Everything else works.",
    });
    parent.replaceChildren(note, textarea);
    return { get: () => textarea.value, set: (text) => { textarea.value = text; } };
  }
}

/** The chrome colours come from palette.json at runtime, so no hex is typed twice: each key becomes `--<key>`. */
function applyChrome(chrome) {
  if (!chrome) throw new Error('Could not use palette.json: it has no "chrome" colours.');
  for (const [name, value] of Object.entries(chrome)) document.documentElement.style.setProperty(`--${name}`, value);
}

async function setUp({ records, roads, palette }) {
  const { record, code, notice, autoRun } = choose({ records });
  document.title = `${record.label} - Playground`;
  $("exercise-label").textContent = record.label;
  $("exercise-task").textContent = record.task ?? "";
  $("exercise-task").hidden = !record.task;
  $("unit-link").href = `../unit/${record.unit_key}/`; // relative: the lecture sits beside the playground wherever the course is served
  $("open-colab").href = record.colab_url;
  const kind = KINDS[record.kind];
  if (!kind) throw new Error(`Exercise ${record.id} has kind "${record.kind}", which this page does not know how to run.`);
  // Only a kind with a player is played on a road; the others name none ("" for an image) and borrow the first to size the canvas.
  const roadName = record.road || (kind.shows.includes("player-view") ? "" : Object.keys(roads)[0]);
  if (!Object.hasOwn(roads, roadName)) {
    throw new Error(record.road
      ? `Exercise ${record.id} names road "${record.road}", which roads.json does not have.`
      : `Exercise ${record.id} is a ${record.kind} exercise, which is played on a road, but names none.`);
  }
  $("road").replaceChildren(...Object.keys(roads).map((name) => new Option(name, name, false, name === roadName)));

  // Fitting a canvas that has no size (a hidden panel) gives NaN, so both canvases are fitted before the groups are hidden.
  const player = new Player($("canvas"), roads[roadName], palette);
  driveRoad = roadName;
  const strips = { ctx: fitCanvas($("strips")).ctx };
  GROUPS.forEach((id) => { $(id).hidden = !kind.shows.includes(id); });
  const editor = await mountEditor($("editor"), code);
  window.__setCode = editor.set; // test hook: the browser tests paste reference solutions with it
  window.__getCode = editor.get; // test hook: the browser tests read the editor back with it
  player.onTick(({ k, row, playing }) => {
    $("play").textContent = playing ? "Pause" : "Play";
    $("scrub").max = Math.max(0, player.length - 1);
    $("scrub").value = k;
    Object.entries(readoutText(row)).forEach(([name, text]) => { $(`r-${name}`).textContent = text; });
    drawStrips(strips.ctx, player.rows, k, palette);
    if (row) wantFrame(k, row);
  });
  new ResizeObserver(() => {
    if (!$("canvas").clientWidth) return; // the player panel is hidden (an image or stats exercise): nothing to fit
    player.fit();
    strips.ctx = fitCanvas($("strips")).ctx;
  }).observe(document.body);
  return { record, kind, roads, palette, player, strips, editor, notice, autoRun };
}

// ---- the worker --------------------------------------------------------------------------------

const missingFeatures = () =>
  [["Web Workers", "Worker"], ["WebAssembly", "WebAssembly"]].filter(([, name]) => typeof globalThis[name] === "undefined").map(([label]) => label);

function startWorker(note = "") {
  worker?.terminate();
  loadNote = note;
  workerReady = false;
  busy = null;
  hasDashboard = false; // a fresh worker has no run to draw
  frames.wanted = null;
  frames.inFlight = false;
  clearTimeout(frames.timer);
  frames.timer = 0;
  clearFailure();
  controls();
  const missing = missingFeatures();
  if (missing.length) {
    return showLoadFailure(`This browser has no ${missing.join(" or ")}, which the playground needs to run Python.`, { retry: false });
  }
  setStatus("loading", note ? `${note} Restarting Python...` : "Starting Python in your browser...");
  worker = new Worker("./worker.js", { type: "module" });
  worker.onmessage = ({ data: message }) => {
    try {
      (FROM_WORKER[message.type] ?? unknownMessage)(message);
    } catch (error) {
      failWorker(error.message); // a bug in the page or the worker is shown, never left as "Running..."
    }
  };
  worker.onerror = (event) => failWorker(`Could not start the worker (${event.filename || "worker.js"}): ${event.message || "see the browser console"}`);
  worker.postMessage({ type: "init" });
}

function unknownMessage(message) {
  throw new Error(`The worker sent a "${message.type}" message, which this page does not know.`); // routed to failWorker by onmessage
}

function start(message) {
  clearOutputs();
  busy = message;
  hasDashboard = false;
  setStatus("running", message === "check" ? "Checking your code..." : "Running your code...");
  worker.postMessage({ type: message, code: app.editor.get(), exercise: app.record, road: $("road").value });
  controls();
}

function stop() {
  app.player.finish(); // the rows so far stay on screen
  startWorker("Stopped.");
}

function onRow(row) {
  if (row.t === 0) { // a new drive begins at t = 0; a check can drive several roads in turn
    const road = app.roads[row.road];
    if (!road) throw new Error(`A streamed step names the road "${row.road}", which roads.json does not have.`);
    driveRoad = row.road;
    app.player.reset(road);
    app.player.play();
  }
  app.player.append(row);
}

function finishWork() {
  busy = null;
  app.player.finish();
  controls();
}

function wantFrame(k, row) {
  frames.wanted = { k, row };
  pumpFrames();
}

function pumpFrames() {
  if (!frames.wanted || frames.inFlight || frames.timer || !workerReady || busy) return; // the worker is single-threaded: frames wait for the run
  const wait = Math.max(0, FRAME_INTERVAL_MS - (performance.now() - frames.sentAt));
  frames.timer = setTimeout(() => {
    frames.timer = 0;
    sendFrame();
  }, wait);
}

function sendFrame() {
  if (!frames.wanted || !workerReady || busy) return;
  const { k, row } = frames.wanted;
  frames.wanted = null;
  frames.inFlight = true;
  frames.sentAt = performance.now();
  worker.postMessage({ type: "frame", road: driveRoad, k, x: row.x, y: row.y, yaw: row.yaw, speed: row.speed, seed: RUN_SEED });
}

function paintRgb(canvas, width, height, rgb) {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const image = new ImageData(width, height);
  const source = new Uint8Array(rgb);
  for (let i = 0, j = 0; i < source.length; i += 3, j += 4) {
    image.data.set([source[i], source[i + 1], source[i + 2], 255], j);
  }
  canvas.getContext("2d").putImageData(image, 0, 0);
}

function showStats(values) {
  const items = Object.entries(values).flatMap(([label, value]) => [
    Object.assign(document.createElement("dt"), { textContent: label }),
    Object.assign(document.createElement("dd"), { textContent: Number(value.toPrecision(4)).toString() }),
  ]);
  $("stats").replaceChildren(...items);
  $("stats").hidden = false;
}

const FROM_WORKER = {
  // After `ready` the only download is the Dashboard's matplotlib, loaded on its first press.
  progress: ({ detail }) =>
    workerReady
      ? setStatus("running", `Loading ${detail} for the dashboard...`)
      : setStatus("loading", [loadNote, `Loading ${detail}...`].filter(Boolean).join(" ")),

  ready: ({ versions, steps_per_s_probe: stepsPerS }) => {
    workerReady = true;
    loadNote = "";
    $("engine").textContent =
      `Pyodide ${versions.pyodide}, Python ${versions.python}, NumPy ${versions.numpy}, Pillow ${versions.pillow}; ` +
      `the simulator runs about ${Math.round(stepsPerS)} steps per second on this device.`;
    setStatus("ready", [app.notice, "Ready."].filter(Boolean).join(" "));
    controls();
    pumpFrames();
    if (app.autoRun) {
      app.autoRun = false; // once: Stop or a later Retry must not start another run
      start(app.kind.message);
    }
  },

  row: ({ row }) => onRow(row),

  stdout: ({ text }) => {
    $("stdout").textContent += `${text}\n`;
  },

  done: ({ metrics }) => {
    const seconds = app.player.rows.at(-1)?.t ?? 0;
    hasDashboard = true;
    finishWork();
    setStatus("done", describeRun(metrics, driveRoad, seconds));
    pumpFrames();
  },

  stats: ({ values }) => {
    finishWork();
    showStats(values);
    setStatus("done", `Ran ${app.record.name} on a recorded drive; ${Object.keys(values).length} values below.`);
  },

  image: ({ width, height, frame, overlay }) => {
    finishWork();
    paintRgb($("frame"), width, height, frame);
    paintRgb($("mask"), width, height, overlay);
    setStatus("done", `Ran ${app.record.name} on the lab's noisy camera frame. Left: the frame. Right: the pixels your masks marked.`);
  },

  verdict: ({ passed, details }) => {
    finishWork();
    $("verdict").dataset.passed = String(passed);
    $("verdict").querySelector("ul").replaceChildren(...details.map((text) => Object.assign(document.createElement("li"), { textContent: text })));
    $("verdict").hidden = false;
    const saved = passed && markPassed(app.record.unit_key, app.record.id);
    setStatus("done", !passed ? "Check did not pass. Details below." : saved ? "Check passed. Your progress is saved." : "Check passed, but this browser blocked saving your progress.");
    pumpFrames();
  },

  frame: (message) => {
    frames.inFlight = false;
    paintRgb($("frame"), message.width, message.height, message.rgb);
    pumpFrames();
  },

  dashboard: ({ png }) => {
    URL.revokeObjectURL(dashboardUrl);
    dashboardUrl = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    $("dashboard-img").src = dashboardUrl;
    $("dashboard-img").hidden = false;
    busy = null;
    setStatus("done", "Dashboard drawn below.");
    controls();
  },

  error: ({ message, line, traceback, request }) => {
    if (request === "init") return failWorker(message);
    frames.inFlight = false;
    if (request === busy) finishWork(); // a late frame error must not end the run that is still streaming
    const said = line ? `${message} (your code, line ${line})` : message;
    setStatus("error", said);
    $("stdout").textContent += `${said}\n`; // the same line where the learner's own prints appear
    if (request === "dashboard") { // not a failed run: the run behind it stands, so its traceback and Dashboard stay, to retry
      $("dashboard-error").textContent = said;
      $("dashboard-error").hidden = false;
      return;
    }
    $("traceback").querySelector("pre").textContent = traceback;
    $("traceback").hidden = !traceback;
  },
};

// ---- user input ---------------------------------------------------------------------------------

async function boot() {
  try {
    data ??= await loadData();
    applyChrome(data.palette.chrome);
    app ??= await setUp(data);
    startWorker();
  } catch (error) {
    showLoadFailure(error.message);
  }
}

$("run").addEventListener("click", () => start(app.kind.message));
$("check").addEventListener("click", () => start("check"));
$("stop").addEventListener("click", stop);
$("retry").addEventListener("click", boot);
$("play").addEventListener("click", () => app.player.toggle());
$("to-end").addEventListener("click", () => app.player.seek(Math.max(0, app.player.length - 1)));
$("scrub").addEventListener("input", (event) => app.player.seek(event.target.valueAsNumber));
$("speed").addEventListener("change", (event) => app.player.setSpeed(Number(event.target.value)));
$("road").addEventListener("change", (event) => {
  driveRoad = event.target.value;
  app.player.reset(app.roads[driveRoad]); // a different road starts a different drive; the old rows do not belong on it
  app.player.finish();
});
$("dashboard").addEventListener("click", () => {
  busy = "dashboard";
  $("dashboard-error").hidden = true;
  setStatus("running", "Drawing the dashboard (the first time this downloads matplotlib)...");
  worker.postMessage({ type: "dashboard" });
  controls();
});

boot();
