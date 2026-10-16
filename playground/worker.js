// Module worker: the protocol between the page and Pyodide, nothing else. Python decides, JS carries.
// Pyodide 314.0.7 from jsDelivr, verified 2026-10-11:
//   $ curl -sI https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs | head -1
//   HTTP/2 200
// It must stay a module worker (`new Worker("./worker.js", {type: "module"})`): the classic worker
// failed in the spike because importScripts threw. The import is dynamic so that a failed load
// reaches the page as an `error` message that names the URL, instead of a bare worker error event.
const PYODIDE_VERSION = "314.0.7";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PACKAGES = ["numpy", "pillow"];
const DASHBOARD_PACKAGES = ["matplotlib"]; // loaded by the first Dashboard request, not at start-up
const QUIET = { messageCallback: () => {} }; // package progress goes out as `progress` messages; it is not the learner's output
const TO_JS = { dict_converter: Object.fromEntries };

let pyodide;
let shim;
let frameSize; // [width, height] of the camera image, read from the camera that renders it
let ready; // the `ready` message once init finished: a repeated init answers with it instead of loading again
let queue = Promise.resolve(); // messages are handled one at a time: an async handler (init, dashboard) must not interleave with the next

const post = (type, fields = {}, transfer = []) => postMessage({ type, ...fields }, transfer);

/** Await `work`; a failure is re-thrown naming the file it was fetching. */
async function loading(url, work) {
  try {
    return await work;
  } catch (cause) {
    throw new Error(`Could not load ${url}: ${cause.message}`);
  }
}

/** The lockfile entry for package `key`; a key the lockfile does not have is named, not left to a TypeError on `undefined`. */
function lockEntry(key) {
  const entry = pyodide.lockfile.packages[key];
  if (!entry) throw new Error(`Pyodide's package list has no "${key}".`);
  return entry;
}

/** The lockfile keys of `keys` and of everything they depend on whose wheel is not loaded. `loadedPackages` is keyed by the
 *  lockfile's display name ("Pillow"), the lockfile by the package key ("pillow"). */
function notLoaded(keys) {
  const seen = new Set();
  const pending = [...keys];
  while (pending.length) {
    const key = pending.pop();
    if (seen.has(key)) continue;
    seen.add(key);
    pending.push(...lockEntry(key).depends);
  }
  return [...seen].filter((key) => !pyodide.loadedPackages[lockEntry(key).name]);
}

/** `loadPackage` reports a failed download through its error callback and still resolves, and it skips the dependencies of
 *  a package that is already loaded. So load exactly what is missing from the whole dependency closure, then read what
 *  loaded: anything still missing is a failure that names its wheels, and the next call retries only those. */
async function loadPackages(keys) {
  await pyodide.loadPackage(notLoaded(keys), QUIET);
  const missing = notLoaded(keys);
  if (missing.length) throw new Error(`the download failed for ${missing.map((key) => pyodide.lockfile.packages[key].file_name).join(", ")}`);
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.arrayBuffer();
}

/** Python value to plain JS; the proxy is released. */
function plain(proxy) {
  try {
    return proxy.toJs(TO_JS);
  } finally {
    proxy.destroy();
  }
}

/** A Python bytes object as a standalone ArrayBuffer (a copy, so it can be transferred). */
function bytesOf(proxy) {
  try {
    return proxy.toJs().buffer;
  } finally {
    proxy.destroy();
  }
}

/** `on_row` for Python: each step goes to the page the moment it exists, tagged with `road`. A step the shim already
 *  tagged keeps its own tag. A run passes the road it was asked for. A check passes the exercise's own road: only a
 *  gate drive streams, and every drive exercise's record road is that gate road (GATE_SCENARIO, grade/_drive.py:12). */
const onRow = (road) => (rowProxy) => post("row", { row: { road, ...rowProxy.toJs(TO_JS) } }); // Pyodide owns a call argument's proxy, so not `plain`

async function init() {
  post("progress", { stage: "pyodide", detail: `Pyodide ${PYODIDE_VERSION}` });
  const { loadPyodide } = await loading(`${PYODIDE_BASE}pyodide.mjs`, import(`${PYODIDE_BASE}pyodide.mjs`));
  pyodide = await loading(PYODIDE_BASE, loadPyodide({ indexURL: PYODIDE_BASE }));
  pyodide.setStdout({ batched: (text) => post("stdout", { text }) });
  post("progress", { stage: "packages", detail: PACKAGES.join(", ") });
  await loading(`${PYODIDE_BASE} (${PACKAGES.join(", ")})`, loadPackages(PACKAGES));
  post("progress", { stage: "bundle", detail: "bundle.zip" });
  pyodide.unpackArchive(await loading("bundle.zip", fetchBytes("bundle.zip")), "zip");
  pyodide.FS.writeFile("shim.py", new Uint8Array(await loading("shim.py", fetchBytes("shim.py"))));
  shim = pyodide.pyimport("shim");
  frameSize = plain(pyodide.runPython("from zero2fsd.sim.camera import Camera\ncamera = Camera()\n[camera.width, camera.height]"));
  post("progress", { stage: "probe", detail: "timing the simulator" });
  const stepsPerS = shim.probe_steps_per_s(100);
  const [python, numpy, pillow] = plain(pyodide.runPython("import sys, numpy, PIL\n[sys.version.split()[0], numpy.__version__, PIL.__version__]"));
  ready = { versions: { pyodide: pyodide.version, python, numpy, pillow }, steps_per_s_probe: stepsPerS };
  post("ready", ready);
}

/** Build the learner's object, hand it to `use`, and release it. */
async function withLearner({ code, exercise }, use) {
  const learner = shim.learner_object(code, exercise.name);
  try {
    return await use(learner);
  } finally {
    learner?.destroy?.(); // a function or Car is a proxy; None arrives as undefined, numbers and strings have no destroy
  }
}

const HANDLERS = {
  init,

  run: (m) => withLearner(m, (learner) => {
    const metrics = plain(shim.preview_run(learner, m.exercise.id, m.road, onRow(m.road)));
    post("done", { metrics });
  }),

  check: (m) => withLearner(m, (learner) => {
    post("verdict", plain(shim.check_code(learner, m.exercise.id, onRow(m.exercise.road))));
  }),

  stats: (m) => withLearner(m, (learner) => {
    post("stats", { values: plain(shim.preview_stats(learner, m.exercise.id)) });
  }),

  frame: ({ road, k, x, y, yaw, speed, seed }) => {
    const rgb = bytesOf(shim.frame_rgb(road, k, x, y, yaw, speed, seed));
    const [width, height] = frameSize;
    post("frame", { k, width, height, rgb }, [rgb]);
  },

  image: (m) => withLearner(m, (learner) => {
    const { width, height, frame, overlay } = plain(shim.preview_image(learner, m.exercise.id));
    post("image", { width, height, frame: frame.buffer, overlay: overlay.buffer }, [frame.buffer, overlay.buffer]);
  }),

  dashboard: async () => {
    if (notLoaded(DASHBOARD_PACKAGES).length) {
      post("progress", { stage: "dashboard-packages", detail: DASHBOARD_PACKAGES.join(", ") }); // its own stage: a late download, not start-up
      await loading(`${PYODIDE_BASE} (${DASHBOARD_PACKAGES.join(", ")})`, loadPackages(DASHBOARD_PACKAGES));
    }
    const png = bytesOf(shim.dashboard_png());
    post("dashboard", { png }, [png]);
  },
};

/** The page's one error shape. A Python exception is formatted by the shim, which finds the learner's line. */
function errorFields(error) {
  if (!(shim && error instanceof pyodide.ffi.PythonError)) return { message: error.message, line: null, traceback: error.stack ?? "" };
  const sys = pyodide.pyimport("sys");
  const exception = sys.last_exc;
  try {
    return plain(shim.format_error(exception));
  } finally {
    exception.destroy();
    sys.destroy();
  }
}

async function handle(data) {
  try {
    const handler = Object.hasOwn(HANDLERS, data.type) ? HANDLERS[data.type] : null;
    if (!handler) throw new Error(`The worker has no handler for message type "${data.type}".`);
    if (data.type === "init" && ready) return post("ready", ready); // already loaded: the same answer, no second download
    if (!ready && data.type !== "init") throw new Error('Send "init" and wait for "ready" first.');
    await handler(data);
  } catch (error) {
    post("error", { ...errorFields(error), request: data.type });
  }
}

onmessage = ({ data }) => {
  queue = queue.then(() => handle(data)).catch(console.error); // handle reports its own failures; this keeps one that escapes from blocking the rest
};
