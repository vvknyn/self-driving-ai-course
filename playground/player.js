// The playback clock. The worker produces rows at full speed; the player replays them by row.t, so
// scrubbing, speed and pause never touch the worker. It owns the rows and calls view.js to draw them.
import { draw, fitCanvas, fitTransform } from "./view.js";

const MAX_FRAME_S = 0.25; // a backgrounded tab must not fast-forward the lap when it wakes

/** `x` when it is a finite number above zero (zero too with `zeroOk`); otherwise a RangeError that says what `name` must be. */
function finitePositive(x, name, { zeroOk = false } = {}) {
  if (!Number.isFinite(x) || x < 0 || (x === 0 && !zeroOk)) {
    throw new RangeError(`${name} must be a finite number ${zeroOk ? "of zero or more" : "above zero"}, not ${x}.`);
  }
  return x;
}

/** Index of the newest row shown at `clock` seconds, searching from `k` (rows are in time order). */
export function rowIndexAt(rows, k, clock) {
  let i = Math.max(0, Math.min(k, rows.length - 1));
  while (i + 1 < rows.length && rows[i + 1].t <= clock) i++;
  while (i > 0 && rows[i].t > clock) i--;
  return i;
}

export class Player {
  #canvas;
  #palette;
  #road;
  #transform;
  #ctx;
  #rows = [];
  #k = 0;
  #clock = 0;
  #speed = 1;
  #playing = false;
  #finished = true; // no more rows are coming
  #dirty = true;
  #frame = 0;
  #last = null;
  #ticks = [];

  constructor(canvas, road, palette) {
    this.#canvas = canvas;
    this.#palette = palette;
    this.reset(road);
    this.#finished = true;
  }

  get k() { return this.#k; }
  get length() { return this.#rows.length; }
  /** The rows so far; read-only, do not mutate. */
  get rows() { return this.#rows; }
  get playing() { return this.#playing; }

  onTick(callback) { this.#ticks.push(callback); }

  /** Start a new drive on `road`: rows cleared, paused, more rows expected until finish(). */
  reset(road) {
    this.#road = road;
    this.#rows = [];
    this.#k = 0;
    this.#clock = 0;
    this.#setPlaying(false);
    this.#finished = false;
    this.fit();
  }

  /** Replace all rows with a finished drive. The player keeps its own copy: a later append() never touches `rows`. */
  setRows(rows) {
    this.#rows = Array.from(rows);
    this.#finished = true;
    this.seek(0);
  }

  /** Live: add the newest step; a playing player keeps following it. */
  append(row) {
    this.#rows.push(row);
    this.#dirty = true;
    this.#kick();
  }

  /** The producer is done: playback stops at the last row instead of waiting for more. */
  finish() {
    this.#finished = true;
    this.#kick();
  }

  /** Re-measure the canvas (window or layout resize) and redraw. */
  fit() {
    const { ctx, width, height } = fitCanvas(this.#canvas);
    this.#ctx = ctx;
    this.#transform = fitTransform(this.#road, width, height);
    this.#dirty = true;
    this.#kick();
  }

  play() {
    if (this.#finished && this.#k >= this.#rows.length - 1) this.seek(0); // finished lap: play it again
    this.#setPlaying(true);
    this.#last = null;
    this.#kick();
  }

  pause() {
    this.#setPlaying(false);
    this.#kick();
  }

  toggle() { this.#playing ? this.pause() : this.play(); }

  seek(k) {
    this.#k = Math.max(0, Math.min(Math.trunc(finitePositive(k, "The step", { zeroOk: true })), this.#rows.length - 1));
    this.#clock = this.#rows[this.#k]?.t ?? 0;
    this.#dirty = true;
    this.#kick();
  }

  setSpeed(multiplier) { this.#speed = finitePositive(multiplier, "The speed"); }

  /** The one place `playing` changes: a change is a frame to draw, so every listener hears it in a tick. */
  #setPlaying(value) {
    if (this.#playing === value) return;
    this.#playing = value;
    this.#dirty = true;
  }

  #kick() {
    if (!this.#frame) this.#frame = requestAnimationFrame(this.#step);
  }

  #step = (now) => {
    this.#frame = 0;
    if (this.#playing) {
      const dt = this.#last === null ? 0 : Math.min((now - this.#last) / 1000, MAX_FRAME_S);
      this.#last = now;
      const newest = this.#rows.at(-1)?.t ?? 0;
      this.#clock = Math.min(this.#clock + dt * this.#speed, newest);
      const k = rowIndexAt(this.#rows, this.#k, this.#clock);
      this.#dirty ||= k !== this.#k;
      this.#k = k;
      if (this.#finished && k >= this.#rows.length - 1) this.#setPlaying(false); // reached the end
    }
    if (this.#playing) this.#kick(); // re-arm before drawing: a listener that throws must not end the loop
    if (this.#dirty) this.#render();
  };

  #render() {
    this.#dirty = false;
    draw(this.#ctx, this.#road, this.#rows, this.#k, this.#palette, this.#transform);
    const tick = { k: this.#k, row: this.#rows[this.#k], playing: this.#playing };
    for (const callback of this.#ticks) {
      try {
        callback(tick);
      } catch (error) {
        console.error("A player tick listener failed", error);
      }
    }
  }
}
