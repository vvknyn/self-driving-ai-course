// Pure presentation, shared by the playground and the lecture sliders: drawing functions take a 2D context
// plus data, text functions take numbers; none keeps state. Colours come only from the palette argument.

const CAR_LENGTH_M = 4.5;
const CAR_WIDTH_M = 1.8;
const MIN_CAR_PX = 12; // a 700 m loop on a phone would draw a true-size car under one pixel; the symbol never goes below this
const LINE_PX = 2;
const STRIPS = [
  { label: "e_off", unit: "m", truth: "lat", estimate: "est_offset", color: "e_off", floor: 0.5 },
  { label: "ψ", unit: "rad", truth: "head_err", estimate: "est_heading", color: "psi", floor: 0.1 },
  { label: "δ", unit: "rad", truth: "steer", color: "delta", floor: 0.1 },
];
const MIN_SCALE = 0.001; // pixels per metre: a canvas with no room (a hidden 0x0 one) still fits the road at a positive scale
const LABEL_MARGIN_PX = 76; // room right of the strip lines for the quantity names
const RANGE_LABEL_PX = 14; // the top of each strip holds its range label; traces stay below it
const STRIP_PAD_PX = 4; // space under a strip's traces
const LABEL_GAP_PX = 12;
const FONT = '10px "IBM Plex Mono", ui-monospace, monospace';

/** Size a canvas's backing store to its CSS box at the device pixel ratio; drawing then uses CSS pixels. */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

/** Scale and offsets that fit the whole tarmac into width x height pixels, centred, world y up. A road needs two distinct
 *  points to have a size; a canvas smaller than the padding still gets a positive scale, never a negative (mirrored) one. */
export function fitTransform(road, width, height, padding = 24) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of [...road.lines.tarmac_left, ...road.lines.tarmac_right]) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  if (!(maxX > minX || maxY > minY)) throw new Error("The road has fewer than two distinct points, so there is nothing to fit to the canvas.");
  const scale = Math.max(MIN_SCALE, Math.min((width - 2 * padding) / (maxX - minX), (height - 2 * padding) / (maxY - minY)));
  return { scale, ox: (width - scale * (minX + maxX)) / 2, oy: (height + scale * (minY + maxY)) / 2, width, height };
}

export const worldToCanvas = (t, x, y) => [t.ox + t.scale * x, t.oy - t.scale * y];

function tracePath(ctx, t, points, close) {
  points.forEach(([x, y], i) => {
    const [px, py] = worldToCanvas(t, x, y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  if (close) ctx.closePath();
}

function strokeLine(ctx, t, points, road, style) {
  ctx.beginPath();
  tracePath(ctx, t, points, road.closed);
  Object.assign(ctx, { strokeStyle: style.color, lineWidth: style.width, globalAlpha: style.alpha ?? 1 });
  ctx.setLineDash(style.dash ?? []);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}

function fillTarmac(ctx, t, road, colour) {
  const { tarmac_left: left, tarmac_right: right } = road.lines;
  ctx.beginPath();
  if (road.closed) {
    tracePath(ctx, t, left, true); // two loops: the fill between them is the road
    tracePath(ctx, t, right, true);
  } else {
    tracePath(ctx, t, [...left, ...[...right].reverse()], true);
  }
  ctx.fillStyle = colour;
  ctx.fill("evenodd");
}

/** The point of a polyline (closed back to its start when `closed`) nearest to (x, y): a vertex or the foot of a perpendicular. */
function nearestPoint(points, closed, x, y) {
  let best = points[0], bestD = Infinity;
  for (let i = 0; i < points.length - (closed ? 0 : 1); i++) {
    const [ax, ay] = points[i], [bx, by] = points[(i + 1) % points.length];
    const u = Math.max(0, Math.min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 || 1)));
    const p = [ax + u * (bx - ax), ay + u * (by - ay)], d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
    if (d < bestD) { bestD = d; best = p; }
  }
  return best;
}

function drawCar(ctx, t, row, fill, outline) {
  const [px, py] = worldToCanvas(t, row.x, row.y);
  const grow = Math.max(1, MIN_CAR_PX / (CAR_LENGTH_M * t.scale));
  const length = CAR_LENGTH_M * t.scale * grow;
  const width = CAR_WIDTH_M * t.scale * grow;
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-row.yaw); // world angles run counter-clockwise, canvas rotation clockwise
  ctx.beginPath();
  ctx.rect(-length / 2, -width / 2, length, width);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = LINE_PX;
  ctx.strokeStyle = outline;
  ctx.stroke();
  ctx.beginPath(); // the nose: which way the car points
  ctx.moveTo(0, 0);
  ctx.lineTo(length / 2, 0);
  ctx.stroke();
  ctx.restore();
}

/** Top-down picture of step `k`: road, e_off brace, the path driven so far, the car. `t` comes from fitTransform. */
export function draw(ctx, road, rows, k, palette, t) {
  const { quantity, scene, chrome } = palette;
  const { lines } = road;
  ctx.fillStyle = chrome.panel;
  ctx.fillRect(0, 0, t.width, t.height);
  fillTarmac(ctx, t, road, scene.road);
  strokeLine(ctx, t, lines.ego_center, road, { color: chrome.text, width: 1, alpha: 0.25 });
  strokeLine(ctx, t, lines.white, road, { color: scene.white, width: LINE_PX });
  const dashOn = road.dash_period * road.dash_fraction * t.scale;
  const dashOff = road.dash_period * (1 - road.dash_fraction) * t.scale;
  strokeLine(ctx, t, lines.yellow, road, { color: scene.yellow, width: LINE_PX, dash: [dashOn, dashOff] });
  const row = rows[k];
  if (!row) return;
  strokeLine(ctx, t, rows.slice(0, k + 1).map((r) => [r.x, r.y]), { closed: false }, { color: quantity.e_off, width: LINE_PX });
  const [bx, by] = nearestPoint(lines.ego_center, road.closed, row.x, row.y); // the e_off brace: car to the lane centre
  strokeLine(ctx, t, [[row.x, row.y], [bx, by]], { closed: false }, { color: quantity.e_off, width: LINE_PX });
  drawCar(ctx, t, row, chrome.background, row.off_road ? quantity.perception : chrome.text);
}

function plotSeries(ctx, rows, xOf, yOf, key, usable) {
  let pen = false, last = null;
  ctx.beginPath();
  for (const row of rows) {
    const v = row[key];
    if (!usable(row) || !Number.isFinite(v)) { pen = false; continue; } // a gap, never an invented value
    last = [xOf(row), yOf(v)];
    if (pen) ctx.lineTo(...last); else ctx.moveTo(...last);
    pen = true;
  }
  ctx.stroke();
  return last;
}

/** Three strip charts of the last `windowS` seconds ending at step `k`; each quantity's name sits at its line's end. */
export function drawStrips(ctx, rows, k, palette, windowS = 10) {
  const { clientWidth: width, clientHeight: height } = ctx.canvas;
  const { quantity, chrome } = palette;
  ctx.fillStyle = chrome.panel;
  ctx.fillRect(0, 0, width, height);
  if (!rows[k]) return;
  const start = rows[k].t - windowS;
  let first = k;
  while (first > 0 && rows[first - 1].t >= start) first--;
  const visible = rows.slice(first, k + 1);
  const band = height / STRIPS.length;
  const plotWidth = width - LABEL_MARGIN_PX;
  const xOf = (row) => ((row.t - start) / windowS) * plotWidth;
  ctx.font = FONT;
  ctx.lineWidth = LINE_PX;
  STRIPS.forEach((strip, i) => {
    const half = (band - RANGE_LABEL_PX - STRIP_PAD_PX) / 2; // traces use the band below the label
    const mid = band * i + RANGE_LABEL_PX + half;
    const estimateOk = (row) => strip.estimate !== undefined && row.est_valid;
    const spread = visible.flatMap((r) => [r[strip.truth], estimateOk(r) ? r[strip.estimate] : NaN]).filter(Number.isFinite);
    const range = Math.max(strip.floor, ...spread.map(Math.abs)) * 1.1;
    const yOf = (v) => mid - (v / range) * half;
    ctx.beginPath(); // yOf already gives canvas pixels, so the zero line skips worldToCanvas (which flips y)
    ctx.moveTo(0, yOf(0));
    ctx.lineTo(plotWidth, yOf(0));
    ctx.strokeStyle = chrome.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.lineWidth = LINE_PX;
    ctx.fillStyle = chrome.text;
    ctx.globalAlpha = 0.6;
    ctx.fillText(`±${range.toFixed(2)} ${strip.unit}`, 4, band * i + RANGE_LABEL_PX - 3);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = quantity[strip.color];
    const end = plotSeries(ctx, visible, xOf, yOf, strip.truth, () => true);
    const labels = [{ text: strip.label, end }];
    if (strip.estimate !== undefined) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = quantity.perception;
      labels.push({ text: `${strip.label} est.`, end: plotSeries(ctx, visible, xOf, yOf, strip.estimate, estimateOk) });
      ctx.setLineDash([]);
    }
    let lastY = -Infinity;
    for (const label of labels.filter((l) => l.end).sort((a, b) => a.end[1] - b.end[1])) {
      const y = Math.max(label.end[1], lastY + LABEL_GAP_PX); // two names that would overlap stack instead
      ctx.fillStyle = label.text.endsWith("est.") ? quantity.perception : quantity[strip.color];
      ctx.fillText(label.text, plotWidth + 6, y + 3);
      lastY = y;
    }
  });
}

/** `+0.12`: fixed digits with an explicit sign, so left and right offsets read at a glance. */
function signed(value, digits) {
  const rounded = Number(value.toFixed(digits));
  return (rounded < 0 ? "" : "+") + rounded.toFixed(digits);
}

/** The five live readouts for one step; a dash when there is no step yet. */
export function readoutText(row) {
  if (!row) return { t: "-", lat: "-", speed: "-", steer: "-", progress: "-" };
  return {
    t: `${row.t.toFixed(1)} s`,
    lat: `${signed(row.lat, 2)} m`,
    speed: `${row.speed.toFixed(1)} m/s`,
    steer: `${signed(row.steer, 3)} rad`,
    progress: `${row.progress_m.toFixed(0)} m`,
  };
}

/** One sentence for a finished drive. `metrics` is the shim's summary(); `seconds` is the last step's t. */
export function describeRun(metrics, road, seconds) {
  const lat = `mean |lat| ${metrics.mean_abs_lat.toFixed(2)} m`;
  if (metrics.completed) return `Completed ${road} in ${seconds.toFixed(1)} s, ${lat}`;
  const how = metrics.off_road ? "Left the road on" : "Ran out of time on";
  return `${how} ${road} after ${seconds.toFixed(1)} s and ${metrics.progress_m.toFixed(0)} m, ${lat}`;
}
