"""Figures for lecture 0.1.2 Images are arrays.  `python scripts/make_figures.py course/0.1.2-images-are-arrays`."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the "3d" projection)

from zero2fsd.car.provided import PAINT_DISTANCE, paint_masks
from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.camera import PALETTE, ROAD, SKY, WHITE, YELLOW
from zero2fsd.sim.scenarios import TARGET_SPEED

VIEW_S, NOISE_SIGMA, SEED = 20.0, 6.0, 0  # the lab renders this same frame
PATCH_ROWS, PATCH_COLS = slice(148, 154), slice(55, 61)  # where the road meets the yellow line
MARK = (150, 59)  # one yellow-paint pixel inside that patch: frame[150, 59]
NAMES = ("sky", "grass", "road", "yellow", "white")
TEXT, PANEL, BACKGROUND, BORDER = CHROME["text"], CHROME["panel"], CHROME["background"], CHROME["border"]
LABEL_BOX = {"boxstyle": "round,pad=0.25", "fc": PANEL, "ec": BORDER}  # text that sits on top of a picture
GREY = LinearSegmentedColormap.from_list("channel", [BACKGROUND, TEXT])  # one channel as brightness: 0 darkest, 255 brightest
BASE = TEXT  # neutral chrome for what names no quantity (the quantity hues are saved for the lessons that measure them)
DIM, DIMMER = shades(BORDER, 5)[3], shades(BORDER, 5)[2]  # two quieter greys from the same chrome
THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")
IOU_TRUTH = (slice(3, 5), slice(1, 6))  # iou_grid: the 10 true paint cells
IOU_MARKED = ((slice(3, 5), slice(2, 6)), (slice(5, 6), slice(2, 6)))  # and the 12 cells the mask marks
FS = 12  # smallest annotation size


def _frames(offset=0.0):
    road = make_scenario("straight").road
    state = BicycleState(*road.ego_pose(VIEW_S, offset), TARGET_SPEED, 0.0)
    camera = Camera()
    clean = camera.render(road, state)
    noisy, labels = camera.render(road, state, noise_sigma=NOISE_SIGMA, seed=SEED, return_labels=True)
    return clean, noisy, labels


def _distance(frame, paint):
    return np.linalg.norm(frame.astype(float) - PALETTE[paint], axis=-1)


def _mask_picture(yellow=None, white=None):
    """A mask pair drawn as a picture: yellow mask in paint yellow, white mask in paint white, everything else dark."""
    shape = (yellow if yellow is not None else white).shape
    picture = np.tile(to_rgb(PANEL), (*shape, 1))
    for mask, colour in ((yellow, SCENE["yellow"]), (white, SCENE["white"])):
        if mask is not None:
            picture[mask] = to_rgb(colour)
    return picture


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _show(ax, picture, title=None, **kwargs):
    ax.imshow(picture, interpolation="nearest", aspect="equal", **kwargs)
    if title:
        ax.set_title(title, fontsize=FS + 1, pad=8)


def _rgb_text(rgb):
    return "\n".join(str(int(v)) for v in rgb)


def _show_patch(ax, patch, title, outline=None, fontsize=FS - 1):
    """A few pixels drawn large, each labelled with its red, green and blue values; `outline` = (row, col) of one pixel to ring."""
    _show(ax, patch)
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            dark = float(np.dot(patch[i, j], (0.299, 0.587, 0.114))) > 150
            ax.text(j, i, _rgb_text(patch[i, j]), ha="center", va="center", fontsize=fontsize, linespacing=1.15, color=BACKGROUND if dark else TEXT)
    if outline is not None:
        for colour, width in ((BACKGROUND, 6), (TEXT, 3)):
            ax.add_patch(Rectangle((outline[1] - 0.5, outline[0] - 0.5), 1, 1, fill=False, edgecolor=colour, lw=width))
    ax.set_xticks(range(patch.shape[1]), range(PATCH_COLS.start, PATCH_COLS.stop))
    ax.set_yticks(range(patch.shape[0]), range(PATCH_ROWS.start, PATCH_ROWS.stop))
    ax.set(xlabel="column index", ylabel="row index")
    ax.set_title(title, fontsize=FS + 1, pad=8)


def _patch_box(ax):
    r, c = PATCH_ROWS, PATCH_COLS
    ax.add_patch(Rectangle((c.start - 0.5, r.start - 0.5), c.stop - c.start, r.stop - r.start, fill=False, edgecolor=TEXT, lw=2))


def _camera_to_masks(path, clean):
    yellow, white = paint_masks(clean)
    _, _, labels = _frames()
    assert (yellow == (labels == YELLOW)).all() and (white == (labels == WHITE)).all()  # the reference masks are the answer key on a clean frame
    fig = Figure(figsize=(11, 3.9))
    for ax, picture, title in zip(fig.subplots(1, 2), (clean, _mask_picture(yellow, white)),
                                  ("the camera frame", f"paint_masks: {yellow.sum()} yellow, {white.sum()} white")):
        _show(ax, picture, title)
        _bare(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


FRAME_GRID_DIGIT = 31  # font size of the RGB digits in frame_grid: they must stay at least 9 px tall when the PNG is shown 330 px wide


def _frame_grid(path, clean):
    r, c = PATCH_ROWS, PATCH_COLS
    fig = Figure(figsize=(11, 15.2))
    full, zoom = fig.subplots(2, 1, gridspec_kw={"height_ratios": [0.2, 1], "hspace": 0.17})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.975, bottom=0.04)
    _show(full, clean, "frame: 180 rows x 320 columns x 3 colours")
    _patch_box(full)
    full.set(xlabel="column index", ylabel="row index (row 0 is the top)")
    _show_patch(zoom, clean[r, c], f"the box: rows {r.start}\u2013{r.stop - 1}, columns {c.start}\u2013{c.stop - 1}", fontsize=FRAME_GRID_DIGIT)
    zoom.tick_params(labelsize=FS + 4)
    fig.savefig(path, dpi=100)


def _row_col(path, clean):
    fig = Figure(figsize=(11, 4.9))
    ax = fig.subplots()
    _show(ax, clean)
    ax.set_yticks([0, 75, 179])
    ax.set_xticks([0, 160, 319])
    ax.xaxis.tick_top()
    ax.tick_params(pad=26)  # keeps the 0 ticks clear of the frame[0, 0] circle that overhangs the corner
    ax.xaxis.set_label_position("top")
    ax.set(xlabel="column index (0 at the left)", ylabel="row index (0 at the top)")
    pixels = ((0, 0, (36, 24)), (179, 160, (200, 128)), (*MARK, (110, 105)))
    for row, col, text_at in pixels:
        ax.add_patch(Circle((col, row), 7, fill=False, edgecolor=TEXT, lw=2.5, clip_on=False))
        ax.annotate(f"frame[{row}, {col}] = {tuple(int(v) for v in clean[row, col])}", xy=(col, row), xytext=text_at, fontsize=FS + 1, va="center",
                    bbox=LABEL_BOX, arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1.5, "shrinkA": 0, "shrinkB": 7})
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _channels(path, clean):
    fig = Figure(figsize=(11, 3.6))
    for k, (ax, name) in enumerate(zip(fig.subplots(1, 3), ("red", "green", "blue"))):
        _show(ax, clean[:, :, k], f"{name}, frame[:, :, {k}]", cmap=GREY, vmin=0, vmax=255)
        _bare(ax)
    blue = fig.axes[2]
    blue.annotate("yellow line: 40, road: 95", xy=(MARK[1], MARK[0]), xytext=(110, 30), fontsize=FS + 1, ha="center", bbox=LABEL_BOX,
                  arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 1.5})
    fig.text(0.5, 0.02, "all three on the same grey scale, 0 (darkest) to 255 (brightest)", ha="center", fontsize=FS)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=100)


def _slices(path, clean):
    fig = Figure(figsize=(11, 5.2))
    ax = fig.subplots()
    _show(ax, clean)
    width = clean.shape[1]
    ax.add_patch(Rectangle((-0.5, -0.5), width, 76, facecolor=BACKGROUND, alpha=0.6, edgecolor=TEXT, lw=2))
    ax.add_patch(Rectangle((-0.5, 99.5), width, 80, facecolor=TEXT, alpha=0.25, edgecolor=TEXT, lw=2))
    ax.text(160, 38, "frame[:76], shape (76, 320, 3), all sky", ha="center", va="center", fontsize=FS + 2, bbox=LABEL_BOX)
    ax.text(160, 112, "frame[100:180], shape (80, 320, 3)", ha="center", va="center", fontsize=FS + 2, bbox=LABEL_BOX)
    ax.set_yticks([0, 76, 100, 179])
    ax.set(xlabel="column index", ylabel="row index (row 0 is the top)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


ROW_TICKS = [80, 100, 120, 140, 160, 179]


def _line_width(path, clean, labels):
    rows = np.arange(79, 180)
    widths = np.array([np.count_nonzero(labels[r] == WHITE) for r in rows])
    camera = Camera()
    ahead = {r: float(camera.pixel_to_ground(camera.cx, r)[0]) for r in (179, 80)}
    fig = Figure(figsize=(11, 4.6))
    frame_ax, ax = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    _show(frame_ax, clean, "the road part of the frame, rows 79 to 179")
    ax.plot(widths, rows, marker="o", markersize=3, color=BASE, lw=1.5)
    ax.set(xlim=(0, 20), ylim=(179.5, 78.5), xlabel="width of the white line (pixels)")
    ax.set_yticks(ROW_TICKS)
    ax.tick_params(labelleft=True)  # the shared axis hides these by default: the rows must be readable on this panel too
    ax.set_title("how wide the 0.2 m line looks", fontsize=FS + 1, pad=8)
    frame_ax.set(ylabel="row index (row 0 is the top)", xlabel="column index", yticks=ROW_TICKS)
    for row, at in ((179, (0.4, 168)), (80, (8.0, 92))):
        w = widths[row - 79]
        ax.annotate(f"row {row}: {w} px (\u2248{ahead[row]:.0f} m)", xy=(w, row), xytext=at, fontsize=FS, va="center", ha="left",
                    arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 1.5})
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _rgb_cube(path):
    fig = Figure(figsize=(8.5, 7))
    ax = fig.add_subplot(projection="3d")
    ax.patch.set_alpha(0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0)  # no grey pane boxes on the dark background
        axis._axinfo["grid"]["color"] = BORDER
    ax.set(xlim=(0, 255), ylim=(0, 255), zlim=(0, 255), xticks=[0, 128, 255], yticks=[0, 128, 255], zticks=[0, 128, 255])
    ax.set(xlabel="red", ylabel="green", zlabel="blue")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(FS + 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-62)
    offsets = {"sky": (0, 0, 24), "grass": (-4, 0, -24), "road": (-26, 0, 26), "yellow": (0, 0, -26), "white": (-50, 0, -16)}
    for name, colour in zip(NAMES, PALETTE.astype(int)):
        ax.scatter(*colour, s=170, color=SCENE[name], edgecolor=TEXT, linewidth=1.2, depthshade=False)
        dx, dy, dz = offsets[name]
        ax.text(colour[0] + dx, colour[1] + dy, colour[2] + dz, f"{name}\n{tuple(int(v) for v in colour)}", fontsize=FS, ha="center", va="center", color=TEXT)
    road, yellow = PALETTE[ROAD].astype(float), PALETTE[YELLOW].astype(float)
    ax.plot(*zip(road, yellow), color=TEXT, lw=2, linestyle="--")
    mid = (road + yellow) / 2
    ax.text(mid[0] + 22, mid[1], mid[2] + 6, f"{np.linalg.norm(road - yellow):.1f}", fontsize=FS + 3, ha="left", va="center", color=TEXT)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=100)


def _distance_build(path):
    red, green, blue = (abs(int(a) - int(b)) for a, b in zip(PALETTE[YELLOW], PALETTE[ROAD]))
    flat = float(np.hypot(red, green))
    full = float(np.hypot(flat, blue))
    gap = 22  # between the two triangles; both are drawn in the same units
    x0 = red + gap
    fig = Figure(figsize=(11, 4.3))
    ax = fig.subplots()
    ax.set_aspect("equal")
    ax.set(xlim=(-12, x0 + flat + 14), ylim=(-30, green + 22))
    ax.axis("off")

    def triangle(origin, run, rise, run_text, rise_text, hyp_text):
        x, y = origin
        ax.plot([x, x + run, x + run], [y, y, y + rise], color=DIM, lw=2.5, solid_capstyle="round")
        ax.plot([x, x + run], [y, y + rise], color=BASE, lw=3.5)
        s = 9
        ax.plot([x + run - s, x + run - s, x + run], [y, y + s, y + s], color=TEXT, lw=1.5)  # the right angle
        ax.text(x + run / 2, y - 6, run_text, ha="center", va="top", fontsize=FS + 2)
        ax.text(x + run + 5, y + rise / 2, rise_text, ha="left", va="center", fontsize=FS + 2)
        ax.text(x + run / 2 - 4, y + rise / 2 + 4, hyp_text, ha="right", va="bottom", fontsize=FS + 3, color=BASE, fontweight="bold")

    triangle((0, 0), red, green, f"red: {red}", f"green: {green}", f"{flat:.1f}")
    triangle((x0, 0), flat, blue, f"{flat:.1f}", f"blue: {blue}", f"{full:.1f}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _nearest_bars(path):
    fig = Figure(figsize=(11, 3.9))
    faded, strong = DIM, BASE
    for k, (paint, left) in enumerate(((YELLOW, 0.17), (WHITE, 0.68))):
        ax = fig.add_axes([left, 0.2, 0.29, 0.62])
        others = [i for i in (2, 1, 0, 3, 4) if i != paint]
        values = [float(np.linalg.norm(PALETTE[paint].astype(float) - PALETTE[i])) for i in others]
        nearest = int(np.argmin(values))
        y = np.arange(len(others))[::-1]
        ax.barh(y, values, height=0.62, color=[strong if i == nearest else faded for i in range(len(others))])
        ax.set(xlim=(0, 360), ylim=(-0.6, 3.6), yticks=[], xlabel="colour distance", title=f"from {NAMES[paint]} paint to ...")
        ax.set_title(ax.get_title(), fontsize=FS + 1, pad=8)
        for yi, v, i in zip(y, values, others):
            ax.text(v + 6, yi, f"{v:.1f}" + ("  nearest" if i == others[nearest] else ""), va="center", fontsize=FS + 1)
            ax.add_patch(Rectangle((-0.46, yi - 0.2), 0.05, 0.4, transform=ax.get_yaxis_transform(),
                                   facecolor=SCENE[NAMES[i]], edgecolor=TEXT, lw=1, clip_on=False))
            ax.text(-0.38, yi, NAMES[i], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=FS + 1)
    fig.savefig(path, dpi=100)


def _noise_bell(path, clean, noisy):
    nudge = noisy.astype(int) - clean.astype(int)
    edge = int(np.abs(nudge).max())
    values = np.arange(-edge, edge + 1)
    counts = np.array([np.count_nonzero(nudge == v) for v in values])
    inner, outer = (np.count_nonzero(np.abs(nudge) <= k) / nudge.size for k in (6, 18))
    faint, mid, strong = DIMMER, DIM, BASE
    colours = [strong if abs(v) <= 6 else mid if abs(v) <= 18 else faint for v in values]
    fig = Figure(figsize=(11, 4.4))
    ax = fig.subplots()
    ax.bar(values, counts, width=1.0, color=colours, edgecolor=BACKGROUND, linewidth=0.4)
    top = counts.max()
    ax.set(xlim=(-edge - 1, edge + 1), ylim=(0, top * 1.42), xlabel="nudge added to one channel", ylabel="number of values (of 172,800)")
    ax.yaxis.set_major_formatter(THOUSANDS)
    for half, share, height in ((6.5, inner, 1.08), (18.5, outer, 1.24)):
        ax.annotate("", xy=(-half, top * height), xytext=(half, top * height), arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 1.5})
        ax.text(0, top * (height + 0.015), f"from −{int(half)} to +{int(half)}: {share * 100:.1f}%".replace("6.5", "6").replace("18.5", "18"),
                ha="center", va="bottom", fontsize=FS + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _noisy_patch(path, clean, noisy):
    fig = Figure(figsize=(11, 4.8))
    outline = (MARK[0] - PATCH_ROWS.start, MARK[1] - PATCH_COLS.start)
    for ax, frame, name in zip(fig.subplots(1, 2), (clean, noisy), ("clean", f"with noise, sigma = {NOISE_SIGMA:g}")):
        _show_patch(ax, frame[PATCH_ROWS, PATCH_COLS], f"the same pixels, {name}", outline)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _exact_fails(path, clean, noisy):
    yellow = PALETTE[YELLOW]
    fig = Figure(figsize=(11, 3.9))
    for ax, frame, name in zip(fig.subplots(1, 2), (clean, noisy), ("clean", "noisy")):
        found = np.all(frame == yellow, axis=-1)
        _show(ax, _mask_picture(yellow=found), f"{name}: {found.sum()} found")
        ax.set_xlabel(f"pixels equal to {tuple(int(v) for v in yellow)}", fontsize=FS + 1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _exact_odds(path):
    sigma = NOISE_SIGMA
    x = np.linspace(-20, 20, 801)
    pdf = lambda v: np.exp(-0.5 * (v / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    from math import erf, sqrt
    sliver = erf(0.5 / (sigma * sqrt(2)))
    fig = Figure(figsize=(11, 4.2))
    ax = fig.subplots()
    ax.plot(x, pdf(x), color=DIM, lw=2.5)
    inside = np.abs(x) <= 0.5
    ax.fill_between(x[inside], pdf(x[inside]), color=BASE, linewidth=0)
    ax.set(xlim=(-20, 20), ylim=(0, pdf(0) * 1.25), xlabel="nudge", ylabel="how common (bell height)")
    ax.annotate(f"−0.5 to +0.5: {sliver * 100:.1f}%", xy=(0, pdf(0) * 0.5), xytext=(7, pdf(0) * 0.95), fontsize=FS + 2, va="center", color=BASE,
                arrowprops={"arrowstyle": "->", "color": BASE, "lw": 2})
    ax.text(-3, pdf(6) * 1.06, "σ = 6", ha="center", fontsize=FS + 1)
    ax.annotate("", xy=(-6, pdf(6)), xytext=(0, pdf(6)), arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 1.2})
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _box(ax, centre, size, text=None, depth=False, fontsize=FS):
    """An array drawn as a box whose proportions hint at its shape; `depth` stacks three sheets for the colour axis."""
    (cx, cy), (w, h) = centre, size
    sheets = ((0.2, 0.2), (0.1, 0.1), (0.0, 0.0)) if depth else ((0.0, 0.0),)
    for dx, dy in sheets:
        ax.add_patch(Rectangle((cx - w / 2 + dx, cy - h / 2 + dy), w, h, facecolor=PANEL, edgecolor=TEXT, lw=1.8))
    if text:
        ax.text(cx, cy - h / 2 - 0.22, text, ha="center", va="top", fontsize=fontsize, family="monospace")


def _arrow(ax, start, end, text=None, above=True):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 2})
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + (0.2 if above else -0.2), text, ha="center", va="bottom" if above else "top", fontsize=FS + 1, family="monospace")


def _broadcast_shapes(path):
    fig = Figure(figsize=(11, 5.6))
    ax = fig.subplots()
    ax.set(xlim=(0, 11), ylim=(0, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")
    big = (3.2, 1.8)  # 320 wide, 180 tall
    # row 1: frame - yellow -> a frame-sized array
    _box(ax, (1.7, 4.6), big, "frame (180, 320, 3)", depth=True)
    ax.add_patch(Circle((3.95, 4.7), 0.3, facecolor=PANEL, edgecolor=TEXT, lw=1.8))
    ax.text(3.95, 4.7, "\u2212", fontsize=26, ha="center", va="center")
    _box(ax, (4.9, 4.7), (0.45, 0.14), "yellow (3,)")
    _arrow(ax, (5.5, 4.7), (6.7, 4.7), "stretched")
    _box(ax, (8.6, 4.6), big, "(180, 320, 3)", depth=True)
    # row 2: norm along the last axis, then the comparison
    _arrow(ax, (8.6, 3.2), (8.6, 2.45), None)
    ax.text(8.85, 2.85, "np.linalg.norm(…, axis=-1)", fontsize=FS + 1, va="center", family="monospace", ha="left")
    _box(ax, (8.6, 1.45), big, "(180, 320)")
    _arrow(ax, (6.9, 1.45), (5.3, 1.45), "< tau")
    _box(ax, (3.7, 1.45), big, "(180, 320) True/False")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _colour_distance(path, noisy, labels):
    fig = Figure(figsize=(11, 4.2))
    top = int(np.ceil(max(_distance(noisy, p).max() for p in (YELLOW, WHITE)) / 4) * 4)
    bins = np.arange(0, top + 1, 4)
    for ax, paint in zip(fig.subplots(1, 2, sharey=True), (YELLOW, WHITE)):
        d = _distance(noisy, paint)
        is_paint = labels == paint
        ax.hist(d[~is_paint], bins=bins, color=TEXT, alpha=0.55, edgecolor=BACKGROUND, linewidth=0.4)
        ax.hist(d[is_paint], bins=bins, color=SCENE[NAMES[paint]], edgecolor=BACKGROUND, linewidth=0.4)
        ax.axvline(PAINT_DISTANCE, color=TEXT, linewidth=2, linestyle="--")
        ax.set(yscale="log", xlim=(0, top + 8), ylim=(0.7, 3e5), xlabel=f"distance to {NAMES[paint]} paint {tuple(int(v) for v in PALETTE[paint])}", title=f"distance to {NAMES[paint]}, every pixel")
        ax.set_title(ax.get_title(), fontsize=FS + 1, pad=8)
        ax.yaxis.set_major_formatter(THOUSANDS)
        gap_lo, gap_hi = d[is_paint].max(), d[~is_paint].min()
        ax.text(PAINT_DISTANCE - 3, 1e5, f"τ = {PAINT_DISTANCE:g}", fontsize=FS + 1, ha="right", va="center")
        ax.text(2, 3e3, f"paint\n≤ {gap_lo:.1f}", fontsize=FS, ha="left", va="center")
        ax.text(PAINT_DISTANCE + 6, 3e4, f"nearest other pixel: {gap_hi:.1f}", fontsize=FS, ha="left", va="center")
    fig.axes[0].set_ylabel("number of pixels (each step up is 10 times more)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _tau_sweep(path, noisy, labels):
    yellow, white = paint_masks(noisy)  # tau = 60
    truth_y, truth_w = labels == YELLOW, labels == WHITE
    assert (yellow == truth_y).all() and (white == truth_w).all()
    loose_y, tight_w = _distance(noisy, YELLOW) < 10, _distance(noisy, WHITE) < 120
    panels = [
        (_mask_picture(yellow=loose_y), f"yellow, τ = 10:\n{loose_y.sum()} of {truth_y.sum()}"),
        (_mask_picture(yellow, white), f"yellow and white, τ = 60:\n{yellow.sum()} and {white.sum()}, nothing else"),
        (_mask_picture(white=tight_w), f"white, τ = 120:\n{tight_w.sum():,} pixels"),
    ]
    fig = Figure(figsize=(11, 3.8))
    for ax, (picture, title) in zip(fig.subplots(1, 3), panels):
        _show(ax, picture, title)
        _bare(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.15)  # crop the empty band under the three masks


def _accuracy_trap(path, labels):
    truth = labels == YELLOW
    right = 1 - truth.sum() / truth.size  # what a mask that marks nothing gets right
    fig = Figure(figsize=(11, 3.9))
    first, second, third = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.8]})
    _show(first, _mask_picture(yellow=truth), f"truth: {truth.sum()} pixels")
    _show(second, _mask_picture(yellow=np.zeros_like(truth)), "a mask that marks nothing")
    for ax in (first, second):
        _bare(ax)
    heights, colours = (right, 0.0), (BASE, BASE)
    third.bar([0, 1], heights, width=0.6, color=colours)
    third.set(xlim=(-0.6, 1.6), ylim=(0, 1.25), xticks=[], yticks=[0, 0.5, 1])
    third.set_title("two scores", fontsize=FS + 1, pad=8)
    for x, text, h, colour in ((0, f"share right\n{right * 100:.1f}%", right, colours[0]), (1, "IoU\n0", 0.0, colours[1])):
        third.text(x, h + 0.04, text, ha="center", va="bottom", fontsize=FS + 1, color=colour)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _iou_cells():
    truth = np.zeros((8, 8), bool)
    truth[IOU_TRUTH] = True
    marked = np.zeros((8, 8), bool)
    for part in IOU_MARKED:
        marked[part] = True
    return truth, marked


def _iou_grid(path):
    truth, marked = _iou_cells()
    both, missed, wrong = marked & truth, truth & ~marked, marked & ~truth
    assert (both.sum(), missed.sum(), wrong.sum()) == (8, 2, 4)
    fig = Figure(figsize=(11, 4.9))
    ax = fig.subplots()
    ax.set_aspect("equal")
    ax.set(xlim=(-6.5, 15.5), ylim=(7.5, -0.5), title=f"score = {both.sum()} / {(marked | truth).sum()} = {both.sum() / (marked | truth).sum():.3f}")
    ax.set_title(ax.get_title(), fontsize=FS + 4, pad=10)
    ax.axis("off")
    for r in range(8):
        for c in range(8):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=PANEL, edgecolor=BORDER, lw=1,
                                   ))
    for r, c in zip(*np.nonzero(marked)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=QUANTITY["perception"], edgecolor=BORDER, lw=1))
    for r, c in zip(*np.nonzero(truth)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=TEXT, lw=3))
    for text, xy, at, ha in ((f"both: {both.sum()}", (5.3, 3.0), (9.3, 2.6), "left"), (f"missed: {missed.sum()}", (0.8, 3.5), (-1.2, 1.0), "right"),
                             (f"wrongly marked: {wrong.sum()}", (5.3, 5.0), (9.3, 5.0), "left")):
        ax.annotate(text, xy=xy, xytext=at, fontsize=FS + 3, va="center", ha=ha, arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 1.5, "shrinkA": 4, "shrinkB": 0})
    ax.text(9.3, 3.7, f"either: {(marked | truth).sum()}  ({truth.sum()} true + {wrong.sum()} wrong)", fontsize=FS + 3, va="center", ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _uint8_wrap(path):
    def angle(v):  # 0 at the top, values growing clockwise
        return np.pi / 2 - 2 * np.pi * np.asarray(v, dtype=float) / 256

    def point(v, r):
        return r * np.cos(angle(v)), r * np.sin(angle(v))

    fig = Figure(figsize=(8.6, 7.4))
    ax = fig.subplots()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set(xlim=(-1.55, 1.55), ylim=(-1.4, 1.4))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color=TEXT, lw=2)
    for v in range(0, 256, 8):
        ax.plot(*zip(point(v, 1.0), point(v, 1.04)), color=BORDER, lw=1)
    for v in (0, 64, 128, 192):
        ax.plot(*zip(point(v, 0.95), point(v, 1.08)), color=TEXT, lw=2.5)
        x, y = point(v + (3 if v == 0 else 0), 1.2)
        ax.text(x, y, str(v), ha="center", va="center", fontsize=FS + 4)
    seam = -0.5  # halfway between 255 and 0
    ax.plot(*zip(point(seam, 0.7), point(seam, 1.12)), color=TEXT, lw=2, linestyle=":")
    ax.text(-0.08, 1.22, "255 | 0", ha="right", va="center", fontsize=FS + 1)
    ax.text(0, 0.1, "the 256 values\nof a uint8", ha="center", va="center", fontsize=FS + 2)
    ax.text(0, -0.25, "the seam: one step down\nfrom 0 is 255", ha="center", va="center", fontsize=FS)
    start, steps = 90, 230
    path_values = start - np.linspace(0, steps, 300)
    arc = np.array([point(v, 0.82) for v in path_values])
    ax.plot(arc[:-6, 0], arc[:-6, 1], color=BASE, lw=4)
    ax.annotate("", xy=arc[-1], xytext=arc[-8], arrowprops={"arrowstyle": "-|>", "color": BASE, "lw": 4, "mutation_scale": 28})
    for v, text, r, ha in ((90, "start 90", 1.22, "left"), (116, "ends at 116", 1.2, "left")):
        ax.plot(*point(v, 0.82), marker="o", markersize=9, color=BASE)
        ax.plot(*zip(point(v, 0.95), point(v, 1.08)), color=BASE, lw=2.5)
        x, y = point(v, r)
        ax.text(x, y, text, ha=ha, va="center", fontsize=FS + 2, color=BASE)
    x, y = point(start - steps / 2, 0.62)
    ax.text(x, y, "−230", ha="center", va="center", fontsize=FS + 4, color=BASE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def _next_lesson(path):
    road = make_scenario("straight").road
    pose = road.ego_pose(VIEW_S, 0.5)  # 0.5 m left of the lane centre
    assert abs(road.ego_truth(*pose)[1] - 0.5) < 1e-6  # the sim's own e: left is positive, as in lesson 0.1.1
    frame = Camera().render(road, BicycleState(*pose, TARGET_SPEED, 0.0))
    yellow, white = paint_masks(frame)
    height, width = frame.shape[:2]
    fig = Figure(figsize=(11, 4.9))
    ax = fig.subplots()
    _show(ax, _mask_picture(yellow, white), "the two masks, with the car 0.5 m left of the lane centre")
    ax.set(ylim=(height + 22, -0.5), xlim=(-0.5, width - 0.5))
    ax.axis("off")
    ax.plot([width / 2], [height - 1], marker="^", markersize=18, color=TEXT, clip_on=False)
    ax.text(width / 2, height + 8, "the car", ha="center", va="center", fontsize=FS + 3)
    ax.text(width / 2, 125, "?", ha="center", va="center", fontsize=60, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def make(out_dir):
    clean, noisy, labels = _frames()
    _camera_to_masks(out_dir / "camera_to_masks.png", clean)
    _frame_grid(out_dir / "frame_grid.png", clean)
    _row_col(out_dir / "row_col.png", clean)
    _channels(out_dir / "channels.png", clean)
    _slices(out_dir / "slices.png", clean)
    _line_width(out_dir / "line_width.png", clean, labels)
    _rgb_cube(out_dir / "rgb_cube.png")
    _distance_build(out_dir / "distance_build.png")
    _nearest_bars(out_dir / "nearest_bars.png")
    _noise_bell(out_dir / "noise_bell.png", clean, noisy)
    _noisy_patch(out_dir / "noisy_patch.png", clean, noisy)
    _exact_fails(out_dir / "exact_fails.png", clean, noisy)
    _exact_odds(out_dir / "exact_odds.png")
    _broadcast_shapes(out_dir / "broadcast_shapes.png")
    _colour_distance(out_dir / "colour_distance.png", noisy, labels)
    _tau_sweep(out_dir / "tau_sweep.png", noisy, labels)
    _accuracy_trap(out_dir / "accuracy_trap.png", labels)
    _iou_grid(out_dir / "iou_grid.png")
    _uint8_wrap(out_dir / "uint8_wrap.png")
    _next_lesson(out_dir / "next_lesson.png")
