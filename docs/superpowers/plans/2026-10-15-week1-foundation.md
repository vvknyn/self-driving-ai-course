# Week 1 Foundation (C1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring all four units (course/0.1.1 … 0.1.4) onto one foundation before Week 1 is rewritten. That means one dark palette with computed gates, lab figures in lecture order, plain-text captions, no dotted ids in learner-visible text, one section-reference form, the "Check my work" label, and the playground fixes carried from the site run. Every check must stay green.

**Architecture:** Every fact has one source of truth, and a gate fails loudly when that source is bypassed.
- `zero2fsd/palette.py` owns every colour. One shared style helper owns the dark figure look.
- `scripts/build_labs.py` owns caption rendering and figure order.
- One shared label module owns "Week N · <title> · Exercise A".
- One grader helper owns section hints.
- The playground gains a per-exercise draft store and preset links.

The gates are pytest tests that read source (AST), built artifacts, or the grader's real stdout.

**Tech Stack:** Python 3 (repo `.venv`), matplotlib, pytest, Playwright for Python (browser tests), vanilla ES-module JavaScript in `playground/`, CodeMirror from a CDN with a textarea fallback.

**Spec:**
- `docs/superpowers/plans/2026-10-15-week1-rulings.md`: BINDING. C1 = R2–R10.
- `docs/superpowers/specs/2026-09-28-zero2fsd-v2-design.md`.
- `docs/superpowers/specs/2026-10-11-playground-design.md`.
- Writing standard: `docs/writing/v1.md` and `docs/writing/v2.md` (v2 adopted by D1; its "DRAFT" header is stale).

## Global Constraints

- **Worktree:** `/Users/vivek/Antigravity/self-driving-ai-course-week1`, branch `feat/week1-meet-your-car`.
  - Use absolute paths or `git -C`.
  - Never touch `/Users/vivek/Antigravity/self-driving-ai-course` (the primary checkout), except to run its `.venv` Python.
- **Python:**
  - Interpreter: `/Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python`.
  - pytest always runs with `-p no:cacheprovider` and env `PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache`.
- **Freshness:** `tests/playground/test_build.py` (dist/playground) and `tests/zero2fsd/test_build_labs.py` (lab.ipynb) fail unless the built artifacts match their inputs. Every task that changes a builder input runs the exact rebuild command(s) and commits the rebuilt artifacts in the same commit.
- **Process safety:**
  - Node/JS tests run in run mode only, never watch.
  - Never bare `playwright install`.
  - Never `git stash`.
  - Never pkill/killall with a pattern.
- **Commits:**
  - Explicit `git add <paths>`, never `-A` or `.`.
  - Message: `<type>(<area>): <what>`.
  - The last line is exactly `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Fix the class:**
  - Each gate or test includes at least one case OTHER than the one that surfaced the problem.
  - A gate is shown RED on a seeded violation before it goes green.
- **Lean code:**
  - No gate that enumerates names: derive the set.
  - No lying fallbacks: a missing value fails loudly.
  - Smallest complete diff.
  - One source of truth per fact.
- **Measured facts only (R18):** every number shown to a learner comes from a run of the course sim.
- **Learner-visible names (R2, R4):**
  - Week N is the unit at syllabus position N.
  - Lessons are named by title, weeks as "Week N · <title>", exercises as "Exercise A/B…".
  - Dotted ids stay in code cells, URLs and function arguments only.

## Interfaces between parts

### Part A: Palette, dark figures, lab figure order, captions (R3, R5, R6)

Worktree for every command: `/Users/vivek/Antigravity/self-driving-ai-course-week1` (branch `feat/week1-meet-your-car`). Every command below is written out in full (absolute paths; `cd` into the worktree is part of the command because shell state does not persist between calls). `PY` below always means `/Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python`; every Python/pytest command carries `PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg` and pytest carries `-p no:cacheprovider`. Never `git stash`; never `git add -A`/`git add .`; the last line of every commit message is exactly `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (Task 1's text keeps its older trailer; the controller rewrites it at assembly).

### Part interfaces

What Part A produces that Part B, Part C and plan C2 consume:

- `zero2fsd/palette.py` (unchanged keys `QUANTITY`, `SCENE`, `CHROME`, `palette_dict()`; `palette.json` output is byte-identical) gains **`shades(hex_colour: str, n: int) -> list[str]`**: `n` shades of one hue for a quantity drawn `n` times; `shades(h, n)[0] == h`, later entries step evenly toward `CHROME["text"]` (at most `MIX = 0.65` of the way). Pure Python (no matplotlib): still importable in the browser.
- **`scripts/figstyle.py`** exports **`RC: dict`** (matplotlib rcParams: dark `figure.facecolor`/`axes.facecolor`, text/tick/label colours, `axes.prop_cycle` = the four `QUANTITY` hues). `scripts/make_figures.py` applies it with `matplotlib.rc_context(RC)` around every figures.py `make(out_dir)`; a figures.py never sets a background, text colour or cycle itself. Any future figures.py (C2) imports its colours only from `zero2fsd.palette` (`CHROME`, `QUANTITY`, `SCENE`, `shades`).
- **Gate G2** = `tests/zero2fsd/test_figure_colours.py::test_no_figure_module_holds_a_string_colour_literal`, parametrized over every `course/*/figures.py` plus `scripts/figstyle.py`. A new unit's figures.py is scanned automatically. What it flags: any `#hex` string anywhere; any string (also inside f-strings and conditionals) under a keyword or dict key whose name is a matplotlib colour alias (`c`, `ec`, `fc`, `mec`, `mfc`, ... read from matplotlib) or contains `color`/`colour`; an all-number tuple/list under such a keyword; any positional string matplotlib would read as a format string naming a colour (`"ro"`, `"C3"`, `"red"`). Not flagged: `QUANTITY["e_off"]` (a subscript key), marker/linestyle formats (`"o"`, `"--"`), prose.
- **Gate G3** = same module: every `QUANTITY` hue and every `shades(hue, 4)` entry has WCAG contrast ≥ 3:1 against `CHROME["background"]` and `CHROME["panel"]`; `CHROME["text"]` ≥ 4.5:1 on the panel. Contrast is computed from the palette values, never tabulated.
- `scripts/build_labs.py` gains **`caption_parts(alt: str) -> tuple[str, str]`** = `(plain, caption_html)`; `figure_html(unit, ref)` keeps its signature and now emits `<img ... alt="<plain>">` and `<figcaption><em>Lecture figure N. <caption_html></em></figcaption>` where `` `code` `` spans of the lecture alt render as `<code>…</code>` (HTML-escaped) and the img alt is the plain text (backticks removed). `BuildError` (from `_course`, a `ValueError`) on an unbalanced backtick or on `*`, `[`, `$` outside a code span. **Part B / C2:** any lecture alt text may hold only plain text and code spans; anything parsing lab captions must accept `<code>` inside `<em>`.
- `scripts/build_labs.py` **`check_lecture_order(unit: Path, shown: list[str]) -> None`**, called by `compile_lab`: the figures a lab shows, in lab order, must be a strict subsequence of the lecture's figure order (`BuildError` otherwise). **Part B / C2:** when adding or moving a `![](figures/x.png)` line in a lab.py, keep lecture order; `python scripts/build_labs.py --check` tells you.
- Lab figure embeds moved by Part A (Part B anchors its text edits on the new positions): `course/0.1.2-images-are-arrays/lab.py` shows `colour_distance.png` before `exact_vs_threshold.png` (same markdown cell as before); `course/0.1.4-closing-the-loop/lab.py` shows `steady_state.png` right after `gain_sweep.png` in the section-4 cell (the one-line cell after Practice Problem 3 is gone).
- Learner-visible colour words Part A relies on staying true: "red" stays the word for the perception hue (`#FC6255`, coral red) in lecture alt texts 0.1.1:31 ("red dot") and 0.1.2:21/:23 ("red box") and in two 0.1.3 figure titles; Part A changes the 0.1.3 title word "black lines" → "grey lines" and the 0.1.4 bend legend to "shaded: left bend / unshaded: right bend" inside figures.py. Part B must not reintroduce colour names that the palette does not draw (no green/purple/orange/cyan/black in figure prose).
- Not touched by Part A (report to the controller, not in scope of R3's figures.py rule): `zero2fsd/sim/topdown.py` keeps `EDGE_COLOR = "#666666"`, `DASH_COLOR = "#c9a800"` (library used by Colab dashboards on white backgrounds); `course/0.1.3-your-first-perception-box/lab.py:64` uses `"tab:orange"`/`"tab:cyan"` in a Colab plot (Part B's file).

### Part C: Playground drafts, presets, back link, link-check punctuation, spec text (R8, R10)

All paths are relative to the worktree `/Users/vivek/Antigravity/self-driving-ai-course-week1` (branch `feat/week1-meet-your-car`, HEAD 335491a). Every command below is run from that directory with absolute paths; never touch `/Users/vivek/Antigravity/self-driving-ai-course`.

Shared values (copy exactly):
- `PY=/Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python` (Python 3.13.5, playwright installed, chromium present).
- Every pytest run: `PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache $PY -m pytest -p no:cacheprovider …`. Browser tests also need `-m browser` (pytest.ini deselects them by default) and are run ONLY by node id; their output always goes to a file under `/private/tmp/claude-501/c1-out/` (`mkdir -p` it first) and you read the tail.
- Rebuild of the committed playground: `$PY scripts/build_course_bundle.py` (prints `wrote …/course/dist/course.json (4 ready units, N figures)` then `wrote …/course/dist/playground`). `tests/playground/test_build.py::test_the_committed_playground_is_what_the_source_builds` (:425) fails until `course/dist/playground/` matches `playground/`; after a rebuild `git status --short course/dist` must list ONLY paths under `course/dist/playground/` (course.json has no input in this part; if it shows modified, stop and report instead of committing it).
- Facts verified in the tree at 335491a: `playground/index.html:41` already reads `<a id="unit-link" target="_top">Back to the lecture</a>` (added in 78bffe7, the first page commit) and `course/dist/playground/index.html` carries it too — so R8(c) needs a pinning test, not a source change. The site repo's `feat/playground-site-seams` branch is not checked out anywhere on this Mac (`/Users/vivek/Antigravity/vivekn` is on `main`), so the site's sync rule is cited from the recon (checklist :265: the sync refuses `exercise=0.1.1.a.`), not re-read.

### Part interfaces

Produced by Part C, consumed by Part B (same plan) and by C2/S1:
- `playground/draft.js` (new ES module, no dependencies): `readDraft(exerciseId) -> string | null` and `saveDraft(exerciseId, code, starter) -> void`. Key: `` `learnfsd.v2.draft.${exerciseId}` `` (e.g. `learnfsd.v2.draft.0.1.1.a`). Rule: a draft is code that differs from the starter — `saveDraft` removes the key when `code === starter`, writes it otherwise; both never throw (blocked storage reads as "no draft").
- `playground/app.js`: `choose({ records }, savedDraft)` returns `{ record, code, notice, autoRun, draft }`; `draft` is the saved text when the URL named a preset and the draft differs from the opened code, else `null`. `mountEditor(parent, doc, onChange)`; `onChange(text)` fires after every edit (CodeMirror and the textarea fallback). Presets are read from `record.presets` by name only — C2's file-backed presets (R13) need no page change.
- `playground/index.html`: new controls `<button id="reset">Reset to starter</button>` (in `.toolbar`, after `#stop`) and `<button id="restore-draft" hidden>Restore my draft</button>` (in `.status-actions`). Part B renames only `#check`'s label; it must leave these two as they are.
- Status copy: `You have a saved draft of this exercise.` is appended to `app.notice` (shown as `… Ready.`) when a preset link finds a different draft.
- `tests/playground/test_browser.py` helpers: `DRAFT_KEY = "learnfsd.v2.draft."`, `stored_draft(page, exercise_id)`, `open_editor(page, base_url, exercise_id, query="")` (waits for the editor only, not the runtime), `editor_text(page)`.
- `scripts/build_playground.py:24` `_VALUE = r"""([^&"'<>#\s)]+)"""`: a lecture link's id/preset runs to a delimiter (`& " ' < > # whitespace )`); sentence punctuation after a bare link is part of the id and fails the build. C2's R15 lecture links must therefore be markdown links `[text](playground/index.html?exercise=0.1.1.a)` with the sentence's punctuation after the `)`.
- Spec `docs/superpowers/specs/2026-10-11-playground-design.md`: link form `playground/index.html?exercise=<id>[&preset=<p>]` at :115 and :118; view.js exports listed at :110 with "lecture runs in `playground/runs/`" (the R14 location C2 builds); image kind marked shipped at :68 and :170.

### Part B: Learner-visible names — no dotted ids, one section form, "Check my work" (R4, R7, R9)

Worktree `/Users/vivek/Antigravity/self-driving-ai-course-week1` (branch feat/week1-meet-your-car). Runs AFTER Parts A and C: every edit below is anchored by function name or by exact text, never by line number alone.

### Part interfaces

- **`zero2fsd/labels.py` (new, Task 11) is the one place the learner-facing names are spelled**; it ships to Colab (pip) and to the playground (bundle.zip packs `zero2fsd/**/*.py`), so it must stay dependency-free: `exercise_name("0.1.1.b") == "Exercise B"`, `exercise_label(1, "Meet your car", "0.1.1.b") == "Week 1 · Meet your car · Exercise B"`, `lecture_section("Meet your car", "Lateral error") == 'lecture "Meet your car", section "Lateral error"'`, `CHECK_MY_WORK == "Check my work"`. `scripts/build_playground.py` (label), `zero2fsd/grade/_report.py` (tick/cross head lines and hints), `playground/shim.py` (boundary TypeErrors) and `scripts/build_labs.py` (the Colab check-cell label) import it. The site's TypeScript copy of the label is out of scope. **Part C2 (R16):** label lab headings through `exercise_name`, not by hand.
- **Grader registration changes shape (Task 12):** `exercise(exercise_id, *, checked, lecture, section)` replaces `hint=`; each `zero2fsd/grade/units/u0_1_n.py` declares `LECTURE = "<syllabus title>"`. The registered function carries `.lecture` and `.section`, read by `scripts/verify_units.py`. Learner output: `✓ Exercise A: …`, `✗ Exercise A: …`, `hint:     lecture "Meet your car", section "Summarising lane error"`; shim errors `Exercise B: must be a Car, got int`. Tests that asserted `{exercise_id}:` prefixes are updated in B2.
- **`scripts/verify_units.py` is gate G5 + R7's class test (Tasks 12, 14).** `verify(root)` additionally reports: any course id (syllabus unit ids + ready units' exercise ids, whole-token match) in the grader's real stdout for the skeleton run and the solution run; any such id in lecture prose (fenced code and link targets excluded, inline code included), lab.py markdown cells, or a yaml string of unit.yaml/quiz.yaml/cards.yaml that is not itself exactly an id; any `section "X"` in those texts, or in a grader hint, whose X is not a `##` heading of that unit's lecture.md. `scripts/_course.py` gains `lecture_sections(text) -> set[str]`. The mini fixture's lecture heading "Measuring lane error" becomes "Summarising lane error" (and its quiz slugs follow) because the fixture grades with the real 0.1.1 grader, whose hint cites that section.
- **Colab check cells (Task 13):** `build_labs.labelled(cell)` returns `# Check my work\n` + the source for every code cell whose source has a line starting `check(`, and the cell as authored otherwise; `compile_lab` feeds code cells through it (one token: `else cell.source` → `else labelled(cell)`). Authored lab.py never spells the label. **Part A owns `compile_lab`** (Tasks 1/2: `caption_parts`, `figure_html`, `check_lecture_order`); B3 adds one function above it and changes that one token, anchored on the expression that supplies a code cell's source, whatever shape Part A leaves the statement in.
- **Text contract after B3 (for C2 and every later unit):** lessons are named by title in quotes (`in "Images are arrays"`), exercises as `Exercise A`, a lab title is `# Lab: <title>`, an exercise heading `### Exercise A · \`fn\``, a section citation `section "<Heading>"` (also inside tutor prompts), a proof `passes Exercise A's "Check my work" in the lab`. Reference values (`id:`, `prerequisites:`, `exercises[].id`, link URLs, code) keep the ids.
- Part B does not touch `#reset`/`#restore-draft` (Part C), `course/*/figures.py`, `zero2fsd/palette.py`, `scripts/make_figures.py`, or the caption/figure-order code of `scripts/build_labs.py` (Part A). `playground/app.js` build-error strings (`Exercise ${record.id} has kind …`) are developer errors that `build_playground.py` makes unreachable with a valid bundle: left alone.

**Constraints for every task (copy into each dispatch):** Python = `/Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python`; every Python command runs with `PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg`; pytest always `-p no:cacheprovider -q` on the named files only (no full suite; browser tests are excluded by pytest.ini); never `playwright install`, never `git stash`, never pkill/killall by pattern; `git add <explicit paths>`; commit message `<type>(<area>): <what>` ending with the trailer line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; a task that changes a builder input rebuilds and commits the built artifact in the same commit (`course/*/lab.ipynb` via `scripts/build_labs.py`; `course/dist/**` via `scripts/build_course_bundle.py`); never touch `/Users/vivek/Antigravity/self-driving-ai-course` (the primary checkout).

## Review Focus

### Part A
Input classes a reviewer should check are really exercised (each was missing from the first draft and now has a test in the owning task):

1. **A colour hidden in a compound format string or a bare positional string** — `ax.plot(x, y, "ro")`, `ax.plot(x, y, "C3")` carry a colour with no colour keyword at all, so a keyword-name rule never sees them. Covered in Task 3 by the seeded cases `format string with a colour` and `bare colour string` in `VIOLATIONS` (`test_a_seeded_colour_literal_is_flagged_with_its_line`), which go through matplotlib's own `_process_plot_format`; the `marker and linestyle formats` case in `ALLOWED` proves `"o"` and `"--"` are not flagged.
2. **A colour given as numbers, not a string** — `color=(1.0, 0.0, 0.0)` is a literal colour with no string in it. Covered in Task 3 by the `rgb tuple` seeded case (all-number tuple/list under a colour keyword or dict key is reported as its unparsed text).
3. **Regenerated PNGs committed without the dist copies** — `course/dist/figures/<slug>/*.png` are plain copies; before Part A nothing failed when they went stale (the playground and notebook freshness tests do not look at them). Covered in Task 4 by `tests/zero2fsd/test_bundle.py::test_the_committed_dist_figures_are_byte_copies_of_the_units_figures`, parametrized over every ready unit with a `figures/` folder, and shown RED in Task 4 Step 6 (after `make_figures.py`, before `build_course_bundle.py`) and GREEN in Step 7.

### Part C
1. **A stored draft equal to the code the link opens.** Another tab (or the site's own iframe) can leave `learnfsd.v2.draft.<id>` holding the starter itself; offering "Restore my draft" there restores nothing and teaches the learner to ignore the control. Test added to Task 7: `test_a_draft_equal_to_the_opened_preset_is_not_offered` (seeds the key through `page.add_init_script`, opens `&preset=nope` so the ready text can be read, asserts no notice and a hidden button).
2. **Trailing punctuation inside an HTML `href`.** The sentence-punctuation rule was surfaced by a bare link; the same text inside `href="…"` ends at `"`, so the `.` before the quote must still be part of the id, and a `)` or `#` must still end it. Cases added to Task 9's pattern test: `'<a href="playground/index.html?exercise=0.1.4.a.">'` → `("0.1.4.a.", None)` next to the `#frag`, `)` and `'` cases; the failing-build test runs over all six marks `.,;:!?`, not only `.`.
3. **The textarea fallback (CDN blocked) writes and resets drafts too.** The CodeMirror `updateListener` is the obvious path; the fallback has its own `input` listener and its `set` must call `onChange` or Reset would leave the stale draft in storage. Test added to Task 6: `test_the_textarea_fallback_keeps_the_draft_too` (aborts `/npm/@codemirror/`, types, reloads, asserts the draft, clicks `#reset`, asserts the key is gone).

### Part B
1. **A lab's exercise heading is hand-written text, not generated from unit.yaml** (`### Exercise A · \`fn\`` in lab.py). G5 catches a dotted id there, but nothing checks that the letter matches the exercise the following cells are tagged with (a `### Exercise B` heading above cells tagged `exercise:0.1.1.a` passes). Reviewer: confirm C2's R16 work labels headings through `exercise_name`; if it does not, add to Task 14 a `text_violations` case that pairs each `Exercise <L>` heading in a markdown cell with the next `exercise:` tag.
2. **`course_ids` swallows a broken unit.yaml** (`except Exception: pass`, Task 12). The unit is still reported by `verify_unit` as `cannot load the unit`, but its exercise ids drop out of the pattern, so another unit's prose could name them unflagged in the same run. Reviewer: add to Task 14's seeded tests a case that breaks one ready unit's unit.yaml and asserts the other units are still scanned with the full id set, or make `course_ids` raise and let `verify` report it once.
3. **`section "X"` is the only cite form the scanner recognises.** Text that still says `the "X" section` or `lecture "X"` (without `section`) is neither flagged nor checked against headings; Task 13's 78 edits removed every such form the planner found, but a later author can reintroduce one. Reviewer: decide whether Task 14 should also flag `lecture "<X>"` / `"<X>" section` as a wrong form (one more regex in `text_violations`, one seeded test), or leave it to the writing rules.

---

### Task 1: Captions are plain text plus code spans (R6)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/build_labs.py` (:23 import, :36-49 `figure_html`, new `caption_parts`)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_build_labs.py` (:140-141 assertion; new tests at the end)
- Rebuild + commit: `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.1-meet-your-car/lab.ipynb`

**Interfaces:** `caption_parts(alt) -> (plain, caption_html)`; `figure_html` output shape as in Part interfaces; `BuildError` on bad alt text.

- [ ] **Step 1: Make the real-repo test expect plain captions (RED).** In `tests/zero2fsd/test_build_labs.py` replace the two lines at :140-141

  ```python
          number, lecture_alt = lecture[file]
          assert (alt, caption) == (lecture_alt, f"Lecture figure {number}. {lecture_alt}")
  ```
  with
  ```python
          number, plain = lecture[file][0], lecture[file][1].replace("`", "")  # a `code` span renders as <code>: no backticks reach the learner
          assert (alt, caption) == (plain, f"Lecture figure {number}. {plain}")
  ```
  Append these tests at the end of the file (after `test_a_lab_figure_with_its_own_alt_text_fails_the_build`):

  ```python
  def lecture_with_alt(mini_repo, alt):
      """The fixture lab showing two.png, whose lecture alt text for two.png is `alt`."""
      lab = mini_lab_showing(mini_repo, "![](figures/two.png)")
      lecture = lab.parent / "lecture.md"
      lecture.write_text(lecture.read_text(encoding="utf-8").replace("![Second]", f"![{alt}]"), encoding="utf-8")
      return lab


  def test_a_code_span_in_the_alt_is_code_in_the_caption_and_plain_text_in_the_img_alt(mini_repo):
      cell = nbformat.reads(build_labs.compile_lab(lecture_with_alt(mini_repo, "Drive `a<b` on `gentle` & stop")), as_version=4).cells[0]
      url = f"https://raw.githubusercontent.com/{REPO_SLUG}/{build_labs.FIGURE_REF}/course/0.1.1-mini/figures/two.png"
      assert f'<img src="{url}" alt="Drive a&lt;b on gentle &amp; stop">' in cell.source
      assert "<figcaption><em>Lecture figure 2. Drive <code>a&lt;b</code> on <code>gentle</code> &amp; stop</em></figcaption>" in cell.source
      assert "`" not in cell.source
      assert shown_images(cell.source) == [(url, "Drive a<b on gentle & stop", "Lecture figure 2. Drive a<b on gentle & stop")]


  def test_markdown_characters_inside_a_code_span_are_allowed(mini_repo):
      cell = nbformat.reads(build_labs.compile_lab(lecture_with_alt(mini_repo, "mask `a[0] * 2` costs `$x`")), as_version=4).cells[0]
      assert "Lecture figure 2. mask <code>a[0] * 2</code> costs <code>$x</code></em>" in cell.source


  @pytest.mark.parametrize("alt", ["*bold* roads", "a [link](x) here", "cost $5 each", "`open span", "two `a` and `b` and `c"],
                           ids=["emphasis", "link", "maths", "unbalanced", "unbalanced-after-spans"])
  def test_markdown_outside_a_code_span_or_an_unbalanced_backtick_fails_the_build(mini_repo, alt):
      with pytest.raises(ValueError, match="alt text"):
          build_labs.compile_lab(lecture_with_alt(mini_repo, alt))
  ```

- [ ] **Step 2: Run the file, expect RED.**
  `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_build_labs.py 2>&1 | tail -15`
  Expected: failures include `test_every_lecture_figure_is_in_the_lab_with_its_caption_and_a_url_colab_can_load[0.1.1-meet-your-car]` (the committed caption still holds backticks), the two code-span tests (`<code>` absent) and all five `test_markdown_outside...` cases (`DID NOT RAISE`). Nothing else fails.

- [ ] **Step 3: Implement.** In `scripts/build_labs.py`:
  - :23 `from _course import FENCE, REPO, REPO_SLUG, parse_lab` → `from _course import FENCE, REPO, REPO_SLUG, BuildError, parse_lab`
  - after :28 (`LECTURE_FIGURE = ...`) add
    ```python
    MARKDOWN = re.compile(r"[*\[$]")  # markdown or maths that alt text may not hold outside a code span
    ```
  - after `lecture_figures` (:31-33) add
    ```python
    def caption_parts(alt: str) -> tuple[str, str]:
        """(plain text for the <img> alt, caption HTML) of a lecture alt text: `code` spans become <code>, all HTML-escaped.

        BuildError on an unbalanced backtick or on `*`, `[`, `$` outside a code span: a caption is plain text plus code,
        because Colab shows the lab's markdown verbatim (R6).
        """
        parts = alt.split("`")
        if len(parts) % 2 == 0:
            raise BuildError(f"unbalanced backticks in figure alt text: {alt!r}")
        if any(MARKDOWN.search(text) for text in parts[::2]):
            raise BuildError(f"figure alt text may hold only plain text and `code` spans (no *, [ or $ outside one): {alt!r}")
        escaped = [html.escape(part) for part in parts]
        return "".join(escaped), "".join(f"<code>{part}</code>" if i % 2 else part for i, part in enumerate(escaped))
    ```
  - in `figure_html` replace :46-49
    ```python
        number, alt = figures[path]
        alt = html.escape(alt)
        src = f"https://raw.githubusercontent.com/{REPO_SLUG}/{FIGURE_REF}/course/{unit.name}/{path}"
        return f'<figure>\n<img src="{src}" alt="{alt}">\n<figcaption><em>Lecture figure {number}. {alt}</em></figcaption>\n</figure>'
    ```
    with
    ```python
        number, alt = figures[path]
        plain, caption = caption_parts(alt)
        src = f"https://raw.githubusercontent.com/{REPO_SLUG}/{FIGURE_REF}/course/{unit.name}/{path}"
        return f'<figure>\n<img src="{src}" alt="{plain}">\n<figcaption><em>Lecture figure {number}. {caption}</em></figcaption>\n</figure>'
    ```
  - module docstring :9: after "the lecture's own alt text as a numbered caption." add the sentence `A \`code\` span in that alt text becomes <code>; any other markdown in it (*, [, $ outside a code span) fails the build.` (write the backticks literally, without the backslashes).

- [ ] **Step 4: Rebuild the notebooks and verify.**
  `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_labs.py && git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 status --short course && grep -c '<code>straight</code>' course/0.1.1-meet-your-car/lab.ipynb`
  Expected: four `built course/0.1.x-.../lab.ipynb` lines; `git status` shows exactly ` M course/0.1.1-meet-your-car/lab.ipynb` (the only lecture alt with backticks is 0.1.1 `test_roads`); grep prints `1`.
  Then the test file:
  `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_build_labs.py 2>&1 | tail -3`
  Expected: last line `NN passed in ...` with no `failed`.

- [ ] **Step 5: Commit.**
  `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add scripts/build_labs.py tests/zero2fsd/test_build_labs.py course/0.1.1-meet-your-car/lab.ipynb && git commit -q -F - <<'MSG'
feat(labs): caption code spans render as <code>; other markdown in alt text fails the build

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 log --oneline -1`

### Task 2: Lab figures follow lecture order (R5)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/build_labs.py` (`compile_lab`; new `check_lecture_order` right above it)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_build_labs.py` (new tests at the end)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.2-images-are-arrays/lab.py` (:93 and :100 swap)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.4-closing-the-loop/lab.py` (:159-161 removed; one line added after :207)
- Rebuild + commit: `course/0.1.2-images-are-arrays/lab.ipynb`, `course/0.1.4-closing-the-loop/lab.ipynb`

**Interfaces:** `check_lecture_order(unit: Path, shown: list[str]) -> None` raises `BuildError` unless the lecture-figure numbers of `shown` (lab order) strictly increase; `compile_lab` collects the `figures/...` paths of every markdown cell's `![](figures/x.png)` lines and calls it once after the cell loop. Only the figure code of `compile_lab` changes (Part B owns the text-cell code in this file). The lab.py edits move two figure lines only; no prose changes.

- [ ] **Step 1: Tests (RED).** Append at the end of `tests/zero2fsd/test_build_labs.py` (after the Task 1 tests):

  ```python
  @pytest.mark.parametrize(("first", "second"), [("two", "one"), ("one", "one")], ids=["reversed", "repeated"])
  def test_lab_figures_out_of_lecture_order_fail_the_build(mini_repo, first, second):
      lab = mini_lab_showing(mini_repo, f"![](figures/{first}.png)")
      lab.write_text(lab.read_text(encoding="utf-8") + f"\n# %% [markdown]\n# ![](figures/{second}.png)\n", encoding="utf-8")
      with pytest.raises(ValueError, match="out of lecture order"):
          build_labs.compile_lab(lab)


  def test_lab_figures_in_lecture_order_across_cells_build(mini_repo):
      lab = mini_lab_showing(mini_repo, "![](figures/one.png)")
      lab.write_text(lab.read_text(encoding="utf-8") + "\n# %%\nx = 1\n\n# %% [markdown]\n# ![](figures/two.png)\n", encoding="utf-8")
      cells = nbformat.reads(build_labs.compile_lab(lab), as_version=4).cells
      shown = shown_images(cells[0].source) + shown_images(cells[2].source)
      assert [url.rsplit("/", 1)[1] for url, _, _ in shown] == ["one.png", "two.png"]
  ```

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_build_labs.py -q -k "lecture_order" 2>&1 | tail -5`
  Expected: `2 failed, 1 passed` — the two `out_of_lecture_order` cases fail with `DID NOT RAISE`; the in-order case already passes.

- [ ] **Step 2: Implement.** In `scripts/build_labs.py`, insert this function directly above `def compile_lab(` (`BuildError` is already imported from `_course` by Task 1):

  ```python
  def check_lecture_order(unit: Path, shown: list[str]) -> None:
      """BuildError unless the lecture figures a lab shows, in lab order, are a strict subsequence of the lecture's figure order."""
      if not shown:
          return
      figures = lecture_figures((unit / "lecture.md").read_text(encoding="utf-8"))
      numbers = [figures[path][0] for path in shown]  # every path is a lecture figure: figure_html already checked
      if any(a >= b for a, b in zip(numbers, numbers[1:])):
          raise BuildError(f"{unit.name}: the lab shows lecture figures out of lecture order: {shown}")
  ```

  In `compile_lab`, replace the loop

  ```python
      cells = []
      for index, cell in enumerate(parse_lab(lab.read_text(encoding="utf-8"))):
          metadata = {"tags": cell.tags} if cell.tags else {}
          source = LAB_FIGURE.sub(lambda ref: figure_html(lab.parent, ref), cell.source) if cell.kind == "markdown" else cell.source
          cells.append(make[cell.kind](source, id=f"c{index:03d}", metadata=metadata))
  ```
  with
  ```python
      cells, shown = [], []
      for index, cell in enumerate(parse_lab(lab.read_text(encoding="utf-8"))):
          metadata = {"tags": cell.tags} if cell.tags else {}
          source = cell.source
          if cell.kind == "markdown":
              shown += [ref["path"] for ref in LAB_FIGURE.finditer(source)]
              source = LAB_FIGURE.sub(lambda ref: figure_html(lab.parent, ref), source)
          cells.append(make[cell.kind](source, id=f"c{index:03d}", metadata=metadata))
      check_lecture_order(lab.parent, shown)
  ```
  Add to the module docstring (top of the file), after the sentence Task 1 added: `A lab shows its lecture's figures in lecture order (a build error otherwise).`

  Run the same pytest command as Step 1. Expected: `3 passed`. Then the whole file: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_build_labs.py -q 2>&1 | tail -3` — expected: all passed, 0 failed (no committed notebook changed yet, so the freshness test stays green). (`test_main_builds_only_the_named_unit_dirs` builds a unit with no lecture.md and no figures: the early return keeps it green.)

- [ ] **Step 3: The gate is RED on the real repo.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_labs.py --check 2>&1 | tail -2`
  Expected: the last line starts with `_course.BuildError: 0.1.2-images-are-arrays: the lab shows lecture figures out of lecture order:` (0.1.2's lecture order is frame_grid, noisy_patch, colour_distance, exact_vs_threshold; its lab shows exact_vs_threshold before colour_distance. 0.1.4 is also wrong — steady_state shown before gain_sweep — but the build stops at the first unit).

- [ ] **Step 4: Reorder the two labs.** In `course/0.1.2-images-are-arrays/lab.py`, line 93 is exactly `# ![](figures/exact_vs_threshold.png)` and line 100 is exactly `# ![](figures/colour_distance.png)` (same markdown cell). Swap them: line 93 becomes `# ![](figures/colour_distance.png)`, line 100 becomes `# ![](figures/exact_vs_threshold.png)`. Nothing else in the file changes.
  In `course/0.1.4-closing-the-loop/lab.py`, delete the three lines 159-161, which are exactly:
  ```
  # %% [markdown]
  # ![](figures/steady_state.png)

  ```
  (the markdown header, the figure line and the blank line after it; line 158 is blank and line 162 is the next `# %% [markdown]`, so one blank line remains between the `practice(...)` call and the next cell). Then, in the section-4 cell, right after the line that is exactly `# ![](figures/gain_sweep.png)` (line 204 after the deletion) and before the `#` line that follows it, insert the two lines:
  ```
  #
  # ![](figures/steady_state.png)
  ```
  so the cell reads `... gain_sweep.png)` / `#` / `# ![](figures/steady_state.png)` / `#` / `# ## 5. Graded exercise 0.1.4.b ...`.
  Verify: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && grep -n '^# !\[\](figures' course/0.1.2-images-are-arrays/lab.py course/0.1.4-closing-the-loop/lab.py`
  Expected (8 lines): 0.1.2 → `66:...frame_grid.png`, `68:...noisy_patch.png`, `93:...colour_distance.png`, `100:...exact_vs_threshold.png`; 0.1.4 → `81:...open_vs_closed.png`, `204:...gain_sweep.png`, `206:...steady_state.png`, `287:...curvy_box.png`.

- [ ] **Step 5: Rebuild the notebooks and prove the gate is GREEN.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_labs.py && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_labs.py --check; echo "check exit $?"; git status --short`
  Expected: `check exit 0`; status shows exactly `M scripts/build_labs.py`, `M tests/zero2fsd/test_build_labs.py`, `M course/0.1.2-images-are-arrays/lab.py`, `M course/0.1.2-images-are-arrays/lab.ipynb`, `M course/0.1.4-closing-the-loop/lab.py`, `M course/0.1.4-closing-the-loop/lab.ipynb` (0.1.1 and 0.1.3 notebooks unchanged). Then the freshness and caption tests: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_build_labs.py tests/zero2fsd/test_bundle.py -q 2>&1 | tail -3` — expected: all passed, 0 failed.

- [ ] **Step 6: Commit.**
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add scripts/build_labs.py tests/zero2fsd/test_build_labs.py course/0.1.2-images-are-arrays/lab.py course/0.1.2-images-are-arrays/lab.ipynb course/0.1.4-closing-the-loop/lab.py course/0.1.4-closing-the-loop/lab.ipynb && git commit -q -F - <<'MSG'
  feat(labs): lab figures must follow lecture order; reorder 0.1.2 and 0.1.4

  compile_lab collects the figures each lab shows and check_lecture_order fails
  the build unless they are a strict subsequence of the lecture's figure order.
  0.1.2 showed exact_vs_threshold before colour_distance; 0.1.4 showed
  steady_state before gain_sweep. Notebooks rebuilt.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 log --oneline -1
  ```

### Task 3: Palette shades, the dark figure style, and the G2/G3 gates (R3)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/zero2fsd/palette.py` (add `shades`; pure Python, no matplotlib)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_palette.py` (import + one test)
- Create: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/figstyle.py`
- Create: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_figure_colours.py` (the G2 scanner with seeded cases; G3)
- Rebuild + commit: `course/dist/playground/bundle.zip` (bundle.zip packs every `zero2fsd/*.py`, so a palette change changes it)

**Interfaces:** `zero2fsd.palette.shades(hex_colour: str, n: int) -> list[str]`; `figstyle.RC: dict`; in the test module, `colour_literals(source: str) -> list[tuple[int, str]]` (sorted `(line, literal)` pairs), `contrast(a: str, b: str) -> float`, `unreadable(hues: dict[str, str], background: str, ratio: float = 3.0) -> dict[str, float]`, `SCANNED` (every `course/*/figures.py` + `scripts/figstyle.py`). The real-repo G2 gate (`test_no_figure_module_holds_a_string_colour_literal`) is added in Task 5, once the figures modules are clean; this task proves the scanner on seeded code. No file under `scripts/build_labs.py` changes.

- [ ] **Step 1: Palette test (RED).** In `tests/zero2fsd/test_palette.py` change the import line `from zero2fsd.palette import CHROME, QUANTITY, SCENE, palette_dict` to `from zero2fsd.palette import CHROME, QUANTITY, SCENE, palette_dict, shades` and append at the end of the file:

  ```python
  @pytest.mark.parametrize("n", [1, 2, 4])
  def test_shades_start_at_the_hue_and_step_evenly_toward_the_text_colour(n):
      family = shades(QUANTITY["e_off"], n)
      assert family[0] == QUANTITY["e_off"] and len(family) == n == len(set(family))
      assert all(HEX.fullmatch(shade) for shade in family)
      text = rgb(CHROME["text"])
      distances = [sum(abs(a - b) for a, b in zip(rgb(shade), text)) for shade in family]
      assert distances == sorted(distances, reverse=True)  # every later shade is nearer the text colour
  ```
  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_palette.py -q 2>&1 | tail -3`
  Expected: collection error `ImportError: cannot import name 'shades'`.

- [ ] **Step 2: `shades` in the palette.** Replace the whole of `zero2fsd/palette.py` with:

  ```python
  # zero2fsd/palette.py
  """The course's colours in one place: `palette_dict()` is what the browser view reads as `palette.json`.

  Paint colours come from the camera's own `PALETTE`, so the top-down view paints the road in the colours
  the camera sees.  The quantity colours name the four things the lectures plot; the chrome colours are the
  site's learnfsd page colours.  `shades(hue, n)` is the one way to draw a quantity `n` times on one figure.
  """
  from __future__ import annotations

  from .sim.camera import GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW

  MIX = 0.65  # how far the lightest shade of a family goes toward the text colour: still clearly its hue


  def _hex(channels) -> str:
      return "#{:02X}{:02X}{:02X}".format(*(int(c) for c in channels))


  def _rgb(hex_colour: str) -> tuple[int, int, int]:
      return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


  QUANTITY = {"e_off": "#58C4DD", "psi": "#D147BD", "delta": "#83C167", "perception": "#FC6255"}
  SCENE = {name: _hex(PALETTE[index]) for name, index in (("sky", SKY), ("grass", GRASS), ("road", ROAD), ("yellow", YELLOW), ("white", WHITE))}
  CHROME = {"background": "#090d16", "panel": "#0d1117", "border": "#30363d", "text": "#c9d1d9", "link": "#58a6ff"}


  def shades(hex_colour: str, n: int) -> list[str]:
      """`n` shades of one hue for a quantity drawn `n` times: the hue itself, then even steps toward CHROME["text"]."""
      base, text = _rgb(hex_colour), _rgb(CHROME["text"])
      return [hex_colour] + [_hex(round(b + (t - b) * MIX * i / (n - 1)) for b, t in zip(base, text)) for i in range(1, n)]


  def palette_dict() -> dict[str, dict[str, str]]:
      return {"quantity": QUANTITY, "scene": SCENE, "chrome": CHROME}
  ```
  (`QUANTITY`, `SCENE`, `CHROME` values are unchanged, so `palette.json` stays byte-identical.) Run the Step 1 command. Expected: `10 passed` (7 existing + 3). Spot-check: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -c "from zero2fsd.palette import QUANTITY, shades; print(shades(QUANTITY['e_off'], 4))"` → `['#58C4DD', '#70C7DC', '#89CADB', '#A1CCDA']`.

- [ ] **Step 3: The style helper.** Create `scripts/figstyle.py`:

  ```python
  # scripts/figstyle.py
  """The one figure style: dark page colours from the palette, applied by make_figures.py with `matplotlib.rc_context(RC)`.

  A figures.py never sets a background, text colour or colour cycle; it only picks palette entries for what it draws.
  """
  from cycler import cycler

  from zero2fsd.palette import CHROME, QUANTITY

  RC = {
      "figure.facecolor": CHROME["background"],
      "axes.facecolor": CHROME["panel"],
      "axes.edgecolor": CHROME["border"],
      "axes.labelcolor": CHROME["text"],
      "xtick.color": CHROME["text"],
      "ytick.color": CHROME["text"],
      "text.color": CHROME["text"],
      "grid.color": CHROME["border"],
      "legend.edgecolor": CHROME["border"],
      "patch.edgecolor": CHROME["text"],  # an un-coloured annotate arrow is a patch: light on dark, not black
      "patch.facecolor": CHROME["text"],
      "axes.prop_cycle": cycler(color=list(QUANTITY.values())),
  }
  ```

- [ ] **Step 4: The G2 scanner and G3, proved on seeded code.** Create `tests/zero2fsd/test_figure_colours.py`:

  ```python
  # tests/zero2fsd/test_figure_colours.py
  """Gate G2: no figures.py (nor scripts/figstyle.py) holds a string colour literal; colours come from zero2fsd.palette.
  Gate G3: every quantity hue, and every shade of it, reads on the dark figure background (WCAG contrast >= 3:1).

  What counts as a colour literal is derived from matplotlib itself, never from a list of names: the kwarg aliases of
  every Artist property whose name contains "color", any kwarg or dict key containing "color"/"colour", any "#hex"
  string anywhere, and any positional string that matplotlib's own format parser reads as naming a colour ("ro", "C3").
  """
  import ast
  import re
  from pathlib import Path

  import pytest
  from matplotlib.artist import Artist, ArtistInspector
  from matplotlib.axes._base import _process_plot_format

  import figstyle
  from zero2fsd.palette import CHROME, QUANTITY, shades

  REPO = Path(__file__).resolve().parents[2]
  SCANNED = sorted(REPO.glob("course/*/figures.py")) + [REPO / "scripts" / "figstyle.py"]
  HEX = re.compile(r"#[0-9a-fA-F]{3,8}")


  def _subclasses(cls):
      for sub in cls.__subclasses__():
          yield sub
          yield from _subclasses(sub)


  def colour_aliases() -> set[str]:
      """Every kwarg alias matplotlib accepts for a property whose name contains "color" (c, ec, fc, mec, mfc, ...)."""
      aliases = set()
      for cls in [Artist, *_subclasses(Artist)]:
          for prop, names in ArtistInspector(cls).aliasd.items():
              if "color" in prop:
                  aliases |= set(names)
      return aliases


  ALIASES = colour_aliases()


  def is_colour_name(name: str) -> bool:
      lowered = name.lower()
      return name in ALIASES or "color" in lowered or "colour" in lowered


  def _str_constants(node):
      """Every string constant under `node`, not descending into subscript keys (QUANTITY["e_off"] names a palette entry)."""
      if isinstance(node, ast.Constant):
          if isinstance(node.value, str):
              yield node
          return
      for name, child in ast.iter_fields(node):
          if isinstance(node, ast.Subscript) and name == "slice":
              continue
          for item in child if isinstance(child, list) else [child]:
              if isinstance(item, ast.AST):
                  yield from _str_constants(item)


  def _colour_values(value):
      """(line, text) for the literals a colour-valued expression holds: its strings, or an all-number tuple/list."""
      if isinstance(value, (ast.Tuple, ast.List)) and value.elts and all(
          isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in value.elts
      ):
          return [(value.lineno, ast.unparse(value))]
      return [(c.lineno, c.value) for c in _str_constants(value)]


  def _fmt_colour(text: str):
      try:
          return _process_plot_format(text)[2]
      except ValueError:
          return None


  def colour_literals(source: str) -> list[tuple[int, str]]:
      """Every (line, literal) in `source` that hands matplotlib a colour as a literal rather than a palette value."""
      found = []
      for node in ast.walk(ast.parse(source)):
          if isinstance(node, ast.Constant) and isinstance(node.value, str) and HEX.search(node.value):
              found.append((node.lineno, node.value))
          elif isinstance(node, ast.keyword) and node.arg and is_colour_name(node.arg):
              found += _colour_values(node.value)
          elif isinstance(node, ast.Dict):
              for key, value in zip(node.keys, node.values):
                  if isinstance(key, ast.Constant) and isinstance(key.value, str) and is_colour_name(key.value):
                      found += _colour_values(value)
          elif isinstance(node, ast.Call):
              for arg in node.args:
                  if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _fmt_colour(arg.value):
                      found.append((arg.lineno, arg.value))
      return sorted(set(found))


  VIOLATIONS = {
      "hex anywhere": 'EDGE = "#666666"\n',
      "colour keyword": 'ax.plot(x, y, color="tab:red")\n',
      "alias keyword": 'ax.plot(x, y, "o", mec="black")\n',
      "keyword containing colour": 'ax.hist(d, edgecolor="0.2")\n',
      "plural keyword": 'ax.eventplot(d, colors=["red"])\n',
      "conditional under a colour keyword": 'ax.text(0, 0, s, color="white" if dark else "black")\n',
      "f-string under a colour keyword": 'ax.plot(t, y, color=f"C{i}")\n',
      "dict key": 'ax.annotate("", xy=p, arrowprops={"arrowstyle": "->", "color": "C3"})\n',
      "format string with a colour": 'ax.plot(x, y, "ro")\n',
      "bare colour string": 'ax.plot(x, y, "C3")\n',
      "rgb tuple": 'ax.plot(x, y, color=(1.0, 0.0, 0.0))\n',
  }
  ALLOWED = {
      "palette subscript": 'ax.plot(x, y, color=QUANTITY["e_off"])\n',
      "palette shades": 'ax.plot(x, y, color=shades(QUANTITY["e_off"], 3)[i])\n',
      "marker and linestyle formats": 'ax.plot(x, y, "o")\nax.plot(x, y, "--", lw=2)\n',
      "prose": 'NAMES = ("sky", "grass", "road", "yellow", "white")\nax.set_title("the yellow line")\n',
      "label text": 'ax.hist(d, label="true white pixels")\n',
  }


  def test_the_colour_aliases_come_from_matplotlib_and_include_the_short_ones():
      assert {"c", "ec", "fc", "mec", "mfc"} <= ALIASES


  @pytest.mark.parametrize("source", VIOLATIONS.values(), ids=VIOLATIONS)
  def test_a_seeded_colour_literal_is_flagged_with_its_line(source):
      hits = colour_literals(source)
      assert hits and all(line == 1 for line, _ in hits), hits


  @pytest.mark.parametrize("source", ALLOWED.values(), ids=ALLOWED)
  def test_palette_lookups_formats_and_prose_are_not_flagged(source):
      assert colour_literals(source) == []


  def test_figstyle_holds_only_palette_colours_and_sets_the_page_colours():
      assert figstyle.RC["figure.facecolor"] == CHROME["background"] and figstyle.RC["axes.facecolor"] == CHROME["panel"]
      assert [c["color"] for c in figstyle.RC["axes.prop_cycle"]] == list(QUANTITY.values())


  # --- G3: contrast, computed from the palette values ---


  def _rgb(hex_colour: str) -> tuple[int, int, int]:
      return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


  def luminance(hex_colour: str) -> float:
      """WCAG 2 relative luminance."""
      def channel(v):
          v /= 255
          return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
      r, g, b = (channel(v) for v in _rgb(hex_colour))
      return 0.2126 * r + 0.7152 * g + 0.0722 * b


  def contrast(a: str, b: str) -> float:
      """WCAG 2 contrast ratio, 1.0 (same colour) to 21.0 (black on white)."""
      lighter, darker = sorted((luminance(a), luminance(b)), reverse=True)
      return (lighter + 0.05) / (darker + 0.05)


  def unreadable(hues: dict[str, str], background: str, ratio: float = 3.0) -> dict[str, float]:
      """{name: contrast} for every hue that falls below `ratio` on `background` (empty when all read)."""
      return {name: round(contrast(hue, background), 2) for name, hue in hues.items() if contrast(hue, background) < ratio}


  def test_contrast_is_the_wcag_ratio():
      assert round(contrast("#000000", "#FFFFFF"), 1) == 21.0 and contrast("#58C4DD", "#58C4DD") == 1.0
      assert round(contrast("#777777", "#FFFFFF"), 2) == 4.48


  def test_a_seeded_dark_hue_is_reported_with_its_ratio():
      assert unreadable({"psi": "#1a1f2b", "e_off": QUANTITY["e_off"]}, CHROME["background"]) == {"psi": 1.18}


  @pytest.mark.parametrize("surface", ["background", "panel"])
  def test_every_quantity_hue_and_every_shade_reads_on_the_figure_surfaces(surface):
      families = {f"{name}[{i}]": shade for name, hue in QUANTITY.items() for i, shade in enumerate(shades(hue, 4))}
      assert unreadable({**QUANTITY, **families}, CHROME[surface]) == {}


  def test_text_reads_on_the_panel_at_the_body_text_ratio():
      assert unreadable({"text": CHROME["text"]}, CHROME["panel"], ratio=4.5) == {}
  ```
  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_figure_colours.py -q 2>&1 | tail -3`
  Expected: `23 passed` (1 alias + 11 seeded violations + 5 allowed + 1 figstyle + 5 contrast). For the record, the measured ratios the parametrized G3 test is checking: on the background e_off 9.58, psi 4.95, delta 9.06, perception 6.49; on the panel 9.33 / 4.82 / 8.82 / 6.32; text on panel 12.26; the lowest of the sixteen `shades(hue, 4)` entries on the panel is 4.82.
  Prove the scanner on the real tree before anything is restyled: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1:/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts:/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -c "import pathlib, test_figure_colours as t; [print(p.parent.name, len(t.colour_literals(p.read_text()))) for p in t.SCANNED]"`
  Expected: `0.1.1-meet-your-car 14`, `0.1.2-images-are-arrays 5`, `0.1.3-your-first-perception-box 35`, `0.1.4-closing-the-loop 15`, `scripts 0`.

- [ ] **Step 5: Rebuild the bundle and run the freshness tests.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py; echo "bundle exit $?"; git status --short`
  Expected: `bundle exit 0`; status lists `M zero2fsd/palette.py`, `M tests/zero2fsd/test_palette.py`, `?? scripts/figstyle.py`, `?? tests/zero2fsd/test_figure_colours.py`, `M course/dist/playground/bundle.zip` and possibly `M course/dist/playground/manifest.json` (if it carries the bundle's size or hash); nothing under `course/dist/figures` or `course/dist/course.json` changes (no figure changed). Then: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/playground/test_build.py tests/zero2fsd/test_palette.py tests/zero2fsd/test_bundle.py -q 2>&1 | tail -3` — expected: all passed, 0 failed (this covers `test_the_committed_playground_is_what_the_source_builds` and the browser import-surface test that imports `zero2fsd.palette` with matplotlib absent).

- [ ] **Step 6: Commit** (add `course/dist/playground/manifest.json` too if Step 5 listed it as modified):
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add zero2fsd/palette.py tests/zero2fsd/test_palette.py scripts/figstyle.py tests/zero2fsd/test_figure_colours.py course/dist/playground/bundle.zip && git commit -q -F - <<'MSG'
  feat(figures): palette shades, dark figure style helper, G2 colour-literal and G3 contrast gates

  palette.shades(hue, n) is the one way to draw a quantity n times. scripts/figstyle.py
  holds the rcParams for the dark figure style. tests/zero2fsd/test_figure_colours.py
  flags string colour literals in figure modules (aliases and format strings read from
  matplotlib, not listed) and computes WCAG contrast for every hue and shade on the
  figure surfaces. Bundle rebuilt (it packs the palette).

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 log --oneline -1
  ```

### Task 4: Dark style through make_figures; palette colours in 0.1.1 and 0.1.2 (R3)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/make_figures.py` (`make_unit_figures`; two imports)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_build_labs.py` (two imports; one test after `test_figures_are_byte_identical_across_runs`)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_bundle.py` (import; one parametrized test at the end)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.1-meet-your-car/figures.py`, `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.2-images-are-arrays/figures.py`
- Regenerate + commit: all 16 PNGs under `course/*/figures/` (the dark background reaches every unit through `rc_context`, 0.1.3 and 0.1.4 included; their foreground colours are Task 5's) and `course/dist/figures/**` (16 copies)

**Interfaces:** `make_figures.make_unit_figures(unit_dir)` runs `module.make(out_dir=...)` inside `matplotlib.rc_context(figstyle.RC)`; nothing else about it changes. New test `tests/zero2fsd/test_bundle.py::test_the_committed_dist_figures_are_byte_copies_of_the_units_figures` (freshness gate for `course/dist/figures`, like the notebook and playground freshness tests; Part C runs `scripts/build_course_bundle.py` after `scripts/make_figures.py`). No build_labs.py change.

- [ ] **Step 1: Wiring test (RED).** In `tests/zero2fsd/test_build_labs.py` add `from matplotlib import image` after `import nbformat` (it is a third-party import: keep it in that block, before `import pytest`) and `from zero2fsd.palette import CHROME` after `from _course import REPO, REPO_SLUG, parse_lab`. Insert after `test_figures_are_byte_identical_across_runs`:

  ```python
  def test_figures_are_drawn_in_the_dark_figure_style(mini_repo):
      assert make_figures.main(["--root", str(mini_repo)]) == 0
      corner = image.imread(mini_repo / "course" / "0.1.1-mini" / "figures" / "curve.png")[0, 0, :3]
      assert tuple(round(float(v) * 255) for v in corner) == tuple(int(CHROME["background"][i:i + 2], 16) for i in (1, 3, 5))
  ```
  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_build_labs.py -q -k "dark_figure_style" 2>&1 | tail -3`
  Expected: `1 failed` — `(255, 255, 255) == (9, 13, 22)` (the fixture figure is still white).

- [ ] **Step 2: Wire the style in.** In `scripts/make_figures.py`:
  - after the line `os.environ.setdefault("MPLBACKEND", "Agg")  # before any figures.py imports matplotlib` and its blank line, add `import matplotlib  # noqa: E402` followed by a blank line;
  - after `from _course import REPO  # noqa: E402` add `from figstyle import RC  # noqa: E402`;
  - replace the line `    module.make(out_dir=out_dir)` in `make_unit_figures` with
    ```python
        with matplotlib.rc_context(RC):  # the one figure style: a figures.py never sets a background or text colour
            module.make(out_dir=out_dir)
    ```
  Run the Step 1 command. Expected: `1 passed`. Then the whole file: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_build_labs.py -q 2>&1 | tail -3` — expected: all passed (`test_figures_are_byte_identical_across_runs` still holds).

- [ ] **Step 3: Dist-figures freshness test (RED later in this task).** In `tests/zero2fsd/test_bundle.py` change `from _course import REPO, BuildError, heading_slug, unit_slug` to `from _course import REPO, BuildError, heading_slug, syllabus_units, unit_slug` and append at the end of the file:

  ```python
  READY_WITH_FIGURES = [u for u in syllabus_units(REPO, "ready") if (REPO / "course" / u["folder"] / "figures").is_dir()]


  @pytest.mark.parametrize("unit", READY_WITH_FIGURES, ids=[u["folder"] for u in READY_WITH_FIGURES])
  def test_the_committed_dist_figures_are_byte_copies_of_the_units_figures(unit):
      """Part C: after scripts/make_figures.py, run scripts/build_course_bundle.py and commit course/dist."""
      pngs = sorted((REPO / "course" / unit["folder"] / "figures").glob("*.png"))
      assert pngs
      for png in pngs:
          copy = REPO / "course" / "dist" / "figures" / unit_slug(unit["id"]) / png.name
          assert copy.is_file() and copy.read_bytes() == png.read_bytes(), copy
  ```
  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_bundle.py -q -k "dist_figures" 2>&1 | tail -3`
  Expected: `4 passed` (the committed tree is in sync; ids are the four unit folders). It goes RED in Step 6 and GREEN in Step 7.

- [ ] **Step 4: Restyle `course/0.1.1-meet-your-car/figures.py`.** Line numbers are the file's before any edit; match on the exact old text. Add the import `from zero2fsd.palette import CHROME, QUANTITY, SCENE, shades` on its own line between `from zero2fsd.car import Car, provided, run` (:7) and `from zero2fsd.score import OFF_LANE_LAT` (:8). Then:
  - :31 `color="C3"` → `color=QUANTITY["perception"]` (in `top.plot(*road.offset(VIEW_S, -LANE_WIDTH / 2), "o", color="C3", label="camera position")`)
  - :49 `ax.plot(x, y, "o", color="C3", zorder=3)` → `ax.plot(x, y, "o", color=QUANTITY["perception"], zorder=3)`
  - :51 `"color": "C3"` → `"color": QUANTITY["perception"]` (inside `arrowprops={"arrowstyle": "-|>", ...}`)
  - :53 `"s", color="0.2", zorder=3)` → `"s", color=CHROME["text"], zorder=3)`
  - :77 `marker="o", color="C3", linestyle=""` → `marker="o", color=QUANTITY["perception"], linestyle=""`
  - :78 `marker="s", color="0.2", linestyle=""` → `marker="s", color=CHROME["text"], linestyle=""`
  - :88 `color="#c9a800"` → `color=SCENE["yellow"]`
  - :89 `ax.axhline(-half, color="0.6", linewidth=3)` → `ax.axhline(-half, color=SCENE["white"], linewidth=3)`
  - :90 `ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)` → `ax.axhline(0.0, color=CHROME["text"], linestyle=":", linewidth=1)`
  - :93 `ax.text(0.2, 0.08, "lane centre, e = 0", fontsize=9, color="0.3")` → `ax.text(0.2, 0.08, "lane centre, e = 0", fontsize=9)` (text colour comes from the style)
  - :94 `for x, e, color in ((4.0, 0.5, "C0"), (8.5, -0.3, "C1")):` → `for (x, e), color in zip(((4.0, 0.5), (8.5, -0.3)), shades(QUANTITY["e_off"], 2)):` (two drawings of the one quantity e: two shades of its hue)
  - :116 `ax.plot(tel.progress_m, tel.lat, color="0.3", linewidth=1)` → `ax.plot(tel.progress_m, tel.lat, color=QUANTITY["e_off"], linewidth=1)`
  - :117 `ax.axhline(signed, color="C3", linewidth=2, ...` → `color=QUANTITY["perception"]`
  - :118 `ax.axhline(mean_abs, color="C0", linewidth=2, linestyle="--", ...` → `color=CHROME["text"]`
  - :120 `ax.axhline(edge, color="0.6", linestyle=":", linewidth=1)` → `ax.axhline(edge, color=CHROME["text"], linestyle=":", linewidth=1)`

- [ ] **Step 5: Restyle `course/0.1.2-images-are-arrays/figures.py`.** Add `from matplotlib.colors import ListedColormap` on its own line before `from matplotlib.figure import Figure` (:3); add `from zero2fsd.palette import CHROME, QUANTITY, SCENE` between `from zero2fsd.car.provided import PAINT_DISTANCE` (:6) and `from zero2fsd.sim import BicycleState, Camera, make_scenario` (:7); after `NAMES = ("sky", "grass", "road", "yellow", "white")` (:14) add the line `MASK = ListedColormap([CHROME["panel"], CHROME["text"]])  # a boolean mask: False in the panel colour, True in the text colour`. Then (`PALETTE` stays imported: :27, :74 and :84 still use it):
  - :40 `color="white" if patch[i, j].mean() < 150 else "black")` → `color=CHROME["text"] if patch[i, j].mean() < 150 else CHROME["background"])`
  - :51 `fill=False, color="red", lw=2))` → `fill=False, color=QUANTITY["perception"], lw=2))`
  - :72 `color=PALETTE[k] / 255, edgecolor="0.2", linewidth=0.4,` → `color=SCENE[name], edgecolor=CHROME["background"], linewidth=0.4,`
  - :73 `ax.axvline(PAINT_DISTANCE, color="red", linewidth=2,` → `ax.axvline(PAINT_DISTANCE, color=QUANTITY["perception"], linewidth=2,`
  - :93 `cmap="gray"` → `cmap=MASK` (the two 2-D panels, `exact` and `near`, are boolean masks)

  Scanner evidence, before Step 4/5 the counts were 14 and 5 (Task 3 Step 4); now: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1:/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts:/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -c "import pathlib, test_figure_colours as t; [print(p.parent.name, t.colour_literals(p.read_text())) for p in t.SCANNED[:2]]"`
  Expected: `0.1.1-meet-your-car []` and `0.1.2-images-are-arrays []`. If a line prints, it names the literal and its line: fix that line with a palette value, never by editing the scanner.

- [ ] **Step 6: Regenerate the figures (about 50 s).** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && time PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/make_figures.py; echo "figures exit $?"; git status --short course | grep -c 'figures/.*\.png'`
  Expected: `figures exit 0`; the count is `16` (every PNG of every unit changed: the background is now dark everywhere). Now the Step 3 test is RED: run the Step 3 pytest command — expected `4 failed`, each assertion naming a `course/dist/figures/0-1-x/....png` copy. Look at two outputs before going on: `Read` `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.1-meet-your-car/figures/sign_convention.png` and `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.2-images-are-arrays/figures/exact_vs_threshold.png` (the `Read` tool renders PNGs): dark background, light text and ticks, cyan/light-cyan error boxes in the first, panel-dark/light masks in the second. If text or ticks are black on dark, `figstyle.RC` is not reaching the figure — stop and report; do not patch colours into a figures.py.

- [ ] **Step 7: Rebuild the bundle (dist copies) and prove GREEN.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py; echo "bundle exit $?"; git status --short | grep -vc 'figures/.*\.png'; git status --short | grep -c 'course/dist/figures/.*\.png'`
  Expected: `bundle exit 0`; the non-PNG count is `5` (`scripts/make_figures.py`, `tests/zero2fsd/test_build_labs.py`, `tests/zero2fsd/test_bundle.py`, the two figures.py), the dist PNG count is `16`; `course/dist/course.json` and `course/dist/playground/**` are unchanged (figure names did not change, nor did any `zero2fsd/*.py`). Then: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_bundle.py tests/zero2fsd/test_build_labs.py tests/zero2fsd/test_figure_colours.py tests/playground/test_build.py -q 2>&1 | tail -3` — expected: all passed, 0 failed.

- [ ] **Step 8: Commit** (source files by name; the 32 PNGs by their two directory globs, which the shell expands to explicit paths — no `-A`, no `.`):
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add scripts/make_figures.py tests/zero2fsd/test_build_labs.py tests/zero2fsd/test_bundle.py course/0.1.1-meet-your-car/figures.py course/0.1.2-images-are-arrays/figures.py course/0.1.*/figures/*.png course/dist/figures/*/*.png && git status --short | grep -v '^[MA] ' ; git commit -q -F - <<'MSG'
  feat(figures): dark style via rc_context; palette colours in 0.1.1 and 0.1.2

  make_figures runs every figures.py inside matplotlib.rc_context(figstyle.RC), so the
  page colours, text colour and quantity cycle are set once. 0.1.1 and 0.1.2 draw only
  palette entries (QUANTITY, SCENE, CHROME, shades). All 16 figures regenerated and
  course/dist/figures rebuilt; test_bundle keeps the dist copies byte-equal from now on.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 log --oneline -1 && git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 status --short | wc -l
  ```
  Expected: the `grep -v` before the commit prints nothing (everything modified is staged); the final `wc -l` prints `0` (clean tree).

### Task 5: Palette colours in 0.1.3 and 0.1.4; the G2 gate on every figures.py (R3)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_figure_colours.py` (the real-repo gate, appended)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.3-your-first-perception-box/figures.py`, `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.4-closing-the-loop/figures.py`
- Regenerate + commit: the 8 PNGs under `course/0.1.3-your-first-perception-box/figures/` and `course/0.1.4-closing-the-loop/figures/`, and their 8 copies under `course/dist/figures/0-1-3/` and `course/dist/figures/0-1-4/`

**Interfaces:** Gate G2 = `tests/zero2fsd/test_figure_colours.py::test_no_figure_module_holds_a_string_colour_literal[<file>]`, parametrized over `SCANNED` (every `course/*/figures.py` + `scripts/figstyle.py`; a new unit is scanned automatically). Two learner-visible words change inside figures.py: the 0.1.3 `perspective_grid` title "black lines: ..." → "grey lines: ..." and the 0.1.4 bend legend "green: left bend / purple: right bend" → "shaded: left bend / unshaded: right bend" (lecture.md and lab.py are Part B's; they do not mention these words — `grep -rn "black lines\|green: left\|purple" course/0.1.3*/lecture.md course/0.1.3*/lab.py course/0.1.4*/lecture.md course/0.1.4*/lab.py` prints nothing; if it does, stop and report instead of editing Part B's files). No build_labs.py change.

- [ ] **Step 1: The gate (RED on 0.1.3 and 0.1.4).** Append at the end of `tests/zero2fsd/test_figure_colours.py`:

  ```python
  # --- G2 on the real tree ---


  @pytest.mark.parametrize("source", SCANNED, ids=[f"{p.parent.name}/{p.name}" for p in SCANNED])
  def test_no_figure_module_holds_a_string_colour_literal(source):
      """Colours come from zero2fsd.palette (QUANTITY, SCENE, CHROME, shades); the style comes from scripts/figstyle.py."""
      assert colour_literals(source.read_text(encoding="utf-8")) == [], f"{source.relative_to(REPO)} holds colour literals (line, text)"
  ```
  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_figure_colours.py -q -k "no_figure_module" 2>&1 | tail -4`
  Expected: `2 failed, 3 passed` — `[0.1.3-your-first-perception-box/figures.py]` (35 literals: `tab:red`, `black`, `grey`, `white`, `none`, `tab:blue`, `tab:orange`, `tab:cyan`) and `[0.1.4-closing-the-loop/figures.py]` (15: `tab:green`, `tab:purple`, `C`, `C0`, `C3`, `black`, `0.4`, `0.5`, `0.7`) fail; 0.1.1, 0.1.2 and figstyle pass.

- [ ] **Step 2: Restyle `course/0.1.3-your-first-perception-box/figures.py`.** Line numbers are the file's before any edit; match on the exact old text.
  - :8 `from zero2fsd.sim.camera import PALETTE, ROAD, WHITE, YELLOW, to_vehicle` → `from zero2fsd.sim.camera import WHITE, YELLOW, to_vehicle`; add `from zero2fsd.palette import CHROME, QUANTITY, SCENE` on its own line between `from zero2fsd.car.provided import FAR_ROWS, NEAR_ROWS, estimate_lane, lane_point, paint_masks` (:6) and `from zero2fsd.sim import BicycleState, Camera, make_scenario` (:7)
  - :18 `GREY = PALETTE[ROAD] / 255` → `PAINT = {YELLOW: SCENE["yellow"], WHITE: SCENE["white"]}  # paint drawn in the colours the camera sees`
  - :34 `ax.set_facecolor(GREY)` → `ax.set_facecolor(SCENE["road"])`
  - :35 `color="tab:red"` → `color=QUANTITY["perception"]`
  - :46 `side.axhline(0, color="black", lw=1)` → `side.axhline(0, color=CHROME["text"], lw=1)`
  - :47 `ls="--", color="tab:blue")` → `ls="--", color=QUANTITY["e_off"])`; :48 `color="tab:blue")` → `color=QUANTITY["e_off"])`
  - :52 `color="tab:red" if bold else "grey"` → `color=QUANTITY["perception"] if bold else CHROME["text"]`
  - :54 `color="tab:red" if bold else "black")` → `color=QUANTITY["perception"] if bold else CHROME["text"])`
  - :55 `side.plot(0, h, marker="s", ms=9, color="black")` → `side.plot(0, h, marker="s", ms=9, color=CHROME["text"])`
  - :65 `curve.semilogy(rows, CAMERA.pixel_to_ground(CAMERA.cx, rows)[0], color="black")` → `color=CHROME["text"])`
  - :66 `curve.axvline(HORIZON, ls="--", color="tab:blue")` → `color=QUANTITY["e_off"])`; :67 `curve.text(HORIZON + 1, 300, "horizon", color="tab:blue")` → `color=QUANTITY["e_off"])`
  - :69 `curve.axvspan(a, b, alpha=0.15, color="tab:orange")` → `curve.axvspan(a, b, alpha=0.25, color=QUANTITY["psi"])`
  - :71 `"o", color="tab:red")` → `"o", color=QUANTITY["perception"])`
  - :86 `color="black", lw=0.6)` → `color=CHROME["text"], lw=0.6)`
  - :89 `ax.axhline(v, color="tab:red", lw=1.5)` → `ax.axhline(v, color=QUANTITY["perception"], lw=1.5)`; :90 `color="tab:red", fontsize=8,` → `color=QUANTITY["perception"], fontsize=8,`
  - :91 `bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})` → `bbox={"facecolor": CHROME["panel"], "linewidth": 0, "pad": 1})`
  - :94 `"black lines: 1 m apart on the ground; red lines: fixed distances ahead"` → `"grey lines: 1 m apart on the ground; red lines: fixed distances ahead"`
  - :108 `image.axhline(edge, color="tab:orange", lw=1.5, ls="--")` → `color=QUANTITY["psi"]`
  - :109 `bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})` → `bbox={"facecolor": CHROME["panel"], "linewidth": 0, "pad": 1})`
  - :117 `top.scatter(left, forward, s=2, color=PALETTE[paint] / 255)` → `color=PAINT[paint])`
  - :118 and :119 `"x", ms=10, mew=2, color="tab:red")` → `"x", ms=10, mew=2, color=QUANTITY["perception"])` (both lines)
  - :124 `color="black", lw=1.5, label="line through the two lane centres")` → `color=CHROME["text"], lw=1.5, ...`
  - :125 `mfc="white", mec="black"` → `mfc=CHROME["text"], mec=CHROME["background"]`
  - :126 `ls=":", color="tab:cyan", lw=2, label="true lane centre")` → `ls=":", color=QUANTITY["e_off"], lw=2, ...`
  - :128 `arrowprops={"arrowstyle": "<->", "color": "white"})` → `arrowprops={"arrowstyle": "<->", "color": CHROME["text"]})`
  - :129 `ha="left", va="center", color="white", fontsize=9)` → `ha="left", va="center", color=CHROME["text"], fontsize=9)`
  - :148 `color=PALETTE[paint] / 255, lw=3)` → `color=PAINT[paint], lw=3)`
  - :149 `ls=":", color="tab:cyan", lw=2, label="true lane centre (a curve)")` → `color=QUANTITY["e_off"]`
  - :150 `ls="--", color="tab:red", lw=1,` → `ls="--", color=QUANTITY["perception"], lw=1,`
  - :152 `color="black", lw=1.5, label="straight line through the two centres")` → `color=CHROME["text"], lw=1.5, ...`
  - :153 `mfc="white", mec="black")` → `mfc=CHROME["text"], mec=CHROME["background"])`

- [ ] **Step 3: Restyle `course/0.1.4-closing-the-loop/figures.py`.** Add `from zero2fsd.palette import CHROME, QUANTITY, shades` on its own line between `from zero2fsd.car.types import Action` (:9) and `from zero2fsd.score import metrics` (:10). Then:
  - :34-38, the whole `_shade_curvy_bends` function, becomes
    ```python
    def _shade_curvy_bends(ax):
        for k in range(0, 6, 2):  # the left bends; the right bends stay unshaded
            a = CURVY_FIRST_ARC + k * CURVY_ARC
            ax.axvspan(a, a + CURVY_ARC, color=CHROME["text"], alpha=0.1)
        ax.text(CURVY_FIRST_ARC + 1, 1.45, "shaded: left bend\nunshaded: right bend", fontsize=8)
    ```
  - :82 `ax.axhline(edge, color="0.5", ls="--", lw=1)` → `ax.axhline(edge, color=CHROME["text"], ls="--", lw=1)`
  - :101-102
    ```python
        for (tel, label), style in zip(runs, ("-", ":", "-")):
            ax.plot(tel.t, tel.lat, ls=style, lw=2, label=f"{label}: {_ends(tel)}")
    ```
    →
    ```python
        for (tel, label), style, colour in zip(runs, ("-", ":", "-"), shades(QUANTITY["e_off"], len(runs))):
            ax.plot(tel.t, tel.lat, ls=style, lw=2, color=colour, label=f"{label}: {_ends(tel)}")
    ```
  - :117 `lat.plot(tel.t, tel.lat, color=f"C{i}", label=...` → `color=shades(QUANTITY["e_off"], len(SWEEP))[i]`
  - :120 `lat.axvline(bend, color="0.7", lw=1)` → `lat.axvline(bend, color=CHROME["text"], lw=1)`
  - :121 `lat.text(bend + 0.2, 1.6, "first bend", color="0.4", fontsize=9)` → `lat.text(bend + 0.2, 1.6, "first bend", fontsize=9)`
  - :128 `steer.plot(tel.t, tel.steer, color=f"C{i}", label=...` → `color=shades(QUANTITY["delta"], len(SWEEP))[i]` (the same `i` as the lat panel, so a gain keeps its shade index across the two panels)
  - :140 `left.plot(gentle.progress_m, gentle.lat, color="C0", label=...` → `color=QUANTITY["e_off"]`
  - :142 `left.axhline(predicted, color="black", ls=":", label=...` → `color=CHROME["text"]`
  - :148 `right.plot(curvy.progress_m, curvy.lat, color="C0", label=...` → `color=QUANTITY["e_off"]`
  - :149-151: `for turn, colour in ((+1, "tab:green"), (-1, "tab:purple")):` → `for turn, ls in ((+1, ":"), (-1, "--")):` and, two lines below, `right.axhline(steady_offset(radius, turn), color=colour, ls=":",` → `right.axhline(steady_offset(radius, turn), color=CHROME["text"], ls=ls,` (the two predictions are told apart by line style, like the bends are told apart by shading)
  - :166 `lat.plot(perfect.progress_m, perfect.lat, color="C0", label=...` → `color=QUANTITY["e_off"]`
  - :167 `lat.plot(box.progress_m, box.lat, color="C3", label=...` → `color=QUANTITY["perception"]`
  - :172 `head.axvspan(CURVY_FIRST_ARC, box.progress_m[-1], color="tab:green", alpha=0.08)` → `color=CHROME["text"], alpha=0.1)`
  - :173 `head.plot(box.progress_m, box.head_err, color="black", label="true heading")` → `color=QUANTITY["psi"]`
  - :174 `head.plot(box.progress_m, box.est_heading, color="C3", label="what the box reported")` → `color=QUANTITY["perception"]`

  Run the Step 1 pytest command. Expected: `5 passed`. If a case still fails, its message lists `(line, literal)` pairs: fix the line with a palette value; never edit the scanner or add an exemption. Then the whole module: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_figure_colours.py -q 2>&1 | tail -2` — expected `28 passed`.

- [ ] **Step 4: Regenerate the two units' figures and the dist copies.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/make_figures.py course/0.1.3-your-first-perception-box course/0.1.4-closing-the-loop; echo "figures exit $?"; PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py; echo "bundle exit $?"; git status --short`
  (`make_figures.py main()` takes unit directories as positional `units`, so only the two units are redrawn.) Expected: both exits `0`; status lists exactly `M tests/zero2fsd/test_figure_colours.py`, the two figures.py, 8 `M course/0.1.3-.../figures/*.png` + `course/0.1.4-.../figures/*.png`, and 8 `M course/dist/figures/0-1-3/*.png` + `course/dist/figures/0-1-4/*.png`; nothing under 0.1.1/0.1.2, `course/dist/course.json` or `course/dist/playground/`.
  Look at the output: `Read` `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.3-your-first-perception-box/figures/perspective_grid.png` (road-grey panel, light grid lines, coral-red distance lines, labels on dark panel-coloured boxes) and `/Users/vivek/Antigravity/self-driving-ai-course-week1/course/0.1.4-closing-the-loop/figures/steady_state.png` (cyan traces; light dotted/dashed predictions; left bends shaded light, right bends unshaded). Any black text, white label box or a legend naming green/purple means a missed edit: find it with the Step 1 pytest command or by `grep -n '"' <figures.py>` and fix the line.
  Then every gate touched by Part A: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider tests/zero2fsd/test_figure_colours.py tests/zero2fsd/test_bundle.py tests/zero2fsd/test_build_labs.py tests/zero2fsd/test_palette.py tests/playground/test_build.py -q 2>&1 | tail -2` — expected: all passed, 0 failed.

- [ ] **Step 5: Commit.**
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add tests/zero2fsd/test_figure_colours.py course/0.1.3-your-first-perception-box/figures.py course/0.1.4-closing-the-loop/figures.py course/0.1.3-your-first-perception-box/figures/*.png course/0.1.4-closing-the-loop/figures/*.png course/dist/figures/0-1-3/*.png course/dist/figures/0-1-4/*.png && git status --short | grep -v '^[MA] ' ; git commit -q -F - <<'MSG'
  feat(figures): palette colours in 0.1.3 and 0.1.4; G2 gate on every figures.py

  Every course/*/figures.py and scripts/figstyle.py now passes
  test_no_figure_module_holds_a_string_colour_literal. 0.1.3 draws paint in the
  camera's colours and lanes/rays in the quantity hues; 0.1.4 tells gains apart by
  shades of one hue and bends by shading instead of green/purple. Figures and
  course/dist/figures regenerated.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 log --oneline -1 && git -C /Users/vivek/Antigravity/self-driving-ai-course-week1 status --short | wc -l
  ```
  Expected: nothing printed by the `grep -v`; the final `wc -l` prints `0`.

### Task 6: Per-exercise draft in localStorage + "Reset to starter"

**Files:**
- Create: `playground/draft.js`
- Modify: `playground/app.js` (imports :2-4, `choose` :97-111, `mountEditor` :113-142, `setUp` :151 and :174-176, listeners :410-413)
- Modify: `playground/index.html:50-53` (the `.toolbar`)
- Test: `tests/playground/test_browser.py` (helpers after `seed_progress` :305-309; tests after `test_failing_check_records_nothing` :613-618, before the `# ---- stop` heading at :670)
- Rebuild + commit: `course/dist/playground/`

**Interfaces:**
- Consumes: `playground/progress.js` as the pattern (try/catch, never throws); `window.__setCode` / `window.__getCode` test hooks (app.js:175-176); `set_code`, `exercise_url`, `EDITOR_READY` from `scripts/playground_browser.py`; `STEER_ID = "0.1.4.a"`, `RETURN_ZERO`, `BY_ID`, `PASSING` constants already in test_browser.py.
- Produces: `readDraft`, `saveDraft`, the key format, `choose(…, savedDraft)`, `mountEditor(…, onChange)`, `#reset`, and the test helpers named in Part interfaces (Task 7 builds on all of them).

- [ ] **Step 1: Add the test helpers and the failing browser test**

In `tests/playground/test_browser.py`, directly after `seed_progress` (ends :309, before `def scroll_width`), add:

```python
DRAFT_KEY = "learnfsd.v2.draft."  # + the exercise id; playground/draft.js


def stored_draft(page, exercise_id):
    return page.evaluate("key => localStorage.getItem(key)", DRAFT_KEY + exercise_id)


def open_editor(page, base_url, exercise_id, query=""):
    """Open an exercise and wait for the editor only: a draft needs no runtime, so the test does not wait for Pyodide."""
    page.goto(exercise_url(base_url, exercise_id, query))
    page.wait_for_function(EDITOR_READY)


def editor_text(page):
    return page.evaluate("window.__getCode()")
```

Directly after `test_failing_check_records_nothing` (:613-618) add:

```python
# ---- drafts (R8) --------------------------------------------------------------------------------------------
# The site's exercise tabs reload the embedded page, so typed code must survive a reload. The draft is per exercise;
# a draft equal to the starter is no draft, so "Reset to starter" leaves nothing behind.


def test_typed_code_is_kept_per_exercise_across_a_reload_and_reset_to_starter_drops_it(page, base_url):
    other = PASSING[1]  # 0.1.3.a: a second exercise, whose editor must not show the first one's draft
    starter = BY_ID[STEER_ID]["presets"]["starter"]
    open_editor(page, base_url, STEER_ID)
    assert stored_draft(page, STEER_ID) is None, "opening the starter must not store it as a draft"
    set_code(page, RETURN_ZERO)
    assert stored_draft(page, STEER_ID) == RETURN_ZERO
    open_editor(page, base_url, STEER_ID)
    assert editor_text(page) == RETURN_ZERO, "a reload lost the typed code"
    expect(page.locator("#restore-draft")).to_be_hidden()  # the draft is already open: nothing to restore
    open_editor(page, base_url, other)
    assert editor_text(page) == BY_ID[other]["presets"]["starter"], "another exercise shows this one's draft"
    assert stored_draft(page, other) is None
    open_editor(page, base_url, STEER_ID)
    page.click("#reset")
    assert editor_text(page) == starter
    assert stored_draft(page, STEER_ID) is None, "Reset to starter left the draft in storage"
    set_code(page, RETURN_ZERO)
    set_code(page, starter)  # typing back to the starter by hand is the same as resetting
    assert stored_draft(page, STEER_ID) is None, "code equal to the starter was stored as a draft"
    open_editor(page, base_url, STEER_ID)
    assert editor_text(page) == starter


def test_the_textarea_fallback_keeps_the_draft_too(page, base_url):
    """With CodeMirror unreachable the plain textarea is the editor (test_editor_import_blocked_falls_back_to_textarea): its
    edits must be drafts as well, and the draft must open in it after a reload."""
    page.route(re.compile(r"/npm/@codemirror/"), lambda route: route.abort())
    open_editor(page, base_url, STEER_ID)
    expect(page.locator("#editor-fallback")).to_be_visible()
    set_code(page, RETURN_ZERO)  # page.fill on the textarea: real input events, not the CodeMirror hook
    assert stored_draft(page, STEER_ID) == RETURN_ZERO
    open_editor(page, base_url, STEER_ID)
    expect(page.locator("#editor-fallback")).to_have_value(RETURN_ZERO)
    page.click("#reset")
    expect(page.locator("#editor-fallback")).to_have_value(BY_ID[STEER_ID]["presets"]["starter"])
    assert stored_draft(page, STEER_ID) is None
```

- [ ] **Step 2: Run it and watch it fail**

```bash
mkdir -p /private/tmp/claude-501/c1-out && cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -m browser -v "tests/playground/test_browser.py::test_typed_code_is_kept_per_exercise_across_a_reload_and_reset_to_starter_drops_it" "tests/playground/test_browser.py::test_the_textarea_fallback_keeps_the_draft_too" > /private/tmp/claude-501/c1-out/C1-red.txt 2>&1; tail -8 /private/tmp/claude-501/c1-out/C1-red.txt
```
Expected: `2 failed`, each with `AssertionError: assert None == 'def p_steer(est, k_off, k_head):\n    return 0.0\n'` at its first `stored_draft(page, STEER_ID) == RETURN_ZERO` (nothing writes a draft yet).

- [ ] **Step 3: Create `playground/draft.js`**

```js
// The learner's code for one exercise, kept in localStorage so a reload keeps it (the site's exercise tabs reload the
// embedded page). One key per exercise; a draft is code that differs from the starter, so the starter itself is never stored.
const key = (exerciseId) => `learnfsd.v2.draft.${exerciseId}`;

/** The saved code for `exerciseId`; null when there is none or storage is unreadable (it then reads as no draft). Never throws. */
export function readDraft(exerciseId) {
  try {
    return localStorage.getItem(key(exerciseId));
  } catch {
    return null;
  }
}

/** Keep `code` as the draft of `exerciseId`, or drop the draft when `code` is the starter. Storage that refuses is left alone:
 *  the page never claims a draft is saved. Never throws. */
export function saveDraft(exerciseId, code, starter) {
  try {
    if (code === starter) localStorage.removeItem(key(exerciseId));
    else localStorage.setItem(key(exerciseId), code);
  } catch {
    // blocked or full storage: nothing to tell the learner on every keystroke
  }
}
```

- [ ] **Step 4: Wire it into `playground/app.js`**

(a) After line 4 `import { markPassed } from "./progress.js";` add:
```js
import { readDraft, saveDraft } from "./draft.js";
```

(b) Replace the whole `choose` function and its doc comment (:97-111, from `/**` through the closing `}`) with:
```js
/**
 * The exercise the URL asks for and the code to open it with. An unknown exercise falls back to the first record and an
 * unknown preset to the starter, each with a notice. A saved draft opens unless the URL names a preset: the preset wins.
 * `autoRun` is true only when the URL named both exercise and preset and both exist.
 */
function choose({ records }, savedDraft) {
  const id = params.get("exercise");
  const known = records.find((r) => r.id === id);
  const record = known ?? records[0];
  const notices = [];
  if (!known && id) notices.push(`Unknown exercise ${id}; showing ${record.label}.`);
  const asked = params.get("preset");
  const preset = asked !== null && Object.hasOwn(record.presets, asked) ? asked : "starter";
  if (asked !== null && preset !== asked) notices.push(`Unknown preset ${asked}; showing the starter.`);
  const draft = savedDraft(record.id);
  const code = asked === null && draft !== null ? draft : record.presets[preset];
  return { record, code, notice: notices.join(" "), autoRun: Boolean(known) && asked === preset };
}
```

(c) `mountEditor` (:113-142). Change its comment and signature from
```js
/** CodeMirror 6 from the import map in index.html; any failure leaves a plain textarea. Returns { get, set } for the code. */
async function mountEditor(parent, doc) {
```
to
```js
/** CodeMirror 6 from the import map in index.html; any failure leaves a plain textarea. Returns { get, set } for the code;
 *  `onChange(text)` is called after every edit, including one made through `set`. */
async function mountEditor(parent, doc, onChange) {
```
In the `extensions: [` list, after the line ending `commands.history(),` add one line:
```js
          view.EditorView.updateListener.of((update) => { if (update.docChanged) onChange(update.state.doc.toString()); }),
```
In the catch branch, after the `const textarea = Object.assign(…)` line add:
```js
    textarea.addEventListener("input", () => onChange(textarea.value));
```
and change the fallback's return from
```js
    return { get: () => textarea.value, set: (text) => { textarea.value = text; } };
```
to
```js
    return { get: () => textarea.value, set: (text) => { textarea.value = text; onChange(text); } };
```

(d) In `setUp`, line 151 becomes:
```js
  const { record, code, notice, autoRun } = choose({ records }, readDraft);
```
and line 174 becomes:
```js
  const editor = await mountEditor($("editor"), code, (text) => saveDraft(record.id, text, record.presets.starter));
```
After line 176 (`window.__getCode = editor.get; …`) add:
```js
  $("reset").disabled = false; // the editor is up: the starter can be put back
```

(e) After line `$("retry").addEventListener("click", boot);` (:413) add:
```js
$("reset").addEventListener("click", () => app.editor.set(app.record.presets.starter)); // the edit listener drops the draft
```

- [ ] **Step 5: Add the control to `playground/index.html`**

After line 52 `<button id="stop" type="button" disabled>Stop</button>` add:
```html
        <button id="reset" type="button" disabled>Reset to starter</button>
```
(It inherits the generic `button` rules in app.css:42-53; no CSS change.)

- [ ] **Step 6: Run the new test and the two affected ones**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -m browser -v "tests/playground/test_browser.py::test_typed_code_is_kept_per_exercise_across_a_reload_and_reset_to_starter_drops_it" "tests/playground/test_browser.py::test_the_textarea_fallback_keeps_the_draft_too" "tests/playground/test_browser.py::test_storage_blocked" "tests/playground/test_browser.py::test_preset_starter_runs_automatically[0.1.4.a]" "tests/playground/test_browser.py::test_editor_import_blocked_falls_back_to_textarea" > /private/tmp/claude-501/c1-out/C1-green.txt 2>&1; tail -8 /private/tmp/claude-501/c1-out/C1-green.txt
```
Expected: `5 passed`. (`test_storage_blocked` asserts `page_errors == []` with every localStorage read throwing: that is the proof `readDraft`/`saveDraft` never throw. The textarea test proves the fallback editor still works with `onChange` attached.)

- [ ] **Step 7: Rebuild the committed playground and run the fast checks**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py && git status --short course/dist && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/playground/test_build.py tests/playground/test_app_css.py > /private/tmp/claude-501/c1-out/C1-build.txt 2>&1; tail -3 /private/tmp/claude-501/c1-out/C1-build.txt
```
Expected: `git status` lists `M course/dist/playground/app.js`, `M course/dist/playground/index.html`, `M course/dist/playground/manifest.json`, `?? course/dist/playground/draft.js` and nothing else under course/dist; pytest `… passed` with 0 failed (test_build.py:315 derives the expected dist file set from `playground/`, so `draft.js` is expected automatically; test_app_css.py:47 scans app.js and index.html for colour literals — none were added).

- [ ] **Step 8: Commit**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add playground/draft.js playground/app.js playground/index.html tests/playground/test_browser.py course/dist/playground && git commit -m "feat(playground): keep a per-exercise draft in localStorage, with Reset to starter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: A `?preset=<name>` link opens the preset and offers "Restore my draft"

**Files:**
- Modify: `playground/app.js` (the `let app` comment :22; `choose` as Task 6 left it; `setUp`'s destructuring line, the `$("reset").disabled = false;` line and its `return { record, kind, … }`; the listeners after the `#reset` one)
- Modify: `playground/index.html` (`.status-actions`, after `#colab-fallback` :58)
- Test: `tests/playground/test_browser.py` (after `test_the_textarea_fallback_keeps_the_draft_too`)
- Rebuild + commit: `course/dist/playground/`

**Interfaces:**
- Consumes: Task 6's `readDraft`, `saveDraft`, `choose({ records }, savedDraft)`, `#reset`, and the test helpers `open_editor`, `editor_text`, `stored_draft`, `DRAFT_KEY`.
- Produces: `choose` also returns `draft` (the saved text to offer, else `null`); `app.draft`; `#restore-draft`; status copy `You have a saved draft of this exercise.` The preset is read from `record.presets` by the name the URL gives: C2's file-backed presets (R13) need no page change.

- [ ] **Step 1: Write the failing browser tests**

After `test_the_textarea_fallback_keeps_the_draft_too` add:

```python
def test_a_preset_link_opens_the_preset_and_offers_the_draft_back(page, base_url):
    """A lecture link names a preset: that code opens, the draft is kept, and one control brings the draft back."""
    starter = BY_ID[STEER_ID]["presets"]["starter"]
    open_editor(page, base_url, STEER_ID)
    set_code(page, RETURN_ZERO)
    open_editor(page, base_url, STEER_ID, "&preset=starter")
    assert editor_text(page) == starter, "the named preset must win over the draft"
    assert stored_draft(page, STEER_ID) == RETURN_ZERO, "opening a preset overwrote the draft"
    expect(page.locator("#restore-draft")).to_be_visible()
    page.click("#restore-draft")
    assert editor_text(page) == RETURN_ZERO
    expect(page.locator("#restore-draft")).to_be_hidden()
    assert stored_draft(page, STEER_ID) == RETURN_ZERO


def test_an_unknown_preset_with_a_draft_says_both_in_the_status_line(page, base_url):
    """An unknown preset shows the starter and does not auto-run, so the ready text stays long enough to read: it names the
    unknown preset and the draft, and the draft is offered."""
    open_editor(page, base_url, STEER_ID)
    set_code(page, RETURN_ZERO)
    open_ready(page, base_url, STEER_ID, "&preset=nope")
    expect(page.locator("#status")).to_contain_text("Unknown preset nope")
    expect(page.locator("#status")).to_contain_text("You have a saved draft of this exercise.")
    expect(page.locator("#restore-draft")).to_be_visible()
    assert editor_text(page) == BY_ID[STEER_ID]["presets"]["starter"]


def test_a_draft_equal_to_the_opened_preset_is_not_offered(page, base_url):
    """A stored draft that is the very code the link opens (left by another tab, say) has nothing to restore."""
    starter = BY_ID[STEER_ID]["presets"]["starter"]
    key, value = json.dumps(DRAFT_KEY + STEER_ID), json.dumps(starter)
    page.add_init_script(f"if (localStorage.getItem({key}) === null) localStorage.setItem({key}, {value});")
    open_ready(page, base_url, STEER_ID, "&preset=nope")  # no auto-run: the ready text can be read
    expect(page.locator("#status")).to_contain_text("Ready.")
    assert "saved draft" not in page.inner_text("#status")
    expect(page.locator("#restore-draft")).to_be_hidden()
```

- [ ] **Step 2: Run them and watch two fail**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -m browser -v "tests/playground/test_browser.py::test_a_preset_link_opens_the_preset_and_offers_the_draft_back" "tests/playground/test_browser.py::test_an_unknown_preset_with_a_draft_says_both_in_the_status_line" "tests/playground/test_browser.py::test_a_draft_equal_to_the_opened_preset_is_not_offered" > /private/tmp/claude-501/c1-out/C2-red.txt 2>&1; tail -8 /private/tmp/claude-501/c1-out/C2-red.txt
```
Expected: `2 failed, 1 passed`. The first fails at `expect(page.locator("#restore-draft")).to_be_visible()` (no such element), the second at `to_contain_text("You have a saved draft…")`; the third is the control and passes already (nothing is offered today either).

- [ ] **Step 3: Return the offer from `choose` and show it**

In `playground/app.js`:

(a) Line 22 becomes:
```js
let app; // the page's long-lived pieces, built once: { record, roads, palette, player, strips, editor, notice, autoRun, draft }
```

(b) In `choose`, the doc comment's sentence `A saved draft opens unless the URL names a preset: the preset wins.` becomes `A saved draft opens unless the URL names a preset: the preset wins and the draft is offered back as \`draft\`.` and the last two lines
```js
  const code = asked === null && draft !== null ? draft : record.presets[preset];
  return { record, code, notice: notices.join(" "), autoRun: Boolean(known) && asked === preset };
```
become
```js
  const code = asked === null && draft !== null ? draft : record.presets[preset];
  const restorable = draft !== null && draft !== code; // a preset link found other code of yours: offer it, never drop it
  if (restorable) notices.push("You have a saved draft of this exercise.");
  return { record, code, notice: notices.join(" "), autoRun: Boolean(known) && asked === preset, draft: restorable ? draft : null };
```

(c) In `setUp`, the destructuring line becomes:
```js
  const { record, code, notice, autoRun, draft } = choose({ records }, readDraft);
```
after `$("reset").disabled = false; …` add:
```js
  $("restore-draft").hidden = draft === null;
```
and the function's last line becomes:
```js
  return { record, kind, roads, palette, player, strips, editor, notice, autoRun, draft };
```

(d) After the `$("reset").addEventListener(…)` line add:
```js
$("restore-draft").addEventListener("click", () => {
  app.editor.set(app.draft); // the edit listener saves it again, unchanged
  app.draft = null;
  $("restore-draft").hidden = true;
});
```

- [ ] **Step 4: Add the control to `playground/index.html`**

After line 58 `<a id="colab-fallback" target="_blank" rel="noopener" hidden>Open this exercise in Colab</a>` add:
```html
        <button id="restore-draft" type="button" hidden>Restore my draft</button>
```
(Generic `button` rules in app.css:42-53 and `.status-actions` :79 apply; no CSS change.)

- [ ] **Step 5: Run the three new tests, Task 6's two, and the two existing preset/notice tests**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -m browser -v "tests/playground/test_browser.py::test_a_preset_link_opens_the_preset_and_offers_the_draft_back" "tests/playground/test_browser.py::test_an_unknown_preset_with_a_draft_says_both_in_the_status_line" "tests/playground/test_browser.py::test_a_draft_equal_to_the_opened_preset_is_not_offered" "tests/playground/test_browser.py::test_typed_code_is_kept_per_exercise_across_a_reload_and_reset_to_starter_drops_it" "tests/playground/test_browser.py::test_the_textarea_fallback_keeps_the_draft_too" "tests/playground/test_browser.py::test_preset_starter_runs_automatically[0.1.4.a]" "tests/playground/test_browser.py::test_unknown_exercise_opens_the_first_exercise_and_says_why[nonsense]" > /private/tmp/claude-501/c1-out/C2-green.txt 2>&1; tail -10 /private/tmp/claude-501/c1-out/C2-green.txt
```
Expected: `7 passed`.

- [ ] **Step 6: Rebuild the committed playground and run the fast checks**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py && git status --short course/dist && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/playground/test_build.py tests/playground/test_app_css.py > /private/tmp/claude-501/c1-out/C2-build.txt 2>&1; tail -3 /private/tmp/claude-501/c1-out/C2-build.txt
```
Expected: `M course/dist/playground/app.js`, `M course/dist/playground/index.html`, `M course/dist/playground/manifest.json` and nothing else; pytest `… passed`, 0 failed.

- [ ] **Step 7: Commit**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add playground/app.js playground/index.html tests/playground/test_browser.py course/dist/playground && git commit -m "feat(playground): a preset link opens the preset and offers the saved draft back

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: Pin `target="_top"` on "Back to the lecture"

The attribute is already there (`playground/index.html:41`, since commit 78bffe7, and in `course/dist/playground/index.html`); the recon's "needs adding" is stale. What is missing is the test: `test_every_exercise_opens_ready_with_its_label_and_links` (:332) pins only `href`, so the attribute could be dropped by any edit of the header without a check going red. The page runs inside the site's `<iframe>` (spec :115); without `target="_top"` the lecture would open inside the iframe.

**Files:**
- Test: `tests/playground/test_browser.py:332` (`test_every_exercise_opens_ready_with_its_label_and_links`)
- Touched only to prove the test bites, then restored: `playground/index.html`

**Interfaces:** none new. The test is parametrized over every exercise record, so it pins the attribute for the class, not for one id.

- [ ] **Step 1: Add the assertion**

After line 332 `expect(page.locator("#unit-link")).to_have_attribute("href", …)` add:
```python
    expect(page.locator("#unit-link")).to_have_attribute("target", "_top")  # the page runs in the site's iframe: leave it
```

- [ ] **Step 2: Seed the violation and watch the test go red**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && sed -i '' 's/ target="_top"//' playground/index.html && grep -c 'target="_top"' playground/index.html; PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -m browser -v "tests/playground/test_browser.py::test_every_exercise_opens_ready_with_its_label_and_links[0.1.4.a]" > /private/tmp/claude-501/c1-out/C3-red.txt 2>&1; tail -6 /private/tmp/claude-501/c1-out/C3-red.txt
```
Expected: `grep -c` prints `0`; pytest `1 failed` with `to_have_attribute("target", "_top")` in the failure (the `base_url` fixture builds the served copy from `playground/`, so the seeded source is what the browser loads).

- [ ] **Step 3: Restore the source and watch it go green**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git checkout -- playground/index.html && git status --short playground && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -m browser -v "tests/playground/test_browser.py::test_every_exercise_opens_ready_with_its_label_and_links[0.1.4.a]" "tests/playground/test_browser.py::test_every_exercise_opens_ready_with_its_label_and_links[0.1.2.a]" > /private/tmp/claude-501/c1-out/C3-green.txt 2>&1; tail -6 /private/tmp/claude-501/c1-out/C3-green.txt
```
Expected: `git status` prints nothing under `playground/` (Task 7 committed index.html, so the checkout restores exactly the committed file); pytest `2 passed`.

- [ ] **Step 4: Commit (tests only; no builder input changed, so no rebuild)**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git status --short course/dist && git add tests/playground/test_browser.py && git commit -m "test(playground): pin target=_top on the back-to-the-lecture link

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: `git status --short course/dist` prints nothing.

### Task 9: The course link check keeps trailing punctuation in the exercise id

Today `_VALUE` (`scripts/build_playground.py:24`) drops a trailing `.,;:!?` from an id or preset, so `Open playground/index.html?exercise=0.1.1.a.` passes the course build and then fails the site's sync (R8, checklist :265). Both checks must read the same id; the stricter one wins. The repo's lectures hold no playground links today (`grep -rn 'index.html?exercise=' course/*/lecture.md` → 0), so the change breaks no committed text; C2's R15 links must be markdown links `[text](playground/index.html?exercise=0.1.1.a)` with the sentence's punctuation after the `)`.

**Files:**
- Modify: `scripts/build_playground.py:24` (`_VALUE`)
- Test: `tests/playground/test_build.py` (:228-238 pattern test, :241-250 resolving links, after :266 two new failing-link tests)

**Interfaces:** `_VALUE = r"""([^&"'<>#\s)]+)"""`: an id or preset runs from `exercise=`/`preset=` to the first `& " ' < > # whitespace )` and includes everything before it. `validate_lecture_links` is unchanged; its messages stay `… (no such exercise)` / `… (exercise <id> has presets [...])`.

- [ ] **Step 1: Write the failing tests**

After `test_a_lecture_link_to_a_missing_exercise_or_preset_names_the_lecture_the_link_and_why` (ends :266) add:

```python
@pytest.mark.parametrize("mark", list(".,;:!?"))
def test_sentence_punctuation_after_a_bare_link_is_part_of_the_id_and_fails_the_build(mini_repo, mark):
    """The site's sync reads the punctuation as part of the id; the course build must refuse the same text (R8)."""
    link_in_lecture(mini_repo, "0.1.1-mini", f"Open playground/index.html?exercise=0.1.1.a{mark} Then read on.")
    with pytest.raises(bp.BuildError) as caught:
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))
    assert f"playground/index.html?exercise=0.1.1.a{mark} (no such exercise)" in str(caught.value)


def test_sentence_punctuation_after_a_bare_preset_is_part_of_the_preset_and_fails_the_build(mini_repo):
    link_in_lecture(mini_repo, "0.1.1-mini", "Try playground/index.html?exercise=0.1.1.a&preset=starter.")
    with pytest.raises(bp.BuildError) as caught:
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))
    assert "playground/index.html?exercise=0.1.1.a&preset=starter. (exercise 0.1.1.a has presets ['starter'])" in str(caught.value)
```

- [ ] **Step 2: Run them and watch all seven fail**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/playground/test_build.py -k "sentence_punctuation" > /private/tmp/claude-501/c1-out/C4-red.txt 2>&1; tail -4 /private/tmp/claude-501/c1-out/C4-red.txt
```
Expected: `7 failed` — each with `Failed: DID NOT RAISE` (the current pattern drops the mark and the link resolves).

- [ ] **Step 3: Change the pattern**

In `scripts/build_playground.py` replace line 24
```python
_VALUE = r"""([^&"'<>#\s)]*[^&"'<>#\s).,;:!?])"""  # a link's id or preset: ends at a delimiter, never on the punctuation of the sentence around it
```
with
```python
_VALUE = r"""([^&"'<>#\s)]+)"""  # a link's id or preset: everything up to a delimiter. Sentence punctuation after a bare link is part of the id, as the site's sync reads it, so the build refuses it (R8)
```

- [ ] **Step 4: Bring the pattern test and the resolving-links test to the new rule**

In `tests/playground/test_build.py`, the parametrize list at :228-236 becomes
```python
@pytest.mark.parametrize(("text", "found"), [
    ("playground/index.html?exercise=0.1.4.a", ("0.1.4.a", None)),
    ("see playground/index.html?exercise=0.1.4.a.", ("0.1.4.a.", None)),
    ("playground/index.html?exercise=0.1.4.a#frag", ("0.1.4.a", None)),
    ("<a href='playground/index.html?exercise=0.1.4.a'>", ("0.1.4.a", None)),
    ('<a href="playground/index.html?exercise=0.1.4.a.">', ("0.1.4.a.", None)),
    ("(playground/index.html?exercise=0.1.4.a&preset=starter), and", ("0.1.4.a", "starter")),
    ("playground/index.html?exercise=0.1.4.a&preset=starter.", ("0.1.4.a", "starter.")),
    ('<a href="playground/index.html?exercise=0.1.4.a&preset=starter#top">', ("0.1.4.a", "starter")),
])
def test_the_link_pattern_stops_the_id_and_the_preset_at_a_delimiter_and_keeps_trailing_punctuation(text, found):
```
(the function body is unchanged). In the list at :241-247 replace the line
```python
    "Open playground/index.html?exercise=0.1.1.a.",
```
with
```python
    "[Open it](playground/index.html?exercise=0.1.1.a).",
```

- [ ] **Step 5: Run the link tests**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/playground/test_build.py > /private/tmp/claude-501/c1-out/C4-green.txt 2>&1; tail -3 /private/tmp/claude-501/c1-out/C4-green.txt
```
Expected: all passed, 0 failed (the file includes `test_every_lecture_link_in_the_repo_resolves` and the dist freshness test; neither changes, since the pattern is not a dist input).

- [ ] **Step 6: Commit**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git status --short course/dist && git add scripts/build_playground.py tests/playground/test_build.py && git commit -m "fix(build): a lecture link keeps its sentence punctuation in the id, as the site's sync does

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: `git status --short course/dist` prints nothing.

### Task 10: Playground spec text brought up to date (R10)

Docs only. Each edit is anchored on the sentence it replaces, not the line number (line numbers below are as of HEAD 335491a). Verified against the code: `playground/view.js` exports `fitCanvas` (:21), `fitTransform` (:34), `worldToCanvas` (:45), `draw` (:113), `drawStrips` (:147), `readoutText` (:206), `describeRun` (:218); the image kind is `zero2fsd/preview/__init__.py:6` `build(learner_fn, frame) -> uint8[H, W, 3]` and exercise 0.1.2.a uses it; the site serves `public/` by exact path, so `playground/?exercise=` is a 404 and `WRONG_FORM_LINK` refuses it.

**Files:**
- Modify: `docs/superpowers/specs/2026-10-11-playground-design.md` (:68, :110, :115, :118, :170)

**Interfaces:** none (prose).

- [ ] **Step 1: Make the five edits**

1. (:68) Replace the sentence
   `` `image` (paint masks on a frame) is added in Phase 3. ``
   with
   `` `image` (`build(learner_fn, frame) -> uint8[H, W, 3]`: the learner's masks painted on a camera frame; `zero2fsd/preview/__init__.py`) shipped in v1 and is used by 0.1.2.a. ``

2. (:110) Replace the whole paragraph
   `` `view.js` is a dependency-free ES module with one export, `draw(canvasCtx, road, rows, k, palette)`. The site's lecture sliders (D5) import it at runtime from `/learnfsd/playground/view.js` and feed it precomputed rows from `sweeps/<name>.json`, so the lecture picture and the playground picture are the same picture. ``
   with
   `` `view.js` is a dependency-free ES module. It exports `fitCanvas`, `fitTransform(road, width, height, padding)`, `worldToCanvas`, `draw(ctx, road, rows, k, palette, t)`, `drawStrips(ctx, rows, k, palette, windowS)`, `readoutText` and `describeRun`. The site's lecture sliders (D5) import it at runtime from `/learnfsd/playground/view.js` and feed it precomputed rows; lecture runs live in `playground/runs/` (`dist/playground/runs/<slug>/<stem>.json`, listed in the playground manifest), so the lecture picture and the playground picture are the same picture. ``

3. (:115) Replace
   `` `<iframe src="/learnfsd/playground/?exercise=<id>">` ``
   with
   `` `<iframe src="/learnfsd/playground/index.html?exercise=<id>[&preset=<p>]">` ``

4. (:118) Replace the sentence
   `` written in `lecture.md` as `../../playground/?exercise=<id>&preset=<name>` so the same text resolves on `/learnfsd/unit/<slug>` today and on `/unit/<slug>/` on GitHub Pages. The bundle validator checks each link against `exercises.json`. ``
   with
   `` written in `lecture.md` as `playground/index.html?exercise=<id>[&preset=<p>]` (the site serves `public/` by exact path, so the directory form is a 404), as a markdown link with the sentence's punctuation outside the `)`. The course build's `validate_lecture_links` refuses any other form, an unknown id or preset, and an id that carries sentence punctuation. ``

5. (:170) Replace
   `` Weeks 2–4 then add the `image` kind and their presets; nothing structural. ``
   with
   `` Weeks 2–4 then add their presets (the `image` kind shipped in v1); nothing structural. ``

- [ ] **Step 2: Verify no stale form or stale claim remains**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && f=docs/superpowers/specs/2026-10-11-playground-design.md; echo "stale link forms: $(grep -c 'playground/?exercise' $f)"; echo "index.html links: $(grep -c 'index.html?exercise' $f)"; echo "one export: $(grep -c 'one export' $f)"; echo "image added later: $(grep -c 'is added in Phase 3\|add the `image` kind' $f)"; echo "runs folder: $(grep -c 'playground/runs/' $f)"; git diff --stat
```
Expected: `stale link forms: 0`, `index.html links: 2`, `one export: 0`, `image added later: 0`, `runs folder: 1`; `git diff --stat` shows only the spec file.

- [ ] **Step 3: Commit**

```bash
cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add docs/superpowers/specs/2026-10-11-playground-design.md && git commit -m "docs(spec): playground spec names the shipped link form, view.js exports, runs folder and image kind

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 11: One module spells the learner-facing names (R4)

**Files:**
- Create: `/Users/vivek/Antigravity/self-driving-ai-course-week1/zero2fsd/labels.py`
- Create: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_labels.py`
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/build_playground.py` (import block; the `"label":` entry inside `exercise_records`)
- Rebuild + commit: `course/dist/playground/bundle.zip`, `course/dist/playground/manifest.json`

**Interfaces:** see Part interfaces, first bullet. `exercise_records` keeps its output byte-identical (the label text does not change in this task).

- [ ] **Step 1: Pin the spellings (RED).** Create `tests/zero2fsd/test_labels.py`:

  ```python
  from zero2fsd.labels import CHECK_MY_WORK, exercise_label, exercise_name, lecture_section


  def test_the_names_a_learner_reads_for_an_exercise_a_lecture_section_and_the_check_action():
      assert exercise_name("0.1.1.b") == "Exercise B"
      assert exercise_label(1, "Meet your car", "0.1.1.a") == "Week 1 · Meet your car · Exercise A"
      assert lecture_section("Meet your car", "Lateral error") == 'lecture "Meet your car", section "Lateral error"'
      assert CHECK_MY_WORK == "Check my work"
  ```

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_labels.py 2>&1 | tail -3`
  Expected: `ModuleNotFoundError: No module named 'zero2fsd.labels'` (collection error, 1 error).

- [ ] **Step 2: Write the module (GREEN).** Create `zero2fsd/labels.py` exactly:

  ```python
  """The names a learner reads for an exercise, a lecture section and the check action, spelled in one place.

  Course ids ("0.1.1", "0.1.1.b") stay in code, URLs and function arguments; a learner reads titles and letters.
  Imported by the grader, the playground shim and the course build scripts: keep it free of other imports."""
  from __future__ import annotations

  CHECK_MY_WORK = "Check my work"  # the action: the playground button and the comment above every Colab check cell


  def exercise_name(exercise_id: str) -> str:
      """'0.1.1.b' -> 'Exercise B'."""
      return f"Exercise {exercise_id.rsplit('.', 1)[-1].upper()}"


  def exercise_label(week: int, title: str, exercise_id: str) -> str:
      """'Week 1 · Meet your car · Exercise B': the playground's heading for an exercise."""
      return f"Week {week} · {title} · {exercise_name(exercise_id)}"


  def lecture_section(title: str, heading: str) -> str:
      """'lecture "Meet your car", section "Lateral error"': how a hint points into the lecture."""
      return f'lecture "{title}", section "{heading}"'
  ```

  Run the Step 1 command. Expected: `1 passed`.

- [ ] **Step 3: The playground builder uses it.** In `scripts/build_playground.py`, insert `from zero2fsd.labels import exercise_label` between the lines `from zero2fsd.export import roads_dict` and `from zero2fsd.palette import palette_dict`, and inside `exercise_records` replace

  ```python
                  "unit_title": meta["title"], "label": f"Week {week} · {meta['title']} · Exercise {exercise['id'].rsplit('.', 1)[-1].upper()}",
  ```
  with
  ```python
                  "unit_title": meta["title"], "label": exercise_label(week, meta["title"], exercise["id"]),
  ```

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/playground/test_build.py 2>&1 | tail -3`
  Expected: all tests pass except possibly one freshness failure naming `course/dist/playground` (the committed bundle.zip predates labels.py); `test_week_and_label_follow_the_unit_position`, `test_labels_never_show_a_dotted_id` and the `Week 2 · Mini unit · Exercise A` test pass unchanged — they compute the label independently.

- [ ] **Step 4: Rebuild the playground bundle.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py && git status --short course`
  Expected: two `wrote …` lines, then exactly ` M course/dist/playground/bundle.zip` and ` M course/dist/playground/manifest.json` (labels.py is now in the bundle; exercises.json is unchanged because the label text is unchanged). Re-run the Step 3 pytest command. Expected: `… passed` with no failures.

- [ ] **Step 5: Commit.**
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add zero2fsd/labels.py tests/zero2fsd/test_labels.py scripts/build_playground.py course/dist/playground/bundle.zip course/dist/playground/manifest.json && git commit -q -F - <<'MSG'
  feat(labels): one module spells the learner-facing names

  exercise_name/exercise_label/lecture_section/CHECK_MY_WORK live in zero2fsd/labels.py; the playground builder
  imports the label instead of formatting it inline. Bundle rebuilt (labels.py ships to the browser).

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git log --oneline -1
  ```
  Expected: one line `<sha> feat(labels): one module spells the learner-facing names`.

### Task 12: The grader and the shim say "Exercise A"; verify_units scans the grader's real output (R4 G5, R7 helper)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/verify_units.py` (imports; `_graded`; `verify_unit`; `verify`; new `course_ids`, `id_pattern`, `output_violations`)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_verify_units.py` (two new tests)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/zero2fsd/grade/_report.py` (imports; `_lines`; `exercise`)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/zero2fsd/grade/units/u0_1_1.py`, `u0_1_2.py`, `u0_1_3.py`, `u0_1_4.py` (one `LECTURE` constant each; every `hint=` line)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/playground/shim.py` (import; `_at_boundary`)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_grade.py` (two assertions; two new tests), `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/playground/test_shim.py` (two assertions)
- Rebuild + commit: `course/dist/playground/bundle.zip`, `course/dist/playground/manifest.json`, `course/dist/playground/shim.py`

**Interfaces:** `exercise(exercise_id, *, checked, lecture, section)`; registered function has `.lecture`/`.section`; `verify_units.course_ids(root) -> set[str]`, `verify_units.id_pattern(ids) -> re.Pattern`, `verify_units._graded(exercise_id, target) -> tuple[bool, str]`, `verify_units.verify_unit(root, unit, pattern)`. Output forms as in Part interfaces.

- [ ] **Step 1: Make verify_units read the grader's stdout and flag course ids (RED on the real course).** In `scripts/verify_units.py`:
  - add `import re` after `import os`;
  - replace the whole `_graded` function with

  ```python
  def course_ids(root: Path) -> set[str]:
      """Every unit id of the syllabus and every exercise id of a ready unit: the ids a learner never reads."""
      ids = set()
      for unit in syllabus_units(root):
          ids.add(unit["id"])
          if unit["status"] == "ready":
              try:
                  ids.update(ex["id"] for ex in load_yaml(root / "course" / unit["folder"] / "unit.yaml")["exercises"])
              except Exception:  # verify_unit reports this unit as "cannot load the unit"; the others are still scanned
                  pass
      return ids


  def id_pattern(ids: set[str]) -> re.Pattern:
      """An id as a whole token: not inside a word or a longer dotted number (a book's §10.1.1 holds no unit 0.1.1, and
      0.1.1.a is the exercise, never unit 0.1.1 followed by '.a'), longest id first."""
      alternatives = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
      return re.compile(rf"(?<![\w.])(?:{alternatives})(?!\w|\.\w)")


  def _graded(exercise_id, target) -> tuple[bool, str]:
      """The verdict and everything the grader printed: exactly what the learner reads."""
      with contextlib.redirect_stdout(io.StringIO()) as out:
          passed = check(exercise_id, target, show=False).passed
      return passed, out.getvalue()


  def output_violations(where: str, run: str, printed: str, pattern: re.Pattern) -> list[str]:
      return [f"{where}: the grader's output for the {run} shows the id {m.group()!r}: {line.strip()!r}"
              for line in printed.splitlines() for m in pattern.finditer(line)]
  ```

  - change the signature to `def verify_unit(root: Path, unit: dict, pattern: re.Pattern) -> list[str]:` and replace the block from `if skeleton is not None and _graded(ex["id"], skeleton):` to the end of the `for ex in exercises:` loop with

  ```python
          if skeleton is not None:
              passed, printed = _graded(ex["id"], skeleton)
              if passed:
                  violations.append(f"{where}: the untouched skeleton PASSES the grader")
              violations += output_violations(where, "skeleton", printed, pattern)
          if not hasattr(solution, ex["name"]):
              violations.append(f"{where}: solutions/course/{unit['id']}/solution.py does not define {ex['name']!r}")
          else:
              passed, printed = _graded(ex["id"], getattr(solution, ex["name"]))
              if not passed:
                  violations.append(f"{where}: the reference solution FAILS the grader")
              violations += output_violations(where, "solution", printed, pattern)
      return violations
  ```

  - replace `verify` with

  ```python
  def verify(root: Path) -> list[str]:
      pattern = id_pattern(course_ids(root))
      return [v for unit in syllabus_units(root, status="ready") for v in verify_unit(root, unit, pattern)]
  ```

  Run the gate on the real course: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/verify_units.py | tail -4`
  Expected (RED, real output): the last line is `21 violation(s)` — 7 exercises × (skeleton `✗ 0.1.n.x:` head line, skeleton `hint: lecture 0.1.n, …` line, solution `✓ 0.1.n.x:` head line); each VIOLATION line reads `… the grader's output for the skeleton|solution shows the id '0.1.…'`. If the count differs, paste the VIOLATION lines into your report and continue — the grader change in Step 3 must bring it to zero either way.

- [ ] **Step 2: Seeded tests for the stdout scan and the token boundaries (RED).** Append to `tests/zero2fsd/test_verify_units.py` (before the `@pytest.mark.slow` test):

  ```python
  def test_an_id_in_the_graders_output_is_a_violation(built_repo, monkeypatch):
      def chatty_check(exercise_id, target, show=False):  # a grader that names the exercise by id, as the old one did
          print(f"\N{CHECK MARK} {exercise_id}: looks fine")
          return verify_units.check(exercise_id, target, show=show)

      monkeypatch.setattr(verify_units, "check", chatty_check)
      violations = verify_units.verify(built_repo)
      assert len(violations) == 4 and all("grader's output" in v for v in violations)  # 2 exercises × (skeleton, solution)
      assert any("skeleton shows the id '0.1.1.a'" in v for v in violations) and any("solution shows the id '0.1.1.b'" in v for v in violations)


  @pytest.mark.parametrize(("text", "found"), [
      ("see unit 0.1.1.", ["0.1.1"]),
      ('check("0.1.1.a")', ["0.1.1.a"]),
      ("Szeliski §10.1.1 and 0.1.1.5", []),  # inside a longer dotted number
      ("v0.1.1 and 0.1.10", []),  # inside a word or a longer id
  ])
  def test_id_pattern_matches_whole_ids_only(text, found):
      assert verify_units.id_pattern({"0.1.1", "0.1.1.a", "0.1.2"}).findall(text) == found
  ```

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_verify_units.py -k "graders_output or id_pattern or correct_unit or every_ready_unit" 2>&1 | tail -4`
  Expected: `test_id_pattern_matches_whole_ids_only` 4 passed; `test_an_id_in_the_graders_output_is_a_violation` FAILS (the real grader also prints ids, so there are more than 4 violations); `test_a_correct_unit_has_no_violations…` and `test_every_ready_unit_in_the_repo_verifies` FAIL with the `grader's output … shows the id` violations. That is the red the grader change below turns green.

- [ ] **Step 3: The grader names the exercise by letter and the lecture by title.** In `zero2fsd/grade/_report.py`:
  - after `from ..car.runner import StepHook` add `from ..labels import exercise_name, lecture_section`;
  - replace `_lines` and `exercise` (everything from `def _lines(` to the end of the file) with

  ```python
  def _lines(failure: Failure, name: str, checked, hint) -> list[str]:
      fields = [("checked", failure.checked or checked),
                *((label, failure.shown[key]) for key, label in _SHOWN_LABELS.items() if key in failure.shown),
                ("hint", hint)]
      return [f"\N{BALLOT X} {name}: {failure.headline}", *(f"    {label + ':':<10}{fmt(value)}" for label, value in fields)]


  def exercise(exercise_id: str, *, checked: str, lecture: str, section: str):
      """Register a checker `(target, show, on_step) -> str` (what passed); it raises `Failure` for the first problem.
      The learner reads the exercise as its letter ("Exercise B") and a hint as `lecture "<title>", section "<## heading>"`."""
      name, hint = exercise_name(exercise_id), lecture_section(lecture, section)

      def register(checker):
          @functools.wraps(checker)
          def graded(target, show, on_step):
              try:
                  message = checker(target, show, on_step)
              except Failure as failure:
                  return Result(False, _lines(failure, name, checked, hint))
              return Result(True, [f"\N{CHECK MARK} {name}: {message}"])

          if exercise_id in EXERCISES:
              raise ValueError(f"duplicate exercise id {exercise_id!r}")
          graded.lecture, graded.section = lecture, section  # scripts/verify_units.py checks them against the course files
          EXERCISES[exercise_id] = graded
          return graded

      return register
  ```

  In each unit module, add the constant after the import block and rewrite every `hint=` keyword (the heading text stays exactly as it is):
  - `zero2fsd/grade/units/u0_1_1.py`: after `TOL = 1e-9` add `LECTURE = "Meet your car"`; `hint='lecture 0.1.1, section "Summarising lane error"',` → `lecture=LECTURE, section="Summarising lane error",`; `hint='lecture 0.1.1, section "Counting steps off the lane"',` → `lecture=LECTURE, section="Counting steps off the lane",`.
  - `u0_1_2.py`: after the line `from .._report import Failure, call, exercise, vs_limit` add a blank line and `LECTURE = "Images are arrays"`; `hint='lecture 0.1.2, section "Paint is a colour threshold"',` → `lecture=LECTURE, section="Paint is a colour threshold",`.
  - `u0_1_3.py`: after `from .._report import Failure, call, exercise, learner_code, vs_limit` add `LECTURE = "Your first perception box"`; `hint='lecture 0.1.3, section "From masks to a lane estimate"',` → `lecture=LECTURE, section="From masks to a lane estimate",`; `hint='lecture 0.1.3, section "Swap your perception into the car"',` → `lecture=LECTURE, section="Swap your perception into the car",`.
  - `u0_1_4.py`: after `from .._report import Failure, call, exercise, fmt, is_real` add `LECTURE = "Closing the loop"`; `hint='lecture 0.1.4, section "The P controller"',` → `lecture=LECTURE, section="The P controller",`; `hint='lecture 0.1.4, section "Closing the loop"',` → `lecture=LECTURE, section="Closing the loop",`.

  Check nothing else passes `hint=`: `grep -rn "hint=" /Users/vivek/Antigravity/self-driving-ai-course-week1/zero2fsd` → expected: no output.

  In `playground/shim.py`: after `from zero2fsd.grade._report import Failure` add `from zero2fsd.labels import exercise_name`; in `_at_boundary` replace both `raise TypeError(f"{exercise_id}: {exc.headline}") from exc` and `raise TypeError(f"{exercise_id}: {exc}") from exc` with `raise TypeError(f"{exercise_name(exercise_id)}: {exc.headline}") from exc` and `raise TypeError(f"{exercise_name(exercise_id)}: {exc}") from exc`.

- [ ] **Step 4: Tests follow the learner's words.** The tests keep computing the letter independently (`exercise_id[-1].upper()`), never through `labels`.
  - `tests/zero2fsd/test_grade.py`: `assert capsys.readouterr().out.startswith(f"\033[32m\N{CHECK MARK} {exercise_id}:")` → `assert capsys.readouterr().out.startswith(f"\033[32m\N{CHECK MARK} Exercise {exercise_id[-1].upper()}:")`; `assert f"\N{CHECK MARK} {exercise_id}:" in capsys.readouterr().out` → `assert f"\N{CHECK MARK} Exercise {exercise_id[-1].upper()}:" in capsys.readouterr().out`. Add `from _course import REPO, syllabus_units` after `import pytest` (the conftest puts `scripts/` on `sys.path`; test_verify_units already imports `_course` this way), and append:

  ```python
  def test_every_hint_names_the_lesson_by_its_syllabus_title():
      titles = {unit["id"]: unit["title"] for unit in syllabus_units(REPO)}
      for exercise_id, graded in EXERCISES.items():
          assert graded.lecture == titles[exercise_id.rsplit(".", 1)[0]], exercise_id


  def test_a_failure_points_at_the_lecture_by_title_and_section(capsys):
      check("0.1.1.a", lambda lat: [1.0, 2.0, 3.0], show=False)
      assert 'hint:     lecture "Meet your car", section "Summarising lane error"' in capsys.readouterr().out
  ```

  - `tests/playground/test_shim.py`: `assert message.startswith(f"{exercise}: ") and expected in message` → `assert message.startswith(f"Exercise {exercise[-1].upper()}: ") and expected in message` (rest of the line unchanged); `assert str(caught.value) == f"0.1.2.a: {expected}"` → `assert str(caught.value) == f"Exercise A: {expected}"`.
  - Confirm no other test or page still expects an id prefix: `grep -rn '{exercise_id}:\|{exercise}: \|"0\.1\.[0-9]\.[ab]: ' /Users/vivek/Antigravity/self-driving-ai-course-week1/tests /Users/vivek/Antigravity/self-driving-ai-course-week1/playground` → expected: no output.

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_grade.py tests/zero2fsd/test_verify_units.py tests/playground/test_shim.py 2>&1 | tail -3`
  Expected: all passed, 0 failed (the Step 2 reds are green now). Then `PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/verify_units.py | tail -1` → expected `all exercises: skeleton fails, solution passes`.

- [ ] **Step 5: Rebuild the bundle (grader + shim changed) and commit.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py && git status --short course`
  Expected: exactly ` M course/dist/playground/bundle.zip`, ` M course/dist/playground/manifest.json`, ` M course/dist/playground/shim.py`. Then:
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add scripts/verify_units.py tests/zero2fsd/test_verify_units.py zero2fsd/grade/_report.py zero2fsd/grade/units/u0_1_1.py zero2fsd/grade/units/u0_1_2.py zero2fsd/grade/units/u0_1_3.py zero2fsd/grade/units/u0_1_4.py playground/shim.py tests/zero2fsd/test_grade.py tests/playground/test_shim.py course/dist/playground/bundle.zip course/dist/playground/manifest.json course/dist/playground/shim.py && git commit -q -F - <<'MSG'
  feat(grade): name exercises by letter and lectures by title; verify_units scans the grader's output for ids

  exercise(..., lecture=, section=) builds every hint through labels.lecture_section; tick/cross lines and shim
  TypeErrors say "Exercise A". verify_units captures the skeleton and solution runs' stdout and reports any course id
  in it (21 real violations before this change, 0 after). Bundle rebuilt.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git log --oneline -1
  ```
  Expected: one line `<sha> feat(grade): name exercises by letter …`.

### Task 13: Learner text names lessons, letters and sections; the check action is "Check my work" (R4, R7, R9)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/build_labs.py` (import block; new `CHECK_CALL`, `labelled`; one token in `compile_lab` — Part A owns the rest of that function)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/playground/index.html` (the `#check` button only; `#reset`/`#restore-draft` are Part C's)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_build_labs.py`, `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/playground/test_build.py` (one new test each)
- Modify (by the script in Step 3): `course/0.1.{1,2,3,4}-*/lecture.md`, `lab.py`, `unit.yaml` — 78 exact substitutions
- Rebuild + commit: `course/0.1.{1,2,3,4}-*/lab.ipynb`, `course/dist/course.json`, `course/dist/playground/index.html`, `course/dist/playground/manifest.json`

**Interfaces:** `build_labs.labelled(cell: Cell) -> str`; `build_labs.CHECK_CALL`; the text contract in Part interfaces. Parts A (Task 2 moved cells in 0.1.2 and 0.1.4 lab.py, renamed colours in figure prose) and C (C4 playground links) have already edited these files: every substitution below is an exact substring that must occur once; the script stops and names any that does not.

- [ ] **Step 1: Tests for the label (RED).** Append to `tests/zero2fsd/test_build_labs.py` (`import nbformat` is already at the top of the file from Part A's tests; add it if not):

  ```python
  def test_every_check_cell_starts_with_check_my_work_and_no_other_cell_does(mini_repo):
      cells = nbformat.reads(build_labs.compile_lab(mini_repo / "course" / "0.1.1-mini" / "lab.py"), as_version=4).cells
      labelled = [c.source.split("\n", 1)[1] for c in cells if c.source.startswith("# Check my work\n")]
      assert labelled == ['check("0.1.1.a", lane_error_stats)', 'check("0.1.1.b", steps_off_lane)']
      assert sum("Check my work" in c.source for c in cells) == 2  # the build adds it to the check cells only; lab.py never spells it
  ```

  Append to `tests/playground/test_build.py`:

  ```python
  def test_the_check_button_is_named_check_my_work():
      from pathlib import Path

      from zero2fsd.labels import CHECK_MY_WORK

      page = (Path(__file__).resolve().parents[2] / "playground" / "index.html").read_text(encoding="utf-8")
      assert f'<button id="check" type="button" disabled>{CHECK_MY_WORK}</button>' in page
  ```

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_build_labs.py tests/playground/test_build.py -k "check_my_work" 2>&1 | tail -3`
  Expected: `2 failed` (`labelled == []`; the button still reads `Check`).

- [ ] **Step 2: Implement the label (GREEN).** In `scripts/build_labs.py`:
  - after the `from _course import …` line add `from zero2fsd.labels import CHECK_MY_WORK` (the `_course` import above it puts the repo root on `sys.path`, so this resolves to the worktree's copy);
  - after the `LECTURE_FIGURE = re.compile(...)` line add
    ```python
    CHECK_CALL = re.compile(r"^check\(", re.M)  # a code cell that grades the learner's answer
    ```
  - directly above `def compile_lab(` insert
    ```python
    def labelled(cell) -> str:
        """A code cell that grades the learner's answer starts with the name of the action; every other cell is as authored."""
        return f"# {CHECK_MY_WORK}\n{cell.source}" if cell.kind == "code" and CHECK_CALL.search(cell.source) else cell.source
    ```
  - inside `compile_lab`, the expression that supplies a CODE cell's source is `cell.source` (the `else cell.source` branch of the `source = … if cell.kind == "markdown" else cell.source` statement, unless Part A reshaped it — then whatever expression now feeds a code cell's source): change that one token to `labelled(cell)`. Do not touch the figure code or the `check_lecture_order` call.

  In `playground/index.html` replace `<button id="check" type="button" disabled>Check</button>` with `<button id="check" type="button" disabled>Check my work</button>`.

  Run the Step 1 command. Expected: `2 passed`.

- [ ] **Step 3: The 78 text substitutions.** Write the following to `/private/tmp/claude-501/c1-b3-rename.py` and run it twice: first `--check` (counts only, writes nothing), then without a flag (applies). Each `old` must occur exactly once in its file; if the script prints a `… occurrences of …` line, Part A's or C's edits moved or reworded that text: open the file, apply the same intent by hand on the current wording, re-run `--check` until it reports `0 problem(s)`, then apply.

  ```python
  """Task 13: every learner-visible dotted id becomes a title or an exercise letter, and every section citation takes the
  form `section "<Heading>"`.  Each `old` must occur exactly once in its file; `--check` only counts, no flag applies."""
  import sys
  from pathlib import Path

  ROOT = Path("/Users/vivek/Antigravity/self-driving-ai-course-week1/course")
  U1, U2, U3, U4 = "0.1.1-meet-your-car", "0.1.2-images-are-arrays", "0.1.3-your-first-perception-box", "0.1.4-closing-the-loop"
  TUTOR_TAIL = "naming these lecture sections: "
  TUTOR_NEW = "naming them as the lecture does: "

  EDITS = {
      f"{U1}/lecture.md": [
          ('starting with perception in unit 0.1.3 and the controller in unit 0.1.4.', 'starting with perception in "Your first perception box" and the controller in "Closing the loop".'),
          ('(you will meet steering gains in unit 0.1.4)', '(you will meet steering gains in "Closing the loop")'),
          ('prepares you for unit 0.1.2.', 'prepares you for "Images are arrays".'),
          ('come back to the steering law after unit 0.1.4.', 'come back to the steering law after "Closing the loop".'),
      ],
      f"{U1}/lab.py": [
          ('# # Lab 0.1.1: Meet your car', '# # Lab: Meet your car'),
          ('Unit 0.1.4 finds out why.', '"Closing the loop" finds out why.'),
          ('### 0.1.1.a `lane_error_stats`', '### Exercise A · `lane_error_stats`'),
          ('### 0.1.1.b `steps_off_lane`', '### Exercise B · `steps_off_lane`'),
          ('In unit 0.1.2 you open the camera picture', 'In "Images are arrays" you open the camera picture'),
          ('(see "The car you were handed")', '(see section "The car you were handed")'),
          ('(lecture: "Lateral error")', '(section "Lateral error")'),
          ('example from "Summarising lane error", one NumPy', 'example from section "Summarising lane error", one NumPy'),
          ('(lecture: "Arrays, not loops")', '(section "Arrays, not loops")'),
          ('Drive B from "Why RMS punishes big errors"', 'Drive B from section "Why RMS punishes big errors"'),
          ('(Lecture: "The signed-mean trap".)', '(section "The signed-mean trap".)'),
          ('Write the three summaries from "Summarising lane error" with NumPy', 'Write the three summaries from section "Summarising lane error" with NumPy'),
          ('(lecture: "Counting steps off the lane")', '(section "Counting steps off the lane")'),
      ],
      f"{U1}/unit.yaml": [
          ('passes check(\\"0.1.1.a\\") in the lab', 'passes Exercise A\'s \\"Check my work\\" in the lab'),
          ('passes check(\\"0.1.1.b\\") in the lab', 'passes Exercise B\'s \\"Check my work\\" in the lab'),
          ('for unit 0.1.1 "Meet your car" of a', 'for the lesson "Meet your car" of a'),
          (TUTOR_TAIL + '"Lateral error", "Summarising lane error", "The signed-mean trap", "Why RMS punishes big errors", "Counting steps off the lane", "Arrays, not loops".',
           TUTOR_NEW + 'section "Lateral error", section "Summarising lane error", section "The signed-mean trap", section "Why RMS punishes big errors", section "Counting steps off the lane", section "Arrays, not loops".'),
      ],
      f"{U2}/lecture.md": [
          ('you will read it properly in unit 0.1.4.', 'you will read it properly in "Closing the loop".'),
      ],
      f"{U2}/lab.py": [
          ('# # Lab 0.1.2: Images are arrays', '# # Lab: Images are arrays'),
          ('### 0.1.2.a `paint_masks`', '### Exercise A · `paint_masks`'),
          ('In unit 0.1.3 you turn these two masks', 'In "Your first perception box" you turn these two masks'),
          ('(lecture: "Paint is a colour threshold"). These are the frames', '(section "Paint is a colour threshold"). These are the frames'),
          ('to the lecture ("A frame is a grid of numbers")', 'to the lecture (section "A frame is a grid of numbers")'),
          ('the whole method from "Paint is a colour threshold", one step', 'the whole method from section "Paint is a colour threshold", one step'),
          ('nearest neighbour ("Paint is a colour threshold"). For paint', 'nearest neighbour (section "Paint is a colour threshold"). For paint'),
          ('perception box (lecture: "Paint is a colour threshold"). Mind the', 'perception box (section "Paint is a colour threshold"). Mind the'),
          ('last section of the lecture, "The uint8 wraparound trap".', 'last section of the lecture, section "The uint8 wraparound trap".'),
      ],
      f"{U2}/unit.yaml": [
          ('passes check(\\"0.1.2.a\\") in the lab', 'passes Exercise A\'s \\"Check my work\\" in the lab'),
          ('for unit 0.1.2 "Images are arrays" of a', 'for the lesson "Images are arrays" of a'),
          (TUTOR_TAIL + '"A frame is a grid of numbers", "Colour is a point in space", "Paint is a colour threshold", "Scoring a mask with IoU", "Why exact equality fails", "The uint8 wraparound trap".',
           TUTOR_NEW + 'section "A frame is a grid of numbers", section "Colour is a point in space", section "Paint is a colour threshold", section "Scoring a mask with IoU", section "Why exact equality fails", section "The uint8 wraparound trap".'),
      ],
      f"{U3}/lecture.md": [
          ('Your `paint_masks` from unit 0.1.2 finds', 'Your `paint_masks` from "Images are arrays" finds'),
          ('rows 0 to 75 of the 0.1.2 frame were all sky', 'rows 0 to 75 of the "Images are arrays" frame were all sky'),
          ('Unit 0.1.4 shows what this bias does to the car on `curvy`.', '"Closing the loop" shows what this bias does to the car on `curvy`.'),
          ('Check `0.1.3.b` drives that car around', "Exercise B's check drives that car around"),
      ],
      f"{U3}/lab.py": [
          ('# # Lab 0.1.3: Your first perception box', '# # Lab: Your first perception box'),
          ('from unit 0.1.2, replace the import', 'from "Images are arrays", replace the import'),
          ('### 0.1.3.a and 0.1.3.b `estimate_lane`', '### Exercise A and Exercise B · `estimate_lane`'),
          ('Check `0.1.3.a` renders 40 frames', "Exercise A's check renders 40 frames"),
          ('Check `0.1.3.b` puts your function in the car', "Exercise B's check puts your function in the car"),
          ('Once 0.1.3.a passes, drive.', 'Once Exercise A passes, drive.'),
          ('only samples straight stretches. Unit 0.1.4 shows what this bias does to the car.', 'only samples straight stretches. "Closing the loop" shows what this bias does to the car.'),
          ('In unit 0.1.4 you write the steering controller,', 'In "Closing the loop" you write the steering controller,'),
          ('the worked example of "From masks to a lane estimate". Then it does the worked example of "Rows are distances": how',
           'the worked example of section "From masks to a lane estimate". Then it does the worked example of section "Rows are distances": how'),
          ('(lecture: "Metres per pixel").', '(section "Metres per pixel").'),
          ('the misconception of\n# "Metres per pixel". A column', 'the misconception of\n# section "Metres per pixel". A column'),
          ('Step 2 of the algorithm in "From masks to a lane estimate", for one line', 'Step 2 of the algorithm in section "From masks to a lane estimate", for one line'),
          ('the four steps of "From masks to a lane estimate", plus the rule of "What valid=False is for".',
           'the four steps of section "From masks to a lane estimate", plus the rule of section "What valid=False is for".'),
          ('(lecture: "Swap your perception into the car")', '(section "Swap your perception into the car")'),
          ('(lecture: "What valid=False is for"). Count the paint', '(section "What valid=False is for"). Count the paint'),
          ('`curvy` (lecture:\n# "Straight roads only").', '`curvy` (section\n# "Straight roads only").'),
          ('the opposite sign ("From masks to a lane estimate": both minus signs', 'the opposite sign (section "From masks to a lane estimate": both minus signs'),
      ],
      f"{U3}/unit.yaml": [
          ('passes check(\\"0.1.3.a\\") in the lab', 'passes Exercise A\'s \\"Check my work\\" in the lab'),
          ('passes check(\\"0.1.3.b\\"), the straight-roads', 'passes Exercise B\'s \\"Check my work\\", the straight-roads'),
          ('for unit 0.1.3 "Your first perception box" of a', 'for the lesson "Your first perception box" of a'),
          (TUTOR_TAIL + '"The horizon", "Rows are distances", "Metres per pixel", "From masks to a lane estimate", "What valid=False is for", "Straight roads only", "Swap your perception into the car".',
           TUTOR_NEW + 'section "The horizon", section "Rows are distances", section "Metres per pixel", section "From masks to a lane estimate", section "What valid=False is for", section "Straight roads only", section "Swap your perception into the car".'),
      ],
      f"{U4}/lecture.md": [
          ("you get unit 0.1.3's positive feedback.", 'you get the positive feedback of "Your first perception box".'),
          ("The cause is unit 0.1.3's chord bias:", 'The cause is the chord bias of "Your first perception box":'),
          ('Check `0.1.4.b` requires a `Car`', "Exercise B's check requires a `Car`"),
      ],
      f"{U4}/lab.py": [
          ('# # Lab 0.1.4: Closing the loop', '# # Lab: Closing the loop'),
          ('## 3. Graded exercise 0.1.4.a `p_steer`', '## 3. Exercise A (graded) · `p_steer`'),
          ('## 5. Graded exercise 0.1.4.b `my_car`', '## 5. Exercise B (graded) · `my_car`'),
          ('paste your `estimate_lane` from lab 0.1.3 into the next cell', 'paste your `estimate_lane` from the "Your first perception box" lab into the next cell'),
          ('The cell sees the same names as lab 0.1.3:', 'The cell sees the same names as that lab:'),
          ('Check `0.1.4.b` needs it to complete', "Exercise B's check needs it to complete"),
          ('(lecture: "Open loop and closed loop"). Last, it sets', '(section "Open loop and closed loop"). Last, it sets'),
          ('The worked example of "Two errors, two gains": the car', 'The worked example of section "Two errors, two gains": the car'),
          ('(Lecture: "Why curves need error".)', '(section "Why curves need error".)'),
          ('Write the P law of "The P controller" and "Two errors, two gains" as a function', 'Write the P law of section "The P controller" and section "Two errors, two gains" as a function'),
          ('(lecture: "Tuning the gains"). It also counts', '(section "Tuning the gains"). It also counts'),
          ('(lecture: "Closing the loop"). `my_controller` returns', '(section "Closing the loop"). `my_controller` returns'),
      ],
      f"{U4}/unit.yaml": [
          ('passes check(\\"0.1.4.a\\") in the lab', 'passes Exercise A\'s \\"Check my work\\" in the lab'),
          ('passes check(\\"0.1.4.b\\") in the lab', 'passes Exercise B\'s \\"Check my work\\" in the lab'),
          ('for unit 0.1.4 "Closing the loop" of a', 'for the lesson "Closing the loop" of a'),
          (TUTOR_TAIL + '"Open loop and closed loop", "The P controller", "Two errors, two gains", "Tuning the gains", "Why curves need error", "Why your car leaves curvy", "Closing the loop", "A 1988 car that learned to steer".',
           TUTOR_NEW + 'section "Open loop and closed loop", section "The P controller", section "Two errors, two gains", section "Tuning the gains", section "Why curves need error", section "Why your car leaves curvy", section "Closing the loop", section "A 1988 car that learned to steer".'),
      ],
  }


  def main(check_only: bool) -> int:
      bad = 0
      for rel, edits in EDITS.items():
          path = ROOT / rel
          text = path.read_text(encoding="utf-8")
          for old, new in edits:
              n = text.count(old)
              if n != 1:
                  bad += 1
                  print(f"{rel}: {n} occurrences of {old[:60]!r}")
              text = text.replace(old, new)
          if not check_only and not bad:
              path.write_text(text, encoding="utf-8")
      total = sum(len(e) for e in EDITS.values())
      print(f"{total} edits in {len(EDITS)} files; {bad} problem(s); {'checked only' if check_only else 'applied'}")
      return 1 if bad else 0


  if __name__ == "__main__":
      sys.exit(main("--check" in sys.argv[1:]))
  ```

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -B /private/tmp/claude-501/c1-b3-rename.py --check && /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -B /private/tmp/claude-501/c1-b3-rename.py`
  Expected: `78 edits in 12 files; 0 problem(s); checked only` then `78 edits in 12 files; 0 problem(s); applied` (the planner dry-ran this list against the tree at 335491a: 78/78 unique). Then confirm the old citation forms are gone and no course id is left in prose: `grep -rnE '\((lecture|Lecture): "|unit 0\.1\.[0-9]|lab 0\.1\.[0-9]|Lab 0\.1\.[0-9]|### 0\.1\.|exercise 0\.1\.' course/*/lecture.md course/*/lab.py course/*/unit.yaml` → expected: no output.

- [ ] **Step 4: Rebuild the notebooks and the bundle.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_labs.py && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/build_course_bundle.py && git status --short course`
  Expected: four `built course/…/lab.ipynb` lines, two `wrote …` lines, then ` M` for the four `lecture.md`, four `lab.py`, four `lab.ipynb`, four `unit.yaml`, `course/dist/course.json`, `course/dist/playground/index.html`, `course/dist/playground/manifest.json` — 19 files. `exercises.json` must NOT appear (no `task:` text held an id); if it does, read its diff before going on. Spot-check the Colab label: `grep -c '"# Check my work\\n",' course/0.1.1-meet-your-car/lab.ipynb` → `2`; `course/0.1.3-your-first-perception-box/lab.ipynb` → `2`; 0.1.2 → `1`; 0.1.4 → `2`.

- [ ] **Step 5: Tests, including the real-course gates that exist so far.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_build_labs.py tests/zero2fsd/test_bundle.py tests/zero2fsd/test_verify_units.py tests/playground/test_build.py 2>&1 | tail -3`
  Expected: all passed, 0 failed (`test_committed_notebooks_are_in_sync_with_their_lab_py`, the course.json/dist freshness tests and `test_every_ready_unit_in_the_repo_verifies` included).

- [ ] **Step 6: Commit.**
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add scripts/build_labs.py playground/index.html tests/zero2fsd/test_build_labs.py tests/playground/test_build.py course/0.1.1-meet-your-car course/0.1.2-images-are-arrays course/0.1.3-your-first-perception-box course/0.1.4-closing-the-loop course/dist/course.json course/dist/playground/index.html course/dist/playground/manifest.json && git status --short && git commit -q -F - <<'MSG'
  feat(course): learner text names lessons, exercise letters and sections; the check action is "Check my work"

  78 substitutions across the four units' lecture.md, lab.py and unit.yaml: no dotted id in prose, every citation in
  the form section "<Heading>", proofs name Exercise A's "Check my work". build_labs prepends "# Check my work" to
  every check cell; the playground button says the same. Notebooks, course.json and the playground page rebuilt.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git log --oneline -1
  ```
  Expected: `git status --short` prints nothing before the commit (everything staged; figures/ are untouched), then one `<sha> feat(course): …` line.

### Task 14: verify_units scans everything a learner reads for ids and checks every section citation against the lecture (R4 G5, R7 class test)

**Files:**
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/_course.py` (new `SECTION`, `lecture_sections`)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/scripts/verify_units.py` (docstring; import line; new `LINK_TARGET`, `SECTION_CITE`, `prose`, `yaml_strings`, `learner_texts`, `text_violations`; `verify_unit`; `verify`; `main`'s green message)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_verify_units.py` (import; five new tests)
- Modify: `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/lecture.md` (one heading), `…/0.1.1-mini/quiz.yaml` (two slugs), `/Users/vivek/Antigravity/self-driving-ai-course-week1/tests/zero2fsd/test_bundle.py` (the two `edit(...)` anchors that spell that slug)
- No rebuild: nothing here feeds `course/dist` or the notebooks.

**Interfaces:** `_course.lecture_sections(text: str) -> set[str]` (the `## ` headings outside code fences, as written); `verify_units.verify_unit(root, unit, ids, pattern)`; `verify_units.learner_texts(folder, ids) -> list[tuple[str, str]]` with sources `"lecture.md"`, `"lab.py cell c000"`, `"unit.yaml"`, `"quiz.yaml"`, `"cards.yaml"`; violation forms `<unit> <source>:<line>: shows the id '<id>': '<line>'`, `<unit> <source>:<line>: cites section '<X>', not a ## heading of lecture.md (headings: [...])`, `<unit> <ex> (<name>): the grader's hint names section '<X>', not a ## heading of lecture.md (headings: [...])`. The CLI's green line becomes `all units: skeleton fails, solution passes, learner text clean` (the substring `test_cli_exit_codes_and_report` asserts is kept).

- [ ] **Step 1: Seeded tests (RED).** In `tests/zero2fsd/test_verify_units.py` change `from _course import REPO` to `from _course import REPO, lecture_sections`, then append before the `@pytest.mark.slow` test:

  ```python
  def test_lecture_sections_are_the_level_two_headings_outside_code_fences():
      text = "# Title\n\n## Lateral error\n\n```python\n## not a heading\n# neither\n```\n\n### Sub-heading\n\n## Arrays, not loops  \n"
      assert lecture_sections(text) == {"Lateral error", "Arrays, not loops"}


  @pytest.mark.parametrize(("rel", "old", "new", "source", "shown"), [
      (f"{UNIT}/lecture.md", "Lateral error is how far", "Lateral error (unit 0.1.1) is how far", "lecture.md:5:", "0.1.1"),
      (f"{UNIT}/lab.py", "# A fixture lab:", "# A fixture lab for 0.1.1.a:", "lab.py cell c000:", "0.1.1.a"),
      (f"{UNIT}/unit.yaml", "proof: lane_error_stats passes its tests", "proof: 0.1.1.a passes in the lab", "unit.yaml:", "0.1.1.a"),
      (f"{UNIT}/quiz.yaml", 'explain: "The absolute values are both 0.5."', 'explain: "The absolute values are both 0.5, as in 0.1.1.b."', "quiz.yaml:", "0.1.1.b"),
      (f"{UNIT}/cards.yaml", 'back: "The sideways distance from lane centre."', 'back: "The sideways distance from lane centre (unit 0.1.1)."', "cards.yaml:", "0.1.1"),
  ])
  def test_a_course_id_in_learner_text_is_a_violation(built_repo, rel, old, new, source, shown):
      edit(built_repo, rel, old, new)
      violations = verify_units.verify(built_repo)
      assert len(violations) == 1 and source in violations[0] and f"shows the id {shown!r}" in violations[0], violations


  def test_ids_in_code_fences_link_targets_and_longer_numbers_are_not_learner_text(built_repo):
      edit(built_repo, f"{UNIT}/lecture.md", "# this comment is not a heading", 'check("0.1.1.a")  # code, not prose')
      edit(built_repo, f"{UNIT}/lecture.md", "Lateral error is how far the car is from lane centre.",
           "Lateral error is how far the car is from lane centre: see [Exercise A](../playground/index.html?exercise=0.1.1.a), Szeliski §10.1.1 and version 0.1.1.5.")
      assert verify_units.verify(built_repo) == []


  def test_a_section_citation_must_be_a_heading_of_the_lecture(built_repo):
      edit(built_repo, f"{UNIT}/cards.yaml", 'back: "The sideways distance from lane centre."', 'back: "See section \\"Counting steps off the lane\\"."')
      assert verify_units.verify(built_repo) == []
      edit(built_repo, f"{UNIT}/cards.yaml", "Counting steps off the lane", "Lane error")
      violations = verify_units.verify(built_repo)
      assert len(violations) == 1 and "cards.yaml:1: cites section 'Lane error', not a ## heading" in violations[0], violations


  def test_a_grader_hint_must_cite_a_heading_of_the_lecture(built_repo):
      edit(built_repo, f"{UNIT}/lecture.md", "## Summarising lane error", "## Measuring lane error")  # the real 0.1.1.a grader cites "Summarising lane error"
      violations = verify_units.verify(built_repo)
      assert violations == ["0.1.1 0.1.1.a (lane_error_stats): the grader's hint names section 'Summarising lane error', not a ## heading of lecture.md (headings: ['Counting steps off the lane', 'Measuring lane error'])"]
  ```

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_verify_units.py -k "lecture_sections or learner_text or section_citation or grader_hint" 2>&1 | tail -3`
  Expected: `ImportError` at collection (`lecture_sections` does not exist yet) — the whole file errors. That is the red.

- [ ] **Step 2: `lecture_sections` and the text scanner (GREEN for the seeded tests).**
  In `scripts/_course.py`, after the `FENCE = re.compile(...)` line add
  ```python
  SECTION = re.compile(r"^## +(.+?) *$", re.M)  # a level-two heading: the unit of citation (R7)
  ```
  and directly above `def syllabus_units(` insert
  ```python
  def lecture_sections(text: str) -> set[str]:
      """The `## ` headings of a lecture outside code fences, spelled as written: what `section "<Heading>"` may cite."""
      return set(SECTION.findall(FENCE.sub("", text)))
  ```

  In `scripts/verify_units.py`:
  - docstring: after the first paragraph (the line ending `a green tick it did not earn.`) add a paragraph
    ```
    It also reads what the learner reads: the grader's own output for both runs, lecture prose, lab markdown cells
    and the strings of unit/quiz/cards.yaml must show no course id, and every `section "<Heading>"` they or the
    grader's hints cite must be a `## ` heading of that unit's lecture.md.
    ```
  - import line: `from _course import REPO, load_solution, load_yaml, syllabus_units` → `from _course import REPO, lecture_sections, load_solution, load_yaml, parse_lab, syllabus_units`;
  - after the `id_pattern` function (from B2) add

  ```python
  LINK_TARGET = re.compile(r"\]\([^)]*\)")  # a markdown link or image target: a URL, never prose
  SECTION_CITE = re.compile(r'section "([^"]+)"')  # the one form every citation of a lecture section takes (R7)


  def prose(markdown: str) -> str:
      """What a learner reads: fenced code blanked line for line (line numbers hold), link targets dropped, inline code kept."""
      blanked = FENCE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), markdown)
      return LINK_TARGET.sub("]()", blanked)


  def yaml_strings(node, ids: set[str]):
      """Every string of a yaml document except one that IS an id: `id:`, `prerequisites:` and `exercises[].id` are references."""
      if isinstance(node, str):
          if node not in ids:
              yield node
      elif isinstance(node, dict):
          for value in node.values():
              yield from yaml_strings(value, ids)
      elif isinstance(node, list):
          for value in node:
              yield from yaml_strings(value, ids)


  def learner_texts(folder: Path, ids: set[str]) -> list[tuple[str, str]]:
      """(source, text) for everything in a unit folder a learner reads outside code."""
      texts = [("lecture.md", prose((folder / "lecture.md").read_text(encoding="utf-8")))]
      cells = parse_lab((folder / "lab.py").read_text(encoding="utf-8"))
      texts += [(f"lab.py cell c{i:03d}", prose(cell.source)) for i, cell in enumerate(cells) if cell.kind == "markdown"]
      for name in ("unit.yaml", "quiz.yaml", "cards.yaml"):
          texts += [(name, text) for text in yaml_strings(load_yaml(folder / name), ids)]
      return texts


  def text_violations(where: str, texts: list[tuple[str, str]], pattern: re.Pattern, sections: set[str]) -> list[str]:
      out = []
      for source, text in texts:
          for n, line in enumerate(text.splitlines(), 1):
              out += [f"{where} {source}:{n}: shows the id {m.group()!r}: {line.strip()!r}" for m in pattern.finditer(line)]
              out += [f"{where} {source}:{n}: cites section {cited!r}, not a ## heading of lecture.md (headings: {sorted(sections)})"
                      for cited in SECTION_CITE.findall(line) if cited not in sections]
      return out
  ```
  (`FENCE` comes from `_course`: extend the import line to `from _course import FENCE, REPO, lecture_sections, load_solution, load_yaml, parse_lab, syllabus_units`.)

  - `verify_unit`: signature → `def verify_unit(root: Path, unit: dict, ids: set[str], pattern: re.Pattern) -> list[str]:`; inside the `try:` block, after the `notebook, solution = …` line add
    ```python
            sections = lecture_sections((folder / "lecture.md").read_text(encoding="utf-8"))
            texts = learner_texts(folder, ids)
    ```
    replace `violations = []` with `violations = text_violations(unit["id"], texts, pattern, sections)`; and after the `continue` of the `the grader has no exercise with this id` check insert
    ```python
            if EXERCISES[ex["id"]].section not in sections:
                violations.append(f"{where}: the grader's hint names section {EXERCISES[ex['id']].section!r}, not a ## heading of lecture.md (headings: {sorted(sections)})")
    ```
  - `verify`:
    ```python
    def verify(root: Path) -> list[str]:
        ids = course_ids(root)
        pattern = id_pattern(ids)
        return [v for unit in syllabus_units(root, status="ready") for v in verify_unit(root, unit, ids, pattern)]
    ```
  - `main`: `"all exercises: skeleton fails, solution passes"` → `"all units: skeleton fails, solution passes, learner text clean"`.

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_verify_units.py --deselect tests/zero2fsd/test_verify_units.py::test_every_ready_unit_in_the_repo_verifies 2>&1 | tail -6`
  Expected: the five `test_a_course_id_in_learner_text_is_a_violation` cases, `test_ids_in_code_fences…`, `test_lecture_sections…` and `test_a_section_citation…` pass; **`test_a_correct_unit_has_no_violations_and_never_runs_untagged_cells` FAILS** with `the grader's hint names section 'Summarising lane error', not a ## heading of lecture.md (headings: ['Counting steps off the lane', 'Measuring lane error'])` — the new check catching the fixture's own mismatch with the real 0.1.1.a grader — and `test_a_grader_hint_must_cite_a_heading_of_the_lecture` fails on `edit`'s assert (`'## Summarising lane error' not in …`). Both go green in Step 3; `test_cli_exit_codes_and_report` passes (its substring survives).

- [ ] **Step 3: The fixture's lecture heading follows the grader it is graded by.**
  - `tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/lecture.md`: `## Measuring lane error` → `## Summarising lane error`.
  - `tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/quiz.yaml`: both `section: measuring-lane-error` → `section: summarising-lane-error` (the bundle build checks quiz slugs against the headings).
  - `tests/zero2fsd/test_bundle.py`: both occurrences of `"section: measuring-lane-error\n- id: q2"` (the `test_bad_content_fails_loudly` row with message `not a heading of the lecture`, and the `old` in `test_a_quiz_section_may_not_be_a_heading_that_only_appears_in_a_code_fence`) → `"section: summarising-lane-error\n- id: q2"`. The `heading_slug` row `("Measuring lane error", "measuring-lane-error")` near the top of that file is a slug-function literal, not the fixture: leave it.
  - Check nothing else spells the old heading: `grep -rn 'measuring-lane-error\|Measuring lane error' tests/ scripts/ zero2fsd/` → expected: only the `test_bundle.py` `heading_slug` literal line.

  Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python -m pytest -p no:cacheprovider -q tests/zero2fsd/test_verify_units.py tests/zero2fsd/test_bundle.py tests/zero2fsd/test_build_labs.py 2>&1 | tail -3`
  Expected: all passed, 0 failed, including `test_every_ready_unit_in_the_repo_verifies` (the real course under the full gate).

- [ ] **Step 4: The gate on the real course, as the CLI.** Run: `cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && PYTHONPATH=/Users/vivek/Antigravity/self-driving-ai-course-week1 PYTHONPYCACHEPREFIX=/private/tmp/claude-501/c1-pycache MPLBACKEND=Agg /Users/vivek/Antigravity/self-driving-ai-course/.venv/bin/python scripts/verify_units.py; echo "exit $?"`
  Expected: `all units: skeleton fails, solution passes, learner text clean` then `exit 0` (the planner's dry run of this scanner on the tree with Task 13's edits applied: 47 violations before B3, 0 after; the seven grader sections all exist as `## ` headings). If a `VIOLATION` line appears, it names the file, line and id or citation: it comes from text Part A or C wrote after the planner's run (for example a playground link's text). Fix it with the text contract (title in quotes, `Exercise A`, `section "<Heading>"`), rebuild what that file feeds (`scripts/build_labs.py` for a lab.py, `scripts/build_course_bundle.py` for a lecture or yaml), add those artifacts to this commit, and re-run until green. Do not widen the scanner's exemptions.

- [ ] **Step 5: Commit.**
  ```
  cd /Users/vivek/Antigravity/self-driving-ai-course-week1 && git add scripts/_course.py scripts/verify_units.py tests/zero2fsd/test_verify_units.py tests/zero2fsd/test_bundle.py tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/lecture.md tests/zero2fsd/fixtures/mini_repo/course/0.1.1-mini/quiz.yaml && git status --short && git commit -q -F - <<'MSG'
  feat(verify): the gate reads what the learner reads: no course id in prose, every section citation a real heading

  verify_units now scans lecture prose (code fences and link targets excluded, inline code kept), lab markdown
  cells and unit/quiz/cards strings for course ids, and checks every `section "<Heading>"` in them and in the
  grader's hints against the lecture's ## headings. The mini fixture's heading follows the real 0.1.1.a grader.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  MSG
  git log --oneline -1
  ```
  Expected: `git status --short` prints nothing (if Step 4 made you fix and rebuild anything, those paths were added too), then one `<sha> feat(verify): …` line.
