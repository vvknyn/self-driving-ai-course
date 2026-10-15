# Week 1 rulings (controller, 2026-10-15)

Authority, in order: Vivek's decisions D1–D6 (2026-10-11, "yes to all") and the writing standard (`docs/writing/v1.md`, approved 2026-10-10; `docs/writing/v2.md`, adopted by D1). Then the v2 design spec (`docs/superpowers/specs/2026-09-28-zero2fsd-v2-design.md`) and the playground spec (`docs/superpowers/specs/2026-10-11-playground-design.md`).

Each ruling is written as: what — why — cost if wrong. Vivek can reverse any of them.

## Scope and order

- **R1 Three plans, run in sequence.**
  - **C1 "Week 1 foundation"** (gap-report Phase 1, plus the playground fixes carried into it).
  - **C2 "Meet your car, v2"** (gap-report Phase 2).
  - **S1**, the site work: lecture sliders, and checking the first real link.

  C1 and C2 run on course branch `feat/week1-meet-your-car`, which is stacked on `feat/playground-design` @335491a. S1 runs on a site branch stacked on `feat/playground-site-seams`.
  — Why: Phase 1 must leave the checks green on all four weeks before Week 1 is rewritten. S1 consumes the runs contract that C2 defines.
  — Cost if wrong: a re-split.

- **R2 "Week 1" means unit 0.1.1 only.** The playground labels already use this mapping: the week number is the unit's position in the syllabus. The playground spec's wider use of "Week 1" is stale.
  — Cost if wrong: none for content.

## C1: foundation (applies to all four units)

- **R3 One palette, dark figures.**
  - Every figures.py takes every colour from `zero2fsd/palette.py`.
  - Figures use a dark background: CHROME background/panel.
  - One shared helper sets the figure style, so no figures.py repeats it.
  - **Gate G2:** an AST test fails on any string colour literal. That covers a hex literal anywhere, plus any string passed as a colour argument: `color`, `c`, `colors`, `facecolor`, `edgecolor`, or any other keyword containing `color`/`colour`.
  - **Gate G3:** a computed test requires every QUANTITY hue to have contrast ≥ 3:1 against the figure background.

  — Why: decision D3 and gates G2/G3.
  — Cost if wrong: one restyle.

- **R4 No dotted ids in learner-visible text.**
  - Lessons are named by their TITLE.
  - Weeks are "Week N · <title>". Exercises are "Exercise A/B…".
  - This covers lecture.md, lab markdown cells, unit.yaml text fields, quiz/cards, grader output, shim errors and hint strings.
  - Ids stay in code: code cells, URLs, function arguments.
  - **Gate G5** scans the learner-visible files. It also scans the grader's REAL output: it runs one passing and one failing check per exercise and scans stdout.

  — Why: the v1 course-creator review, plus gate G5.
  — Cost if wrong: copy edits.

- **R5 Lab figures appear in lecture order.**
  - A test checks that the lecture figures a lab embeds appear as a subsequence of the order in the lecture.
  - Fix 0.1.2 and 0.1.4 to pass it.

  — Cost if wrong: a reorder.

- **R6 Captions are plain text plus code spans.**
  - The lab builder renders `code` spans in a caption as `<code>`, HTML-escaped.
  - The build fails on any other markdown in alt text (`*`, `[`, `$` outside a code span).
  - This fixes the stray backticks in Colab for every caption, not just one.

  — Cost if wrong: a caption needing emphasis has to be reworded.

- **R7 Every place that cites a section uses one form.** Learner-visible text names a section as `section "<Heading>"`. The grader builds its hints through one helper, which names the lesson by title.
  - **Class test:** every `section "X"` in a unit's lab.py, unit.yaml, grader hints and cards must be a `##` heading of that unit's lecture.
  - Quiz `section` slugs are already checked by the bundle build.

  — Why: a rename should break loudly. Today it silently leaves dead pointers.
  — Cost if wrong: one more convention.

- **R8 Playground fixes from the site run (T4-I1, T4-X1).**
  - **Per-exercise draft** in localStorage (key `learnfsd.v2.draft.<exercise id>`). It is restored on load, and a "Reset to starter" control clears it.
  - **A link that names `?preset=`** loads that preset. When a different draft exists, a notice offers "Restore my draft".
  - **"Back to the lecture"** carries `target="_top"`.
  - **The course link check** keeps trailing `.,;:!?` as part of the id. So `exercise=0.1.1.a.` fails the course build, as it already fails the site sync (checklist :265).

  — Cost if wrong: a small UI change.

- **R9 The "Check my work" label.** The action is named "Check my work" both in the playground and in the Colab check cell. (Gap-report Phase 2 wording.)
  — Cost if wrong: one label.

- **R10 Bring the stale spec text up to date** (docs only):
  - playground spec :118 and :115 → the link form `playground/index.html?exercise=<id>[&preset=<p>]`;
  - :110 → "view.js exports draw, drawStrips, fitCanvas, fitTransform, …; lecture runs in `playground/runs/`";
  - :68 and :170 → the image kind is shipped.

## C2: Meet your car, v2

- **R11 A unit declares v2 with `standard: 2` in unit.yaml.**
  - G1 (no stretch over 300 words without a visual), G4 (every figure is named "Figure N" in the prose next to it), the word cap and goal-within-150-words bind declared units only.
  - G2, G3 and G5 bind all units.
  - The word cap is ≤ 2,000 words outside `<details>`.

  — Why: v2 rule 20 says collapsed boxes are not reading, and the first-delivery plan's word cap is "around 1,200–2,000".
  — Cost if wrong: a cap tweak.

- **R12 The `stats` kind shows the drive.**
  - The page plays the provided car on a road the learner chooses: road, car, player and strips.
  - The road selector is shown for `stats` as it is for `drive`. The record's `road` is the default.
  - The learner's function runs on that drive's lateral error.
  - This applies to EVERY stats exercise (the class).

  — Why: the gap report's Week 1 row ("Drive the provided car on all three roads… predict which road it fails on before you press Go"). Today stats shows no panels.
  — Cost if wrong: the panel set for one kind.

- **R13 Presets are files (P5).**
  - A preset is a file `course/<unit>/presets/<exercise id>/<name>.py`. The builder adds it to `record.presets` next to `starter`.
  - A preset named `starter` is a build error.
  - Week 1 ships at least one teaching preset for v2 rule 14 ("let the naive version fail first"). An example is a lane_error_stats that reports the signed mean, which a lecture link opens.

  — Cost if wrong: preset content moves.

- **R14 Lecture sliders are precomputed runs (D5, v2 rule 27).**
  - figures.py can write `<stem>.runs.json` next to `<stem>.png`. Its shape is `{"runs": [{"label", "road", "rows": [<playground row>…]}]}`.
  - The course build validates it and copies it to `dist/playground/runs/<slug>/<stem>.json`. It is listed in the playground manifest with its sha256, so the site mirrors it with no sync change.
  - course.json maps the figure to its runs file.
  - The PNG stays the fallback (GitHub, Colab, no JS).
  - There is a size budget per runs file, enforced by a test.
  - Week 1's slider is a drive scrubber over the provided car on the three roads.

  — Cost if wrong: one more file kind in the bundle.

- **R15 The first real lecture link (site ruling F9).**
  - The rewritten lecture links at least `playground/index.html?exercise=0.1.1.a`, plus the R13 preset link.
  - S1 shows the site's committed-bundle link checks scanning ≥ 1 link.

- **R16 Lab cells link back to the playground (P15).**
  - One constant, `PLAYGROUND_BASE_URL = "https://vivekn.xyz/learnfsd/playground/index.html"`, lives in the lab builder.
  - Every exercise cell gets a link to its playground page.

  — Cost if wrong: one constant moves when the course goes open-source.

- **R17 Who writes what.**
  - The lecture prose is drafted by the most capable model to `docs/writing/` v1+v2.
  - Code and builds are typed by sonnet.
  - A separate course-creator review reads the rendered lecture before the task closes.

- **R18 Measured facts only.** Every number in the prose or on a figure comes from a run of the course sim (spec DoD 2 and v2 rule 28). Examples: the provided car leaves curvy at 48.3 m after 6.10 s; it completes gentle in 88.85 s.

## S1: site

- **R19 The site renders lecture sliders.**
  - The site adds `data-runs="<url>"` to a figure that course.json maps to a runs file. The URL is always a URL, because the KaTeX pre-pass mangles `$` in attributes.
  - One client island, mounted beside the lecture article, finds `figure[data-runs]`. It imports `/learnfsd/playground/view.js` and swaps the image for a canvas with a run selector and a time scrubber.
  - The site validator refuses any runs path that is not in the manifest.
