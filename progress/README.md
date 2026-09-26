# Zero2FSD learner progress

Local engagement state for Zero2FSD Modules 00–09 (drafted as LearnFSD). Same rules as the design doc (`docs/redesign/03_modules_00_01_design.md`, section **Engagement & habit loop**).

- **Schema:** [`schema.json`](schema.json)
- **Example:** [`learner.example.json`](learner.example.json) — illustration only; the trainer does not overwrite this file.

**Live file:** [`artifacts/progress.json`](../artifacts/progress.json) (gitignored). Created when a module ships an artifact or records a real XP event via `modules/common/progress.py`.

**XP:** tests green, artifact exported, or a written principle cell (non-placeholder). Opening a notebook awards 0.

**Streak:** Counts when you pass tests for a fill or export an artifact that day. `pause_week: true` freezes the count. A gap resets to 1 on the next qualifying session.

**Badges** are cosmetic. The next module is never locked.
