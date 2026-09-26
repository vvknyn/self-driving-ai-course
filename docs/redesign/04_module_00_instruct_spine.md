# Module 00 — Instruct spine (Driving ML Gym)

Status: **lecture path** for `notebooks/00_driving_ml_gym.ipynb`, generated from `scripts/build_notebooks_00_01.py` → `build_m00()`.

## Time sketch

| Segment | Minutes (guide) |
|---|---|
| Beat 0 — Cold open | ~10 |
| Beats 1–7 (theory → demo → check each) | ~35–50 |
| Beat 8 fills, break-it, pytest, artifact | ~10–30 |
| **Full notebook** | **55–90** (25 min slot = session card + one beat only) |

## Pedagogy

- **Course direction:** FastAI-style top-down — working crop classifier before full theory stack.
- **Inside each beat:** **What** (definition or formula) plus **Why** (intuition) inside the Theory markdown cell, then **Demo**, then **Check**. MicroMasters-style bottom-up per topic; load-bearing fills only after beats 2–7 teach the mechanisms.
- **Why blocks:** Every major concept gets a greppable `**Why — short name.**` paragraph of about 3–6 sentences: why the idea exists (driving or learning problem), why this mathematical or engineering shape, and one concrete driving sentence when it fits.
- **Fills after motivation:** `focal_loss` after beat 7’s Why on the objective; `minority_recall` and `build_error_gallery` in beat 8 after the inspection Whys.
- **First principles** live in the spine (including a Why before beat 0); assignments force applying them in `.py` fills and written cells.

### Governing principles

1. A model is a function from data to scores; learning adjusts parameters so a chosen loss gets small on the training distribution.
2. The loss defines what good means. If the loss ignores rare classes, the model will too.
3. You cannot improve what you do not inspect.

## Lecture beats (0–8)

| Beat | Title | Teaches |
|---|---|---|
| 0 | Cold open | Task, untrained fail hook, always-road accuracy trap |
| 1 | Images as numbers | HWC vs CHW, [0,1] scale, batch shape |
| 2 | Classification as scores | Softmax, CE derivation, manual vs `F.cross_entropy` |
| 3 | Neuron and layer | Affine + ReLU, XOR vs linear |
| 4 | The train loop | forward / loss / backward / step, LR break, `zero_grad` |
| 5 | Why convolutions | locality, sharing, scaffold `DrivingClassifier` shapes |
| 6 | Metrics that lie | accuracy trap, confusion matrix, CE training baseline |
| 7 | The loss is the objective | focal factor derivation, numeric table, **fill `focal_loss`** |
| 8 | Inspect and ship | **fill `minority_recall`**, **from-scratch `build_error_gallery`**, break-it, pytest, artifacts |

Each beat uses **Theory** (What + Why), **Demo**, and **Check** subsections. Checks are answered by the following code cell (assert + print) so headless execution passes.

## Scaffold vs fill vs from-scratch

| Kind | Share | Module 00 items |
|---|---|---|
| Scaffold | ~60–70% | `TrainConfig`, `DrivingPatchDataset`, `DrivingClassifier`, `cross_entropy_loss`, `train_epoch`, `evaluate`, `train.main`, break-it |
| Fill | ~25–30% | `focal_loss`, `minority_recall` (`NotImplementedError` until student implements) |
| From scratch | ~5–10% | `build_error_gallery` |

Student code stays in `modules/00_ml_gym/*.py`. No solutions pasted into the notebook.

## Engagement (unchanged)

- Session card via `session_card_text("m00")`; XP not awarded for opening the notebook.
- `come_back_cue("m00")` at end; progress.json cell.
- Break-it demo, pytest, artifact export.

## Non-goals (this spine)

- Autograd / Micrograd on the **default Week 1 critical path** — optional appendix **00b** only (`modules/00_nn_scratch`, `notebooks/00_neural_networks_and_autograd.ipynb`).
- Module 01 rewrite (separate pass).
- Detection, BEV, or camera geometry.

## Optional deepeners (not substitutes)

- [StatQuest — cross-entropy](https://www.youtube.com/watch?v=6ArSys5qHAU)
- [3Blue1Brown — neural networks](https://www.3blue1brown.com/lessons/neural-networks)
- [3Blue1Brown — backpropagation](https://www.3blue1brown.com/lessons/backpropagation)
