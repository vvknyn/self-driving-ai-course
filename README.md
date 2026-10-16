# Zero2FSD

Zero2FSD is a hands-on course on how a self-driving car works. You drive a simulated car that ships with a working perception box and controller, then replace those parts with your own code, one lab at a time. Each unit has a lecture, figures, a graded Python lab, a quiz and review cards.

Take the course at **[zero2fsd.com](https://zero2fsd.com)**. This repository holds the source of the course: lectures, labs, the simulator and the grader.

Zero2FSD is an independent educational project. Not affiliated with Tesla, Inc.

## Run it locally

Needs Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/build_labs.py --check          # lab.ipynb files are up to date with each lab.py
python scripts/verify_units.py                # every starter fails its grader, every reference solution passes
python -m pytest tests/zero2fsd/test_car.py   # run one test file; drop the path to run the whole suite
```

Open a unit's `lab.ipynb` in Jupyter, or in Colab through the badge inside the notebook. After editing a `lab.py`, rebuild its notebook with `python scripts/build_labs.py` (no `--check`).

## What is where

- `course/`: the maintained course. One folder per unit (`lecture.md`, `lab.py`, `lab.ipynb`, `quiz.yaml`, `cards.yaml`, `figures/`), plus `syllabus.yaml`. `course/dist/` is generated output.
- `zero2fsd/`: the simulator, the car, the provided boxes and the grader behind the labs.
- `playground/`: the in-browser Python playground (Pyodide).
- `scripts/`, `tests/`, `solutions/`, `data/`: build tools, tests, reference solutions and small synthetic datasets.

Legacy v1 material (not reviewed) lives in `modules/`, `notebooks/`, `docs/` and `visual_explainers/`; the maintained course is `course/`.

## Licence

Free for non-commercial use (source-available).

- Lessons, figures, quizzes, flashcards and lab text: [CC BY-NC-SA 4.0](LICENSE-content). Code: [PolyForm Noncommercial 1.0.0](LICENSE).
- Free for personal study, teaching, research and nonprofit use.
- Commercial use (selling it, paywalls, paid courses or training, commercial products) needs a licence: licensing@zero2fsd.com
- Attribution: "Zero2FSD by Vivek Narayanan (https://zero2fsd.com), CC BY-NC-SA 4.0"
- Details: https://zero2fsd.com/license. Third-party software the course loads is listed in [NOTICE](NOTICE).

## Contributing

Found an error, a broken lab or an unclear explanation? Open an issue. Fixes are welcome as pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md).
