# Mathematical walkthrough

[Teaching notes](lecture7_ot_sbm_notes.md) contain the complete derivation sequence, literal equation readings, intuition, proof assumptions, numerical examples, and source links. [Equation readings](EQUATION_READINGS.md) link directly to the new reveal stages. [Slide-to-code map](SLIDE_CODE_MAP.md) connects implementations to their native slides. [Exercises](EXERCISES.md) provide small extensions and answers.

For a compact starting point, run `python lecture_7/numerical_examples.py --output lecture_7/outputs/numerical`, then `python -m unittest discover -s lecture_7/tests -v`. The LP dual certificate, Sinkhorn marginals, finite bridge probabilities, CTMC master equation, and score-to-drift factor are checked independently of the neural training reports.
