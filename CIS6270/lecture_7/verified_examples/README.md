# Recorded execution

These are actual seeded CPU runs of all 17 methods at their default training settings. Each run saves a checkpoint, reloads it, and checks that the same sampling seed produces exactly the same generated values. `config.json` records the Python, PyTorch, NumPy, seed, and training settings. `run_all_report.json` records every reload check.

The reports include approximation residuals rather than replacing them with analytic targets. In particular, the soft Branch constraints leave small negative weights and a small mass residual; these are reported. The cone example is a deterministic geometry calculation. Neural examples use small synthetic problems; these results are separate from the attributed paper benchmarks in the lecture.

`mathematics/` supplies learned-run diagnostics to the independent mathematical tests. Checkpoints are created by the runner and are not bundled in the repository update. Reproduce with `python lecture_7/run_all.py`.
