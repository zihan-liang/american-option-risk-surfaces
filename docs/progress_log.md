# Progress Log

Use this file to record dated progress, commands, metrics, and decisions.

## Template

- Date:
- Stage:
- Command:
- Key metrics:
- Decision:
- Failure modes observed:
- Next step:

## 2026-06-13

- Stage: CN/PSOR solver validation audit.
- Commands:
  - `python -m unittest discover -s tests -p test_solver_audit.py`
  - `python experiments/01_solver_validation.py --config configs/solver_validation.yaml`
- Key metrics:
  - Smoke grid 120 European put max error: `2.781746966190046e-4`.
  - Smoke grid 120 no-dividend American call max error: `2.781740085198575e-4`.
  - Refined grid 240 European put max error: `6.937489127123442e-5`.
  - Refined grid 240 no-dividend American call max error: `6.937471924713112e-5`.
  - American put obstacle violation: `0.0`.
  - American put complementarity error: `4.0088005617245605e-12`.
  - Dividend-paying American call obstacle violation: `0.0`.
  - Dividend-paying American call complementarity error: `5.2696412383000194e-12`.
- Decision: solver-validation gate passes for continuing classical risk-surface experiments, but not yet as publication-grade final accuracy evidence.
- Failure modes observed: finite-difference error is still visible at moderate grids; `rannacher_steps` is configured but not implemented; boundary values move in grid increments because boundary extraction uses the current uniform grid.
- Next step: use validated solver settings for boundary/Greek diagnostics and keep grid/domain/runtime evidence attached to any reported surfaces.
