# M09 Pre-Canonical Corrective Note

The first M09 smoke run completed before any canonical run and exposed two protocol defects in implementation commit `16576a6ce43fc883194d126af2fcb145e9bf08eb`:

1. the coarse MLP was instantiated before its declared campaign seed was applied;
2. ordinary-test fine surfaces and metrics were generated for every fraction before validation-only matched-accuracy selection was frozen.

The corrective commit preserves the original runner as `experiments/standalone/m09_multifidelity_legacy.py`, makes model initialization explicitly seed-first, and defers ordinary-test oracle generation/evaluation until after validation freezes the selected fractions. It also moves `total_seconds` measurement after selected test and inference timing.

No canonical M09 run occurred before this correction. The branch therefore has a documented extra pre-canonical software-fix commit rather than rewriting or force-pushing the published implementation history.
