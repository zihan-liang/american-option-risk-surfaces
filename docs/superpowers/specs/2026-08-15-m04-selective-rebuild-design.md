# M04 Selective Rebuild Design

M04 compares exactly three American-put boundary representations: a positive-premium price surrogate whose validation boundary is extracted solely from predicted premium, a scalar graph head, and an implicit signed level set. All labels are derived from complete training CN/PSOR surfaces; validation and descriptive legacy-test data supply features only during inference.

Every source split is a disjoint collection of verified rectangular surfaces. The selected protocol counts are ten train, five validation, and five legacy-test regimes. Training uses CPU float64 and fixed 30 epochs without validation checkpoint selection or threshold tuning. Each input representation has a training-only normalization record.

Nested accepted-current-M01 CN/PSOR grids provide explicit oracle provenance and convergence/refinement diagnostics. Boundary extraction returns a structured state for every time level, including the B(0)=K maturity convention. Missing or non-monotone roots remain missing; no numerical fallback is invented.

The scalar head has no natural representation for no finite boundary. It reports this limitation in the evidence schema rather than creating a synthetic numeric label; the price-surrogate and implicit representations preserve their structured missing-boundary states.

The screen records separate oracle, preprocessing, training, inference, extraction, fallback, and memory costs, checks the 3600-second budget before each model/seed and timing block, and writes the compact M04 evidence set. The implementation prepares a screen only; it does not run a canonical screen or access the blind holdout.

## M04 gate-evidence repair notes

The dataset remains deliberately unfrozen in the checked-in configuration: a separate data-freeze phase must set the dataset SHA-256 and explicit disjoint train, validation, and legacy-test regime IDs before any screen can run. The shared campaign dispatcher must require an explicit M04 config path during manager integration; this repair phase does not modify that dispatcher. Boundary-only scalar and implicit heads retain price, Delta, Gamma, and LCP diagnostics as `not_applicable_by_design`; the scalar run evidence also records its no-finite-boundary limitation.
