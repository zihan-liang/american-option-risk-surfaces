# M04 Implementation Review

## Decision

The selective M04 implementation is accepted for integration and may advance
to a separate data-freeze and smoke-preparation phase. It is not accepted as
experimental evidence, and neither smoke nor canonical execution is currently
authorized.

Reviewed commits:

- `c9020163c9279ef44d0f4a2a78bf108f92484af1`: selective rebuild on the
  accepted campaign base.
- `8c9ec7718bba49f35bce1846cf7eb851638721fb`: bounded gate and evidence repair.

The old `exp/m04-free-boundary-representation` history is rejected for
integration because it contains a tracked virtual environment and invalid
held-out-label comparisons. No old M04 commit was merged or cherry-picked.

## Accepted Implementation Properties

- Three explicit representations: predicted-premium boundary extraction,
  scalar boundary head, and implicit level set.
- No validation-label input to model inference or boundary extraction.
- Fixed-epoch CPU float64 training with seeds
  `[101, 203, 307, 409, 503]` and training-only normalization.
- Complete-surface loading, explicit split-ID interface, and dataset SHA-256
  gate.
- Accepted-current-M01 CN/PSOR oracle with per-level nested-grid validation.
- Structured missing, no-boundary, non-monotone, and maturity states.
- Paired complete-regime inference with Holm correction for the two direct
  candidates.
- Training-only premium-contact tolerance selection.
- Wall-clock budget checkpoints and finite compact output writers.
- Boundary-only price, Greek, and LCP diagnostics remain explicitly not
  applicable by design; U-Net remains in M19.

## Verification

- Worker focused suite: 29 tests passed.
- Manager rerun of the same M04 suite: 29 tests passed.
- Manager M01--M04 campaign/dispatcher suite: 71 tests passed.
- Focused compileall and `git diff --check` passed.
- No smoke, canonical, or blind-holdout run occurred.

## Mandatory Before Smoke

The next phase is limited to data freeze and smoke preparation. It must:

1. make the ignored real dataset available without committing it;
2. verify and freeze its SHA-256 plus explicit disjoint regime IDs;
3. persist coarse/fine per-level oracle diagnostics in compact evidence;
4. ensure output-preparation time and any resource-blocked output carry full
   provenance and budget accounting;
5. require an explicit M04 config in the shared dispatcher;
6. add an explicitly non-evidence smoke configuration and run it only after
   the preceding gates pass.

Failure of any item blocks smoke. Canonical execution remains a later,
separately authorized phase.
