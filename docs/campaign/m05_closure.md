# M05 Schrödingerization Closure

## Decision

M05 is closed as `diagnostic_only`. Both bounded numerical gates pass, but the
experiment is a classical dissipative-recovery proxy rather than a quantum
circuit or a complete American-option solver. It supports a follow-up study;
it does not support a quantum speedup or replacement claim.

## Evidence

- CN/PSOR medium/fine grids: 32/64 spatial and time intervals.
- Classical discretization floor: `9.739070937476127e-4` normalized RMSE.
- Frozen-continuation inner RMSE: `9.204467659744154e-13`.
- One-live-obstacle-update inner RMSE: `1.6667828623921668e-12`.
- Active-set mismatch: `0.0` for both diagnostics.
- Live-update obstacle violation: `0.0`.
- Live-update LCP natural residual: `6.144432185273274e-12`.
- Blind holdout accessed: `false`.

## Limits

The screen covers one normalized American put, one mid-maturity CN step, one
frozen policy system, and one live policy update. The finite-time state is
computed classically from the normal-equation dissipative flow. No auxiliary
Schrödingerization grid, Hamiltonian circuit, state preparation, postselection,
readout, full maturity march, Greek study, or speed comparison is included.
These omissions prevent any broader claim.
