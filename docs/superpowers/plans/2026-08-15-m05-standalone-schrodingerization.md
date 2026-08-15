# M05 Standalone Schrödingerization Diagnostic

**Scope:** Implement only the M05 screening diagnostic. Do not run a quantum
circuit and do not claim speedup.

1. Reconstruct one mid-maturity CN linear-complementarity system from the
   existing American-put CN/PSOR oracle.
2. Emulate finite-time dissipative state recovery for a frozen PSOR policy
   system and for one live policy update. This is the classical inner evolution
   that Schrödingerization would embed, not a QPU implementation.
3. Estimate the classical discretization floor from matched medium/fine
   CN/PSOR grids. Pass a numerical gate only when both inner errors are below
   that floor and the projected live update respects the obstacle.
4. Persist compact, hashed evidence and retain the overall conclusion
   `diagnostic_only`, regardless of the numerical gate.
5. Verify focused tests, a short screen, compileall, artifact hashes, the full
   test suite, and repository cleanliness before committing.
