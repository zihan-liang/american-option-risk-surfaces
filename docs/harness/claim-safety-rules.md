# Claim-Safety Rules

These rules protect the project from overstating what the experiments prove.

## Allowed claims

- CN/PSOR is the reference solver and audit/label engine within the tested regimes.
- Boundary heads are supervised diagnostics for approximate exercise-boundary behavior.
- Delta heads are bounded diagnostic surrogates evaluated under strict masks.
- Integrated workflows support scenario screening and audit-case comparison.
- Round-2 boundary methods are experimental unless they pass the full metric and provenance gates.

## Forbidden claims

Avoid language that says or implies:

- a surrogate replaces PSOR;
- a neural model is an exact American-option solver;
- one network solves price, boundary, Delta, and Gamma for all regimes;
- Round-2 failures are proof that the method class cannot work;
- smoke-scale outputs are full-scale evidence;
- Gamma surrogate results are in scope without explicit authorization.

## Safe phrasing examples

- “The surrogate is evaluated against CN/PSOR labels.”
- “This boundary head is diagnostic and remains subject to PSOR audit.”
- “Round-2 did not justify a best-combo branch under the current gates.”
- “The result is future work until full-scale evidence and claim audit pass.”

## Unsafe phrasing examples

- “The surrogate replaces PSOR.”
- “The model solves American option pricing.”
- “The boundary network is an exact free-boundary solver.”
- “The integrated workflow removes the need for PSOR audit.”

When in doubt, prefer narrower language and point to the exact tested regime, seed, artifact, and metric.
