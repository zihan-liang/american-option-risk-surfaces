# Expanded Mathematical Derivation Notes

These notes are a compact navigation companion to the expanded LaTeX reports:

- `reports/expanded/stage01_math_notes_expanded.tex`
- `reports/expanded/stage02_solver_validation_expanded.tex`
- `reports/expanded/final_report_expanded_math.tex`

They do not replace the concise publication-facing reports and do not introduce
new numerical experiments or stronger scientific claims.

## 1. Continuous American-option problem

For payoff \(\Phi(S)\), time to maturity is

\[
\tau=T-t,\qquad U(S,\tau)=V(S,T-\tau),\qquad U(S,0)=\Phi(S).
\]

The Black--Scholes operator is

\[
\mathcal L U
=
\frac12\sigma^2S^2U_{SS}+(r-q)SU_S-rU.
\]

American exercise produces the variational inequality

\[
U\geq\Phi,\qquad
U_\tau-\mathcal L U\geq0,\qquad
(U-\Phi)(U_\tau-\mathcal L U)=0.
\]

The continuation premium

\[
C(S,\tau)=U(S,\tau)-\Phi(S)\geq0
\]

is zero in exercise and positive in continuation.

## 2. Spatial discretization

\[
S_i=i\Delta S,\quad i=0,\ldots,M,
\qquad
\tau_n=n\Delta\tau,\quad n=0,\ldots,N.
\]

For the interior vector

\[
U^n=(U_1^n,\ldots,U_{M-1}^n)^\top,
\]

central differences yield

\[
(L_hU)_i=a_iU_{i-1}+b_iU_i+c_iU_{i+1},
\]

with

\[
a_i=\frac{\tfrac12\sigma^2S_i^2}{(\Delta S)^2}
-\frac{(r-q)S_i}{2\Delta S},
\]

\[
b_i=-\frac{\sigma^2S_i^2}{(\Delta S)^2}-r,
\qquad
c_i=\frac{\tfrac12\sigma^2S_i^2}{(\Delta S)^2}
+\frac{(r-q)S_i}{2\Delta S}.
\]

Boundary nodes \(U_0^n\) and \(U_M^n\) are known values, so they are excluded
from the interior matrix and added to the first and last right-hand-side
entries.

## 3. Crank--Nicolson and the LCP

\[
\frac{U^{n+1}-U^n}{\Delta\tau}
=
\frac12L_hU^{n+1}+\frac12L_hU^n,
\]

\[
A=I-\frac{\Delta\tau}{2}L_h,\qquad
B=I+\frac{\Delta\tau}{2}L_h.
\]

After adding the boundary vector \(g^{n,n+1}\),

\[
AU^{n+1}=BU^n+g^{n,n+1}
\]

is the European step. The American step is

\[
AU^{n+1}\geq d^n,\qquad
U^{n+1}\geq\Phi,\qquad
(U^{n+1}-\Phi)^\top(AU^{n+1}-d^n)=0,
\]

where \(d^n=BU^n+g^{n,n+1}\).

PSOR computes a relaxed Gauss--Seidel candidate and projects it:

\[
x_i^{\mathrm{new}}
=
\max(\Phi_i,x_i^{\mathrm{relaxed}}).
\]

## 4. Diagnostics and learning limits

- Boundary extraction follows low-spot exercise for puts and high-spot
  exercise for dividend-paying calls.
- Missing boundaries are recorded, not fabricated.
- Delta is a first derivative; Gamma is a more fragile second derivative.
- Strict Greek masks exclude payoff kinks, boundary bands, and near-zero spot.
- Grid 240 is an internal numerical reference, not analytic truth.
- Dataset splits are by parameter regime to avoid surface-level leakage.
- Positive-premium reconstruction guarantees only \(\widehat U\geq\Phi\).
- Boundary and Delta are separate diagnostic heads.
- The integrated workflow is scenario screening with fresh PSOR audit.
- The project makes no Gamma-surrogate or PSOR-replacement claim.

## 5. Numerical source of truth

All rounded headline values in the expanded reports must agree with
`docs/verified_metric_ledger.md`. Raw CSV and JSON artifacts remain the
numerical source of truth.
