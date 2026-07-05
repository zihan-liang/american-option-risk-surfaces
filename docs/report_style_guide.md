# Report Style Guide

## Audience and Voice

Write for a computationally capable AI/ML undergraduate who may not have advanced
financial-mathematics training. Introduce each finance concept in one or two
sentences at first use, then use the technical term consistently. Prefer direct,
evidence-bound academic prose over promotional language.

## Canonical Terminology

- Crank--Nicolson; CN/PSOR; projected successive over-relaxation.
- payoff obstacle; continuation premium; early-exercise boundary; free boundary.
- strict Greek mask; regime-level split; direct-price baseline.
- positive-premium surrogate; boundary diagnostic head.
- bounded supervised Delta head; scenario screening with PSOR audit.

Use "free boundary" for the mathematical object and "early-exercise boundary"
when explaining its financial interpretation. Use "diagnostic head" or
"supervised estimator" for boundary and Delta models.

## Canonical Notation

- Spot, strike, and time to maturity: \(S\), \(K\), and \(\tau=T-t\).
- American value and payoff: \(U(S,\tau)\) and \(\Phi(S)\).
- Continuation premium: \(C(S,\tau)=U(S,\tau)-\Phi(S)\).
- Early-exercise boundary: \(B(\tau)\), reported as \(B/K\) when normalized.
- Greeks: \(\Delta=U_S\), \(\Gamma=U_{SS}\), and normalized Gamma \(K\Gamma\).
- Predictions: \(\widehat U\), \(\widehat C\), \(\widehat B\), and
  \(\widehat\Delta\).
- Neural features:
  \[
  x=(\log(S/K),\tau/T,r,q,\sigma,T,\mathbf{1}_{\mathrm{call}}).
  \]

Do not use \(P\) for continuation premium in report prose because it can be
confused with a put price.

## Numerical Presentation

- Preserve full precision in CSV and JSON evidence artifacts.
- Use approximately four significant figures in prose and paper-facing tables.
- Use scientific notation for residuals and values below \(10^{-3}\).
- State when a displayed value is rounded.
- Use zero only when the underlying evidence reports zero; otherwise report a
  tolerance-scale value.

## Claim Language

Preferred phrases include:

- "supports within the tested regime envelope";
- "audited evidence suggests";
- "obstacle-consistent by construction";
- "retained as negative evidence";
- "scenario screening with PSOR audit";
- "requires further validation before stronger claims".

Do not state that one network solves price, boundary, and Greeks; that the
workflow replaces CN/PSOR; that the price surrogate is a Greek engine; that a
Gamma surrogate is validated; or that future extensions are implemented.

## LaTeX Conventions

- Use `\texttt{}` or `\path{}` for paths and artifact names.
- Use `booktabs` rules and descriptive table headings.
- Captions must identify the quantity, evidence scope, and any limitation that
  could otherwise be overread.
- Use `Crank--Nicolson`, `early-exercise`, and `finite-difference` with LaTeX
  en-dash or hyphen conventions consistently.
- Keep the required eight-subsection structure in every stage report.

