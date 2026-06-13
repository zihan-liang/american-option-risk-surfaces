# Mathematical Formulation

Under the risk-neutral measure,

```text
dS_t = (r - q) S_t dt + sigma S_t dW_t.
```

For payoff `Phi(S)`, the American value is

```text
V(S,t) = sup_tau E[exp(-r(tau-t)) Phi(S_tau) | S_t = S].
```

Using time-to-maturity `tau = T - t` and `U(S,tau) = V(S,T-tau)`, the continuation-region PDE is

```text
U_tau = L U
L U = 0.5 sigma^2 S^2 U_SS + (r-q) S U_S - r U.
```

The variational inequality / LCP form is

```text
U >= Phi,
U_tau - L U >= 0,
(U - Phi)(U_tau - L U) = 0.
```

The continuation premium is

```text
C(S,tau) = U(S,tau) - Phi(S) >= 0.
```

This inequality is the design rule for the price surrogate: predict nonnegative `C_hat`, then reconstruct `V_hat = Phi + C_hat`.
