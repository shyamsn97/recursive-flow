# NumPy Linear Algebra

Use this skill when the user asks for a concrete matrix/vector computation,
least-squares fit, eigenvalue calculation, or numerical verification.

## Instructions

- Use NumPy in the REPL for the actual arithmetic. Do not do matrix algebra by
  hand when code can compute it exactly enough.
- Print the key intermediate arrays or scalars so the trace is auditable.
- Use `np.linalg.solve` for square nonsingular systems and `np.linalg.lstsq`
  for overdetermined systems.
- Verify the answer by computing a residual, reconstruction, or direct
  substitution check.
- In the final answer, include the numeric result and the verification residual.

## Example

For a least-squares line `y = m*x + b` through points:

```python
import numpy as np

x = np.array([0, 1, 2, 3], dtype=float)
y = np.array([1, 2, 2, 4], dtype=float)
A = np.column_stack([x, np.ones_like(x)])

coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
m, b = coeffs
pred = A @ coeffs
residual = y - pred
residual_norm = np.linalg.norm(residual)

print("A =", A)
print("coeffs =", coeffs)
print("pred =", pred)
print("residual =", residual)
print("residual_norm =", residual_norm)
```

Then report `m`, `b`, `pred`, and `residual_norm`.
