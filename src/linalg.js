// Matrix-free numerical helpers: SIMD-free, dependency-free JS.
// Sizes here are tiny (<= 6 parameters, <= a few hundred rows), so clarity wins.

/** Cholesky solve for symmetric positive definite A (n x n, row-major flat). */
export function choleskySolve(A, b, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k += 1) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) throw new Error('matrix not positive definite');
        L[i * n + j] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    let s = b[i];
    for (let k = 0; k < i; k += 1) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i -= 1) {
    let s = y[i];
    for (let k = i + 1; k < n; k += 1) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

export function invertSPD(A, n) {
  const inv = new Float64Array(n * n);
  for (let col = 0; col < n; col += 1) {
    const e = new Float64Array(n);
    e[col] = 1;
    const x = choleskySolve(A, e, n);
    for (let row = 0; row < n; row += 1) inv[row * n + col] = x[row];
  }
  return inv;
}

/** Solve a small dense linear system with partial pivoting (fallback only). */
export function solveGeneral(A0, b0, n) {
  const A = Float64Array.from(A0);
  const b = Float64Array.from(b0);
  for (let col = 0; col < n; col += 1) {
    let piv = col;
    for (let r = col + 1; r < n; r += 1) if (Math.abs(A[r * n + col]) > Math.abs(A[piv * n + col])) piv = r;
    if (Math.abs(A[piv * n + col]) < 1e-14) throw new Error('singular system');
    if (piv !== col) {
      for (let k = 0; k < n; k += 1) {
        const t = A[col * n + k];
        A[col * n + k] = A[piv * n + k];
        A[piv * n + k] = t;
      }
      const t = b[col];
      b[col] = b[piv];
      b[piv] = t;
    }
    for (let r = col + 1; r < n; r += 1) {
      const f = A[r * n + col] / A[col * n + col];
      if (f === 0) continue;
      for (let k = col; k < n; k += 1) A[r * n + k] -= f * A[col * n + k];
      b[r] -= f * b[col];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r -= 1) {
    let s = b[r];
    for (let k = r + 1; k < n; k += 1) s -= A[r * n + k] * x[k];
    x[r] = s / A[r * n + r];
  }
  return x;
}

export function quantile(sorted, p) {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo);
}

/** Two-sided normal quantile via Acklam's approximation (abs err < 1.2e-9). */
export function normalQuantile(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  let q;
  let r;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - plow) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export function mean(xs) {
  if (xs.length === 0) return NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

export function sd(xs) {
  if (xs.length < 2) return NaN;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) ** 2;
  return Math.sqrt(s / (xs.length - 1));
}
