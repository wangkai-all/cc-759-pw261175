// Calibration engine for two imperfect devices.
//
// Structural errors-in-variables model. Neither device is treated as exact:
//
//   x_i                  true value of reference sample i (latent)
//   u_ijk = x_i + e      old-device reading,  e ~ N(0, sRef^2)
//   y_ijk = g(x_i) + c_b + eps,                eps ~ N(0, sNew^2)
//   g(x)  = a + b*x                            (linear)
//   g(x)  = a + b*x + d*max(0, x-k)            (continuous kink at k)
//
// The reference certificate supplies a nominal value with its own uncertainty,
// entering as a weak Gaussian prior on x_i; the old-device repeats then refine
// x_i. Batch shared offsets c_b get a weak N(0, 0.5^2) prior. Readings below
// the limit of detection are left-censored; saturated readings are right-
// censored at the instrument plateau. Latent x_i are profiled out by a 1-D
// Newton solve, so the outer optimiser searches only the response shape, the
// two repeatabilities and the batch offsets. Parameter uncertainty comes from
// the observed information of the profile likelihood; inverse prediction
// propagates both that covariance and the single-reading error by Monte Carlo.
import { normal, mulberry32 } from './random.js';
import { sd, mean, normalQuantile } from './linalg.js';

const LOG_2PI = Math.log(2 * Math.PI);
const MIN_SIGMA = 1e-3;

function clampResidual(r, cap) {
  if (!Number.isFinite(r)) return Number.isFinite(r) ? r : cap;
  return Math.max(-cap, Math.min(cap, r));
}

export function normalLogPdf(r, mu, sigma) {
  const z = (r - mu) / sigma;
  return -0.5 * (LOG_2PI + 2 * Math.log(sigma) + z * z);
}

// ------------------------------------------------------- gross-error screening

/**
 * Flag candidate gross outliers using within-cell repeat scatter only, so the
 * screen does not depend on the fitted response curve or on the repeatability
 * it is trying to protect. For every cell the median and MAD are taken across
 * the repeats at that (reference, batch); a reading more than `cut` robust
 * standard deviations from its cell median is a candidate. Rows already marked
 * by the import (`importedFlag`) are always candidates.
 *
 * Returns the candidate ids and, for transparency, the per-row score.
 */
export function screenOutliers(measurements, { cut = 4, minCells = 3 } = {}) {
  const byCell = new Map();
  for (const m of measurements) {
    if (m.belowLod || m.saturated || !Number.isFinite(m.reading)) continue;
    const key = `${m.referenceId}|${m.batchId}`;
    if (!byCell.has(key)) byCell.set(key, []);
    byCell.get(key).push(m);
  }
  const scores = new Map();
  let cellsUsed = 0;
  for (const rows of byCell.values()) {
    if (rows.length < 3) continue;
    cellsUsed += 1;
    const vals = rows.map((r) => r.reading).sort((a, b) => a - b);
    const mid = vals[Math.floor(vals.length / 2)];
    const dev = vals.map((v) => Math.abs(v - mid)).sort((a, b) => a - b);
    const mad = dev[Math.floor(dev.length / 2)];
    // 1.4826 rescales the MAD to a Gaussian sigma; floor it so a cell that
    // happens to repeat almost identically cannot flag ordinary noise.
    const sigma = Math.max(1.4826 * mad, 0.15);
    for (const r of rows) {
      const score = Math.abs(r.reading - mid) / sigma;
      scores.set(r.id, score);
    }
  }
  const flagged = new Set();
  if (cellsUsed >= minCells) {
    for (const m of measurements) {
      const s = scores.get(m.id);
      if (m.importedFlag) flagged.add(m.id);
      else if (Number.isFinite(s) && s > cut) flagged.add(m.id);
    }
  } else {
    for (const m of measurements) if (m.importedFlag) flagged.add(m.id);
  }
  return { flagged, scores };
}

// --------------------------------------------------------------- normal tails

function logUpperTail(x) {
  if (x > 8) return -0.5 * x * x - Math.log(x) - 0.5 * LOG_2PI;
  let t = x + 1 / (x + 2 / (x + 3 / (x + 4 / x)));
  for (let i = 40; i >= 1; i -= 1) t = x + i / t;
  return -0.5 * x * x - 0.5 * LOG_2PI - Math.log(t);
}

/** Complementary error function (Numerical Recipes erfcc; rel. err. < 1.2e-7). */
function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const ans =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? ans : 2 - ans;
}

export function logPhi(z) {
  if (z < -10) {
    const z2 = z * z;
    return -0.5 * z2 - Math.log(-z) - 0.5 * LOG_2PI + Math.log1p(-1 / z2 + 3 / (z2 * z2));
  }
  if (z > 8) return -Math.exp(logUpperTail(z));
  const v = 0.5 * erfc(-z / Math.SQRT2);
  return Math.log(Math.max(v, 1e-308));
}

/** phi(z)/Phi(z), the inverse Mills ratio. */
function invMills(z) {
  return Math.exp(-0.5 * z * z - 0.5 * LOG_2PI - logPhi(z));
}

// ---------------------------------------------------------------- responses

export function makeResponse({ model, intercept, slope, kink, kinkDelta }) {
  const hasKink = model === 'kink' && Number.isFinite(kink);
  return {
    model,
    intercept,
    slope,
    kink: hasKink ? kink : null,
    kinkDelta: hasKink ? kinkDelta : 0,
    forward(x) {
      const v = intercept + slope * x;
      return hasKink && x > kink ? v + kinkDelta * (x - kink) : v;
    },
    slopeAt(x) {
      return hasKink && x > kink ? slope + kinkDelta : slope;
    },
    inverse(y) {
      if (!hasKink) return (y - intercept) / slope;
      const yKink = intercept + slope * kink;
      if (y <= yKink) return (y - intercept) / slope;
      return kink + (y - yKink) / (slope + kinkDelta);
    },
  };
}

// ---------------------------------------------------------- data preparation

/**
 * Collapse measurements to reference groups and (reference, batch) cells.
 * Each cell keeps the new-device readings (some censored) and the old-device
 * readings, so within-cell repeatability of both devices is modelled.
 */
export function prepareGroups(measurements, references, options = {}) {
  const {
    withdrawnRefs = new Set(),
    excludedBatches = new Set(),
    excludedMeasurementIds = new Set(),
    includeBelowLod = true,
    includeSaturated = true,
    excludeCells = new Set(),
  } = options;
  const refById = new Map(references.map((r) => [r.id, r]));
  const byRef = new Map();
  for (const m of measurements) {
    if (withdrawnRefs.has(m.referenceId)) continue;
    if (excludedBatches.has(m.batchId)) continue;
    if (excludedMeasurementIds.has(m.id)) continue;
    if (m.belowLod && !includeBelowLod) continue;
    if (m.saturated && !includeSaturated) continue;
    if (!Number.isFinite(m.reading)) continue;
    if (excludeCells.has(`${m.referenceId}|${m.batchId}`)) continue;
    let g = byRef.get(m.referenceId);
    if (!g) {
      const ref = refById.get(m.referenceId);
      g = {
        referenceId: m.referenceId,
        range: m.range,
        nominal: ref?.certified ?? NaN,
        certSd: ref?.certSd ?? 0.4,
        cells: new Map(),
        rows: [],
      };
      byRef.set(m.referenceId, g);
    }
    g.rows.push(m);
    let cell = g.cells.get(m.batchId);
    if (!cell) {
      cell = {
        batchId: m.batchId,
        nNew: 0,
        sumNew: 0,
        newValues: [],
        censored: [],
        nRef: 0,
        sumRef: 0,
        sumRef2: 0,
      };
      g.cells.set(m.batchId, cell);
    }
    if (m.belowLod) cell.censored.push({ bound: m.reading, kind: 'left' });
    else if (m.saturated) cell.censored.push({ bound: m.reading, kind: 'right' });
    else {
      cell.nNew += 1;
      cell.sumNew += m.reading;
      cell.newValues.push(m.reading);
    }
    const old = Number.isFinite(m.oldReading) ? m.oldReading : m.refReading;
    if (Number.isFinite(old)) {
      cell.nRef += 1;
      cell.sumRef += old;
      cell.sumRef2 += old * old;
    }
  }
  const groups = [...byRef.values()];
  groups.sort((a, b) => a.range.localeCompare(b.range) || a.nominal - b.nominal);
  for (const g of groups) {
    g.cellList = [...g.cells.values()];
    g.batchIds = [...new Set(g.rows.map((r) => r.batchId))];
    // The certificate is a measurement with its own uncertainty, not truth.
    g.priorSd = Number.isFinite(g.certSd) ? Math.max(0.05, g.certSd) : 0.4;
    g.xInit = g.nominal;
  }
  return groups;
}

export function priorCentre(group) {
  return group.nominal;
}

// ------------------------------------------------------------------- fitting

export function fitModel({
  measurements,
  references,
  batches = [],
  model = 'linear',
  sRef: fixedSRef = null,
  kinkRange = null,
  groups: suppliedGroups = null,
  options = {},
  optimizer = {},
  init = null,
  screen = true,
  debug = null,
}) {
  // Gross errors are screened out before fitting by a rule that uses only
  // within-cell repeat scatter (see `screenOutliers`). Screening can be turned
  // off for diagnostics; the app leaves it on and reports what was dropped.
  let screenReport = null;
  let fitMeasurements = measurements;
  if (screen && Number.isFinite(measurements?.[0]?.reading)) {
    screenReport = screenOutliers(measurements, { cut: options.outlierCut ?? 4 });
    if (screenReport.flagged.size > 0) {
      fitMeasurements = measurements.filter((m) => !screenReport.flagged.has(m.id));
    }
  }
  const groups = suppliedGroups || prepareGroups(fitMeasurements, references, options);
  if (groups.length < 3) throw new Error('校准数据不足：至少需要 3 个有效参考样本');

  const batchIds = [...new Set(groups.flatMap((g) => g.batchIds))].sort();
  const nBatch = batchIds.length;
  const batchIndex = new Map(batchIds.map((id, i) => [id, i]));
  const isKink = model === 'kink';

  // theta = [intercept, slope, (kinkDelta, kink), log(sNew), (log(sRef)), batches...]
  const iA = 0;
  const iB = 1;
  const iD = isKink ? 2 : -1;
  const iK = isKink ? 3 : -1;
  const iSigNew = isKink ? 4 : 2;
  const iSigRef = fixedSRef == null ? iSigNew + 1 : -1;
  const iBatch0 = (fixedSRef == null ? iSigNew + 2 : iSigNew + 1);
  const nPar = iBatch0 + nBatch;

  const bounds = new Array(nPar);
  const setBound = (i, lo, hi) => {
    if (i >= 0) bounds[i] = [lo, hi];
  };
  setBound(iA, -2000, 2000);
  setBound(iB, 0.2, 5);
  setBound(iD, -1.5, 1.5);
  setBound(iK, ...(kinkRange || [-1000, 1000]));
  setBound(iSigNew, Math.log(0.02), Math.log(30));
  setBound(iSigRef, Math.log(0.005), Math.log(10));
  for (let b = 0; b < nBatch; b += 1) bounds[iBatch0 + b] = [-3, 3];

  const clampToBounds = (v, i) => Math.min(bounds[i][1], Math.max(bounds[i][0], v));

  // --- initial values: median per reference against certificate nominal
  const pairs = [];
  for (const g of groups) {
    const vals = g.rows.filter((r) => !r.belowLod && !r.saturated).map((r) => r.reading);
    if (vals.length && Number.isFinite(g.nominal)) {
      vals.sort((a, b) => a - b);
      pairs.push([vals[Math.floor(vals.length / 2)], g.nominal]);
    }
  }
  let bInit = 1.03;
  let aInit = 0;
  if (pairs.length >= 2) {
    const mx = mean(pairs.map((p) => p[1]));
    const my = mean(pairs.map((p) => p[0]));
    let sxy = 0;
    let sxx = 0;
    for (const [y, x] of pairs) {
      sxy += (x - mx) * (y - my);
      sxx += (x - mx) ** 2;
    }
    if (sxx > 1e-9) bInit = sxy / sxx;
    aInit = my - bInit * mx;
  }
  // Initial new-device repeatability from within-cell repeat readings, taken as
  // the median of per-cell sample variances. The mean would let a few bumped
  // fixture rows (gross outliers) inflate the start and send the search into a
  // shallow local optimum of the profiled likelihood.
  const cellVars = [];
  const cellVals = [];
  for (const g of groups) {
    for (const c of g.cellList) {
      if (c.nNew >= 2) {
        const vals = c.newValues;
        const m = mean(vals);
        cellVars.push(vals.reduce((acc, v) => acc + (v - m) ** 2, 0) / (c.nNew - 1));
        vals.sort((a, b) => a - b);
        cellVals.push(vals[Math.floor(vals.length / 2)]);
      }
    }
  }
  cellVars.sort((a, b) => a - b);
  let sNew0 = cellVars.length ? Math.sqrt(cellVars[Math.floor(cellVars.length / 2)]) : NaN;
  if (!Number.isFinite(sNew0) || sNew0 < 0.05) sNew0 = 0.5;
  sNew0 = Math.min(20, sNew0);
  const refPool = groups.flatMap((g) =>
    g.rows.map((r) => (Number.isFinite(r.oldReading) ? r.oldReading : r.refReading)).filter(Number.isFinite),
  );
  let sRef0 = fixedSRef;
  if (sRef0 == null) {
    // Pooled within-cell SD of old readings.
    let ss = 0;
    let df = 0;
    for (const g of groups) {
      for (const c of g.cellList) {
        if (c.nRef >= 2) {
          ss += c.sumRef2 - (c.sumRef * c.sumRef) / c.nRef;
          df += c.nRef - 1;
        }
      }
    }
    sRef0 = df > 0 ? Math.sqrt(ss / df) : sd(refPool) || 0.05;
    if (!Number.isFinite(sRef0) || sRef0 < 0.005) sRef0 = 0.02;
  }

  const theta = new Float64Array(nPar);
  theta[iA] = clampToBounds(aInit, iA);
  theta[iB] = clampToBounds(bInit, iB);
  if (isKink) {
    const xs = groups.map((g) => g.nominal).filter(Number.isFinite);
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    theta[iK] = clampToBounds(Number.isFinite(kinkRange?.[0]) ? (kinkRange[0] + kinkRange[1]) / 2 : xmin + 0.35 * (xmax - xmin), iK);
    theta[iD] = 0;
  }
  theta[iSigNew] = clampToBounds(Math.log(sNew0), iSigNew);
  if (iSigRef >= 0) theta[iSigRef] = clampToBounds(Math.log(sRef0), iSigRef);
  for (let b = 0; b < nBatch; b += 1) theta[iBatch0 + b] = 0;
  if (init) {
    if (Number.isFinite(init.intercept)) theta[iA] = init.intercept;
    if (Number.isFinite(init.slope)) theta[iB] = init.slope;
    if (isKink && Number.isFinite(init.kink)) theta[iK] = init.kink;
    if (isKink && Number.isFinite(init.kinkDelta)) theta[iD] = init.kinkDelta;
    if (Number.isFinite(init.sNew)) theta[iSigNew] = Math.log(init.sNew);
    if (iSigRef >= 0 && Number.isFinite(init.sRef)) theta[iSigRef] = Math.log(init.sRef);
  }

  const responseFrom = (th) =>
    makeResponse({ model, intercept: th[iA], slope: th[iB], kink: isKink ? th[iK] : NaN, kinkDelta: isKink ? th[iD] : 0 });

  const xWork = groups.map((g) => g.xInit);
  const offAt = (th, id) => th[iBatch0 + batchIndex.get(id)];
  const refSigmaFrom = (th) => (iSigRef >= 0 ? Math.exp(th[iSigRef]) : fixedSRef);

  /** Conditional NLL and derivatives for one reference group at a given x. */
  function groupNll(x, g, resp, sigNew, sigRef, offs, wantDeriv = false) {
    let nll = 0;
    let grad = 0;
    let hess = 0;
    const pc = g.nominal;
    const dprior = x - pc;
    nll += 0.5 * (dprior / g.priorSd) ** 2;
    grad += dprior / (g.priorSd * g.priorSd);
    hess += 1 / (g.priorSd * g.priorSd);
    const dg = resp.slopeAt(x);
    const s2 = sigNew * sigNew;
    for (const cell of g.cellList) {
      const off = offs.get(cell.batchId) || 0;
      if (cell.nNew > 0) {
        const mu = resp.forward(x) + off;
        // Gaussian repeats on the new device. Gross outliers (bumped fixture)
        // are not absorbed here: they are pre-trimmed from the fit by
        // `trimOutliers` and reported in the residual table instead. A mixture
        // with a wide component was tried first, but a whole high-range cell
        // could flip to the wide component, giving the profile likelihood two
        // discrete modes and a jumpy optimiser surface.
        let cellNll = 0;
        let cellGrad = 0;
        let cellHess = 0;
        for (const y of cell.newValues) {
          const r = clampResidual(y - mu, sigNew * 1e4);
          cellNll += 0.5 * (LOG_2PI + 2 * Math.log(sigNew)) + (r * r) / (2 * s2);
          cellGrad += (r / s2) * dg;
          cellHess += Math.max((dg * dg) / s2, 1e-9);
        }
        nll += cellNll;
        // cellGrad = (r/sigma^2)*dg; d(nll)/dx = -(r/sigma^2)*dg.
        grad -= cellGrad;
        hess += cellHess;
      }
      for (const c of cell.censored) {
        const mu = resp.forward(x) + off;
        if (c.kind === 'left') {
          const z = (c.bound - mu) / sigNew;
          const lam = invMills(z);
          nll -= logPhi(z);
          grad += (dg / sigNew) * lam;
          hess += ((dg * dg) / s2) * Math.max(lam * (lam + z), 0);
        } else {
          const zp = (mu - c.bound) / sigNew;
          const lam = invMills(zp);
          nll -= logPhi(zp);
          grad -= (dg / sigNew) * lam;
          hess += ((dg * dg) / s2) * Math.max(lam * (lam + zp), 0);
        }
      }
      if (cell.nRef > 0) {
        // Full within-cell residual sum of squares of the old-device readings:
        // this is what identifies the reference repeatability (a cell-mean-only
        // statistic would lose it and let sRef shrink without penalty).
        const sr2 = sigRef * sigRef;
        const rss = cell.sumRef2 - 2 * x * cell.sumRef + cell.nRef * x * x;
        nll += 0.5 * cell.nRef * LOG_2PI + cell.nRef * Math.log(sigRef) + rss / (2 * sr2);
        // d rss / dx = 2 (nRef * x - sumRef)
        grad += (cell.nRef * x - cell.sumRef) / sr2;
        hess += cell.nRef / sr2;
      }
    }
    return wantDeriv ? { nll, grad, hess } : nll;
  }

  function solveLatent(g, gi, th, resp, sigNew, sigRef, offs) {
    let x = xWork[gi];
    if (!Number.isFinite(x)) x = g.xInit;
    for (let iter = 0; iter < 80; iter += 1) {
      const d = groupNll(x, g, resp, sigNew, sigRef, offs, true);
      let h = d.hess;
      if (!(h > 1e-9)) h = 1e-9;
      let step = -d.grad / h;
      if (!Number.isFinite(step)) break;
      const cap = Math.max(0.5, g.priorSd * 3);
      if (Math.abs(step) > cap) step = Math.sign(step) * cap;
      x += step;
      if (Math.abs(step) < 1e-9) break;
    }
    return x;
  }

  function evaluate(th, coldStart = false) {
    if (globalThis.__CAL_PROF) globalThis.__CAL_PROF.n += 1;
    for (let i = 0; i < nPar; i += 1) {
      if (th[i] < bounds[i][0] || th[i] > bounds[i][1]) return { f: Infinity, grad: null };
    }
    const resp = responseFrom(th);
    const sigNew = Math.exp(th[iSigNew]);
    const sigRef = refSigmaFrom(th);
    const offs = new Map(batchIds.map((id, b) => [id, th[iBatch0 + b]]));
    if (coldStart) for (let gi = 0; gi < groups.length; gi += 1) xWork[gi] = groups[gi].xInit;
    if (debug) debug.lastTheta = th;
    const grad = new Float64Array(nPar);
    let total = 0;
    const s2 = sigNew * sigNew;
    const sr2 = sigRef * sigRef;

    for (let gi = 0; gi < groups.length; gi += 1) {
      const g = groups[gi];
      const x = solveLatent(g, gi, th, resp, sigNew, sigRef, offs);
      xWork[gi] = x;
      const dg = resp.slopeAt(x);
      const fwd = resp.forward(x);
      const kinkActive = isKink && x > th[iK] ? 1 : 0;

      // certificate prior on x
      const dp = x - g.nominal;
      total += 0.5 * (dp / g.priorSd) ** 2;

      for (const cell of g.cellList) {
        const ob = batchIndex.get(cell.batchId);
        const off = th[iBatch0 + ob];
        const mu = fwd + off;

        if (cell.nNew > 0) {
          for (const y of cell.newValues) {
            const r = y - mu;
            total += 0.5 * (LOG_2PI + 2 * Math.log(sigNew)) + (r * r) / (2 * s2);
            // d nll / d mu = -r/s^2
            const gMu = -r / s2;
            grad[iA] += gMu;
            grad[iB] += gMu * x;
            if (isKink) {
              grad[iD] += gMu * Math.max(0, x - th[iK]);
              grad[iK] += -gMu * th[iD] * kinkActive;
            }
            grad[iBatch0 + ob] += gMu;
            // d nll / d log(sigNew) = 1 - r^2/s^2
            grad[iSigNew] += 1 - (r * r) / s2;
          }
        }

        for (const c of cell.censored) {
          if (c.kind === 'left') {
            const z = (c.bound - mu) / sigNew;
            const lam = invMills(z);
            total -= logPhi(z);
            const gMu = lam / sigNew;
            grad[iA] += gMu;
            grad[iB] += gMu * x;
            if (isKink) {
              grad[iD] += gMu * Math.max(0, x - th[iK]);
              grad[iK] += -gMu * th[iD] * kinkActive;
            }
            grad[iBatch0 + ob] += gMu;
            grad[iSigNew] += lam * z;
          } else {
            const z = (mu - c.bound) / sigNew;
            const lam = invMills(z);
            total -= logPhi(z);
            const gMu = -lam / sigNew;
            grad[iA] += gMu;
            grad[iB] += gMu * x;
            if (isKink) {
              grad[iD] += gMu * Math.max(0, x - th[iK]);
              grad[iK] += -gMu * th[iD] * kinkActive;
            }
            grad[iBatch0 + ob] += gMu;
            grad[iSigNew] += lam * z;
          }
        }

        if (cell.nRef > 0) {
          const rss = cell.sumRef2 - 2 * x * cell.sumRef + cell.nRef * x * x;
          total += 0.5 * cell.nRef * LOG_2PI + cell.nRef * Math.log(sigRef) + rss / (2 * sr2);
          if (iSigRef >= 0) grad[iSigRef] += cell.nRef - rss / sr2;
        }
      }
    }
    for (let b = 0; b < nBatch; b += 1) {
      const v = th[iBatch0 + b];
      total += 0.5 * (v / 0.5) ** 2;
      grad[iBatch0 + b] += v / 0.25;
    }
    return { f: total, grad };
  }

  function nll(th) {
    return evaluate(th).f;
  }

  const free = [];
  for (let i = 0; i < nPar; i += 1) free.push(i);
  // Every evaluation cold-starts the latent solver so the objective is a
  // deterministic function of theta (no path dependence across optimiser steps).
  const objective = (th) => evaluate(th, true);
  if (debug) {
    debug.objective = objective;
    debug.theta0 = Float64Array.from(theta);
    debug.startNll = objective(theta);
    debug.meta = { iA, iB, iD, iK, iSigNew, iSigRef, iBatch0, nPar, bounds };
    debug.xWork = xWork;
    debug.groupNll = (x, g, th) => {
      const resp = responseFrom(th);
      const offs = new Map(batchIds.map((id, b) => [id, th[iBatch0 + b]]));
      return groupNll(x, g, resp, Math.exp(th[iSigNew]), refSigmaFrom(th), offs, true);
    };
    debug.groups = groups;
    debug.xAt = () => Array.from(xWork);
    debug.groupGrad = (gi) => {
      const th = debug.lastTheta;
      const resp = responseFrom(th);
      const offs = new Map(batchIds.map((id, b) => [id, th[iBatch0 + b]]));
      return groupNll(xWork[gi], groups[gi], resp, Math.exp(th[iSigNew]), refSigmaFrom(th), offs, true);
    };
  }
  const numericObjective = (th) => objective(th).f;
  // Start the quasi-Newton search from the inverse of the observed information
  // at theta0 rather than a diagonal guess: slope and intercept are strongly
  // anti-correlated, and a diagonal start made BFGS crawl along that ridge.
  const hess0 = numericHessian(numericObjective, theta, free);
  const scale = estimateCurvatureDiag(numericObjective, theta, free);
  const cov0 = invertInformation(hess0, free, nPar);
  const h0 = new Float64Array(nPar * nPar);
  for (let i = 0; i < nPar; i += 1) for (let j = 0; j < nPar; j += 1) h0[i * nPar + j] = cov0[i * nPar + j];
  // `objective` returns the profiled NLL together with its exact gradient, so
  // the quasi-Newton search runs on analytic derivatives.
  const result = minimise(objective, theta, free, {
    ...optimizer,
    scale,
    h0,
    refreshHessian: (xx) => numericHessian(numericObjective, xx, free),
  });
  const fOpt = result.f;
  if (!Number.isFinite(fOpt)) throw new Error('模型未能收敛到有效解');

  const hess = numericHessian(numericObjective, result.x, free);
  const cov = invertInformation(hess, free, nPar);
  const se = (i) => {
    if (i < 0) return null;
    const v = cov[i * nPar + i];
    return Number.isFinite(v) && v >= 0 ? Math.sqrt(v) : NaN;
  };

  const params = {
    model,
    intercept: result.x[iA],
    slope: result.x[iB],
    kink: isKink ? result.x[iK] : null,
    kinkDelta: isKink ? result.x[iD] : 0,
    sNew: Math.exp(result.x[iSigNew]),
    sRef: refSigmaFrom(result.x),
    slopeSe: se(iB),
    interceptSe: se(iA),
    kinkSe: se(iK),
    kinkDeltaSe: se(iD),
    sNewSe: Math.exp(result.x[iSigNew]) * se(iSigNew),
    sRefSe: iSigRef >= 0 ? Math.exp(result.x[iSigRef]) * se(iSigRef) : null,
  };

  const latent = groups.map((g, gi) => {
    const x = xWork[gi];
    const d = groupNll(x, g, responseFrom(result.x), params.sNew, params.sRef, new Map(batchIds.map((id, b) => [id, result.x[iBatch0 + b]])), true);
    return {
      referenceId: g.referenceId,
      range: g.range,
      nominal: g.nominal,
      x,
      se: d.hess > 0 ? Math.sqrt(1 / d.hess) : NaN,
      nRows: g.rows.length,
    };
  });

  const batchOffsets = batchIds.map((id, b) => {
    const j = iBatch0 + b;
    return { id, offset: result.x[j], se: se(j) };
  });

  return {
    model,
    params,
    cov,
    nPar,
    free,
    index: { iA, iB, iD, iK, iSigNew, iSigRef, iBatch0 },
    batchIds,
    batchOffsets,
    latent,
    groups,
    nll: fOpt,
    nObservations: groups.reduce((s, g) => s + g.rows.length, 0),
    nReferences: groups.length,
    optimizer: result,
    nobsParams: { nPar, nBatch },
  };
}

// --------------------------------------------------------------- optimisation

/** fn may return a number or { f, grad }; gradients are used when supplied. */
function fxOf(v) {
  return typeof v === 'number' ? v : v.f;
}

function minimise(fn, x0, free, { maxIter = 300, tol = 1e-10, scale = null, h0 = null, refreshHessian = null, trace = null } = {}) {
  const n = free.length;
  const x = Float64Array.from(x0);
  let first = fn(x);
  let fx = fxOf(first);
  if (!Number.isFinite(fx)) throw new Error('初始参数无法给出有限似然');
  const analytic = first && typeof first === 'object' && first.grad;
  const diag = (i) => (scale && Number.isFinite(scale[free[i]]) ? scale[free[i]] : 1);
  // B holds an approximation to the *inverse* Hessian, so the search direction
  // is a matrix-vector product rather than a linear solve. (Earlier versions
  // solved B·dir = g here, which silently used the inverse-Hessian as a
  // Hessian and degraded every step to steepest descent.)
  const B = new Float64Array(n * n);
  if (h0) {
    B.set(h0);
  } else {
    for (let i = 0; i < n; i += 1) B[i * n + i] = diag(i);
  }
  const dirFrom = (gv) => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      let v = 0;
      for (let j = 0; j < n; j += 1) v += B[i * n + j] * gv[j];
      out[i] = -v;
    }
    return out;
  };
  const gradAt = (v, at) => {
    if (analytic) {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i += 1) out[i] = v.grad[free[i]];
      return out;
    }
    return grad((t) => fxOf(fn(t)), at, free);
  };
  let g = gradAt(first, x);
  let iter = 0;
  let converged = false;
  let stall = 0;
  for (; iter < maxIter; iter += 1) {
    let dir = dirFrom(g);
    let slope = 0;
    for (let i = 0; i < n; i += 1) slope += g[i] * dir[i];
    if (!(slope < 0)) {
      dir = Float64Array.from(g, (v) => -v);
      for (let i = 0; i < n * n; i += 1) B[i] = 0;
      for (let i = 0; i < n; i += 1) B[i * n + i] = diag(i);
      slope = 0;
      for (let i = 0; i < n; i += 1) slope -= g[i] * g[i];
    }
    let step = 1;
    let xn = null;
    let fnx = Infinity;
    let fvn = null;
    for (let ls = 0; ls < 40; ls += 1) {
      const trial = Float64Array.from(x);
      for (let i = 0; i < n; i += 1) trial[free[i]] += step * dir[i];
      const fv = fn(trial);
      const ft = fxOf(fv);
      if (Number.isFinite(ft) && ft <= fx + 1e-4 * step * slope) {
        xn = trial;
        fnx = ft;
        fvn = fv;
        break;
      }
      step *= 0.5;
    }
    if (!xn) break;
    const gn = gradAt(fvn, xn);
    const s = new Float64Array(n);
    const yv = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      s[i] = xn[free[i]] - x[free[i]];
      yv[i] = gn[i] - g[i];
    }
    let sy = 0;
    for (let i = 0; i < n; i += 1) sy += s[i] * yv[i];
    for (let i = 0; i < n; i += 1) x[free[i]] = xn[free[i]];
    const improvement = fx - fnx;
    fx = fnx;
    g = gn;
    if (trace) trace.push({ iter, fx, step, g0: g[0], gmax: Math.max(...g.map(Math.abs)) });
    if (sy > 1e-12) {
      // BFGS update of the inverse Hessian.
      const Bs = new Float64Array(n);
      for (let i = 0; i < n; i += 1) {
        let v = 0;
        for (let j = 0; j < n; j += 1) v += B[i * n + j] * s[j];
        Bs[i] = v;
      }
      const yBs = (() => {
        let v = 0;
        for (let i = 0; i < n; i += 1) v += yv[i] * Bs[i];
        return v;
      })();
      const phi = (1 + yBs / sy) / sy;
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
          B[i * n + j] += (s[i] * s[j] * phi) - (Bs[i] * s[j] + s[i] * Bs[j]) / sy;
        }
      }
    }
    if (trace) trace[trace.length - 1].gmaxOut = Math.max(...g.map(Math.abs));
    if (improvement < tol * (1 + Math.abs(fx)) && step > 1e-3) {
      converged = true;
      break;
    }
    // Near-flat directions (the two repeatabilities, a batch offset resting on
    // one reference) make plain BFGS crawl. If progress stalls for several
    // iterations, rebuild the inverse-Hessian approximation from a fresh
    // numeric Hessian at the current point.
    if (improvement < 1e-4 * (1 + Math.abs(fx))) stall += 1;
    else stall = 0;
    if (stall >= 8 && refreshHessian) {
      const Hn = refreshHessian(x);
      if (Hn) {
        const Bn = invertSPDOrNull(Hn, n);
        if (Bn) B.set(Bn);
      }
      stall = 0;
    }
  }
  return { x, f: fx, iter, converged, grad: g };
}

/** Invert a symmetric matrix; return null when it is not usable. */
function invertSPDOrNull(H, n) {
  const inv = new Float64Array(n * n);
  for (let col = 0; col < n; col += 1) {
    const e = new Float64Array(n);
    e[col] = 1;
    const x = solveLinear(H, e, n);
    if (!x) return null;
    for (let row = 0; row < n; row += 1) {
      if (!Number.isFinite(x[row])) return null;
      inv[row * n + col] = x[row];
    }
  }
  return inv;
}

function grad(fn, x, free) {
  const n = free.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const j = free[i];
    const h = Math.max(1e-6, Math.abs(x[j]) * 1e-6 + 1e-7);
    const xp = Float64Array.from(x);
    const xm = Float64Array.from(x);
    xp[j] += h;
    xm[j] -= h;
    const fp = fn(xp);
    const fm = fn(xm);
    out[i] = Number.isFinite(fp) && Number.isFinite(fm) ? (fp - fm) / (2 * h) : 0;
  }
  return out;
}

/**
 * Diagonal preconditioner for the quasi-Newton search: the inverse of the
 * observed curvature, so H0 approximates the inverse Hessian that BFGS updates.
 */
function estimateCurvatureDiag(fn, x, free) {
  const out = new Float64Array(x.length).fill(1);
  for (let i = 0; i < free.length; i += 1) {
    const j = free[i];
    const h = Math.max(1e-4, Math.abs(x[j]) * 1e-4 + 1e-5);
    const xp = Float64Array.from(x);
    const xm = Float64Array.from(x);
    xp[j] += h;
    xm[j] -= h;
    const f0 = fxOf(fn(x));
    const curv = (fxOf(fn(xp)) - 2 * f0 + fxOf(fn(xm))) / (h * h);
    out[j] = Number.isFinite(curv) && curv > 1e-6 ? 1 / Math.min(1e6, curv) : 1;
  }
  return out;
}

export function numericHessian(fn, x, free) {
  const n = free.length;
  const hess = new Float64Array(n * n);
  const f0 = fn(x);
  const hstep = (i) => Math.max(2e-4, Math.abs(x[free[i]]) * 2e-4 + 1e-5);
  for (let i = 0; i < n; i += 1) {
    const hi = hstep(i);
    const xp = Float64Array.from(x);
    const xm = Float64Array.from(x);
    xp[free[i]] += hi;
    xm[free[i]] -= hi;
    hess[i * n + i] = (fn(xp) - 2 * f0 + fn(xm)) / (hi * hi);
    for (let k = i + 1; k < n; k += 1) {
      const hk = hstep(k);
      const xpp = Float64Array.from(x);
      const xpm = Float64Array.from(x);
      const xmp = Float64Array.from(x);
      const xmm = Float64Array.from(x);
      xpp[free[i]] += hi;
      xpp[free[k]] += hk;
      xpm[free[i]] += hi;
      xpm[free[k]] -= hk;
      xmp[free[i]] -= hi;
      xmp[free[k]] += hk;
      xmm[free[i]] -= hi;
      xmm[free[k]] -= hk;
      const v = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * hi * hk);
      hess[i * n + k] = v;
      hess[k * n + i] = v;
    }
  }
  return hess;
}

function solveLinear(A0, b0, n) {
  const A = Float64Array.from(A0);
  const b = Float64Array.from(b0);
  for (let col = 0; col < n; col += 1) {
    let piv = col;
    for (let r = col + 1; r < n; r += 1) if (Math.abs(A[r * n + col]) > Math.abs(A[piv * n + col])) piv = r;
    if (Math.abs(A[piv * n + col]) < 1e-12) return null;
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
    if (Math.abs(A[r * n + r]) < 1e-12) return null;
    x[r] = s / A[r * n + r];
  }
  return x;
}

function invertInformation(hess, free, nPar) {
  const n = free.length;
  const inv = new Float64Array(n * n);
  let scale = 0;
  for (let i = 0; i < n; i += 1) scale = Math.max(scale, Math.abs(hess[i * n + i]));
  const ridge = Math.max(1e-8, scale * 1e-8);
  const A = Float64Array.from(hess);
  for (let i = 0; i < n; i += 1) A[i * n + i] += ridge;
  for (let col = 0; col < n; col += 1) {
    const e = new Float64Array(n);
    e[col] = 1;
    const x = solveLinear(A, e, n);
    if (!x) continue;
    for (let r = 0; r < n; r += 1) inv[r * n + col] = x[r];
  }
  const out = new Float64Array(nPar * nPar);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < n; k += 1) out[free[i] * nPar + free[k]] = inv[i * n + k];
  }
  return out;
}

export function cholesky(A, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k += 1) s -= L[i * n + k] * L[j * n + k];
      if (i === j) L[i * n + j] = Math.sqrt(Math.max(s, 1e-14));
      else L[i * n + j] = s / L[j * n + j];
    }
  }
  return L;
}

// -------------------------------------------------------- inverse prediction

/**
 * Convert one new-device reading to the reference scale.
 *
 * Two independent sources of uncertainty are propagated by Monte Carlo over the
 * fitted parameter covariance: the response parameters themselves, and the
 * single-reading measurement error of the reading being converted. The interval
 * is therefore wider than a confidence band for the calibration curve, which is
 * the correct interval for a *new* reading.
 *
 * `batchId` (optional) adds that batch's shared offset, drawn from its own
 * posterior; passing null converts as if the offset were perfectly known at the
 * fitted value minus its estimate, i.e. it treats the batch as identified.
 *
 * Returns `refused: true` when the range has no identifiable correspondence.
 */
export function invertReading(reading, fit, { seed = 7, draws = 3000, coverage = 0.95, batchId = null, ranges = null, references = null } = {}) {
  const { params, cov, nPar, index, model } = fit;
  const rand = mulberry32(seed);
  const L = cholesky(cov, nPar);
  const xs = [];
  const { iA, iB, iD, iK, iSigNew, iBatch0 } = index;
  let batchIndex = -1;
  if (batchId) {
    const bi = fit.batchIds.indexOf(batchId);
    if (bi >= 0) batchIndex = iBatch0 + bi;
  }
  const batchOffset = batchIndex >= 0 ? fit.batchOffsets[fit.batchIds.indexOf(batchId)] : null;
  const offPoint = batchOffset ? batchOffset.offset : 0;
  for (let d = 0; d < draws; d += 1) {
    const delta = new Float64Array(nPar);
    for (let i = 0; i < nPar; i += 1) {
      let s = 0;
      for (let j = 0; j <= i; j += 1) s += L[i * nPar + j] * normal(rand);
      delta[i] = s;
    }
    const b = params.slope + delta[iB];
    const sig = Math.exp(Math.log(params.sNew) + delta[iSigNew]);
    if (!(b > 0.05) || !(sig > 1e-4) || sig > 100) continue;
    const off = batchIndex >= 0 ? offPoint + delta[batchIndex] : 0;
    const resp = makeResponse({
      model,
      intercept: params.intercept + delta[iA],
      slope: b,
      kink: model === 'kink' ? params.kink + delta[iK] : NaN,
      kinkDelta: model === 'kink' ? params.kinkDelta + delta[iD] : 0,
    });
    const y = reading + normal(rand, 0, sig);
    const x = resp.inverse(y - off);
    if (Number.isFinite(x)) xs.push(x);
  }
  xs.sort((p, q) => p - q);
  const point = makeResponse(params).inverse(reading - offPoint);
  const q = (p) => {
    if (!xs.length) return NaN;
    const h = (xs.length - 1) * p;
    const lo = Math.floor(h);
    const hi = Math.ceil(h);
    return xs[lo] + (xs[hi] - xs[lo]) * (h - lo);
  };
  const half = (1 - coverage) / 2;
  const range = ranges && references ? ranges.find((r) => reading >= r.nominal && reading < r.upper) : null;
  return {
    reading,
    batchId,
    batchOffset: offPoint,
    batchOffsetSe: batchOffset ? batchOffset.se : null,
    estimate: point,
    lo: q(half),
    hi: q(1 - half),
    sd: sd(xs),
    accepted: xs.length,
    draws,
    coverage,
    rangeId: range ? range.id : null,
  };
}

// ---------------------------------------------------------------- diagnostics

/** Residuals in both directions for every retained row. */
export function residuals(fit, { excludeIds = new Set() } = {}) {
  const resp = makeResponse(fit.params);
  const batchMap = new Map(fit.batchOffsets.map((b) => [b.id, b.offset]));
  const latentMap = new Map(fit.latent.map((l) => [l.referenceId, l]));
  const rows = [];
  for (const g of fit.groups) {
    const lat = latentMap.get(g.referenceId);
    for (const m of g.rows) {
      if (excludeIds.has(m.id)) continue;
      const x = lat.x;
      const off = batchMap.get(m.batchId) || 0;
      const predicted = resp.forward(x) + off;
      // Censored rows get no finite numeric residual: the measurement only says
      // "below/above this bound". They still appear in the table, flagged.
      const censored = m.belowLod || m.saturated;
      const forwardResid = censored ? null : m.reading - predicted;
      const standardized = censored ? null : forwardResid / fit.params.sNew;
      const back = censored ? NaN : resp.inverse(m.reading - off);
      const old = Number.isFinite(m.oldReading) ? m.oldReading : m.refReading;
      const reverseResid = Number.isFinite(back) && Number.isFinite(old) ? back - old : null;
      const revStd =
        Number.isFinite(reverseResid) && fit.params.sNew > 0
          ? reverseResid / Math.sqrt(fit.params.sNew ** 2 + fit.params.sRef ** 2)
          : null;
      rows.push({
        id: m.id,
        referenceId: m.referenceId,
        range: g.range,
        batchId: m.batchId,
        reading: m.reading,
        oldReading: Number.isFinite(old) ? old : NaN,
        latentX: x,
        fitted: predicted,
        forwardResid,
        standardized,
        reverseResid,
        reverseStandardized: revStd,
        oldDeviceResid: Number.isFinite(old) ? old - x : NaN,
        belowLod: m.belowLod,
        saturated: m.saturated,
        censored,
        importedFlag: m.importedFlag || '',
        flagged: Number.isFinite(standardized) && Math.abs(standardized) > 3.5,
      });
    }
  }
  return rows;
}

export function residualSummary(rows) {
  // Censored rows carry only an inequality, so they never enter the residual
  // statistics; they are counted separately.
  const f = rows.filter((r) => !r.censored && Number.isFinite(r.forwardResid));
  const rv = rows.filter((r) => !r.censored && Number.isFinite(r.reverseResid));
  return {
    n: f.length,
    mean: mean(f.map((r) => r.forwardResid)),
    rmse: Math.sqrt(mean(f.map((r) => r.forwardResid ** 2))),
    reverseMean: mean(rv.map((r) => r.reverseResid)),
    reverseRmse: Math.sqrt(mean(rv.map((r) => r.reverseResid ** 2))),
    flagged: rows.filter((r) => r.flagged).length,
    belowLod: rows.filter((r) => r.belowLod).length,
    saturated: rows.filter((r) => r.saturated).length,
  };
}

// ---------------------------------------------------------------- comparison

/**
 * Compare models by repeated holdout over whole reference samples.
 * Training sees only the training references; scoring uses the held-out rows.
 */
export function holdoutCompare({
  measurements,
  references,
  models = ['linear', 'kink'],
  holdoutFraction = 0.3,
  seed = 99,
  repeats = 5,
  fitOptions = {},
  kinkRange = null,
}) {
  const active = references.filter((r) => r.status === 'active');
  if (active.length < 6) throw new Error('独立留样比较至少需要 6 个有效参考样本');
  const rand = mulberry32(seed);
  const ids = active.map((r) => r.id);
  const results = [];
  const usedHoldouts = [];
  for (let rep = 0; rep < repeats; rep += 1) {
    const shuffled = ids.slice();
    for (let i = shuffled.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const nHold = Math.max(4, Math.round(shuffled.length * holdoutFraction));
    const holdIds = new Set(shuffled.slice(0, nHold));
    usedHoldouts.push([...holdIds]);
    const trainMeasurements = measurements.filter((m) => !holdIds.has(m.referenceId));
    const trainReferences = references.filter((r) => !holdIds.has(r.id));
    for (const model of models) {
      let fit;
      try {
        fit = fitModel({
          measurements: trainMeasurements,
          references: trainReferences,
          model,
          kinkRange,
          ...fitOptions,
        });
      } catch (err) {
        results.push({ rep, model, ok: false, error: err.message });
        continue;
      }
      // Score on the held-out samples, which the fit never saw: predict the new
      // device's reading for each held-out cell from the certificate value and
      // the batch offset the trained model assigns to that batch, then compare
      // with the cell's observed uncensored mean. Whole samples move together,
      // so the comparison never leaks repeats of one sample across the split.
      const resp = makeResponse(fit.params);
      const batchMap = new Map(fit.batchOffsets.map((b) => [b.id, b.offset]));
      const holdRows = [];
      for (const m of measurements) {
        if (!holdIds.has(m.referenceId)) continue;
        if (m.belowLod || m.saturated || !Number.isFinite(m.reading)) continue;
        holdRows.push(m);
      }
      const byCell = new Map();
      for (const m of holdRows) {
        const key = `${m.referenceId}|${m.batchId}`;
        if (!byCell.has(key)) byCell.set(key, []);
        byCell.get(key).push(m);
      }
      const certOf = new Map(references.map((r) => [r.id, r.certified]));
      const errs = [];
      let uncoveredCells = 0;
      for (const cellRows of byCell.values()) {
        const batchId = cellRows[0].batchId;
        if (!batchMap.has(batchId)) {
          uncoveredCells += 1;
          continue;
        }
        const x = certOf.get(cellRows[0].referenceId);
        if (!Number.isFinite(x)) continue;
        const predicted = resp.forward(x) + batchMap.get(batchId);
        const observed = mean(cellRows.map((r) => r.reading));
        errs.push(observed - predicted);
      }
      const trainRows = residuals(fit).filter((r) => !r.belowLod && !r.saturated && Number.isFinite(r.forwardResid));
      results.push({
        rep,
        model,
        ok: errs.length > 0,
        error: errs.length ? undefined : '留出样本没有可评分的批次',
        nHoldRows: holdRows.length,
        nHoldCells: errs.length,
        nHoldRefs: holdIds.size,
        uncoveredCells,
        rmse: errs.length ? Math.sqrt(mean(errs.map((e) => e * e))) : NaN,
        mae: errs.length ? mean(errs.map(Math.abs)) : NaN,
        bias: errs.length ? mean(errs) : NaN,
        trainRmse: trainRows.length ? Math.sqrt(mean(trainRows.map((r) => r.forwardResid ** 2))) : NaN,
        params: {
          intercept: fit.params.intercept,
          slope: fit.params.slope,
          kink: fit.params.kink,
          kinkDelta: fit.params.kinkDelta,
        },
      });
    }
  }
  const summary = {};
  for (const model of models) {
    const ok = results.filter((r) => r.model === model && r.ok && Number.isFinite(r.rmse));
    if (!ok.length) {
      summary[model] = { ok: false, rmse: NaN, mae: NaN, bias: NaN, reps: 0, preferred: false };
      continue;
    }
    summary[model] = {
      ok: true,
      reps: ok.length,
      rmse: mean(ok.map((r) => r.rmse)),
      rmseSe: sd(ok.map((r) => r.rmse)) / Math.sqrt(ok.length),
      mae: mean(ok.map((r) => r.mae)),
      bias: mean(ok.map((r) => r.bias)),
      trainRmse: mean(ok.filter((r) => Number.isFinite(r.trainRmse)).map((r) => r.trainRmse)),
      preferred: false,
    };
  }
  const usable = models.filter((m) => summary[m]?.ok);
  let preferred = null;
  if (usable.length) preferred = usable.reduce((best, m) => (summary[m].rmse < summary[best].rmse ? m : best), usable[0]);
  if (preferred) summary[preferred].preferred = true;
  const alt = usable.find((m) => m !== preferred);
  const delta = preferred && alt ? summary[alt].rmse - summary[preferred].rmse : NaN;
  return { results, summary, preferred, delta, seed, repeats, holdoutFraction, usedHoldouts };
}

// -------------------------------------------------------------- identifiability

/**
 * Which ranges have an identifiable correspondence.
 *
 * A range is convertible only when at least `minReferences` distinct reference
 * samples have an uncensored new-device reading there and the latent reference
 * values span more than a single point. R4 in this dataset fails that test: all
 * of its readings are at the instrument plateau, so no correspondence exists and
 * pointwise conversion must be refused.
 *
 * `uncovered` is informational: parts of the range's nominal interval that lie
 * outside the reference span. Those readings need an extrapolation decision,
 * which `convertReading` makes per reading, not a blanket refusal.
 */
export function rangeIdentifiability(fit, ranges, references, { minReferences = 2 } = {}) {
  const active = new Set(references.filter((r) => r.status === 'active').map((r) => r.id));
  const support = new Map();
  for (const g of fit.groups) {
    if (!active.has(g.referenceId)) continue;
    const hasUncensored = g.rows.some((r) => !r.belowLod && !r.saturated);
    const lat = fit.latent.find((l) => l.referenceId === g.referenceId);
    if (!lat) continue;
    let s = support.get(g.range);
    if (!s) {
      s = { refs: new Set(), xs: [], uncensoredRefs: new Set() };
      support.set(g.range, s);
    }
    s.refs.add(g.referenceId);
    if (hasUncensored) {
      s.uncensoredRefs.add(g.referenceId);
      s.xs.push(lat.x);
    }
  }
  const out = {};
  for (const r of ranges) {
    const s = support.get(r.id);
    const xs = s ? s.xs.slice().sort((a, b) => a - b) : [];
    const lo = xs.length ? xs[0] : NaN;
    const hi = xs.length ? xs[xs.length - 1] : NaN;
    const identifiableRefs = s ? s.uncensoredRefs.size : 0;
    let reason = '';
    if (!s || identifiableRefs === 0) reason = '该量程没有未饱和/未截尾的参考读数，对应关系不可识别';
    else if (identifiableRefs < minReferences) reason = `该量程仅 ${identifiableRefs} 个可用参考样本，低于最低 ${minReferences} 个`;
    else if (!(hi - lo > 1e-6)) reason = '该量程参考样本集中在单点，无法识别斜率';
    const uncovered = [];
    if (!reason) {
      const margin = Math.max(0.5, 0.01 * (r.upper - r.nominal));
      if (r.nominal < lo - margin) uncovered.push([r.nominal, Math.max(r.nominal, lo)]);
      if (r.upper > hi + margin) uncovered.push([Math.min(r.upper, hi), r.upper]);
    }
    out[r.id] = {
      ok: !reason,
      nReferences: s ? s.refs.size : 0,
      identifiableReferences: identifiableRefs,
      span: { lo, hi },
      uncovered,
      extrapolation: uncovered.length > 0,
      reason,
    };
  }
  return out;
}

/** Saturation plateau supported by the retained data, if any. */
export function saturationInfo(measurements, references) {
  const active = new Set(references.filter((r) => r.status === 'active').map((r) => r.id));
  const sat = measurements.filter((m) => m.saturated && active.has(m.referenceId));
  if (sat.length < 2) return null;
  return {
    n: sat.length,
    plateau: mean(sat.map((m) => m.reading)),
    min: Math.min(...sat.map((m) => m.reading)),
    max: Math.max(...sat.map((m) => m.reading)),
    ranges: [...new Set(sat.map((m) => m.range))],
  };
}

/** Lowest identifiable reference-scale value the fitted curve supports. */
export function identifiabilityWindow(fit, ranges, references) {
  const ident = rangeIdentifiability(fit, ranges, references);
  const trainingMin = Math.min(...fit.groups.map((g) => g.rows[0]?.reading ?? Infinity));
  return {
    ident,
    okRanges: Object.entries(ident).filter(([, d]) => d.ok).map(([id]) => id),
    blockedRanges: Object.entries(ident).filter(([, d]) => !d.ok).map(([id]) => id),
    trainingMin,
  };
}

/**
 * Decision layer for converting operational readings.
 *
 * A reading is converted only when its range has an identifiable
 * correspondence. Within such a range the reading must also lie inside the
 * reference span the fit actually covers (with a small margin); otherwise the
 * conversion would be an extrapolation and is refused with a reason instead of
 * a number. Saturated readings are reported as "no correspondence", because the
 * new device's plateau carries no information about the reference value.
 */
export function convertReading(reading, fit, { ranges, references, batchId = null, coverage = 0.95, seed = 7, saturated = false, belowLod = false } = {}) {
  const range = ranges.find((r) => reading >= r.nominal && reading < r.upper) || ranges[ranges.length - 1];
  const ident = rangeIdentifiability(fit, ranges, references)[range.id];
  const base = { reading, rangeId: range.id, rangeLabel: range.label, batchId, identifiable: ident.ok };
  if (!ident.ok) {
    return { ...base, ok: false, refused: true, reason: ident.reason || '该量程没有可识别的对应关系', warnings: [] };
  }
  if (saturated) {
    return {
      ...base,
      ok: false,
      refused: true,
      reason: `读数已到新设备饱和平台，不能反推参考值（${range.label}仍可识别，但此读数不可用）`,
      warnings: [],
    };
  }
  const warnings = [];
  if (belowLod) warnings.push('读数在检出限附近，参考值区间偏宽');
  if (batchId) {
    const has = fit.batchIds.includes(batchId);
    if (!has) {
      return { ...base, ok: false, refused: true, reason: `批次 ${batchId} 不在本次标定中，缺少该批次的共同偏移`, warnings };
    }
    const bo = fit.batchOffsets[fit.batchIds.indexOf(batchId)];
    if (Number.isFinite(bo.se) && bo.se > 0.3) warnings.push(`批次 ${batchId} 的共同偏移不确定度较大（±${bo.se.toFixed(2)}）`);
  }
  const margin = Math.max(0.5, 0.01 * (range.upper - range.nominal));
  if (!(Number.isFinite(ident.span.lo) && Number.isFinite(ident.span.hi))) {
    return { ...base, ok: false, refused: true, reason: '该量程没有可用的参考跨度', warnings };
  }
  const pointReading = reading - (batchId && fit.batchIds.includes(batchId) ? fit.batchOffsets[fit.batchIds.indexOf(batchId)].offset : 0);
  const point = makeResponse(fit.params).inverse(pointReading);
  if (point < ident.span.lo - 2 * margin || point > ident.span.hi + 2 * margin) {
    return { ...base, ok: false, refused: true, reason: '该读数超出参考跨度，单点换算需要外推，已拒绝', warnings };
  }
  const inv = invertReading(reading, fit, { batchId, coverage, seed });
  return { ...base, ok: true, refused: false, reason: '', warnings, conversion: inv };
}

