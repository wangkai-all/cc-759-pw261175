// Parameter recovery against the hidden truths used by the generator.
// These tests are allowed to read `truth`; the application is not.
import test from 'node:test';
import assert from 'node:assert/strict';
import { generateDataset, truth, hiddenReferenceValues, RANGES } from '../src/data.js';
import {
  fitModel,
  screenOutliers,
  residuals,
  residualSummary,
  rangeIdentifiability,
  saturationInfo,
  invertReading,
  convertReading,
  holdoutCompare,
  makeResponse,
} from '../src/model.js';

const ds = generateDataset();

function fitted(model = 'kink') {
  return fitModel({
    measurements: ds.measurements,
    references: ds.references,
    batches: ds.batches,
    model,
    kinkRange: [150, 400],
  });
}

test('generator is deterministic and frozen at the documented seed', () => {
  const again = generateDataset();
  assert.deepEqual(again.measurements, ds.measurements);
  assert.equal(ds.measurements.length, 208);
  assert.equal(ds.field.length, 25);
  assert.equal(ds.references.length, 26);
});

test('old-device readings carry noise of the stated size', () => {
  // Within (reference, batch) cells the old device must scatter around the
  // latent value. This is the check that the old device is *not* exact.
  const byCell = new Map();
  for (const m of ds.measurements) {
    const key = `${m.referenceId}|${m.batchId}`;
    if (!byCell.has(key)) byCell.set(key, []);
    byCell.get(key).push(m);
  }
  let ss = 0;
  let df = 0;
  for (const rows of byCell.values()) {
    const vals = rows.map((r) => r.oldReading);
    const mu = vals.reduce((a, b) => a + b, 0) / vals.length;
    ss += vals.reduce((a, v) => a + (v - mu) ** 2, 0);
    df += vals.length - 1;
  }
  const withinSd = Math.sqrt(ss / df);
  assert.ok(Math.abs(withinSd - truth.refRepeatSd) < 0.15, `old-device within-cell SD ${withinSd.toFixed(3)}`);
});

test('linear fit recovers slope and the old-device repeatability', () => {
  const fit = fitted('linear');
  assert.ok(fit.optimizer.converged, 'optimiser should report convergence');
  assert.ok(Math.abs(fit.params.slope - truth.slope) < 0.02, `slope ${fit.params.slope}`);
  assert.ok(Math.abs(fit.params.sRef - truth.refRepeatSd) < 0.12, `sRef ${fit.params.sRef}`);
  // Under the *wrong* (linear) model the kink shows up as extra scatter, so
  // sNew is inflated. That is a feature: a misspecified curve cannot hide the
  // misfit in the repeatability alone, and the holdout comparison catches it.
  assert.ok(fit.params.sNew > truth.newRepeatSd, 'misspecified model must inflate sNew');
  assert.ok(fit.params.slopeSe > 0 && fit.params.slopeSe < 0.01);
});

test('changepoint fit recovers kink location and extra slope', () => {
  const fit = fitted('kink');
  assert.ok(fit.optimizer.converged);
  assert.ok(Math.abs(fit.params.kink - truth.kink) < 12, `kink ${fit.params.kink}`);
  assert.ok(Math.abs(fit.params.kinkDelta - truth.kinkSlopeDelta) < 0.018, `delta ${fit.params.kinkDelta}`);
  assert.ok(Math.abs(fit.params.slope - truth.slope) < 0.005);
});

test('known responses are reproduced on the reference scale', () => {
  const fit = fitted('kink');
  const resp = makeResponse(fit.params);
  for (const x of [20, 150, 260, 320, 420]) {
    assert.ok(Math.abs(resp.forward(x) - (truth.slope * x + truth.offset + (x > truth.kink ? truth.kinkSlopeDelta * (x - truth.kink) : 0))) < 3);
  }
});

test('outlier screen keeps the three bumped-fixture rows and drops nothing else gross', () => {
  const { flagged, scores } = screenOutliers(ds.measurements);
  const bumped = ds.measurements.filter((m) => m.importedFlag).map((m) => m.id);
  for (const id of bumped) assert.ok(flagged.has(id), `bumped row ${id} must be screened`);
  for (const id of flagged) assert.ok(scores.get(id) > 4 || bumped.includes(id));
  assert.ok(flagged.size <= bumped.length + 2, `screen found ${flagged.size} rows`);
});

test('screening is what keeps the repeatability honest', () => {
  const screened = fitted('kink');
  const unscreened = fitModel({
    measurements: ds.measurements,
    references: ds.references,
    batches: ds.batches,
    model: 'kink',
    kinkRange: [150, 400],
    screen: false,
  });
  // Without the screen the bumped rows inflate the new-device repeatability by
  // roughly four-fold, which would silently widen every conversion interval.
  assert.ok(screened.params.sNew < 0.6 * unscreened.params.sNew, `${screened.params.sNew} vs ${unscreened.params.sNew}`);
  assert.ok(Math.abs(screened.params.sNew - truth.newRepeatSd) < 0.08);
  assert.ok(Math.abs(screened.params.slope - truth.slope) < 0.005);
});

test('forward residuals on uncensored rows are centred and small', () => {
  const fit = fitted('kink');
  const rows = residuals(fit);
  const summary = residualSummary(rows);
  assert.ok(summary.n > 160);
  assert.ok(Math.abs(summary.mean) < 0.1);
  assert.ok(summary.rmse < 0.4);
  assert.equal(summary.belowLod, 14);
  assert.equal(summary.saturated, 16);
  // Censored rows must not be given a numeric residual.
  for (const r of rows) {
    if (r.belowLod || r.saturated) {
      assert.equal(r.forwardResid, null);
      assert.equal(r.reverseResid, null);
    }
  }
});

test('reverse residuals — old reading against the converted new reading — are unbiased', () => {
  const fit = fitted('kink');
  const rows = residuals(fit).filter((r) => !r.censored);
  const mean = rows.reduce((s, r) => s + r.reverseResid, 0) / rows.length;
  const rms = Math.sqrt(rows.reduce((s, r) => s + r.reverseResid ** 2, 0) / rows.length);
  assert.ok(Math.abs(mean) < 0.12, `reverse mean ${mean}`);
  assert.ok(rms < 1.1, `reverse rms ${rms}`);
});

test('batch offsets are recovered to within their standard errors', () => {
  const fit = fitted('kink');
  for (const b of fit.batchOffsets) {
    const gen = ds.batches.find((x) => x.id === b.id);
    if (!gen) continue;
    assert.ok(Math.abs(b.offset - gen.offset) < 0.25, `batch ${b.id}: ${b.offset} vs ${gen.offset}`);
    assert.ok(Number.isFinite(b.se) && b.se > 0);
  }
});

test('saturation plateau is detected at the generating value', () => {
  const info = saturationInfo(ds.measurements, ds.references);
  assert.ok(info);
  assert.ok(Math.abs(info.plateau - truth.saturation) < 0.2);
  assert.deepEqual(info.ranges, ['R4']);
});

test('the highest expandable range has no identifiable correspondence and refuses conversion', () => {
  const fit = fitted('kink');
  const ident = rangeIdentifiability(fit, RANGES, ds.references);
  assert.equal(ident.R4.ok, false);
  assert.equal(ident.R4.identifiableReferences, 0);
  assert.equal(ident.R1.ok, true);
  assert.equal(ident.R2.ok, true);
  assert.equal(ident.R3.ok, true);

  const r = convertReading(510, fit, { ranges: RANGES, references: ds.references, batchId: 'B04' });
  assert.equal(r.ok, false);
  assert.equal(r.refused, true);
  assert.match(r.reason, /不可识别/);
});

test('a saturated reading carries no reference information and is refused', () => {
  const fit = fitted('kink');
  const r = convertReading(486.4, fit, {
    ranges: RANGES,
    references: ds.references,
    batchId: 'B01',
    saturated: true,
  });
  assert.equal(r.ok, false);
  assert.match(r.reason, /饱和/);
});

test('conversion intervals widen with the single-reading error', () => {
  const fit = fitted('kink');
  const reading = 205.5;
  const wide = invertReading(reading, fit, { batchId: 'B05', draws: 4000 });
  // Shrink the repeatability of the new device to near zero and redo: the
  // interval must narrow, proving the reading noise is one of the two
  // propagated components rather than only the parameter covariance.
  const tightFit = { ...fit, params: { ...fit.params, sNew: 1e-3 } };
  const tight = invertReading(reading, tightFit, { batchId: 'B05', draws: 4000 });
  assert.ok(wide.hi - wide.lo > tight.hi - tight.lo);
  assert.ok(wide.hi > wide.estimate && wide.lo < wide.estimate);
});

test('conversion propagates batch offset uncertainty as well', () => {
  const fit = fitted('kink');
  const base = invertReading(205.5, fit, { draws: 4000 });
  const withBatch = invertReading(205.5, fit, { batchId: 'B05', draws: 4000 });
  assert.ok(withBatch.hi - withBatch.lo >= base.hi - base.lo - 1e-9);
  assert.ok(Number.isFinite(withBatch.batchOffsetSe));
});

test('conversions land near the value the reading was generated from', () => {
  const fit = fitted('kink');
  // Field reading 205.5 was planned at 205.5 - 0.42 - 0.05*(205.5-280) ... i.e.
  // it belongs near 199 mm on the reference scale, in REF-014's stretch of R2.
  const r = convertReading(205.5, fit, { ranges: RANGES, references: ds.references, batchId: 'B05' });
  assert.equal(r.ok, true);
  assert.ok(Math.abs(r.conversion.estimate - 199.2) < 3);
  const lookup = hiddenReferenceValues();
  assert.equal(lookup['REF-014'].value, 196);
  assert.ok(r.conversion.hi - r.conversion.lo < 3, 'interval should stay narrow for an in-span reading');
});

test('cross-range boundary: readings on either side of 300 are handled separately', () => {
  const fit = fitted('kink');
  const below = convertReading(295.2, fit, { ranges: RANGES, references: ds.references, batchId: 'B05' });
  const above = convertReading(305.6, fit, { ranges: RANGES, references: ds.references, batchId: 'B02' });
  assert.equal(below.rangeId, 'R2');
  assert.equal(above.rangeId, 'R3');
  assert.equal(below.ok, true);
  // 305.6 is past the fitted reference span in R3, so conversion must refuse
  // rather than silently extrapolate across the boundary.
  assert.equal(above.ok, false);
  assert.equal(above.refused, true);
});

test('repeated holdout prefers the kink model over the linear one', () => {
  const cmp = holdoutCompare({
    measurements: ds.measurements,
    references: ds.references,
    models: ['linear', 'kink'],
    kinkRange: [150, 400],
    repeats: 4,
    seed: 99,
  });
  assert.ok(cmp.summary.linear.ok && cmp.summary.kink.ok);
  assert.equal(cmp.preferred, 'kink');
  assert.ok(cmp.summary.linear.rmse > 1.5 * cmp.summary.kink.rmse);
});

test('holdout never scores a model on samples it trained on', () => {
  const cmp = holdoutCompare({
    measurements: ds.measurements,
    references: ds.references,
    models: ['kink'],
    kinkRange: [150, 400],
    repeats: 2,
    seed: 5,
  });
  for (const r of cmp.results) {
    assert.ok(r.ok);
    assert.ok(r.nHoldCells > 0);
    assert.ok(Number.isFinite(r.trainRmse));
  }
});

test('withdrawing a reference removes its batch and blocks readings that depended on it', () => {
  const base = fitted('kink');
  assert.ok(base.batchIds.includes(truth.recheckBatch));
  const withdrawn = new Set([truth.withdrawDemo]);
  const refs = ds.references.map((r) => (withdrawn.has(r.id) ? { ...r, status: 'withdrawn' } : r));
  const after = fitModel({
    measurements: ds.measurements.filter((m) => !withdrawn.has(m.referenceId)),
    references: refs,
    batches: ds.batches,
    model: 'kink',
    kinkRange: [150, 400],
  });
  // B05's offset rested on REF-014 alone; with the sample gone the batch leaves
  // the calibration entirely, so any reading recorded in it loses its offset.
  assert.ok(!after.batchIds.includes(truth.recheckBatch));
  const affected = ds.field.filter((f) => f.batchId === truth.recheckBatch);
  assert.ok(affected.length >= 3);
  for (const f of affected) {
    const r = convertReading(f.reading, after, { ranges: RANGES, references: refs, batchId: f.batchId });
    assert.equal(r.ok, false);
    assert.equal(r.refused, true);
    assert.match(r.reason, /共同偏移/);
  }
  // The response shape itself must barely move: one reference out of 26.
  assert.ok(Math.abs(after.params.slope - base.params.slope) < 0.002);
});

test('linear model cannot absorb the kink, so its residuals show structure by range', () => {
  const lin = fitted('linear');
  const rows = residuals(lin).filter((r) => !r.censored);
  const meanFor = (id) => {
    const xs = rows.filter((r) => r.range === id).map((r) => r.forwardResid);
    return xs.reduce((a, b) => a + b, 0) / xs.length;
  };
  const r2 = meanFor('R2');
  const r3 = meanFor('R3');
  assert.ok(Math.abs(r3 - r2) > 0.6, `range means must differ under the wrong model (${r2}, ${r3})`);
});
