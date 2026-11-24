// Fictional metrology-lab data. Every number here is synthetic by construction;
// nothing is a real verification result, certificate, or reference value.
//
// The `truth` block and `hiddenReferenceValues()` are for the test suite only.
// The application imports `generateDataset` and never reads the truth block.
import { mulberry32, normal, uniform, shuffle, hashSeed } from './random.js';

export const RANGES = [
  { id: 'R1', label: '低量程', nominal: 0, upper: 100, unit: 'mm' },
  { id: 'R2', label: '中量程', nominal: 100, upper: 300, unit: 'mm' },
  { id: 'R3', label: '高量程', nominal: 300, upper: 500, unit: 'mm' },
  { id: 'R4', label: '扩容量程', nominal: 500, upper: 600, unit: 'mm' },
];

// Known truths. Tests only.
export const truth = {
  slope: 1.028,
  offset: 0.42,
  newRepeatSd: 0.3,
  refRepeatSd: 0.75,
  certSd: 0.4,
  batchSd: 0.2,
  kink: 280,
  kinkSlopeDelta: 0.05,
  saturation: 472.0,
  detected: { R1: 2.4, R2: 5.0, R3: 12.0, R4: 20.0 },
  withdrawDemo: 'REF-014',
  recheckBatch: 'B05',
};

// id, range, hidden realized value
const REFERENCE_VALUES = [
  ['REF-001', 'R1', 0.8],
  ['REF-002', 'R1', 1.6],
  ['REF-003', 'R1', 2.6],
  ['REF-004', 'R1', 7.5],
  ['REF-005', 'R1', 18.5],
  ['REF-006', 'R1', 31.0],
  ['REF-007', 'R1', 47.2],
  ['REF-008', 'R1', 63.8],
  ['REF-009', 'R1', 82.4],
  ['REF-010', 'R1', 95.0],
  ['REF-011', 'R2', 112.0],
  ['REF-012', 'R2', 138.5],
  ['REF-013', 'R2', 167.0],
  ['REF-014', 'R2', 196.0],
  ['REF-015', 'R2', 224.0],
  ['REF-016', 'R2', 258.0],
  ['REF-017', 'R2', 272.0],
  ['REF-018', 'R2', 286.0],
  ['REF-019', 'R2', 289.5],
  ['REF-020', 'R3', 318.0],
  ['REF-021', 'R3', 352.0],
  ['REF-022', 'R3', 389.0],
  ['REF-023', 'R3', 425.0],
  ['REF-024', 'R3', 448.0],
  ['REF-025', 'R4', 525.0],
  ['REF-026', 'R4', 545.0],
];

export function hiddenReferenceValues() {
  return Object.fromEntries(REFERENCE_VALUES.map(([id, range, v]) => [id, { range, value: v }]));
}

/** Response of the new device before noise: continuous piecewise line. */
export function truthResponse(reference) {
  const { slope, offset, kink, kinkSlopeDelta } = truth;
  const base = slope * reference + offset;
  if (reference <= kink) return base;
  return base + kinkSlopeDelta * (reference - kink);
}

function rangeFor(value) {
  if (value < 100) return 'R1';
  if (value < 300) return 'R2';
  if (value < 500) return 'R3';
  return 'R4';
}

export function generateDataset({ seed = 20261124, batches = 4, reps = 4 } = {}) {
  const rand = mulberry32(seed);
  const batchList = [];
  for (let b = 1; b <= batches; b += 1) {
    batchList.push({
      id: `B${String(b).padStart(2, '0')}`,
      date: `2026-${String(2 + b).padStart(2, '0')}-${String(3 + b * 2).padStart(2, '0')}`,
      offset: Number(normal(rand, 0, truth.batchSd).toFixed(4)),
      kind: 'calibration',
    });
  }
  // A recheck batch holds a single reference sample. Its shared offset rests on
  // that one sample, which makes the effect of withdrawing it visible.
  batchList.push({
    id: truth.recheckBatch,
    date: '2026-06-18',
    offset: Number(normal(rand, 0, truth.batchSd).toFixed(4)),
    kind: 'recheck',
  });
  const calibrationBatches = batchList.filter((b) => b.kind === 'calibration').map((b) => b.id);

  const references = REFERENCE_VALUES.map(([id, range, value]) => ({
    id,
    range,
    certified: Number((value + normal(rand, 0, truth.certSd)).toFixed(3)),
    certSd: truth.certSd,
    status: 'active',
    withdrawnAt: null,
    withdrawnNote: '',
    version: 1,
    receivedAt: `2026-0${1 + (Number(id.slice(4)) % 3)}-1${Number(id.slice(5)) % 9}`,
  }));

  const measurements = [];
  let mid = 0;
  for (let i = 0; i < references.length; i += 1) {
    const ref = references[i];
    const realized = REFERENCE_VALUES[i][2];
    const rng = rangeFor(realized);
    const lod = truth.detected[rng];
    const usedBatches =
      ref.id === truth.withdrawDemo
        ? [truth.recheckBatch, ...shuffle(rand, calibrationBatches).slice(0, 1)]
        : shuffle(rand, calibrationBatches).slice(0, 2);
    for (const batchId of usedBatches) {
      const batch = batchList.find((b) => b.id === batchId);
      for (let r = 0; r < reps; r += 1) {
        // Both devices have measurement error; the certificate is not exact.
        const oldReading = realized + normal(rand, 0, truth.refRepeatSd);
        const clean = truthResponse(realized) + batch.offset + normal(rand, 0, truth.newRepeatSd);
        let reading = clean;
        let saturated = false;
        if (clean > truth.saturation) {
          reading = truth.saturation + Math.abs(normal(rand, 0, 0.03));
          saturated = true;
        }
        const belowLod = !saturated && reading < lod;
        if (belowLod) reading = lod;
        measurements.push({
          id: `M${String(++mid).padStart(4, '0')}`,
          referenceId: ref.id,
          range: rng,
          batchId,
          rep: r + 1,
          reading: Number(reading.toFixed(3)),
          oldReading: Number(oldReading.toFixed(3)),
          reported: belowLod ? `<${lod.toFixed(1)}` : reading.toFixed(2),
          belowLod,
          saturated,
          lod,
          importedFlag: '',
        });
      }
    }
  }

  // A few gross outliers from a bumped fixture: exactly 3 eligible rows.
  const eligible = [];
  for (let k = 0; k < measurements.length; k += 1) {
    const m = measurements[k];
    if (!m.saturated && !m.belowLod && m.reading > 20) eligible.push(k);
  }
  const bumped = shuffle(rand, eligible).slice(0, 3);
  for (const k of bumped) {
    const m = measurements[k];
    const dir = rand() < 0.5 ? -1 : 1;
    m.reading = Number((m.reading + dir * uniform(rand, 6, 12)).toFixed(3));
    m.importedFlag = '夹具碰撞';
    m.reported = m.reading.toFixed(2);
  }

  // Operational readings from the new device; no reference measurement exists
  // for these. `batchId` is the measurement batch whose offset must be known
  // before a reading can be converted.
  const planned = [
    [1.2, 'A', 'B01'], [4.6, 'A', 'B01'], [12.8, 'A', 'B02'], [25.4, 'A', 'B02'], [44.1, 'A', 'B03'],
    [61.9, 'B', 'B01'], [78.3, 'B', 'B03'], [90.5, 'B', 'B04'], [108.2, 'B', 'B04'], [136.7, 'B', 'B02'],
    [160.4, 'C', 'B01'], [188.9, 'C', 'B03'], [205.5, 'C', truth.recheckBatch], [247.0, 'C', truth.recheckBatch], [279.8, 'C', 'B04'],
    [295.2, 'C', truth.recheckBatch], [305.6, 'A', 'B02'], [334.8, 'A', 'B03'], [362.1, 'A', 'B01'], [401.7, 'B', 'B04'],
    [428.4, 'B', 'B03'], [442.9, 'B', 'B02'], [486.4, 'C', 'B01'], [510.0, 'C', 'B04'], [533.8, 'C', 'B02'],
  ];
  const field = planned.map(([reading, lot, batchId], i) => {
    const range = RANGES.find((r) => reading >= r.nominal && reading < r.upper) || RANGES[RANGES.length - 1];
    return {
      id: `F${String(i + 1).padStart(3, '0')}`,
      reading,
      range: range.id,
      batchId,
      lotId: `LOT-2603-${lot}`,
      note: reading === 486.4 ? '疑似饱和' : reading === 1.2 ? '接近检出限' : '',
      sampledAt: `2026-06-${String(3 + (i % 24)).padStart(2, '0')}`,
    };
  });

  return {
    seed,
    unit: 'mm',
    ranges: RANGES.map((r) => ({ ...r })),
    batches: batchList,
    references,
    measurements,
    field,
    lots: ['LOT-2603-A', 'LOT-2603-B', 'LOT-2603-C'],
    generatedBy: 'synthetic-v1',
  };
}

export function datasetFingerprint(dataset) {
  return `SYN-${dataset.seed}-${hashSeed(JSON.stringify(dataset.measurements)).toString(16).slice(0, 8)}`;
}
