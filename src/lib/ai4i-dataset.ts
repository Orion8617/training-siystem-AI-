// AI4I 2020 Predictive Maintenance Dataset types and constants

export const NUM_CLASSES = 6;

export const FAILURE_LABELS = [
  'No Failure (OK)',
  'Tool Wear Failure (TWF)',
  'Heat Dissipation Failure (HDF)',
  'Power Failure (PWF)',
  'Overstrain Failure (OSF)',
  'Random Failures (RNF)',
];

export interface TrainingMetrics {
  /** Current epoch (1-based) */
  epoch: number;
  /** Total epochs */
  totalEpochs: number;
  /** Current sample index */
  sample: number;
  /** Total samples */
  totalSamples: number;
  /** Training phase */
  phase: 'training' | 'testing' | 'done';
  /** Accuracy (0-1) */
  accuracy: number;
  /** Macro F1 score (0-1) */
  f1Macro: number;
  /** Sparsity (fraction of silent neurons) */
  sparsity: number;
  /** Training loss */
  loss: number;
  /** Confusion matrix [true x predicted] */
  confusion?: number[][];
  /** Per-class accuracy */
  perClassAccuracy?: number[];
  /** Elapsed time in ms */
  elapsedMs: number;
  /** Synaptic operations count */
  synOps: number;
}

/** Generate synthetic AI4I dataset samples */
export function generateDataset(size: number): { features: number[][]; labels: number[] } {
  const features: number[][] = [];
  const labels: number[] = [];

  const rng = (min: number, max: number) => Math.random() * (max - min) + min;

  for (let i = 0; i < size; i++) {
    // Features: [airTemp, processTemp, rotationalSpeed, torque, toolWear, typeOHE_L, typeOHE_M, typeOHE_H]
    const airTemp = rng(295, 305);
    const processTemp = airTemp + rng(8, 12);
    const rotSpeed = rng(1168, 2860);
    const torque = rng(3.8, 76.6);
    const toolWear = rng(0, 253);
    const typeVal = Math.floor(rng(0, 3));

    features.push([
      (airTemp - 295) / 10,
      (processTemp - 305) / 10,
      (rotSpeed - 1168) / 1692,
      torque / 76.6,
      toolWear / 253,
      typeVal === 0 ? 1 : 0,
      typeVal === 1 ? 1 : 0,
      typeVal === 2 ? 1 : 0,
    ]);

    // Generate label with realistic class distribution (96.5% no failure)
    const r = Math.random();
    if (r < 0.965) labels.push(0);
    else if (r < 0.975) labels.push(1);
    else if (r < 0.983) labels.push(2);
    else if (r < 0.990) labels.push(3);
    else if (r < 0.997) labels.push(4);
    else labels.push(5);
  }

  return { features, labels };
}

/** Split dataset into train/test */
export function splitDataset(
  features: number[][],
  labels: number[],
  trainRatio = 0.8,
): { trainFeatures: number[][]; trainLabels: number[]; testFeatures: number[][]; testLabels: number[] } {
  const n = features.length;
  const trainSize = Math.floor(n * trainRatio);
  // Shuffle indices
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const trainIdx = indices.slice(0, trainSize);
  const testIdx = indices.slice(trainSize);
  return {
    trainFeatures: trainIdx.map(i => features[i]),
    trainLabels: trainIdx.map(i => labels[i]),
    testFeatures: testIdx.map(i => features[i]),
    testLabels: testIdx.map(i => labels[i]),
  };
}
