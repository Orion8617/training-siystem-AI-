// STBP (Surrogate Gradient Training with Back-Propagation) for SNNs
// Uses surrogate gradient approach for training spiking neural networks

import type { NeuroEngine } from './neuro-engine';
import type { TrainingMetrics } from './ai4i-dataset';
import { NUM_CLASSES, generateDataset, splitDataset } from './ai4i-dataset';
import type { ProgressCallback, TrainOptions } from './ai4i-trainer';

/** Surrogate gradient function (Heaviside approximation) */
function surrogateGrad(v: number, threshold = -65): number {
  const beta = 0.5;
  return beta / (Math.abs(v - threshold) * beta + 1) ** 2;
}

function computeMetrics(
  predictions: number[],
  labels: number[],
): { accuracy: number; f1Macro: number; confusion: number[][]; perClassAccuracy: number[] } {
  const n = predictions.length;
  const confusion = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
  let correct = 0;

  for (let i = 0; i < n; i++) {
    confusion[labels[i]][predictions[i]]++;
    if (predictions[i] === labels[i]) correct++;
  }

  const perClassAccuracy = Array.from({ length: NUM_CLASSES }, (_, c) => {
    const total = confusion[c].reduce((a, b) => a + b, 0);
    return total > 0 ? confusion[c][c] / total : 0;
  });

  let f1Sum = 0;
  for (let c = 0; c < NUM_CLASSES; c++) {
    const tp = confusion[c][c];
    const fp = confusion.reduce((s, row) => s + row[c], 0) - tp;
    const fn = confusion[c].reduce((s, v) => s + v, 0) - tp;
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    f1Sum += prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  }

  return {
    accuracy: correct / n,
    f1Macro: f1Sum / NUM_CLASSES,
    confusion,
    perClassAccuracy,
  };
}

export async function trainSTBP(
  engine: NeuroEngine,
  options: TrainOptions = {},
  onProgress?: ProgressCallback,
  signal?: AbortSignal,
): Promise<TrainingMetrics> {
  const epochs = options.epochs ?? 5;
  const datasetSize = options.datasetSize ?? 500;
  const lr = options.learningRate ?? 0.001;
  const startTime = Date.now();

  const { features, labels } = generateDataset(datasetSize);
  const { trainFeatures, trainLabels, testFeatures, testLabels } = splitDataset(features, labels);

  let synOps = 0;

  // STBP training with surrogate gradients
  for (let epoch = 0; epoch < epochs; epoch++) {
    if (signal?.aborted) break;

    let totalLoss = 0;

    for (let s = 0; s < trainFeatures.length; s++) {
      if (signal?.aborted) break;

      const scores = engine.forward(trainFeatures[s]);
      synOps += engine.synapses.length;

      // Softmax + cross-entropy with surrogate gradient
      const maxScore = Math.max(...scores);
      const expScores = scores.map(x => Math.exp(x - maxScore));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map(x => x / sumExp);
      const target = trainLabels[s];
      totalLoss += -Math.log(probs[target] + 1e-9);

      // Compute surrogate gradients and apply
      const n = engine.neurons.length;
      const outputStart = n - NUM_CLASSES;
      for (const syn of engine.synapses) {
        if (syn.to >= outputStart) {
          const outIdx = syn.to - outputStart;
          const grad = probs[outIdx] - (outIdx === target ? 1 : 0);
          const sg = surrogateGrad(engine.neurons[syn.to].v);
          syn.weight -= lr * grad * sg * (engine.neurons[syn.from].fired ? 1 : 0.01);
          syn.weight = Math.max(-2, Math.min(2, syn.weight));
        }
      }

      if (s % 50 === 0 && onProgress) {
        onProgress({
          epoch: epoch + 1,
          totalEpochs: epochs,
          sample: s,
          totalSamples: trainFeatures.length,
          phase: 'training',
          accuracy: 0,
          f1Macro: 0,
          sparsity: 1 - engine.neurons.filter(n => n.fired).length / engine.neurons.length,
          loss: totalLoss / (s + 1),
          elapsedMs: Date.now() - startTime,
          synOps,
        });
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  }

  if (signal?.aborted) {
    return {
      epoch: epochs, totalEpochs: epochs, sample: 0, totalSamples: testFeatures.length,
      phase: 'done', accuracy: 0, f1Macro: 0, sparsity: 0, loss: 0,
      elapsedMs: Date.now() - startTime, synOps,
    };
  }

  // Evaluate
  const predictions: number[] = [];
  for (let s = 0; s < testFeatures.length; s++) {
    if (signal?.aborted) break;
    const scores = engine.forward(testFeatures[s]);
    predictions.push(scores.indexOf(Math.max(...scores)));

    if (s % 50 === 0 && onProgress) {
      onProgress({
        epoch: epochs, totalEpochs: epochs, sample: s, totalSamples: testFeatures.length,
        phase: 'testing', accuracy: 0, f1Macro: 0, sparsity: 0, loss: 0,
        elapsedMs: Date.now() - startTime, synOps,
      });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  const elapsedMs = Date.now() - startTime;
  const { accuracy, f1Macro, confusion, perClassAccuracy } = computeMetrics(
    predictions, testLabels.slice(0, predictions.length),
  );

  const finalMetrics: TrainingMetrics = {
    epoch: epochs, totalEpochs: epochs, sample: testFeatures.length, totalSamples: testFeatures.length,
    phase: 'done', accuracy, f1Macro,
    sparsity: 1 - engine.neurons.filter(n => n.fired).length / engine.neurons.length,
    loss: 0, confusion, perClassAccuracy, elapsedMs, synOps,
  };

  onProgress?.(finalMetrics);
  return finalMetrics;
}
