// R-STDP AI4I trainer
// Reward-modulated Spike-Timing Dependent Plasticity training on AI4I dataset

import type { NeuroEngine } from './neuro-engine';
import type { TrainingMetrics } from './ai4i-dataset';
import { NUM_CLASSES, generateDataset, splitDataset } from './ai4i-dataset';

export type ProgressCallback = (metrics: TrainingMetrics) => void;

export interface TrainOptions {
  epochs?: number;
  datasetSize?: number;
  learningRate?: number;
}

function computeMetrics(
  predictions: number[],
  labels: number[],
  elapsedMs: number,
  synOps: number,
): { accuracy: number; f1Macro: number; confusion: number[][]; perClassAccuracy: number[] } {
  const n = predictions.length;
  const confusion = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
  let correct = 0;

  for (let i = 0; i < n; i++) {
    const p = predictions[i];
    const t = labels[i];
    confusion[t][p]++;
    if (p === t) correct++;
  }

  const perClassAccuracy = Array.from({ length: NUM_CLASSES }, (_, c) => {
    const total = confusion[c].reduce((a, b) => a + b, 0);
    return total > 0 ? confusion[c][c] / total : 0;
  });

  // Macro F1
  let f1Sum = 0;
  for (let c = 0; c < NUM_CLASSES; c++) {
    const tp = confusion[c][c];
    const fp = confusion.reduce((s, row) => s + row[c], 0) - tp;
    const fn = confusion[c].reduce((s, v) => s + v, 0) - tp;
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    f1Sum += prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  }
  const f1Macro = f1Sum / NUM_CLASSES;
  const accuracy = correct / n;

  return { accuracy, f1Macro, confusion, perClassAccuracy };
}

export async function trainOnAI4I(
  engine: NeuroEngine,
  options: TrainOptions = {},
  onProgress?: ProgressCallback,
  signal?: AbortSignal,
): Promise<TrainingMetrics> {
  const epochs = options.epochs ?? 5;
  const datasetSize = options.datasetSize ?? 500;
  const startTime = Date.now();

  const { features, labels } = generateDataset(datasetSize);
  const { trainFeatures, trainLabels, testFeatures, testLabels } = splitDataset(features, labels);

  let synOps = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (signal?.aborted) break;

    let loss = 0;
    const epochStart = Date.now();

    for (let s = 0; s < trainFeatures.length; s++) {
      if (signal?.aborted) break;

      const scores = engine.forward(trainFeatures[s]);
      synOps += engine.synapses.length;

      const pred = scores.indexOf(Math.max(...scores));
      const target = trainLabels[s];
      const reward = pred === target ? 1.0 : -0.5;
      const baselineCorrect = 1 / NUM_CLASSES;
      engine.applyReward(reward - baselineCorrect, target);
      loss += pred !== target ? 1 : 0;

      // Report progress every 50 samples
      if (s % 50 === 0 && onProgress) {
        const elapsed = Date.now() - startTime;
        onProgress({
          epoch: epoch + 1,
          totalEpochs: epochs,
          sample: s,
          totalSamples: trainFeatures.length,
          phase: 'training',
          accuracy: 1 - loss / (s + 1),
          f1Macro: 0,
          sparsity: 1 - engine.neurons.filter(n => n.fired).length / engine.neurons.length,
          loss: loss / (s + 1),
          elapsedMs: elapsed,
          synOps,
        });
        // Yield to event loop
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    console.log(`Epoch ${epoch + 1}/${epochs} done in ${Date.now() - epochStart}ms`);
  }

  if (signal?.aborted) {
    return {
      epoch: epochs,
      totalEpochs: epochs,
      sample: 0,
      totalSamples: testFeatures.length,
      phase: 'done',
      accuracy: 0,
      f1Macro: 0,
      sparsity: 0,
      loss: 0,
      elapsedMs: Date.now() - startTime,
      synOps,
    };
  }

  // Test phase
  const predictions: number[] = [];
  for (let s = 0; s < testFeatures.length; s++) {
    if (signal?.aborted) break;
    const scores = engine.forward(testFeatures[s]);
    predictions.push(scores.indexOf(Math.max(...scores)));

    if (s % 50 === 0 && onProgress) {
      onProgress({
        epoch: epochs,
        totalEpochs: epochs,
        sample: s,
        totalSamples: testFeatures.length,
        phase: 'testing',
        accuracy: 0,
        f1Macro: 0,
        sparsity: 0,
        loss: 0,
        elapsedMs: Date.now() - startTime,
        synOps,
      });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  const elapsedMs = Date.now() - startTime;
  const { accuracy, f1Macro, confusion, perClassAccuracy } = computeMetrics(
    predictions, testLabels.slice(0, predictions.length), elapsedMs, synOps,
  );

  const finalMetrics: TrainingMetrics = {
    epoch: epochs,
    totalEpochs: epochs,
    sample: testFeatures.length,
    totalSamples: testFeatures.length,
    phase: 'done',
    accuracy,
    f1Macro,
    sparsity: 1 - engine.neurons.filter(n => n.fired).length / engine.neurons.length,
    loss: 0,
    confusion,
    perClassAccuracy,
    elapsedMs,
    synOps,
  };

  onProgress?.(finalMetrics);
  return finalMetrics;
}
