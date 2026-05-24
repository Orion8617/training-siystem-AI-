// Training Worker - offloads SNN training to a background thread

import { NeuroEngine } from './neuro-engine';
import { trainOnAI4I } from './ai4i-trainer';
import { trainSTBP } from './surrogate-trainer';
import type { TrainingMetrics } from './ai4i-dataset';
import type { ConnectomeSubset } from './connectome-loader';

export interface WorkerRequest {
  type: 'train';
  method: 'rstdp' | 'stbp';
  useConnectome: boolean;
  connectomeSubset: ConnectomeSubset;
  epochs: number;
  runId: string;
  neuronCount: number;
}

export interface WorkerMessage {
  type: 'progress' | 'done' | 'error';
  runId: string;
  metrics: TrainingMetrics;
  diagnostics?: unknown;
  message?: string;
}

// Only run worker code in a worker context
if (typeof self !== 'undefined' && typeof (self as unknown as { WorkerGlobalScope: unknown }).WorkerGlobalScope !== 'undefined') {
  self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
    const req = e.data;
    if (req.type !== 'train') return;

    const engine = new NeuroEngine(req.neuronCount, req.useConnectome, req.connectomeSubset);

    const onProgress = (metrics: TrainingMetrics) => {
      const msg: WorkerMessage = { type: 'progress', runId: req.runId, metrics };
      self.postMessage(msg);
    };

    try {
      const finalMetrics = req.method === 'stbp'
        ? await trainSTBP(engine, { epochs: req.epochs }, onProgress)
        : await trainOnAI4I(engine, { epochs: req.epochs }, onProgress);

      const diagnostics = engine.getDiagnostics();
      const doneMsg: WorkerMessage = {
        type: 'done',
        runId: req.runId,
        metrics: finalMetrics,
        diagnostics,
      };
      self.postMessage(doneMsg);
    } catch (err: unknown) {
      const errMsg: WorkerMessage = {
        type: 'error',
        runId: req.runId,
        metrics: {} as TrainingMetrics,
        message: err instanceof Error ? err.message : String(err),
      };
      self.postMessage(errMsg);
    }
  };
}
