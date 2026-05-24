import { useState, useRef, useCallback, useEffect } from 'react';
import { NeuroEngine } from '@/lib/neuro-engine';
import type { ConnectomeSubset } from '@/lib/connectome-loader';
import { trainOnAI4I, type ProgressCallback } from '@/lib/ai4i-trainer';
import { trainSTBP } from '@/lib/surrogate-trainer';
import { FAILURE_LABELS, NUM_CLASSES, type TrainingMetrics } from '@/lib/ai4i-dataset';
import { neuralBus } from '@/lib/neural-bus';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import type { WorkerRequest, WorkerMessage } from '@/lib/training-worker';
import {
  Play, Loader2, CheckCircle2,
  Activity, Brain, Target, Factory,
  RotateCcw, Server, Monitor, ShieldCheck,
  GitCompareArrows, Trophy,
} from 'lucide-react';

// ─── Confusion Matrix ───
const ConfusionMatrix = ({ matrix }: { matrix: number[][] }) => {
  const max = Math.max(1, ...matrix.flat());
  const shortLabels = ['OK', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'];
  return (
    <div className="space-y-1">
      <div className="grid gap-px" style={{ gridTemplateColumns: `40px repeat(${NUM_CLASSES}, 1fr)` }}>
        <div />
        {shortLabels.map(l => (
          <div key={l} className="text-[7px] text-muted-foreground text-center font-mono">{l}</div>
        ))}
        {matrix.map((row, r) => (
          <div key={`row-${r}`} className="contents">
            <div className="text-[7px] text-muted-foreground font-mono flex items-center justify-end pr-1">{shortLabels[r]}</div>
            {row.map((val, c) => {
              const intensity = val / max;
              const isDiag = r === c;
              return (
                <div key={`${r}-${c}`}
                  className="aspect-square flex items-center justify-center text-[7px] font-mono rounded-sm"
                  style={{
                    backgroundColor: isDiag
                      ? `hsl(var(--chart-2) / ${0.15 + intensity * 0.85})`
                      : `hsl(var(--destructive) / ${intensity * 0.5})`,
                    color: val > 0 ? 'hsl(var(--foreground))' : 'transparent',
                    minWidth: 14,
                  }}>
                  {val > 0 ? val : ''}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── Metric Card ───
const MetricCard = ({ icon: Icon, label, value, unit, color }: {
  icon: typeof Activity; label: string; value: string | number; unit?: string; color: string;
}) => (
  <div className="bg-card border border-border rounded-lg p-2 flex items-center gap-2">
    <Icon size={14} style={{ color }} />
    <div className="flex-1 min-w-0">
      <div className="text-[10px] text-muted-foreground truncate">{label}</div>
      <div className="text-sm font-mono font-bold" style={{ color }}>
        {value}{unit && <span className="text-[10px] text-muted-foreground ml-0.5">{unit}</span>}
      </div>
    </div>
  </div>
);

// ─── Per-class accuracy bars ───
const ClassBars = ({ perClass }: { perClass: number[] }) => (
  <div className="space-y-1">
    {FAILURE_LABELS.map((label, i) => (
      <div key={i} className="flex items-center gap-1">
        <span className="text-[8px] font-mono text-muted-foreground w-8 text-right">{(perClass[i] * 100).toFixed(0)}%</span>
        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${perClass[i] * 100}%`,
              backgroundColor: i === 0 ? 'hsl(var(--chart-2))' : 'hsl(var(--chart-4))',
            }}
          />
        </div>
        <span className="text-[7px] text-muted-foreground truncate max-w-[60px]">{label.split('(')[0].trim()}</span>
      </div>
    ))}
  </div>
);

// ─── Six Sigma: compute from accuracy ───
function computeSigmaFromAccuracy(accuracy: number, totalOps: number): { sigma: number; dpmo: number; defects: number; totalOps: number } {
  const defects = Math.round((1 - accuracy) * totalOps);
  const dpmo = totalOps > 0 ? Math.round((defects / totalOps) * 1_000_000) : 1_000_000;
  let sigma = 0;
  if (dpmo <= 0) {
    sigma = 6;
  } else if (dpmo >= 1_000_000) {
    sigma = 0;
  } else {
    const pDefect = dpmo / 1_000_000;
    const t = Math.sqrt(-2 * Math.log(pDefect));
    const z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
    sigma = z + 1.5; // Traditional 1.5σ shift
  }
  return { sigma, dpmo, defects, totalOps };
}

// ─── Six Sigma Badge ───
const SixSigmaBadge = ({ sigma, dpmo }: { sigma: number; dpmo: number }) => {
  const color = sigma >= 6 ? 'hsl(142, 76%, 46%)' : sigma >= 4 ? 'hsl(var(--chart-2))' : sigma >= 3 ? 'hsl(var(--chart-3))' : 'hsl(var(--destructive))';
  const level = sigma >= 6 ? '🏆 World Class' : sigma >= 5 ? '⭐ Excellent' : sigma >= 4 ? '✅ Good' : sigma >= 3 ? '⚠️ Acceptable' : '❌ Needs Improvement';
  return (
    <div className="bg-card border border-border rounded-lg p-2">
      <div className="flex items-center gap-2">
        <ShieldCheck size={14} style={{ color }} />
        <div>
          <div className="text-[10px] text-muted-foreground">Six Sigma</div>
          <div className="text-sm font-mono font-bold" style={{ color }}>{sigma.toFixed(1)}σ</div>
        </div>
        <div className="text-[9px] font-mono" style={{ color }}>{level}</div>
        <div className="ml-auto text-right">
          <div className="text-[10px] text-muted-foreground">DPMO</div>
          <div className="text-xs font-mono text-muted-foreground">{dpmo.toLocaleString()}</div>
        </div>
      </div>
      <div className="text-[8px] text-muted-foreground mt-1">
        6σ = 99.99966% (DPMO ≤ 3.4) • 5σ = 99.977% • 4σ = 99.38% • 3σ = 93.3%
      </div>
    </div>
  );
};

// ─── Comparison Result Card ───
const ComparisonCard = ({ label, metrics, color, winner }: {
  label: string; metrics: TrainingMetrics; color: string; winner: boolean;
}) => (
  <div className={`bg-card border rounded-xl p-2.5 ${winner ? 'border-chart-2 ring-1 ring-chart-2/30' : 'border-border'}`}>
    <div className="flex items-center gap-1.5 mb-2">
      {winner && <Trophy size={12} className="text-chart-2" />}
      <span className="text-[10px] font-bold" style={{ color }}>{label}</span>
      {winner && <span className="text-[8px] bg-chart-2/20 text-chart-2 px-1 rounded font-mono">WINNER</span>}
    </div>
    <div className="grid grid-cols-2 gap-1.5">
      <div className="bg-muted rounded p-1.5 text-center">
        <div className="text-[8px] text-muted-foreground">Accuracy</div>
        <div className="text-sm font-mono font-bold" style={{ color }}>{(metrics.accuracy * 100).toFixed(1)}%</div>
      </div>
      <div className="bg-muted rounded p-1.5 text-center">
        <div className="text-[8px] text-muted-foreground">F1 Macro</div>
        <div className="text-sm font-mono font-bold" style={{ color }}>{(metrics.f1Macro * 100).toFixed(1)}%</div>
      </div>
      <div className="bg-muted rounded p-1.5 text-center">
        <div className="text-[8px] text-muted-foreground">Latency</div>
        <div className="text-xs font-mono text-muted-foreground">{(metrics.elapsedMs / 1000).toFixed(1)}s</div>
      </div>
      <div className="bg-muted rounded p-1.5 text-center">
        <div className="text-[8px] text-muted-foreground">SynOps</div>
        <div className="text-xs font-mono text-muted-foreground">{(metrics.synOps / 1000).toFixed(0)}K</div>
      </div>
    </div>
    {metrics.confusion && (
      <div className="mt-2">
        <ConfusionMatrix matrix={metrics.confusion} />
      </div>
    )}
  </div>
);

interface ServerResult {
  accuracy: number;
  f1Macro: number;
  confusion: number[][];
  perClassAccuracy: number[];
  epochs: number;
  datasetSize: number;
  trainSize: number;
  testSize: number;
  sixSigma: { totalOps: number; defects: number; dpmo: number; sigma: number };
  synOps: number;
}

interface DiagnosticData {
  columnScores: { colId: number; fitnessScore: number; avgReward: number }[];
  cumulativeReward: number;
  recentRewards: number[];
  avgRecentReward: number;
  generation: number;
  taskActive: boolean;
  correlation: number;
}

// ─── Worker helper (inline to avoid IIFE code-splitting error) ───
function createTrainingWorker(): Worker {
  return new Worker(new URL('./lib/training-worker.ts', import.meta.url), { type: 'module' });
}

const AI4IDashboard = () => {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [serverResult, setServerResult] = useState<ServerResult | null>(null);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<'local' | 'server' | 'compare' | 'decussation'>('server');
  const [connectomeSubset, setConnectomeSubset] = useState<ConnectomeSubset>('full');
  const [epochs, setEpochs] = useState(50);
  const [trainMethod, setTrainMethod] = useState<'rstdp' | 'stbp'>('rstdp');
  const [diagnostics, setDiagnostics] = useState<DiagnosticData | null>(null);
  const [diagMode, setDiagMode] = useState(false);
  const [useWorker, setUseWorker] = useState(true);
  const [compareResults, setCompareResults] = useState<{ rstdp?: TrainingMetrics; stbp?: TrainingMetrics } | null>(null);
  const [decussationResults, setDecussationResults] = useState<{ ipsilateral?: TrainingMetrics; contralateral?: TrainingMetrics } | null>(null);
  const [comparePhase, setComparePhase] = useState<string>('');
  const abortRef = useRef<AbortController | null>(null);
  const workerRef = useRef<Worker | null>(null);

  // Reactive engine config from neuralBus (synced with NeuroComputer organism selector)
  const [busConfig, setBusConfig] = useState(neuralBus.engineConfig);
  useEffect(() => {
    return neuralBus.onConfigChange((config) => setBusConfig(config));
  }, []);
  const useConnectome = busConfig.useConnectome;
  const busNeurons = busConfig.neurons;
  const organismLabel = busConfig.organism || (useConnectome ? `elegans_${busNeurons}` : `mini_${busNeurons}`);

  const validateConfig = (): string | null => {
    if (epochs < 1 || epochs > 100) return 'Epochs must be between 1 and 100';
    if (mode === 'local' && epochs > 100) return 'Local mode limited to 100 epochs';
    return null;
  };

  // ── Server training ──
  const handleServerTrain = useCallback(async () => {
    setRunning(true);
    setServerResult(null);
    setMetrics(null);
    try {
      const { data, error } = await supabase.functions.invoke('train-ai4i', {
        body: { epochs, learningRate: 0.01, datasetSize: 10000 },
      });
      if (error) throw error;
      setServerResult(data as ServerResult);
      toast.success(`Server training complete: F1=${((data as ServerResult).f1Macro * 100).toFixed(1)}%`);
    } catch (e: unknown) {
      const err = e as { message?: string };
      console.error('Server training error:', e);
      toast.error(`Error: ${err.message ?? 'Server error'}`);
    }
    setRunning(false);
  }, [epochs]);

  // ── Worker-based training ──
  const handleWorkerTrain = useCallback((method: 'stbp' | 'rstdp', runId: string): Promise<TrainingMetrics> => {
    return new Promise((resolve, reject) => {
      const worker = createTrainingWorker();
      workerRef.current = worker;

      worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
        const msg = e.data;
        if (msg.runId !== runId) return;

        if (msg.type === 'progress') {
          setMetrics({ ...msg.metrics });
        } else if (msg.type === 'done') {
          if (msg.diagnostics) setDiagnostics(msg.diagnostics as DiagnosticData);
          worker.terminate();
          workerRef.current = null;
          resolve(msg.metrics);
        } else if (msg.type === 'error') {
          worker.terminate();
          workerRef.current = null;
          reject(new Error(msg.message));
        }
      };

      worker.onerror = (err) => {
        worker.terminate();
        workerRef.current = null;
        reject(new Error(err.message));
      };

      const request: WorkerRequest = {
        type: 'train',
        method,
        useConnectome,
        connectomeSubset,
        epochs: diagMode ? 3 : epochs,
        runId,
        neuronCount: busNeurons,
      };
      worker.postMessage(request);
    });
  }, [useConnectome, connectomeSubset, epochs, diagMode, busNeurons]);

  // ── Local training (main thread fallback) ──
  const handleMainThreadTrain = useCallback(async (method: 'stbp' | 'rstdp'): Promise<{ metrics: TrainingMetrics; diagnostics?: DiagnosticData }> => {
    abortRef.current = new AbortController();
    const engine = new NeuroEngine(busNeurons, useConnectome, connectomeSubset);
    const actualEpochs = diagMode ? 3 : epochs;

    const onProgress: ProgressCallback = (m) => {
      setMetrics({ ...m });
      const stats = engine.getStats();
      neuralBus.emit({
        stats,
        spikePositions: engine.neurons.filter(n => n.fired).map(n => ({
          x: n.x, y: n.y, hemisphere: n.hemisphere, type: n.type, izhType: n.izhType,
        })),
        firedCount: engine.neurons.filter(n => n.fired).length,
        gabaLevel: engine.bridge.gabaModulation,
        coherence: stats.quantumCoherence ?? 0,
        entropy: stats.entropy ?? 0,
        bridgeActive: engine.bridge.transfers > 0,
        fieldAmp: engine.energy,
        tick: engine.tick,
        rewardSignal: engine.rewardSignal,
        burstRate: stats.burstRate ?? 0,
        webgpuActive: false,
        modulators: engine.modulators,
        episodicCount: engine.episodicMemory.length,
      });
    };

    let finalMetrics: TrainingMetrics;
    if (method === 'stbp') {
      finalMetrics = await trainSTBP(engine, { epochs: actualEpochs }, onProgress, abortRef.current.signal);
    } else {
      finalMetrics = await trainOnAI4I(engine, { epochs: actualEpochs }, onProgress, abortRef.current.signal);
    }

    const diag = engine.getDiagnostics();
    return { metrics: finalMetrics, diagnostics: diag };
  }, [useConnectome, connectomeSubset, epochs, diagMode, busNeurons]);

  // ── Unified training handler ──
  const runTraining = useCallback(async (method: 'stbp' | 'rstdp'): Promise<TrainingMetrics> => {
    if (useWorker) {
      try {
        return await handleWorkerTrain(method, `${method}-${Date.now()}`);
      } catch (e) {
        console.warn('Worker failed, falling back to main thread:', e);
        toast.info('Worker unavailable, using main thread');
      }
    }
    const result = await handleMainThreadTrain(method);
    if (result.diagnostics) setDiagnostics(result.diagnostics);
    return result.metrics;
  }, [useWorker, handleWorkerTrain, handleMainThreadTrain]);

  // ── Compare mode: runs both methods sequentially ──
  const handleCompare = useCallback(async () => {
    setRunning(true);
    setCompareResults(null);
    setMetrics(null);
    const compareEpochs = Math.min(epochs, 5);

    try {
      // 1. R-STDP
      setComparePhase('🧬 R-STDP (BioEvol)...');
      const rstdpResult = await (async () => {
        abortRef.current = new AbortController();
        const engine = new NeuroEngine(busNeurons, useConnectome, connectomeSubset);
        return trainOnAI4I(engine, { epochs: compareEpochs }, (m) => setMetrics({ ...m }), abortRef.current.signal);
      })();
      setCompareResults(prev => ({ ...prev, rstdp: rstdpResult }));

      // 2. STBP
      setComparePhase('⚡ STBP (Surrogate)...');
      setMetrics(null);
      const stbpResult = await (async () => {
        abortRef.current = new AbortController();
        const engine = new NeuroEngine(busNeurons, useConnectome, connectomeSubset);
        return trainSTBP(engine, { epochs: compareEpochs }, (m) => setMetrics({ ...m }), abortRef.current.signal);
      })();
      setCompareResults(prev => ({ ...prev, stbp: stbpResult }));

      setComparePhase('');
      toast.success('Comparison complete!');

      // Save both to DB
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const base = {
          user_id: user.id,
          epochs: compareEpochs,
          engine_type: `${organismLabel}`,
        };
        await Promise.all([
          supabase.from('training_sessions').insert({
            ...base, engine_type: base.engine_type + '_rstdp_compare',
            accuracy: rstdpResult.accuracy, f1_macro: rstdpResult.f1Macro,
            sparsity: rstdpResult.sparsity, loss: rstdpResult.loss,
            confusion_matrix: rstdpResult.confusion, per_class_accuracy: rstdpResult.perClassAccuracy,
            elapsed_ms: rstdpResult.elapsedMs, syn_ops: rstdpResult.synOps,
          } as Record<string, unknown>),
          supabase.from('training_sessions').insert({
            ...base, engine_type: base.engine_type + '_stbp_compare',
            accuracy: stbpResult.accuracy, f1_macro: stbpResult.f1Macro,
            sparsity: stbpResult.sparsity, loss: stbpResult.loss,
            confusion_matrix: stbpResult.confusion, per_class_accuracy: stbpResult.perClassAccuracy,
            elapsed_ms: stbpResult.elapsedMs, syn_ops: stbpResult.synOps,
          } as Record<string, unknown>),
        ]);
      }
    } catch (e: unknown) {
      const err = e as { message?: string };
      console.error('Compare error:', e);
      toast.error(`Error: ${err.message}`);
    }
    setRunning(false);
    setComparePhase('');
  }, [epochs, useConnectome, connectomeSubset, busNeurons, organismLabel]);

  // ── Decussation A/B: ipsilateral vs contralateral ──
  const handleDecussation = useCallback(async () => {
    setRunning(true);
    setDecussationResults(null);
    setMetrics(null);
    const compareEpochs = Math.min(epochs, 5);

    try {
      // 1. IPSILATERAL (legacy — no crossing)
      setComparePhase('🔀 Ipsilateral (sin cruce)...');
      const ipsiResult = await (async () => {
        const ab = new AbortController();
        abortRef.current = ab;
        const engine = new NeuroEngine(busNeurons, useConnectome, connectomeSubset);
        engine.contralateralDecussation = false;
        return trainOnAI4I(engine, { epochs: compareEpochs }, (m) => setMetrics({ ...m }), ab.signal);
      })();
      setDecussationResults(prev => ({ ...prev, ipsilateral: ipsiResult }));

      // 2. CONTRALATERAL (bio-correct crossing)
      setComparePhase('🧬 Contralateral (cruce bio)...');
      setMetrics(null);
      const contraResult = await (async () => {
        const ab = new AbortController();
        abortRef.current = ab;
        const engine = new NeuroEngine(busNeurons, useConnectome, connectomeSubset);
        engine.contralateralDecussation = true;
        return trainOnAI4I(engine, { epochs: compareEpochs }, (m) => setMetrics({ ...m }), ab.signal);
      })();
      setDecussationResults(prev => ({ ...prev, contralateral: contraResult }));

      setComparePhase('');
      toast.success('Decussation A/B complete!');

      // Save to DB
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const base = { user_id: user.id, epochs: compareEpochs, engine_type: organismLabel };
        await Promise.all([
          supabase.from('training_sessions').insert({
            ...base, engine_type: base.engine_type + '_ipsilateral',
            accuracy: ipsiResult.accuracy, f1_macro: ipsiResult.f1Macro,
            sparsity: ipsiResult.sparsity, loss: ipsiResult.loss,
            confusion_matrix: ipsiResult.confusion, per_class_accuracy: ipsiResult.perClassAccuracy,
            elapsed_ms: ipsiResult.elapsedMs, syn_ops: ipsiResult.synOps,
          } as Record<string, unknown>),
          supabase.from('training_sessions').insert({
            ...base, engine_type: base.engine_type + '_contralateral',
            accuracy: contraResult.accuracy, f1_macro: contraResult.f1Macro,
            sparsity: contraResult.sparsity, loss: contraResult.loss,
            confusion_matrix: contraResult.confusion, per_class_accuracy: contraResult.perClassAccuracy,
            elapsed_ms: contraResult.elapsedMs, syn_ops: contraResult.synOps,
          } as Record<string, unknown>),
        ]);
      }
    } catch (e: unknown) {
      const err = e as { message?: string };
      console.error('Decussation compare error:', e);
      toast.error(`Error: ${err.message}`);
    }
    setRunning(false);
    setComparePhase('');
  }, [epochs, connectomeSubset, useConnectome, busNeurons, organismLabel]);

  // ── Local single-method training ──
  const handleLocalTrain = useCallback(async () => {
    if (running) {
      abortRef.current?.abort();
      workerRef.current?.terminate();
      workerRef.current = null;
      setRunning(false);
      return;
    }

    setRunning(true);
    setMetrics(null);
    setServerResult(null);
    setCompareResults(null);
    setDecussationResults(null);

    try {
      const finalMetrics = await runTraining(trainMethod);

      if (finalMetrics.phase === 'done') {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          await supabase.from('training_sessions').insert({
            user_id: user.id,
            engine_type: `${organismLabel}_${trainMethod}`,
            epochs: diagMode ? 3 : epochs,
            accuracy: finalMetrics.accuracy,
            f1_macro: finalMetrics.f1Macro,
            sparsity: finalMetrics.sparsity,
            loss: finalMetrics.loss,
            confusion_matrix: finalMetrics.confusion,
            per_class_accuracy: finalMetrics.perClassAccuracy,
            elapsed_ms: finalMetrics.elapsedMs,
            syn_ops: finalMetrics.synOps,
          } as Record<string, unknown>);
          toast.success('Session saved');
        }
      }
    } catch (e) {
      console.error('Training error:', e);
    }
    setRunning(false);
    if (diagMode) setDiagMode(false);
  }, [running, trainMethod, runTraining, epochs, diagMode, organismLabel]);

  const handleStart = useCallback(() => {
    const error = validateConfig();
    if (error) { toast.error(`⚠️ Poka-Yoke: ${error}`); return; }
    if (mode === 'server') handleServerTrain();
    else if (mode === 'compare') handleCompare();
    else if (mode === 'decussation') handleDecussation();
    else handleLocalTrain();
  }, [mode, handleServerTrain, handleLocalTrain, handleCompare, handleDecussation, validateConfig]);

  const phaseBadge = metrics?.phase === 'done'
    ? { icon: CheckCircle2, text: 'Complete', color: 'hsl(var(--chart-2))' }
    : metrics?.phase === 'testing'
      ? { icon: Target, text: 'Evaluating...', color: 'hsl(var(--chart-4))' }
      : { icon: Loader2, text: 'Training...', color: 'hsl(var(--chart-1))' };

  const progress = metrics
    ? metrics.phase === 'training'
      ? ((metrics.epoch * 350 + metrics.sample) / (epochs * 350) * 100)
      : metrics.phase === 'testing'
        ? (metrics.sample / metrics.totalSamples * 100)
        : 100
    : 0;

  const displayAccuracy = serverResult?.accuracy ?? metrics?.accuracy ?? 0;
  const displayF1 = serverResult?.f1Macro ?? metrics?.f1Macro ?? 0;
  const displayConfusion = serverResult?.confusion ?? metrics?.confusion;
  const displayPerClass = serverResult?.perClassAccuracy ?? metrics?.perClassAccuracy;
  const isDone = serverResult !== null || metrics?.phase === 'done';

  // Six Sigma calculation
  const totalOps = serverResult?.sixSigma?.totalOps ?? (isDone ? 10000 : 0);
  const sigmaData = isDone ? computeSigmaFromAccuracy(displayAccuracy, totalOps) : null;

  return (
    <div className="space-y-3 max-w-2xl mx-auto">
      {/* Header */}
      <div className="bg-card border border-border rounded-xl p-3">
        <div className="flex items-center gap-2 mb-2">
          <Factory size={18} className="text-chart-1" />
          <h2 className="text-sm font-bold text-foreground">AI4I 2020 — Predictive Maintenance</h2>
        </div>
        <p className="text-[10px] text-muted-foreground mb-3">
          {mode === 'server'
            ? '10,000 industrial samples • Serverless training with Six Sigma & Kanban'
            : mode === 'compare'
              ? 'Compare R-STDP vs STBP side-by-side • Same dataset, same connectome'
              : mode === 'decussation'
                ? 'Decussation A/B: Ipsilateral vs Contralateral — same C. elegans connectome'
                : '500 samples • SNN training • Browser-side'}
        </p>

        {/* Mode selector */}
        <div className="flex gap-0.5 mb-3 bg-muted rounded-lg p-0.5">
          {([
            { key: 'server', label: 'Server', icon: Server },
            { key: 'local', label: 'Local', icon: Monitor },
            { key: 'compare', label: 'Compare', icon: GitCompareArrows },
            { key: 'decussation', label: 'Decusación', icon: Brain },
          ] as const).map(({ key, label, icon: Icon }) => (
            <button key={key}
              onClick={() => !running && setMode(key)}
              className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-md text-[10px] font-medium transition-colors ${
                mode === key ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon size={11} /> {label}
            </button>
          ))}
        </div>

        {/* Training Method selector (local only) */}
        {mode === 'local' && (
          <div className="flex gap-1 mb-3 bg-muted rounded-lg p-0.5">
            <button
              onClick={() => !running && setTrainMethod('rstdp')}
              className={`flex-1 py-1 rounded-md text-[10px] font-medium transition-colors ${
                trainMethod === 'rstdp' ? 'bg-chart-4 text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              🧬 R-STDP (Bio)
            </button>
            <button
              onClick={() => !running && setTrainMethod('stbp')}
              className={`flex-1 py-1 rounded-md text-[10px] font-medium transition-colors ${
                trainMethod === 'stbp' ? 'bg-chart-1 text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              ⚡ STBP (Surrogate)
            </button>
          </div>
        )}

        {/* Config */}
        <div className="flex flex-wrap gap-2 mb-3">
          {mode !== 'server' && (
            <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Brain size={12} /> {useConnectome ? `C. elegans (${busNeurons}N)` : `Mini SNN (${busNeurons}N)`}
            </label>
          )}
          {mode !== 'server' && useConnectome && (
            <label className="flex items-center gap-1.5 text-xs">
              <Target size={12} />
              <select value={connectomeSubset} onChange={e => setConnectomeSubset(e.target.value as ConnectomeSubset)}
                className="bg-secondary border border-border rounded px-1 py-0.5 text-xs font-mono" disabled={running}>
                <option value="full">Full (302n)</option>
                <option value="sensory">Sensory</option>
                <option value="motor">Motor</option>
                <option value="interneuron">Inter</option>
                <option value="sensory-motor">Sensory+Motor</option>
              </select>
            </label>
          )}
          <label className="flex items-center gap-1.5 text-xs">
            <RotateCcw size={12} />
            Epochs:
            <select value={epochs} onChange={e => setEpochs(+e.target.value)}
              className="bg-secondary border border-border rounded px-1 py-0.5 text-xs font-mono" disabled={running}>
              {(mode === 'server' ? [5, 10, 20, 50] : (mode === 'compare' || mode === 'decussation') ? [3, 5, 10] : [1, 2, 3, 5, 10, 20, 50]).map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          {mode === 'local' && (
            <label className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <input type="checkbox" checked={useWorker} onChange={e => setUseWorker(e.target.checked)}
                className="rounded border-border" disabled={running} />
              Web Worker
            </label>
          )}
          {mode === 'local' && trainMethod === 'stbp' && (
            <span className="text-[9px] text-chart-1 bg-chart-1/10 px-1.5 py-0.5 rounded font-mono">
              +Delays +SurrogateGrad
            </span>
          )}
        </div>

        {/* Start/Stop */}
        <button onClick={handleStart}
          className={`w-full py-2 rounded-lg font-bold text-sm flex items-center justify-center gap-2 transition-all ${
            running
              ? 'bg-destructive text-destructive-foreground'
              : mode === 'compare' || mode === 'decussation'
                ? 'bg-gradient-to-r from-chart-4 to-chart-1 text-primary-foreground hover:opacity-90'
                : 'bg-primary text-primary-foreground hover:opacity-90'
          }`}
          disabled={running && mode === 'server'}
        >
          {running
            ? mode === 'server'
              ? <><Loader2 size={14} className="animate-spin" /> Processing on server...</>
              : (mode === 'compare' || mode === 'decussation')
                ? <><Loader2 size={14} className="animate-spin" /> {comparePhase || 'Comparing...'}</>
                : <><Loader2 size={14} className="animate-spin" /> Stop</>
            : mode === 'compare'
              ? <><GitCompareArrows size={14} /> Compare R-STDP vs STBP ({epochs}ep)</>
              : mode === 'decussation'
                ? <><Brain size={14} /> Decusación A/B ({epochs}ep)</>
                : <><Play size={14} /> Start {trainMethod === 'stbp' ? 'STBP' : 'R-STDP'} ({mode === 'server' ? '10K records' : '500 samples'})</>}
        </button>

        {/* Progress bar */}
        {metrics && mode !== 'server' && !compareResults && (
          <div className="mt-2">
            <div className="flex items-center gap-2 mb-1">
              <phaseBadge.icon
                size={11}
                style={{ color: phaseBadge.color }}
                className={metrics.phase !== 'done' ? 'animate-spin' : ''}
              />
              <span className="text-[10px] font-mono" style={{ color: phaseBadge.color }}>{phaseBadge.text}</span>
              {metrics.phase === 'training' && (
                <span className="text-[9px] text-muted-foreground ml-auto font-mono">
                  Epoch {metrics.epoch}/{metrics.totalEpochs} • Sample {metrics.sample}
                </span>
              )}
              {metrics.phase === 'testing' && (
                <span className="text-[9px] text-muted-foreground ml-auto font-mono">
                  {metrics.sample}/{metrics.totalSamples} samples
                </span>
              )}
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, progress)}%`, backgroundColor: phaseBadge.color }}
              />
            </div>
            {metrics.phase === 'training' && (
              <div className="flex gap-2 mt-1">
                <span className="text-[9px] text-muted-foreground font-mono">Loss: {metrics.loss.toFixed(3)}</span>
                <span className="text-[9px] text-muted-foreground font-mono">Sparsity: {(metrics.sparsity * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Results: Compare mode */}
      {compareResults && (compareResults.rstdp || compareResults.stbp) && (
        <div className="space-y-2">
          <div className="text-xs font-bold text-muted-foreground flex items-center gap-1.5">
            <GitCompareArrows size={12} /> Algorithm Comparison
          </div>
          <div className="grid grid-cols-2 gap-2">
            {compareResults.rstdp && (
              <ComparisonCard
                label="🧬 R-STDP"
                metrics={compareResults.rstdp}
                color="hsl(var(--chart-4))"
                winner={(compareResults.rstdp?.f1Macro ?? 0) >= (compareResults.stbp?.f1Macro ?? 0)}
              />
            )}
            {compareResults.stbp && (
              <ComparisonCard
                label="⚡ STBP"
                metrics={compareResults.stbp}
                color="hsl(var(--chart-1))"
                winner={(compareResults.stbp?.f1Macro ?? 0) > (compareResults.rstdp?.f1Macro ?? 0)}
              />
            )}
          </div>
        </div>
      )}

      {/* Results: Decussation mode */}
      {decussationResults && (decussationResults.ipsilateral || decussationResults.contralateral) && (
        <div className="space-y-2">
          <div className="text-xs font-bold text-muted-foreground flex items-center gap-1.5">
            <Brain size={12} /> Decussation A/B
          </div>
          <div className="grid grid-cols-2 gap-2">
            {decussationResults.ipsilateral && (
              <ComparisonCard
                label="🔀 Ipsilateral"
                metrics={decussationResults.ipsilateral}
                color="hsl(var(--chart-3))"
                winner={(decussationResults.ipsilateral?.f1Macro ?? 0) >= (decussationResults.contralateral?.f1Macro ?? 0)}
              />
            )}
            {decussationResults.contralateral && (
              <ComparisonCard
                label="🧬 Contralateral"
                metrics={decussationResults.contralateral}
                color="hsl(var(--chart-2))"
                winner={(decussationResults.contralateral?.f1Macro ?? 0) > (decussationResults.ipsilateral?.f1Macro ?? 0)}
              />
            )}
          </div>
        </div>
      )}

      {/* Results: Single run */}
      {isDone && !compareResults && !decussationResults && (
        <div className="space-y-3">
          {/* Six Sigma */}
          {sigmaData && (
            <SixSigmaBadge sigma={sigmaData.sigma} dpmo={sigmaData.dpmo} />
          )}

          {/* Key metrics */}
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              icon={Activity}
              label="Accuracy"
              value={`${(displayAccuracy * 100).toFixed(2)}`}
              unit="%"
              color="hsl(var(--chart-2))"
            />
            <MetricCard
              icon={Target}
              label="F1 Macro"
              value={`${(displayF1 * 100).toFixed(2)}`}
              unit="%"
              color="hsl(var(--chart-4))"
            />
            {(serverResult?.synOps ?? metrics?.synOps) !== undefined && (
              <MetricCard
                icon={Brain}
                label="SynOps"
                value={((serverResult?.synOps ?? metrics?.synOps ?? 0) / 1000).toFixed(0)}
                unit="K"
                color="hsl(var(--chart-1))"
              />
            )}
            {(serverResult?.testSize ?? metrics?.elapsedMs) !== undefined && (
              <MetricCard
                icon={RotateCcw}
                label={mode === 'server' ? 'Test Samples' : 'Elapsed'}
                value={mode === 'server'
                  ? String(serverResult?.testSize ?? 0)
                  : `${((metrics?.elapsedMs ?? 0) / 1000).toFixed(1)}`}
                unit={mode === 'server' ? '' : 's'}
                color="hsl(var(--chart-3))"
              />
            )}
          </div>

          {/* Per-class accuracy bars */}
          {displayPerClass && (
            <div className="bg-card border border-border rounded-xl p-3">
              <div className="text-[10px] font-bold text-muted-foreground mb-2">Per-Class Accuracy</div>
              <ClassBars perClass={displayPerClass} />
            </div>
          )}

          {/* Confusion matrix */}
          {displayConfusion && (
            <div className="bg-card border border-border rounded-xl p-3">
              <div className="text-[10px] font-bold text-muted-foreground mb-2">Confusion Matrix</div>
              <ConfusionMatrix matrix={displayConfusion} />
              <div className="mt-2 flex gap-3 text-[8px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: 'hsl(var(--chart-2) / 0.7)' }} />
                  Correct (diagonal)
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: 'hsl(var(--destructive) / 0.4)' }} />
                  Misclassified
                </span>
              </div>
            </div>
          )}

          {/* Diagnostics */}
          {diagnostics && (
            <div className="bg-card border border-border rounded-xl p-3">
              <div className="text-[10px] font-bold text-muted-foreground mb-2">Engine Diagnostics</div>
              <div className="grid grid-cols-2 gap-2 text-[9px] font-mono">
                <div className="bg-muted rounded p-1.5">
                  <div className="text-muted-foreground">Cumulative Reward</div>
                  <div className="text-chart-2">{diagnostics.cumulativeReward.toFixed(2)}</div>
                </div>
                <div className="bg-muted rounded p-1.5">
                  <div className="text-muted-foreground">Avg Recent Reward</div>
                  <div className="text-chart-2">{diagnostics.avgRecentReward.toFixed(3)}</div>
                </div>
                <div className="bg-muted rounded p-1.5">
                  <div className="text-muted-foreground">Correlation</div>
                  <div className="text-chart-4">{diagnostics.correlation.toFixed(3)}</div>
                </div>
                <div className="bg-muted rounded p-1.5">
                  <div className="text-muted-foreground">Generation</div>
                  <div className="text-chart-1">{diagnostics.generation}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Engine info footer */}
      <div className="text-[9px] text-muted-foreground text-center font-mono pb-2">
        {organismLabel} • {mode === 'server' ? 'Edge Function' : useWorker ? 'Web Worker' : 'Main Thread'} • AI4I 2020 UCI
      </div>
    </div>
  );
};

export default AI4IDashboard;
