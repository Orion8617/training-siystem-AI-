// NeuralBus - reactive event bus for neural engine state synchronization

export interface EngineConfig {
  useConnectome: boolean;
  neurons: number;
  organism?: string;
}

export interface BusEvent {
  stats: {
    burstRate: number;
    quantumCoherence: number;
    entropy: number;
    meanFiringRate: number;
  };
  spikePositions: Array<{ x: number; y: number; hemisphere: string; type: string; izhType: string }>;
  firedCount: number;
  gabaLevel: number;
  coherence: number;
  entropy: number;
  bridgeActive: boolean;
  fieldAmp: number;
  tick: number;
  rewardSignal: number;
  burstRate: number;
  webgpuActive: boolean;
  modulators: Record<string, number>;
  episodicCount: number;
}

type ConfigChangeCallback = (config: EngineConfig) => void;

class NeuralBus {
  engineConfig: EngineConfig = {
    useConnectome: false,
    neurons: 128,
    organism: 'mini_128',
  };

  private configListeners: Set<ConfigChangeCallback> = new Set();
  private lastEvent: BusEvent | null = null;

  emit(event: BusEvent): void {
    this.lastEvent = event;
  }

  getLastEvent(): BusEvent | null {
    return this.lastEvent;
  }

  onConfigChange(callback: ConfigChangeCallback): () => void {
    this.configListeners.add(callback);
    return () => this.configListeners.delete(callback);
  }

  setConfig(config: Partial<EngineConfig>): void {
    this.engineConfig = { ...this.engineConfig, ...config };
    for (const listener of this.configListeners) {
      listener(this.engineConfig);
    }
  }
}

export const neuralBus = new NeuralBus();
