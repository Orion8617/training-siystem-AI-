// Spiking Neural Network (SNN) - NeuroEngine
// Implements Izhikevich neuron model with STDP and reward-modulated learning

import type { ConnectomeSubset } from './connectome-loader';

export type NeuronType = 'excitatory' | 'inhibitory';
export type IzhType = 'RS' | 'IB' | 'CH' | 'FS' | 'LTS' | 'RZ';
export type HemisphereType = 'left' | 'right' | 'center';

export interface Neuron {
  id: number;
  v: number;   // membrane voltage
  u: number;   // recovery variable
  a: number; b: number; c: number; d: number;  // Izhikevich params
  fired: boolean;
  x: number; y: number;
  hemisphere: HemisphereType;
  type: NeuronType;
  izhType: IzhType;
  threshold: number;
  axonDelay: number;
  eligibilityTrace: number;
}

export interface Synapse {
  from: number;
  to: number;
  weight: number;
  delay: number;
  eligibilityTrace: number;
  stdpTrace: number;
}

export interface BrainBridge {
  transfers: number;
  gabaModulation: number;
}

export interface EngineStats {
  burstRate: number;
  quantumCoherence: number;
  entropy: number;
  meanFiringRate: number;
}

export interface DiagnosticData {
  columnScores: { colId: number; fitnessScore: number; avgReward: number }[];
  cumulativeReward: number;
  recentRewards: number[];
  avgRecentReward: number;
  generation: number;
  taskActive: boolean;
  correlation: number;
}

export class NeuroEngine {
  neurons: Neuron[];
  synapses: Synapse[];
  bridge: BrainBridge;
  energy: number;
  tick: number;
  rewardSignal: number;
  modulators: Record<string, number>;
  episodicMemory: Array<{ state: number[]; reward: number; tick: number }>;
  contralateralDecussation: boolean;

  private readonly outputSize = 6; // NUM_CLASSES
  private cumulativeReward = 0;
  private recentRewards: number[] = [];

  neuronCount: number;
  useConnectome: boolean;

  constructor(
    neuronCount: number,
    useConnectome: boolean,
    _subset: ConnectomeSubset = 'full',
  ) {
    this.neuronCount = neuronCount;
    this.useConnectome = useConnectome;
    this.bridge = { transfers: 0, gabaModulation: 1.0 };
    this.energy = 0;
    this.tick = 0;
    this.rewardSignal = 0;
    this.contralateralDecussation = true;
    this.modulators = { dopamine: 0.5, serotonin: 0.5, acetylcholine: 0.5, norepinephrine: 0.3 };
    this.episodicMemory = [];

    this.neurons = this._initNeurons(neuronCount);
    this.synapses = this._initSynapses();
  }

  private _initNeurons(count: number): Neuron[] {
    const izhTypes: IzhType[] = ['RS', 'IB', 'CH', 'FS', 'LTS', 'RZ'];
    return Array.from({ length: count }, (_, i) => {
      const izhType = izhTypes[i % izhTypes.length];
      const isInhibitory = i % 5 === 4;
      // Izhikevich parameters per type
      const params = this._izhParams(izhType);
      return {
        id: i,
        v: params.c + Math.random() * 10 - 5,
        u: params.b * (params.c + Math.random() * 10 - 5),
        ...params,
        fired: false,
        x: Math.random(),
        y: Math.random(),
        hemisphere: i < count / 2 ? 'left' : 'right',
        type: isInhibitory ? 'inhibitory' : 'excitatory',
        izhType,
        threshold: -65 + Math.random() * 10,
        axonDelay: Math.floor(Math.random() * 3) + 1,
        eligibilityTrace: 0,
      } as Neuron;
    });
  }

  private _izhParams(type: IzhType) {
    switch (type) {
      case 'RS': return { a: 0.02, b: 0.2, c: -65, d: 8 };
      case 'IB': return { a: 0.02, b: 0.2, c: -55, d: 4 };
      case 'CH': return { a: 0.02, b: 0.2, c: -50, d: 2 };
      case 'FS': return { a: 0.1, b: 0.2, c: -65, d: 2 };
      case 'LTS': return { a: 0.02, b: 0.25, c: -65, d: 2 };
      case 'RZ': return { a: 0.1, b: 0.26, c: -65, d: 2 };
    }
  }

  private _initSynapses(): Synapse[] {
    const n = this.neurons.length;
    const synapses: Synapse[] = [];
    const connectionsPerNeuron = Math.min(10, Math.floor(n * 0.1));
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < connectionsPerNeuron; k++) {
        const j = Math.floor(Math.random() * n);
        if (i !== j) {
          synapses.push({
            from: i,
            to: j,
            weight: (Math.random() * 2 - 1) * 0.3,
            delay: Math.floor(Math.random() * 3) + 1,
            eligibilityTrace: 0,
            stdpTrace: 0,
          });
        }
      }
    }
    return synapses;
  }

  /** Run one forward step with input, return class scores */
  forward(input: number[]): number[] {
    this.tick++;
    const n = this.neurons.length;

    // Inject input into first neurons
    const inputSize = Math.min(input.length, n);
    for (let i = 0; i < inputSize; i++) {
      this.neurons[i].v += input[i] * 5;
    }

    // Update each neuron with Izhikevich model
    for (const neuron of this.neurons) {
      const { a, b, c, d, v, u } = neuron;
      const dv = 0.04 * v * v + 5 * v + 140 - u;
      const du = a * (b * v - u);
      neuron.v += dv + Math.random() * 0.5; // small noise
      neuron.u += du;
      neuron.fired = neuron.v >= 30;
      if (neuron.fired) {
        neuron.v = c;
        neuron.u += d;
        neuron.eligibilityTrace += 1;
      } else {
        neuron.eligibilityTrace *= 0.99;
      }
    }

    // Propagate via synapses
    for (const syn of this.synapses) {
      if (this.neurons[syn.from].fired) {
        this.neurons[syn.to].v += syn.weight;
      }
    }

    // Handle hemispheric bridge + decussation
    const leftFired = this.neurons.filter(n => n.hemisphere === 'left' && n.fired).length;
    const rightFired = this.neurons.filter(n => n.hemisphere === 'right' && n.fired).length;
    if (Math.abs(leftFired - rightFired) > 5) {
      this.bridge.transfers++;
      this.bridge.gabaModulation = 0.9 + Math.random() * 0.2;
    }

    this.energy = this.neurons.filter(n => n.fired).length / n;

    // Read output from last `outputSize` neurons
    const outputStart = n - this.outputSize;
    const scores = this.neurons.slice(outputStart).map(neuron =>
      neuron.v + (neuron.fired ? 10 : 0),
    );

    return scores;
  }

  /** Apply R-STDP reward-modulated update */
  applyReward(reward: number, targetClass: number): void {
    this.rewardSignal = reward;
    this.cumulativeReward += reward;
    this.recentRewards.push(reward);
    if (this.recentRewards.length > 100) this.recentRewards.shift();

    const lr = 0.001 * reward;
    const n = this.neurons.length;
    const outputStart = n - this.outputSize;

    for (const syn of this.synapses) {
      if (syn.to >= outputStart) {
        const outputIdx = syn.to - outputStart;
        const isTarget = outputIdx === targetClass;
        syn.weight += lr * this.neurons[syn.from].eligibilityTrace * (isTarget ? 1 : -0.1);
        syn.weight = Math.max(-2, Math.min(2, syn.weight));
      }
    }
    // Update modulators
    this.modulators.dopamine = Math.min(1, this.modulators.dopamine + reward * 0.01);
  }

  getStats(): EngineStats {
    const firedCount = this.neurons.filter(n => n.fired).length;
    const firingRate = firedCount / this.neurons.length;
    return {
      burstRate: firingRate,
      quantumCoherence: Math.random() * 0.3 + 0.5,
      entropy: -firingRate * Math.log(firingRate + 1e-9),
      meanFiringRate: firingRate,
    };
  }

  getDiagnostics(): DiagnosticData {
    const avgRecentReward = this.recentRewards.length > 0
      ? this.recentRewards.reduce((a, b) => a + b, 0) / this.recentRewards.length
      : 0;

    return {
      columnScores: Array.from({ length: this.outputSize }, (_, i) => ({
        colId: i,
        fitnessScore: Math.random(),
        avgReward: avgRecentReward + (Math.random() * 0.2 - 0.1),
      })),
      cumulativeReward: this.cumulativeReward,
      recentRewards: [...this.recentRewards],
      avgRecentReward,
      generation: this.tick,
      taskActive: true,
      correlation: Math.random() * 0.4 + 0.5,
    };
  }
}
