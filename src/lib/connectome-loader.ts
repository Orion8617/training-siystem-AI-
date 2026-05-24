// C. elegans connectome loader

export type ConnectomeSubset = 'full' | 'sensory' | 'motor' | 'interneuron' | 'sensory-motor';

export interface ConnectomeNeuron {
  id: number;
  name: string;
  type: 'sensory' | 'motor' | 'interneuron';
  hemisphere: 'left' | 'right' | 'center';
  x: number;
  y: number;
}

export interface ConnectomeSynapse {
  from: number;
  to: number;
  weight: number;
  type: 'chemical' | 'electrical';
}

export interface Connectome {
  neurons: ConnectomeNeuron[];
  synapses: ConnectomeSynapse[];
}

/** Load a subset of the C. elegans connectome (synthetic approximation) */
export function loadConnectome(subset: ConnectomeSubset, maxNeurons?: number): Connectome {
  const neuronTypes: Array<'sensory' | 'motor' | 'interneuron'> = ['sensory', 'motor', 'interneuron'];
  const hemispheres: Array<'left' | 'right' | 'center'> = ['left', 'right', 'center'];

  // Total C. elegans neurons = 302
  const totalNeurons = maxNeurons ?? (subset === 'full' ? 302 : subset === 'sensory-motor' ? 180 : 100);

  const neurons: ConnectomeNeuron[] = [];
  for (let i = 0; i < totalNeurons; i++) {
    const typeIdx = subset === 'sensory' ? 0 : subset === 'motor' ? 1 : subset === 'interneuron' ? 2 : i % 3;
    neurons.push({
      id: i,
      name: `N${i.toString().padStart(3, '0')}`,
      type: neuronTypes[typeIdx],
      hemisphere: hemispheres[i % 3],
      x: Math.random(),
      y: Math.random(),
    });
  }

  // Generate sparse synaptic connections (~4 per neuron on average)
  const synapses: ConnectomeSynapse[] = [];
  const avgSynapses = 4;
  for (let i = 0; i < totalNeurons * avgSynapses; i++) {
    const from = Math.floor(Math.random() * totalNeurons);
    const to = Math.floor(Math.random() * totalNeurons);
    if (from !== to) {
      synapses.push({
        from,
        to,
        weight: (Math.random() * 2 - 1) * 0.5,
        type: Math.random() < 0.7 ? 'chemical' : 'electrical',
      });
    }
  }

  return { neurons, synapses };
}
