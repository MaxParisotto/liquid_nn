use ndarray::{Array1, Array2};
use crate::{Forward, Backward, Initialize, Result, NeuronConfig, ActivationType};
use crate::neuron::Neuron;

/// Represents a layer of liquid neurons with specific dynamics
pub struct LiquidLayer {
    neurons: Vec<Neuron>,
    time_constants: Array1<f64>,
    layer_index: usize,
}

impl LiquidLayer {
    pub fn new(size: usize, weight_matrix: Array2<f64>, input_weight: Array1<f64>, layer_idx: usize) -> Self {
        let config = NeuronConfig {
            input_dim: weight_matrix.shape()[1],
            hidden_dim: weight_matrix.shape()[0],
            activation: ActivationType::Tanh,
            use_bias: true,
        };
        
        let neurons = (0..size)
            .map(|_| Neuron::new(weight_matrix.clone(), input_weight.clone()))
            .collect();
        let time_constants = Array1::ones(size);
        
        LiquidLayer {
            neurons,
            time_constants,
            layer_index: layer_idx,
        }
    }
}

impl Forward for LiquidLayer {
    fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut output = Array1::zeros(self.neurons.len());
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let state = neuron.update(input[0]);
            output[i] = state;
        }
        Ok(output)
    }
}

impl Backward for LiquidLayer {
    fn backward(&mut self, grad: &Array1<f64>) -> Result<Array1<f64>> {
        let mut output = Array1::zeros(self.neurons[0].forward(&Array1::zeros(1))?.len());
        for (neuron, &g) in self.neurons.iter_mut().zip(grad.iter()) {
            output += &neuron.backward(&Array1::from_vec(vec![g]))?;
        }
        Ok(output)
    }
}

impl Initialize for LiquidLayer {
    fn initialize(&mut self) -> Result<()> {
        for neuron in &mut self.neurons {
            neuron.initialize()?;
        }
        Ok(())
    }
}

/// Main topology structure organizing multiple liquid layers
pub struct Topology {
    layers: Vec<LiquidLayer>,
    dt: f64,
    total_steps: usize,
}

impl Topology {
    pub fn new(layer_sizes: Vec<usize>, weight_matrix: Array2<f64>, input_weight: Array1<f64>) -> Self {
        let layers = layer_sizes.into_iter()
            .enumerate()
            .map(|(idx, size)| {
                LiquidLayer::new(size, weight_matrix.clone(), input_weight.clone(), idx)
            })
            .collect();

        Topology {
            layers,
            dt: 0.01, // Default timestep
            total_steps: 0,
        }
    }
}

impl Forward for Topology {
    fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }
}

impl Backward for Topology {
    fn backward(&mut self, grad: &Array1<f64>) -> Result<Array1<f64>> {
        let mut current = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current = layer.backward(&current)?;
        }
        Ok(current)
    }
}

impl Initialize for Topology {
    fn initialize(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.initialize()?;
        }
        Ok(())
    }
} 