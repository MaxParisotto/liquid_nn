use crate::neuron::Neuron;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::time::Instant;

/// Represents a layer of liquid neurons with specific dynamics
pub struct LiquidLayer {
    neurons: Vec<Neuron>,
    time_constants: Array1<f64>,
    layer_index: usize,
}

impl LiquidLayer {
    pub fn new(size: usize, weight_matrix: Array2<f64>, input_weight: Array1<f64>, layer_idx: usize) -> Self {
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

    pub fn update_time_constants(&mut self, inputs: &Array1<f64>) {
        // Adaptive time constants based on input magnitude
        self.time_constants.iter_mut()
            .zip(inputs.iter())
            .for_each(|(tau, &input)| {
                *tau = 1.0 + (input.abs() * 0.1).tanh(); // Bounded adaptation
            });
    }
}

/// Main topology structure organizing multiple liquid layers
pub struct Topology {
    layers: Vec<LiquidLayer>,
    dt: f64,
    total_steps: usize,
    performance_stats: PerformanceStats,
}

/// Track performance metrics
struct PerformanceStats {
    forward_pass_times: Vec<f64>,
    max_state_values: Vec<f64>,
    min_state_values: Vec<f64>,
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
            performance_stats: PerformanceStats {
                forward_pass_times: Vec::new(),
                max_state_values: Vec::new(),
                min_state_values: Vec::new(),
            },
        }
    }

    /// Forward pass with RK4 integration and parallel computation
    pub fn forward(&mut self, inputs: Array1<f64>) {
        let timer = Instant::now();

        // Update time constants for each layer based on inputs
        self.layers.iter_mut().for_each(|layer| {
            layer.update_time_constants(&inputs);
        });

        // Convert inputs to Vec for parallel iteration
        let inputs_vec = inputs.to_vec();

        // Parallel forward pass through layers
        self.layers.par_iter_mut().for_each(|layer| {
            layer.neurons.par_iter_mut()
                .zip(inputs_vec.par_iter())
                .for_each(|(neuron, &input)| {
                    // RK4 integration step
                    let k1 = neuron.compute_derivative(input);
                    let k2 = neuron.compute_derivative(input + 0.5 * self.dt * k1);
                    let k3 = neuron.compute_derivative(input + 0.5 * self.dt * k2);
                    let k4 = neuron.compute_derivative(input + self.dt * k3);
                    
                    let delta = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
                    neuron.step(input, self.dt * delta);
                });
        });

        // Update performance statistics
        self.update_stats(timer.elapsed().as_secs_f64());
        self.total_steps += 1;
    }

    /// Update performance statistics
    fn update_stats(&mut self, forward_time: f64) {
        self.performance_stats.forward_pass_times.push(forward_time);

        // Calculate state value ranges
        let mut max_val = f64::MIN;
        let mut min_val = f64::MAX;

        for layer in &self.layers {
            for neuron in &layer.neurons {
                let state = neuron.state();
                max_val = max_val.max(state);
                min_val = min_val.min(state);
            }
        }

        self.performance_stats.max_state_values.push(max_val);
        self.performance_stats.min_state_values.push(min_val);

        // Keep only recent statistics
        const MAX_STATS_HISTORY: usize = 1000;
        if self.performance_stats.forward_pass_times.len() > MAX_STATS_HISTORY {
            self.performance_stats.forward_pass_times.remove(0);
            self.performance_stats.max_state_values.remove(0);
            self.performance_stats.min_state_values.remove(0);
        }
    }

    /// Adjust integration timestep based on state stability
    pub fn adapt_timestep(&mut self) {
        let recent_max_vals: Vec<f64> = self.performance_stats.max_state_values.iter()
            .rev()
            .take(100)
            .cloned()
            .collect();

        if let (Some(max), Some(min)) = (recent_max_vals.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
                                       recent_max_vals.iter().min_by(|a, b| a.partial_cmp(b).unwrap())) {
            let range = max - min;
            // Adjust dt based on state range
            if range > 10.0 {
                self.dt *= 0.5; // Decrease timestep for stability
            } else if range < 0.1 {
                self.dt *= 1.2; // Increase timestep for efficiency
            }
            // Ensure dt stays within reasonable bounds
            self.dt = self.dt.clamp(0.001, 0.1);
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> (f64, f64, f64) {
        let avg_forward_time: f64 = self.performance_stats.forward_pass_times.iter().sum::<f64>() 
            / self.performance_stats.forward_pass_times.len() as f64;
        let max_state: f64 = *self.performance_stats.max_state_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let min_state: f64 = *self.performance_stats.min_state_values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        
        (avg_forward_time, max_state, min_state)
    }
}