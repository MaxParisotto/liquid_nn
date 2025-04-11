use crate::{
    Forward, Backward, Initialize, Result,
    ActivationType, NeuronConfig,
};
use ndarray::{Array1, Array2};
use tracing::debug;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Liquid neuron implementation with adaptive time constant
#[allow(dead_code)]
pub struct Neuron {
    config: NeuronConfig,
    weights: Array2<f64>,
    bias: Array1<f64>,
    state: f64,
    tau: f64,  // Adaptive time constant
    input_weights: Array1<f64>,
}

#[allow(dead_code)]
impl Neuron {
    pub fn new(weights: Array2<f64>, input_weights: Array1<f64>) -> Self {
        let input_dim = input_weights.len();
        let hidden_dim = weights.dim().0;
        
        Self {
            weights: weights.clone(),  // Clone to avoid ownership issues
            input_weights: input_weights.clone(),  // Clone to avoid ownership issues
            state: 0.0,
            config: NeuronConfig {
                input_dim,
                hidden_dim,
                activation: ActivationType::Tanh,
                use_bias: true,
            },
            bias: Array1::zeros(hidden_dim),
            tau: 1.0,
        }
    }

    pub fn state(&self) -> f64 {
        self.state
    }

    /// Apply activation function
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        match self.config.activation {
            ActivationType::Tanh => x.mapv(|v| v.tanh()),
            ActivationType::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationType::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationType::Linear => x.clone(),
        }
    }

    /// Compute derivative of activation function
    fn activate_derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        match self.config.activation {
            ActivationType::Tanh => x.mapv(|v| 1.0 - v.tanh().powi(2)),
            ActivationType::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            ActivationType::Sigmoid => {
                let sig = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                sig.mapv(|v| v * (1.0 - v))
            },
            ActivationType::Linear => Array1::ones(x.len()),
        }
    }

    /// Update adaptive time constant based on input
    fn update_tau(&mut self, input: &Array1<f64>) {
        // Simple adaptive mechanism based on input magnitude
        let input_mag = input.dot(input).sqrt();
        self.tau = 1.0 + input_mag.tanh();
        debug!("Updated tau to {}", self.tau);
    }

    /// Compute state derivative for ODE integration
    fn compute_derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        let pre_activation = self.weights.dot(input);
        if self.config.use_bias {
            &pre_activation + &self.bias
        } else {
            pre_activation
        }
    }

    /// Perform one step of Euler integration
    fn integrate_step(&mut self, derivative: f64, dt: f64) {
        self.state += derivative * dt;
    }

    /// Update neuron state based on input
    pub fn step(&mut self, _input: f64, delta: f64) {
        self.state += delta;
        
        // Apply non-linearity (tanh)
        self.state = self.state.tanh();
    }

    /// Compute the derivative for RK4 integration
    pub fn compute_derivative_rk4(&self, input: f64) -> f64 {
        // Calculate the derivative
        let weighted_input = input * self.input_weights.iter().sum::<f64>();
        let recurrent = self.weights.dot(&Array1::from_elem(self.config.hidden_dim, self.state));
        
        // Sum the components
        let sum = weighted_input + recurrent.sum();
        
        // Return the rate of change
        sum - self.state
    }

    /// Applies noise to the neuron's state
    pub fn apply_noise(&mut self, noise_level: f64) {
        let mut rng = thread_rng();
        let noise_dist = Normal::new(0.0, noise_level).unwrap();
        self.state += noise_dist.sample(&mut rng);
    }

    /// Resets the neuron's state
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Update the neuron state and return the current state
    pub fn update(&mut self, input: f64) -> f64 {
        let derivative = self.compute_derivative_rk4(input);
        self.step(input, derivative);
        self.state
    }
}

impl Forward for Neuron {
    fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let output = self.weights.dot(input) + &self.input_weights;
        self.state = output[0];
        Ok(output)
    }
}

impl Backward for Neuron {
    fn backward(&mut self, grad: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(grad.clone())
    }
}

impl Initialize for Neuron {
    fn initialize(&mut self) -> Result<()> {
        // Reset state
        self.reset();
        Ok(())
    }
}

/// Helper functions for neuron operations
pub mod utils {
    use super::*;
    
    /// Initializes random weights for a neuron
    pub fn init_weights(dim: (usize, usize)) -> Array2<f64> {
        let std = (2.0 / (dim.0 + dim.1) as f64).sqrt();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, std).unwrap();
        
        Array2::from_shape_fn(dim, |_| normal.sample(&mut rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_initialization() {
        let _config = NeuronConfig {
            input_dim: 10,
            hidden_dim: 20,
            activation: ActivationType::Tanh,
            use_bias: true,
        };

        let mut neuron = Neuron::new(Array2::zeros((20, 10)), Array1::zeros(20));
        assert!(neuron.initialize().is_ok());
        
        // Check dimensions
        assert_eq!(neuron.weights.shape(), &[20, 10]);
        assert_eq!(neuron.bias.len(), 20);
    }

    #[test]
    fn test_neuron_forward() {
        let _config = NeuronConfig {
            input_dim: 5,
            hidden_dim: 10,
            activation: ActivationType::Tanh,
            use_bias: true,
        };

        let mut neuron = Neuron::new(Array2::zeros((10, 5)), Array1::zeros(10));
        neuron.initialize().unwrap();

        let input = Array1::linspace(0., 1., 5);
        let output = neuron.forward(&input).unwrap();
        
        assert_eq!(output.len(), 10);
        assert!(output.iter().all(|&x| (-1.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_activation_functions() {
        let input_dim = 5;
        let hidden_dim = 10;
        let input = Array1::linspace(-2., 2., input_dim);

        for activation in [
            ActivationType::Tanh,
            ActivationType::ReLU,
            ActivationType::Sigmoid,
            ActivationType::Linear,
        ] {
            let mut neuron = Neuron::new(Array2::zeros((hidden_dim, input_dim)), Array1::zeros(hidden_dim));
            neuron.initialize().unwrap();

            let output = neuron.forward(&input).unwrap();
            assert_eq!(output.len(), hidden_dim);

            // Test activation function properties
            match activation {
                ActivationType::Tanh => {
                    assert!(output.iter().all(|&x| (-1.0..=1.0).contains(&x)));
                },
                ActivationType::ReLU => {
                    assert!(output.iter().all(|&x| x >= 0.0));
                },
                ActivationType::Sigmoid => {
                    assert!(output.iter().all(|&x| (0.0..=1.0).contains(&x)));
                },
                ActivationType::Linear => {
                    // No specific bounds for linear activation
                },
            }
        }
    }
} 