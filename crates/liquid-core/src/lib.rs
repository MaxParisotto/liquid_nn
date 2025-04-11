pub mod neuron;
pub mod topology;
pub mod solvers;
pub mod error;
pub mod types;
pub mod state;
pub mod liquid_layer;
pub mod scheduler;
//

/// Core LNN types and feedback API
pub use topology::{Topology, FeedbackMode};
pub use state::RecurrentState;
pub use crate::liquid_layer::{LiquidLayer, LayerMode, LnnModeConfig};
pub use crate::scheduler::{
    LnnScheduler, SchedulerDecision, FeedbackPolicy, LnnMetrics, LnnState, SchedulerConfig,
};

use ndarray::Array1;
use serde::{Deserialize, Serialize};

pub use crate::error::LiquidError;
pub type Result<T> = std::result::Result<T, LiquidError>;

/// Core traits for neural network components
pub trait Forward {
    fn forward(&mut self, input: &Array1<f64>) -> Result<Array1<f64>>;
}

pub trait Backward {
    fn backward(&mut self, grad: &Array1<f64>) -> Result<Array1<f64>>;
}

pub trait Initialize {
    fn initialize(&mut self) -> Result<()>;
}

/// Configuration for neural network components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub activation: ActivationType,
    pub use_bias: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    Tanh,
    ReLU,
    Sigmoid,
    Linear,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 128,
            activation: ActivationType::Tanh,
            use_bias: true,
        }
    }
}

/// Utility functions for numerical stability
pub mod numerical {
    use ndarray::Array1;
    use std::f64;

    pub fn stable_softmax(x: &Array1<f64>) -> Array1<f64> {
        let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp = x.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        exp.mapv(|v| v / sum)
    }

    pub fn log_sum_exp(x: &Array1<f64>) -> f64 {
        let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum = x.mapv(|v| (v - max).exp()).sum();
        max + sum.ln()
    }
} 