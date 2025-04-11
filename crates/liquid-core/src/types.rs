use ndarray::Array1;

/// Common types used throughout the library
pub type Vector = Array1<f64>;

/// State type for liquid neurons
#[derive(Debug, Clone)]
pub struct LiquidState {
    pub values: Vector,
    pub time: f64,
} 