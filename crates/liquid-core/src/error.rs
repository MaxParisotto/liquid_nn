use thiserror::Error;

#[derive(Error, Debug)]
pub enum LiquidError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Initialization error: {0}")]
    InitializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Parameter error: {0}")]
    ParameterError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

impl LiquidError {
    pub fn shape_mismatch(expected: Vec<usize>, got: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, got }
    }

    pub fn invalid_dimension(msg: impl Into<String>) -> Self {
        Self::InvalidDimension(msg.into())
    }

    pub fn numerical_error(msg: impl Into<String>) -> Self {
        Self::NumericalError(msg.into())
    }

    pub fn initialization_error(msg: impl Into<String>) -> Self {
        Self::InitializationError(msg.into())
    }

    pub fn parameter_error(msg: impl Into<String>) -> Self {
        Self::ParameterError(msg.into())
    }

    pub fn runtime_error(msg: impl Into<String>) -> Self {
        Self::RuntimeError(msg.into())
    }
} 