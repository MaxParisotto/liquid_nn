use std::error::Error;
use std::fmt;

/// Custom error type for liquid neural network operations
#[derive(Debug)]
pub enum LiquidError {
    InvalidShape(String),
    InvalidDimension(String),
    UnsupportedModality(String),
    ConfigurationError(String),
    RuntimeError(String),
}

impl fmt::Display for LiquidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiquidError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            LiquidError::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            LiquidError::UnsupportedModality(msg) => write!(f, "Unsupported modality: {}", msg),
            LiquidError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            LiquidError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl Error for LiquidError {} 