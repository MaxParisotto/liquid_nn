use crate::error::LiquidError;

/// Result type for liquid neural network operations
pub type Result<T> = std::result::Result<T, LiquidError>; 