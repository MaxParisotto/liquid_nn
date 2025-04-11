use ndarray::{Array1, Array2, Array3, prelude::*};
use crate::{Forward, Backward, Initialize};
use rand::Rng;
use tracing::{debug, trace};
use std::result::Result as StdResult;
//

// === BEGIN: Minimal stubs for compilation and testing ===

use std::fmt;

#[derive(Debug, Clone)]
pub struct LiquidConfig {
    pub embedding_dim: usize,
    pub attention_heads: usize,
    pub dropout: f32,
}
impl Default for LiquidConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 8,
            attention_heads: 2,
            dropout: 0.0,
        }
    }
}


#[derive(Debug, Clone)]
pub enum InputModality {
    Text(String),
    // Add other variants as needed
}

#[derive(Debug, Clone)]
pub enum OutputModality {
    Text(String),
    // Add other variants as needed
}

// Minimal stub for text <-> embedding conversion
mod moe {
    use super::*;
    pub fn text_to_embedding(_text: &str) -> Result<Array1<f64>, String> {
        // Return a fixed-size vector for testing
        Ok(Array1::from(vec![1.0; 8]))
    }
    pub fn embedding_to_text(_embedding: &Array1<f64>) -> Result<String, String> {
        Ok("stub".to_string())
    }
}

// Minimal stub for Xavier initialization
mod utils {
    use super::*;
    pub mod init {
        use super::*;
        pub fn xavier_init(shape: (usize, usize)) -> Array2<f64> {
            Array2::from_elem(shape, 0.5)
        }
    }
    pub mod numerical {
        use super::*;
        pub fn stable_softmax(x: &Array1<f64>) -> Array1<f64> {
            let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp = x.mapv(|v| (v - max).exp());
            let sum = exp.sum();
            exp.mapv(|v| v / sum)
        }
    }
}

// === END: Minimal stubs for compilation and testing ===


/// Custom error type for attention operations
#[derive(Debug)]
pub enum AttentionError {
    InvalidShape,
    InvalidDimension,
}

impl std::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionError::InvalidShape => write!(f, "Invalid array shape"),
            AttentionError::InvalidDimension => write!(f, "Invalid array dimension"),
        }
    }
}

impl std::error::Error for AttentionError {}


/// Multi-head attention layer
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    query_proj: Array2<f64>,
    key_proj: Array2<f64>,
    value_proj: Array2<f64>,
    output_proj: Array2<f64>,
    dropout: f64,
}

impl MultiHeadAttention {
    pub fn new(config: &LiquidConfig) -> Self {
        let head_dim = config.embedding_dim / config.attention_heads;
        
        Self {
            num_heads: config.attention_heads,
            head_dim,
            query_proj: Array2::zeros((config.embedding_dim, config.embedding_dim)),
            key_proj: Array2::zeros((config.embedding_dim, config.embedding_dim)),
            value_proj: Array2::zeros((config.embedding_dim, config.embedding_dim)),
            output_proj: Array2::zeros((config.embedding_dim, config.embedding_dim)),
            dropout: config.dropout as f64,
        }
    }

    /// Public getters for testing
    pub fn query_proj(&self) -> &Array2<f64> {
        &self.query_proj
    }
    pub fn key_proj(&self) -> &Array2<f64> {
        &self.key_proj
    }
    pub fn value_proj(&self) -> &Array2<f64> {
        &self.value_proj
    }
    pub fn output_proj(&self) -> &Array2<f64> {
        &self.output_proj
    }

    /// Split input into multiple attention heads
    fn split_heads(&self, x: &Array2<f64>) -> Result<Array3<f64>, String> {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        if seq_len % self.num_heads != 0 {
            return Err("Invalid shape in split_heads".to_string());
        }
        
        // Create a new array with the desired shape
        let mut result = Array3::zeros((batch_size, self.num_heads, seq_len / self.num_heads));
        
        // Copy data from input array
        for i in 0..batch_size {
            for j in 0..self.num_heads {
                for k in 0..(seq_len / self.num_heads) {
                    result[[i, j, k]] = x[[i, j * (seq_len / self.num_heads) + k]];
                }
            }
        }
        
        Ok(result)
    }

    /// Compute scaled dot-product attention
    fn attention(&mut self, query: &Array3<f64>, key: &Array3<f64>, value: &Array3<f64>, mask: Option<&Array2<f64>>) -> Result<Array3<f64>, String> {
        // Compute attention scores
        let scores = self.compute_attention_scores(query, key)?;
        
        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            let mut masked_scores = scores.clone();
            for i in 0..masked_scores.shape()[0] {
                for j in 0..masked_scores.shape()[1] {
                    for k in 0..masked_scores.shape()[2] {
                        masked_scores[[i, j, k]] += mask[[j, k]];
                    }
                }
            }
            masked_scores
        } else {
            scores
        };
        
        // Apply softmax
        let weights = self.softmax(scores)?;
        
        // Apply dropout
        let mut weights = weights;
        self.dropout(&mut weights, self.dropout)?;
        
        // Compute weighted sum
        let output = self.weighted_sum(&weights, value)?;
        
        Ok(output)
    }

    /// Compute attention scores for expert selection
    pub fn compute_scores(&self, query: &Array1<f64>, key_matrix: &Array2<f64>) -> Result<Array1<f64>, String> {
        debug!("Computing attention scores for expert selection");
        
        // Project query
        let projected_query = self.query_proj.dot(query);
        
        // Project keys
        let projected_keys = self.key_proj.dot(&key_matrix.t());
        
        // Compute scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = projected_query.dot(&projected_keys) / scale;
        
        // Apply softmax to get attention weights
        let attention_weights = utils::numerical::stable_softmax(&scores);
        
        trace!("Computed attention weights: {:?}", attention_weights);
        
        Ok(attention_weights)
    }

    /// Applies dropout to the input tensor with probability p
    /// x: Input tensor to apply dropout to
    /// p: Dropout probability (elements are zeroed with probability p)
    fn dropout(&self, x: &mut Array3<f64>, p: f64) -> Result<(), String> {
        // Validate dropout probability
        if p < 0.0 || p >= 1.0 {
            return Err("Invalid dropout probability".to_string());
        }
        
        // If dropout is zero, no need to do anything
        if p == 0.0 {
            return Ok(());
        }
        
        let scale = 1.0 / (1.0 - p);
        let mut rng = rand::thread_rng();
        
        // Generate and apply dropout mask in a single pass for better cache locality
        for elem in x.iter_mut() {
            // Keep value with probability (1-p)
            if rng.gen::<f64>() >= p {
                *elem *= scale;
            } else {
                *elem = 0.0;
            }
        }
        
        // Check for any NaN values after dropout
        if x.iter().any(|&v| v.is_nan()) {
            debug!("NaN detected after dropout");
            return Err("NaN detected after dropout".to_string());
        }
        
        Ok(())
    }

    fn compute_attention_scores(&self, query: &Array3<f64>, key: &Array3<f64>) -> Result<Array3<f64>, String> {
        // Validate input shapes
        let q_shape = query.shape();
        let k_shape = key.shape();
        
        if q_shape[0] != k_shape[0] || q_shape[1] != k_shape[1] {
            return Err("Shape mismatch in compute_attention_scores".to_string());
        }
        
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let q_seq_len = q_shape[2];
        let k_seq_len = k_shape[2];
        
        // Compute scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        
        // Pre-allocate the scores array
        let mut scores = Array3::zeros((batch_size, num_heads, q_seq_len));
        
        // Process each batch and head in parallel
        for b in 0..batch_size {
            for h in 0..num_heads {
                // Pre-compute all key values for this batch and head
                let q_batch_head = query.slice(s![b, h, ..]);
                let k_batch_head = key.slice(s![b, h, ..]);
                
                // Compute dot product for this batch and head
                for i in 0..q_seq_len {
                    let q_i = q_batch_head[i];
                    let mut dot_product = 0.0;
                    
                    // Vectorized dot product with cache locality
                    for j in 0..k_seq_len {
                        dot_product += q_i * k_batch_head[j];
                    }
                    
                    // Apply scaling for numerical stability
                    scores[[b, h, i]] = dot_product / scale;
                }
            }
        }
        
        // Check for NaN values
        if scores.iter().any(|&v| v.is_nan()) {
            debug!("NaN detected in attention scores");
            return Err("NaN detected in attention scores".to_string());
        }
        
        Ok(scores)
    }

    fn softmax(&self, x: Array3<f64>) -> Result<Array3<f64>, String> {
        // Get shape information
        let shape = x.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        
        // Create output array
        let mut result = Array3::zeros(x.dim());
        
        // Apply softmax along the last dimension for each batch and head
        for b in 0..batch_size {
            for h in 0..num_heads {
                // Find max for numerical stability (log-sum-exp trick)
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..seq_len {
                    max_val = max_val.max(x[[b, h, i]]);
                }
                
                // Early check for numerical issues
                if max_val.is_nan() || max_val.is_infinite() {
                    debug!("Invalid max value in softmax: {}", max_val);
                    return Err("Invalid max value in softmax".to_string());
                }
                
                // Compute shifted exp values for better numerical stability
                let mut sum = 0.0;
                let mut exp_values = vec![0.0; seq_len];
                
                for i in 0..seq_len {
                    let shifted_val = x[[b, h, i]] - max_val;
                    let exp_val = shifted_val.exp();
                    exp_values[i] = exp_val;
                    sum += exp_val;
                }
                
                // Avoid division by zero
                if sum <= f64::EPSILON {
                    debug!("Sum is too small in softmax: {}", sum);
                    return Err("Sum is too small in softmax".to_string());
                }
                
                // Normalize and store in result
                for i in 0..seq_len {
                    result[[b, h, i]] = exp_values[i] / sum;
                }
                
                // Verify results are valid probabilities
                let prob_sum: f64 = (0..seq_len).map(|i| result[[b, h, i]]).sum();
                if (prob_sum - 1.0).abs() > 1e-5 {
                    debug!("Softmax probabilities do not sum to 1.0: {}", prob_sum);
                    return Err("Softmax probabilities do not sum to 1.0".to_string());
                }
            }
        }
        
        Ok(result)
    }

    /// Compute weighted sum of value vectors with attention weights
    fn weighted_sum(&self, weights: &Array3<f64>, values: &Array3<f64>) -> Result<Array3<f64>, String> {
        // Validate input shapes
        let w_shape = weights.shape();
        let v_shape = values.shape();
        
        if w_shape[0] != v_shape[0] || w_shape[1] != v_shape[1] || w_shape[2] != v_shape[2] {
            return Err("Shape mismatch in weighted_sum".to_string());
        }
        
        let batch_size = w_shape[0];
        let num_heads = w_shape[1];
        let seq_len = w_shape[2];
        
        // Pre-allocate the output array
        let mut output = Array3::zeros((batch_size, num_heads, seq_len));
        
        // Compute weighted sum
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    output[[b, h, i]] = weights[[b, h, i]] * values[[b, h, i]];
                }
            }
        }
        
        Ok(output)
    }

    /// Forward pass for self-attention with efficient processing
    pub fn forward(&self, x: &Array3<f64>) -> Result<Array3<f64>, String> {
        // Validate input shape
        if x.ndim() != 3 {
            return Err("Input to forward must be 3D".to_string());
        }
        
        // Extract query, key, and value from the input (they're the same for self-attention)
        let query = x;
        let key = x;
        let value = x;
        
        // Compute attention scores directly
        let scores = self.compute_attention_scores(query, key)?;
        
        // Apply softmax to get attention weights
        let weights = self.softmax(scores)?;
        
        // Apply dropout if needed
        let mut weights_with_dropout = weights.clone();
        if self.dropout > 0.0 {
            self.dropout(&mut weights_with_dropout, self.dropout)?;
        }
        
        // Compute weighted sum of values
        let output = self.weighted_sum(&weights_with_dropout, value)?;
        
        Ok(output)
    }

    fn combine_heads(&self, x: &Array3<f64>) -> Result<Array2<f64>, String> {
        let shape = x.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        
        // Create a new array with the desired shape
        let mut result = Array2::zeros((batch_size, seq_len * num_heads));
        
        // Copy data from input array
        for i in 0..batch_size {
            for j in 0..num_heads {
                for k in 0..seq_len {
                    result[[i, j * seq_len + k]] = x[[i, j, k]];
                }
            }
        }
        
        Ok(result)
    }
}


