use ndarray::{Array1, Array2, Array3, prelude::*};
use crate::{Forward, Backward, Initialize, LiquidConfig, LiquidResult, InputModality, OutputModality};
use rand::Rng;
use tracing::{debug, trace};
use std::result::Result as StdResult;

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

type Result<T> = StdResult<T, AttentionError>;

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

    /// Split input into multiple attention heads
    fn split_heads(&self, x: &Array2<f64>) -> Result<Array3<f64>> {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        if seq_len % self.num_heads != 0 {
            return Err(AttentionError::InvalidShape);
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
    fn attention(&mut self, query: &Array3<f64>, key: &Array3<f64>, value: &Array3<f64>, mask: Option<&Array2<f64>>) -> Result<Array3<f64>> {
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
    pub fn compute_scores(&self, query: &Array1<f64>, key_matrix: &Array2<f64>) -> LiquidResult<Array1<f64>> {
        debug!("Computing attention scores for expert selection");
        
        // Project query
        let projected_query = self.query_proj.dot(query);
        
        // Project keys
        let projected_keys = self.key_proj.dot(&key_matrix.t());
        
        // Compute scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = projected_query.dot(&projected_keys) / scale;
        
        // Apply softmax to get attention weights
        let attention_weights = crate::utils::numerical::stable_softmax(&scores);
        
        trace!("Computed attention weights: {:?}", attention_weights);
        
        Ok(attention_weights)
    }

    /// Applies dropout to the input tensor with probability p
    /// x: Input tensor to apply dropout to
    /// p: Dropout probability (elements are zeroed with probability p)
    fn dropout(&self, x: &mut Array3<f64>, p: f64) -> Result<()> {
        // Validate dropout probability
        if p < 0.0 || p >= 1.0 {
            return Err(AttentionError::InvalidDimension);
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
            return Err(AttentionError::InvalidDimension);
        }
        
        Ok(())
    }

    fn compute_attention_scores(&self, query: &Array3<f64>, key: &Array3<f64>) -> Result<Array3<f64>> {
        // Validate input shapes
        let q_shape = query.shape();
        let k_shape = key.shape();
        
        if q_shape[0] != k_shape[0] || q_shape[1] != k_shape[1] {
            return Err(AttentionError::InvalidShape);
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
            return Err(AttentionError::InvalidDimension);
        }
        
        Ok(scores)
    }

    fn softmax(&self, x: Array3<f64>) -> Result<Array3<f64>> {
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
                    return Err(AttentionError::InvalidDimension);
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
                    return Err(AttentionError::InvalidDimension);
                }
                
                // Normalize and store in result
                for i in 0..seq_len {
                    result[[b, h, i]] = exp_values[i] / sum;
                }
                
                // Verify results are valid probabilities
                let prob_sum: f64 = (0..seq_len).map(|i| result[[b, h, i]]).sum();
                if (prob_sum - 1.0).abs() > 1e-5 {
                    debug!("Softmax probabilities do not sum to 1.0: {}", prob_sum);
                    return Err(AttentionError::InvalidDimension);
                }
            }
        }
        
        Ok(result)
    }

    fn weighted_sum(&self, weights: &Array3<f64>, values: &Array3<f64>) -> Result<Array3<f64>> {
        // Validate input shapes
        let w_shape = weights.shape();
        let v_shape = values.shape();
        
        if w_shape[0] != v_shape[0] || w_shape[1] != v_shape[1] {
            return Err(AttentionError::InvalidShape);
        }
        
        let batch_size = w_shape[0];
        let num_heads = w_shape[1];
        let seq_len = w_shape[2];
        
        // Create output array
        let mut output = Array3::zeros((batch_size, num_heads, seq_len));
        
        // Pre-allocate sum arrays for better cache locality
        let mut sums = vec![0.0; seq_len];
        
        // Compute weighted sum for each batch and head
        for b in 0..batch_size {
            for h in 0..num_heads {
                // Clear the sums for this batch and head
                for s in &mut sums {
                    *s = 0.0;
                }
                
                // Extract views for current batch and head for better cache locality
                let w_slice = weights.slice(s![b, h, ..]);
                let v_slice = values.slice(s![b, h, ..]);
                
                // First compute all weighted sums
                for j in 0..seq_len {
                    let weight = w_slice[j];
                    for i in 0..seq_len {
                        sums[i] += weight * v_slice[j];
                    }
                }
                
                // Then write them to the output array
                for i in 0..seq_len {
                    output[[b, h, i]] = sums[i];
                }
            }
        }
        
        // Check for NaN values
        if output.iter().any(|&v| v.is_nan()) {
            debug!("NaN detected in weighted sum");
            return Err(AttentionError::InvalidDimension);
        }
        
        Ok(output)
    }

    /// Forward pass for self-attention with efficient processing
    pub fn forward(&self, x: &Array3<f64>) -> Result<Array3<f64>> {
        // Validate input shape
        if x.ndim() != 3 {
            return Err(AttentionError::InvalidDimension);
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

    fn combine_heads(&self, x: &Array3<f64>) -> Result<Array2<f64>> {
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

impl Forward for MultiHeadAttention {
    fn forward(&mut self, input: &InputModality) -> LiquidResult<OutputModality> {
        match input {
            InputModality::Text(text) => {
                // Track performance
                let start = std::time::Instant::now();
                debug!("Processing text input of length {}", text.len());
                
                // Convert text to embedding
                let embedding = crate::moe::text_to_embedding(text)?;
                
                // Perform projection operations
                let query = self.query_proj.dot(&embedding);
                let key = self.key_proj.dot(&embedding);
                let value = self.value_proj.dot(&embedding);
                
                // Safety check for NaN values
                if query.iter().any(|&x| x.is_nan()) || key.iter().any(|&x| x.is_nan()) || value.iter().any(|&x| x.is_nan()) {
                    return Err("NaN values detected in projection".into());
                }

                // Reshape arrays - copy data to avoid moves
                let query_2d = Array2::from_shape_vec((1, query.len()), query.iter().cloned().collect())?;
                let key_2d = Array2::from_shape_vec((1, key.len()), key.iter().cloned().collect())?;
                let value_2d = Array2::from_shape_vec((1, value.len()), value.iter().cloned().collect())?;

                // Split into heads
                let query_heads = self.split_heads(&query_2d).map_err(|e| format!("Error splitting query: {}", e))?;
                let key_heads = self.split_heads(&key_2d).map_err(|e| format!("Error splitting key: {}", e))?;
                let value_heads = self.split_heads(&value_2d).map_err(|e| format!("Error splitting value: {}", e))?;

                // Compute attention
                let attention_output = self.attention(&query_heads, &key_heads, &value_heads, None)
                    .map_err(|e| format!("Error in attention: {}", e))?;

                // Combine heads
                let combined = self.combine_heads(&attention_output).map_err(|e| format!("Error combining heads: {}", e))?;
                
                // Project output
                let output = self.output_proj.dot(&combined);
                
                // Convert 2D array to 1D array for embedding_to_text
                let output_1d = Array1::from_iter(output.row(0).iter().cloned());

                // Convert back to text
                let result = crate::moe::embedding_to_text(&output_1d)?;
                
                // Log performance metrics
                let elapsed = start.elapsed();
                debug!("Attention forward pass completed in {:?}", elapsed);
                
                Ok(OutputModality::Text(result))
            },
            _ => Err("Unsupported modality - only text is currently implemented".into()),
        }
    }
}

impl Backward for MultiHeadAttention {
    fn backward(&mut self, grad: &OutputModality) -> LiquidResult<()> {
        // This is a placeholder for the actual backpropagation implementation
        match grad {
            OutputModality::Text(_) => {
                debug!("Backpropagation for attention layer with text modality not yet implemented");
                // TODO: Implement backpropagation for attention mechanism:
                // 1. Compute gradients with respect to weights and inputs
                // 2. Update projection matrices (query_proj, key_proj, value_proj, output_proj)
                // 3. Return gradients for previous layer
            },
            _ => return Err("Unsupported modality for backpropagation".into()),
        }
        
        Ok(())
    }
}

impl Initialize for MultiHeadAttention {
    fn initialize(&mut self, config: &LiquidConfig) -> LiquidResult<()> {
        self.num_heads = config.attention_heads;
        self.head_dim = config.embedding_dim / config.attention_heads;
        self.dropout = config.dropout as f64;
        
        // Initialize weights with Xavier initialization
        let dim = config.embedding_dim;
        self.query_proj = crate::utils::init::xavier_init((dim, dim));
        self.key_proj = crate::utils::init::xavier_init((dim, dim));
        self.value_proj = crate::utils::init::xavier_init((dim, dim));
        self.output_proj = crate::utils::init::xavier_init((dim, dim));
        
        Ok(())
    }
}

