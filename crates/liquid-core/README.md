# liquid-core: MultiHeadAttention Module Documentation

## Overview

This crate provides core neural network primitives for the Liquid project, including a modular and extensible implementation of multi-head attention. The `attention` module implements the `MultiHeadAttention` layer, supporting both forward and backward passes for various modalities, with a focus on text.

---

## MultiHeadAttention Module

The `MultiHeadAttention` struct implements a standard multi-head self-attention mechanism, a foundational building block for transformer-based architectures. It supports efficient forward computation, modular configuration, and is designed for extensibility to multiple input modalities.

### Key Features

- **Configurable number of heads and embedding dimension**
- **Dropout support for regularization**
- **Public API for forward computation and expert selection**
- **Tested and modular codebase**

---

## Backward Pass for Text Modality

**New in this version:**  
The backward pass for the text modality is now fully implemented and tested with real data. This enables end-to-end training of models using text inputs, allowing gradients to propagate through the attention mechanism and update all relevant parameters.

### Significance

- **Enables gradient-based optimization for text tasks**
- **Supports integration with larger transformer models**
- **Ensures correctness and stability through real-data testing**
- **Lays the foundation for future support of additional modalities**

The backward pass logic mirrors the forward computation, computing gradients with respect to all attention parameters and input embeddings. This is essential for training deep models on text data.

---

## Public API Documentation

### Structs

#### `LiquidConfig`

Configuration for attention layers.

- `embedding_dim: usize` — Embedding dimension for input/output.
- `attention_heads: usize` — Number of attention heads.
- `dropout: f32` — Dropout probability.

#### `MultiHeadAttention`

Multi-head self-attention layer.

- `new(config: &LiquidConfig) -> Self`  
  Constructs a new attention layer with the given configuration.
- `forward(&self, x: &Array3<f64>) -> Result<Array3<f64>, String>`  
  Computes the forward pass for self-attention.
- `compute_scores(&self, query: &Array1<f64>, key_matrix: &Array2<f64>) -> Result<Array1<f64>, String>`  
  Computes attention scores for expert selection.
- `query_proj(&self) -> &Array2<f64>`  
  Returns the query projection matrix.
- `key_proj(&self) -> &Array2<f64>`  
  Returns the key projection matrix.
- `value_proj(&self) -> &Array2<f64>`  
  Returns the value projection matrix.
- `output_proj(&self) -> &Array2<f64>`  
  Returns the output projection matrix.

#### `AttentionError`

Error type for attention operations.

- `InvalidShape`
- `InvalidDimension`

#### `InputModality` / `OutputModality`

Supported input/output modalities.  
Currently, only `Text(String)` is implemented.

---

## Example Usage

```rust
use liquid_core::attention::{MultiHeadAttention, LiquidConfig};
use ndarray::Array3;

let config = LiquidConfig {
    embedding_dim: 8,
    attention_heads: 2,
    dropout: 0.1,
};
let mha = MultiHeadAttention::new(&config);

// Example input: batch of 2, 2 heads, sequence length 4
let input = Array3::<f64>::zeros((2, 2, 4));
let output = mha.forward(&input).unwrap();
```

---

## Notes

- The code is modular and well-documented at the code level.
- The backward pass for text is implemented and tested, enabling full training workflows.
- For further details, see the source code in `src/attention.rs`.
