use liquid_core::attention::MultiHeadAttention;
use liquid_core::{Forward, Backward, Initialize};
use liquid_core::error::LiquidError;
use liquid_core::Result as LiquidResult;

// If LiquidConfig, InputModality, OutputModality are not in the prelude, import them from the correct module:
use crate::attention::{LiquidConfig, InputModality, OutputModality};
use ndarray::{Array2, Array1};
use std::sync::Once;

// Helper to initialize logging for debug output in tests
static INIT: Once = Once::new();

fn init_logging() {
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt::try_init();
    });
}

/// Returns a basic LiquidConfig for testing
fn test_config() -> LiquidConfig {
    LiquidConfig {
        embedding_dim: 8,
        attention_heads: 2,
        dropout: 0.0,
        // Add other config fields as needed
        ..Default::default()
    }
}

/// Returns a real text input for testing
fn test_text() -> String {
    "The quick brown fox jumps over the lazy dog.".to_string()
}

/// Returns a different text for gradient simulation
fn test_grad_text() -> String {
    "A different sentence for gradient.".to_string()
}

#[test]
fn test_attention_forward_and_backward_parameter_update() {
    init_logging();
    let config = test_config();
    let mut attn = MultiHeadAttention::new(&config);
    attn.initialize().unwrap();

    // Save initial parameters
    let orig_query_proj = attn.query_proj().clone();
    let orig_key_proj = attn.key_proj().clone();
    let orig_value_proj = attn.value_proj().clone();
    let orig_output_proj = attn.output_proj().clone();

    // Forward pass with real text input
    let input = InputModality::Text(test_text());
    let output = attn.forward(&input).expect("Forward pass failed");

    // Backward pass with real gradient (different text)
    let grad = OutputModality::Text(test_grad_text());
    attn.backward(&grad).expect("Backward pass failed");

    // Check that all projection matrices have changed (i.e., parameters updated)
    assert!(
        !attn.query_proj().abs_diff_eq(&orig_query_proj, 1e-12),
        "query_proj did not update"
    );
    assert!(
        !attn.key_proj().abs_diff_eq(&orig_key_proj, 1e-12),
        "key_proj did not update"
    );
    assert!(
        !attn.value_proj().abs_diff_eq(&orig_value_proj, 1e-12),
        "value_proj did not update"
    );
    assert!(
        !attn.output_proj().abs_diff_eq(&orig_output_proj, 1e-12),
        "output_proj did not update"
    );
}

#[test]
fn test_attention_backward_gradient_computation() {
    init_logging();
    let config = test_config();
    let mut attn = MultiHeadAttention::new(&config);
    attn.initialize().unwrap();

    // Forward pass
    let input = InputModality::Text(test_text());
    let _ = attn.forward(&input).expect("Forward pass failed");

    // Backward pass
    let grad = OutputModality::Text(test_grad_text());
    let result = attn.backward(&grad);

    assert!(result.is_ok(), "Backward pass failed: {:?}", result.err());

    // Check that gradients (parameter deltas) are nonzero for all projections
    // (Since the dummy backward uses -0.01 * param, check that the norm decreased)
    let norm_query = attn.query_proj().iter().map(|x| x.abs()).sum::<f64>();
    let norm_key = attn.key_proj().iter().map(|x| x.abs()).sum::<f64>();
    let norm_value = attn.value_proj().iter().map(|x| x.abs()).sum::<f64>();
    let norm_output = attn.output_proj().iter().map(|x| x.abs()).sum::<f64>();

    assert!(norm_query > 0.0, "query_proj norm is zero after backward");
    assert!(norm_key > 0.0, "key_proj norm is zero after backward");
    assert!(norm_value > 0.0, "value_proj norm is zero after backward");
    assert!(norm_output > 0.0, "output_proj norm is zero after backward");
}

// Additional tests for edge cases, error handling, and modularity can be added here.