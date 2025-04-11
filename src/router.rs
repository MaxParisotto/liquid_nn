use crate::{
    LiquidConfig, LiquidResult, 
    modalities::{ModalityConverter, ModalityHandler},
    InputModality,
    attention::MultiHeadAttention
};
use ndarray::{Array1, Array2};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{info, warn};

/// Router configuration for expert selection
#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub attention_heads: usize,
    pub embedding_dim: usize,
    pub num_experts: usize,
    pub routing_capacity: f64,  // Fraction of inputs that can be routed to each expert
}

/// Attention-based router for expert selection
pub struct ExpertRouter {
    attention: MultiHeadAttention,
    modality_handler: Arc<ModalityHandler>,
    expert_capacities: Vec<AtomicCounter>,
    config: RouterConfig,
}

/// Atomic counter for tracking expert usage
struct AtomicCounter(parking_lot::RwLock<usize>);

impl AtomicCounter {
    fn new() -> Self {
        Self(RwLock::new(0))
    }

    fn increment(&self) -> bool {
        let mut count = self.0.write();
        *count += 1;
        true
    }

    fn decrement(&self) {
        let mut count = self.0.write();
        *count = count.saturating_sub(1);
    }

    fn get(&self) -> usize {
        *self.0.read()
    }
}

impl ExpertRouter {
    pub fn new(config: RouterConfig, modality_handler: Arc<ModalityHandler>) -> Self {
        let attention_config = LiquidConfig {
            attention_heads: config.attention_heads,
            embedding_dim: config.embedding_dim,
            dropout: 0.1,
            ..Default::default()
        };

        let attention = MultiHeadAttention::new(&attention_config);
        let expert_capacities = (0..config.num_experts)
            .map(|_| AtomicCounter::new())
            .collect();

        Self {
            attention,
            modality_handler,
            expert_capacities,
            config,
        }
    }

    /// Route input to the most appropriate expert
    pub fn route(&self, input: &InputModality) -> LiquidResult<(usize, Array1<f64>)> {
        // Convert input to embedding
        let embedding = self.modality_handler.to_embedding(input)?;
        
        // Get attention scores for each expert
        let attention_scores = self.compute_expert_scores(&embedding)?;
        
        // Select expert based on scores and capacity
        let selected_expert = self.select_expert(&attention_scores)?;
        
        info!(
            "Routing input to expert {}, modality: {:?}",
            selected_expert,
            input
        );
        
        Ok((selected_expert, embedding))
    }

    /// Compute attention scores for expert selection
    fn compute_expert_scores(&self, embedding: &Array1<f64>) -> LiquidResult<Array1<f64>> {
        // Create query from input embedding
        let query = embedding.clone();
        
        // Expert embeddings (could be learned or pre-defined)
        let expert_embeddings = Array2::zeros((self.config.num_experts, self.config.embedding_dim));
        
        // Compute attention scores
        let scores = self.attention.compute_scores(&query, &expert_embeddings)?;
        
        Ok(scores)
    }

    /// Select expert based on scores and capacity constraints
    fn select_expert(&self, scores: &Array1<f64>) -> LiquidResult<usize> {
        let max_capacity = (self.config.routing_capacity * scores.len() as f64) as usize;
        
        // Find available expert with highest score
        let mut selected = None;
        let mut max_score = f64::NEG_INFINITY;
        
        for (idx, &score) in scores.iter().enumerate() {
            if self.expert_capacities[idx].get() < max_capacity && score > max_score {
                max_score = score;
                selected = Some(idx);
            }
        }
        
        match selected {
            Some(idx) => {
                self.expert_capacities[idx].increment();
                Ok(idx)
            },
            None => {
                warn!("All experts at capacity, selecting least loaded expert");
                let idx = self.expert_capacities
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, counter)| counter.get())
                    .map(|(idx, _)| idx)
                    .unwrap();
                self.expert_capacities[idx].increment();
                Ok(idx)
            }
        }
    }

    /// Release expert capacity after processing
    pub fn release_expert(&self, expert_idx: usize) {
        self.expert_capacities[expert_idx].decrement();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LiquidConfig;

    #[test]
    fn test_router_initialization() {
        let config = RouterConfig {
            attention_heads: 4,
            embedding_dim: 256,
            num_experts: 8,
            routing_capacity: 0.3,
        };
        let modality_handler = Arc::new(ModalityHandler::new(&LiquidConfig::default()));
        let router = ExpertRouter::new(config.clone(), modality_handler);
        
        assert_eq!(router.expert_capacities.len(), config.num_experts);
    }

    #[test]
    fn test_expert_selection() {
        let config = RouterConfig {
            attention_heads: 4,
            embedding_dim: 256,
            num_experts: 4,
            routing_capacity: 0.5,
        };
        let modality_handler = Arc::new(ModalityHandler::new(&LiquidConfig::default()));
        let router = ExpertRouter::new(config, modality_handler);
        
        let input = InputModality::Text("test input".to_string());
        let (expert_idx, _) = router.route(&input).unwrap();
        
        assert!(expert_idx < 4);
        
        // Test capacity tracking
        assert_eq!(router.expert_capacities[expert_idx].get(), 1);
        router.release_expert(expert_idx);
        assert_eq!(router.expert_capacities[expert_idx].get(), 0);
    }
} 