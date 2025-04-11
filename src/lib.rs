#![recursion_limit = "256"]

pub mod neuron;
pub mod topology;
pub mod solvers;
pub mod moe;
pub mod attention;
pub mod modalities;
pub mod utils;
pub mod router;
pub mod error;
pub mod types;
pub mod configuration;

use ndarray::{Array2, Array3};
use std::error::Error;
use tracing::{debug, trace};

pub use configuration::LiquidConfig;
pub type LiquidResult<T> = std::result::Result<T, Box<dyn Error>>;

/// Supported input modalities
#[derive(Debug, Clone)]
pub enum InputModality {
    Text(String),
    Audio(Vec<f32>),
    Video(Array3<f32>),
    Image(Array2<f32>),
}

/// Supported output modalities
#[derive(Debug, Clone)]
pub enum OutputModality {
    Text(String),
    Audio(Vec<f32>),
    Video(Array3<f32>),
    Image(Array2<f32>),
}

/// Storage configuration for model parameters
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub ram_cache_size: usize,      // Size of RAM cache in bytes
    pub nvme_path: std::path::PathBuf,         // Path to NVMe storage
    pub qdrant_url: String,         // URL for Qdrant vector database
    pub iggy_config: IggyConfig,    // Configuration for Iggy message broker
}

/// Iggy message broker configuration
#[derive(Debug, Clone)]
pub struct IggyConfig {
    pub host: String,
    pub port: u16,
    pub topic: String,
}

/// Common traits for model components
pub trait Forward {
    fn forward(&mut self, input: &InputModality) -> LiquidResult<OutputModality>;
}

pub trait Backward {
    fn backward(&mut self, grad: &OutputModality) -> LiquidResult<()>;
}

pub trait Initialize {
    fn initialize(&mut self, config: &LiquidConfig) -> LiquidResult<()>;
}

/// Main LNN model combining all components
pub struct LiquidNeuralNetwork {
    router: router::ExpertRouter,
    experts: Vec<moe::MoELayer>,
}

impl LiquidNeuralNetwork {
    pub fn new(config: LiquidConfig) -> LiquidResult<Self> {
        let modality_handler = std::sync::Arc::new(modalities::ModalityHandler::new(&config));
        
        let router_config = router::RouterConfig {
            attention_heads: config.attention_heads,
            embedding_dim: config.embedding_dim,
            num_experts: config.num_experts,
            routing_capacity: 0.3,
        };
        
        let router = router::ExpertRouter::new(router_config, modality_handler.clone());
        
        let experts = (0..config.num_experts)
            .map(|_| moe::MoELayer::new(config.clone()))
            .collect();
            
        Ok(Self {
            router,
            experts,
        })
    }
    
    pub fn process(&mut self, input: &InputModality) -> LiquidResult<OutputModality> {
        trace!("Processing input through LNN");
        
        // Route the input to the appropriate expert
        let (expert_idx, _embedding) = self.router.route(input)?;
        
        // Use the selected expert
        let expert = &mut self.experts[expert_idx];
        let output = expert.forward(input)?;
        
        debug!("LNN processing complete, output: {:?}", output);
        Ok(output)
    }
} 