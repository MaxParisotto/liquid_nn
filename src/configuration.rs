use std::path::PathBuf;
use crate::InputModality;

/// Iggy message broker configuration
#[derive(Debug, Clone)]
pub struct IggyConfig {
    pub host: String,
    pub port: u16,
    pub topic: String,
}

/// Storage configuration for model parameters
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub ram_cache_size: usize,      // Size of RAM cache in bytes
    pub nvme_path: PathBuf,         // Path to NVMe storage
    pub qdrant_url: String,         // URL for Qdrant vector database
    pub iggy_config: IggyConfig,    // Configuration for Iggy message broker
}

/// Configuration for the Liquid Neural Network
#[derive(Debug, Clone)]
pub struct LiquidConfig {
    // MoE configuration
    pub num_experts: usize,
    pub expert_size: usize,
    pub num_layers: usize,
    
    // Attention configuration
    pub attention_heads: usize,
    pub embedding_dim: usize,
    pub dropout: f32,
    
    // Input/Output configuration
    pub modalities: Vec<InputModality>,
    pub max_sequence_length: usize,
    
    // Hardware configuration
    pub num_gpus: usize,
    pub gpu_memory_limit: usize,
    pub num_cpu_threads: usize,
    
    // Storage configuration
    pub storage: StorageConfig,
    
    // Training configuration
    pub batch_size: usize,
    pub learning_rate: f32,
    pub gradient_accumulation_steps: usize,
    pub mixed_precision: bool,
}

impl Default for LiquidConfig {
    fn default() -> Self {
        Self {
            // MoE configuration
            num_experts: 8,
            expert_size: 512,
            num_layers: 12,
            
            // Attention configuration
            attention_heads: 8,
            embedding_dim: 768,
            dropout: 0.1,
            
            // Input/Output configuration
            modalities: vec![InputModality::Text(String::new())],
            max_sequence_length: 2048,
            
            // Hardware configuration
            num_gpus: 1,
            gpu_memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
            num_cpu_threads: num_cpus::get(),
            
            // Storage configuration
            storage: StorageConfig {
                ram_cache_size: 32 * 1024 * 1024 * 1024, // 32GB
                nvme_path: PathBuf::from("/mnt/nvme/liquid_nn"),
                qdrant_url: "http://localhost:6333".to_string(),
                iggy_config: IggyConfig {
                    host: "localhost".to_string(),
                    port: 8090,
                    topic: "liquid_nn".to_string(),
                },
            },
            
            // Training configuration
            batch_size: 32,
            learning_rate: 1e-4,
            gradient_accumulation_steps: 8,
            mixed_precision: true,
        }
    }
} 