use liquid_core::{Forward, Backward, Result};
use ndarray::Array1;
use std::sync::Arc;
use tracing::{info, warn};

mod optimizer;
mod scheduler;
mod metrics;
mod distributed;
mod storage;

pub use optimizer::{Optimizer, OptimizerConfig};
pub use scheduler::{LearningRateScheduler, SchedulerConfig};
pub use metrics::{Metrics, MetricsConfig};
pub use distributed::{DistributedTrainer, TrainerConfig};
pub use storage::{ModelStorage, StorageConfig};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gradient_clip: Option<f64>,
    pub mixed_precision: bool,
    pub optimizer: OptimizerConfig,
    pub scheduler: SchedulerConfig,
    pub metrics: MetricsConfig,
    pub storage: StorageConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_epochs: 100,
            learning_rate: 1e-3,
            weight_decay: 1e-5,
            gradient_clip: Some(1.0),
            mixed_precision: true,
            optimizer: OptimizerConfig::default(),
            scheduler: SchedulerConfig::default(),
            metrics: MetricsConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

/// Training state for checkpointing and resumption
#[derive(Debug)]
pub struct TrainingState {
    pub epoch: usize,
    pub step: usize,
    pub best_loss: f64,
    pub model_state: Vec<Array1<f64>>,
    pub optimizer_state: Vec<u8>,
}

/// Main trainer for model optimization
pub struct Trainer<M> 
where 
    M: Forward + Backward + Send + Sync + 'static
{
    model: Arc<M>,
    config: TrainingConfig,
    optimizer: Box<dyn Optimizer>,
    scheduler: Box<dyn LearningRateScheduler>,
    metrics: Metrics,
    storage: Box<dyn ModelStorage>,
    state: TrainingState,
}

impl<M> Trainer<M> 
where 
    M: Forward + Backward + Send + Sync + 'static
{
    pub fn new(model: M, config: TrainingConfig) -> Result<Self> {
        let model = Arc::new(model);
        let optimizer = optimizer::create_optimizer(&config.optimizer)?;
        let scheduler = scheduler::create_scheduler(&config.scheduler)?;
        let metrics = Metrics::new(config.metrics.clone())?;
        let storage = storage::create_storage(&config.storage)?;
        
        let state = TrainingState {
            epoch: 0,
            step: 0,
            best_loss: f64::INFINITY,
            model_state: Vec::new(),
            optimizer_state: Vec::new(),
        };

        Ok(Self {
            model,
            config,
            optimizer,
            scheduler,
            metrics,
            storage,
            state,
        })
    }

    pub fn train(&mut self, train_data: &[Array1<f64>], valid_data: &[Array1<f64>]) -> Result<()> {
        info!("Starting training with config: {:?}", self.config);

        for epoch in 0..self.config.num_epochs {
            self.train_epoch(train_data)?;
            let valid_loss = self.validate(valid_data)?;

            // Update learning rate
            self.scheduler.step(valid_loss);

            // Save checkpoint if best so far
            if valid_loss < self.state.best_loss {
                self.state.best_loss = valid_loss;
                self.save_checkpoint()?;
            }

            info!(
                "Epoch {}/{}: valid_loss = {:.4}, lr = {:.6}", 
                epoch + 1,
                self.config.num_epochs,
                valid_loss,
                self.scheduler.get_lr()
            );
        }

        Ok(())
    }

    fn train_epoch(&mut self, data: &[Array1<f64>]) -> Result<()> {
        // Training loop implementation
        Ok(())
    }

    fn validate(&self, data: &[Array1<f64>]) -> Result<f64> {
        // Validation loop implementation
        Ok(0.0)
    }

    fn save_checkpoint(&self) -> Result<()> {
        self.storage.save_checkpoint(&self.state)
    }

    fn load_checkpoint(&mut self) -> Result<()> {
        self.state = self.storage.load_checkpoint()?;
        Ok(())
    }
} 