use std::sync::{Arc, Mutex};

/// Performance metrics for scheduler decision-making.
#[derive(Debug, Clone, Default)]
pub struct LnnMetrics {
    pub loss: Option<f32>,
    pub accuracy: Option<f32>,
    pub reward: Option<f32>,
    pub novelty: Option<f32>,
    pub history: Vec<f32>,
}

/// State snapshot for scheduler (can be extended as needed).
#[derive(Debug, Clone, Default)]
pub struct LnnState {
    pub timestep: usize,
    pub last_decision: Option<SchedulerDecision>,
    pub last_input: Option<Vec<f32>>,
}

/// Enum for scheduler decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerDecision {
    Train,
    Feedback,
    Explore,
    Exploit,
    None,
}

/// Configurable, thread-safe adaptive scheduler/controller for LNN.
pub struct LnnScheduler {
    pub config: Arc<Mutex<SchedulerConfig>>,
    pub metrics: Arc<Mutex<LnnMetrics>>,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub feedback_threshold: f32,
    pub autotrain_threshold: f32,
    pub exploration_rate: f32, // e.g., epsilon for epsilon-greedy
    pub feedback_policy: FeedbackPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FeedbackPolicy {
    Always,
    OnNovelty,
    OnPerformanceDrop,
    Never,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            feedback_threshold: 0.1,
            autotrain_threshold: 0.2,
            exploration_rate: 0.1,
            feedback_policy: FeedbackPolicy::OnPerformanceDrop,
        }
    }
}

impl LnnScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config: Arc::new(Mutex::new(config)),
            metrics: Arc::new(Mutex::new(LnnMetrics::default())),
        }
    }

    /// Main decision function: determines what action to take.
    pub fn decide(
        &self,
        metrics: &LnnMetrics,
        _state: &LnnState,
        _input: &[f32],
        user_config: &crate::topology::LnnModeConfig,
    ) -> SchedulerDecision {
        let config = self.config.lock().unwrap().clone();

        // 1. Check for explicit user triggers
        if let Some(policy) = user_config.feedback_policy {
            match policy {
                FeedbackPolicy::Always => return SchedulerDecision::Feedback,
                FeedbackPolicy::Never => return SchedulerDecision::None,
                _ => {}
            }
        }

        // 2. Performance-based triggers
        if let Some(loss) = metrics.loss {
            if loss > config.autotrain_threshold {
                return SchedulerDecision::Train;
            }
            if loss > config.feedback_threshold && config.feedback_policy == FeedbackPolicy::OnPerformanceDrop {
                return SchedulerDecision::Feedback;
            }
        }

        // 3. Novelty-based triggers
        if let Some(novelty) = metrics.novelty {
            if config.feedback_policy == FeedbackPolicy::OnNovelty && novelty > config.feedback_threshold {
                return SchedulerDecision::Feedback;
            }
        }

        // 4. Exploration vs. exploitation (e.g., epsilon-greedy)
        let explore = rand::random::<f32>() < config.exploration_rate;
        if explore {
            SchedulerDecision::Explore
        } else {
            SchedulerDecision::Exploit
        }
    }

    /// Update metrics after feedback/training.
    pub fn update_metrics(&self, new_metrics: LnnMetrics) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = new_metrics;
    }
}