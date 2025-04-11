use super::state::RecurrentState;

/// Describes the network topology, including recurrent (feedback) connections.
/// Enum for feedback mode in the network.

#[derive()]
#[derive(Default)]
pub enum FeedbackMode {
    #[default]
    None, // No feedback (default)
    Direct, // Output is routed directly as part of next input
    Transform(Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>), // Output is transformed before feedback
}


impl Clone for FeedbackMode {
    fn clone(&self) -> Self {
        match self {
            FeedbackMode::None => FeedbackMode::None,
            FeedbackMode::Direct => FeedbackMode::Direct,
            FeedbackMode::Transform(_) => panic!("Cannot clone FeedbackMode::Transform"),
        }
    }
}

impl std::fmt::Debug for FeedbackMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedbackMode::None => write!(f, "None"),
            FeedbackMode::Direct => write!(f, "Direct"),
            FeedbackMode::Transform(_) => write!(f, "Transform(<function>)"),
        }
    }
}

// Note: Serialize/Deserialize are not implemented for FeedbackMode due to the closure in Transform.

/// Configuration struct for topology/layer mode and feedback, usable from CLI/config.
#[derive(Clone, Debug)]
pub struct LnnModeConfig {
    pub mode: crate::liquid_layer::LayerMode,
    pub feedback_mode: FeedbackMode,
    pub feedback_policy: Option<crate::scheduler::FeedbackPolicy>,
    pub feedback_threshold: Option<f32>,
    pub autotrain_threshold: Option<f32>,
    pub exploration_rate: Option<f32>,
}

impl Default for LnnModeConfig {
    fn default() -> Self {
        Self {
            mode: crate::liquid_layer::LayerMode::Feedforward,
            feedback_mode: FeedbackMode::None,
            feedback_policy: None,
            feedback_threshold: None,
            autotrain_threshold: None,
            exploration_rate: None,
        }
    }
}

pub struct Topology {
    /// Number of input nodes.
    pub input_size: usize,
    /// Number of output nodes.
    pub output_size: usize,
    /// Number of recurrent state nodes.
    pub recurrent_size: usize,
    /// Feedforward connection weights (input to output).
    pub weights: Vec<Vec<f32>>,
    /// Recurrent connection weights (state to input or hidden).
    pub recurrent_weights: Vec<Vec<f32>>,
    /// Feedback mode for this topology.
    pub feedback_mode: FeedbackMode,
}

impl Topology {
    /// Create a new Topology with given sizes, weights, and config.
    pub fn new_with_config(
        input_size: usize,
        output_size: usize,
        recurrent_size: usize,
        weights: Vec<Vec<f32>>,
        recurrent_weights: Vec<Vec<f32>>,
        config: &LnnModeConfig,
    ) -> Self {
        Self {
            input_size,
            output_size,
            recurrent_size,
            weights,
            recurrent_weights,
            feedback_mode: config.feedback_mode.clone(),
        }
    }

    /// Legacy constructor for Topology.
    pub fn new(
        input_size: usize,
        output_size: usize,
        recurrent_size: usize,
        weights: Vec<Vec<f32>>,
        recurrent_weights: Vec<Vec<f32>>,
        feedback_mode: FeedbackMode,
    ) -> Self {
        Self {
            input_size,
            output_size,
            recurrent_size,
            weights,
            recurrent_weights,
            feedback_mode,
        }
    }

    /// Flexible forward: supports both feedforward and recurrent, with explicit state and feedback.
    pub fn forward_flexible(
        &self,
        input: &[f32],
        prev_state: &RecurrentState,
        feedback: Option<&[f32]>,
        mode: crate::liquid_layer::LayerMode,
    ) -> (Vec<f32>, RecurrentState) {
        // Prepare input with feedback if enabled
        let mut modified_input = input.to_vec();
        match &self.feedback_mode {
            FeedbackMode::None => { /* do nothing */ }
            FeedbackMode::Direct => {
                if let Some(fb) = feedback {
                    modified_input.extend(fb.iter());
                } else if let Some(ref fb) = prev_state.last_output {
                    modified_input.extend(fb.iter());
                }
            }
            FeedbackMode::Transform(f) => {
                if let Some(fb) = feedback {
                    let transformed = f(fb);
                    modified_input.extend(transformed.iter());
                } else if let Some(ref fb) = prev_state.last_output {
                    let transformed = f(fb);
                    modified_input.extend(transformed.iter());
                }
            }
        }
        // Simple matrix-vector multiplication for demonstration.
        let output: Vec<f32> = self.weights
            .iter()
            .map(|row| row.iter().zip(&modified_input).map(|(w, x)| w * x).sum())
            .collect();

        // Update recurrent state if in recurrent mode
        let new_state = if mode == crate::liquid_layer::LayerMode::Recurrent {
            RecurrentState::with_output(output.clone(), Some(output.clone()))
        } else {
            RecurrentState::with_output(vec![], Some(output.clone()))
        };

        (output, new_state)
    }

    /// Online learning step for the topology (autotrain).
    pub fn update_weights(
        &mut self,
        input: &[f32],
        target: &[f32],
        prev_state: &RecurrentState,
        feedback: Option<&[f32]>,
        learning_rate: f32,
        mode: crate::liquid_layer::LayerMode,
    ) -> (Vec<f32>, RecurrentState) {
        // Forward pass
        let (output, new_state) = self.forward_flexible(input, prev_state, feedback, mode);

        // Compute error (simple MSE for demonstration)
        let error: Vec<f32> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| t - o)
            .collect();

        // Update weights (simple gradient step for demonstration)
        for (row, e) in self.weights.iter_mut().zip(error.iter()) {
            for w in row.iter_mut() {
                *w += learning_rate * e;
            }
        }

        (output, new_state)
    }

    /// Feedforward only: propagate input through the network.
    pub fn forward_feedforward(&self, input: &[f32], prev_state: &RecurrentState) -> Vec<f32> {
        // Prepare input with feedback if enabled
        let mut modified_input = input.to_vec();
        match &self.feedback_mode {
            FeedbackMode::None => { /* do nothing */ }
            FeedbackMode::Direct => {
                if let Some(ref feedback) = prev_state.last_output {
                    modified_input.extend(feedback.iter());
                }
            }
            FeedbackMode::Transform(f) => {
                if let Some(ref feedback) = prev_state.last_output {
                    let transformed = f(feedback);
                    modified_input.extend(transformed.iter());
                }
            }
        }
        // Simple matrix-vector multiplication for demonstration.
        self.weights
            .iter()
            .map(|row| row.iter().zip(&modified_input).map(|(w, x)| w * x).sum())
            .collect()
    }

    /// Recurrent: propagate input and previous state, return output and new state.
    pub fn forward(&self, input: &[f32], state: &RecurrentState) -> (Vec<f32>, RecurrentState) {
        // Prepare input with feedback if enabled
        let mut combined_input = input.to_vec();
        match &self.feedback_mode {
            FeedbackMode::None => { /* do nothing */ }
            FeedbackMode::Direct => {
                if let Some(ref feedback) = state.last_output {
                    combined_input.extend(feedback.iter());
                }
            }
            FeedbackMode::Transform(f) => {
                if let Some(ref feedback) = state.last_output {
                    let transformed = f(feedback);
                    combined_input.extend(transformed.iter());
                }
            }
        }
        // Also combine recurrent state values
        if !state.values.is_empty() {
            combined_input.extend(&state.values);
        }

        // Compute output using both feedforward and recurrent weights.
        let output: Vec<f32> = self
            .weights
            .iter()
            .map(|row| row.iter().zip(&combined_input).map(|(w, x)| w * x).sum())
            .collect();

        // Update recurrent state, store output for feedback
        let new_state = RecurrentState {
            values: output.clone(),
            last_output: Some(output.clone()),
        };

        (output, new_state)
    }
}

impl Topology {
    pub fn step_with_scheduler_decision(
        &mut self,
        input: &[f32],
        prev_state: &crate::state::RecurrentState,
        decision: crate::scheduler::SchedulerDecision,
        mode: crate::liquid_layer::LayerMode,
    ) -> (Vec<f32>, crate::state::RecurrentState) {
        match decision {
            crate::scheduler::SchedulerDecision::Train => {
                // For demonstration, just call forward_flexible and then update_weights.
                let (output, new_state) = self.forward_flexible(input, prev_state, None, mode);
                // In a real implementation, you would update weights here.
                // self.update_weights(...);
                (output, new_state)
            }
            crate::scheduler::SchedulerDecision::Feedback => {
                // Call forward_flexible with feedback enabled.
                let (output, new_state) = self.forward_flexible(input, prev_state, prev_state.last_output.as_deref(), mode);
                (output, new_state)
            }
            crate::scheduler::SchedulerDecision::Explore => {
                // Exploration: could add noise or randomize input/actions.
                let mut noisy_input = input.to_vec();
                for x in &mut noisy_input {
                    *x += rand::random::<f32>() * 0.01; // Small random noise
                }
                let (output, new_state) = self.forward_flexible(&noisy_input, prev_state, None, mode);
                (output, new_state)
            }
            crate::scheduler::SchedulerDecision::Exploit => {
                // Exploitation: normal forward.
                self.forward_flexible(input, prev_state, None, mode)
            }
            crate::scheduler::SchedulerDecision::None => {
                // No action: return previous state/output.
                (input.to_vec(), prev_state.clone())
            }
        }
    }
}