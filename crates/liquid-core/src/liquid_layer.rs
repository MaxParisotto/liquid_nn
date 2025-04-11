use crate::state::RecurrentState;
use crate::topology::FeedbackMode;

/// Operation mode for the LiquidLayer.
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LayerMode {
    #[default]
    Feedforward,
    Recurrent,
}

/// Main layer logic for the LNN, supporting both feedforward and recurrent operation.
/// Configuration struct for layer mode and feedback, usable from CLI/config.
#[derive(Clone, Debug)]
pub struct LnnModeConfig {
    pub mode: LayerMode,
    pub feedback_mode: FeedbackMode,
}

impl Default for LnnModeConfig {
    fn default() -> Self {
        Self {
            mode: LayerMode::Feedforward,
            feedback_mode: FeedbackMode::None,
        }
    }
}

pub struct LiquidLayer {
    pub mode: LayerMode,
    pub feedback_mode: FeedbackMode,
    pub state: RecurrentState,
    pub input_size: usize,
    pub output_size: usize,
    /// Feedforward weights (input to output).
    pub feedforward_weights: Vec<f32>,
    /// Recurrent weights (state to output), only used in recurrent mode.
    pub recurrent_weights: Vec<f32>,
}

impl LiquidLayer {
    /// Create a new LiquidLayer.
    /// Create a new LiquidLayer from config.
    pub fn new_with_config(input_size: usize, output_size: usize, config: &LnnModeConfig) -> Self {
        Self {
            mode: config.mode,
            feedback_mode: config.feedback_mode.clone(),
            state: RecurrentState::zeros(output_size),
            input_size,
            output_size,
            feedforward_weights: vec![0.5; input_size], // simple init
            recurrent_weights: vec![0.5; output_size], // simple init
        }
    }

    /// Create a new LiquidLayer (legacy, direct mode/feedback).
    pub fn new(input_size: usize, output_size: usize, mode: LayerMode, feedback_mode: FeedbackMode) -> Self {
        Self {
            mode,
            feedback_mode,
            state: RecurrentState::zeros(output_size),
            input_size,
            output_size,
            feedforward_weights: vec![0.5; input_size], // simple init
            recurrent_weights: vec![0.5; output_size], // simple init
        }
    }

    /// Step the layer for one timestep, using input, previous state, and optional feedback.
    /// Online learning step for the layer (autotrain).
    pub fn update_weights(
        &mut self,
        input: &[f32],
        target: &[f32],
        prev_state: &RecurrentState,
        feedback: Option<&[f32]>,
        learning_rate: f32,
    ) -> (Vec<f32>, RecurrentState) {
        // Forward pass
        let (output, new_state) = self.step(input, prev_state, feedback);

        // Compute error (simple MSE for demonstration)
        let error: Vec<f32> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| t - o)
            .collect();

        // Update weights (simple gradient step for demonstration)
        for (w, e) in self.feedforward_weights.iter_mut().zip(error.iter()) {
            *w += learning_rate * e;
        }

        (output, new_state)
    }

    /// Step the layer for one timestep, using input and previous state.
    /// Step the layer for one timestep, using input, previous state, and optional feedback.
    pub fn step(
        &mut self,
        input: &[f32],
        prev_state: &RecurrentState,
        feedback: Option<&[f32]>,
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

        let output = match self.mode {
            LayerMode::Feedforward => {
                // Feedforward: ignore prev_state except for feedback.
                self.feedforward(&modified_input)
            }
            LayerMode::Recurrent => {
                // Recurrent: use prev_state and feedback.
                self.recurrent(&modified_input, prev_state)
            }
        };

        // Store output in new state for feedback in next step
        (
            output.clone(),
            RecurrentState {
                values: output.clone(),
                last_output: Some(output),
            },
        )
    }

    /// Feedforward computation (no state).
    fn feedforward(&self, input: &[f32]) -> Vec<f32> {
        // Placeholder: identity mapping.
        input.to_vec()
    }

    /// Recurrent computation (uses previous state).
    fn recurrent(&self, input: &[f32], prev_state: &RecurrentState) -> Vec<f32> {
        // Placeholder: sum input and previous state elementwise.
        input
            .iter()
            .zip(&prev_state.values)
            .map(|(x, s)| x + s)
            .collect()
    }

    /// Reset the internal state to zeros.
    pub fn reset_state(&mut self) {
        self.state = RecurrentState::zeros(self.output_size);
    }

    /// Set the operation mode (feedforward or recurrent).
    pub fn set_mode(&mut self, mode: LayerMode) {
        self.mode = mode;
    }
}