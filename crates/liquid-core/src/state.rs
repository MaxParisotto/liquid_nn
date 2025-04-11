/// RecurrentState encapsulates all feedback/recurrent values for the LNN.
/// This struct is passed explicitly between timesteps and is thread-safe.
#[derive(Clone, Debug, PartialEq)]
pub struct RecurrentState {
    /// The recurrent values (e.g., neuron activations from previous timestep).
    pub values: Vec<f32>,
    /// The last output of the network, for feedback (if enabled).
    pub last_output: Option<Vec<f32>>,
}

impl RecurrentState {
    /// Create a new RecurrentState with the given values.
    pub fn new(values: Vec<f32>) -> Self {
        Self { values, last_output: None }
    }

    /// Create a new RecurrentState with values and last_output.
    pub fn with_output(values: Vec<f32>, last_output: Option<Vec<f32>>) -> Self {
        Self { values, last_output }
    }

    /// Create a zero-initialized RecurrentState of given size.
    pub fn zeros(size: usize) -> Self {
        Self {
            values: vec![0.0; size],
            last_output: None,
        }
    }
}