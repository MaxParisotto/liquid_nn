use ndarray::Array1;

pub struct Neuron {
    pub state: Array1<f64>,
    weight_matrix: ndarray::Array2<f64>,
    input_weight: Array1<f64>,
}

impl Neuron {
    pub fn new(weight_matrix: ndarray::Array2<f64>, input_weight: Array1<f64>) -> Self {
        Neuron {
            state: Array1::zeros(weight_matrix.nrows()),
            weight_matrix,
            input_weight,
        }
    }

    /// Compute the derivative of the neuron's state
    pub fn compute_derivative(&self, input: f64) -> f64 {
        let recurrent_contribution = self.weight_matrix.dot(&self.state);
        let input_contribution = &self.input_weight * input;
        (recurrent_contribution + input_contribution).sum()
    }

    /// Get the current state of the neuron (first component for simplicity)
    pub fn state(&self) -> f64 {
        self.state[0]
    }

    pub fn step(&mut self, input: f64, dt: f64) {
        // Update using provided derivative
        let derivative = self.compute_derivative(input);
        self.state += &Array1::from_elem(self.state.len(), derivative * dt);
    }
}