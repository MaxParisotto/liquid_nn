mod neuron;
mod topology;
mod solvers;

use ndarray::Array1;
use topology::Topology;

fn main() {
    // Example weight and input vectors
    let weight_matrix = ndarray::Array2::ones((5, 5));
    let input_weight = ndarray::Array1::ones(5);

    // Create layer sizes for the network (e.g., 3 layers of 100 neurons each)
    let layer_sizes = vec![100, 100, 100];

    // Create the network topology
    let mut network = Topology::new(layer_sizes, weight_matrix, input_weight);

    // Example input (size must match the first layer)
    let input = Array1::from(vec![1.0; 100]);

    // Run a single forward pass
    network.forward(input);

    // Print performance statistics
    let (avg_time, max_state, min_state) = network.get_performance_stats();
    println!("LNN forward pass completed.");
    println!("Average forward pass time: {:.6}s", avg_time);
    println!("State range: [{:.6}, {:.6}]", min_state, max_state);
}