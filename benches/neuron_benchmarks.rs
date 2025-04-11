use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use liquid_nn::neuron::Neuron;
use ndarray::{Array1, Array2};
use rand::random;

fn bench_neuron_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_forward");
    group.sample_size(100);
    
    let input_sizes = [64, 128, 256, 512];
    let hidden_sizes = [128, 256, 512, 1024];
    
    for &input_size in &input_sizes {
        for &hidden_size in &hidden_sizes {
            // Create weight matrix and input weights
            let weight_matrix = Array2::from_shape_fn((hidden_size, hidden_size), |_| random::<f64>() * 0.1);
            let input_weights = Array1::from_shape_fn(input_size, |_| random::<f64>() * 0.1);
            
            let mut neuron = Neuron::new(weight_matrix, input_weights);
            
            // Create input
            let input = Array1::linspace(0., 1., input_size);
            
            group.bench_with_input(
                BenchmarkId::new("size", format!("{}x{}", input_size, hidden_size)),
                &(input_size, hidden_size),
                |b, _| {
                    b.iter(|| {
                        // Use compute_derivative and step
                        let derivative = black_box(neuron.compute_derivative(black_box(input[0])));
                        black_box(neuron.step(input[0], derivative));
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_neuron_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_batch");
    group.sample_size(50);
    
    let batch_sizes = [32, 64, 128, 256];
    let input_size = 256;
    let hidden_size = 512;
    
    // Create weight matrix and input weights
    let weight_matrix = Array2::from_shape_fn((hidden_size, hidden_size), |_| random::<f64>() * 0.1);
    let input_weights = Array1::from_shape_fn(input_size, |_| random::<f64>() * 0.1);
    
    let mut neuron = Neuron::new(weight_matrix, input_weights);
    
    for &batch_size in &batch_sizes {
        let batch_input = Array2::from_shape_fn((batch_size, input_size), |(i, j)| {
            (i * input_size + j) as f64 / (batch_size * input_size) as f64
        });
        
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for i in 0..batch_size {
                        let input = batch_input.row(i).to_owned();
                        // Use compute_derivative and step
                        let derivative = black_box(neuron.compute_derivative(black_box(input[0])));
                        black_box(neuron.step(input[0], derivative));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");
    group.sample_size(200);
    
    let input_size = 256;
    let hidden_size = 512;
    
    // Create weight matrix and input weights for different neuron configurations
    let weight_matrix = Array2::from_shape_fn((hidden_size, hidden_size), |_| random::<f64>() * 0.1);
    let input_weights = Array1::from_shape_fn(input_size, |_| random::<f64>() * 0.1);
    
    let mut neuron = Neuron::new(weight_matrix, input_weights);
    
    let input = Array1::from_vec((0..input_size).map(|i| -2.0 + 4.0 * (i as f64 / input_size as f64)).collect());
    
    group.bench_with_input(
        BenchmarkId::new("activation", "tanh"),
        &input_size,
        |b, _| {
            b.iter(|| {
                // Use compute_derivative and step
                let derivative = black_box(neuron.compute_derivative(black_box(input[0])));
                black_box(neuron.step(input[0], derivative));
            });
        },
    );
    
    group.finish();
}

criterion_group!(
    benches,
    bench_neuron_forward,
    bench_neuron_batch,
    bench_activation_functions
);
criterion_main!(benches); 