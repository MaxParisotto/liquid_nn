use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use liquid_nn::neuron::{Neuron, NeuronConfig};
use liquid_nn::{Forward, Initialize, ActivationType};
use ndarray::{Array1, Array2};

fn bench_neuron_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_forward");
    group.sample_size(100);
    
    let input_sizes = [64, 128, 256, 512];
    let hidden_sizes = [128, 256, 512, 1024];
    
    for &input_size in &input_sizes {
        for &hidden_size in &hidden_sizes {
            let config = NeuronConfig {
                input_dim: input_size,
                hidden_dim: hidden_size,
                activation: ActivationType::Tanh,
                use_bias: true,
            };
            
            let mut neuron = Neuron::new(config);
            neuron.initialize().unwrap();
            
            let input = Array1::from_vec((0..input_size).map(|i| i as f64 / input_size as f64).collect());
            
            group.bench_with_input(
                BenchmarkId::new("size", format!("{}x{}", input_size, hidden_size)),
                &(input_size, hidden_size),
                |b, _| {
                    b.iter(|| {
                        black_box(neuron.forward(black_box(&input))).unwrap();
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
    
    let config = NeuronConfig {
        input_dim: input_size,
        hidden_dim: hidden_size,
        activation: ActivationType::Tanh,
        use_bias: true,
    };
    
    let mut neuron = Neuron::new(config);
    neuron.initialize().unwrap();
    
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
                        black_box(neuron.forward(black_box(&batch_input.row(i).to_owned()))).unwrap();
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
    let activations = [
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::Sigmoid,
        ActivationType::Linear,
    ];
    
    for &activation in &activations {
        let config = NeuronConfig {
            input_dim: input_size,
            hidden_dim: hidden_size,
            activation,
            use_bias: true,
        };
        
        let mut neuron = Neuron::new(config);
        neuron.initialize().unwrap();
        
        let input = Array1::from_vec((0..input_size).map(|i| -2.0 + 4.0 * (i as f64 / input_size as f64)).collect());
        
        group.bench_with_input(
            BenchmarkId::new("activation", format!("{:?}", activation)),
            &activation,
            |b, _| {
                b.iter(|| {
                    black_box(neuron.forward(black_box(&input))).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_neuron_forward,
    bench_neuron_batch,
    bench_activation_functions
);
criterion_main!(benches); 