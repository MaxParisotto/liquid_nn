use ndarray::{Array1, Array2};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Performance tracking for model operations
pub struct PerformanceTracker {
    operation_count: AtomicUsize,
    total_time: AtomicUsize, // in microseconds
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            operation_count: AtomicUsize::new(0),
            total_time: AtomicUsize::new(0),
        }
    }

    pub fn record_operation(&self, duration_micros: u64) {
        self.operation_count.fetch_add(1, Ordering::Relaxed);
        self.total_time.fetch_add(duration_micros as usize, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> (usize, f64) {
        let count = self.operation_count.load(Ordering::Relaxed);
        let total_time = self.total_time.load(Ordering::Relaxed) as f64 / 1_000_000.0; // Convert to seconds
        (count, total_time)
    }
}

/// Numerical stability utilities
pub mod numerical {
    use super::*;

    pub fn stable_softmax(x: &Array1<f64>) -> Array1<f64> {
        let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp = x.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        exp.mapv(|v| v / sum)
    }

    pub fn log_sum_exp(x: &Array1<f64>) -> f64 {
        let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum = x.mapv(|v| (v - max).exp()).sum();
        max + sum.ln()
    }
}

/// Gradient clipping and normalization
pub mod gradients {
    use super::*;
    use std::f64;

    pub fn clip_by_norm(grad: &mut Array1<f64>, max_norm: f64) {
        let norm = grad.mapv(|x| x * x).sum().sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            grad.mapv_inplace(|x| x * scale);
        }
    }

    pub fn clip_by_value(grad: &mut Array1<f64>, min_val: f64, max_val: f64) {
        grad.mapv_inplace(|x| x.max(min_val).min(max_val));
    }
}

/// Initialization utilities
pub mod init {
    use super::*;
    use rand::Rng;
    use rand::thread_rng;

    pub fn xavier_init(shape: (usize, usize)) -> Array2<f64> {
        let std = (2.0 / (shape.0 + shape.1) as f64).sqrt();
        let mut rng = thread_rng();
        
        Array2::from_shape_fn(shape, |_| {
            // Box-Muller transform to generate normal distribution
            let u1 = rng.gen::<f64>();
            let u2 = rng.gen::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            z * std
        })
    }

    pub fn zeros_init(shape: (usize, usize)) -> Array2<f64> {
        Array2::zeros(shape)
    }
}

/// Logging utilities
pub mod logging {
    use std::fs::OpenOptions;
    use std::io::Write;
    use chrono::Local;

    pub fn log_event(message: &str, log_file: &str) -> std::io::Result<()> {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();
        let log_message = format!("[{}] {}\n", timestamp, message);
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file)?;
            
        file.write_all(log_message.as_bytes())?;
        Ok(())
    }
} 