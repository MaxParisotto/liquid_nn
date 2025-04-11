use ndarray::Array1;
use crate::Result;

/// Numerical solvers for liquid neural networks
pub trait Solver {
    fn step(&mut self, state: &mut Array1<f64>, dt: f64) -> Result<()>;
}

/// Simple Euler solver
pub struct EulerSolver;

impl Solver for EulerSolver {
    fn step(&mut self, state: &mut Array1<f64>, dt: f64) -> Result<()> {
        *state *= dt;
        Ok(())
    }
} 