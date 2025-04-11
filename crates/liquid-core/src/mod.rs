//! Public interface for the Liquid Neural Network (LNN) core, supporting both feedforward and recurrent operation.

pub mod state;
pub mod topology;
pub mod liquid_layer;

pub use state::RecurrentState;
pub use topology::Topology;
pub use liquid_layer::{LiquidLayer, LayerMode};