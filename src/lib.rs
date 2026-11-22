//! A reproducible, CPU-first research environment for adaptive sensor-network control.
pub mod agents;
pub mod config;
pub mod environment;
pub mod experiments;
pub mod mdp;
pub mod planning;
pub mod visualization;

pub use config::{EnvConfig, TrainingConfig};
pub use environment::{Action, NetworkEnvironment, Observation};
