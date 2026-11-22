use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvConfig {
    pub nodes: usize,
    pub horizon: usize,
    pub initial_energy: f64,
    pub failure_probability: f64,
    pub link_drop_probability: f64,
    pub degradation_rate: f64,
    pub environment_volatility: f64,
    pub discount: f64,
    pub quality_weight: f64,
    pub energy_weight: f64,
    pub coverage_weight: f64,
    pub constraint_penalty: f64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            nodes: 16,
            horizon: 120,
            initial_energy: 100.0,
            failure_probability: 0.002,
            link_drop_probability: 0.12,
            degradation_rate: 0.0015,
            environment_volatility: 0.08,
            discount: 0.97,
            quality_weight: 1.0,
            energy_weight: 0.07,
            coverage_weight: 2.0,
            constraint_penalty: 4.0,
        }
    }
}

impl EnvConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(8..=32).contains(&self.nodes) {
            return Err("nodes must be in 8..=32".into());
        }
        if self.horizon == 0 || self.initial_energy <= 0.0 {
            return Err("horizon and initial_energy must be positive".into());
        }
        for p in [
            self.failure_probability,
            self.link_drop_probability,
            self.degradation_rate,
        ] {
            if !(0.0..=1.0).contains(&p) {
                return Err("probabilities must be in [0,1]".into());
            }
        }
        if !(0.0..1.0).contains(&self.discount) {
            return Err("discount must be in [0,1)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub episodes: usize,
    pub alpha: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub seed: u64,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            episodes: 400,
            alpha: 0.12,
            epsilon_start: 0.25,
            epsilon_end: 0.02,
            epsilon_decay: 0.992,
            seed: 7,
        }
    }
}
