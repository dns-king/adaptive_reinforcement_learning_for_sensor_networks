//! Compact MDP abstractions and POMDP belief state utilities.
use serde::{Deserialize, Serialize};

/// Finite MDP representation: P[s][a] is a sparse probability distribution over next states.
#[derive(Clone, Debug)]
pub struct FiniteMdp {
    pub states: usize,
    pub actions: usize,
    pub transitions: Vec<Vec<Vec<(usize, f64)>>>,
    pub rewards: Vec<Vec<f64>>,
    pub gamma: f64,
}
impl FiniteMdp {
    pub fn validate(&self) -> Result<(), String> {
        for row in &self.transitions {
            for p in row {
                let total: f64 = p.iter().map(|x| x.1).sum();
                if (total - 1.0).abs() > 1e-9 {
                    return Err("each transition distribution must sum to one".into());
                }
            }
        }
        Ok(())
    }
}

/// Bayesian belief over a binary link state. `observe(true)` is an ACK; false is a timeout.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LinkBelief {
    pub up_probability: f64,
}
impl LinkBelief {
    pub fn predict(&mut self, p_drop: f64, p_recover: f64) {
        self.up_probability =
            self.up_probability * (1.0 - p_drop) + (1.0 - self.up_probability) * p_recover;
    }
    pub fn observe(&mut self, ack: bool, sensitivity: f64, false_ack: f64) {
        let prior = self.up_probability;
        let l_up = if ack { sensitivity } else { 1.0 - sensitivity };
        let l_down = if ack { false_ack } else { 1.0 - false_ack };
        self.up_probability =
            (prior * l_up / (prior * l_up + (1.0 - prior) * l_down)).clamp(0.0, 1.0);
    }
}
