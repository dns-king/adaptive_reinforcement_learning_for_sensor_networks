use crate::config::EnvConfig;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Sleep = 0,
    SenseLow = 1,
    SenseHigh = 2,
    SenseTransmit = 3,
}
impl Action {
    pub const ALL: [Action; 4] = [
        Action::Sleep,
        Action::SenseLow,
        Action::SenseHigh,
        Action::SenseTransmit,
    ];
    pub fn cost(self) -> f64 {
        match self {
            Self::Sleep => 0.02,
            Self::SenseLow => 0.35,
            Self::SenseHigh => 1.0,
            Self::SenseTransmit => 1.45,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub energy: f64,
    pub health: f64,
    pub link_up: bool,
    pub quality: f64,
    pub uncertainty: f64,
}
impl Observation {
    /// Compact five-bin local state: energy, health, link, quality, environmental uncertainty.
    pub fn key(self) -> u16 {
        let bin = |x: f64| (x.clamp(0.0, 0.999) * 4.0) as u16;
        bin(self.energy)
            | (bin(self.health) << 2)
            | ((self.link_up as u16) << 4)
            | (bin(self.quality) << 5)
            | (bin(self.uncertainty) << 7)
    }
    pub fn features(self) -> [f64; 5] {
        [
            self.energy,
            self.health,
            self.link_up as u8 as f64,
            self.quality,
            self.uncertainty,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepInfo {
    pub monitoring_quality: f64,
    pub energy_used: f64,
    pub coverage: f64,
    pub violations: usize,
    pub active_nodes: usize,
}
#[derive(Clone, Debug)]
struct Node {
    energy: f64,
    health: f64,
    link_up: bool,
    last_quality: f64,
}

/// Stochastic independent-node network. A shared latent environment affects measurement quality;
/// links, degradation and failures evolve independently conditional on that environment.
pub struct NetworkEnvironment {
    cfg: EnvConfig,
    nodes: Vec<Node>,
    rng: ChaCha8Rng,
    step: usize,
    latent: f64,
}
impl NetworkEnvironment {
    pub fn new(cfg: EnvConfig, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        let mut e = Self {
            cfg,
            nodes: vec![],
            rng: ChaCha8Rng::seed_from_u64(seed),
            step: 0,
            latent: 0.5,
        };
        e.reset();
        Ok(e)
    }
    pub fn config(&self) -> &EnvConfig {
        &self.cfg
    }
    pub fn reset(&mut self) -> Vec<Observation> {
        self.step = 0;
        self.latent = self.rng.gen_range(0.3..0.7);
        self.nodes = (0..self.cfg.nodes)
            .map(|_| Node {
                energy: self.cfg.initial_energy,
                health: self.rng.gen_range(0.85..1.0),
                link_up: true,
                last_quality: 0.5,
            })
            .collect();
        self.observations()
    }
    pub fn observations(&self) -> Vec<Observation> {
        self.nodes
            .iter()
            .map(|n| Observation {
                energy: n.energy / self.cfg.initial_energy,
                health: n.health,
                link_up: n.link_up,
                quality: n.last_quality,
                uncertainty: (self.latent - 0.5).abs() * 2.0,
            })
            .collect()
    }
    pub fn step(&mut self, actions: &[Action]) -> (Vec<Observation>, f64, bool, StepInfo) {
        assert_eq!(
            actions.len(),
            self.nodes.len(),
            "one action per sensor node"
        );
        self.step += 1;
        let disturbance = if self.cfg.environment_volatility > 0.0 {
            self.rng
                .gen_range(-self.cfg.environment_volatility..self.cfg.environment_volatility)
        } else {
            0.0
        };
        self.latent = (0.88 * self.latent + 0.12 * self.rng.gen_range(0.0..1.0) + disturbance)
            .clamp(0.0, 1.0);
        let mut quality = 0.0;
        let mut energy_used = 0.0;
        let mut covered = 0usize;
        let mut violations = 0usize;
        let mut active = 0usize;
        for (node, action) in self.nodes.iter_mut().zip(actions) {
            if node.energy <= 0.0 || node.health <= 0.02 {
                node.energy = 0.0;
                node.link_up = false;
                continue;
            }
            if self.rng.gen_bool(self.cfg.failure_probability) {
                node.health *= self.rng.gen_range(0.4..0.8);
            }
            node.health = (node.health
                - self.cfg.degradation_rate * (1.0 + (*action as u8 as f64) * 0.25))
                .max(0.0);
            node.link_up = !self
                .rng
                .gen_bool(self.cfg.link_drop_probability * (1.0 + (1.0 - node.health) * 0.5));
            let used = action.cost().min(node.energy);
            node.energy -= used;
            energy_used += used;
            active += 1;
            let sensing = match action {
                Action::Sleep => 0.0,
                Action::SenseLow => 0.45,
                Action::SenseHigh | Action::SenseTransmit => 1.0,
            };
            let delivered = if *action == Action::SenseTransmit && node.link_up {
                1.0
            } else if *action == Action::SenseTransmit {
                0.2
            } else {
                0.65
            };
            node.last_quality = (sensing
                * node.health
                * (0.55 + 0.45 * self.latent)
                * self.rng.gen_range(0.85..1.15))
            .clamp(0.0, 1.0);
            quality += node.last_quality * delivered;
            if node.last_quality > 0.2 {
                covered += 1;
            }
            if node.energy / self.cfg.initial_energy < 0.05 && *action != Action::Sleep {
                violations += 1;
            }
        }
        let coverage = covered as f64 / self.cfg.nodes as f64;
        let reward = self.cfg.quality_weight * quality / self.cfg.nodes as f64
            + self.cfg.coverage_weight * coverage
            - self.cfg.energy_weight * energy_used
            - self.cfg.constraint_penalty * violations as f64;
        let done = self.step >= self.cfg.horizon || active == 0;
        (
            self.observations(),
            reward,
            done,
            StepInfo {
                monitoring_quality: quality / self.cfg.nodes as f64,
                energy_used,
                coverage,
                violations,
                active_nodes: active,
            },
        )
    }
    pub fn node_health(&self) -> Vec<f64> {
        self.nodes.iter().map(|n| n.health).collect()
    }
}
