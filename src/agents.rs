use crate::environment::{Action, Observation};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

pub trait Agent: Send {
    fn select(&mut self, observation: Observation, training: bool) -> Action;
    fn update(
        &mut self,
        state: Observation,
        action: Action,
        reward: f64,
        next: Observation,
        done: bool,
    );
    fn name(&self) -> &'static str;
}

pub struct RandomPolicy {
    rng: ChaCha8Rng,
}
impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}
impl Agent for RandomPolicy {
    fn select(&mut self, _: Observation, _: bool) -> Action {
        Action::ALL[self.rng.gen_range(0..4)]
    }
    fn update(&mut self, _: Observation, _: Action, _: f64, _: Observation, _: bool) {}
    fn name(&self) -> &'static str {
        "random"
    }
}

/// A transparent non-learning baseline: act only when local state can support it.
pub struct GreedyPolicy;
impl Agent for GreedyPolicy {
    fn select(&mut self, o: Observation, _: bool) -> Action {
        if o.energy < 0.08 || o.health < 0.10 {
            Action::Sleep
        } else if o.link_up && o.quality > 0.35 {
            Action::SenseTransmit
        } else if o.quality < 0.25 {
            Action::SenseHigh
        } else {
            Action::SenseLow
        }
    }
    fn update(&mut self, _: Observation, _: Action, _: f64, _: Observation, _: bool) {}
    fn name(&self) -> &'static str {
        "greedy"
    }
}

fn argmax(q: &[f64; 4]) -> Action {
    let mut best = 0;
    for i in 1..4 {
        if q[i] > q[best] {
            best = i;
        }
    }
    Action::ALL[best]
}
fn entropy_epsilon(epsilon: f64, steps: usize) -> f64 {
    // modest entropy bonus early, bounded for deterministic evaluation
    (epsilon * (1.0 + 1.0 / (steps as f64 + 1.0).sqrt())).min(1.0)
}

pub struct QLearning {
    q: HashMap<u16, [f64; 4]>,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    decay: f64,
    floor: f64,
    steps: usize,
    rng: ChaCha8Rng,
}
impl QLearning {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, decay: f64, floor: f64, seed: u64) -> Self {
        Self {
            q: HashMap::new(),
            alpha,
            gamma,
            epsilon,
            decay,
            floor,
            steps: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
    pub fn states_seen(&self) -> usize {
        self.q.len()
    }
}
impl Agent for QLearning {
    fn select(&mut self, o: Observation, training: bool) -> Action {
        let eps = if training {
            entropy_epsilon(self.epsilon, self.steps)
        } else {
            0.0
        };
        self.steps += 1;
        if self.rng.gen_bool(eps) {
            Action::ALL[self.rng.gen_range(0..4)]
        } else {
            argmax(self.q.get(&o.key()).unwrap_or(&[0.0; 4]))
        }
    }
    fn update(&mut self, s: Observation, a: Action, r: f64, n: Observation, done: bool) {
        let target = r + if done {
            0.0
        } else {
            self.gamma
                * self
                    .q
                    .get(&n.key())
                    .unwrap_or(&[0.0; 4])
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
        };
        let q = self.q.entry(s.key()).or_insert([0.0; 4]);
        q[a as usize] += self.alpha * (target - q[a as usize]);
        self.epsilon = (self.epsilon * self.decay).max(self.floor);
    }
    fn name(&self) -> &'static str {
        "q_learning"
    }
}

pub struct Sarsa {
    q: HashMap<u16, [f64; 4]>,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    decay: f64,
    floor: f64,
    rng: ChaCha8Rng,
}
impl Sarsa {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, decay: f64, floor: f64, seed: u64) -> Self {
        Self {
            q: HashMap::new(),
            alpha,
            gamma,
            epsilon,
            decay,
            floor,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}
impl Agent for Sarsa {
    fn select(&mut self, o: Observation, training: bool) -> Action {
        if training && self.rng.gen_bool(self.epsilon) {
            Action::ALL[self.rng.gen_range(0..4)]
        } else {
            argmax(self.q.get(&o.key()).unwrap_or(&[0.; 4]))
        }
    }
    fn update(&mut self, s: Observation, a: Action, r: f64, n: Observation, done: bool) {
        let next = if done {
            0.0
        } else {
            let b = self.select(n, true);
            self.q.get(&n.key()).unwrap_or(&[0.; 4])[b as usize]
        };
        let q = self.q.entry(s.key()).or_insert([0.; 4]);
        q[a as usize] += self.alpha * (r + self.gamma * next - q[a as usize]);
        self.epsilon = (self.epsilon * self.decay).max(self.floor);
    }
    fn name(&self) -> &'static str {
        "sarsa"
    }
}

/// Linear semi-gradient Q approximation with feature vector [bias, local observation].
pub struct LinearQ {
    weights: [[f64; 6]; 4],
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    rng: ChaCha8Rng,
}
impl LinearQ {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, seed: u64) -> Self {
        Self {
            weights: [[0.; 6]; 4],
            alpha,
            gamma,
            epsilon,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
    fn x(o: Observation) -> [f64; 6] {
        let f = o.features();
        [1., f[0], f[1], f[2], f[3], f[4]]
    }
    fn value(&self, o: Observation, a: Action) -> f64 {
        Self::x(o)
            .iter()
            .zip(self.weights[a as usize])
            .map(|(x, w)| x * w)
            .sum()
    }
}
impl Agent for LinearQ {
    fn select(&mut self, o: Observation, training: bool) -> Action {
        if training && self.rng.gen_bool(self.epsilon) {
            Action::ALL[self.rng.gen_range(0..4)]
        } else {
            Action::ALL
                .into_iter()
                .max_by(|a, b| self.value(o, *a).partial_cmp(&self.value(o, *b)).unwrap())
                .unwrap()
        }
    }
    fn update(&mut self, s: Observation, a: Action, r: f64, n: Observation, done: bool) {
        let next = if done {
            0.
        } else {
            Action::ALL
                .into_iter()
                .map(|x| self.value(n, x))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let delta = r + self.gamma * next - self.value(s, a);
        for (w, x) in self.weights[a as usize].iter_mut().zip(Self::x(s)) {
            *w += self.alpha * delta * x;
        }
    }
    fn name(&self) -> &'static str {
        "linear_q"
    }
}

#[cfg(feature = "dqn")]
pub mod dqn {
    //! Burn is isolated behind `--features dqn`; use tabular/linear agents for the default CPU workflow.
    pub const STATUS: &str = "DQN extension enabled; supply replay buffer and Burn model architecture for the selected backend.";
}
