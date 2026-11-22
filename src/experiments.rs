use crate::{
    agents::Agent,
    config::{EnvConfig, TrainingConfig},
    environment::{Action, NetworkEnvironment, Observation, StepInfo},
};
use nalgebra::DVector;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EpisodeMetrics {
    pub episode: usize,
    pub reward: f64,
    pub energy: f64,
    pub coverage: f64,
    pub violations: usize,
    pub latency_us: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Summary {
    pub policy: String,
    pub seeds: usize,
    pub mean_reward: f64,
    pub ci95: f64,
    pub mean_energy: f64,
    pub mean_coverage: f64,
    pub mean_violations: f64,
}

pub fn run_episode<A: Agent>(
    env: &mut NetworkEnvironment,
    agent: &mut A,
    training: bool,
    episode: usize,
) -> EpisodeMetrics {
    let mut obs = env.reset();
    let mut out = EpisodeMetrics {
        episode,
        ..Default::default()
    };
    let begin = std::time::Instant::now();
    loop {
        let acts: Vec<Action> = obs
            .iter()
            .copied()
            .map(|x| agent.select(x, training))
            .collect();
        let (next, reward, done, info): (Vec<Observation>, f64, bool, StepInfo) = env.step(&acts);
        if training {
            for ((s, a), n) in obs.iter().copied().zip(acts).zip(next.iter().copied()) {
                agent.update(s, a, reward / env.config().nodes as f64, n, done);
            }
        }
        out.reward += reward;
        out.energy += info.energy_used;
        out.coverage += info.coverage;
        out.violations += info.violations;
        if done {
            break;
        }
        obs = next;
    }
    out.coverage /= env.config().horizon as f64;
    out.latency_us = begin.elapsed().as_secs_f64() * 1e6 / env.config().horizon as f64;
    out
}
pub fn train<A: Agent>(
    env: &mut NetworkEnvironment,
    agent: &mut A,
    cfg: &TrainingConfig,
) -> Vec<EpisodeMetrics> {
    (0..cfg.episodes)
        .map(|e| run_episode(env, agent, true, e))
        .collect()
}

/// Independent seeds execute in parallel, without sharing mutable agent state.
pub fn parallel_evaluate<F, A>(env_cfg: EnvConfig, seeds: &[u64], factory: F) -> Summary
where
    F: Fn(u64) -> A + Sync + Send,
    A: Agent,
{
    assert!(!seeds.is_empty(), "evaluation requires at least one seed");
    let records: Vec<EpisodeMetrics> = seeds
        .par_iter()
        .map(|&seed| {
            let mut env = NetworkEnvironment::new(env_cfg.clone(), seed).unwrap();
            let mut a = factory(seed);
            run_episode(&mut env, &mut a, false, 0)
        })
        .collect();
    summarize(factory(seeds[0]).name(), &records)
}

/// Train and evaluate independent replicas on disjoint deterministic trajectories.
pub fn parallel_train_evaluate<F, A>(
    env_cfg: EnvConfig,
    training: &TrainingConfig,
    seeds: &[u64],
    factory: F,
) -> Summary
where
    F: Fn(u64) -> A + Sync + Send,
    A: Agent,
{
    assert!(!seeds.is_empty(), "evaluation requires at least one seed");
    let records: Vec<EpisodeMetrics> = seeds
        .par_iter()
        .map(|&seed| {
            let mut agent = factory(seed);
            let mut train_env = NetworkEnvironment::new(env_cfg.clone(), seed).unwrap();
            train(&mut train_env, &mut agent, training);
            let mut test_env =
                NetworkEnvironment::new(env_cfg.clone(), seed ^ 0x9e37_79b9).unwrap();
            run_episode(&mut test_env, &mut agent, false, 0)
        })
        .collect();
    summarize(factory(seeds[0]).name(), &records)
}

fn summarize(policy: &str, records: &[EpisodeMetrics]) -> Summary {
    let n = records.len() as f64;
    // A dense vector keeps the statistical reduction explicit and leaves a natural path
    // for covariance / least-squares analyses without storing joint network states.
    let rewards = DVector::from_iterator(records.len(), records.iter().map(|r| r.reward));
    let mean = rewards.iter().sum::<f64>() / n;
    let sd = if n > 1.0 {
        (rewards.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.)).sqrt()
    } else {
        0.
    };
    Summary {
        policy: policy.to_string(),
        seeds: records.len(),
        mean_reward: mean,
        ci95: 1.96 * sd / n.sqrt(),
        mean_energy: records.iter().map(|x| x.energy).sum::<f64>() / n,
        mean_coverage: records.iter().map(|x| x.coverage).sum::<f64>() / n,
        mean_violations: records.iter().map(|x| x.violations as f64).sum::<f64>() / n,
    }
}
pub fn write_csv(path: &Path, records: &[EpisodeMetrics]) -> std::io::Result<()> {
    let mut s = String::from("episode,reward,energy,coverage,violations,latency_us\n");
    for r in records {
        s.push_str(&format!(
            "{},{:.8},{:.8},{:.8},{}, {:.4}\n",
            r.episode, r.reward, r.energy, r.coverage, r.violations, r.latency_us
        ));
    }
    fs::write(path, s)
}
