# Experiment protocol

## Minimum reproducible protocol

1. Save each `EnvConfig`, `TrainingConfig`, source revision, and seed list as JSON alongside results.
2. For every algorithm/condition/seed, create a fresh environment and fresh agent, train for the same episode budget, then set exploration to zero for evaluation.
3. Evaluate at least 20 independent held-out seeds. Report mean and `mean +/- 1.96 * sample_sd / sqrt(n)`; for small samples or non-normal metrics, prefer bootstrap intervals.
4. Store per-episode CSVs, not just aggregate scores. Plot reward, energy, coverage, violations, and decision latency.

## Factorial study matrix

| Factor | Suggested levels |
|---|---|
| Network size | 8, 16, 24, 32 |
| Initial energy | 40, 100, 160 |
| Failure probability | 0.000, 0.002, 0.010 |
| Link drop probability | 0.02, 0.12, 0.30 |
| Discount gamma | 0.90, 0.97, 0.995 |
| Reward energy weight | 0.02, 0.07, 0.15 |
| Environmental volatility | 0.02, 0.08, 0.20 |

Run one factor at a time for diagnosis, then a fractional factorial design for interactions. Keep a random and greedy baseline in every table. `rayon` is used only across independent seeds, so results do not share agent state.

## Metrics

- **Cumulative reward:** sum of simulator rewards.
- **Resource consumption:** total energy action cost.
- **Network coverage:** fraction of nodes yielding usable quality, averaged across steps.
- **Decision latency:** microseconds per step, including agent selection/update.
- **Convergence:** moving mean and variance of training returns, without declaring convergence solely from a plot.
- **Policy stability:** disagreement rate among greedy actions for a fixed held-out observation set across late checkpoints.
- **Constraint violations:** count of non-sleep actions below 5% energy.

## Running baseline outputs

`cargo run --release -- train` produces a deterministic single-seed Q-learning trace. `cargo run --release -- compare` independently trains Q-learning, SARSA, and linear-Q policies across 24 seeds, evaluates them on separate deterministic trajectories, and includes random and greedy baselines. For a full factorial study, change the configs/seed list in a small experiment driver and preserve the same train/test separation.
