# Adaptive Reinforcement Learning for Resource-Constrained Sensor Networks

An offline, CPU-first Rust research environment for distributed infrastructure-monitoring sensor networks. It models 8–32 nodes with bounded energy, sensor health degradation/failure, communication drops, measurement-quality variation and a shared uncertain environment. No service, GPU, or precomputed policy is required.

## Quick start

Install stable Rust from [rustup](https://rustup.rs/) (the toolchain was not installed in the development environment), then run:

```powershell
cargo test
cargo run --release -- train
cargo run --release -- sarsa
cargo run --release -- linear
cargo run --release -- compare
cargo bench
```

`train` writes `results/q_learning_training.csv`, `results/learning_curve.svg`, and `results/topology.svg`. Open the SVG files locally in any browser. `compare` independently trains Q-learning, SARSA, and linear-Q policies, evaluates each on separate deterministic test trajectories, and writes means and normal-approximation 95% confidence intervals to `results/comparison.json`.

## Model and scope

Each node selects one local action at each decision point: `Sleep`, `SenseLow`, `SenseHigh`, or `SenseTransmit`. A single shared policy is trained over compact local observations, which is an intentional parameter-sharing approximation for independent, homogeneous nodes. It is memory-conscious: the tabular state key uses 9 bits (binned energy, health, link, quality, uncertainty), not a joint 32-node state.

The environment deliberately makes no claim that it is a calibrated model of a particular radio stack. Its transition mechanisms are documented in [MDP_MATHEMATICS.md](MDP_MATHEMATICS.md), and all reported outputs are generated from runs—not fabricated convergence figures.

## Project layout

- `src/environment.rs` — stochastic sensor network and reward/termination.
- `src/agents.rs` — random, greedy, Q-learning, SARSA, and linear Q approximation.
- `src/planning.rs` — generic finite-MDP value/policy iteration.
- `src/experiments.rs` — training, seed-parallel evaluation, metrics and CSV output.
- `src/visualization.rs` — offline SVG topology and learning curves.
- `tests/integration.rs` — determinism, termination, and belief-update coverage.

## Reproducibility and experiments

`ChaCha8Rng` is seeded explicitly. Hold the configuration and a seed list fixed to reproduce a run. For defensible comparison, train each learning algorithm separately on each seed, freeze exploration for evaluation, then report the returned mean, interval, coverage, energy, latency, and violations. The shipped `compare` command is a quick baseline smoke test; [EXPERIMENTS.md](EXPERIMENTS.md) gives the full protocol.

## Optional neural extension

Burn is intentionally optional (`cargo build --features dqn`). Default workflows use no neural networ. The feature boundary reserves Burn for a small replay-buffer DQN; it does not silently substitute an untested neural result for a classical baseline.
