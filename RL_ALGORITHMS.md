# RL algorithms

## Implemented methods

| Method | Role | Update / selection |
|---|---|---|
| Random | lower-bound exploration baseline | uniform action |
| Greedy | interpretable non-learning baseline | energy/health/link thresholds |
| Value iteration | exact planning on a supplied finite reduced MDP | Bellman optimality backup |
| Policy iteration | exact planning on a supplied finite reduced MDP | evaluate then improve |
| Q-learning | model-free off-policy control | max next-Q TD target |
| SARSA | model-free on-policy control | sampled next-action TD target |
| Linear Q | scalable baseline | semi-gradient linear approximation |

All learning agents use online transitions from the simulator. There is no hardcoded policy or convergence label. A trained model should be evaluated with exploration disabled and on held-out seeds.

## Advanced methods roadmap

- **POMDP / Bayesian beliefs:** `LinkBelief` is implemented; append beliefs and uncertainty history for a belief-state policy.
- **Entropy exploration:** the tabular Q agent boosts early epsilon modestly. A more principled extension optimizes reward plus `beta H(pi(.|s))` using a softmax actor.
- **Reward shaping:** preserve the base monitoring objective, log each reward term, and test whether shaping changes the target policy.
- **Constrained MDP:** replace static `constraint_penalty` with a primal-dual multiplier and report feasibility confidence intervals.
- **Actor-critic:** a small linear softmax actor plus linear value critic is appropriate before any neural model; use advantage `G-V` and an entropy regularizer.
- **DQN:** Burn is gated behind `dqn`. Add a small CPU ndarray backend, replay buffer, target network, Huber loss, and held-out evaluation before considering results. Large networks are explicitly out of scope.

## Practical cautions

The joint network action space is exponential. Parameter sharing across local node observations keeps the initial study tractable but does not learn coordinated joint policies. Compare it with centralized reduced-state planning only on carefully defined small models. Do not compare algorithms using different numbers of environment interactions or report training-seed results as generalization.
