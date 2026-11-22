# MDP and mathematical model

## Assumptions and MDP

At time `t`, node `i` has physical state `x_i=(e_i,h_i,l_i,q_i)`: normalized remaining energy, health, binary link state, and most recent quality. The network also has a latent environmental condition `z_t`. The true state is `s_t=(x_1,...,x_N,z_t)` and the joint action is `a_t=(a_1,...,a_N)`, with four actions per node. This is a discounted MDP

`M = (S, A, P, R, gamma)`, with `0 <= gamma < 1`.

The simulator uses conditional independent node transitions given `z_t`; `z_t` is a clipped, mean-reverting noisy process. Health degrades faster when action cost is higher; failures apply a random health shock; links drop stochastically; measurement quality is proportional to sensing effort, health, latent environment, and a small multiplicative noise. These are lightweight assumptions for algorithm comparison, not radio-physics claims.

The learner receives local observation `o_i=(e_i,h_i,l_i,q_i,u_t)`, where `u_t=2|z_t-0.5|`, rather than full state. Thus the learning task is operationally a POMDP. The tabular agent bins this observation into a compact key. An episode ends at the configured horizon or when no active node remains.

The per-step reward is

`r_t = w_q Q_t + w_c C_t - w_e E_t - w_v V_t`,

where `Q_t` is mean delivered monitoring quality, `C_t` coverage, `E_t` action energy, and `V_t` actions that violate the low-energy constraint. Reward shaping is explicit through these weights; changing weights is an experiment, not a post-hoc score adjustment.

## Bellman equations

For policy `pi`, return is `G_t = sum_(k=0)^infinity gamma^k R_(t+k+1)` and the value is `V^pi(s)=E_pi[G_t | S_t=s]`.

The Bellman expectation equation is

`V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a)[R(s,a,s') + gamma V^pi(s')]`.

The optimality equation is

`V*(s) = max_a sum_s' P(s'|s,a)[R(s,a,s') + gamma V*(s')]`.

`Q^pi(s,a)` replaces the outer action expectation with a fixed first action. In the generic finite-MDP module, policy evaluation repeatedly applies the expectation backup; policy improvement picks the maximizing action. Value iteration applies the optimality backup directly.

## Learning updates

For a sampled transition, temporal-difference error is

`delta_t = r_(t+1) + gamma V(S_(t+1)) - V(S_t)`.

Off-policy Q-learning uses

`Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]`.

On-policy SARSA uses its sampled next action:

`Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]`.

Linear Q uses `Q_w(s,a)=w_a^T phi(s)` and the semi-gradient update `w_a <- w_a + alpha delta phi(s)`. The selected feature vector is a bias plus the five normalized local observation fields.

## Partial observability, beliefs, and constraints

`LinkBelief` implements Bayes filtering for a hidden binary link. It first predicts with drop/recovery probabilities, then conditions on an ACK or timeout with specified sensitivity/false-ACK rates. A future policy can append this belief to the observation.

The low-energy action count is a measurable cost `V_t`; the present implementation uses a Lagrangian-style penalty. A strict constrained-MDP extension should maintain `E[sum gamma^t V_t] <= d` and update a nonnegative multiplier from observed violation excess, reporting the constraint separately. Exploration is epsilon-greedy with an early entropy-inspired boost; its benefit is coverage of uncertain actions, while its cost is temporary resource use and slower exploitation.
