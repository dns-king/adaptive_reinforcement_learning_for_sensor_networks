# Results record

This repository intentionally contains no claimed numerical results. Run outputs are generated locally in `results/` and should be versioned or archived with the exact configuration and seed list used.

Use the following reporting template:

| Policy | Train seeds | Test seeds | Cumulative reward (95% CI) | Energy | Coverage | Violations | Latency (us/step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | | | | | | | |
| Greedy | | | | | | | |
| Q-learning | | | | | | | |
| SARSA | | | | | | | |
| Linear Q | | | | | | | |

Interpret resource use together with monitoring quality and constraints: a higher reward is not sufficient if its confidence interval overlaps a baseline, violations grow, or it was measured on training seeds. Include the generated topology/health and learning-curve SVGs with every report.
