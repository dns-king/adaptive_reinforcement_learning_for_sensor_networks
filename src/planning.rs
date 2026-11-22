use crate::mdp::FiniteMdp;

pub fn value_iteration(
    mdp: &FiniteMdp,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<usize>) {
    mdp.validate().expect("valid MDP");
    let mut v = vec![0.0; mdp.states];
    for _ in 0..max_iterations {
        let old = v.clone();
        let mut delta: f64 = 0.0;
        for s in 0..mdp.states {
            v[s] = (0..mdp.actions)
                .map(|a| {
                    mdp.rewards[s][a]
                        + mdp.gamma
                            * mdp.transitions[s][a]
                                .iter()
                                .map(|(n, p)| p * old[*n])
                                .sum::<f64>()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            delta = delta.max((v[s] - old[s]).abs());
        }
        if delta < tolerance {
            break;
        }
    }
    let policy = (0..mdp.states)
        .map(|s| {
            (0..mdp.actions)
                .max_by(|&a, &b| {
                    let q = |x: usize| {
                        mdp.rewards[s][x]
                            + mdp.gamma
                                * mdp.transitions[s][x]
                                    .iter()
                                    .map(|(n, p)| p * v[*n])
                                    .sum::<f64>()
                    };
                    q(a).partial_cmp(&q(b)).unwrap()
                })
                .unwrap()
        })
        .collect();
    (v, policy)
}

pub fn policy_iteration(
    mdp: &FiniteMdp,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<usize>) {
    mdp.validate().expect("valid MDP");
    let mut pi = vec![0; mdp.states];
    let mut v = vec![0.0; mdp.states];
    for _ in 0..max_iterations {
        // iterative policy evaluation
        for _ in 0..max_iterations {
            let old = v.clone();
            let mut delta: f64 = 0.0;
            for s in 0..mdp.states {
                let a = pi[s];
                v[s] = mdp.rewards[s][a]
                    + mdp.gamma
                        * mdp.transitions[s][a]
                            .iter()
                            .map(|(n, p)| p * old[*n])
                            .sum::<f64>();
                delta = delta.max((v[s] - old[s]).abs());
            }
            if delta < tolerance {
                break;
            }
        }
        let mut stable = true;
        for s in 0..mdp.states {
            let old = pi[s];
            pi[s] = (0..mdp.actions)
                .max_by(|&a, &b| {
                    let q = |x: usize| {
                        mdp.rewards[s][x]
                            + mdp.gamma
                                * mdp.transitions[s][x]
                                    .iter()
                                    .map(|(n, p)| p * v[*n])
                                    .sum::<f64>()
                    };
                    q(a).partial_cmp(&q(b)).unwrap()
                })
                .unwrap();
            stable &= old == pi[s];
        }
        if stable {
            break;
        }
    }
    (v, pi)
}
