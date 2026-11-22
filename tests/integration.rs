use adaptive_sensor_rl::{
    agents::QLearning,
    config::EnvConfig,
    environment::{Action, NetworkEnvironment},
    experiments::run_episode,
    mdp::LinkBelief,
};

#[test]
fn deterministic_reset_and_step() {
    let c = EnvConfig::default();
    let mut a = NetworkEnvironment::new(c.clone(), 4).unwrap();
    let mut b = NetworkEnvironment::new(c, 4).unwrap();
    let actions = vec![Action::SenseLow; 16];
    assert_eq!(a.step(&actions).1, b.step(&actions).1);
}

#[test]
fn episode_is_finite() {
    let c = EnvConfig {
        horizon: 4,
        ..Default::default()
    };
    let mut e = NetworkEnvironment::new(c.clone(), 2).unwrap();
    let mut q = QLearning::new(0.1, c.discount, 0.1, 0.9, 0.01, 2);
    let m = run_episode(&mut e, &mut q, true, 0);
    assert!(m.energy >= 0.0);
}

#[test]
fn bayes_ack_increases_link_belief() {
    let mut b = LinkBelief {
        up_probability: 0.5,
    };
    b.observe(true, 0.95, 0.05);
    assert!(b.up_probability > 0.5);
}
