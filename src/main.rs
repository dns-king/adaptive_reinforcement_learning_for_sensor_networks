use adaptive_sensor_rl::{
    agents::{GreedyPolicy, LinearQ, QLearning, RandomPolicy, Sarsa},
    config::{EnvConfig, TrainingConfig},
    environment::NetworkEnvironment,
    experiments::{parallel_evaluate, parallel_train_evaluate, run_episode, train, write_csv},
    visualization,
};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("demo");
    let out = Path::new("results");
    fs::create_dir_all(out)?;
    let ec = EnvConfig::default();
    let tc = TrainingConfig::default();
    match command {
        "train" | "demo" => {
            let mut e = NetworkEnvironment::new(ec.clone(), tc.seed)?;
            let mut a = QLearning::new(
                tc.alpha,
                ec.discount,
                tc.epsilon_start,
                tc.epsilon_decay,
                tc.epsilon_end,
                tc.seed,
            );
            let rows = train(&mut e, &mut a, &tc);
            write_csv(&out.join("q_learning_training.csv"), &rows)?;
            visualization::learning_curve(&out.join("learning_curve.svg"), &rows)?;
            visualization::topology(&out.join("topology.svg"), &e.node_health())?;
            println!("trained {} states; wrote results/", a.states_seen());
        }
        "compare" => {
            let seeds: Vec<u64> = (0..24).collect();
            let qs = parallel_train_evaluate(ec.clone(), &tc, &seeds, |s| {
                QLearning::new(
                    tc.alpha,
                    ec.discount,
                    tc.epsilon_start,
                    tc.epsilon_decay,
                    tc.epsilon_end,
                    s,
                )
            });
            let ss = parallel_train_evaluate(ec.clone(), &tc, &seeds, |s| {
                Sarsa::new(
                    tc.alpha,
                    ec.discount,
                    tc.epsilon_start,
                    tc.epsilon_decay,
                    tc.epsilon_end,
                    s,
                )
            });
            let ls = parallel_train_evaluate(ec.clone(), &tc, &seeds, |s| {
                LinearQ::new(0.01, ec.discount, 0.1, s)
            });
            let rs = parallel_evaluate(ec.clone(), &seeds, RandomPolicy::new);
            let gs = parallel_evaluate(ec.clone(), &seeds, |_| GreedyPolicy);
            fs::write(
                out.join("comparison.json"),
                serde_json::to_string_pretty(&vec![qs, ss, ls, rs, gs])?,
            )?;
            println!("wrote results/comparison.json");
        }
        "sarsa" => {
            let mut e = NetworkEnvironment::new(ec.clone(), tc.seed)?;
            let mut a = Sarsa::new(
                tc.alpha,
                ec.discount,
                tc.epsilon_start,
                tc.epsilon_decay,
                tc.epsilon_end,
                tc.seed,
            );
            let rows = train(&mut e, &mut a, &tc);
            write_csv(&out.join("sarsa_training.csv"), &rows)?;
        }
        "linear" => {
            let mut e = NetworkEnvironment::new(ec.clone(), tc.seed)?;
            let mut a = LinearQ::new(0.01, ec.discount, 0.1, tc.seed);
            let rows = train(&mut e, &mut a, &tc);
            write_csv(&out.join("linear_q_training.csv"), &rows)?;
        }
        "evaluate" => {
            let mut e = NetworkEnvironment::new(ec, tc.seed)?;
            let mut a = GreedyPolicy;
            println!("{:#?}", run_episode(&mut e, &mut a, false, 0));
        }
        _ => eprintln!("usage: cargo run -- [train|compare|sarsa|linear|evaluate]"),
    };
    Ok(())
}
