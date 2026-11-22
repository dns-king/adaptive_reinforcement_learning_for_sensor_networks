use adaptive_sensor_rl::{
    config::EnvConfig,
    environment::{Action, NetworkEnvironment},
};
use criterion::{criterion_group, criterion_main, Criterion};
fn step(c: &mut Criterion) {
    c.bench_function("network_step_32", |b| {
        b.iter(|| {
            let mut e = NetworkEnvironment::new(
                EnvConfig {
                    nodes: 32,
                    ..Default::default()
                },
                1,
            )
            .unwrap();
            e.step(&vec![Action::SenseLow; 32])
        })
    });
}
criterion_group!(benches, step);
criterion_main!(benches);
