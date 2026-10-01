//! Probe benchmarks recorded for the BunnyBox fork (see docs/bunnybox/).
//!
//! Four expressions: two Arc-sharing probes over a large `request` payload, the optional-clone
//! repro, and the unbounded nested-comprehension (budget) probe.
use cel::objects::OptionalValue;
use cel::{Context, Program, Value};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// `request` with 128 query entries and 100 headers of 4 KiB each (~400 KiB).
fn request() -> Value {
    let query: HashMap<String, Value> = (0..128)
        .map(|i| (format!("q{i}"), Value::from(format!("value-{i}"))))
        .collect();
    let headers: HashMap<String, Value> = (0..100)
        .map(|i| (format!("h{i}"), Value::from("x".repeat(4096))))
        .collect();
    let mut request: HashMap<String, Value> = HashMap::new();
    request.insert("query".into(), Value::from(query));
    request.insert("headers".into(), Value::from(headers));
    Value::from(request)
}

fn probes(c: &mut Criterion) {
    let mut group = c.benchmark_group("probes");
    // The slower probes take 0.1-1.5 s per iteration: keep the sample count small.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let x16 = format!(
        "size(request.query.map(k, [{}])) > 0",
        vec!["request"; 16].join(", ")
    );
    let cases: [(&str, String); 4] = [
        (
            "query_map_request",
            "size(request.query.map(k, request)) > 0".into(),
        ),
        ("query_map_request_x16", x16),
        ("optional_clone", "[1].map(x, o)[0].hasValue()".into()),
        (
            "nested_all_4m",
            "size(l.all(x, l.all(y, true)) ? [1] : []) > 0".into(),
        ),
    ];
    for (name, expr) in cases {
        let program = Program::compile(&expr).expect("compile");
        let mut ctx = Context::default();
        ctx.add_variable_from_value("request", request());
        ctx.add_variable_from_value("l", (0..2000i64).collect::<Vec<_>>());
        ctx.add_variable_from_value("o", Value::Opaque(Arc::new(OptionalValue::of(1i64.into()))));
        group.bench_function(name, |b| {
            // The optional probe errors on 4d469a8 (the bug); the result is only black-boxed.
            b.iter(|| black_box(program.execute(&ctx)))
        });
    }
    group.finish();
}

criterion_group!(benches, probes);
criterion_main!(benches);
