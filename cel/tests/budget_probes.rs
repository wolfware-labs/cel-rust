//! The BunnyBox probe expressions: legal CEL whose cost grows with its input.
//! Unbounded they complete; under a modest budget they abort with
//! `BudgetExceeded` instead of pinning the worker.
use cel::{BudgetKind, Context, ExecutionError, Program, RuntimeOptions, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

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

fn context() -> Context<'static, 'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("request", request());
    ctx.add_variable_from_value("l", (0..2000i64).collect::<Vec<_>>());
    ctx
}

fn probes() -> Vec<(&'static str, String)> {
    let x16 = format!(
        "size(request.query.map(k, [{}])) > 0",
        vec!["request"; 16].join(", ")
    );
    vec![
        (
            "query_map_request",
            "size(request.query.map(k, request)) > 0".into(),
        ),
        ("query_map_request_x16", x16),
        (
            "nested_all_4m",
            "size(l.all(x, l.all(y, true)) ? [1] : []) > 0".into(),
        ),
    ]
}

/// A budget an API gateway could set per request: room for real rules, none
/// for work that grows with the payload.
fn modest() -> RuntimeOptions {
    RuntimeOptions::default()
        .with_max_steps(256)
        .with_max_bytes(16 * 1024)
}

#[test]
fn probes_complete_without_a_budget() {
    let ctx = context();
    for (name, src) in probes() {
        let program = Program::compile(&src).unwrap();
        let (result, usage) = program.execute_with_usage(&ctx);
        assert_eq!(result, Ok(true.into()), "{name}");
        assert!(usage.steps > 256, "{name}: {usage:?}");
    }
}

#[test]
fn probes_abort_under_a_modest_budget() {
    let mut ctx = context();
    ctx.set_budget(modest());
    for (name, src) in probes() {
        let program = Program::compile(&src).unwrap();
        let started = Instant::now();
        let result = program.execute(&ctx);
        assert!(
            matches!(result, Err(ExecutionError::BudgetExceeded { .. })),
            "{name}: {result:?}"
        );
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "{name} took {:?}",
            started.elapsed()
        );
    }
    // and a rule of the kind the budget is meant for fits it
    let rule = Program::compile(
        "request.query.q1 == 'value-1' && size(request.headers.h0) > 0 \
         && 'q7' in request.query",
    )
    .unwrap();
    assert_eq!(rule.execute(&ctx), Ok(true.into()));
}

#[test]
fn the_nested_probe_trips_the_steps_budget() {
    let mut ctx = context();
    ctx.set_budget(modest());
    let program = Program::compile("size(l.all(x, l.all(y, true)) ? [1] : []) > 0").unwrap();
    assert_eq!(
        program.execute(&ctx),
        Err(ExecutionError::BudgetExceeded {
            kind: BudgetKind::Steps,
            limit: 256
        })
    );
}

#[test]
fn returning_the_shared_payload_many_times_trips_the_bytes_budget() {
    // cheap to build (the payload is shared), but converting the result
    // would materialise the ~400 KiB payload 128 times
    let mut ctx = context();
    ctx.set_budget(RuntimeOptions::default().with_max_bytes(1024 * 1024));
    let program = Program::compile("request.query.map(k, request)").unwrap();
    let started = Instant::now();
    assert_eq!(
        program.execute(&ctx),
        Err(ExecutionError::BudgetExceeded {
            kind: BudgetKind::Bytes,
            limit: 1024 * 1024
        })
    );
    assert!(started.elapsed() < Duration::from_millis(100));
}
