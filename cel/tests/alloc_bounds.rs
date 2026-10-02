//! Heap allocation bounds for evaluating over large inputs.
//!
//! Lists, maps, strings and bytes are shared rather than deep-copied when a
//! value is cloned, bound as a variable or converted to and from [`Value`].
//! These tests pin that down by counting the bytes allocated while a single
//! expression runs over a ~400 KiB input: a deep copy of the input on every
//! iteration shows up as tens of MiB.
//!
//! Run with `cargo test -p cel --features dhat-heap --test alloc_bounds -- --test-threads=1`.
#![cfg(feature = "dhat-heap")]

use cel::{Context, Program, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Only one dhat profiler may be live at a time.
static PROFILER: Mutex<()> = Mutex::new(());

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;

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

/// The bytes allocated while `f` runs.
fn bytes_allocated<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = dhat::HeapStats::get().total_bytes;
    let out = f();
    let after = dhat::HeapStats::get().total_bytes;
    (out, after - before)
}

/// Asserts that `bytes` is under `limit`, and reports it either way.
#[track_caller]
fn assert_below(bytes: u64, limit: u64) {
    println!("allocated {bytes} bytes (limit {limit})");
    assert!(bytes < limit, "allocated {bytes} bytes, limit {limit}");
}

/// Runs `test` under the dhat profiler.
fn profiled(test: impl FnOnce()) {
    let _guard = PROFILER.lock().unwrap_or_else(|e| e.into_inner());
    let _profiler = dhat::Profiler::builder().testing().build();
    test();
}

/// The bytes allocated by executing `expr` against a context holding
/// `request`, excluding the setup of the context.
fn execute_with_request(expr: &str) -> (Value, u64) {
    let program = Program::compile(expr).unwrap();
    let mut ctx = Context::default();
    ctx.add_variable_from_value("request", request());
    let (result, bytes) = bytes_allocated(|| program.execute(&ctx));
    (result.unwrap(), bytes)
}

#[test]
fn map_over_request_does_not_copy_it() {
    profiled(|| {
        let (result, bytes) = execute_with_request("size(request.query.map(k, request)) > 0");
        assert_eq!(result, Value::Bool(true));
        assert_below(bytes, MIB);
    });
}

#[test]
fn map_over_request_x16_does_not_copy_it() {
    profiled(|| {
        let expr = format!(
            "size(request.query.map(k, [{}])) > 0",
            vec!["request"; 16].join(", ")
        );
        let (result, bytes) = execute_with_request(&expr);
        assert_eq!(result, Value::Bool(true));
        assert_below(bytes, MIB);
    });
}

#[test]
fn map_to_large_strings_does_not_copy_them() {
    profiled(|| {
        let (result, bytes) =
            execute_with_request("size(request.headers.map(k, request.headers[k])) == 100");
        assert_eq!(result, Value::Bool(true));
        // 100 headers of 4 KiB each: a copy of each would be 400 KiB
        assert_below(bytes, 64 * KIB);
    });
}

#[test]
fn filter_over_request_does_not_copy_it() {
    profiled(|| {
        let (result, bytes) = execute_with_request(
            "size([request, request, request].filter(r, size(r.headers) == 100)) == 3",
        );
        assert_eq!(result, Value::Bool(true));
        assert_below(bytes, 64 * KIB);
    });
}

#[test]
fn binding_a_value_does_not_copy_its_strings() {
    profiled(|| {
        let request = request();
        let mut ctx = Context::default();
        let ((), bytes) =
            bytes_allocated(|| ctx.add_variable_from_value("request", request.clone()));
        // the headers alone are 400 KiB
        assert_below(bytes, 64 * KIB);
    });
}

#[test]
fn returning_a_value_does_not_copy_its_strings() {
    profiled(|| {
        let (result, bytes) = execute_with_request("request.headers");
        let Value::Map(headers) = result else {
            panic!("expected a map, got {result:?}")
        };
        assert_eq!(headers.map.len(), 100);
        assert_below(bytes, 64 * KIB);
    });
}

#[test]
fn map_still_linear() {
    profiled(|| {
        let bytes_for = |n: i64| {
            let program = Program::compile("l.map(x, [x, x]).filter(p, p[0] >= 0)").unwrap();
            let mut ctx = Context::default();
            ctx.add_variable_from_value("l", (0..n).collect::<Vec<_>>());
            let (result, bytes) = bytes_allocated(|| program.execute(&ctx));
            match result.unwrap() {
                Value::List(l) => assert_eq!(l.len() as i64, n),
                other => panic!("expected a list, got {other:?}"),
            }
            bytes
        };
        let small = bytes_for(1_000);
        let large = bytes_for(4_000);
        // 4x the input allocates about 4x the bytes; a quadratic accumulator would be 16x
        assert!(
            large < small * 6,
            "1,000 elements allocated {small} bytes, 4,000 allocated {large}"
        );
    });
}

#[test]
fn shared_string_variable_is_not_copied_into_a_function() {
    profiled(|| {
        let big = Arc::new("x".repeat(64 * KIB as usize));
        let program = Program::compile("len(s) == 65536").unwrap();
        let mut ctx = Context::default();
        ctx.add_function("len", |s: Arc<String>| s.len() as i64)
            .unwrap();
        ctx.add_variable_from_value("s", Value::String(big.clone()));
        let (result, bytes) = bytes_allocated(|| program.execute(&ctx));
        assert_eq!(result.unwrap(), Value::Bool(true));
        assert_below(bytes, 4 * KIB);
    });
}
