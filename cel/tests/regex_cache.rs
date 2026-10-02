//! The Env's regex cache, seen from outside: the standard library's `matches`
//! compiles a pattern once per size limit, a budget is charged the compile on
//! a miss only, and the results never depend on the cache.
#![cfg(feature = "regex")]

use cel::common::functions::EvalCtx;
use cel::common::types::{CelBool, CelString, STRING_TYPE};
use cel::common::value::CowVal;
use cel::{
    BudgetKind, Context, Env, EvalUsage, ExecutionError, Program, RegexCacheOptions, ResolveResult,
    RuntimeOptions,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A ~100-byte pattern a gateway rule could use on a request path.
const PATH_PATTERN: &str = r"^/api/v[0-9]+/(users|orders|items|carts)/[0-9]+(/[a-z_]+)?(\?[a-z_]+=[a-z0-9_]+(&[a-z_]+=[a-z0-9_]+)*)?$";
const PATH: &str = "/api/v1/users/42/profile?view=full&lang=en";

/// The budget bb-cel sets per request.
fn gateway_budget() -> RuntimeOptions {
    RuntimeOptions::default()
        .with_max_steps(10_000)
        .with_max_bytes(1 << 20)
        .with_regex_size_limit(1 << 20)
        .with_max_regex_len(512)
}

fn context(env: Env) -> Context<'static, 'static> {
    let mut ctx = Context::with_env(Arc::new(env));
    ctx.add_variable_from_value("path", PATH);
    ctx.add_variable_from_value("p", PATH_PATTERN);
    ctx
}

fn run(ctx: &Context, src: &str) -> (ResolveResult, EvalUsage) {
    Program::compile(src)
        .expect("must parse")
        .execute_with_usage(ctx)
}

#[test]
fn the_probe_pattern_is_about_100_bytes() {
    assert!(
        (95..=110).contains(&PATH_PATTERN.len()),
        "{}",
        PATH_PATTERN.len()
    );
    assert!(regex::Regex::new(PATH_PATTERN).unwrap().is_match(PATH));
}

#[test]
fn a_hit_is_charged_only_the_lookup_and_the_match() {
    let mut ctx = context(Env::stdlib());
    ctx.set_budget(gateway_budget());
    let (first, miss) = run(&ctx, "path.matches(p)");
    let (second, hit) = run(&ctx, "path.matches(p)");
    assert_eq!(first, Ok(true.into()));
    assert_eq!(second, Ok(true.into()));
    // the miss pays parsing, translating and compiling the pattern
    assert!(miss.steps > 1_000, "{miss:?}");
    // the hit pays its three nodes, the dispatch, the lookup (2 steps) and
    // the scan: 2 steps per KiB of automaton, 24 KiB for this pattern
    assert!(hit.steps <= 60, "{hit:?}");
    assert!(miss.steps > 50 * hit.steps, "{miss:?} {hit:?}");
}

#[test]
fn a_prewarmed_pattern_is_a_hit() {
    let env = Env::stdlib();
    env.regex_cache().prewarm(PATH_PATTERN, 1 << 20).unwrap();
    // another limit is another entry
    env.regex_cache().prewarm(PATH_PATTERN, 0).unwrap();
    assert_eq!(env.regex_cache().len(), 2);
    assert_eq!(env.regex_cache().pinned_len(), 2);
    let mut ctx = context(env);
    ctx.set_budget(gateway_budget());
    let (result, usage) = run(&ctx, "path.matches(p)");
    assert_eq!(result, Ok(true.into()));
    assert!(usage.steps <= 60, "{usage:?}");
}

#[test]
fn a_cached_error_is_a_cheap_hit() {
    for pattern in ["(", r"\p{Bogus}", r"\w{1000}"] {
        let mut ctx = context(Env::stdlib());
        ctx.add_variable_from_value("bad", pattern);
        ctx.set_budget(
            RuntimeOptions::default()
                .with_max_steps(10_000_000)
                .with_regex_size_limit(64 * 1024),
        );
        let (first, _) = run(&ctx, "'a'.matches(bad)");
        let (second, hit) = run(&ctx, "'a'.matches(bad)");
        assert!(
            matches!(first, Err(ExecutionError::FunctionError { .. })),
            "{first:?}"
        );
        assert_eq!(first, second, "{pattern}");
        assert!(hit.steps <= 10, "{pattern}: {hit:?}");
    }
}

#[test]
fn a_pattern_is_never_served_under_a_smaller_size_limit() {
    // `\w{10}` compiles to ~561 KB: within the regex crate's 10 MiB default
    let mut ctx = context(Env::stdlib());
    assert_eq!(run(&ctx, r"'a'.matches(r'\w{10}')").0, Ok(false.into()));
    ctx.set_budget(
        RuntimeOptions::default()
            .with_max_steps(10_000_000)
            .with_regex_size_limit(16 * 1024),
    );
    assert_eq!(
        run(&ctx, r"'a'.matches(r'\w{10}')").0,
        Err(ExecutionError::function_error(
            "matches",
            "'\\w{10}' not a valid regex:\nCompiled regex exceeds size limit of 16384 bytes."
        ))
    );
    // and a larger limit compiles it again: the key is the limit itself
    ctx.set_budget(
        RuntimeOptions::default()
            .with_max_steps(10_000_000)
            .with_regex_size_limit(1 << 20),
    );
    assert_eq!(
        run(&ctx, r"'aaaaaaaaaa'.matches(r'\w{10}')").0,
        Ok(true.into())
    );
}

/// The regression probe: the same ~100-byte pattern matched once per element
/// of `l`. Without the cache every call paid the parse, the translation and
/// the compile, ~4,400 steps for this pattern, so the gateway budget refused
/// the rule at its third element.
///
/// 2,000 calls cannot fit 10k steps, whatever `matches` costs:
/// `l.all(x, true)` over 2,000 elements is already 10,004 steps, and each
/// call adds three nodes and a dispatch. A hit costs ~56 steps here, most
/// of it the scan charge, which is priced by the compiled automaton.
#[test]
fn a_repeated_pattern_fits_the_budget_through_cache_hits() {
    let mut ctx = context(Env::stdlib());
    ctx.add_variable_from_value("l", (0..80i64).collect::<Vec<_>>());
    ctx.set_budget(gateway_budget());
    let started = Instant::now();
    let (result, usage) = run(&ctx, "l.all(x, path.matches(p))");
    assert_eq!(result, Ok(true.into()), "{usage:?}");
    assert!(started.elapsed() < Duration::from_millis(50));

    // 2,000 calls fit 150k steps: ~61 a call, the pattern compiled once
    let mut ctx = context(Env::stdlib());
    ctx.add_variable_from_value("l", (0..2_000i64).collect::<Vec<_>>());
    ctx.set_budget(gateway_budget().with_max_steps(150_000));
    let (result, usage) = run(&ctx, "l.all(x, path.matches(p))");
    assert_eq!(result, Ok(true.into()), "{usage:?}");
    // the same pattern built anew by every call is a hit too
    let (result, usage) = run(&ctx, "l.all(x, path.matches('^/api/v' + '[0-9]+/.*$'))");
    assert_eq!(result, Ok(true.into()), "{usage:?}");
}

#[test]
fn distinct_patterns_still_pay_their_compile() {
    let mut ctx = context(Env::stdlib());
    ctx.add_variable_from_value("l", (0..2_000i64).collect::<Vec<_>>());
    ctx.set_budget(gateway_budget());
    let (result, _) = run(
        &ctx,
        "l.all(x, !path.matches('a{' + string(x % 1000 + 1) + '}b'))",
    );
    assert_eq!(
        result,
        Err(ExecutionError::BudgetExceeded {
            kind: BudgetKind::Steps,
            limit: 10_000
        })
    );
}

#[test]
fn without_a_budget_results_match_the_regex_crate() {
    let ctx = context(Env::stdlib());
    for pattern in [
        "a.c",
        "^b",
        "(?i)ABC",
        r"^\p{Han}+$",
        "[",
        "(?P<x>a",
        "a{2,1}",
        r"\p{Bogus}",
        r"\w{1000}",
        "",
        r"\bc\b",
    ] {
        for subject in ["abc", "日本語", "a c", ""] {
            let mut ctx = Context::new_inner_scope(&ctx);
            ctx.add_variable_from_value("s", subject);
            ctx.add_variable_from_value("re", pattern);
            let expected = match regex::Regex::new(pattern) {
                Ok(re) => Ok(re.is_match(subject).into()),
                Err(err) => Err(ExecutionError::function_error(
                    "matches",
                    format!("'{pattern}' not a valid regex:\n{err}"),
                )),
            };
            // twice: a miss, then a hit
            assert_eq!(run(&ctx, "s.matches(re)").0, expected, "{pattern}");
            assert_eq!(run(&ctx, "s.matches(re)").0, expected, "{pattern}");
        }
    }
}

#[test]
fn the_cache_compiles_each_pattern_once() {
    let env = Arc::new(Env::stdlib());
    let ctx = Context::with_env(env.clone());
    let program = Program::compile("'abc'.matches('a.c')").unwrap();
    for _ in 0..1_000 {
        assert_eq!(program.execute(&ctx), Ok(true.into()));
    }
    assert_eq!(env.regex_cache().len(), 1);
}

#[test]
fn a_zero_capacity_caches_nothing() {
    let mut env = Env::stdlib();
    env.set_regex_cache_options(RegexCacheOptions::default().with_capacity(0));
    let env = Arc::new(env);
    let mut ctx = Context::with_env(env.clone());
    assert_eq!(run(&ctx, "'abc'.matches('a.c')").0, Ok(true.into()));
    assert_eq!(
        run(&ctx, "'abc'.matches('[')").0,
        run(&ctx, "'abc'.matches('[')").0
    );
    ctx.set_budget(gateway_budget());
    let (_, first) = run(&ctx, "'abc'.matches('a.c')");
    let (_, second) = run(&ctx, "'abc'.matches('a.c')");
    // every call is a miss, charged in full
    assert_eq!(first, second);
    assert!(second.steps > 300, "{second:?}");
    assert_eq!(env.regex_cache().len(), 0);
}

#[test]
fn the_cache_is_shared_across_threads() {
    let env = Arc::new(Env::stdlib());
    let capacity = env.regex_cache().options().capacity();
    std::thread::scope(|scope| {
        for t in 0..8 {
            let env = env.clone();
            scope.spawn(move || {
                let mut ctx = Context::with_env(env);
                if t % 2 == 0 {
                    ctx.set_budget(gateway_budget());
                }
                let program = Program::compile("s.matches(re)").unwrap();
                for i in 0..1_250 {
                    let n = (i * 7 + t) % 100;
                    let mut ctx = Context::new_inner_scope(&ctx);
                    ctx.add_variable_from_value("re", format!("^a{{{n}}}$"));
                    ctx.add_variable_from_value("s", "a".repeat(n + i % 2));
                    assert_eq!(program.execute(&ctx), Ok((i % 2 == 0).into()));
                }
            });
        }
    });
    assert!(env.regex_cache().len() <= capacity);
}

/// `starts_with_slash` that also reports whether it was handed the Env.
fn env_aware<'b, 'v>(
    ectx: &EvalCtx<'_>,
    args: Vec<CowVal<'b, 'v>>,
) -> Result<CowVal<'b, 'v>, ExecutionError> {
    let this = args[0].downcast_ref::<CelString>().unwrap();
    let pattern = args[1].downcast_ref::<CelString>().unwrap();
    let seen = ectx.env().regex_cache().options().capacity() == 7;
    Ok(CowVal::owned(CelBool::from(
        seen && this.inner() == pattern.inner(),
    )))
}

#[test]
fn an_env_function_is_handed_the_env() {
    let mut env = Env::default();
    env.set_regex_cache_options(RegexCacheOptions::default().with_capacity(7));
    env.add_member_overload_with_env(
        "matches",
        "my_matches",
        STRING_TYPE,
        vec![STRING_TYPE],
        env_aware,
    )
    .unwrap();
    env.add_overload_with_env("same", "my_same", vec![STRING_TYPE, STRING_TYPE], env_aware)
        .unwrap();
    // an Env-aware `matches` of the embedder's own is never replaced by
    // the standard library's, with or without a budget, whatever the pattern
    let long = "a".repeat(600);
    let mut ctx = context(env);
    for budget in [false, true] {
        if budget {
            ctx.set_budget(gateway_budget());
        }
        assert_eq!(run(&ctx, "'a.c'.matches('a.c')").0, Ok(true.into()));
        assert_eq!(run(&ctx, "'abc'.matches('a.c')").0, Ok(false.into()));
        assert_eq!(
            run(&ctx, &format!("'{long}'.matches('{long}')")).0,
            Ok(true.into())
        );
        assert_eq!(run(&ctx, r"'a'.matches(r'\w{1000}')").0, Ok(false.into()));
        assert_eq!(run(&ctx, "same('x', 'x')").0, Ok(true.into()));
    }
    // and it is not offered as a plain function
    let env = Env::default();
    assert!(env.find_overload("same", &[]).is_none());
}

#[test]
fn env_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Env>();
    assert_send_sync::<cel::RegexCache>();
}

/// Other tenants of a shared Env cannot evict what was prewarmed: their
/// misses would otherwise flip a rule sized for warm patterns to refused.
#[test]
fn prewarmed_patterns_survive_other_tenants() {
    let env = Arc::new(Env::stdlib());
    env.regex_cache().prewarm(PATH_PATTERN, 1 << 20).unwrap();
    // far more distinct patterns than the cache holds
    let mut attacker = Context::with_env(env.clone());
    attacker.add_variable_from_value("l", (0..500i64).collect::<Vec<_>>());
    attacker.add_variable_from_value("path", PATH);
    assert_eq!(
        run(
            &attacker,
            "l.all(x, !path.matches('^/z' + string(x) + '$'))"
        )
        .0,
        Ok(true.into())
    );
    let mut ctx = Context::with_env(env);
    ctx.add_variable_from_value("path", PATH);
    ctx.add_variable_from_value("p", PATH_PATTERN);
    ctx.set_budget(gateway_budget());
    let (result, usage) = run(&ctx, "path.matches(p)");
    assert_eq!(result, Ok(true.into()));
    assert!(usage.steps <= 60, "{usage:?}");
}

/// Under a budget, the cache keeps only what an evaluation paid for: a
/// compile whose charge stops the evaluation is dropped.
#[test]
fn a_compile_the_budget_refuses_is_not_kept() {
    let env = Arc::new(Env::stdlib());
    let mut ctx = Context::with_env(env.clone());
    ctx.add_variable_from_value("path", PATH);
    ctx.set_budget(gateway_budget());
    for pattern in [r"(?:\w{17}|z1)y", r"\w{1000}"] {
        ctx.add_variable_from_value("p", pattern);
        let (result, _) = run(&ctx, "'a'.matches(p) || true");
        assert_eq!(
            result,
            Err(ExecutionError::BudgetExceeded {
                kind: BudgetKind::Steps,
                limit: 10_000
            }),
            "{pattern}"
        );
    }
    assert_eq!(env.regex_cache().len(), 0);
    // what it can pay for is kept
    ctx.add_variable_from_value("p", PATH_PATTERN);
    assert_eq!(run(&ctx, "path.matches(p)").0, Ok(true.into()));
    assert_eq!(env.regex_cache().len(), 1);
}
