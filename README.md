# Common Expression Language (Rust)

[![Rust](https://github.com/cel-rust/cel-rust/actions/workflows/rust.yml/badge.svg)](https://github.com/cel-rust/cel-rust/actions/workflows/rust.yml)

The [Common Expression Language (CEL)](https://github.com/google/cel-spec) is a non-Turing complete language designed
for simplicity, speed, safety, and
portability. CEL's C-like syntax looks nearly identical to equivalent expressions in C++, Go, Java, and TypeScript. CEL
is ideal for lightweight expression evaluation when a fully sandboxed scripting language is too resource intensive.

```java
// Check whether a resource name starts with a group name.
resource.name.startsWith("/groups/" + auth.claims.group)
```

```go
// Determine whether the request is in the permitted time window.
request.time - resource.age < duration("24h")
```

```typescript
// Check whether all resource names in a list match a given filter.
auth.claims.email_verified && resources.all(r, r.startsWith(auth.claims.email))
```

## Getting Started

Add `cel` to your `Cargo.toml`:

```shell
cargo add cel
```

Create and execute a simple CEL expression:

```rust
use cel::{Context, Program};

fn main() {
    let program = Program::compile("add(2, 3) == 5").unwrap();
    let mut context = Context::default();
    context.add_function("add", |a: i64, b: i64| a + b).unwrap();
    let value = program.execute(&context).unwrap();
    assert_eq!(value, true.into());
}
```

### Examples

Check out these other examples to learn how to use this library:

- [Simple](./example/src/simple.rs) - A simple example of how to use the library.
- [Variables](./example/src/variables.rs) - Passing variables and using them in your program.
- [Functions](./example/src/functions.rs) - Defining and using custom functions in your program.
- [Concurrent Execution](./example/src/threads.rs) - Executing the same program concurrently.
- [Interrupting Evaluation](./example/src/interrupt.rs) - Cancelling an evaluation from a deadline or another thread, and capping the work it may do.

### Bounding evaluation

An evaluation can be cancelled cooperatively and bounded in the amount of work it does. Both are checked
while iterating comprehensions (`all`, `exists`, `map`, `filter`, ...), and neither can be swallowed by
`||`, `&&` or optional accessors: once tripped, the evaluation as a whole fails.

Cancellation is per evaluation, so the handle is set on the `Context`. Anything implementing
`cel::Interrupt` works: an `AtomicBool`, a `cel::Deadline`, or any `Fn() -> bool + Send + Sync` closure.

```rust
use cel::{Context, Deadline, ExecutionError, Program};
use std::time::Duration;

let program = Program::compile("list.all(x, list.exists(y, y > x))").unwrap();
let deadline = Deadline::after(Duration::from_millis(50));
let mut context = Context::default();
context.add_variable("list", (0..100_000).collect::<Vec<i64>>()).unwrap();
context.set_interrupt(&deadline);
assert_eq!(program.execute(&context), Err(ExecutionError::Interrupted));
```

The iteration budget is runtime policy, so it lives on the `Env`. It is off by default.

```rust
use cel::{Context, Env, ExecutionError, Program, RuntimeOptions};
use std::sync::Arc;

let mut env = Env::stdlib();
env.set_options(RuntimeOptions::default().with_max_iterations(10_000));
let context = Context::with_env(Arc::new(env));
let program = Program::compile("[1, 2, 3].all(x, x > 0)").unwrap();
assert_eq!(program.execute(&context), Ok(true.into()));
```

Steps (`with_max_steps`: nodes evaluated and functions called) and bytes (`with_max_bytes`: bytes
allocated for the values created) budgets bound the rest of what an expression does, and fail with
`ExecutionError::BudgetExceeded { kind, limit }`. `Context::set_budget` overrides the `Env`'s options for
one context, and `Program::execute_with_usage` reports what an evaluation used. Budget errors are never
absorbed by `||`, `&&`, comprehension macros or optional accessors.

```rust
use cel::{Context, Program, RuntimeOptions};

let mut context = Context::default();
context.set_budget(RuntimeOptions::default().with_max_steps(1_000).with_max_bytes(64 * 1024));
let program = Program::compile("[1, 2, 3].all(x, x > 0)").unwrap();
let (result, usage) = program.execute_with_usage(&context);
assert_eq!(result, Ok(true.into()));
assert_eq!(usage.iterations, 3);
```

Custom functions can poll the handle through `FunctionContext::is_interrupted()` to return early from
long-running work.

The standard library's `matches` compiles each pattern once: the `Env` keeps the compiled regexes in a
`cel::RegexCache` (64 patterns by default, see `Env::set_regex_cache_options`) shared by every evaluation
under it. Under a steps budget, only the call that compiles a pattern pays for parsing and compiling it;
later calls pay the lookup and the match, and a pattern is kept only once its compile was paid for. Other
evaluations can evict a kept pattern, so size budgets for compiling every pattern, unless the patterns are
pinned with `RegexCache::prewarm`. The cache keeps at most 64 MiB of automata by default
(`RegexCacheOptions::with_max_bytes`), plus the pinned ones (`RegexCache::pinned_bytes`). Both count a
pattern's automata for matching without a budget and the NFA it matches with under one. On top of that, each
kept regex keeps the matching scratch space of each thread matching it, which is not counted: up to ~2 MiB per
pattern and thread without a budget, and under one up to `RegexCacheOptions::with_budget_dfa_bytes` (256 KiB by
default) of lazy DFA states beyond the least the pattern needs (~20 KB for a small one, ~430 KB near 1 MiB).
Under a budget, building those states is charged as they are built, so a smaller cache costs steps, not
unpriced work.
