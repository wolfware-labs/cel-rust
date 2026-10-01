# Probe baseline at 4d469a8 (upstream master)

Bench: `cel/benches/probes.rs`. Release profile, criterion, `sample_size(10)`, `measurement_time(10s)`
(capped because the slow probes take 1+ s per iteration; criterion warns it cannot finish 10 samples
in 10 s for those and extends the run).

Command: `cargo bench -p cel --bench probes`

Machine: 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz, 16 logical CPUs (8 cores), Linux 7.0.0-31-generic.
Toolchain: rustc 1.97.1 (8bab26f4f 2026-07-14).

Inputs: `request` = map with `query` (128 string entries) and `headers` (100 strings of 4 KiB, ~400 KiB);
`l` = list `0..2000`; `o` = `Value::Opaque(OptionalValue::of(1))`. Variables are bound with
`Context::add_variable_from_value`, evaluated with `Program::execute`.

| Probe | Expression | Time (low / median / high) |
|---|---|---|
| `query_map_request` | `size(request.query.map(k, request)) > 0` | 76.9 ms / 78.1 ms / 79.2 ms |
| `query_map_request_x16` | `size(request.query.map(k, [request, ... x16])) > 0` | 1.245 s / 1.364 s / 1.470 s |
| `optional_clone` | `[1].map(x, o)[0].hasValue()` | 1.45 us / 1.50 us / 1.58 us |
| `nested_all_4m` | `size(l.all(x, l.all(y, true)) ? [1] : []) > 0` | 1.362 s / 1.441 s / 1.529 s |

Notes:
- `optional_clone` returns `Err(NoSuchOverload)` at this commit (the optional-clone bug); the bench only
  black-boxes the result, so its time is that of the failing path. After the fix it measures the success path.
- `nested_all_4m` does 4 M iterations with no way to stop it; this is the case the budget `Frame` bounds.
- An earlier, noisier first run (cold machine, build just finished) gave 142 ms / 2.70 s / 3.3 us / 1.49 s.
