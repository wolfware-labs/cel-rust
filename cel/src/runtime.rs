//! Per-evaluation runtime state: cooperative interruption and iteration budgets.
//!
//! Evaluation of a CEL expression is bounded by two independent mechanisms:
//!
//! - an [`Interrupt`] handle, set per evaluation on the [`Context`](crate::Context)
//!   with [`Context::set_interrupt`](crate::Context::set_interrupt), that the
//!   interpreter polls while iterating comprehensions (`all`, `exists`, `map`,
//!   `filter`, ...);
//! - an iteration budget, configured on the [`Env`](crate::Env) through
//!   [`RuntimeOptions`], that caps the total number of comprehension iterations
//!   performed by a single evaluation, nested comprehensions included.
//!
//! When either trips, evaluation aborts with
//! [`ExecutionError::Interrupted`](crate::ExecutionError::Interrupted) or
//! [`ExecutionError::IterationBudgetExceeded`](crate::ExecutionError::IterationBudgetExceeded).
//! Unlike ordinary CEL errors, these are never absorbed by `||`, `&&`, or
//! optional accessors: once tripped, the evaluation as a whole fails.

use crate::common::value::{BuiltinRef, Val};
use crate::ExecutionError;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// A cancellation signal consulted by the interpreter during evaluation.
///
/// Implement this for whatever drives cancellation in your application: a
/// flag flipped from another thread, a deadline, a runtime's cancellation
/// token. The interpreter polls it while iterating comprehensions, and custom
/// functions can poll it through
/// [`FunctionContext::is_interrupted`](crate::FunctionContext::is_interrupted).
///
/// Implementations are provided for [`AtomicBool`], [`Deadline`], and any
/// `Fn() -> bool + Send + Sync` closure.
///
/// # Example
/// ```
/// use cel::{Context, Deadline, ExecutionError, Program};
/// use std::time::Duration;
///
/// let program = Program::compile("[1, 2, 3].all(x, x > 0)").unwrap();
/// let deadline = Deadline::after(Duration::from_millis(50));
/// let mut ctx = Context::default();
/// ctx.set_interrupt(&deadline);
/// assert_eq!(program.execute(&ctx), Ok(true.into()));
/// ```
pub trait Interrupt: Send + Sync {
    /// Returns `true` once evaluation should stop.
    fn is_interrupted(&self) -> bool;
}

impl Interrupt for AtomicBool {
    fn is_interrupted(&self) -> bool {
        self.load(Ordering::Relaxed)
    }
}

impl<F> Interrupt for F
where
    F: Fn() -> bool + Send + Sync,
{
    fn is_interrupted(&self) -> bool {
        self()
    }
}

/// An [`Interrupt`] that trips once a point in time has passed.
///
/// # Example
/// ```
/// use cel::Deadline;
/// use std::time::Duration;
///
/// let deadline = Deadline::after(Duration::from_secs(1));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadline(Instant);

impl Deadline {
    /// A deadline that trips at the given instant.
    pub fn at(instant: Instant) -> Self {
        Deadline(instant)
    }

    /// A deadline that trips once `timeout` has elapsed from now.
    ///
    /// Saturates at the far future if `Instant::now() + timeout` would overflow.
    pub fn after(timeout: Duration) -> Self {
        let now = Instant::now();
        Deadline(
            now.checked_add(timeout)
                .unwrap_or(now + Duration::from_secs(u32::MAX as u64)),
        )
    }

    /// The instant at which this deadline trips.
    pub fn instant(&self) -> Instant {
        self.0
    }
}

impl Interrupt for Deadline {
    fn is_interrupted(&self) -> bool {
        Instant::now() >= self.0
    }
}

/// Runtime policy shared by every evaluation performed under an [`Env`](crate::Env).
///
/// # Example
/// ```
/// use cel::{Context, Env, ExecutionError, Program, RuntimeOptions};
/// use std::sync::Arc;
///
/// let mut env = Env::stdlib();
/// env.set_options(RuntimeOptions::default().with_max_iterations(5));
/// let ctx = Context::with_env(Arc::new(env));
///
/// let program = Program::compile("[1, 2, 3].all(x, x > 0)").unwrap();
/// assert_eq!(program.execute(&ctx), Ok(true.into()));
///
/// let program = Program::compile("[1, 2, 3, 4, 5, 6].all(x, x > 0)").unwrap();
/// assert_eq!(
///     program.execute(&ctx),
///     Err(ExecutionError::IterationBudgetExceeded { limit: 5 })
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RuntimeOptions {
    max_iterations: u64,
    interrupt_check_frequency: u64,
    max_steps: u64,
    max_bytes: u64,
}

impl Default for RuntimeOptions {
    /// No budget of any kind, and the interrupt handle polled on every iteration.
    fn default() -> Self {
        RuntimeOptions {
            max_iterations: 0,
            interrupt_check_frequency: 1,
            max_steps: 0,
            max_bytes: 0,
        }
    }
}

impl RuntimeOptions {
    /// Caps the total number of comprehension iterations a single evaluation may
    /// perform, across all comprehensions in the expression, nested ones included.
    ///
    /// Zero, the default, means unlimited.
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Polls the [`Interrupt`] handle every `frequency` comprehension iterations.
    ///
    /// The default is one, polling on every iteration. Zero is clamped to one:
    /// setting a frequency never disables interruption.
    pub fn with_interrupt_check_frequency(mut self, frequency: u64) -> Self {
        self.interrupt_check_frequency = frequency.max(1);
        self
    }

    /// The iteration budget, zero meaning unlimited.
    pub fn max_iterations(&self) -> u64 {
        self.max_iterations
    }

    /// How many comprehension iterations run between polls of the interrupt handle.
    pub fn interrupt_check_frequency(&self) -> u64 {
        self.interrupt_check_frequency
    }

    /// Caps the number of evaluation steps a single evaluation may take.
    ///
    /// One step is charged for every expression node evaluated and for every
    /// function or overload dispatched. Operations whose cost grows with their
    /// operands are charged in proportion: `in` on a list and `==`/`!=` on
    /// lists or maps cost one step per element walked, and the string
    /// functions `contains`, `startsWith`, `endsWith` and `matches` one step
    /// per 64 bytes of the string searched. Comprehension iterations are
    /// charged through the nodes they evaluate, so a steps budget bounds them
    /// too, nested ones included.
    ///
    /// When exceeded, evaluation fails with
    /// [`ExecutionError::BudgetExceeded`](crate::ExecutionError::BudgetExceeded)
    /// of kind [`BudgetKind::Steps`]. Zero, the default, means unlimited.
    ///
    /// # Example
    /// ```
    /// use cel::{BudgetKind, Context, Env, ExecutionError, Program, RuntimeOptions};
    /// use std::sync::Arc;
    ///
    /// let mut env = Env::stdlib();
    /// env.set_options(RuntimeOptions::default().with_max_steps(100));
    /// let ctx = Context::with_env(Arc::new(env));
    ///
    /// let program = Program::compile("[1, 2, 3].all(x, x > 0)").unwrap();
    /// assert_eq!(program.execute(&ctx), Ok(true.into()));
    ///
    /// // ten outer iterations, each running ten inner ones
    /// let l = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]";
    /// let program = Program::compile(&format!("{l}.all(x, {l}.all(y, y >= 0))")).unwrap();
    /// assert_eq!(
    ///     program.execute(&ctx),
    ///     Err(ExecutionError::BudgetExceeded { kind: BudgetKind::Steps, limit: 100 })
    /// );
    /// ```
    pub fn with_max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// The steps budget, zero meaning unlimited.
    pub fn max_steps(&self) -> u64 {
        self.max_steps
    }

    /// Caps the number of bytes a single evaluation may allocate for the
    /// values it creates.
    ///
    /// Bytes are charged when a value is created (a string or bytes value its
    /// length, a list 16 bytes per element, a map 64 bytes per entry, scalars
    /// nothing), when a value is really copied (sharing a string, list or map
    /// is free), and when the result, or a custom function's argument, is
    /// converted into a [`Value`](crate::Value). The count is cumulative:
    /// it bounds the total allocated, not what is live at once.
    ///
    /// When exceeded, evaluation fails with
    /// [`ExecutionError::BudgetExceeded`](crate::ExecutionError::BudgetExceeded)
    /// of kind [`BudgetKind::Bytes`]. Zero, the default, means unlimited.
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// The bytes budget, zero meaning unlimited.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }
}

/// The resource a [`ExecutionError::BudgetExceeded`](crate::ExecutionError::BudgetExceeded)
/// error ran out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BudgetKind {
    /// Evaluation steps, see [`RuntimeOptions::with_max_steps`].
    Steps,
    /// Allocated bytes, see [`RuntimeOptions::with_max_bytes`].
    Bytes,
}

impl std::fmt::Display for BudgetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BudgetKind::Steps => "steps",
            BudgetKind::Bytes => "bytes",
        })
    }
}

const ABORT_NONE: u8 = 0;
const ABORT_BUDGET: u8 = 1;
const ABORT_INTERRUPTED: u8 = 2;
const ABORT_STEPS: u8 = 3;
const ABORT_BYTES: u8 = 4;

/// Per-evaluation state, created by the outermost evaluation entry point and
/// shared by every nested scope and re-entrant call underneath it.
///
/// This type is opaque: it is managed entirely by the interpreter and exposes
/// no public API. It is only nameable so that it may appear as a field of
/// [`Context::Child`](crate::Context::Child).
///
/// Counters are atomics rather than `Cell`s only so that `Context` stays `Sync`;
/// evaluation itself is single-threaded, so relaxed ordering is sufficient.
pub struct Frame<'a> {
    interrupt: Option<&'a dyn Interrupt>,
    max_iterations: u64,
    check_frequency: u64,
    iterations: AtomicU64,
    polls: AtomicU64,
    abort: AtomicU8,
    max_steps: u64,
    steps: AtomicU64,
    max_bytes: u64,
    bytes: AtomicU64,
}

impl<'a> Frame<'a> {
    pub(crate) fn new(options: &RuntimeOptions, interrupt: Option<&'a dyn Interrupt>) -> Self {
        Frame {
            interrupt,
            max_iterations: options.max_iterations,
            check_frequency: options.interrupt_check_frequency.max(1),
            iterations: AtomicU64::new(0),
            polls: AtomicU64::new(0),
            abort: AtomicU8::new(ABORT_NONE),
            max_steps: options.max_steps,
            steps: AtomicU64::new(0),
            max_bytes: options.max_bytes,
            bytes: AtomicU64::new(0),
        }
    }

    /// Accounts for one comprehension iteration: enforces the budget and polls
    /// the interrupt handle at the configured frequency.
    ///
    /// Once an abort has been recorded, every subsequent call fails immediately.
    #[inline]
    pub(crate) fn tick(&self) -> Result<(), ExecutionError> {
        if let Some(err) = self.abort_error() {
            return Err(err);
        }
        let iterations = self.iterations.fetch_add(1, Ordering::Relaxed) + 1;
        if self.max_iterations > 0 && iterations > self.max_iterations {
            self.record(ABORT_BUDGET);
            return Err(self.abort_error().expect("abort was just recorded"));
        }
        if let Some(interrupt) = self.interrupt {
            let polls = self.polls.fetch_add(1, Ordering::Relaxed) + 1;
            if polls % self.check_frequency == 0 && interrupt.is_interrupted() {
                self.record(ABORT_INTERRUPTED);
                return Err(ExecutionError::Interrupted);
            }
        }
        Ok(())
    }

    /// Charges `steps` evaluation steps against the steps budget.
    ///
    /// Fails once the budget is exceeded, and on every call after any abort
    /// has been recorded. Evaluation is single-threaded, so a relaxed load and
    /// store suffice: no read-modify-write is needed.
    #[inline(always)]
    pub(crate) fn charge_steps(&self, steps: u64) -> Result<(), ExecutionError> {
        let total = self.steps.load(Ordering::Relaxed).saturating_add(steps);
        self.steps.store(total, Ordering::Relaxed);
        if self.max_steps > 0 && total > self.max_steps {
            self.record(ABORT_STEPS);
        }
        match self.abort.load(Ordering::Relaxed) {
            ABORT_NONE => Ok(()),
            _ => Err(self.abort_error().expect("an abort is recorded")),
        }
    }

    /// Charges `bytes` allocated bytes against the bytes budget.
    ///
    /// Fails once the budget is exceeded, and on every call after any abort
    /// has been recorded.
    #[inline(always)]
    pub(crate) fn charge_bytes(&self, bytes: u64) -> Result<(), ExecutionError> {
        let total = self.bytes.load(Ordering::Relaxed).saturating_add(bytes);
        self.bytes.store(total, Ordering::Relaxed);
        if self.max_bytes > 0 && total > self.max_bytes {
            self.record(ABORT_BYTES);
        }
        match self.abort.load(Ordering::Relaxed) {
            ABORT_NONE => Ok(()),
            _ => Err(self.abort_error().expect("an abort is recorded")),
        }
    }

    /// Charges the bytes of creating `value`, see [`fresh_size`].
    #[inline(always)]
    pub(crate) fn charge_fresh(&self, value: &dyn Val) -> Result<(), ExecutionError> {
        self.charge_bytes(fresh_size(value))
    }

    /// Charges the bytes of copying `value`, see [`clone_size`].
    #[inline(always)]
    pub(crate) fn charge_clone(&self, value: &dyn Val) -> Result<(), ExecutionError> {
        self.charge_bytes(clone_size(value))
    }

    /// Charges the bytes of converting `value` into a [`Value`](crate::Value),
    /// see [`conversion_size`]. The walk stops once it passes the budget left,
    /// so a value sharing a large one many times is refused before it is
    /// converted, not after.
    pub(crate) fn charge_conversion(&self, value: &dyn Val) -> Result<(), ExecutionError> {
        let cap = match self.max_bytes {
            0 => u64::MAX,
            max => max.saturating_sub(self.bytes.load(Ordering::Relaxed)),
        };
        self.charge_bytes(conversion_size(value, cap))
    }

    /// Polls the interrupt handle directly, bypassing the frequency gate.
    ///
    /// Observing an interrupt is sticky: the evaluation will fail with
    /// [`ExecutionError::Interrupted`] even if the caller goes on to return a value.
    pub(crate) fn observe_interrupt(&self) -> bool {
        if self.abort.load(Ordering::Relaxed) == ABORT_INTERRUPTED {
            return true;
        }
        match self.interrupt {
            Some(interrupt) if interrupt.is_interrupted() => {
                self.record(ABORT_INTERRUPTED);
                true
            }
            _ => false,
        }
    }

    /// Replaces a result with the recorded abort, if any.
    ///
    /// This is the backstop guaranteeing an abort surfaces even if the error
    /// value was swallowed somewhere along the way.
    pub(crate) fn finish<T>(&self, result: Result<T, ExecutionError>) -> Result<T, ExecutionError> {
        match self.abort_error() {
            Some(err) => Err(err),
            None => result,
        }
    }

    /// Records an abort. An interrupt always wins over a budget overrun, since
    /// the embedder asked for it explicitly.
    fn record(&self, abort: u8) {
        if abort == ABORT_INTERRUPTED {
            self.abort.store(ABORT_INTERRUPTED, Ordering::Relaxed);
        } else {
            let _ = self.abort.compare_exchange(
                ABORT_NONE,
                abort,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }

    fn abort_error(&self) -> Option<ExecutionError> {
        match self.abort.load(Ordering::Relaxed) {
            ABORT_INTERRUPTED => Some(ExecutionError::Interrupted),
            ABORT_BUDGET => Some(ExecutionError::IterationBudgetExceeded {
                limit: self.max_iterations,
            }),
            ABORT_STEPS => Some(ExecutionError::BudgetExceeded {
                kind: BudgetKind::Steps,
                limit: self.max_steps,
            }),
            ABORT_BYTES => Some(ExecutionError::BudgetExceeded {
                kind: BudgetKind::Bytes,
                limit: self.max_bytes,
            }),
            _ => None,
        }
    }
}

/// Bytes charged per list element: the `Box<dyn Val>` it occupies.
const LIST_SLOT: u64 = 16;
/// Bytes charged per map entry: its key, its value's box and table overhead.
const MAP_SLOT: u64 = 64;
/// Bytes charged per list element when a list is converted into a [`Value`](crate::Value).
const VALUE_SLOT: u64 = std::mem::size_of::<crate::Value>() as u64;
/// Bytes charged for an optional converted into a [`Value`](crate::Value): its `Arc`.
const OPTIONAL_SLOT: u64 = 16;

/// The bytes a freshly created value allocated at its top level: a string or
/// bytes its length, a list [`LIST_SLOT`] per element, a map [`MAP_SLOT`] per
/// entry, a struct [`MAP_SLOT`] per field, scalars nothing. Elements were
/// charged when they were created themselves.
pub(crate) fn fresh_size(value: &dyn Val) -> u64 {
    match value.as_builtin() {
        BuiltinRef::String(s) => s.inner().len() as u64,
        BuiltinRef::Bytes(b) => b.inner().len() as u64,
        BuiltinRef::List(l) => l.inner().len() as u64 * LIST_SLOT,
        BuiltinRef::Map(m) => m.inner().len() as u64 * MAP_SLOT,
        #[cfg(feature = "structs")]
        BuiltinRef::Struct(s) => s.field_count() as u64 * MAP_SLOT,
        _ => 0,
    }
}

/// The bytes [`Val::clone_as_boxed`] copies for `value`: O(1).
///
/// Strings, bytes, lists, maps and optionals share their contents behind an
/// `Arc` (or borrow them), so a copy allocates nothing but its box, which is
/// not charged. A struct copies its field table, [`MAP_SLOT`] per field.
/// Values of other types are not charged: their cost is unknown.
pub(crate) fn clone_size(value: &dyn Val) -> u64 {
    match value.as_builtin() {
        #[cfg(feature = "structs")]
        BuiltinRef::Struct(s) => s.field_count() as u64 * MAP_SLOT,
        _ => 0,
    }
}

/// The bytes converting `value` into a [`Value`](crate::Value) allocates,
/// stopping once the count passes `cap`.
///
/// The conversion rebuilds every list and map ([`VALUE_SLOT`] per list
/// element, [`MAP_SLOT`] per map entry) and copies borrowed strings and bytes,
/// while shared ones are not copied. It visits a value shared `n` times `n`
/// times, which is why the walk is capped: it runs before the conversion.
pub(crate) fn conversion_size(value: &dyn Val, cap: u64) -> u64 {
    fn walk(value: &dyn Val, cap: u64, total: &mut u64) {
        if *total > cap {
            return;
        }
        match value.as_builtin() {
            BuiltinRef::String(s) if s.as_arc().is_none() => *total += s.inner().len() as u64,
            BuiltinRef::Bytes(b) if b.as_arc().is_none() => *total += b.inner().len() as u64,
            BuiltinRef::List(l) => {
                *total += l.inner().len() as u64 * VALUE_SLOT;
                for item in l.inner() {
                    walk(item.as_ref(), cap, total);
                    if *total > cap {
                        return;
                    }
                }
            }
            BuiltinRef::Map(m) => {
                *total += m.inner().len() as u64 * MAP_SLOT;
                for item in m.inner().values() {
                    walk(item.as_ref(), cap, total);
                    if *total > cap {
                        return;
                    }
                }
            }
            BuiltinRef::Optional(o) => {
                *total += OPTIONAL_SLOT;
                if let Some(inner) = o.option() {
                    walk(inner, cap, total);
                }
            }
            #[cfg(feature = "structs")]
            BuiltinRef::Struct(s) => {
                *total += s.field_count() as u64 * MAP_SLOT;
                for item in s.fields() {
                    walk(item, cap, total);
                    if *total > cap {
                        return;
                    }
                }
            }
            _ => {}
        }
    }
    let mut total = 0;
    walk(value, cap, &mut total);
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, Env, ExecutionError, FunctionContext, Program, ResolveResult, Value};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn budgeted(max_iterations: u64) -> Context<'static, 'static> {
        let mut env = Env::stdlib();
        env.set_options(RuntimeOptions::default().with_max_iterations(max_iterations));
        Context::with_env(Arc::new(env))
    }

    fn run(ctx: &Context, src: &str) -> ResolveResult {
        Program::compile(src).expect("must parse").execute(ctx)
    }

    fn run_optional(ctx: &Context, src: &str) -> ResolveResult {
        let expr = crate::parser::Parser::new()
            .enable_optional_syntax(true)
            .parse(src)
            .expect("must parse");
        ctx.resolve(&expr)
    }

    fn budget_exceeded(limit: u64) -> ResolveResult {
        Err(ExecutionError::IterationBudgetExceeded { limit })
    }

    fn with_options(options: RuntimeOptions) -> Context<'static, 'static> {
        let mut env = Env::stdlib();
        env.set_options(options);
        Context::with_env(Arc::new(env))
    }

    fn steps_budget(max_steps: u64) -> Context<'static, 'static> {
        with_options(RuntimeOptions::default().with_max_steps(max_steps))
    }

    fn steps_exceeded(limit: u64) -> ResolveResult {
        Err(ExecutionError::BudgetExceeded {
            kind: BudgetKind::Steps,
            limit,
        })
    }

    #[test]
    fn steps_are_off_by_default() {
        let options = RuntimeOptions::default();
        assert_eq!(options.max_steps(), 0);
        assert_eq!(options.max_bytes(), 0);
    }

    #[test]
    fn steps_budget_trips_at_limit() {
        // `1 + 1` is three nodes: the call and its two operands.
        let ctx = steps_budget(3);
        assert_eq!(run(&ctx, "1 + 1"), Ok(2.into()));
        let ctx = steps_budget(2);
        assert_eq!(run(&ctx, "1 + 1"), steps_exceeded(2));
    }

    #[test]
    fn steps_budget_charges_function_dispatch() {
        // `size('a')`: two nodes plus one dispatch.
        let ctx = steps_budget(3);
        assert_eq!(run(&ctx, "size('a')"), Ok(1.into()));
        let ctx = steps_budget(2);
        assert_eq!(run(&ctx, "size('a')"), steps_exceeded(2));
    }

    #[test]
    fn steps_budget_bounds_comprehensions_without_an_iteration_budget() {
        let mut ctx = steps_budget(1_000);
        ctx.add_variable("l", (0..2_000i64).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(run(&ctx, "l.all(x, l.all(y, true))"), steps_exceeded(1_000));
        assert_eq!(run(&ctx, "size(l.map(x, x)) > 0"), steps_exceeded(1_000));
        assert_eq!(
            run(&ctx, "size(l.filter(x, true)) > 0"),
            steps_exceeded(1_000)
        );
    }

    #[test]
    fn steps_budget_charges_data_proportional_work() {
        // Each of these is a handful of nodes, but walks a large operand.
        let mut ctx = steps_budget(500);
        ctx.add_variable("l", (0..1_000i64).collect::<Vec<_>>())
            .unwrap();
        ctx.add_variable("s", "x".repeat(64 * 1_000)).unwrap();
        assert_eq!(run(&ctx, "-1 in l"), steps_exceeded(500));
        assert_eq!(run(&ctx, "l == l"), steps_exceeded(500));
        assert_eq!(run(&ctx, "l != l"), steps_exceeded(500));
        assert_eq!(run(&ctx, "s.contains('y')"), steps_exceeded(500));
        assert_eq!(run(&ctx, "s.startsWith('y')"), steps_exceeded(500));
        assert_eq!(run(&ctx, "s.endsWith('y')"), steps_exceeded(500));
        // and the same expressions fit a budget sized for them
        let mut ctx = steps_budget(1_100);
        ctx.add_variable("l", (0..1_000i64).collect::<Vec<_>>())
            .unwrap();
        ctx.add_variable("s", "x".repeat(64 * 1_000)).unwrap();
        assert_eq!(run(&ctx, "-1 in l"), Ok(false.into()));
        assert_eq!(run(&ctx, "s.contains('y')"), Ok(false.into()));
    }

    #[cfg(feature = "regex")]
    #[test]
    fn steps_budget_charges_matches() {
        let mut ctx = steps_budget(500);
        ctx.add_variable("s", "x".repeat(64 * 1_000)).unwrap();
        assert_eq!(run(&ctx, "s.matches('y')"), steps_exceeded(500));
    }

    fn bytes_budget(max_bytes: u64) -> Context<'static, 'static> {
        let mut ctx = with_options(RuntimeOptions::default().with_max_bytes(max_bytes));
        ctx.add_variable("s", "x".repeat(1_000)).unwrap();
        ctx.add_variable("l", (0..1_000i64).collect::<Vec<_>>())
            .unwrap();
        ctx
    }

    fn bytes_exceeded(limit: u64) -> ResolveResult {
        Err(ExecutionError::BudgetExceeded {
            kind: BudgetKind::Bytes,
            limit,
        })
    }

    #[test]
    fn bytes_budget_charges_created_strings() {
        // the concatenation allocates 2000 bytes
        assert_eq!(run(&bytes_budget(2_100), "size(s + s)"), Ok(2_000.into()));
        assert_eq!(
            run(&bytes_budget(1_900), "size(s + s)"),
            bytes_exceeded(1_900)
        );
    }

    #[test]
    fn bytes_budget_charges_created_lists_and_maps() {
        // 1000 + 1000 elements of 16 bytes
        assert_eq!(run(&bytes_budget(33_000), "size(l + l)"), Ok(2_000.into()));
        assert_eq!(
            run(&bytes_budget(31_000), "size(l + l)"),
            bytes_exceeded(31_000)
        );
        // a list literal: 16 bytes per element
        assert_eq!(
            run(&bytes_budget(100), "size([1, 2, 3, 4, 5, 6, 7, 8])"),
            bytes_exceeded(100)
        );
        assert_eq!(
            run(&bytes_budget(200), "size([1, 2, 3, 4, 5, 6, 7, 8])"),
            Ok(8.into())
        );
        // a map literal: 64 bytes per entry
        assert_eq!(
            run(&bytes_budget(100), "size({1: 1, 2: 2})"),
            bytes_exceeded(100)
        );
        assert_eq!(run(&bytes_budget(200), "size({1: 1, 2: 2})"), Ok(2.into()));
    }

    #[test]
    fn bytes_budget_charges_map_and_filter_results() {
        // one `[x]` of 16 bytes per element
        assert_eq!(
            run(&bytes_budget(10_000), "size(l.map(x, x))"),
            bytes_exceeded(10_000)
        );
        assert_eq!(
            run(&bytes_budget(10_000), "size(l.filter(x, true))"),
            bytes_exceeded(10_000)
        );
        assert_eq!(
            run(&bytes_budget(20_000), "size(l.map(x, x))"),
            Ok(1_000.into())
        );
    }

    #[test]
    fn bytes_budget_does_not_charge_sharing() {
        // `[l, l, l]` shares `l`: three slots, no copy of its elements
        assert_eq!(run(&bytes_budget(100), "size([l, l, l])"), Ok(3.into()));
        assert_eq!(run(&bytes_budget(100), "size([s, s, s])"), Ok(3.into()));
    }

    #[test]
    fn bytes_budget_charges_the_result_conversion() {
        // returning `l` builds a `Value` list of 1000 elements
        assert_eq!(run(&bytes_budget(1_000), "l"), bytes_exceeded(1_000));
        // a result sharing `l` a thousand times would build a million values:
        // the conversion is refused before it is made
        let started = std::time::Instant::now();
        assert_eq!(
            run(&bytes_budget(1_000_000), "l.map(x, l)"),
            bytes_exceeded(1_000_000)
        );
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn bytes_budget_charges_function_results_and_arguments() {
        fn big(ftx: &FunctionContext) -> ResolveResult {
            let _ = ftx;
            Ok(Value::String(Arc::new("y".repeat(5_000))))
        }
        fn length(v: Value) -> i64 {
            match v {
                Value::List(l) => l.len() as i64,
                _ => -1,
            }
        }
        let mut ctx = bytes_budget(4_000);
        ctx.add_function("big", big).unwrap();
        ctx.add_function("length", length).unwrap();
        assert_eq!(run(&ctx, "size(big())"), bytes_exceeded(4_000));
        // converting `l` into a `Value` argument builds a 1000 element list
        assert_eq!(run(&ctx, "length(l)"), bytes_exceeded(4_000));
        assert_eq!(run(&ctx, "length([1])"), Ok(1.into()));
    }

    #[test]
    fn budget_exceeded_displays_its_kind() {
        let err = ExecutionError::BudgetExceeded {
            kind: BudgetKind::Steps,
            limit: 7,
        };
        assert_eq!(err.to_string(), "steps budget of 7 exceeded");
        let err = ExecutionError::BudgetExceeded {
            kind: BudgetKind::Bytes,
            limit: 7,
        };
        assert_eq!(err.to_string(), "bytes budget of 7 exceeded");
    }

    #[test]
    fn budget_off_by_default() {
        let ctx = Context::default();
        assert_eq!(ctx.env().options().max_iterations(), 0);
        assert_eq!(ctx.env().options().interrupt_check_frequency(), 1);
        let list: Vec<i64> = (0..20_000).collect();
        let mut ctx = Context::default();
        ctx.add_variable("list", list).unwrap();
        assert_eq!(run(&ctx, "list.all(x, x >= 0)"), Ok(true.into()));
    }

    #[test]
    fn zero_frequency_clamps_to_one() {
        let options = RuntimeOptions::default().with_interrupt_check_frequency(0);
        assert_eq!(options.interrupt_check_frequency(), 1);
    }

    #[test]
    fn budget_trips_at_limit() {
        let ctx = budgeted(3);
        assert_eq!(run(&ctx, "[1, 2, 3].all(x, x > 0)"), Ok(true.into()));
        assert_eq!(run(&ctx, "[1, 2, 3, 4].all(x, x > 0)"), budget_exceeded(3));
    }

    #[test]
    fn map_and_filter_are_budgeted() {
        // `map` and `filter` run through a dedicated append loop, which must
        // spend the budget like any other comprehension.
        let ctx = budgeted(6);
        assert_eq!(
            run(&ctx, "[1, 2, 3, 4, 5, 6].map(x, x)"),
            Ok(vec![1, 2, 3, 4, 5, 6].into())
        );
        let ctx = budgeted(5);
        assert_eq!(
            run(&ctx, "[1, 2, 3, 4, 5, 6].map(x, x)"),
            budget_exceeded(5)
        );
        assert_eq!(
            run(&ctx, "[1, 2, 3, 4, 5, 6].filter(x, x > 0)"),
            budget_exceeded(5)
        );
        assert_eq!(
            run(&ctx, "[1, 2, 3, 4, 5, 6].map(x, x > 0, x)"),
            budget_exceeded(5)
        );
    }

    #[test]
    fn map_and_filter_are_interruptible() {
        let interrupt = || true;
        let mut ctx = Context::default();
        ctx.set_interrupt(&interrupt);
        assert_eq!(
            run(&ctx, "[1, 2].map(x, x)"),
            Err(ExecutionError::Interrupted)
        );
        assert_eq!(
            run(&ctx, "[1, 2].filter(x, true)"),
            Err(ExecutionError::Interrupted)
        );
    }

    #[test]
    fn budget_is_global_across_nested_comprehensions() {
        // 2 outer + 2 * 2 inner = 6 iterations
        let ctx = budgeted(6);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, [1, 2].all(y, y > 0))"),
            Ok(true.into())
        );
        let ctx = budgeted(5);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, [1, 2].all(y, y > 0))"),
            budget_exceeded(5)
        );
    }

    #[test]
    fn budget_is_global_across_sibling_comprehensions() {
        let ctx = budgeted(4);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0) && [1, 2].all(x, x > 0)"),
            Ok(true.into())
        );
        let ctx = budgeted(3);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0) && [1, 2].all(x, x > 0)"),
            budget_exceeded(3)
        );
    }

    #[test]
    fn budget_is_per_evaluation() {
        let ctx = budgeted(3);
        for _ in 0..5 {
            assert_eq!(run(&ctx, "[1, 2, 3].all(x, x > 0)"), Ok(true.into()));
        }
    }

    #[test]
    fn budget_is_not_absorbed_by_logical_or() {
        let ctx = budgeted(1);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0) || true"),
            budget_exceeded(1)
        );
        assert_eq!(
            run(&ctx, "false || [1, 2].all(x, x > 0)"),
            budget_exceeded(1)
        );
    }

    #[test]
    fn budget_is_not_absorbed_by_logical_and() {
        let ctx = budgeted(1);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0) && false"),
            budget_exceeded(1)
        );
        assert_eq!(
            run(&ctx, "true && [1, 2].all(x, x > 0)"),
            budget_exceeded(1)
        );
    }

    #[test]
    fn budget_is_not_absorbed_by_all_macro_predicate() {
        // `all` wraps its predicate in @not_strictly_false, which turns errors into `true`.
        let ctx = budgeted(2);
        assert_eq!(
            run(&ctx, "[1].all(x, [1, 2].all(y, y > 0))"),
            budget_exceeded(2)
        );
    }

    #[test]
    fn budget_is_not_absorbed_by_optional_accessors() {
        let ctx = budgeted(1);
        assert_eq!(
            run_optional(&ctx, "{'a': 1}[?'a'].orValue([1, 2].all(x, x > 0))"),
            budget_exceeded(1)
        );
        assert_eq!(
            run_optional(&ctx, "[[1, 2].all(x, x > 0)][?0].hasValue()"),
            budget_exceeded(1)
        );
        assert_eq!(
            run_optional(&ctx, "{'a': [1, 2].all(x, x > 0)}.?a.hasValue()"),
            budget_exceeded(1)
        );
    }

    #[test]
    fn budget_is_not_absorbed_by_ternary() {
        let ctx = budgeted(1);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0) ? 1 : 2"),
            budget_exceeded(1)
        );
    }

    #[test]
    fn budget_is_shared_with_reentrant_function_calls() {
        fn reenter(ftx: &FunctionContext) -> ResolveResult {
            let program = Program::compile("[1, 2, 3].all(x, x > 0)").unwrap();
            program.execute(ftx.ptx)
        }
        let mut ctx = budgeted(4);
        ctx.add_function("reenter", reenter).unwrap();
        // 2 outer iterations, each re-entering for 3 more: 8 > 4.
        assert_eq!(run(&ctx, "[1, 2].all(x, reenter())"), budget_exceeded(4));
        // A single nested execute fits: 1 + 3 = 4.
        assert_eq!(run(&ctx, "[1].all(x, reenter())"), Ok(true.into()));
    }

    #[test]
    fn nested_execute_cannot_discard_the_abort() {
        fn swallow(ftx: &FunctionContext) -> ResolveResult {
            let program = Program::compile("[1, 2, 3].all(x, x > 0)").unwrap();
            let _ = program.execute(ftx.ptx);
            Ok(Value::Bool(true))
        }
        let mut ctx = budgeted(2);
        ctx.add_function("swallow", swallow).unwrap();
        assert_eq!(run(&ctx, "swallow()"), budget_exceeded(2));
    }

    #[test]
    fn context_resolve_is_budgeted() {
        let ctx = budgeted(1);
        let program = Program::compile("[1, 2].all(x, x > 0)").unwrap();
        assert_eq!(ctx.resolve(program.expression()), budget_exceeded(1));
    }

    #[test]
    fn atomic_bool_interrupts_from_another_thread() {
        let flag = AtomicBool::new(false);
        let list: Vec<i64> = (0..1_000_000).collect();
        let mut ctx = Context::default();
        ctx.add_variable("list", list).unwrap();
        ctx.set_interrupt(&flag);
        // Each iteration runs another `all` over 1000 elements, so a full run takes seconds.
        let program = Program::compile("list.all(x, list.exists(y, y > x + 1000))").unwrap();
        let result = std::thread::scope(|scope| {
            let flag = &flag;
            scope.spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                flag.store(true, Ordering::Relaxed);
            });
            program.execute(&ctx)
        });
        assert_eq!(result, Err(ExecutionError::Interrupted));
    }

    #[test]
    fn deadline_interrupts() {
        let deadline = Deadline::after(Duration::from_millis(10));
        let list: Vec<i64> = (0..1_000_000).collect();
        let mut ctx = Context::default();
        ctx.add_variable("list", list).unwrap();
        ctx.set_interrupt(&deadline);
        let program = Program::compile("list.all(x, list.exists(y, y > x + 1000))").unwrap();
        assert_eq!(program.execute(&ctx), Err(ExecutionError::Interrupted));
    }

    #[test]
    fn closure_interrupts() {
        let interrupt = || true;
        let mut ctx = Context::default();
        ctx.set_interrupt(&interrupt);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0)"),
            Err(ExecutionError::Interrupted)
        );
        // No comprehension, nothing polled: the expression completes.
        assert_eq!(run(&ctx, "1 + 1"), Ok(2.into()));
    }

    #[test]
    fn interrupt_set_on_a_child_scope_is_honoured() {
        let interrupt = || true;
        let root = Context::default();
        let mut child = root.new_inner_scope();
        child.set_interrupt(&interrupt);
        assert_eq!(
            run(&child, "[1, 2].all(x, x > 0)"),
            Err(ExecutionError::Interrupted)
        );
    }

    #[test]
    fn interrupt_is_not_absorbed_by_logical_or() {
        let interrupt = || true;
        let mut ctx = Context::default();
        ctx.set_interrupt(&interrupt);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0) || true"),
            Err(ExecutionError::Interrupted)
        );
    }

    #[test]
    fn interrupt_check_frequency_gates_polling() {
        let polls = AtomicU64::new(0);
        let interrupt = || {
            polls.fetch_add(1, Ordering::Relaxed);
            false
        };
        let mut env = Env::stdlib();
        env.set_options(RuntimeOptions::default().with_interrupt_check_frequency(10));
        let mut ctx = Context::with_env(Arc::new(env));
        ctx.set_interrupt(&interrupt);
        let list: Vec<i64> = (0..100).collect();
        ctx.add_variable("list", list).unwrap();
        assert_eq!(run(&ctx, "list.all(x, x >= 0)"), Ok(true.into()));
        assert_eq!(polls.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn function_observing_interrupt_is_sticky() {
        fn observe(ftx: &FunctionContext) -> ResolveResult {
            assert!(ftx.is_interrupted());
            Ok(Value::Bool(false))
        }
        let interrupt = || true;
        let mut ctx = Context::default();
        ctx.set_interrupt(&interrupt);
        ctx.add_function("observe", observe).unwrap();
        assert_eq!(run(&ctx, "observe()"), Err(ExecutionError::Interrupted));
    }

    #[test]
    fn function_without_interrupt_is_not_interrupted() {
        fn observe(ftx: &FunctionContext) -> ResolveResult {
            Ok(Value::Bool(ftx.is_interrupted()))
        }
        let mut ctx = Context::default();
        ctx.add_function("observe", observe).unwrap();
        assert_eq!(run(&ctx, "observe()"), Ok(false.into()));
    }

    #[test]
    fn interrupt_wins_over_budget() {
        let interrupt = || true;
        let mut ctx = budgeted(1);
        ctx.set_interrupt(&interrupt);
        assert_eq!(
            run(&ctx, "[1, 2].all(x, x > 0)"),
            Err(ExecutionError::Interrupted)
        );
    }

    #[test]
    fn frame_records_first_budget_only() {
        let options = RuntimeOptions::default().with_max_iterations(1);
        let frame = Frame::new(&options, None);
        assert!(frame.tick().is_ok());
        assert_eq!(
            frame.tick(),
            Err(ExecutionError::IterationBudgetExceeded { limit: 1 })
        );
        // Sticky: still failing, and finish() overrides a value.
        assert_eq!(
            frame.tick(),
            Err(ExecutionError::IterationBudgetExceeded { limit: 1 })
        );
        assert_eq!(
            frame.finish(Ok(())),
            Err(ExecutionError::IterationBudgetExceeded { limit: 1 })
        );
    }

    #[test]
    fn every_scope_carries_the_frame_itself() {
        // Looking the frame up must not walk the parent chain: every scope
        // under the frame scope holds a reference to it.
        let root = Context::default();
        let frame = root.new_frame();
        let scope = root.new_frame_scope(&frame);
        let a = scope.new_inner_scope();
        let b = a.new_inner_scope();
        let c = b.new_inner_scope();
        for ctx in [&scope, &a, &b, &c] {
            match ctx {
                Context::Child { frame: Some(f), .. } => assert!(std::ptr::eq(*f, &frame)),
                _ => panic!("scope lost the frame"),
            }
        }
    }

    #[test]
    fn context_stays_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Context>();
    }
}
