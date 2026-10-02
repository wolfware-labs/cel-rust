//! Per-evaluation runtime state: cooperative interruption and budgets.
//!
//! Evaluation of a CEL expression is bounded by independent mechanisms:
//!
//! - an [`Interrupt`] handle, set per evaluation on the [`Context`](crate::Context)
//!   with [`Context::set_interrupt`](crate::Context::set_interrupt), that the
//!   interpreter polls while iterating comprehensions (`all`, `exists`, `map`,
//!   `filter`, ...);
//! - budgets, set through [`RuntimeOptions`] on the [`Env`](crate::Env) or,
//!   per context, with [`Context::set_budget`](crate::Context::set_budget):
//!   - an iteration budget capping the comprehension iterations of a single
//!     evaluation, nested comprehensions included;
//!   - a steps budget capping the expression nodes evaluated and functions
//!     dispatched, with data-proportional charges for operations that walk
//!     their operands;
//!   - a bytes budget capping the bytes allocated for the values created.
//!
//! When one trips, evaluation aborts with
//! [`ExecutionError::Interrupted`](crate::ExecutionError::Interrupted),
//! [`ExecutionError::IterationBudgetExceeded`](crate::ExecutionError::IterationBudgetExceeded) or
//! [`ExecutionError::BudgetExceeded`](crate::ExecutionError::BudgetExceeded).
//! Unlike ordinary CEL errors, these are never absorbed by `||`, `&&`,
//! comprehension macros or optional accessors: once tripped, the evaluation as
//! a whole fails.
//!
//! An evaluation with no budget and no interrupt handle does none of this
//! bookkeeping.

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
    max_regex_len: u64,
    regex_size_limit: u64,
}

impl Default for RuntimeOptions {
    /// No budget of any kind, and the interrupt handle polled on every iteration.
    fn default() -> Self {
        RuntimeOptions {
            max_iterations: 0,
            // zero: not set, which reads as polling on every iteration
            interrupt_check_frequency: 0,
            max_steps: 0,
            max_bytes: 0,
            max_regex_len: 0,
            regex_size_limit: 0,
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
        self.interrupt_check_frequency.max(1)
    }

    /// Caps the number of evaluation steps a single evaluation may take.
    ///
    /// One step is charged for every expression node evaluated and for every
    /// function or overload dispatched. Operations whose cost grows with their
    /// operands are charged in proportion: `in` on a list and `==`/`!=` on
    /// lists or maps cost one step per element walked, the string
    /// functions `contains`, `startsWith` and `endsWith` one step per 64
    /// bytes of the string searched, and the standard library's `matches`
    /// by its pattern (parsing, translating and compiling it, priced before
    /// each runs, and paid only by the call that compiles it: the
    /// [`Env`](crate::Env)'s [`RegexCache`](crate::RegexCache) keeps it for
    /// later calls) and by the size of its compiled automaton times the
    /// subject's length, charged before matching. Other evaluations sharing
    /// the Env can evict a kept pattern: size the budget for compiling every
    /// pattern the expression matches, unless they are pinned with
    /// [`RegexCache::prewarm`](crate::RegexCache::prewarm).
    /// Comparisons (`==`, `!=`, `in`) are charged by a walk of their
    /// operands, one step per element and per 64 bytes of string. Creating a
    /// value costs one step per element or entry, or per 64 bytes of a string
    /// (so concatenation and conversions are priced by their result), and so
    /// does converting the result, or a custom function's argument, into a
    /// [`Value`](crate::Value). Comprehension iterations are
    /// charged through the nodes they evaluate, so a steps budget bounds them
    /// too, nested ones included.
    ///
    /// A function you add, with [`Context::add_function`](crate::Context::add_function),
    /// [`Env::add_overload`](crate::Env::add_overload) or
    /// [`Env::add_member_overload`](crate::Env::add_member_overload), is
    /// charged one dispatch step and the conversion of its arguments and
    /// result: the budget does not see the work it does, so it must bound
    /// that itself. This holds for a function of your own named `matches`
    /// too: only the standard library's `string.matches(string)` is priced
    /// by its pattern, and limited by [`with_max_regex_len`](Self::with_max_regex_len)
    /// and [`with_regex_size_limit`](Self::with_regex_size_limit).
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

    /// Caps the length, in bytes, of a regular expression pattern passed to
    /// the standard library's `matches`. Longer patterns fail with an
    /// [`ExecutionError::FunctionError`](crate::ExecutionError::FunctionError)
    /// before they are compiled. A function of your own named `matches`
    /// is not limited.
    ///
    /// Zero, the default, means unlimited.
    pub fn with_max_regex_len(mut self, max_regex_len: u64) -> Self {
        self.max_regex_len = max_regex_len;
        self
    }

    /// The regex pattern length limit, zero meaning unlimited.
    pub fn max_regex_len(&self) -> u64 {
        self.max_regex_len
    }

    /// Caps the size, in bytes, of the automaton `matches` compiles a
    /// pattern into, and of its lazy DFA cache, while a frame is enforcing a
    /// budget. A pattern whose automaton would be larger fails `matches` with
    /// an [`ExecutionError::FunctionError`](crate::ExecutionError::FunctionError),
    /// after at most the work of building that much automaton.
    ///
    /// Short patterns can compile to large automata: `\w{30}` is over 1 MiB
    /// with Unicode word characters, and compiling 1 MiB takes about 10 ms.
    /// Zero, the default, keeps the `regex` crate's limits (10 MiB, with a
    /// 2 MiB cache).
    ///
    /// The [`RegexCache`](crate::RegexCache) keeps a pattern under the limit
    /// it was compiled with: it is never served to an evaluation with
    /// another limit, which compiles it again.
    pub fn with_regex_size_limit(mut self, bytes: u64) -> Self {
        self.regex_size_limit = bytes;
        self
    }

    /// The compiled regex size limit, zero meaning the `regex` crate's default.
    pub fn regex_size_limit(&self) -> u64 {
        self.regex_size_limit
    }

    /// These options laid over `base`: every field these options leave unset
    /// (zero, the default) takes the value it has in `base`.
    pub(crate) fn over(&self, base: &RuntimeOptions) -> RuntimeOptions {
        let pick = |own: u64, inherited: u64| if own == 0 { inherited } else { own };
        RuntimeOptions {
            max_iterations: pick(self.max_iterations, base.max_iterations),
            interrupt_check_frequency: pick(
                self.interrupt_check_frequency,
                base.interrupt_check_frequency,
            ),
            max_steps: pick(self.max_steps, base.max_steps),
            max_bytes: pick(self.max_bytes, base.max_bytes),
            max_regex_len: pick(self.max_regex_len, base.max_regex_len),
            regex_size_limit: pick(self.regex_size_limit, base.regex_size_limit),
        }
    }

    /// Whether these options set no budget of any kind.
    pub(crate) fn is_unbounded(&self) -> bool {
        self.max_iterations == 0
            && self.max_steps == 0
            && self.max_bytes == 0
            && self.max_regex_len == 0
            && self.regex_size_limit == 0
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

/// The resources one evaluation used, as reported by
/// [`Program::execute_with_usage`](crate::Program::execute_with_usage) and
/// [`Value::resolve_with_usage`](crate::Value::resolve_with_usage).
///
/// The counts are those the budgets of [`RuntimeOptions`] are enforced
/// against, whether or not a budget is set. When a budget is exceeded they
/// include the charge that exceeded it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EvalUsage {
    /// Evaluation steps, see [`RuntimeOptions::with_max_steps`].
    pub steps: u64,
    /// Allocated bytes, see [`RuntimeOptions::with_max_bytes`].
    pub bytes: u64,
    /// Comprehension iterations, see [`RuntimeOptions::with_max_iterations`].
    pub iterations: u64,
}

impl EvalUsage {
    /// The usage accrued since `earlier`, a snapshot of the same frame.
    pub(crate) fn since(&self, earlier: &EvalUsage) -> EvalUsage {
        EvalUsage {
            steps: self.steps.saturating_sub(earlier.steps),
            bytes: self.bytes.saturating_sub(earlier.bytes),
            iterations: self.iterations.saturating_sub(earlier.iterations),
        }
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
    #[cfg_attr(not(feature = "regex"), allow(dead_code))]
    max_regex_len: u64,
    #[cfg_attr(not(feature = "regex"), allow(dead_code))]
    regex_size_limit: u64,
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
            max_regex_len: options.max_regex_len,
            regex_size_limit: options.regex_size_limit,
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

    /// The steps left before the steps budget is exceeded, `u64::MAX` when
    /// there is none: the cap for walks charged before the work they price.
    pub(crate) fn steps_left(&self) -> u64 {
        match self.max_steps {
            0 => u64::MAX,
            max => max.saturating_sub(self.steps.load(Ordering::Relaxed)),
        }
    }

    /// How many list elements the budget left can still pay for, each costing
    /// at least a step and a [`LIST_SLOT`]: the cap on a preallocation.
    pub(crate) fn elements_left(&self) -> usize {
        let elements = self.steps_left().min(self.bytes_left() / LIST_SLOT);
        usize::try_from(elements).unwrap_or(usize::MAX)
    }

    /// Matches `pattern` against `subject`, as `matches` does, charging the
    /// steps it costs. A pattern longer than
    /// [`RuntimeOptions::with_max_regex_len`] is refused first.
    ///
    /// The pattern is compiled with the same configuration as
    /// `regex::Regex::new` (so the result and the error messages are the
    /// same), bounded by [`RuntimeOptions::with_regex_size_limit`], and kept
    /// in `cache` under that limit, its error included. The charges are
    /// calibrated on measured CPU, about 80 ns per step:
    ///
    /// - the lookup: one step per 64 bytes of pattern (rounded up), for
    ///   hashing and comparing it;
    /// - on a miss only, parsing: the pattern is parsed twice, to price it
    ///   and to compile it, each charged per byte before either runs, see
    ///   `regex_cost::parse_steps`;
    /// - on a miss only, translating: priced from the parsed pattern before
    ///   compiling, see `regex_cost::translation_steps` (Unicode and Perl
    ///   classes, and under `(?i)` every case fold: of each Unicode class,
    ///   and of each bracket and set operand, Perl classes, ranges and nested
    ///   brackets included);
    /// - on a miss only, compiling: one step per 8 bytes of the compiled
    ///   automaton, or of the size limit when the compile fails for size,
    ///   as the work up to the limit was done anyway. The compiled pattern
    ///   is kept in `cache` only once this charge succeeds;
    /// - matching: two steps per KiB of automaton per 64 bytes of subject
    ///   (rounded up), charged before matching, as a match can cost up to
    ///   the automaton's states times the subject.
    ///
    /// A pattern compiled by an earlier call, or an earlier evaluation, is
    /// so charged the lookup and the match only.
    #[cfg(feature = "regex")]
    pub(crate) fn is_match(
        &self,
        cache: &crate::RegexCache,
        subject: &str,
        pattern: &str,
    ) -> Result<bool, ExecutionError> {
        let (len, limit) = (pattern.len() as u64, self.max_regex_len);
        if limit > 0 && len > limit {
            return Err(ExecutionError::function_error(
                "matches",
                format!("regex pattern of {len} bytes exceeds the limit of {limit} bytes"),
            ));
        }
        if !self.add_steps(len / 64 + 1) {
            return Err(self.exceeded());
        }
        let size_limit = crate::regex_cache::size_limit(self.regex_size_limit);
        let compiled = match cache.lookup(pattern, size_limit) {
            Some(compiled) => compiled,
            None => self.compile(cache, pattern, size_limit)?,
        };
        let regex = compiled.regex()?;
        let size = compiled.built();
        let scan = (size / 1024 + 1)
            .saturating_mul(subject.len() as u64 / 64 + 1)
            .saturating_mul(2);
        if !self.add_steps(scan) {
            return Err(self.exceeded());
        }
        Ok(regex.is_match(subject))
    }

    /// Compiles `pattern` under `size_limit` into `cache`, charging the
    /// parsing and the translation before it, and the compile after.
    #[cfg(feature = "regex")]
    #[inline(never)]
    fn compile(
        &self,
        cache: &crate::RegexCache,
        pattern: &str,
        size_limit: usize,
    ) -> Result<crate::regex_cache::Compiled, ExecutionError> {
        // parsing the pattern twice, to price it and to compile it, then
        // translating it, priced from its syntax: each charged before it runs
        if !self.add_steps(crate::regex_cost::parse_steps(pattern)) {
            return Err(self.exceeded());
        }
        if !self.add_steps(crate::regex_cost::translation_steps(
            pattern,
            self.steps_left(),
        )) {
            return Err(self.exceeded());
        }
        let compiled = crate::regex_cache::compile(pattern, size_limit);
        let steps = match compiled.built() {
            0 => pattern.len() as u64 / 64 + 1,
            built => built / 8,
        };
        // kept only once paid for: a compile the budget refuses is dropped,
        // so evaluations that are refused cannot fill the cache
        match self.add_steps(steps) {
            true => Ok(cache.insert(pattern, size_limit, compiled)),
            false => Err(self.exceeded()),
        }
    }

    /// The resources used so far by the evaluation.
    pub(crate) fn usage(&self) -> EvalUsage {
        EvalUsage {
            steps: self.steps.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            iterations: self.iterations.load(Ordering::Relaxed),
        }
    }

    // The hot charges below report success as a `bool` and leave building the
    // error to the cold `exceeded`: a `Result` would cost every interpreter
    // node a large return slot even when no frame exists.

    /// Charges the step of evaluating one expression node. Returns `false`
    /// once the evaluation must stop, the error then being
    /// [`exceeded`](Self::exceeded).
    ///
    /// Like every counter of the frame, the steps are counted with a relaxed
    /// `fetch_add`: exact even if a custom function evaluates on the frame
    /// from several threads at once. Cannot overflow in practice: a `u64` of
    /// steps is centuries of evaluation.
    #[inline(always)]
    pub(crate) fn step(&self) -> bool {
        let total = self.steps.fetch_add(1, Ordering::Relaxed) + 1;
        (self.max_steps == 0 || total <= self.max_steps)
            && self.abort.load(Ordering::Relaxed) == ABORT_NONE
    }

    /// Charges `steps` evaluation steps, see [`step`](Self::step).
    #[inline(never)]
    pub(crate) fn add_steps(&self, steps: u64) -> bool {
        let total = self
            .steps
            .fetch_add(steps, Ordering::Relaxed)
            .saturating_add(steps);
        if self.max_steps > 0 && total > self.max_steps {
            self.record(ABORT_STEPS);
        }
        self.abort.load(Ordering::Relaxed) == ABORT_NONE
    }

    /// Charges `bytes` allocated bytes against the bytes budget. Returns
    /// `false` once the evaluation must stop, the error then being
    /// [`exceeded`](Self::exceeded).
    #[inline(never)]
    pub(crate) fn add_bytes(&self, bytes: u64) -> bool {
        let total = self
            .bytes
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes);
        if self.max_bytes > 0 && total > self.max_bytes {
            self.record(ABORT_BYTES);
        }
        self.abort.load(Ordering::Relaxed) == ABORT_NONE
    }

    /// Charges the bytes and the steps of creating `value`, see
    /// [`fresh_size`], [`fresh_steps`] and [`add_bytes`](Self::add_bytes).
    #[inline(never)]
    pub(crate) fn add_fresh(&self, value: &dyn Val) -> bool {
        self.add_bytes(fresh_size(value)) && self.add_steps(fresh_steps(value))
    }

    /// Charges the bytes of copying `value`, see [`clone_size`] and
    /// [`add_bytes`](Self::add_bytes).
    #[inline(never)]
    pub(crate) fn add_clone(&self, value: &dyn Val) -> bool {
        self.add_bytes(clone_size(value))
    }

    /// The error of an evaluation a charge has stopped: the abort recorded,
    /// recording first the steps overrun [`step`](Self::step) leaves to it.
    #[cold]
    #[inline(never)]
    pub(crate) fn exceeded(&self) -> ExecutionError {
        if self.max_steps > 0 && self.steps.load(Ordering::Relaxed) > self.max_steps {
            self.record(ABORT_STEPS);
        }
        self.abort_error()
            .expect("a stopped evaluation has an abort")
    }

    /// Charges the bytes and steps of converting `value` into a
    /// [`Value`](crate::Value), see [`conversion_cost`]. The walk stops once it
    /// passes the bytes or the steps left, so a value sharing a large one many
    /// times is refused before it is converted, not after.
    pub(crate) fn charge_conversion(&self, value: &dyn Val) -> Result<(), ExecutionError> {
        let cost = conversion_cost(value, self.bytes_left(), self.steps_left());
        match self.add_bytes(cost.bytes) && self.add_steps(cost.steps) {
            true => Ok(()),
            false => Err(self.exceeded()),
        }
    }

    /// The bytes left before the bytes budget is exceeded, `u64::MAX` when
    /// there is none.
    pub(crate) fn bytes_left(&self) -> u64 {
        match self.max_bytes {
            0 => u64::MAX,
            max => max.saturating_sub(self.bytes.load(Ordering::Relaxed)),
        }
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
///
/// Values of other types are not charged (and [`compare_size`] prices them
/// at 0): see the cost section of the [`Val`] docs. For the built-in scalars (`int`,
/// `bool`, `timestamp`, ...) and opaque values (an `Arc`) the copy is a
/// small fixed-size box, paid for by the steps of the node that copies it.
/// A custom [`Val`] decides in its own `clone_as_boxed` what a copy costs,
/// which the interpreter cannot see: an embedder exposing a custom value
/// whose copy is expensive should make it cheap to clone (share it behind an
/// `Arc`), as the built-in containers are.
pub(crate) fn clone_size(value: &dyn Val) -> u64 {
    match value.as_builtin() {
        #[cfg(feature = "structs")]
        BuiltinRef::Struct(s) => s.field_count() as u64 * MAP_SLOT,
        _ => 0,
    }
}

/// The steps comparing `value` with another value can cost, stopping once
/// the count passes `cap`: one per list element or map entry, and one per 64
/// bytes of every string or bytes value compared, recursively. Scalars and
/// short strings cost nothing beyond the node that produced them, and so do
/// custom [`Val`]s, whose comparison cost the interpreter cannot see.
///
/// Equality and `in` compare deeply, so a few nodes can compare megabytes;
/// the walk is charged before comparing, and capped so that it costs no more
/// than the budget it is about to exceed.
pub(crate) fn compare_size(value: &dyn Val, cap: u64) -> u64 {
    fn walk(value: &dyn Val, cap: u64, total: &mut u64) {
        match value.as_builtin() {
            BuiltinRef::String(s) => *total += s.inner().len() as u64 / 64,
            BuiltinRef::Bytes(b) => *total += b.inner().len() as u64 / 64,
            BuiltinRef::List(l) => {
                *total += l.inner().len() as u64;
                for item in l.inner() {
                    if *total > cap {
                        return;
                    }
                    walk(item.as_ref(), cap, total);
                }
            }
            BuiltinRef::Map(m) => {
                *total += m.inner().len() as u64;
                for (key, item) in m.inner() {
                    if *total > cap {
                        return;
                    }
                    walk(key.inner(), cap, total);
                    walk(item.as_ref(), cap, total);
                }
            }
            BuiltinRef::Optional(o) => {
                if let Some(inner) = o.option() {
                    walk(inner, cap, total);
                }
            }
            #[cfg(feature = "structs")]
            BuiltinRef::Struct(s) => {
                *total += s.field_count() as u64;
                for item in s.fields() {
                    if *total > cap {
                        return;
                    }
                    walk(item, cap, total);
                }
            }
            _ => {}
        }
    }
    let mut total = 0;
    walk(value, cap, &mut total);
    total
}

/// The cost of converting a value into a [`Value`](crate::Value): the bytes
/// it allocates and the steps it takes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cost {
    pub(crate) bytes: u64,
    pub(crate) steps: u64,
}

/// The cost of converting `value` into a [`Value`](crate::Value), stopping
/// once either count passes its cap.
///
/// The conversion rebuilds every list and map ([`VALUE_SLOT`] bytes and one
/// step per list element, [`MAP_SLOT`] bytes and one step per map entry) and
/// copies borrowed strings and bytes (their length in bytes, one step per 64),
/// while shared ones are not copied. It visits a value shared `n` times `n`
/// times, which is why the walk is capped: it runs before the conversion, and
/// stops once the conversion would exceed the budget left.
pub(crate) fn conversion_cost(value: &dyn Val, bytes_cap: u64, steps_cap: u64) -> Cost {
    fn walk(value: &dyn Val, caps: &Cost, total: &mut Cost) {
        let over = |total: &Cost| total.bytes > caps.bytes || total.steps > caps.steps;
        if over(total) {
            return;
        }
        match value.as_builtin() {
            BuiltinRef::String(s) if s.as_arc().is_none() => {
                let len = s.inner().len() as u64;
                total.bytes += len;
                total.steps += len / 64;
            }
            BuiltinRef::Bytes(b) if b.as_arc().is_none() => {
                let len = b.inner().len() as u64;
                total.bytes += len;
                total.steps += len / 64;
            }
            BuiltinRef::List(l) => {
                let len = l.inner().len() as u64;
                total.bytes += len * VALUE_SLOT;
                total.steps += len;
                for item in l.inner() {
                    walk(item.as_ref(), caps, total);
                    if over(total) {
                        return;
                    }
                }
            }
            BuiltinRef::Map(m) => {
                let len = m.inner().len() as u64;
                total.bytes += len * MAP_SLOT;
                total.steps += len;
                for item in m.inner().values() {
                    walk(item.as_ref(), caps, total);
                    if over(total) {
                        return;
                    }
                }
            }
            BuiltinRef::Optional(o) => {
                total.bytes += OPTIONAL_SLOT;
                total.steps += 1;
                if let Some(inner) = o.option() {
                    walk(inner, caps, total);
                }
            }
            #[cfg(feature = "structs")]
            BuiltinRef::Struct(s) => {
                let len = s.field_count() as u64;
                total.bytes += len * MAP_SLOT;
                total.steps += len;
                for item in s.fields() {
                    walk(item, caps, total);
                    if over(total) {
                        return;
                    }
                }
            }
            _ => {}
        }
    }
    let caps = Cost {
        bytes: bytes_cap,
        steps: steps_cap,
    };
    let mut total = Cost::default();
    walk(value, &caps, &mut total);
    total
}

/// The steps creating `value` took, beyond the node that created it: one per
/// list element or map entry and one per 64 bytes of a string or bytes value,
/// copied into it. Concatenation and conversions are priced by their result.
pub(crate) fn fresh_steps(value: &dyn Val) -> u64 {
    match value.as_builtin() {
        BuiltinRef::String(s) => s.inner().len() as u64 / 64,
        BuiltinRef::Bytes(b) => b.inner().len() as u64 / 64,
        BuiltinRef::List(l) => l.inner().len() as u64,
        BuiltinRef::Map(m) => m.inner().len() as u64,
        #[cfg(feature = "structs")]
        BuiltinRef::Struct(s) => s.field_count() as u64,
        _ => 0,
    }
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

    #[cfg(feature = "regex")]
    #[test]
    fn steps_budget_charges_matches_by_pattern_times_subject() {
        // Matching is O(pattern x subject): an 8 KiB string matched against
        // itself takes most of a second, and must be refused before it runs.
        let mut ctx = steps_budget(10_000);
        ctx.add_variable("s", "a".repeat(8 * 1024)).unwrap();
        let started = std::time::Instant::now();
        assert_eq!(run(&ctx, "s.matches(s)"), steps_exceeded(10_000));
        assert_eq!(
            run(&ctx, "[1, 2, 3].all(x, s.matches(s))"),
            steps_exceeded(10_000)
        );
        assert!(started.elapsed() < Duration::from_millis(100));
        // a short pattern over the same subject still fits
        assert_eq!(run(&ctx, "s.matches('^a+$')"), Ok(true.into()));
    }

    #[cfg(feature = "regex")]
    #[test]
    fn steps_budget_charges_matches_by_compiled_size() {
        // Short patterns can compile to huge automata (`\w` is every Unicode
        // word character): the charge follows the compiled size, so a failed
        // or huge compile swallowed by `|| true` still exhausts the budget.
        let mut ctx = with_options(
            RuntimeOptions::default()
                .with_max_steps(10_000)
                .with_regex_size_limit(64 * 1024),
        );
        ctx.add_variable("s", "a".repeat(8 * 1024)).unwrap();
        ctx.add_variable("l", (0..2_000i64).collect::<Vec<_>>())
            .unwrap();
        for src in [
            r"l.all(x, 'a'.matches(r'\w{1000}') || true)",
            r"l.all(x, 'a'.matches(r'(\w{100}){100}') || true)",
            r"l.all(x, 'a'.matches(r'(?:\w{50}){40}') || true)",
            r"l.all(x, !s.matches(r'\p{L}{200}x'))",
            r"l.all(x, !s.matches(r'.{1000}x') || true)",
            r"l.all(x, !s.matches(r'[\w\W]{300}b') || true)",
            r"l.all(x, !s.matches(r'.{60}x'))",
        ] {
            let started = std::time::Instant::now();
            let result = run(&ctx, src);
            // the budget runs out, or (unswallowed) a pattern over the size
            // limit fails, as a pattern does without a budget
            assert!(
                result == steps_exceeded(10_000)
                    || matches!(
                        &result,
                        Err(ExecutionError::FunctionError { message, .. })
                            if message.ends_with("exceeds size limit of 65536 bytes.")
                    ),
                "{src}: {result:?}"
            );
            assert!(
                started.elapsed() < Duration::from_millis(200),
                "{src} took {:?}",
                started.elapsed()
            );
        }
        // ordinary patterns stay cheap
        assert_eq!(
            run(
                &ctx,
                "'/api/v1/users/42'.matches('^/api/v[0-9]+/users/[0-9]+$')"
            ),
            Ok(true.into())
        );
    }

    #[cfg(feature = "regex")]
    #[test]
    fn steps_budget_charges_regex_translation() {
        // Long or case-insensitive Unicode classes cost far more to translate
        // than their small compiled automata suggest: priced from the syntax,
        // before compiling, with or without a pattern length cap.
        for max_regex_len in [0, 512] {
            for pattern in [
                format!("(?i)[{}]", r"\p{Lu}".repeat(200)),
                format!("(?i)[{}]", r"\p{Greek}".repeat(113)),
                format!("(?i)[{}]", r"\p{Greek}".repeat(12_800)),
                format!("[{}]", r"\p{L}".repeat(12_800)),
                r"(?i)\p{Lu}".to_owned(),
                r"(?i)[\p{Any}]".to_owned(),
            ] {
                let mut ctx = with_options(
                    RuntimeOptions::default()
                        .with_max_steps(10_000)
                        .with_regex_size_limit(1 << 20)
                        .with_max_regex_len(max_regex_len),
                );
                ctx.add_variable("p", pattern.clone()).unwrap();
                ctx.add_variable("l", (0..2_000i64).collect::<Vec<_>>())
                    .unwrap();
                let started = std::time::Instant::now();
                let result = run(&ctx, "l.all(x, 'a'.matches(p) || true)");
                let elapsed = started.elapsed();
                let head: std::string::String = pattern.chars().take(24).collect();
                assert!(
                    result == steps_exceeded(10_000),
                    "{head} (max_regex_len {max_regex_len}): {result:?}"
                );
                assert!(
                    elapsed < Duration::from_millis(200),
                    "{head} (max_regex_len {max_regex_len}) took {elapsed:?}"
                );
            }
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn steps_budget_charges_regex_parsing() {
        // A pattern is parsed twice, to price it and to compile it, at up to
        // ~470 ns a byte each, even when the parse fails: too deep a nesting
        // fails only once the whole pattern is parsed. Both parses are
        // charged per byte before either runs.
        for pattern in [
            format!("{}a{}", "(".repeat(4_000), ")".repeat(4_000)),
            format!("{}a{}", "[".repeat(4_000), "]".repeat(4_000)),
            format!("[{}\\w]", r"\w&&".repeat(4_095)),
            "(".repeat(16_384),
        ] {
            let len = pattern.len() as u64;
            // a third of what parsing it twice costs
            let limit = len * 3;
            let mut ctx = with_options(
                RuntimeOptions::default()
                    .with_max_steps(limit)
                    .with_regex_size_limit(1 << 20),
            );
            ctx.add_variable("p", pattern.clone()).unwrap();
            let head: std::string::String = pattern.chars().take(24).collect();
            let result = run(&ctx, "'a'.matches(p)");
            let shown: std::string::String = format!("{result:?}").chars().take(80).collect();
            assert!(
                result == steps_exceeded(limit),
                "{head} ({len} bytes): {shown}"
            );
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn steps_budget_charges_regex_refolds_and_age_classes() {
        // Each takes 40-60 ms to translate in release, over 500,000 steps at
        // the reference 80 ns: a single call must not fit in 200,000.
        let mut failed = vec![];
        for pattern in [
            // under (?i), every set operation folds its Perl class operand
            format!("(?i)[{}\\w]", r"\w&&".repeat(100)),
            r"(?i:[\w--\d])".repeat(50),
            // every nested bracket refolds all it contains
            format!("(?i)[{}\\w{}]", "a[".repeat(120), "]".repeat(120)),
            // an Age class unions the tables of every earlier version
            format!("[{}]", r"\p{Age=15.0}".repeat(200)),
        ] {
            let limit = 200_000;
            let mut ctx = with_options(
                RuntimeOptions::default()
                    .with_max_steps(limit)
                    .with_regex_size_limit(1 << 20),
            );
            ctx.add_variable("p", pattern.clone()).unwrap();
            let head: std::string::String = pattern.chars().take(24).collect();
            let result = run(&ctx, "'a'.matches(p)");
            if result != steps_exceeded(limit) {
                let shown: std::string::String = format!("{result:?}").chars().take(80).collect();
                failed.push(format!("{head}: {shown}"));
            }
        }
        assert!(failed.is_empty(), "{failed:#?}");
    }

    #[cfg(feature = "regex")]
    #[test]
    fn steps_budget_charges_refolding_negated_brackets() {
        // A nested `[^..]` is folded, then negated: nearly all of Unicode,
        // still marked folded. A literal beside it makes its parent fold it
        // again, ~10 ms. Each takes over 1 ms in release: a single call
        // must not fit in 10,000 steps.
        let mut failed = vec![];
        for pattern in [
            r"(?i)[a[^\x{0}]]".to_owned(),
            r"(?i)[a[^\p{Han}]]".to_owned(),
            format!("(?i){}[[:^alpha:]]{}", "[a".repeat(10), "]".repeat(10)),
            format!("(?i)[a{}]", "[^a]".repeat(10)),
            format!("(?i){}[^a]{}", "[a".repeat(100), "]".repeat(100)),
            r"(?i)[a[:^alpha:]]".to_owned(),
        ] {
            let limit = 10_000;
            let mut ctx = with_options(
                RuntimeOptions::default()
                    .with_max_steps(limit)
                    .with_regex_size_limit(1 << 20),
            );
            ctx.add_variable("p", pattern.clone()).unwrap();
            let result = run(&ctx, "'a'.matches(p)");
            if result != steps_exceeded(limit) {
                let shown: std::string::String = format!("{result:?}").chars().take(80).collect();
                failed.push(format!("{pattern}: {shown}"));
            }
        }
        assert!(failed.is_empty(), "{failed:#?}");
        // under bb-cel's settings, the first call already exhausts the budget
        let mut ctx = with_options(
            RuntimeOptions::default()
                .with_max_steps(10_000)
                .with_max_bytes(1 << 20)
                .with_regex_size_limit(1 << 20)
                .with_max_regex_len(512),
        );
        let pattern = format!("(?i){}[^\\x{{0}}]{}", "[a".repeat(10), "]".repeat(10));
        ctx.add_variable("p", pattern).unwrap();
        ctx.add_variable("l", (0..2_000i64).collect::<Vec<_>>())
            .unwrap();
        let (result, usage) = Program::compile("l.all(x, 'a'.matches(p) || true)")
            .unwrap()
            .execute_with_usage(&ctx);
        assert_eq!(result, steps_exceeded(10_000));
        assert!(usage.iterations <= 1, "{usage:?}");
    }

    #[cfg(feature = "regex")]
    #[test]
    fn regex_size_limit_fails_big_compiles() {
        let ctx = with_options(RuntimeOptions::default().with_regex_size_limit(16 * 1024));
        let started = std::time::Instant::now();
        assert_eq!(
            run(&ctx, r"'a'.matches(r'\w{1000}')"),
            Err(ExecutionError::function_error(
                "matches",
                "'\\w{1000}' not a valid regex:\nCompiled regex exceeds size limit of 16384 bytes."
            ))
        );
        // a 16 KiB limit is reached in well under the 10 MiB default's time
        assert!(started.elapsed() < Duration::from_millis(50));
        assert_eq!(run(&ctx, "'abc'.matches('^a.c$')"), Ok(true.into()));
        // unset by default, so `matches` keeps the regex crate's limits
        assert_eq!(RuntimeOptions::default().regex_size_limit(), 0);
    }

    /// Runs `src` against `ctx` without a budget and with one, with the
    /// regex limits bb-cel sets, which must not change the result.
    #[cfg(feature = "regex")]
    fn same_with_and_without_budget(
        mut ctx: Context<'static, 'static>,
        src: &str,
    ) -> ResolveResult {
        let unbudgeted = run(&ctx, src);
        ctx.set_budget(
            RuntimeOptions::default()
                .with_max_steps(1_000_000)
                .with_max_regex_len(512)
                .with_regex_size_limit(1 << 20),
        );
        let budgeted = run(&ctx, src);
        assert_eq!(unbudgeted, budgeted, "{src}");
        budgeted
    }

    #[cfg(feature = "regex")]
    #[test]
    fn a_budget_does_not_add_matches_to_an_env_without_it() {
        // a pattern over `max_regex_len` included: the limit is the stdlib's
        let long = "a".repeat(576);
        for src in [
            "'abc'.matches('a.c')".to_owned(),
            format!("'abc'.matches('{long}')"),
            format!("matches('abc', '{long}')"),
        ] {
            let ctx = Context::with_env(Arc::new(Env::default()));
            assert_eq!(
                same_with_and_without_budget(ctx, &src),
                Err(ExecutionError::UndeclaredReference(Arc::new(
                    "matches".into()
                ))),
                "{}",
                &src[..24]
            );
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn a_budget_keeps_a_context_function_named_matches() {
        fn glob(this: crate::extractors::This<Arc<String>>, pattern: Arc<String>) -> bool {
            pattern.as_str() == "*" || this.0 == pattern
        }
        // a pattern over `max_regex_len` included: the limit is the stdlib's
        let long = "a".repeat(576);
        for (src, expected) in [
            ("'abc'.matches('*')".to_owned(), true),
            ("'abc'.matches('a.c')".to_owned(), false),
            ("matches('abc', '*')".to_owned(), true),
            (format!("'abc'.matches('{long}')"), false),
            (format!("matches('abc', '{long}')"), false),
        ] {
            let mut ctx = Context::with_env(Arc::new(Env::default()));
            ctx.add_function("matches", glob).unwrap();
            assert_eq!(
                same_with_and_without_budget(ctx, &src),
                Ok(expected.into()),
                "{}",
                &src[..src.len().min(24)]
            );
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn a_budget_keeps_an_env_overload_named_matches() {
        use crate::common::functions::EvalCtx;
        use crate::common::types::{CelBool, CelString, STRING_TYPE};
        use crate::common::value::CowVal;
        fn glob(this: &CelString, pattern: &CelString) -> CelBool {
            CelBool::from(pattern.inner() == "*" || this.inner() == pattern.inner())
        }
        fn plain<'b, 'v>(args: Vec<CowVal<'b, 'v>>) -> Result<CowVal<'b, 'v>, ExecutionError> {
            let string = |i: usize| args[i].downcast_ref::<CelString>().unwrap();
            Ok(CowVal::owned(glob(string(0), string(1))))
        }
        fn with_env<'b, 'v>(
            _: &EvalCtx<'_>,
            args: Vec<CowVal<'b, 'v>>,
        ) -> Result<CowVal<'b, 'v>, ExecutionError> {
            plain(args)
        }
        // a pattern over `max_regex_len`, and one over the size limit,
        // included: the limits are the stdlib's
        let long = "a".repeat(576);
        let cases = [
            ("'abc'.matches('*')".to_owned(), true),
            ("'abc'.matches('a.c')".to_owned(), false),
            (r"'a'.matches(r'\w{1000}')".to_owned(), false),
            (format!("'abc'.matches('{long}')"), false),
            (format!("'{long}'.matches('{long}')"), true),
        ];
        // an embedder's `matches`, plain or Env-aware, member or global
        type MakeEnv = fn() -> Env;
        let envs: [(&str, MakeEnv); 4] = [
            ("plain member", || {
                let mut env = Env::default();
                env.add_member_overload("matches", "m", STRING_TYPE, vec![STRING_TYPE], plain)
                    .unwrap();
                env
            }),
            ("Env-aware member", || {
                let mut env = Env::default();
                env.add_member_overload_with_env(
                    "matches",
                    "m",
                    STRING_TYPE,
                    vec![STRING_TYPE],
                    with_env,
                )
                .unwrap();
                env
            }),
            ("plain global", || {
                let mut env = Env::default();
                env.add_overload("matches", "m", vec![STRING_TYPE, STRING_TYPE], plain)
                    .unwrap();
                env
            }),
            ("Env-aware global", || {
                let mut env = Env::default();
                env.add_overload_with_env("matches", "m", vec![STRING_TYPE, STRING_TYPE], with_env)
                    .unwrap();
                env
            }),
        ];
        for (kind, env) in envs {
            for (src, expected) in &cases {
                let src = match kind.ends_with("member") {
                    true => src.clone(),
                    // `'s'.matches('p')` as `matches('s', 'p')`
                    false => {
                        let (this, rest) = src.split_once(".matches(").unwrap();
                        format!("matches({this}, {rest}")
                    }
                };
                let ctx = Context::with_env(Arc::new(env()));
                assert_eq!(
                    same_with_and_without_budget(ctx, &src),
                    Ok((*expected).into()),
                    "{kind}: {}",
                    &src[..src.len().min(24)]
                );
            }
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn stdlib_matches_gives_the_same_results_with_a_budget() {
        for src in [
            "'abc'.matches('a.c')",
            "'abc'.matches('^b')",
            "'ABC'.matches('(?i)abc')",
            r"'日本語'.matches(r'^\p{Han}+$')",
            "'abc'.matches('[')",
            "'abc'.matches('(?P<x>a')",
            "'a'.matches('a{2,1}')",
        ] {
            let _ = same_with_and_without_budget(Context::default(), src);
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn max_regex_len_rejects_longer_patterns() {
        let mut ctx = with_options(RuntimeOptions::default().with_max_regex_len(8));
        ctx.add_variable("p", "a".repeat(9)).unwrap();
        assert_eq!(run(&ctx, "'aaa'.matches('^a+$')"), Ok(true.into()));
        assert_eq!(
            run(&ctx, "'aaa'.matches(p)"),
            Err(ExecutionError::function_error(
                "matches",
                "regex pattern of 9 bytes exceeds the limit of 8 bytes"
            ))
        );
        // off by default
        assert_eq!(RuntimeOptions::default().max_regex_len(), 0);
    }

    /// `s`: 8 KiB; `ls`: 2000 shares of `s`; `m`: 128 entries of 8 KiB.
    fn deep_values(ctx: &mut Context) {
        let s = Arc::new("a".repeat(8 * 1024));
        ctx.add_variable_from_value("s", Value::String(s.clone()));
        ctx.add_variable_from_value(
            "ls",
            (0..2_000)
                .map(|_| Value::String(s.clone()))
                .collect::<Vec<_>>(),
        );
        let m: std::collections::HashMap<String, Value> = (0..128)
            .map(|i| {
                (
                    format!("k{i}"),
                    Value::from(format!("{i}{}", "v".repeat(8 * 1024))),
                )
            })
            .collect();
        ctx.add_variable_from_value("m", m);
    }

    #[test]
    fn steps_budget_charges_deep_equality_and_membership() {
        // Comparison walks the whole operands: a walk of the same depth is
        // charged, capped at the steps left, before comparing.
        let mut ctx = steps_budget(10_000);
        deep_values(&mut ctx);
        for src in [
            "ls == ls",
            "[ls] == [ls]",
            "[ls, ls] != [ls, ls]",
            "[m] == [m]",
            "{'a': ls} == {'a': ls}",
            "m in [m, m, m, m]",
            "s in ls",
            "[1].all(x, [ls] == [ls])",
        ] {
            let started = std::time::Instant::now();
            assert_eq!(run(&ctx, src), steps_exceeded(10_000), "{src}");
            assert!(
                started.elapsed() < Duration::from_millis(20),
                "{src} took {:?}",
                started.elapsed()
            );
        }
        // small comparisons stay cheap
        assert_eq!(run(&ctx, "[1, 2] == [1, 2] && 'k1' in m"), Ok(true.into()));
    }

    #[test]
    fn steps_budget_alone_bounds_creation_and_conversion() {
        // Without a bytes budget, the steps budget must still price work that
        // grows with the data: concatenation copies every element, and the
        // result conversion visits every node.
        let mut ctx = steps_budget(10_000);
        ctx.add_variable("l", (0..2_000i64).collect::<Vec<_>>())
            .unwrap();
        ctx.add_variable("s", "a".repeat(8 * 1024)).unwrap();
        for src in [
            "size(l.map(x, l + l)) > 0",
            "l.map(x, l)",
            "size([1, 2].map(x, l + l + l + l + l + l)) > 0",
            "size([1, 2, 3].map(x, s + s + s + s + s + s + s + s + s + s)) > 0",
            "[1, 2, 3, 4, 5, 6].map(k, l)",
        ] {
            let started = std::time::Instant::now();
            assert_eq!(run(&ctx, src), steps_exceeded(10_000), "{src}");
            assert!(
                started.elapsed() < Duration::from_millis(50),
                "{src} took {:?}",
                started.elapsed()
            );
        }
        assert_eq!(run(&ctx, "size(l + l)"), Ok(4_000.into()));
    }

    #[test]
    fn steps_budget_charges_large_keys_sizes_and_orderings() {
        // Hashing a map key, counting a string's characters and ordering two
        // strings or bytes values all walk them: one step per 64 bytes.
        let mut ctx = steps_budget(10_000);
        let big = "a".repeat(1 << 20);
        ctx.add_variable("big", big.clone()).unwrap();
        ctx.add_variable("big2", big.clone()).unwrap();
        ctx.add_variable_from_value("bb", Value::Bytes(Arc::new(big.clone().into_bytes())));
        ctx.add_variable_from_value("mb", std::collections::HashMap::from([(big, 1i64)]));
        ctx.add_variable("l", (0..2_000i64).collect::<Vec<_>>())
            .unwrap();
        for src in [
            "l.all(x, mb[big] == 1)",
            "l.all(x, mb[?big].hasValue())",
            "l.all(x, size(big) > 0)",
            "l.all(x, big.size() > 0)",
            "l.all(x, big <= big2)",
            "l.all(x, big > big2 || true)",
            "l.all(x, bb < bb || true)",
        ] {
            let started = std::time::Instant::now();
            assert_eq!(run_optional(&ctx, src), steps_exceeded(10_000), "{src}");
            assert!(
                started.elapsed() < Duration::from_millis(20),
                "{src} took {:?}",
                started.elapsed()
            );
        }
        // small ones stay cheap
        assert_eq!(run(&ctx, "size('abc') == 3 && 'a' < 'b'"), Ok(true.into()));
    }

    #[test]
    fn a_repeated_long_map_key_is_truncated_in_the_error() {
        // The error is built outside any budget: it must not copy the key.
        let mut ctx = Context::default();
        let s = "a".repeat(8 * 1024);
        ctx.add_variable("s", s.clone()).unwrap();
        match run(&ctx, "{s: 1, s: 2}") {
            Err(ExecutionError::DuplicateKey(Value::String(key))) => {
                assert_eq!(*key, format!("{}…", &s[..64]));
            }
            other => panic!("unexpected {other:?}"),
        }
        // short keys are reported whole
        assert_eq!(
            run(&ctx, "{'k': 1, 'k': 2}"),
            Err(ExecutionError::DuplicateKey(Value::String(Arc::new(
                "k".into()
            ))))
        );
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

    /// Every construct that catches or short-circuits errors, with `BOOM`
    /// standing for a sub-expression that exhausts the budget.
    const ERROR_CATCHING_SITES: &[&str] = &[
        "BOOM || true",
        "false || BOOM",
        "BOOM && false",
        "true && BOOM",
        "!BOOM",
        "BOOM ? 1 : 2",
        "true ? BOOM : false",
        "false ? 1 : BOOM",
        "[1].all(x, BOOM)",
        "[1].exists(x, BOOM)",
        "[1, 2].exists(x, x == 2 || BOOM)",
        "[1].exists_one(x, BOOM)",
        "size([1].filter(x, BOOM)) == 0",
        "size([1].map(x, BOOM)) == 0",
        "size([1].map(x, BOOM, x)) == 0",
        "[1].all(x, [1].exists(y, BOOM)) || true",
        "has({'a': BOOM}.a)",
        "has({'a': {'b': BOOM}}.a.b)",
        "{'a': BOOM}.?a.hasValue()",
        "{'a': {'b': BOOM}}.?a.?b.hasValue()",
        "optional.of({'a': BOOM}).a.hasValue()",
        "[BOOM][?0].hasValue()",
        "{'a': BOOM}[?'a'].hasValue()",
        "{'a': 1}[?'a'].orValue(BOOM) == 1",
        "optional.none().orValue(BOOM)",
        "optional.of(BOOM).hasValue()",
        "BOOM in [true]",
        "[BOOM] == [true]",
        "{'k': BOOM} != {}",
    ];

    /// Asserts that `boom` exhausting the budget of `ctx` fails every site
    /// with `expected`.
    fn assert_fatal_everywhere(ctx: &Context, boom: &str, expected: ResolveResult) {
        for site in ERROR_CATCHING_SITES {
            let src = site.replace("BOOM", boom);
            assert_eq!(run_optional(ctx, &src), expected, "{src}");
        }
    }

    #[test]
    fn steps_budget_is_fatal_everywhere() {
        let mut ctx = steps_budget(2_000);
        ctx.add_variable("l", (0..1_000i64).collect::<Vec<_>>())
            .unwrap();
        // the sites themselves fit the budget...
        for site in ERROR_CATCHING_SITES {
            let src = site.replace("BOOM", "true");
            assert!(run_optional(&ctx, &src).is_ok(), "{src}");
        }
        // ...and none of them absorbs running out of it
        assert_fatal_everywhere(&ctx, "l.all(x, true)", steps_exceeded(2_000));
    }

    #[test]
    fn bytes_budget_is_fatal_everywhere() {
        let ctx = bytes_budget(1_500);
        for site in ERROR_CATCHING_SITES {
            let src = site.replace("BOOM", "true");
            assert!(run_optional(&ctx, &src).is_ok(), "{src}");
        }
        assert_fatal_everywhere(&ctx, "size(s + s) > 0", bytes_exceeded(1_500));
    }

    #[test]
    fn iteration_budget_is_fatal_everywhere() {
        let mut ctx = budgeted(500);
        ctx.add_variable("l", (0..1_000i64).collect::<Vec<_>>())
            .unwrap();
        assert_fatal_everywhere(&ctx, "l.all(x, true)", budget_exceeded(500));
    }

    #[test]
    fn interrupt_is_fatal_everywhere() {
        let interrupt = || true;
        let mut ctx = Context::default();
        ctx.set_interrupt(&interrupt);
        assert_fatal_everywhere(&ctx, "[1].all(x, true)", Err(ExecutionError::Interrupted));
    }

    #[test]
    fn a_swallowed_budget_error_still_fails_the_evaluation() {
        fn swallow(ftx: &FunctionContext) -> ResolveResult {
            let program = Program::compile("size(s + s) > 0 && l.all(x, true)").unwrap();
            let _ = program.execute(ftx.ptx);
            Ok(Value::Bool(true))
        }
        let mut ctx = bytes_budget(1_500);
        ctx.add_function("swallow", swallow).unwrap();
        assert_eq!(run(&ctx, "swallow() || true"), bytes_exceeded(1_500));
        let mut ctx = steps_budget(100);
        ctx.add_variable("s", "x").unwrap();
        ctx.add_variable("l", (0..1_000i64).collect::<Vec<_>>())
            .unwrap();
        ctx.add_function("swallow", swallow).unwrap();
        assert_eq!(run(&ctx, "swallow() || true"), steps_exceeded(100));
    }

    #[test]
    fn budget_exceeded_is_fatal() {
        assert!(ExecutionError::BudgetExceeded {
            kind: BudgetKind::Steps,
            limit: 1
        }
        .is_fatal());
        assert!(ExecutionError::BudgetExceeded {
            kind: BudgetKind::Bytes,
            limit: 1
        }
        .is_fatal());
    }

    #[test]
    fn context_budget_overrides_the_env() {
        // the env sets no budget: the context's applies
        let mut ctx = Context::default();
        ctx.set_budget(RuntimeOptions::default().with_max_steps(2));
        assert_eq!(run(&ctx, "1 + 1"), steps_exceeded(2));
        // the env sets one: the context's replaces it
        let mut ctx = steps_budget(2);
        ctx.set_budget(RuntimeOptions::default().with_max_steps(3));
        assert_eq!(run(&ctx, "1 + 1"), Ok(2.into()));
        // an override leaving the steps unset inherits the env's
        ctx.set_budget(RuntimeOptions::default());
        assert_eq!(run(&ctx, "1 + 1"), steps_exceeded(2));
    }

    #[test]
    fn context_budget_inherits_unset_fields() {
        // the Env caps iterations and polls every 10th iteration...
        let mut env = Env::stdlib();
        env.set_options(
            RuntimeOptions::default()
                .with_max_iterations(5)
                .with_interrupt_check_frequency(10),
        );
        let mut ctx = Context::with_env(Arc::new(env));
        // ...a context adding a steps budget keeps both
        ctx.set_budget(RuntimeOptions::default().with_max_steps(1_000));
        let effective = ctx.budget();
        assert_eq!(effective.max_steps(), 1_000);
        assert_eq!(effective.max_iterations(), 5);
        assert_eq!(effective.interrupt_check_frequency(), 10);
        assert_eq!(
            run(&ctx, "[1, 2, 3, 4, 5, 6].all(x, x > 0)"),
            budget_exceeded(5)
        );
        // a child scope overrides only what it sets
        let mut child = ctx.new_inner_scope();
        child.set_budget(RuntimeOptions::default().with_max_iterations(10));
        let effective = child.budget();
        assert_eq!(effective.max_iterations(), 10);
        assert_eq!(effective.max_steps(), 1_000);
        assert_eq!(effective.interrupt_check_frequency(), 10);
        assert_eq!(
            run(&child, "[1, 2, 3, 4, 5, 6].all(x, x > 0)"),
            Ok(true.into())
        );
    }

    #[test]
    fn a_budget_set_inside_a_running_evaluation_has_no_effect() {
        fn loosen(ftx: &FunctionContext) -> ResolveResult {
            let mut scope = ftx.ptx.new_inner_scope();
            scope.set_budget(RuntimeOptions::default().with_max_steps(1_000_000));
            Program::compile("[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(x, x > 0)")
                .unwrap()
                .execute(&scope)
        }
        let mut ctx = steps_budget(20);
        ctx.add_function("loosen", loosen).unwrap();
        assert_eq!(run(&ctx, "loosen()"), steps_exceeded(20));
    }

    #[test]
    fn steps_are_counted_exactly_across_threads() {
        // A custom function may evaluate on the running frame from several
        // threads at once: the counters are read-modify-write, so no step is
        // lost.
        fn fan_out(ftx: &FunctionContext) -> ResolveResult {
            let program = Program::compile("1 + 1").unwrap();
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| {
                        for _ in 0..10_000 {
                            program.execute(ftx.ptx).unwrap();
                        }
                    });
                }
            });
            Ok(Value::Bool(true))
        }
        let mut ctx = steps_budget(u64::MAX / 2);
        ctx.add_function("fanOut", fan_out).unwrap();
        let (result, usage) = Program::compile("fanOut()")
            .unwrap()
            .execute_with_usage(&ctx);
        assert_eq!(result, Ok(true.into()));
        // one node and one dispatch, then 4 x 10,000 x 3 steps
        assert_eq!(usage.steps, 2 + 4 * 10_000 * 3);
    }

    #[test]
    fn nearest_scope_budget_applies() {
        let mut root = Context::default();
        root.set_budget(RuntimeOptions::default().with_max_steps(2));
        let mut child = root.new_inner_scope();
        assert_eq!(run(&child, "1 + 1"), steps_exceeded(2));
        child.set_budget(RuntimeOptions::default().with_max_steps(3));
        assert_eq!(run(&child, "1 + 1"), Ok(2.into()));
        assert_eq!(run(&root, "1 + 1"), steps_exceeded(2));
    }

    #[test]
    fn contexts_sharing_an_env_vary_their_budgets() {
        let env = Arc::new(Env::stdlib());
        let mut tight = Context::with_env(env.clone());
        tight.set_budget(RuntimeOptions::default().with_max_bytes(10));
        let loose = Context::with_env(env);
        assert_eq!(run(&tight, "size('abc' + 'defghijkl')"), bytes_exceeded(10));
        assert_eq!(run(&loose, "size('abc' + 'defghijkl')"), Ok(12.into()));
    }

    #[test]
    fn unbounded_evaluation_creates_no_frame() {
        fn has_frame(ftx: &FunctionContext) -> ResolveResult {
            Ok(Value::Bool(ftx.ptx.frame().is_some()))
        }
        let mut ctx = Context::default();
        ctx.add_function("hasFrame", has_frame).unwrap();
        assert_eq!(run(&ctx, "hasFrame()"), Ok(false.into()));
        ctx.set_budget(RuntimeOptions::default().with_max_steps(100));
        assert_eq!(run(&ctx, "hasFrame()"), Ok(true.into()));
        let interrupt = || false;
        let mut ctx = Context::default();
        ctx.add_function("hasFrame", has_frame).unwrap();
        ctx.set_interrupt(&interrupt);
        assert_eq!(run(&ctx, "hasFrame()"), Ok(true.into()));
    }

    #[test]
    fn usage_is_reported() {
        let ctx = Context::default();
        let (result, usage) = Program::compile("1 + 1").unwrap().execute_with_usage(&ctx);
        assert_eq!(result, Ok(2.into()));
        assert_eq!((usage.steps, usage.bytes, usage.iterations), (3, 0, 0));

        let (result, usage) = Program::compile("'abc' + 'def'")
            .unwrap()
            .execute_with_usage(&ctx);
        assert_eq!(result, Ok("abcdef".into()));
        assert_eq!(usage.bytes, 6);

        let (result, usage) = Program::compile("[1, 2, 3].all(x, x > 0)")
            .unwrap()
            .execute_with_usage(&ctx);
        assert_eq!(result, Ok(true.into()));
        assert_eq!(usage.iterations, 3);
        assert!(usage.steps > 3);
    }

    #[test]
    fn usage_matches_value_resolve() {
        let ctx = Context::default();
        let program = Program::compile("[1, 2].map(x, x * 2)").unwrap();
        let (result, usage) = Value::resolve_with_usage(program.expression(), &ctx);
        assert_eq!(result, program.execute(&ctx));
        assert_eq!(usage, program.execute_with_usage(&ctx).1);
        assert_eq!(usage.iterations, 2);
    }

    #[test]
    fn usage_is_reported_when_the_budget_is_exceeded() {
        let ctx = steps_budget(2);
        let (result, usage) = Program::compile("1 + 1").unwrap().execute_with_usage(&ctx);
        assert_eq!(result, steps_exceeded(2));
        assert!(usage.steps > 2);
    }

    #[test]
    fn nested_usage_reports_the_nested_evaluation_only() {
        fn nested(ftx: &FunctionContext) -> ResolveResult {
            let program = Program::compile("1 + 1").unwrap();
            let (result, usage) = program.execute_with_usage(ftx.ptx);
            assert_eq!(usage.steps, 3);
            result
        }
        let mut ctx = steps_budget(1_000);
        ctx.add_function("nested", nested).unwrap();
        let (result, usage) = Program::compile("1 + 1 + nested()")
            .unwrap()
            .execute_with_usage(&ctx);
        assert_eq!(result, Ok(4.into()));
        // 5 nodes, one dispatch and the nested evaluation's 3 steps
        assert_eq!(usage.steps, 9);
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
        let interrupt = || false;
        let mut root = Context::default();
        root.set_interrupt(&interrupt);
        let frame = root.new_frame().expect("an interrupt needs a frame");
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
