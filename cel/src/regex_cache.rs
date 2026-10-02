//! The regular expressions `matches` compiled, kept by the [`Env`](crate::Env)
//! for the next call of the same pattern.
//!
//! An entry is keyed by the pattern and the size limit it was compiled
//! under, so a pattern compiled under one limit is never served under
//! another: a smaller limit could refuse it, and the lazy DFA's cache is
//! sized by the limit too. A pattern that fails to compile, for its syntax
//! or its size, is kept as its error.
//!
//! Hits do not exclude each other. The index of the entries is replicated,
//! a replica per core: a hit takes its thread's replica shared, finds the
//! entry and clones its `Arc`, so hits on threads of different groups touch
//! no common lock, and hits sharing a replica only share its lock. Compiling
//! and matching happen outside every lock. An insertion is decided under
//! one mutex, and then applied to each replica under its lock, exclusively.
//!
//! When the cache is full, the entry used least recently is evicted, found
//! by a scan of the entries under the mutex: a capacity in the low thousands
//! at most is sensible. Every insertion takes a tick from a counter, and a
//! hit marks its entry used at the latest tick, so that recency is tracked
//! between insertions: the entries used since the last insertion count as
//! equally recent, and among them the earliest inserted is evicted first. A
//! sequence of uses on one thread always evicts the same entries; uses
//! racing on several threads are ordered by the ticks they saw.

use crate::ExecutionError;
use regex_automata::{meta, util::syntax, MatchKind};
use regex_syntax::hir::{Hir, HirKind, LookSet};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// The `regex` crate's default limit on the size of a compiled pattern.
const NFA_SIZE_LIMIT: usize = 10 << 20;
/// The `regex` crate's default capacity of the lazy DFA's cache.
const DFA_SIZE_LIMIT: usize = 2 << 20;

/// The size limit a pattern is compiled under for a
/// [`RuntimeOptions::with_regex_size_limit`](crate::RuntimeOptions::with_regex_size_limit)
/// of `regex_size_limit`: zero is the `regex` crate's default.
pub(crate) fn size_limit(regex_size_limit: u64) -> usize {
    match regex_size_limit {
        0 => NFA_SIZE_LIMIT,
        limit => usize::try_from(limit).unwrap_or(usize::MAX),
    }
}

/// How many compiled patterns an [`Env`](crate::Env) keeps, see
/// [`Env::set_regex_cache_options`](crate::Env::set_regex_cache_options).
///
/// The cache changes no result: a pattern is compiled exactly as without
/// it. Under an evaluation budget, a pattern found in the cache is charged
/// the lookup and the match, but not its parsing and compiling again, and a
/// pattern is kept only once an evaluation has paid for compiling it.
///
/// # Memory
///
/// The cache holds the compiled automata of up to
/// [`capacity`](Self::with_capacity) patterns, in all at most
/// [`max_bytes`](Self::with_max_bytes), plus the pinned ones (see
/// [`RegexCache::prewarm`]), which count against neither.
///
/// Each kept regex also keeps the scratch space of its matches: a pool of
/// caches, one for each thread that matched it at the same time as
/// another, which stay with the regex. A cache holds a lazy DFA of up to
/// 2 MiB, or the size limit when smaller, and the state of the other
/// engines, about the size of the automaton; about 0.6 MiB per pattern and
/// thread was measured for patterns near 1 MiB. This memory is not counted
/// by `max_bytes`. With `T` threads matching, the cache can so hold up to
///
/// `max_bytes + capacity × T × (2 MiB + automaton)`, plus the pinned patterns.
///
/// Without an evaluation budget, patterns compile under the `regex` crate's
/// default limits, so an automaton is up to 10 MiB: with the defaults, at
/// most 64 MiB of automata, plus 64 × `T` × ~2 MiB of scratch space. Under
/// a budget, a pattern is kept only once its compile was paid for, which a
/// 10k-step budget allows up to ~80 KB of automaton.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RegexCacheOptions {
    capacity: usize,
    max_pattern_len: usize,
    max_bytes: u64,
}

impl Default for RegexCacheOptions {
    /// 64 patterns, of up to 4,096 bytes each, with up to 64 MiB of
    /// automata in all.
    fn default() -> Self {
        RegexCacheOptions {
            capacity: 64,
            max_pattern_len: 4096,
            max_bytes: 64 << 20,
        }
    }
}

impl RegexCacheOptions {
    /// How many compiled patterns the cache keeps at most. Zero turns the
    /// cache off: every call compiles its pattern.
    ///
    /// A compiled pattern takes up to the size limit it was compiled under
    /// (10 MiB by default, see
    /// [`RuntimeOptions::with_regex_size_limit`](crate::RuntimeOptions::with_regex_size_limit)),
    /// plus its scratch space, see [Memory](Self#memory).
    ///
    /// Making room scans the entries, under the lock insertions take: a
    /// capacity in the low thousands at most is sensible.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// How many compiled patterns the cache keeps at most.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The longest pattern, in bytes, the cache keeps: a longer one is
    /// compiled on every call.
    pub fn with_max_pattern_len(mut self, max_pattern_len: usize) -> Self {
        self.max_pattern_len = max_pattern_len;
        self
    }

    /// The longest pattern, in bytes, the cache keeps.
    pub fn max_pattern_len(&self) -> usize {
        self.max_pattern_len
    }

    /// How many bytes of compiled automata the cache keeps at most, in all:
    /// the sum of their sizes, an invalid pattern counting its error message.
    /// The entries used least recently are evicted to stay under it, and a
    /// pattern larger on its own is not kept. Pinned patterns do not count,
    /// and neither does the scratch space of matches, see
    /// [Memory](Self#memory).
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// How many bytes of compiled automata the cache keeps at most.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }
}

/// The compiled regular expressions of an [`Env`](crate::Env)'s `matches`,
/// shared by every evaluation under it, see [`RegexCacheOptions`].
pub struct RegexCache {
    options: RegexCacheOptions,
    /// The last tick handed out, see [`next_tick`](Self::next_tick).
    tick: AtomicU64,
    /// The entries, as insertions and evictions see them: each is decided
    /// under this lock, and then applied to every replica.
    state: Mutex<State>,
    /// Whether any entry is pinned, read by lookups without the mutex.
    any_pinned: AtomicBool,
    /// Copies of the index of the entries, one per group of threads. A hit
    /// takes its own replica's lock, shared: hits on threads of different
    /// groups touch no common lock, and hits in one group do not exclude
    /// each other.
    replicas: Box<[Replica]>,
}

/// The entries of each pattern, one per size limit.
type Index = HashMap<Arc<str>, Vec<Arc<Slot>>>;

/// A replica of the index, on a cache line of its own.
#[derive(Default)]
#[repr(align(128))]
struct Replica(RwLock<Index>);

#[derive(Default)]
struct State {
    index: Index,
    /// The entries that can be evicted.
    len: usize,
    /// The bytes of the entries that can be evicted, see [`Slot::bytes`].
    bytes: u64,
    /// The pinned entries, which are never evicted.
    pinned: usize,
}

fn find<'a>(index: &'a Index, pattern: &str, size_limit: usize) -> Option<&'a Arc<Slot>> {
    index
        .get(pattern)?
        .iter()
        .find(|slot| slot.size_limit == size_limit)
}

impl State {
    /// Removes the entry used least recently, the earliest inserted first
    /// among those used as recently, which makes it one entry, whatever the
    /// order of the map. The cache is not empty.
    fn evict_least_recently_used(&mut self) -> (Arc<str>, Arc<Slot>) {
        let (pattern, index) = self
            .index
            .iter()
            .flat_map(|(pattern, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| !slot.pinned)
                    .map(move |(index, slot)| ((slot.last_used(), slot.inserted), pattern, index))
            })
            .min_by_key(|(recency, _, _)| *recency)
            .map(|(_, pattern, index)| (pattern.clone(), index))
            .expect("a full cache has an entry it can evict");
        let slot = remove(&mut self.index, &pattern, |slots| slots.swap_remove(index));
        self.len -= 1;
        self.bytes -= slot.bytes;
        (pattern, slot)
    }
}

/// Removes the slot `take` picks from the entries of `pattern`, and the
/// pattern with its last entry.
fn remove(
    index: &mut Index,
    pattern: &str,
    take: impl FnOnce(&mut Vec<Arc<Slot>>) -> Arc<Slot>,
) -> Arc<Slot> {
    let slots = index.get_mut(pattern).expect("the pattern is kept");
    let slot = take(slots);
    if slots.is_empty() {
        index.remove(pattern);
    }
    slot
}

struct Slot {
    size_limit: usize,
    compiled: Compiled,
    /// Kept by [`RegexCache::prewarm`]: never evicted.
    pinned: bool,
    /// What the entry counts against [`RegexCacheOptions::with_max_bytes`]:
    /// the size of the automaton, or of the error message.
    bytes: u64,
    /// The tick the entry was inserted at, unique.
    inserted: u64,
    /// When the entry was last used: twice the tick of its insertion, or
    /// twice the tick current at a hit, plus one.
    last_used: AtomicU64,
}

impl Slot {
    fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }

    /// Marks the entry used now, `tick` being the latest tick handed out:
    /// as recent as any use since that insertion, and more than it. Writes
    /// only the first time in a tick, so hits mostly only read.
    fn used_at(&self, tick: u64) {
        let stamp = tick.saturating_mul(2) | 1;
        if self.last_used() < stamp {
            self.last_used.fetch_max(stamp, Ordering::Relaxed);
        }
    }
}

/// A pattern compiled under a size limit: the regex, or the error `matches`
/// reports for it.
#[derive(Clone)]
pub(crate) struct Compiled {
    regex: Result<Arc<meta::Regex>, Arc<str>>,
    /// The bytes of automaton compiling built: the regex's size, the size
    /// limit when it was exceeded, zero when the pattern is invalid.
    built: u64,
    /// How many Unicode word-boundary assertions the automaton holds, see
    /// [`unicode_word_looks`]: with any, the lazy DFA cannot scan a
    /// non-ASCII subject.
    word_looks: u64,
}

impl Compiled {
    /// The bytes the entry holds: the size of the automaton, or of the
    /// error message.
    fn bytes(&self) -> u64 {
        match &self.regex {
            Ok(_) => self.built,
            Err(message) => message.len() as u64,
        }
    }

    /// The bytes of automaton compiling built, see the field.
    pub(crate) fn built(&self) -> u64 {
        self.built
    }

    /// The steps of matching the regex against `subject`, charged before
    /// matching, at about 80 ns a step.
    ///
    /// The lazy DFA scans a subject at a cost that grows with the automaton
    /// it builds: two steps per KiB of automaton per 64 bytes of subject
    /// (rounded up), as a match can cost up to the automaton's states times
    /// the subject. It cannot evaluate a Unicode word boundary (`\b`, `\B`,
    /// `\b{start}`, ...) on a non-ASCII byte, though, and then quits: a
    /// slower engine scans the subject instead, at up to ~100 ns a byte plus
    /// ~35 ns a byte per word-boundary assertion of the automaton, each
    /// evaluated at every position by decoding the characters around it. A
    /// non-ASCII subject is so charged at least 1.25 steps per byte plus
    /// half a step per byte per assertion.
    pub(crate) fn scan_steps(&self, subject: &str) -> u64 {
        let len = subject.len() as u64;
        let scan = (self.built / 1024 + 1)
            .saturating_mul(len / 64 + 1)
            .saturating_mul(2);
        if self.word_looks == 0 || subject.is_ascii() {
            return scan;
        }
        let per_four_bytes = self.word_looks.saturating_mul(2).saturating_add(5);
        let slow = len.saturating_mul(per_four_bytes).div_ceil(4);
        scan.max(slow)
    }

    /// The regex, or the error `matches` reports for the pattern.
    pub(crate) fn regex(&self) -> Result<&meta::Regex, ExecutionError> {
        match &self.regex {
            Ok(regex) => Ok(regex),
            Err(message) => Err(ExecutionError::function_error("matches", message)),
        }
    }

    /// Whether the regex matches `subject`, as `regex::Regex::is_match`.
    pub(crate) fn is_match(&self, subject: &str) -> Result<bool, ExecutionError> {
        self.regex().map(|regex| regex.is_match(subject))
    }
}

impl Default for RegexCache {
    fn default() -> Self {
        RegexCache::new(RegexCacheOptions::default())
    }
}

/// The group of threads this thread belongs to: a number given out in
/// turn, at the thread's first hit.
fn thread_group() -> usize {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static GROUP: usize = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    GROUP.with(|group| *group)
}

impl RegexCache {
    /// An empty cache.
    pub fn new(options: RegexCacheOptions) -> Self {
        // a replica per core: threads beyond share them
        let replicas = std::thread::available_parallelism()
            .map_or(1, |cores| cores.get())
            .clamp(1, 64);
        RegexCache {
            options,
            tick: AtomicU64::new(0),
            state: Mutex::default(),
            any_pinned: AtomicBool::new(false),
            replicas: (0..replicas).map(|_| Replica::default()).collect(),
        }
    }

    /// The options the cache was made with.
    pub fn options(&self) -> &RegexCacheOptions {
        &self.options
    }

    /// How many compiled patterns the cache holds, counting a pattern once
    /// per size limit it was compiled under, the pinned ones included.
    pub fn len(&self) -> usize {
        let state = self.state();
        state.len + state.pinned
    }

    /// How many of the compiled patterns are pinned, see
    /// [`prewarm`](Self::prewarm).
    pub fn pinned_len(&self) -> usize {
        self.state().pinned
    }

    /// Whether the cache holds no compiled pattern.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every compiled pattern, the pinned ones included.
    pub fn clear(&self) {
        let mut state = self.state();
        self.any_pinned.store(false, Ordering::Relaxed);
        let mut evicted = vec![std::mem::take(&mut state.index)];
        state.len = 0;
        state.bytes = 0;
        state.pinned = 0;
        for replica in self.replicas.iter() {
            evicted.push(std::mem::take(&mut *write(replica)));
        }
        drop(state);
        drop(evicted);
    }

    /// Compiles `pattern` as `matches` does in an evaluation with a
    /// [`RuntimeOptions::with_regex_size_limit`](crate::RuntimeOptions::with_regex_size_limit)
    /// of `regex_size_limit` (zero meaning the `regex` crate's default), and
    /// pins it: every such evaluation that matches it is then charged as a
    /// hit, whatever other evaluations add to the cache.
    ///
    /// A pinned pattern is never evicted, and counts neither against the
    /// [capacity](RegexCacheOptions::with_capacity) nor against the
    /// [bytes](RegexCacheOptions::with_max_bytes) of the cache, whatever its
    /// size or length: only [`unpin`](Self::unpin) and [`clear`](Self::clear)
    /// drop it. Pin only the patterns you trust, such as those of the rules
    /// you load.
    ///
    /// Under a budget, a pattern that is not pinned is charged its compile
    /// whenever the cache does not hold it, which any evaluation sharing the
    /// Env can cause by filling the cache: size budgets for that cold case,
    /// unless every pattern the expression matches is pinned.
    ///
    /// # Errors
    ///
    /// The error `matches` reports for the pattern, which is pinned too.
    ///
    /// # Example
    /// ```
    /// use cel::{Context, Env, Program, RuntimeOptions};
    /// use std::sync::Arc;
    ///
    /// let env = Env::stdlib();
    /// // the patterns of the rules, compiled as their evaluations will
    /// env.regex_cache().prewarm("^/api/v[0-9]+/", 1 << 20).unwrap();
    /// assert!(env.regex_cache().prewarm("(", 1 << 20).is_err());
    ///
    /// let mut ctx = Context::with_env(Arc::new(env));
    /// ctx.set_budget(RuntimeOptions::default().with_max_steps(100).with_regex_size_limit(1 << 20));
    /// let program = Program::compile("'/api/v1/users'.matches('^/api/v[0-9]+/')").unwrap();
    /// assert_eq!(program.execute(&ctx), Ok(true.into()));
    /// ```
    pub fn prewarm(&self, pattern: &str, regex_size_limit: u64) -> Result<(), ExecutionError> {
        let size_limit = size_limit(regex_size_limit);
        let kept = find(&self.state().index, pattern, size_limit).map(|slot| slot.compiled.clone());
        let compiled = kept.unwrap_or_else(|| compile(pattern, size_limit));
        self.pin(pattern, size_limit, compiled).regex().map(|_| ())
    }

    /// Unpins `pattern` under `regex_size_limit` (see [`prewarm`](Self::prewarm))
    /// and drops it. Returns whether it was pinned.
    pub fn unpin(&self, pattern: &str, regex_size_limit: u64) -> bool {
        let size_limit = size_limit(regex_size_limit);
        let mut state = self.state();
        let slot = match find(&state.index, pattern, size_limit) {
            Some(slot) if slot.pinned => slot.clone(),
            _ => return false,
        };
        let pattern = remove_slot(&mut state.index, pattern, &slot);
        state.pinned -= 1;
        self.any_pinned.store(state.pinned > 0, Ordering::Relaxed);
        self.apply(&[(pattern, slot)], None);
        true
    }

    /// Keeps `compiled` pinned, replacing the entry of `pattern` under
    /// `size_limit` if it is not, and hands back the pinned entry.
    fn pin(&self, pattern: &str, size_limit: usize, compiled: Compiled) -> Compiled {
        let mut state = self.state();
        let mut removed = Vec::new();
        if let Some(slot) = find(&state.index, pattern, size_limit) {
            if slot.pinned {
                return slot.compiled.clone();
            }
            let slot = slot.clone();
            state.len -= 1;
            state.bytes -= slot.bytes;
            removed.push((remove_slot(&mut state.index, pattern, &slot), slot));
        }
        let pattern = self.add(&mut state, pattern, size_limit, compiled.clone(), true);
        state.pinned += 1;
        self.any_pinned.store(true, Ordering::Relaxed);
        self.apply(&removed, Some(&pattern));
        drop(state);
        drop(removed);
        compiled
    }

    // The maps are consistent between statements: a panic cannot leave one
    // half-updated, so a poisoned lock is used as is.
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// This thread's replica.
    fn replica(&self) -> RwLockReadGuard<'_, Index> {
        let replica = &self.replicas[thread_group() % self.replicas.len()];
        replica.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// A tick no other insertion has, later than every earlier one.
    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Whether the cache keeps `pattern`.
    fn keeps(&self, pattern: &str) -> bool {
        self.options.capacity > 0 && pattern.len() <= self.options.max_pattern_len
    }

    /// The compiled `pattern` under `size_limit`, if the cache holds it,
    /// marked as used. Takes this thread's replica shared, and only reads
    /// shared memory, but for the first hit on an entry since an insertion.
    pub(crate) fn lookup(&self, pattern: &str, size_limit: usize) -> Option<Compiled> {
        if !self.keeps(pattern) && !self.any_pinned.load(Ordering::Relaxed) {
            return None;
        }
        let index = self.replica();
        let slot = find(&index, pattern, size_limit)?;
        slot.used_at(self.tick.load(Ordering::Relaxed));
        Some(slot.compiled.clone())
    }

    /// Keeps `compiled`, the `pattern` compiled under `size_limit`, evicting
    /// the entries used least recently to make room, and hands back the
    /// entry the cache holds: a racing thread's, if it inserted first.
    pub(crate) fn insert(&self, pattern: &str, size_limit: usize, compiled: Compiled) -> Compiled {
        if !self.keeps(pattern) {
            return compiled;
        }
        let mut state = self.state();
        if let Some(slot) = find(&state.index, pattern, size_limit) {
            slot.used_at(self.tick.load(Ordering::Relaxed));
            return slot.compiled.clone();
        }
        let bytes = compiled.bytes();
        if bytes > self.options.max_bytes {
            return compiled;
        }
        let mut evicted = Vec::new();
        while state.len >= self.options.capacity || state.bytes + bytes > self.options.max_bytes {
            evicted.push(state.evict_least_recently_used());
        }
        let added = self.add(&mut state, pattern, size_limit, compiled.clone(), false);
        state.len += 1;
        state.bytes += bytes;
        self.apply(&evicted, Some(&added));
        // the evicted regexes are freed after the locks are released
        drop(state);
        drop(evicted);
        compiled
    }

    /// Adds a new entry to the index of `state`, and hands it back with its
    /// pattern, for [`apply`](Self::apply).
    fn add(
        &self,
        state: &mut State,
        pattern: &str,
        size_limit: usize,
        compiled: Compiled,
        pinned: bool,
    ) -> (Arc<str>, Arc<Slot>) {
        let tick = self.next_tick();
        let slot = Arc::new(Slot {
            size_limit,
            bytes: compiled.bytes(),
            compiled,
            pinned,
            inserted: tick,
            last_used: AtomicU64::new(tick.saturating_mul(2)),
        });
        let pattern: Arc<str> = match state.index.get_key_value(pattern) {
            Some((kept, _)) => kept.clone(),
            None => pattern.into(),
        };
        add(&mut state.index, &pattern, &slot);
        (pattern, slot)
    }

    /// Applies to every replica the removal of `removed` and the addition
    /// of `added`, already made to the state, whose lock the caller holds.
    fn apply(&self, removed: &[(Arc<str>, Arc<Slot>)], added: Option<&(Arc<str>, Arc<Slot>)>) {
        for replica in self.replicas.iter() {
            let mut index = write(replica);
            for (pattern, slot) in removed {
                remove_slot(&mut index, pattern, slot);
            }
            if let Some((pattern, slot)) = added {
                add(&mut index, pattern, slot);
            }
        }
    }

    /// The compiled `pattern` under `size_limit`, compiling it, outside the
    /// locks, when the cache does not hold it.
    pub(crate) fn get(&self, pattern: &str, size_limit: usize) -> Compiled {
        match self.lookup(pattern, size_limit) {
            Some(compiled) => compiled,
            None => self.insert(pattern, size_limit, compile(pattern, size_limit)),
        }
    }
}

fn write(replica: &Replica) -> RwLockWriteGuard<'_, Index> {
    replica.0.write().unwrap_or_else(PoisonError::into_inner)
}

/// Removes `slot` from the entries of `pattern`, and hands back the pattern
/// as the index keeps it.
fn remove_slot(index: &mut Index, pattern: &str, slot: &Arc<Slot>) -> Arc<str> {
    let kept = index
        .get_key_value(pattern)
        .map(|(kept, _)| kept.clone())
        .expect("the pattern is kept");
    remove(index, pattern, |slots| {
        let at = slots
            .iter()
            .position(|other| Arc::ptr_eq(other, slot))
            .expect("the index holds the entry");
        slots.swap_remove(at)
    });
    kept
}

fn add(index: &mut Index, pattern: &Arc<str>, slot: &Arc<Slot>) {
    index.entry(pattern.clone()).or_default().push(slot.clone());
}
/// Compiles `pattern` with the configuration of `regex::Regex::new`, so the
/// result and the error messages are the same, under `size_limit`, which
/// also caps the lazy DFA's cache.
pub(crate) fn compile(pattern: &str, size_limit: usize) -> Compiled {
    // parsed as `meta::Builder::build` parses it, so the error is the same,
    // and the syntax looked at once before it is compiled
    let hir = match syntax::parse_with(pattern, &syntax::Config::new().utf8(true)) {
        Ok(hir) => hir,
        Err(err) => {
            return Compiled {
                regex: Err(format!("'{pattern}' not a valid regex:\n{err}").into()),
                built: 0,
                word_looks: 0,
            }
        }
    };
    let built = meta::Builder::new()
        .configure(
            meta::Config::new()
                .nfa_size_limit(Some(size_limit))
                .hybrid_cache_capacity(size_limit.min(DFA_SIZE_LIMIT))
                .match_kind(MatchKind::LeftmostFirst)
                .utf8_empty(true),
        )
        .build_from_hir(&hir);
    match built {
        Ok(regex) => Compiled {
            built: regex.memory_usage() as u64,
            regex: Ok(Arc::new(regex)),
            word_looks: unicode_word_looks(&hir),
        },
        Err(err) => {
            // as `regex::Error` reports a `meta::BuildError`
            let (built, message) = match (err.size_limit(), err.syntax_error()) {
                (Some(limit), _) => (
                    limit as u64,
                    format!("Compiled regex exceeds size limit of {limit} bytes."),
                ),
                (None, Some(syntax)) => (0, syntax.to_string()),
                (None, None) => (0, err.to_string()),
            };
            Compiled {
                regex: Err(format!("'{pattern}' not a valid regex:\n{message}").into()),
                built,
                word_looks: 0,
            }
        }
    }
}

/// How many Unicode word-boundary assertions the automaton compiled from
/// `hir` holds: a bounded repetition holds as many copies of its body as its
/// maximum, an unbounded one one more than its minimum.
fn unicode_word_looks(hir: &Hir) -> u64 {
    if !hir.properties().look_set().contains_word_unicode() {
        return 0;
    }
    let mut looks = 0u64;
    let mut walk = vec![(hir, 1u64)];
    while let Some((hir, copies)) = walk.pop() {
        match hir.kind() {
            HirKind::Look(look) => {
                if LookSet::singleton(*look).contains_word_unicode() {
                    looks = looks.saturating_add(copies);
                }
            }
            HirKind::Repetition(repetition) => {
                let body = repetition
                    .max
                    .map_or(u64::from(repetition.min).saturating_add(1), u64::from);
                walk.push((&repetition.sub, copies.saturating_mul(body)));
            }
            HirKind::Capture(capture) => walk.push((&capture.sub, copies)),
            HirKind::Concat(hirs) | HirKind::Alternation(hirs) => {
                walk.extend(hirs.iter().map(|hir| (hir, copies)));
            }
            HirKind::Empty | HirKind::Literal(_) | HirKind::Class(_) => {}
        }
    }
    looks
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT: usize = NFA_SIZE_LIMIT;

    fn same(a: &Compiled, b: &Compiled) -> bool {
        match (&a.regex, &b.regex) {
            (Ok(a), Ok(b)) => Arc::ptr_eq(a, b),
            (Err(a), Err(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    fn keys(cache: &RegexCache) -> Vec<String> {
        let state = cache.state();
        let mut keys: Vec<String> = state
            .index
            .iter()
            .flat_map(|(pattern, slots)| {
                slots
                    .iter()
                    .map(move |slot| format!("{pattern}@{}", slot.size_limit))
            })
            .collect();
        keys.sort();
        // every replica holds the same entries
        for replica in cache.replicas.iter() {
            let index = replica.0.read().unwrap();
            let mut replicated: Vec<String> = index
                .iter()
                .flat_map(|(pattern, slots)| {
                    slots
                        .iter()
                        .map(move |slot| format!("{pattern}@{}", slot.size_limit))
                })
                .collect();
            replicated.sort();
            assert_eq!(replicated, keys);
        }
        keys
    }

    #[test]
    fn a_hit_reuses_the_compiled_regex() {
        let cache = RegexCache::default();
        let first = cache.get("a.c", DEFAULT);
        for _ in 0..1_000 {
            let hit = cache.get("a.c", DEFAULT);
            assert!(same(&first, &hit));
            assert_eq!(hit.is_match("abc"), Ok(true));
        }
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn the_size_limit_is_part_of_the_key() {
        let cache = RegexCache::default();
        // `\w{10}` compiles to ~561 KB
        assert_eq!(cache.get(r"\w{10}", DEFAULT).is_match("a"), Ok(false));
        assert_eq!(
            cache.get(r"\w{10}", 16 * 1024).is_match("a"),
            Err(ExecutionError::function_error(
                "matches",
                "'\\w{10}' not a valid regex:\nCompiled regex exceeds size limit of 16384 bytes."
            ))
        );
        assert_eq!(cache.get(r"\w{10}", 16 * 1024).built(), 16 * 1024);
        assert_eq!(cache.len(), 2);
        assert_eq!(
            keys(&cache),
            [&format!(r"\w{{10}}@{DEFAULT}"), r"\w{10}@16384"]
        );
    }

    #[test]
    fn eviction_drops_the_least_recently_used() {
        let cache = RegexCache::new(RegexCacheOptions::default().with_capacity(2));
        let a = cache.get("a", DEFAULT);
        cache.get("b", DEFAULT);
        assert!(same(&a, &cache.get("a", DEFAULT)));
        cache.get("c", DEFAULT);
        assert_eq!(
            keys(&cache),
            [format!("a@{DEFAULT}"), format!("c@{DEFAULT}")]
        );
        // the same sequence always evicts the same entries
        for _ in 0..100 {
            let cache = RegexCache::new(RegexCacheOptions::default().with_capacity(3));
            for pattern in ["a", "b", "c", "a", "d", "b", "e", "a", "f"] {
                cache.get(pattern, DEFAULT);
            }
            assert_eq!(
                keys(&cache),
                [
                    format!("a@{DEFAULT}"),
                    format!("e@{DEFAULT}"),
                    format!("f@{DEFAULT}")
                ]
            );
        }
    }

    #[test]
    fn an_invalid_pattern_is_kept_as_its_error() {
        let cache = RegexCache::default();
        let first = cache.get("(", DEFAULT);
        // through a variable: clippy rejects the literal invalid pattern
        let pattern = "(";
        let expected = regex::Regex::new(pattern).unwrap_err();
        assert_eq!(
            first.is_match("a"),
            Err(ExecutionError::function_error(
                "matches",
                format!("'(' not a valid regex:\n{expected}")
            ))
        );
        assert_eq!(first.built(), 0);
        assert!(same(&first, &cache.get("(", DEFAULT)));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn an_over_limit_pattern_is_an_error_not_a_panic() {
        let cache = RegexCache::default();
        let tiny = cache.get(r"\w{100}", 1024);
        assert!(tiny.is_match("a").is_err());
        assert!(same(&tiny, &cache.get(r"\w{100}", 1024)));
    }

    #[test]
    fn matching_happens_outside_the_lock() {
        let cache = RegexCache::default();
        let compiled = cache.get("^a+$", DEFAULT);
        // the entry is held, and the lock is free
        let guard = cache.state.try_lock();
        assert!(guard.is_ok());
        drop(guard);
        for replica in cache.replicas.iter() {
            assert!(replica.0.try_write().is_ok());
        }
        assert_eq!(compiled.is_match(&"a".repeat(1 << 16)), Ok(true));
        // and another thread can use the cache meanwhile
        std::thread::scope(|scope| {
            scope.spawn(|| assert_eq!(cache.get("b", DEFAULT).is_match("b"), Ok(true)));
        });
        assert_eq!(compiled.is_match("b"), Ok(false));
    }

    #[test]
    fn a_hit_takes_no_exclusive_lock() {
        let cache = RegexCache::default();
        let first = cache.get("a.c", DEFAULT);
        // an insertion is under way, and other hits hold every replica
        let inserting = cache.state();
        let held: Vec<_> = cache
            .replicas
            .iter()
            .map(|replica| replica.0.read().unwrap())
            .collect();
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| send.send(cache.lookup("a.c", DEFAULT)).unwrap());
            let hit = receive
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("a hit waits for no other use")
                .expect("a hit");
            assert!(same(&first, &hit));
        });
        drop(held);
        drop(inserting);
    }

    #[test]
    fn a_hit_marks_its_entry_once_per_tick() {
        let cache = RegexCache::default();
        cache.get("a", DEFAULT);
        let state = cache.state();
        let slot = find(&state.index, "a", DEFAULT).unwrap().clone();
        drop(state);
        // inserted at tick 1
        assert_eq!(slot.last_used(), 2);
        cache.lookup("a", DEFAULT);
        assert_eq!(slot.last_used(), 3);
        // a racing hit that saw an earlier tick does not go back
        slot.used_at(0);
        assert_eq!(slot.last_used(), 3);
        cache.get("b", DEFAULT);
        cache.lookup("a", DEFAULT);
        assert_eq!(slot.last_used(), 5);
    }

    #[test]
    fn hits_since_the_last_insertion_count_as_equally_recent() {
        let cache = RegexCache::new(RegexCacheOptions::default().with_capacity(3));
        for pattern in ["a", "b", "c", "c", "b", "a", "d"] {
            cache.get(pattern, DEFAULT);
        }
        // `a`, `b` and `c` were all used since `c` was inserted: the
        // earliest inserted of them goes
        assert_eq!(
            keys(&cache),
            [
                format!("b@{DEFAULT}"),
                format!("c@{DEFAULT}"),
                format!("d@{DEFAULT}")
            ]
        );
    }

    #[test]
    fn a_racing_insert_keeps_the_first_entry() {
        let cache = RegexCache::default();
        let first = cache.insert("a", DEFAULT, compile("a", DEFAULT));
        let second = cache.insert("a", DEFAULT, compile("a", DEFAULT));
        assert!(same(&first, &second));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn zero_capacity_and_long_patterns_are_not_kept() {
        let cache = RegexCache::new(RegexCacheOptions::default().with_capacity(0));
        let first = cache.get("a", DEFAULT);
        assert!(!same(&first, &cache.get("a", DEFAULT)));
        assert_eq!(cache.get("a", DEFAULT).is_match("a"), Ok(true));
        assert!(cache.is_empty());

        let cache = RegexCache::new(RegexCacheOptions::default().with_max_pattern_len(4));
        assert_eq!(cache.get("aaaaa", DEFAULT).is_match("aaaaa"), Ok(true));
        cache.get("aaaa", DEFAULT);
        assert_eq!(keys(&cache), [format!("aaaa@{DEFAULT}")]);
    }

    #[test]
    fn many_threads_share_the_cache() {
        let cache = RegexCache::new(RegexCacheOptions::default().with_capacity(16));
        std::thread::scope(|scope| {
            for t in 0..8 {
                let cache = &cache;
                scope.spawn(move || {
                    for i in 0..10_000 {
                        let n = (i * 7 + t) % 100;
                        let compiled = cache.get(&format!("^a{{{n}}}$"), DEFAULT);
                        let subject = "a".repeat(n + i % 2);
                        assert_eq!(compiled.is_match(&subject), Ok(i % 2 == 0));
                    }
                });
            }
        });
        assert!(cache.len() <= 16);
        let len = cache.state().index.values().map(Vec::len).sum::<usize>();
        assert_eq!(cache.len(), len);
        keys(&cache);
    }

    #[test]
    fn a_pinned_entry_is_never_evicted() {
        let cache = RegexCache::new(RegexCacheOptions::default().with_capacity(2));
        cache.prewarm("p", 0).unwrap();
        let pinned = cache.get("p", DEFAULT);
        for i in 0..100 {
            cache.get(&format!("x{i}"), DEFAULT);
        }
        assert!(same(&pinned, &cache.lookup("p", DEFAULT).unwrap()));
        // outside the capacity
        assert_eq!((cache.len(), cache.pinned_len()), (3, 1));
        assert_eq!(
            keys(&cache),
            [
                format!("p@{DEFAULT}"),
                format!("x98@{DEFAULT}"),
                format!("x99@{DEFAULT}")
            ]
        );
    }

    #[test]
    fn pinning_takes_over_a_kept_entry() {
        let cache = RegexCache::new(RegexCacheOptions::default().with_capacity(2));
        let kept = cache.get("p", DEFAULT);
        cache.prewarm("p", 0).unwrap();
        // the same regex, compiled once, now pinned
        assert!(same(&kept, &cache.lookup("p", DEFAULT).unwrap()));
        assert_eq!((cache.len(), cache.pinned_len()), (1, 1));
        cache.prewarm("p", 0).unwrap();
        assert_eq!((cache.len(), cache.pinned_len()), (1, 1));
        // an invalid pattern is pinned as its error
        assert!(cache.prewarm("(", 0).is_err());
        assert_eq!(cache.pinned_len(), 2);
        assert_eq!(cache.get("(", DEFAULT).built(), 0);
    }

    #[test]
    fn pins_bypass_the_capacity_and_the_pattern_length() {
        let cache = RegexCache::new(
            RegexCacheOptions::default()
                .with_capacity(0)
                .with_max_pattern_len(2),
        );
        cache.prewarm("long pattern", 0).unwrap();
        assert!(cache.lookup("long pattern", DEFAULT).is_some());
        // and nothing else is kept
        cache.get("other pattern", DEFAULT);
        assert_eq!((cache.len(), cache.pinned_len()), (1, 1));
    }

    #[test]
    fn unpin_and_clear_drop_pins() {
        let cache = RegexCache::default();
        cache.prewarm("p", 0).unwrap();
        cache.prewarm("p", 1024).unwrap();
        assert!(!cache.unpin("q", 0));
        assert!(cache.unpin("p", 0));
        assert!(!cache.unpin("p", 0));
        assert!(cache.lookup("p", DEFAULT).is_none());
        assert_eq!(keys(&cache), ["p@1024"]);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.pinned_len(), 0);
    }

    #[test]
    fn the_bytes_of_the_automata_are_capped() {
        let size = |pattern: &str| compile(pattern, DEFAULT).bytes();
        let (a, b, c) = (size("a+"), size("b+"), size("c+"));
        let cache = RegexCache::new(RegexCacheOptions::default().with_max_bytes(a + b));
        cache.get("a+", DEFAULT);
        cache.get("b+", DEFAULT);
        assert_eq!(cache.len(), 2);
        // `c+` needs the room of the least recently used, `a+`
        cache.get("c+", DEFAULT);
        assert_eq!(
            keys(&cache),
            [format!("b+@{DEFAULT}"), format!("c+@{DEFAULT}")]
        );
        assert_eq!(cache.state().bytes, b + c);
        // a pattern larger than the cap on its own is not kept
        assert!(size(r"\w{10}") > a + b);
        assert_eq!(cache.get(r"\w{10}", DEFAULT).is_match("a"), Ok(false));
        assert_eq!(
            keys(&cache),
            [format!("b+@{DEFAULT}"), format!("c+@{DEFAULT}")]
        );
        // pinned patterns are not counted
        cache.prewarm(r"\w{10}", 0).unwrap();
        assert_eq!(cache.state().bytes, b + c);
        assert_eq!(cache.len(), 3);
        // an error counts its message
        let error = compile("(", DEFAULT);
        assert_eq!(
            error.bytes(),
            "'(' not a valid regex:\n".len() as u64 + {
                let pattern = "(";
                regex::Regex::new(pattern).unwrap_err().to_string().len() as u64
            }
        );
        cache.clear();
        assert_eq!(cache.state().bytes, 0);
    }

    #[test]
    fn unicode_word_boundaries_are_counted_per_copy() {
        let looks = |pattern: &str| compile(pattern, DEFAULT).word_looks;
        assert_eq!(looks(r"^/api/v[0-9]+$"), 0);
        // ASCII word boundaries leave the lazy DFA able to scan
        assert_eq!(looks(r"(?-u:\b)x(?-u:\B)"), 0);
        assert_eq!(looks(r"\b\w+\b"), 2);
        assert_eq!(looks(r"\b{start}x\b{end-half}"), 2);
        assert_eq!(looks(r"(?:\b\B|\B\b|\b\b\B|\B\B\b)"), 10);
        assert_eq!(looks(r"(\bx\B){3}"), 6);
        assert_eq!(looks(r"(?:\bx\B){2,}"), 6);
        assert_eq!(looks(r"(?:\bx)*"), 1);
        // regex-syntax keeps one copy of a repeated empty-width body
        assert_eq!(looks(r"(\b\B){3}"), 2);
        // an invalid pattern holds none
        assert_eq!(looks(r"\b("), 0);
    }

    #[test]
    fn a_non_ascii_subject_pays_for_the_slow_scan() {
        let pattern = compile(r"(?:\b\B|\B\b|\b\b\B|\B\B\b)", DEFAULT);
        let ascii = "a".repeat(8 << 10);
        let mixed = "é".repeat(4 << 10);
        // ~400 ns a byte measured: over 40,000 steps for 8 KiB
        assert!(
            pattern.scan_steps(&mixed) > 40_000,
            "{}",
            pattern.scan_steps(&mixed)
        );
        assert!(pattern.scan_steps(&ascii) < 1_000);
        // a pattern without Unicode word boundaries is charged as before
        let plain = compile(r"[a-z]+x", DEFAULT);
        assert_eq!(plain.scan_steps(&mixed), plain.scan_steps(&ascii));
    }

    #[test]
    fn errors_are_reported_as_meta_reports_them() {
        for pattern in ["(", r"\p{Bogus}", "a{2,1}", "(?P<x>a)(?P<x>b)"] {
            let built = meta::Builder::new()
                .syntax(syntax::Config::new().utf8(true))
                .build(pattern)
                .unwrap_err();
            let message = built.syntax_error().unwrap().to_string();
            let Err(compiled) = compile(pattern, DEFAULT).regex else {
                panic!("{pattern} compiles");
            };
            assert_eq!(
                &*compiled,
                format!("'{pattern}' not a valid regex:\n{message}").as_str()
            );
        }
    }

    #[test]
    fn clear_drops_every_entry() {
        let cache = RegexCache::default();
        cache.get("a", DEFAULT);
        cache.clear();
        assert!(cache.is_empty());
    }
}
