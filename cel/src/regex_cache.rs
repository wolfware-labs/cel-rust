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
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
/// the lookup and the match, but not its parsing and compiling again.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RegexCacheOptions {
    capacity: usize,
    max_pattern_len: usize,
}

impl Default for RegexCacheOptions {
    /// 64 patterns, of up to 4,096 bytes each.
    fn default() -> Self {
        RegexCacheOptions {
            capacity: 64,
            max_pattern_len: 4096,
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
    /// plus, for each thread matching it at once, a lazy DFA cache of up to
    /// 2 MiB (or the size limit, when smaller).
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
    len: usize,
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
        let (pattern, index) =
            self.index
                .iter()
                .flat_map(|(pattern, slots)| {
                    slots.iter().enumerate().map(move |(index, slot)| {
                        ((slot.last_used(), slot.inserted), pattern, index)
                    })
                })
                .min_by_key(|(recency, _, _)| *recency)
                .map(|(_, pattern, index)| (pattern.clone(), index))
                .expect("a full cache has an entry");
        let slot = remove(&mut self.index, &pattern, |slots| slots.swap_remove(index));
        self.len -= 1;
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
}

impl Compiled {
    /// The bytes of automaton compiling built, see the field.
    pub(crate) fn built(&self) -> u64 {
        self.built
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
            replicas: (0..replicas).map(|_| Replica::default()).collect(),
        }
    }

    /// The options the cache was made with.
    pub fn options(&self) -> &RegexCacheOptions {
        &self.options
    }

    /// How many compiled patterns the cache holds, counting a pattern once
    /// per size limit it was compiled under.
    pub fn len(&self) -> usize {
        self.state().len
    }

    /// Whether the cache holds no compiled pattern.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every compiled pattern.
    pub fn clear(&self) {
        let mut state = self.state();
        let mut evicted = vec![std::mem::take(&mut *state)];
        for replica in self.replicas.iter() {
            evicted.push(State {
                index: std::mem::take(&mut *write(replica)),
                len: 0,
            });
        }
        drop(state);
        drop(evicted);
    }

    /// Compiles `pattern` as `matches` does in an evaluation with a
    /// [`RuntimeOptions::with_regex_size_limit`](crate::RuntimeOptions::with_regex_size_limit)
    /// of `regex_size_limit` (zero meaning the `regex` crate's default), and
    /// keeps it: the first such evaluation to match it is then charged as a
    /// hit. Patterns the cache does not keep (see [`RegexCacheOptions`]) are
    /// compiled and dropped.
    ///
    /// # Errors
    ///
    /// The error `matches` reports for the pattern, which is kept too.
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
        self.get(pattern, size_limit(regex_size_limit))
            .regex()
            .map(|_| ())
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
        if !self.keeps(pattern) {
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
        let mut evicted = Vec::new();
        while state.len >= self.options.capacity {
            evicted.push(state.evict_least_recently_used());
        }
        let tick = self.next_tick();
        let slot = Arc::new(Slot {
            size_limit,
            compiled: compiled.clone(),
            inserted: tick,
            last_used: AtomicU64::new(tick.saturating_mul(2)),
        });
        let pattern: Arc<str> = match state.index.get_key_value(pattern) {
            Some((kept, _)) => kept.clone(),
            None => pattern.into(),
        };
        add(&mut state.index, &pattern, &slot);
        state.len += 1;
        for replica in self.replicas.iter() {
            let mut index = write(replica);
            for (pattern, slot) in &evicted {
                remove(&mut index, pattern, |slots| {
                    let at = slots
                        .iter()
                        .position(|kept| Arc::ptr_eq(kept, slot))
                        .expect("the replica holds the entry");
                    slots.swap_remove(at)
                });
            }
            add(&mut index, &pattern, &slot);
        }
        // the evicted regexes are freed after the locks are released
        drop(state);
        drop(evicted);
        compiled
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

fn add(index: &mut Index, pattern: &Arc<str>, slot: &Arc<Slot>) {
    index.entry(pattern.clone()).or_default().push(slot.clone());
}
/// Compiles `pattern` with the configuration of `regex::Regex::new`, so the
/// result and the error messages are the same, under `size_limit`, which
/// also caps the lazy DFA's cache.
pub(crate) fn compile(pattern: &str, size_limit: usize) -> Compiled {
    let built = meta::Builder::new()
        .configure(
            meta::Config::new()
                .nfa_size_limit(Some(size_limit))
                .hybrid_cache_capacity(size_limit.min(DFA_SIZE_LIMIT))
                .match_kind(MatchKind::LeftmostFirst)
                .utf8_empty(true),
        )
        .syntax(syntax::Config::new().utf8(true))
        .build(pattern);
    match built {
        Ok(regex) => Compiled {
            built: regex.memory_usage() as u64,
            regex: Ok(Arc::new(regex)),
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
            }
        }
    }
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
    fn clear_drops_every_entry() {
        let cache = RegexCache::default();
        cache.get("a", DEFAULT);
        cache.clear();
        assert!(cache.is_empty());
    }
}
