//! The regular expressions `matches` compiled, kept by the [`Env`](crate::Env)
//! for the next call of the same pattern.
//!
//! An entry is keyed by the pattern and the size limit it was compiled
//! under, so a pattern compiled under one limit is never served under
//! another: a smaller limit could refuse it, and the lazy DFA's cache is
//! sized by the limit too. A pattern that fails to compile, for its syntax
//! or its size, is kept as its error.
//!
//! Compiling happens outside the lock, and so does matching: a lookup only
//! clones the entry's `Arc`. When the cache is full, the entry used least
//! recently is evicted: every lookup and insertion takes a unique tick, so
//! the choice is deterministic.

use crate::ExecutionError;
use regex_automata::{meta, util::syntax, MatchKind};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

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
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// The entries of each pattern, one per size limit.
    map: HashMap<Arc<str>, Vec<Slot>>,
    len: usize,
    tick: u64,
}

impl Inner {
    /// A tick no other use of the cache has.
    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn slot(&mut self, pattern: &str, size_limit: usize) -> Option<&mut Slot> {
        self.map
            .get_mut(pattern)?
            .iter_mut()
            .find(|slot| slot.size_limit == size_limit)
    }

    /// Removes the entry used least recently, which the unique ticks make
    /// one entry, whatever the order of the map. The cache is not empty.
    fn evict_least_recently_used(&mut self) -> Compiled {
        let (pattern, index) = self
            .map
            .iter()
            .flat_map(|(pattern, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .map(move |(index, slot)| (slot.last_used, pattern, index))
            })
            .min_by_key(|(last_used, _, _)| *last_used)
            .map(|(_, pattern, index)| (pattern.clone(), index))
            .expect("a full cache has an entry");
        let slots = self.map.get_mut(&pattern).expect("the pattern is kept");
        let slot = slots.swap_remove(index);
        if slots.is_empty() {
            self.map.remove(&pattern);
        }
        self.len -= 1;
        slot.compiled
    }
}

struct Slot {
    size_limit: usize,
    compiled: Compiled,
    last_used: u64,
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

impl RegexCache {
    /// An empty cache.
    pub fn new(options: RegexCacheOptions) -> Self {
        RegexCache {
            options,
            inner: Mutex::default(),
        }
    }

    /// The options the cache was made with.
    pub fn options(&self) -> &RegexCacheOptions {
        &self.options
    }

    /// How many compiled patterns the cache holds, counting a pattern once
    /// per size limit it was compiled under.
    pub fn len(&self) -> usize {
        self.lock().len
    }

    /// Whether the cache holds no compiled pattern.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every compiled pattern.
    pub fn clear(&self) {
        let evicted = std::mem::take(&mut *self.lock());
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

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // the map is consistent between statements: a panic cannot leave it
        // half-updated
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether the cache keeps `pattern`.
    fn keeps(&self, pattern: &str) -> bool {
        self.options.capacity > 0 && pattern.len() <= self.options.max_pattern_len
    }

    /// The compiled `pattern` under `size_limit`, if the cache holds it,
    /// marked as used.
    pub(crate) fn lookup(&self, pattern: &str, size_limit: usize) -> Option<Compiled> {
        if !self.keeps(pattern) {
            return None;
        }
        let mut inner = self.lock();
        let tick = inner.next_tick();
        let slot = inner.slot(pattern, size_limit)?;
        slot.last_used = tick;
        Some(slot.compiled.clone())
    }

    /// Keeps `compiled`, the `pattern` compiled under `size_limit`, evicting
    /// the entries used least recently to make room, and hands back the
    /// entry the cache holds: a racing thread's, if it inserted first.
    pub(crate) fn insert(&self, pattern: &str, size_limit: usize, compiled: Compiled) -> Compiled {
        if !self.keeps(pattern) {
            return compiled;
        }
        let mut evicted = Vec::new();
        let mut inner = self.lock();
        let tick = inner.next_tick();
        if let Some(slot) = inner.slot(pattern, size_limit) {
            slot.last_used = tick;
            return slot.compiled.clone();
        }
        while inner.len >= self.options.capacity {
            evicted.push(inner.evict_least_recently_used());
        }
        let slot = Slot {
            size_limit,
            compiled: compiled.clone(),
            last_used: tick,
        };
        match inner.map.get_mut(pattern) {
            Some(slots) => slots.push(slot),
            None => {
                inner.map.insert(pattern.into(), vec![slot]);
            }
        }
        inner.len += 1;
        // the evicted regexes are freed after the lock is released
        drop(inner);
        drop(evicted);
        compiled
    }

    /// The compiled `pattern` under `size_limit`, compiling it, outside the
    /// lock, when the cache does not hold it.
    pub(crate) fn get(&self, pattern: &str, size_limit: usize) -> Compiled {
        match self.lookup(pattern, size_limit) {
            Some(compiled) => compiled,
            None => self.insert(pattern, size_limit, compile(pattern, size_limit)),
        }
    }
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
        let inner = cache.lock();
        let mut keys: Vec<String> = inner
            .map
            .iter()
            .flat_map(|(pattern, slots)| {
                slots
                    .iter()
                    .map(move |slot| format!("{pattern}@{}", slot.size_limit))
            })
            .collect();
        keys.sort();
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
        let guard = cache.inner.try_lock();
        assert!(guard.is_ok());
        drop(guard);
        assert_eq!(compiled.is_match(&"a".repeat(1 << 16)), Ok(true));
        // and another thread can use the cache meanwhile
        std::thread::scope(|scope| {
            scope.spawn(|| assert_eq!(cache.get("b", DEFAULT).is_match("b"), Ok(true)));
        });
        assert_eq!(compiled.is_match("b"), Ok(false));
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
        let inner = cache.lock();
        assert_eq!(inner.len, inner.map.values().map(Vec::len).sum::<usize>());
    }

    #[test]
    fn clear_drops_every_entry() {
        let cache = RegexCache::default();
        cache.get("a", DEFAULT);
        cache.clear();
        assert!(cache.is_empty());
    }
}
