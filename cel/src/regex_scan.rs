//! Matching a compiled pattern under a budget, charging what it costs as it
//! goes.
//!
//! The `meta` regex `matches` uses without a budget runs a lazy DFA first,
//! and falls back to a slower engine when the DFA gives up. Neither is
//! priced by the size of the automaton:
//!
//! - the lazy DFA builds its states as the subject needs them, and a state
//!   costs up to ~30 ns per byte of its NFA state set, plus a lookup per
//!   NFA state: a pattern that needs a new state at every byte, and so
//!   keeps clearing its cache, costs ~15 µs a byte with a small automaton;
//! - the slower engine, which runs when the lazy DFA quits (a Unicode word
//!   boundary on a non-ASCII byte), steps every live position of the
//!   pattern at every byte: up to ~40 ns per position and per Unicode
//!   word-boundary assertion.
//!
//! So under a budget, a [`Scanner`] runs a lazy DFA of its own, with no
//! fallback, a byte at a time, charging each state it builds as it builds
//! it. Only if the DFA quits, or finds an empty match that splits a
//! character (which the `meta` regex would skip), does it charge the slower
//! engine's bound, before running a PikeVM. The answer is the `meta`
//! regex's: both engines search the same pattern for any match.
//!
//! Calibrated with regex-automata 0.4 in release, at the budget's reference
//! of about 80 ns per step.
use regex_automata::{
    hybrid,
    nfa::thompson::{self, pikevm},
    util::pool::Pool,
    Input,
};
use regex_syntax::hir::{Hir, HirKind, LookSet};
use std::panic::{RefUnwindSafe, UnwindSafe};

/// Nanoseconds every lazy DFA state costs to build, whatever its size.
const STATE_NS: u64 = 600;
/// Nanoseconds a state costs per byte of its NFA state set.
const STATE_SET_BYTE_NS: u64 = 30;
/// Nanoseconds a state costs per entry of its transition table.
const STATE_TRANSITION_NS: u64 = 12;
/// Nanoseconds a state costs per five NFA states of the automaton (each
/// looked up when computing the set).
const STATE_NFA_STATES_PER_NS: u64 = 5;
/// The nanoseconds per step the charges are calibrated at.
const STEP_NS: u64 = 80;
/// Nanoseconds a byte costs the slower engine, and per live position and
/// per Unicode word-boundary assertion.
const SLOW_BYTE_NS: u64 = 100;
const SLOW_POSITION_NS: u64 = 40;

/// A pattern's automata for matching under a budget.
pub(crate) struct Scanner {
    /// The lazy DFA, unless the cache capacity cannot hold the states of
    /// this automaton: then every match runs the slower engine.
    dfa: Option<hybrid::dfa::DFA>,
    pikevm: pikevm::PikeVM,
    caches: Pool<Caches, CreateCaches>,
    /// The bytes of the NFA both engines run.
    nfa_bytes: u64,
    /// How many NFA states, see [`STATE_NFA_STATES_PER_NS`].
    nfa_states: u64,
    /// The entries of a state's transition table.
    stride: u64,
    /// The literal characters and classes of the pattern, a repetition
    /// counting its copies, plus its Unicode word-boundary assertions: at
    /// most how many NFA threads the slower engine steps at each byte,
    /// within a few states each.
    positions: u64,
}

type CreateCaches = Box<dyn Fn() -> Caches + Send + Sync + UnwindSafe + RefUnwindSafe>;

struct Caches {
    dfa: Option<hybrid::dfa::Cache>,
    pikevm: pikevm::Cache,
}

/// A charge was refused: the evaluation must stop.
pub(crate) struct Stopped;

/// How a lazy DFA walk of a subject ended.
enum Walk {
    /// A match ending at this offset.
    Match(usize),
    NoMatch,
    /// The lazy DFA cannot decide: it quit, or could not build a state.
    Quit,
}

impl Scanner {
    /// The automata of the pattern `hir` under `size_limit`, the lazy DFA's
    /// cache capped at `cache_capacity`, as the `meta` regex's. `None` if
    /// the NFA exceeds the limit.
    pub(crate) fn new(hir: &Hir, size_limit: usize, cache_capacity: usize) -> Option<Scanner> {
        let nfa = thompson::Compiler::new()
            .configure(
                thompson::Config::new()
                    .nfa_size_limit(Some(size_limit))
                    .which_captures(thompson::WhichCaptures::None),
            )
            .build_from_hir(hir)
            .ok()?;
        let pikevm = pikevm::PikeVM::new_from_nfa(nfa.clone()).ok()?;
        // never gives up: every state it builds is charged, thrashing
        // included
        let dfa = hybrid::dfa::Builder::new()
            .configure(
                hybrid::dfa::Config::new()
                    .cache_capacity(cache_capacity)
                    .unicode_word_boundary(true)
                    .minimum_cache_clear_count(None),
            )
            .build_from_nfa(nfa.clone())
            .ok();
        let stride = dfa
            .as_ref()
            .map_or(0, |dfa| 1u64 << dfa.byte_classes().stride2());
        let create: CreateCaches = {
            let (dfa, pikevm) = (dfa.clone(), pikevm.clone());
            Box::new(move || Caches {
                dfa: dfa.as_ref().map(hybrid::dfa::DFA::create_cache),
                pikevm: pikevm.create_cache(),
            })
        };
        Some(Scanner {
            dfa,
            pikevm,
            caches: Pool::new(create),
            nfa_bytes: nfa.memory_usage() as u64,
            nfa_states: nfa.states().len() as u64,
            stride,
            positions: positions(hir),
        })
    }

    /// The bytes of the automata, not counting the caches of the threads
    /// matching them.
    pub(crate) fn bytes(&self) -> u64 {
        self.nfa_bytes
    }

    /// Whether the pattern matches `subject`, as the `meta` regex answers,
    /// charging with `charge` each lazy DFA state built, as it is built,
    /// and the slower engine's bound before running it. Stops as soon as a
    /// charge is refused.
    pub(crate) fn is_match(
        &self,
        subject: &str,
        charge: &mut dyn FnMut(u64) -> bool,
    ) -> Result<bool, Stopped> {
        let mut caches = self.caches.get();
        let caches = &mut *caches;
        if let (Some(dfa), Some(cache)) = (&self.dfa, caches.dfa.as_mut()) {
            match self.walk(dfa, cache, subject.as_bytes(), charge)? {
                // a non-empty match ends on a character boundary
                Walk::Match(at) if subject.is_char_boundary(at) => return Ok(true),
                Walk::NoMatch => return Ok(false),
                // an empty match splitting a character, which `meta`
                // skips, or a quit
                Walk::Match(_) | Walk::Quit => {}
            }
        }
        if !charge(self.slow_steps(subject.len())) {
            return Err(Stopped);
        }
        Ok(self
            .pikevm
            .is_match(&mut caches.pikevm, Input::new(subject)))
    }

    /// The steps the slower engine costs to scan `len` bytes.
    fn slow_steps(&self, len: usize) -> u64 {
        let per_byte = self
            .positions
            .saturating_mul(SLOW_POSITION_NS)
            .saturating_add(SLOW_BYTE_NS);
        (len as u64).saturating_mul(per_byte).div_ceil(STEP_NS)
    }

    /// The steps of building a lazy DFA state that grew the cache by
    /// `grown` bytes: its transition table, its bookkeeping, and its NFA
    /// state set.
    fn state_steps(&self, grown: u64) -> u64 {
        let set = grown.saturating_sub(self.stride * 4 + 36);
        let ns = STATE_NS
            + self.stride * STATE_TRANSITION_NS
            + self.nfa_states / STATE_NFA_STATES_PER_NS
            + set.saturating_mul(STATE_SET_BYTE_NS);
        ns.div_ceil(STEP_NS)
    }

    /// Walks `haystack` with the lazy DFA for any match, a byte at a time,
    /// as `hybrid::dfa::DFA::try_search_fwd` does for an unanchored,
    /// earliest search, charging each state as the cache grows by it.
    fn walk(
        &self,
        dfa: &hybrid::dfa::DFA,
        cache: &mut hybrid::dfa::Cache,
        haystack: &[u8],
        charge: &mut dyn FnMut(u64) -> bool,
    ) -> Result<Walk, Stopped> {
        let mut used = cache.memory_usage() as u64;
        let mut clears = cache.clear_count();
        // charges the states built since the last call
        let mut account = |cache: &hybrid::dfa::Cache| -> Result<(), Stopped> {
            let now = cache.memory_usage() as u64;
            let grown = if cache.clear_count() != clears {
                // the cache was full and cleared: a state was built, its
                // set's size unknown, so priced at the largest
                clears = cache.clear_count();
                self.stride * 4 + 36 + self.nfa_states * 5
            } else {
                now.saturating_sub(used)
            };
            used = now;
            match grown {
                0 => Ok(()),
                grown => match charge(self.state_steps(grown)) {
                    true => Ok(()),
                    false => Err(Stopped),
                },
            }
        };
        let input = Input::new(haystack);
        let Ok(mut sid) = dfa.start_state_forward(cache, &input) else {
            return Ok(Walk::Quit);
        };
        account(cache)?;
        for (at, &byte) in haystack.iter().enumerate() {
            let Ok(next) = dfa.next_state(cache, sid, byte) else {
                return Ok(Walk::Quit);
            };
            sid = next;
            account(cache)?;
            if sid.is_tagged() {
                // matches are reported a byte late
                if sid.is_match() {
                    return Ok(Walk::Match(at));
                }
                if sid.is_dead() {
                    return Ok(Walk::NoMatch);
                }
                if sid.is_quit() {
                    return Ok(Walk::Quit);
                }
            }
        }
        let Ok(sid) = dfa.next_eoi_state(cache, sid) else {
            return Ok(Walk::Quit);
        };
        account(cache)?;
        Ok(if sid.is_match() {
            Walk::Match(haystack.len())
        } else if sid.is_quit() {
            Walk::Quit
        } else {
            Walk::NoMatch
        })
    }
}

/// The literal characters and classes of `hir`, plus its Unicode
/// word-boundary assertions, a bounded repetition counting as many copies
/// of its body as its maximum, an unbounded one one more than its minimum,
/// as the NFA compiled from it holds them.
fn positions(hir: &Hir) -> u64 {
    let mut positions = 0u64;
    let mut walk = vec![(hir, 1u64)];
    while let Some((hir, copies)) = walk.pop() {
        let own = match hir.kind() {
            HirKind::Literal(literal) => match std::str::from_utf8(&literal.0) {
                Ok(text) => text.chars().count() as u64,
                Err(_) => literal.0.len() as u64,
            },
            HirKind::Class(_) => 1,
            HirKind::Look(look) => u64::from(LookSet::singleton(*look).contains_word_unicode()),
            HirKind::Repetition(repetition) => {
                let body = repetition
                    .max
                    .map_or(u64::from(repetition.min).saturating_add(1), u64::from);
                walk.push((&repetition.sub, copies.saturating_mul(body)));
                0
            }
            HirKind::Capture(capture) => {
                walk.push((&capture.sub, copies));
                0
            }
            HirKind::Concat(hirs) | HirKind::Alternation(hirs) => {
                walk.extend(hirs.iter().map(|hir| (hir, copies)));
                0
            }
            HirKind::Empty => 0,
        };
        positions = positions.saturating_add(own.saturating_mul(copies));
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex_automata::util::syntax;

    fn scanner(pattern: &str) -> Scanner {
        let hir = syntax::parse_with(pattern, &syntax::Config::new().utf8(true)).unwrap();
        Scanner::new(&hir, 1 << 20, 1 << 20).unwrap()
    }

    fn charged(pattern: &str, subject: &str) -> (bool, u64) {
        let mut steps = 0u64;
        let matched = scanner(pattern)
            .is_match(subject, &mut |n| {
                steps += n;
                true
            })
            .ok()
            .unwrap();
        (matched, steps)
    }

    #[test]
    fn positions_count_literals_classes_and_word_boundaries_per_copy() {
        let positions = |pattern: &str| {
            positions(&syntax::parse_with(pattern, &syntax::Config::new().utf8(true)).unwrap())
        };
        assert_eq!(positions("abc"), 3);
        assert_eq!(positions("é[a-z]"), 2);
        assert_eq!(positions(r"\b\w+\b"), 4);
        assert_eq!(positions(r"(?-u:\b)a"), 1);
        assert_eq!(positions("(?:a|bc){3}"), 9);
        assert_eq!(positions("(?:ab){2,}"), 6);
        assert_eq!(
            positions(r"(?:[ab]|a[ab]|b[ab]{2}){0,80}a[ab]{18}[^ab/]"),
            500
        );
    }

    #[test]
    fn answers_as_the_meta_regex() {
        let subjects = [
            "",
            "abc",
            "日本語 abc",
            "aé",
            "é",
            "x\r\ny",
            "ab\n",
            "zzz yyy",
        ];
        let patterns = [
            "a",
            "^abc$",
            r"\b\w+\b",
            r"\Bc",
            r"\b",
            r"(?-u:\B)",
            "",
            "^$",
            "(?m)^y$",
            "(?R)^y$",
            "é",
            "[^a]",
            r"\p{Han}+",
            "(?i)ABC",
            "x*",
            r"\b{start}a",
            "a$",
            "b(?:c|é)?",
        ];
        for pattern in patterns {
            let meta = regex_automata::meta::Regex::new(pattern).unwrap();
            for subject in subjects {
                assert_eq!(
                    charged(pattern, subject).0,
                    meta.is_match(subject),
                    "{pattern:?} on {subject:?}"
                );
            }
        }
    }

    #[test]
    fn states_are_charged_as_they_are_built() {
        // a new state at nearly every byte: ~15 µs each
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let subject: String = (0..4096)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if (state >> 33) % 2 == 0 {
                    'a'
                } else {
                    'b'
                }
            })
            .collect();
        let (_, steps) = charged(r"(?:[ab]|a[ab]|b[ab]{2}){0,80}a[ab]{18}[^ab/]", &subject);
        assert!(steps > 300_000, "{steps}");
        // a few states, then none
        let (matched, steps) = charged("^/api/v[0-9]+/", &"/api/v1/".repeat(512));
        assert!(matched);
        assert!(steps < 1_000, "{steps}");
        // a charge refused stops the walk
        let refused = scanner(r"(?:[ab]|a[ab]|b[ab]{2}){0,80}a[ab]{18}[^ab/]")
            .is_match(&subject, &mut |_| false);
        assert!(refused.is_err());
    }

    #[test]
    fn the_slower_engine_is_charged_by_its_positions() {
        // the lazy DFA quits on a Unicode word boundary at a non-ASCII byte
        let subject = "é".repeat(1024);
        let (_, few) = charged(r"\b\B", &subject);
        let (_, many) = charged(r"[a-zé]{0,200}\b\B", &subject);
        assert!(few >= 2048 * 2, "{few}");
        assert!(many > 2048 * 100, "{many}");
        // an ASCII subject never reaches it
        let (_, ascii) = charged(r"[a-zé]{0,200}\b\B", &"a ".repeat(1024));
        assert!(ascii < 10_000, "{ascii}");
    }
}
