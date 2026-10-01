//! The steps translating a regex pattern costs, priced before compiling it.
//!
//! Compiling a pattern first translates it: every Unicode and Perl class is
//! looked up and unioned into its bracketed class, and under `(?i)` every
//! Unicode class and range is case folded. That work is not reflected in the
//! size of the compiled automaton (`(?i)[\p{Greek}...]` repeated thousands
//! of times compiles small), so it is priced from the parsed pattern, before
//! the compile runs.
//!
//! Calibrated with regex 1.13 / regex-syntax 0.8 in release, at the
//! budget's reference of about 80 ns per step:
//!
//! - parsing costs up to ~470 ns per byte of pattern (a byte inside a
//!   bracketed class, or an open group), and a pattern is parsed twice: once
//!   to price it, once to compile it. A parse that fails costs as much: too
//!   deep a nesting fails only once the whole pattern is parsed;
//! - every Unicode or Perl class costs up to ~12 µs to look up and union;
//! - case folding a Unicode class or range iterates every codepoint of each
//!   of its ranges that contains a case-mapped codepoint (~10 ns each), does
//!   a lookup per case-mapped codepoint (up to ~150 ns each), and sorts the
//!   ranges it produces: `(?i)\p{Lu}` folds in ~200 µs, `(?i)\p{Any}` in
//!   ~12 ms.
use regex_syntax::ast::{self, Ast, ClassSetItem, Flag};
use regex_syntax::hir::{Class, ClassUnicode, HirKind};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Steps for looking a Unicode or Perl class up and unioning it in.
const CLASS_ITEM: u64 = 150;
/// Steps every compile costs, whatever the pattern: building the meta
/// regex's engines takes ~25 µs even for a short pattern.
const COMPILE: u64 = 300;
/// Steps per byte of pattern for its two parses, the pricing one and the
/// compile's, at up to ~470 ns a byte each.
const PARSE_BYTE: u64 = 10;
/// Steps per literal character when case insensitive: each becomes a
/// small class and the literal prefilters are built over the variants.
const FOLDED_LITERAL: u64 = 50;

/// The codepoints that have a simple case mapping, as sorted ranges: those a
/// case fold looks up. Approximated by `Changes_When_Casemapped`; if that
/// property is unavailable, every codepoint, which only overcharges.
static CASED: LazyLock<Vec<(u32, u32)>> =
    LazyLock::new(|| unicode_ranges(r"\p{CWCM}").unwrap_or_else(|| vec![(0, char::MAX as u32)]));

/// The ranges of the Unicode class `item` (e.g. `\p{Greek}`), translated on
/// its own, without case folding.
fn unicode_ranges(item: &str) -> Option<Vec<(u32, u32)>> {
    let hir = regex_syntax::ParserBuilder::new()
        .build()
        .parse(item)
        .ok()?;
    match hir.kind() {
        HirKind::Class(Class::Unicode(class)) => Some(ranges(class)),
        _ => None,
    }
}

fn ranges(class: &ClassUnicode) -> Vec<(u32, u32)> {
    class
        .ranges()
        .iter()
        .map(|r| (r.start() as u32, r.end() as u32))
        .collect()
}

/// The steps of case folding the codepoints `lo..=hi`.
fn fold_range(lo: u32, hi: u32) -> u64 {
    let cased = &*CASED;
    let first = cased.partition_point(|&(_, end)| end < lo);
    let mut mapped = 0u64;
    for &(start, end) in cased[first..].iter().take_while(|&&(start, _)| start <= hi) {
        mapped += u64::from(end.min(hi) - start.max(lo)) + 1;
    }
    if mapped == 0 {
        // no case-mapped codepoint: the range is skipped after one lookup
        return 3;
    }
    (u64::from(hi - lo) + 1) / 7 + 2 * mapped + 3
}

/// The steps of case folding a class made of `ranges`.
fn fold_class(ranges: &[(u32, u32)]) -> u64 {
    ranges.iter().map(|&(lo, hi)| fold_range(lo, hi)).sum()
}

/// The complement of the sorted, disjoint `ranges`.
fn complement(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(ranges.len() + 1);
    let mut next = 0u32;
    for &(lo, hi) in ranges {
        if lo > next {
            out.push((next, lo - 1));
        }
        next = hi.saturating_add(1);
    }
    if next <= char::MAX as u32 {
        out.push((next, char::MAX as u32));
    }
    out
}

/// What the translation of a pattern does, as collected from its syntax.
#[derive(Default)]
struct Items<'p> {
    /// Whether case insensitivity is turned on anywhere. Conservatively, it
    /// is then assumed to apply to the whole pattern.
    case_insensitive: bool,
    /// The Unicode classes, by their text in the pattern, and whether each
    /// is negated (`\P{..}`, `\p{^..}`): a negated class is folded before
    /// its negation, so its fold is that of the class it negates.
    unicode: Vec<(&'p str, bool)>,
    /// How many Perl classes.
    perl: u64,
    /// How many literal characters.
    literals: u64,
    /// The bracketed ranges.
    ranges: Vec<(u32, u32)>,
}

struct Collect<'p> {
    pattern: &'p str,
    items: Items<'p>,
}

impl<'p> Collect<'p> {
    fn text(&self, span: &ast::Span) -> &'p str {
        &self.pattern[span.start.offset..span.end.offset]
    }

    fn flags(&mut self, flags: &ast::Flags) {
        if flags.flag_state(Flag::CaseInsensitive) == Some(true) {
            self.items.case_insensitive = true;
        }
    }
}

impl<'p> ast::Visitor for Collect<'p> {
    type Output = Items<'p>;
    type Err = ();

    fn finish(self) -> Result<Items<'p>, ()> {
        Ok(self.items)
    }

    fn visit_pre(&mut self, ast: &Ast) -> Result<(), ()> {
        match ast {
            Ast::Flags(set) => self.flags(&set.flags),
            Ast::Group(group) => {
                if let ast::GroupKind::NonCapturing(flags) = &group.kind {
                    self.flags(flags);
                }
            }
            Ast::ClassUnicode(class) => {
                let text = self.text(&class.span);
                self.items.unicode.push((text, class.is_negated()));
            }
            Ast::ClassPerl(_) => self.items.perl += 1,
            Ast::Literal(_) => self.items.literals += 1,
            _ => {}
        }
        Ok(())
    }

    fn visit_class_set_item_pre(&mut self, item: &ClassSetItem) -> Result<(), ()> {
        match item {
            ClassSetItem::Unicode(class) => {
                let text = self.text(&class.span);
                self.items.unicode.push((text, class.is_negated()));
            }
            ClassSetItem::Perl(_) => self.items.perl += 1,
            ClassSetItem::Literal(_) => self.items.literals += 1,
            ClassSetItem::Range(range) => {
                let (lo, hi) = (range.start.c as u32, range.end.c as u32);
                self.items.ranges.push((lo.min(hi), lo.max(hi)));
            }
            _ => {}
        }
        Ok(())
    }
}

/// The steps compiling `pattern` costs before its translation: [`COMPILE`]
/// steps and [`PARSE_BYTE`] steps per byte, for the parse that prices it and
/// the compile's own, whether or not they succeed. Charged before either
/// parse runs.
pub(crate) fn parse_steps(pattern: &str) -> u64 {
    (pattern.len() as u64)
        .saturating_mul(PARSE_BYTE)
        .saturating_add(COMPILE)
}

/// The steps translating `pattern` costs, counted until they pass `cap`.
///
/// [`CLASS_ITEM`] steps per Unicode or Perl class and, when the pattern is
/// case insensitive, [`FOLDED_LITERAL`] steps per literal character and the
/// steps of folding every Unicode class and range. A pattern that does not
/// parse costs nothing more than [`parse_steps`]: its compile stops at the
/// parse error.
pub(crate) fn translation_steps(pattern: &str, cap: u64) -> u64 {
    let mut total = 0u64;
    let Ok(ast) = ast::parse::Parser::new().parse(pattern) else {
        return total;
    };
    let Ok(items) = ast::visit(
        &ast,
        Collect {
            pattern,
            items: Items::default(),
        },
    ) else {
        return total;
    };
    let classes = items.unicode.len() as u64 + items.perl;
    total = total.saturating_add(classes.saturating_mul(CLASS_ITEM));
    if !items.case_insensitive || total > cap {
        return total;
    }
    total = total.saturating_add(items.literals.saturating_mul(FOLDED_LITERAL));
    // Each distinct class is translated on its own once, to price its fold:
    // that costs less than the CLASS_ITEM steps already charged for it.
    let mut folds: HashMap<&str, u64> = HashMap::new();
    for (item, negated) in items.unicode {
        let fold = *folds.entry(item).or_insert_with(|| {
            unicode_ranges(item).map_or(0, |ranges| match negated {
                true => fold_class(&complement(&ranges)),
                false => fold_class(&ranges),
            })
        });
        total = total.saturating_add(fold);
        if total > cap {
            return total;
        }
    }
    for (lo, hi) in items.ranges {
        total = total.saturating_add(fold_range(lo, hi));
        if total > cap {
            return total;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::{parse_steps, translation_steps};

    #[test]
    fn parsing_is_priced_per_byte() {
        // a parse that fails at the nesting limit still parsed every byte
        let nested = format!("{}a{}", "(".repeat(4_000), ")".repeat(4_000));
        assert!(parse_steps(&nested) >= 10 * nested.len() as u64);
        assert_eq!(translation_steps(&nested, u64::MAX), 0);
        assert!(parse_steps("^/api/v[0-9]+/users/[0-9]+$") < 1_000);
    }

    #[test]
    fn plain_patterns_are_cheap() {
        assert!(translation_steps("^/api/v[0-9]+/users/[0-9]+$", u64::MAX) < 1_000);
        assert!(translation_steps(r"\w+@\w+", u64::MAX) < 1_000);
    }

    #[test]
    fn classes_are_priced_per_item() {
        let one = translation_steps(r"[\p{L}]", u64::MAX);
        let many = translation_steps(&format!("[{}]", r"\p{L}".repeat(100)), u64::MAX);
        assert!(many - one >= 99 * super::CLASS_ITEM, "{one} {many}");
    }

    #[test]
    fn case_folding_is_priced_by_the_class() {
        // folding every codepoint costs ~12 ms: about 150,000 steps
        assert!(translation_steps(r"(?i)[\p{Any}]", u64::MAX) > 100_000);
        assert!(translation_steps(r"[\p{Any}]", u64::MAX) < 1_000);
        // a script without case is cheap to fold, a cased one is not
        assert!(translation_steps(r"(?i)\p{Han}", u64::MAX) < 1_000);
        assert!(translation_steps(r"(?i)\p{Lu}", u64::MAX) > 2_000);
        // a negated class folds the class it negates
        let lu = translation_steps(r"(?i)\p{Lu}", u64::MAX);
        let not_lu = translation_steps(r"(?i)\P{Lu}", u64::MAX);
        assert!(not_lu < 2 * lu, "{lu} {not_lu}");
    }

    #[test]
    fn the_count_stops_past_the_cap() {
        let pattern = format!("(?i)[{}{}]", r"\p{Any}", r"\p{Lu}".repeat(1_000));
        let capped = translation_steps(&pattern, 1_000);
        assert!(capped > 1_000);
        assert!(capped < translation_steps(&pattern, u64::MAX));
    }
}
