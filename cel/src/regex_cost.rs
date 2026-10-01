//! The steps translating a regex pattern costs, priced before compiling it.
//!
//! Compiling a pattern first translates it: every Unicode and Perl class is
//! looked up and unioned into its bracketed class, and under `(?i)` every
//! Unicode class is case folded, and so is every bracket and every operand
//! of a set operation (`&&`, `--`, `~~`) not known to be folded already:
//! one that holds a range, a literal or a Perl class. Folding a bracket
//! folds all it holds, nested brackets included, so `(?i)[a[a[\w]]]` folds
//! `\w` three times.
//! That work is not reflected in the size of the compiled automaton
//! (`(?i)[\p{Greek}...]` repeated thousands of times compiles small), so it
//! is priced from the parsed pattern, before the compile runs.
//!
//! Calibrated with regex 1.13 / regex-syntax 0.8 in release, at the
//! budget's reference of about 80 ns per step:
//!
//! - parsing costs up to ~470 ns per byte of pattern (a byte inside a
//!   bracketed class, or an open group), and a pattern is parsed twice: once
//!   to price it, once to compile it. A parse that fails costs as much: too
//!   deep a nesting fails only once the whole pattern is parsed;
//! - every Unicode or Perl class costs up to ~12 µs to look up and union,
//!   except an `Age` class, the union of the tables of every Unicode version
//!   up to its own: up to ~300 µs;
//! - case folding a Unicode class or range iterates every codepoint of each
//!   of its ranges that contains a case-mapped codepoint (~10 ns each), does
//!   a lookup per case-mapped codepoint (up to ~150 ns each), and sorts the
//!   ranges it produces: `(?i)\p{Lu}` folds in ~200 µs, `(?i)\p{Any}` in
//!   ~12 ms.
use regex_syntax::ast::{self, Ast, ClassSetBinaryOp, ClassSetBinaryOpKind, ClassSetItem, Flag};
use regex_syntax::hir::{Class, ClassUnicode, HirKind};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Steps for looking a Unicode or Perl class up and unioning it in.
const CLASS_ITEM: u64 = 150;
/// Steps for looking an `Age` class up: it is the union of the tables of
/// every Unicode version up to its own, ~300 µs for a recent one.
const AGE_ITEM: u64 = 5_000;
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

/// One step of building the classes of a pattern, as its translation takes
/// them, in order: enough to price the case folding it does.
enum Event<'p> {
    /// An empty class opens: a bracket, or an operand of a set operation.
    Open,
    /// An item is unioned into the open class.
    Item(Item<'p>),
    /// A bracket nested in another closes: it is folded (unless known to be
    /// folded already) and unioned into its parent.
    CloseNested,
    /// An outermost bracket closes: it is folded, unless known to be.
    Close,
    /// A set operation: its two operands, the last two classes opened, are
    /// each folded (unless known to be) and combined into the class below.
    SetOp(ClassSetBinaryOpKind),
    /// A Unicode class outside brackets, folded as it is translated.
    Unicode(&'p str, bool),
}

/// An item of a bracketed class.
enum Item<'p> {
    /// A Unicode class, by its text, and whether it is negated. It is folded
    /// as it is translated, and its translation is then known to be folded.
    Unicode(&'p str, bool),
    /// A Perl class (`\w`, `\D`, ...), by its text. It is not folded as it
    /// is translated, so the class it is unioned into is folded again.
    Perl(&'p str),
    /// A range of codepoints, or a single one.
    Range(u32, u32),
}

/// What the translation of a pattern does, as collected from its syntax.
#[derive(Default)]
struct Items<'p> {
    /// Whether case insensitivity is turned on anywhere. Conservatively, it
    /// is then assumed to apply to the whole pattern.
    case_insensitive: bool,
    /// The steps of looking up every Unicode and Perl class.
    lookups: u64,
    /// How many literal characters.
    literals: u64,
    /// How the classes are built.
    events: Vec<Event<'p>>,
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

    /// Counts the lookup of the Unicode class `class`, returning its text.
    fn unicode(&mut self, class: &ast::ClassUnicode) -> &'p str {
        self.items.lookups = self.items.lookups.saturating_add(lookup(class));
        self.text(&class.span)
    }

    /// Counts the lookup of the Perl class `class`, returning its text.
    fn perl(&mut self, class: &ast::ClassPerl) -> &'p str {
        self.items.lookups = self.items.lookups.saturating_add(CLASS_ITEM);
        self.text(&class.span)
    }
}

/// The steps of looking up the Unicode class `class`.
fn lookup(class: &ast::ClassUnicode) -> u64 {
    let ast::ClassUnicodeKind::NamedValue { name, .. } = &class.kind else {
        return CLASS_ITEM;
    };
    // the property name as regex-syntax matches it: ASCII only, without
    // case, spaces, underscores, hyphens or an `is` prefix
    let name: String = name
        .chars()
        .filter(|c| c.is_ascii() && !matches!(c, ' ' | '_' | '-'))
        .map(|c| c.to_ascii_lowercase())
        .collect();
    match name.as_str() {
        "age" | "isage" => AGE_ITEM,
        _ => CLASS_ITEM,
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
                let text = self.unicode(class);
                self.items
                    .events
                    .push(Event::Unicode(text, class.is_negated()));
            }
            Ast::ClassPerl(class) => {
                // not folded outside brackets
                self.perl(class);
            }
            Ast::ClassBracketed(_) => self.items.events.push(Event::Open),
            Ast::Literal(_) => self.items.literals += 1,
            _ => {}
        }
        Ok(())
    }

    fn visit_post(&mut self, ast: &Ast) -> Result<(), ()> {
        if let Ast::ClassBracketed(_) = ast {
            self.items.events.push(Event::Close);
        }
        Ok(())
    }

    fn visit_class_set_item_pre(&mut self, item: &ClassSetItem) -> Result<(), ()> {
        let item = match item {
            ClassSetItem::Bracketed(_) => {
                self.items.events.push(Event::Open);
                return Ok(());
            }
            ClassSetItem::Unicode(class) => Item::Unicode(self.unicode(class), class.is_negated()),
            ClassSetItem::Perl(class) => Item::Perl(self.perl(class)),
            ClassSetItem::Literal(literal) => {
                self.items.literals += 1;
                Item::Range(literal.c as u32, literal.c as u32)
            }
            ClassSetItem::Range(range) => {
                let (lo, hi) = (range.start.c as u32, range.end.c as u32);
                Item::Range(lo.min(hi), lo.max(hi))
            }
            // `[:alpha:]` and the like: ASCII at most
            ClassSetItem::Ascii(_) => Item::Range(0, 0x7F),
            ClassSetItem::Empty(_) | ClassSetItem::Union(_) => return Ok(()),
        };
        self.items.events.push(Event::Item(item));
        Ok(())
    }

    fn visit_class_set_item_post(&mut self, item: &ClassSetItem) -> Result<(), ()> {
        if let ClassSetItem::Bracketed(_) = item {
            self.items.events.push(Event::CloseNested);
        }
        Ok(())
    }

    fn visit_class_set_binary_op_pre(&mut self, _: &ClassSetBinaryOp) -> Result<(), ()> {
        self.items.events.push(Event::Open);
        Ok(())
    }

    fn visit_class_set_binary_op_in(&mut self, _: &ClassSetBinaryOp) -> Result<(), ()> {
        self.items.events.push(Event::Open);
        Ok(())
    }

    fn visit_class_set_binary_op_post(&mut self, op: &ClassSetBinaryOp) -> Result<(), ()> {
        self.items.events.push(Event::SetOp(op.kind));
        Ok(())
    }
}

/// A class being built, as the steps of case folding it.
#[derive(Clone, Copy)]
struct Building {
    /// The steps of folding its ranges.
    fold: u64,
    /// Whether it is known to be folded already: regex-syntax then skips
    /// folding it. A union is known folded when both sides are.
    folded: bool,
}

impl Building {
    /// An empty class, which is folded.
    const EMPTY: Building = Building {
        fold: 0,
        folded: true,
    };

    fn union(&mut self, other: Building) {
        self.fold = self.fold.saturating_add(other.fold);
        self.folded &= other.folded;
    }

    /// Folds the class, returning the steps it costs.
    fn fold(&mut self) -> u64 {
        match std::mem::replace(&mut self.folded, true) {
            true => 0,
            false => self.fold,
        }
    }
}

/// The steps of case folding a Unicode class as it is translated, and of
/// folding its translation again: those of the class it negates, as it is
/// folded before its negation, then its own.
fn unicode_folds(item: &str, negated: bool) -> (u64, u64) {
    unicode_ranges(item).map_or((0, 0), |ranges| {
        let own = fold_class(&ranges);
        match negated {
            true => (fold_class(&complement(&ranges)), own),
            false => (own, own),
        }
    })
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
/// [`CLASS_ITEM`] steps per Unicode or Perl class ([`AGE_ITEM`] per `Age`
/// class) and, when the pattern is case insensitive, [`FOLDED_LITERAL`]
/// steps per literal character and the steps of every case fold the
/// translation does: of each Unicode class, and of each bracket and set
/// operand not known to be folded already, which folds again all it
/// contains. A pattern that does not parse costs nothing more than
/// [`parse_steps`]: its compile stops at the parse error.
pub(crate) fn translation_steps(pattern: &str, cap: u64) -> u64 {
    let Ok(ast) = ast::parse::Parser::new().parse(pattern) else {
        return 0;
    };
    let Ok(items) = ast::visit(
        &ast,
        Collect {
            pattern,
            items: Items::default(),
        },
    ) else {
        return 0;
    };
    let mut total = items.lookups;
    if !items.case_insensitive || total > cap {
        return total;
    }
    total = total.saturating_add(items.literals.saturating_mul(FOLDED_LITERAL));
    // Each distinct class is translated on its own once, to price its folds:
    // that costs less than the lookup steps already charged for it.
    let mut folds: HashMap<&str, (u64, u64)> = HashMap::new();
    let mut fold_of = |item, negated| -> (u64, u64) {
        *folds
            .entry(item)
            .or_insert_with(|| unicode_folds(item, negated))
    };
    let mut open: Vec<Building> = Vec::new();
    for event in items.events {
        let steps = match event {
            Event::Open => {
                open.push(Building::EMPTY);
                0
            }
            Event::Item(item) => {
                let (steps, built) = match item {
                    Item::Unicode(text, negated) => {
                        let (now, again) = fold_of(text, negated);
                        let built = Building {
                            fold: again,
                            folded: true,
                        };
                        (now, built)
                    }
                    Item::Perl(text) => {
                        let built = Building {
                            fold: fold_of(text, false).1,
                            folded: false,
                        };
                        (0, built)
                    }
                    Item::Range(lo, hi) => {
                        let built = Building {
                            fold: fold_range(lo, hi),
                            folded: false,
                        };
                        (0, built)
                    }
                };
                if let Some(class) = open.last_mut() {
                    class.union(built);
                }
                steps
            }
            Event::CloseNested => {
                let mut nested = open.pop().unwrap_or(Building::EMPTY);
                let steps = nested.fold();
                if let Some(class) = open.last_mut() {
                    class.union(nested);
                }
                steps
            }
            Event::Close => open.pop().unwrap_or(Building::EMPTY).fold(),
            Event::SetOp(kind) => {
                let mut rhs = open.pop().unwrap_or(Building::EMPTY);
                let mut lhs = open.pop().unwrap_or(Building::EMPTY);
                let steps = rhs.fold().saturating_add(lhs.fold());
                // the result is folded, and no larger than its operands
                let fold = match kind {
                    ClassSetBinaryOpKind::Intersection => lhs.fold.min(rhs.fold),
                    ClassSetBinaryOpKind::Difference => lhs.fold,
                    ClassSetBinaryOpKind::SymmetricDifference => lhs.fold.saturating_add(rhs.fold),
                };
                if let Some(class) = open.last_mut() {
                    class.union(Building { fold, folded: true });
                }
                steps
            }
            Event::Unicode(text, negated) => fold_of(text, negated).0,
        };
        total = total.saturating_add(steps);
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
    fn perl_classes_are_folded_with_their_bracket() {
        // `\w` is not folded as it is translated, so its bracket folds it:
        // ~1 ms under (?i)
        let plain = translation_steps(r"[\w]", u64::MAX);
        let folded = translation_steps(r"(?i)[\w]", u64::MAX);
        assert!(folded - plain > 8_000, "{plain} {folded}");
        // outside brackets it is not folded at all
        assert_eq!(translation_steps(r"(?i)\w", u64::MAX), plain);
    }

    #[test]
    fn set_operands_and_nested_brackets_are_folded_again() {
        let one = translation_steps(r"(?i)[\w]", u64::MAX);
        // every operand of a set operation is folded
        let ops = translation_steps(&format!("(?i)[{}\\w]", r"\w&&".repeat(100)), u64::MAX);
        assert!(ops > 100 * one, "{one} {ops}");
        // every nested bracket folds again all it contains: here `\w` again
        let nest = |depth: usize| {
            let pattern = format!("(?i)[{}\\w{}]", "a[".repeat(depth), "]".repeat(depth));
            translation_steps(&pattern, u64::MAX)
        };
        assert!(nest(120) > 100 * one, "{one} {}", nest(120));
        assert!(nest(120) > nest(60) * 19 / 10, "{} {}", nest(60), nest(120));
        // a Unicode class is folded once, as it is translated
        let lu = translation_steps(r"(?i)\p{Lu}", u64::MAX);
        let lu2 = translation_steps(r"(?i)[\p{Lu}\p{Lu}]", u64::MAX);
        assert!(lu2 < 2 * lu + 500, "{lu} {lu2}");
    }

    #[test]
    fn age_classes_are_priced_by_their_union() {
        // ~300 µs each: the union of every earlier version's table
        for class in [r"\p{Age=15.0}", r"\p{age:V15_0}", r"\P{ is_A-g e = 15.0}"] {
            assert!(translation_steps(&format!("[{class}]"), u64::MAX) >= super::AGE_ITEM);
        }
        assert!(translation_steps(r"[\p{scx=Greek}]", u64::MAX) < 1_000);
    }

    #[test]
    fn the_count_stops_past_the_cap() {
        let pattern = format!("(?i)[{}{}]", r"\p{Any}", r"\p{Lu}".repeat(1_000));
        let capped = translation_steps(&pattern, 1_000);
        assert!(capped > 1_000);
        assert!(capped < translation_steps(&pattern, u64::MAX));
    }
}
