use crate::common::traits::{Sizer, Zeroer};
use crate::common::types::{CelInt, CelString, Type};
use crate::common::value::{Builtin, BuiltinRef, CowVal, Val};
use crate::Value;
use crate::{common::traits, ExecutionError};
use std::borrow::Cow;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use traits::{Adder, Comparer};

/// CEL bytes. Shares the buffer, or borrows it for `'a`.
///
/// An owned buffer is held in an [`Arc`], so cloning is O(1) whether the
/// bytes are owned or borrowed, and converting to and from
/// [`Value::Bytes`](crate::Value::Bytes) shares the buffer.
#[derive(Clone)]
pub struct Bytes<'a>(Repr<'a>);

#[derive(Clone)]
enum Repr<'a> {
    Borrowed(&'a [u8]),
    Shared(Arc<Vec<u8>>),
}

impl<'a> Bytes<'a> {
    /// The bytes, moved out when they are owned and not shared, copied
    /// otherwise.
    pub fn into_inner(self) -> Vec<u8> {
        match self.0 {
            Repr::Borrowed(b) => b.to_vec(),
            Repr::Shared(b) => Arc::unwrap_or_clone(b),
        }
    }

    pub fn inner(&self) -> &[u8] {
        match &self.0 {
            Repr::Borrowed(b) => b,
            Repr::Shared(b) => b.as_slice(),
        }
    }

    /// Copies the bytes out if they were borrowed, so the result owns them.
    pub fn into_static(self) -> Bytes<'static> {
        match self.0 {
            Repr::Borrowed(b) => Bytes::from(b.to_vec()),
            Repr::Shared(b) => Bytes(Repr::Shared(b)),
        }
    }

    /// The bytes, with the lifetime of the borrow itself, if they are borrowed
    /// rather than owned.
    ///
    /// Unlike [`inner`](Bytes::inner), whose result is bounded by the borrow
    /// of `self`, this lets a slice of the bytes outlive the `Bytes` that
    /// handed it out: a value read out of a borrowed container can be
    /// re-borrowed, without a copy, for as long as the container's own data.
    pub fn as_borrowed(&self) -> Option<&'a [u8]> {
        match &self.0 {
            Repr::Borrowed(b) => Some(b),
            Repr::Shared(_) => None,
        }
    }

    /// The shared buffer, if the bytes are owned rather than borrowed.
    ///
    /// Cloning the `Arc` hands the bytes out without copying them.
    pub fn as_arc(&self) -> Option<&Arc<Vec<u8>>> {
        match &self.0 {
            Repr::Borrowed(_) => None,
            Repr::Shared(b) => Some(b),
        }
    }

    /// The bytes as an `Arc`: the shared buffer when owned, a copy when
    /// borrowed.
    pub(crate) fn to_arc(&self) -> Arc<Vec<u8>> {
        match &self.0 {
            Repr::Borrowed(b) => Arc::new(b.to_vec()),
            Repr::Shared(b) => Arc::clone(b),
        }
    }
}

impl Default for Bytes<'_> {
    fn default() -> Self {
        Bytes(Repr::Borrowed(&[]))
    }
}

impl Debug for Bytes<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Bytes").field(&self.inner()).finish()
    }
}

impl PartialEq for Bytes<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Deref for Bytes<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl<'a> Val for Bytes<'a> {
    fn get_type(&self) -> &Type {
        <Self as Val>::cel_type()
    }

    fn cel_type() -> &'static Type {
        &super::BYTES_TYPE
    }

    fn as_adder<'b, 'v>(&'b self) -> Option<&'b (dyn Adder + 'v)>
    where
        Self: 'v,
    {
        Some(self)
    }

    fn as_comparer(&self) -> Option<&dyn Comparer> {
        Some(self)
    }

    fn as_sizer(&self) -> Option<&dyn Sizer> {
        Some(self)
    }

    fn as_zeroer(&self) -> Option<&dyn Zeroer> {
        Some(self)
    }

    fn equals(&self, other: &dyn Val) -> bool {
        other
            .downcast_ref::<Bytes>()
            .is_some_and(|a| self.inner() == a.inner())
    }

    fn clone_as_boxed<'v>(&self) -> Box<dyn Val + 'v>
    where
        Self: 'v,
    {
        Box::new(self.clone())
    }

    fn as_builtin<'b, 'v>(&'b self) -> BuiltinRef<'b, 'v>
    where
        Self: 'v,
    {
        BuiltinRef::Bytes(self)
    }

    fn into_builtin<'v>(self: Box<Self>) -> Option<Builtin<'v>>
    where
        Self: 'v,
    {
        Some(Builtin::Bytes(*self))
    }
}

impl<'a> Adder for Bytes<'a> {
    fn add<'b, 'v>(&'b self, other: &(dyn Val + 'v)) -> Result<CowVal<'b, 'v>, ExecutionError>
    where
        Self: 'v,
    {
        if let Some(bytes) = other.downcast_ref::<Bytes>() {
            let mut v = Vec::with_capacity(self.len() + bytes.len());
            v.extend_from_slice(self);
            v.extend_from_slice(bytes);
            Ok(CowVal::owned(Bytes::from(v)))
        } else {
            Err(ExecutionError::UnsupportedBinaryOperator(
                "add",
                (self as &dyn Val).try_into().unwrap_or(Value::Null),
                other.try_into().unwrap_or(Value::Null),
            ))
        }
    }
}

impl Comparer for Bytes<'_> {
    fn compare(&self, other: &dyn Val) -> Result<std::cmp::Ordering, ExecutionError> {
        if let Some(bytes) = other.downcast_ref::<Bytes>() {
            Ok(self.inner().cmp(bytes.inner()))
        } else {
            Err(crate::ExecutionError::values_not_comparable(self, other))
        }
    }
}

impl Sizer for Bytes<'_> {
    fn size(&self) -> CelInt {
        (self.inner().len() as i64).into()
    }
}

impl Zeroer for Bytes<'_> {
    fn is_zero_value(&self) -> bool {
        self.inner().is_empty()
    }
}

impl From<Vec<u8>> for Bytes<'_> {
    fn from(value: Vec<u8>) -> Self {
        Bytes(Repr::Shared(Arc::new(value)))
    }
}

/// Borrows the slice: no copy is made.
impl<'a> From<&'a [u8]> for Bytes<'a> {
    fn from(value: &'a [u8]) -> Self {
        Bytes(Repr::Borrowed(value))
    }
}

/// Shares the buffer: no copy is made.
impl From<Arc<Vec<u8>>> for Bytes<'_> {
    fn from(v: Arc<Vec<u8>>) -> Self {
        Bytes(Repr::Shared(v))
    }
}

/// Reinterprets the string's bytes, keeping a borrow borrowed.
impl<'a> From<CelString<'a>> for Bytes<'a> {
    fn from(value: CelString<'a>) -> Self {
        match value.into_cow() {
            Cow::Borrowed(s) => Bytes::from(s.as_bytes()),
            Cow::Owned(s) => Bytes::from(s.into_bytes()),
        }
    }
}

impl From<Bytes<'_>> for Vec<u8> {
    fn from(value: Bytes<'_>) -> Self {
        value.into_inner()
    }
}

impl<'v> TryFrom<Box<dyn Val + 'v>> for Vec<u8> {
    type Error = Box<dyn Val + 'v>;

    fn try_from(value: Box<dyn Val + 'v>) -> Result<Self, Self::Error> {
        match super::into_builtin(value) {
            Ok(Builtin::Bytes(b)) => Ok(b.into_inner()),
            Ok(other) => Err(other.into_boxed()),
            Err(value) => Err(value),
        }
    }
}

impl<'a, 'v> TryFrom<&'a (dyn Val + 'v)> for &'a [u8] {
    type Error = &'a (dyn Val + 'v);

    fn try_from(value: &'a (dyn Val + 'v)) -> Result<Self, Self::Error> {
        if let Some(bytes) = value.downcast_ref::<Bytes>() {
            return Ok(bytes.inner());
        }
        Err(value)
    }
}

fn string_to_bytes<'b, 'v>(
    mut args: Vec<CowVal<'b, 'v>>,
) -> Result<CowVal<'b, 'v>, ExecutionError> {
    match super::string::take_string(args.remove(0)) {
        Ok(s) => Ok(CowVal::owned(Bytes::from(s))),
        Err(e) => Err(ExecutionError::UnexpectedType {
            got: e.get_type().name().to_owned(),
            want: "Bytes".to_owned(),
        }),
    }
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_type(crate::common::types::BYTES_TYPE)
        .expect("Must be unique");
    // Hand-written: both keep the caller's bytes in place. See `string_to_bytes`.
    env.add_overload(
        "bytes",
        "string_to_bytes",
        vec![super::STRING_TYPE],
        string_to_bytes,
    )
    .expect("Must be unique id");
    env.add_overload(
        "bytes",
        "bytes_to_bytes",
        vec![super::BYTES_TYPE],
        super::noop,
    )
    .expect("Must be unique id");
    crate::add_overload!(env, fn size: (Bytes) -> CelInt, id = "size_bytes");
    crate::add_member_overload!(env, fn size: (Bytes) -> CelInt,
        id = "bytes_size");
}

fn size(this: &Bytes<'_>) -> CelInt {
    this.size()
}

#[cfg(test)]
mod tests {
    use super::Bytes;
    use crate::common::types::CelString;
    use crate::common::value::{CowVal, Val};
    use std::sync::Arc;

    #[test]
    fn as_borrowed_outlives_the_bytes() {
        let owned = vec![1u8, 2, 3, 4];
        let tail = {
            let b = Bytes::from(owned.as_slice());
            // `b` is dropped at the end of this block; the slice is not tied to it
            b.as_borrowed().map(|b| &b[1..])
        };
        assert_eq!(tail, Some(&[2u8, 3, 4][..]));
        assert!(std::ptr::eq(tail.unwrap().as_ptr(), owned[1..].as_ptr()));
        assert_eq!(Bytes::from(vec![1u8]).as_borrowed(), None);
    }

    #[test]
    fn clone_is_shallow() {
        let b = Bytes::from(vec![1u8, 2, 3]);
        let cloned = b.clone();
        assert!(std::ptr::eq(b.inner(), cloned.inner()));
        let boxed = b.clone_as_boxed();
        let back = boxed.downcast_ref::<Bytes>().unwrap();
        assert!(std::ptr::eq(b.inner(), back.inner()));
    }

    #[test]
    fn from_a_shared_arc_shares_it() {
        let arc = Arc::new(vec![1u8, 2, 3]);
        let b = Bytes::from(arc.clone());
        assert!(std::ptr::eq(b.inner(), arc.as_slice()));
        assert!(std::ptr::eq(
            b.clone().into_static().inner(),
            arc.as_slice()
        ));
    }

    #[test]
    fn into_inner_of_shared_bytes_leaves_them_intact() {
        let b = Bytes::from(vec![1u8]);
        let shared = b.clone();
        let mut inner = b.into_inner();
        inner.push(2);
        assert_eq!(shared.inner(), &[1u8]);
    }

    #[test]
    fn debug_shows_the_bytes() {
        assert_eq!(format!("{:?}", Bytes::from(&[1u8, 2][..])), "Bytes([1, 2])");
        assert_eq!(format!("{:?}", Bytes::from(vec![1u8, 2])), "Bytes([1, 2])");
    }

    #[test]
    fn bytes_of_owned_string_moves_the_buffer() {
        let s = CelString::from(String::from("cel"));
        let ptr = s.inner().as_ptr();
        let arg: CowVal<'_, '_> = CowVal::owned(s);
        let out = super::string_to_bytes(vec![arg]).unwrap();
        let b = out.downcast_ref::<Bytes>().unwrap();
        assert!(std::ptr::eq(b.inner().as_ptr(), ptr));
    }

    #[test]
    fn bytes_of_borrowed_string_keeps_the_borrow() {
        let owned = String::from("cel");
        let arg: CowVal<'_, '_> = CowVal::owned(CelString::from(owned.as_str()));
        let out = super::string_to_bytes(vec![arg]).unwrap();
        let b = out.downcast_ref::<Bytes>().unwrap();
        assert!(std::ptr::eq(b.inner(), owned.as_bytes()));
    }
}
