use crate::common::traits::TraitSet;
use crate::common::value::CowVal;
use crate::runtime::Frame;
use crate::{Env, ExecutionError};

#[allow(dead_code)]
pub struct Overload {
    operator: String,
    operand_trait: TraitSet,
    op: Function,
}

/// A function overload. It receives its arguments as [`CowVal`]s bounded by
/// the caller's `'b` borrow and `'v` value lifetime, and may hand one of
/// them back unchanged.
pub type Function = for<'b, 'v> fn(Vec<CowVal<'b, 'v>>) -> Result<CowVal<'b, 'v>, ExecutionError>;

/// A function overload that is also handed the evaluation it runs in, see
/// [`EvalCtx`]: registered with
/// [`Env::add_overload_with_env`] or [`Env::add_member_overload_with_env`].
///
/// ```
/// use cel::common::functions::EvalCtx;
/// use cel::common::types::{CelInt, STRING_TYPE};
/// use cel::common::value::CowVal;
/// use cel::{Context, Env, ExecutionError, Program, RuntimeOptions};
/// use std::sync::Arc;
///
/// // the steps budget the Env sets
/// fn max_steps<'b, 'v>(
///     ectx: &EvalCtx<'_>,
///     _args: Vec<CowVal<'b, 'v>>,
/// ) -> Result<CowVal<'b, 'v>, ExecutionError> {
///     let max_steps = ectx.env().options().max_steps() as i64;
///     Ok(CowVal::owned(CelInt::from(max_steps)))
/// }
///
/// let mut env = Env::stdlib();
/// env.set_options(RuntimeOptions::default().with_max_steps(100));
/// env.add_overload_with_env("maxSteps", "max_steps_string", vec![STRING_TYPE], max_steps)
///     .unwrap();
/// let ctx = Context::with_env(Arc::new(env));
/// let program = Program::compile("maxSteps('') == 100").unwrap();
/// assert_eq!(program.execute(&ctx), Ok(true.into()));
/// ```
pub type EnvFunction =
    for<'b, 'v> fn(&EvalCtx<'_>, Vec<CowVal<'b, 'v>>) -> Result<CowVal<'b, 'v>, ExecutionError>;

/// The evaluation an [`EnvFunction`] is called from: the [`Env`] it runs
/// under, and, crate-internally, the budget frame it is charged to.
pub struct EvalCtx<'a> {
    env: &'a Env,
    #[cfg_attr(not(feature = "regex"), allow(dead_code))]
    frame: Option<&'a Frame<'a>>,
}

impl<'a> EvalCtx<'a> {
    /// The environment the expression is evaluated under.
    pub fn env(&self) -> &'a Env {
        self.env
    }

    /// The budget frame of the evaluation, if it has one.
    #[cfg_attr(not(feature = "regex"), allow(dead_code))]
    pub(crate) fn frame(&self) -> Option<&'a Frame<'a>> {
        self.frame
    }
}

/// How an overload is called: with its arguments only, or with the
/// evaluation too.
#[derive(Clone, Copy)]
pub(crate) enum Op {
    Plain(Function),
    WithEnv(EnvFunction),
}

impl Op {
    /// Calls the overload on `args`, in the evaluation of `env` and `frame`.
    #[inline(always)]
    pub(crate) fn call<'b, 'v>(
        &self,
        env: &Env,
        frame: Option<&Frame<'_>>,
        args: Vec<CowVal<'b, 'v>>,
    ) -> Result<CowVal<'b, 'v>, ExecutionError> {
        match *self {
            Op::Plain(op) => op(args),
            Op::WithEnv(op) => op(&EvalCtx { env, frame }, args),
        }
    }

    /// The function, when the overload takes its arguments only.
    pub(crate) fn plain(self) -> Option<Function> {
        match self {
            Op::Plain(op) => Some(op),
            Op::WithEnv(_) => None,
        }
    }
}
