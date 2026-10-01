use crate::common::functions::{Function, Op};
use crate::common::types::Type;
use crate::common::value::{CowVal, Val};
use crate::DeclarationError;

pub struct FunctionDecl {
    pub name: String,
    overloads: Vec<OverloadDecl>,
}

impl FunctionDecl {
    pub fn new(name: &str) -> FunctionDecl {
        FunctionDecl {
            name: name.to_string(),
            overloads: Vec::default(),
        }
    }

    /// Finds the overload `args` call, when it takes its arguments only:
    /// `None` for an overload registered with an
    /// [`EnvFunction`](crate::common::functions::EnvFunction), which cannot
    /// be called without the evaluation.
    pub fn find_overload(
        &self,
        member_function: bool,
        args: &[CowVal<'_, '_>],
    ) -> Option<Function> {
        self.find_op(member_function, args).and_then(Op::plain)
    }

    /// How to call the overload `args` call, whatever its kind.
    pub(crate) fn find_op(&self, member_function: bool, args: &[CowVal<'_, '_>]) -> Option<Op> {
        self.find_overload_decl(member_function, args)
            .map(|overload| overload.op)
    }

    /// The overload [`find_op`](Self::find_op) would call.
    fn find_overload_decl(
        &self,
        member_function: bool,
        args: &[CowVal<'_, '_>],
    ) -> Option<&OverloadDecl> {
        self.overloads.iter().find(|overload| {
            overload.member_function == member_function
                && args.len() == overload.arg_types.len()
                && overload
                    .arg_types
                    .iter()
                    .enumerate()
                    .all(|(i, t)| t.is_assignable(args[i].as_ref()))
        })
    }

    pub(crate) fn has_overload(&self, member_function: bool) -> bool {
        self.overloads
            .iter()
            .any(|overload| overload.member_function == member_function)
    }

    pub(crate) fn add_overload(
        &mut self,
        id: String,
        member_function: bool,
        arg_types: Vec<Type>,
        op: Op,
    ) -> Result<(), DeclarationError> {
        if self.is_present(&id, member_function, &arg_types) {
            return Err(DeclarationError::duplicate_overload(&self.name, &id));
        }
        self.overloads.push(OverloadDecl {
            id,
            arg_types,
            member_function,
            op,
        });
        Ok(())
    }

    fn is_present(&self, name: &str, member_function: bool, arg_types: &[Type]) -> bool {
        for overload in &self.overloads {
            if overload.id == name
                || (overload.member_function == member_function && overload.arg_types == arg_types)
            {
                return true;
            }
        }
        false
    }
}

pub struct OverloadDecl {
    pub id: String,
    arg_types: Vec<Type>,
    //result_type: &'a Type<'a>,
    member_function: bool,
    //operand_traits: TraitSet,
    op: Op,
}

#[allow(dead_code)]
struct VariableDecl<'a, 'b> {
    name: String,
    var_type: &'a Type,
    value: &'b dyn Val,
}
