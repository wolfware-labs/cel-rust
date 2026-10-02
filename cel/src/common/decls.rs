use crate::common::functions::Function;
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

    pub fn find_overload(
        &self,
        member_function: bool,
        args: &[CowVal<'_, '_>],
    ) -> Option<Function> {
        self.find_overload_decl(member_function, args)
            .map(|overload| overload.op)
    }

    /// The overload [`find_overload`](Self::find_overload) would call.
    pub(crate) fn find_overload_decl(
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

    /// Marks the overload `id` as the standard library's `builtin`, which the
    /// interpreter may run itself. Only the standard library marks overloads,
    /// so an overload an embedder declares is never mistaken for one.
    #[cfg_attr(not(feature = "regex"), allow(dead_code))]
    pub(crate) fn mark_builtin(&mut self, id: &str, builtin: Builtin) {
        let overload = self
            .overloads
            .iter_mut()
            .find(|overload| overload.id == id)
            .expect("the builtin overload is declared");
        overload.builtin = Some(builtin);
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
        op: Function,
    ) -> Result<(), DeclarationError> {
        if self.is_present(&id, member_function, &arg_types) {
            return Err(DeclarationError::duplicate_overload(&self.name, &id));
        }
        self.overloads.push(OverloadDecl {
            id,
            arg_types,
            member_function,
            op,
            builtin: None,
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
    op: Function,
    /// Which standard library overload this is, for the few the interpreter
    /// runs itself under an evaluation budget.
    #[cfg_attr(not(feature = "regex"), allow(dead_code))]
    builtin: Option<Builtin>,
}

impl OverloadDecl {
    pub(crate) fn op(&self) -> Function {
        self.op
    }

    #[cfg_attr(not(feature = "regex"), allow(dead_code))]
    pub(crate) fn builtin(&self) -> Option<Builtin> {
        self.builtin
    }
}

/// A standard library overload the interpreter runs itself, identified by a
/// mark only the standard library can set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "regex"), allow(dead_code))]
pub(crate) enum Builtin {
    /// `string.matches(string)`, run under a budget by
    /// [`Frame::is_match`](crate::runtime::Frame::is_match).
    StringMatches,
}

#[allow(dead_code)]
struct VariableDecl<'a, 'b> {
    name: String,
    var_type: &'a Type,
    value: &'b dyn Val,
}
