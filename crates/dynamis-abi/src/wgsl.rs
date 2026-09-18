use crate::{body, constant, constraint, counter, shape};

macro_rules! declare_constants {
    ($($(#[$meta:meta])* pub const $name:ident: $ty:ty = $value:expr;)*) => {
        $( $(#[$meta])* pub const $name: $ty = $value; )*

        pub(crate) fn emit_constants(out: &mut String) {
            $(
                assert!(
                    $name as u128 <= u32::MAX as u128,
                    concat!(stringify!($name), " must fit a 32 bit shader constant")
                );
                out.push_str(&format!("const {}: u32 = {}u;\n", stringify!($name), $name));
            )*
        }
    };
}

pub(crate) use declare_constants;

const DOF_PREDICATES: &str = "fn dof_locked(flags: u32, index: u32) -> bool { return (flags & (DOF_LOCKED << index)) != 0u; }
fn dof_limited(flags: u32, index: u32) -> bool { return (flags & (DOF_LIMITED << index)) != 0u; }
fn dof_driven(flags: u32, index: u32) -> bool { return (flags & (DOF_DRIVEN << index)) != 0u; }
";

/// Emits the predicate a declaration's capability answers, naming every kind it holds for. A kind
/// added to a declaration therefore reaches the device without the shaders naming it again.
pub(crate) fn predicate(out: &mut String, name: &str, covered: impl Iterator<Item = String>) {
    out.push_str("fn ");
    out.push_str(name);
    out.push_str("(kind: u32) -> bool { return ");
    let mut named = false;
    for kind in covered {
        if named {
            out.push_str(" || ");
        }
        out.push_str("kind == ");
        out.push_str(&kind);
        named = true;
    }
    if !named {
        out.push_str("false");
    }
    out.push_str("; }\n");
}

pub fn constants_wgsl() -> String {
    let mut out = String::new();
    constant::emit_constants(&mut out);
    counter::constants_wgsl(&mut out);
    out.push_str(&format!(
        "const NO_HIT: f32 = {:e};
",
        constant::NO_HIT
    ));
    for (name, value) in [
        ("SOLVER_VELOCITY_SCALE", constant::SOLVER_VELOCITY_SCALE),
        ("SOLVER_POSITION_SCALE", constant::SOLVER_POSITION_SCALE),
    ] {
        out.push_str(&format!(
            "const {name}: f32 = {:e};
",
            value
        ));
    }
    out.push_str(DOF_PREDICATES);
    body::emit_predicates(&mut out);
    constraint::emit_predicates(&mut out);
    shape::emit_predicates(&mut out);
    out
}

pub fn step_reset_wgsl() -> String {
    let mut out = String::new();
    counter::step_reset_wgsl(&mut out);
    out
}
