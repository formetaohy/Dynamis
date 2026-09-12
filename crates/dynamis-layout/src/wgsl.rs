use crate::{constant, counter};

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

pub fn constants_wgsl() -> String {
    let mut out = String::new();
    constant::emit_constants(&mut out);
    counter::emit_constants(&mut out);
    out.push_str(&format!("const NO_HIT: f32 = {:e};\n", constant::NO_HIT));
    out
}
