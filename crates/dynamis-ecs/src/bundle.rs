use crate::component::{Component, component_meta};
use std::mem::{forget, size_of};

pub trait ComponentBundle: Sized {
    fn into_parts(self) -> Vec<(u32, Vec<u8>)>;
}

impl ComponentBundle for () {
    fn into_parts(self) -> Vec<(u32, Vec<u8>)> {
        Vec::new()
    }
}

fn component_bytes<T: Component>(value: T) -> (u32, Vec<u8>) {
    let (index, _) = component_meta::<T>();
    let mut bytes = vec![0u8; size_of::<T>()];
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&value as *const T).cast::<u8>(),
            bytes.as_mut_ptr(),
            size_of::<T>(),
        );
    }
    forget(value);
    (index, bytes)
}

macro_rules! impl_bundle {
    ($($name:ident: $typ:ident),+ $(,)?) => {
        impl<$($typ: Component),+> ComponentBundle for ($($typ,)+) {
            fn into_parts(self) -> Vec<(u32, Vec<u8>)> {
                let ($($name,)+) = self;
                vec![$(component_bytes::<$typ>($name)),+]
            }
        }
    };
}

impl_bundle!(a: A);
impl_bundle!(a: A, b: B);
impl_bundle!(a: A, b: B, c: C);
impl_bundle!(a: A, b: B, c: C, d: D);
impl_bundle!(a: A, b: B, c: C, d: D, e: E);
impl_bundle!(a: A, b: B, c: C, d: D, e: E, f: F);
