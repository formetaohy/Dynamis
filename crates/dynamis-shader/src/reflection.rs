use dynamis_gpu::BindingKind;
use naga::front::wgsl;
use naga::{
    AddressSpace, GlobalVariable, Handle, Module, ResourceBinding, Scalar, ScalarKind,
    StorageAccess, Type, TypeInner, VectorSize,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderBinding {
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
    pub name: String,
    pub element: String,
}

pub fn reflect(source: &str) -> Vec<ShaderBinding> {
    let module = wgsl::parse_str(source).unwrap_or_else(|error| {
        panic!(
            "the assembled shader is not valid WGSL:\n{}",
            error.emit_to_string(source)
        )
    });
    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| {
            variable
                .binding
                .map(|binding| binding_of(&module, variable, binding))
        })
        .collect::<Vec<_>>();
    bindings.sort_by_key(|binding| (binding.group, binding.binding));
    assert_unique(&bindings);
    bindings
}

fn binding_of(
    module: &Module,
    variable: &GlobalVariable,
    binding: ResourceBinding,
) -> ShaderBinding {
    ShaderBinding {
        group: binding.group,
        binding: binding.binding,
        kind: kind_of(&variable.space),
        name: variable
            .name
            .clone()
            .unwrap_or_else(|| panic!("every shader binding needs a name")),
        element: element_of(module, variable.ty),
    }
}

fn kind_of(space: &AddressSpace) -> BindingKind {
    match *space {
        AddressSpace::Uniform => BindingKind::Uniform,
        AddressSpace::Storage { access } => {
            if access.contains(StorageAccess::STORE) {
                BindingKind::ReadWriteStorage
            } else {
                BindingKind::ReadOnlyStorage
            }
        }
        ref other => panic!("a stage binding must be a uniform or storage buffer, not {other:?}"),
    }
}

fn element_of(module: &Module, ty: Handle<Type>) -> String {
    match module.types[ty].inner {
        TypeInner::Array { base, .. } => match module.types[base].inner {
            TypeInner::Array { .. } => {
                panic!("a stream binding cannot expose an array of arrays")
            }
            _ => element_of(module, base),
        },
        TypeInner::Atomic(scalar) => scalar_name(scalar),
        TypeInner::Scalar(scalar) => scalar_name(scalar),
        TypeInner::Vector { size, scalar } => vector_name(size, scalar),
        TypeInner::Struct { .. } => module.types[ty]
            .name
            .clone()
            .unwrap_or_else(|| panic!("every stream record needs a name")),
        _ => panic!("a stream binding must expose a scalar, a vector or a named record"),
    }
}

fn scalar_name(scalar: Scalar) -> String {
    match (scalar.kind, scalar.width) {
        (ScalarKind::Uint, 4) => "u32".to_owned(),
        (ScalarKind::Sint, 4) => "i32".to_owned(),
        (ScalarKind::Float, 4) => "f32".to_owned(),
        (kind, width) => panic!("a stream binding cannot expose a {width} byte {kind:?} scalar"),
    }
}

fn vector_name(size: VectorSize, scalar: Scalar) -> String {
    let suffix = match scalar.kind {
        ScalarKind::Uint => "u",
        ScalarKind::Sint => "i",
        ScalarKind::Float => "f",
        ref other => panic!("a stream binding cannot expose a {other:?} vector"),
    };
    assert_eq!(
        scalar.width, 4,
        "a stream binding cannot expose a {} byte vector component",
        scalar.width
    );
    format!("vec{}{suffix}", size as u32)
}

fn assert_unique(bindings: &[ShaderBinding]) {
    for (position, binding) in bindings.iter().enumerate() {
        for other in &bindings[..position] {
            assert!(
                other.group != binding.group || other.binding != binding.binding,
                "the shader declares binding {} of group {} twice",
                binding.binding,
                binding.group,
            );
        }
    }
}
