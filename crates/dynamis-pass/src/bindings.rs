use dynamis_gpu::{
    BindingKind, BindingSpec, PipelineHandle, Resources, SlotRef, StorageId, StreamElement,
    TypedSlot,
};
use dynamis_shader::ShaderBinding;
use wgpu::{BindGroup, BindGroupEntry, Device};

#[derive(Clone, Copy)]
pub(crate) struct Binding {
    pub(crate) index: u32,
    pub(crate) slot: SlotRef,
}

impl Binding {
    pub(crate) fn storage_id<R: Resources>(&self, resources: &R) -> StorageId {
        self.slot.resolve(resources).storage_id()
    }
}

pub struct Bindings {
    declarations: Vec<ShaderBinding>,
}

impl Bindings {
    pub fn new(declarations: Vec<ShaderBinding>) -> Self {
        Self { declarations }
    }

    pub fn specs(&self, group: u32) -> Vec<BindingSpec> {
        self.ordered(group)
            .into_iter()
            .map(|entry| {
                assert!(
                    group == 0 || entry.kind == BindingKind::ReadOnlyStorage,
                    "group {group} bindings must be read only"
                );
                BindingSpec {
                    binding: entry.binding,
                    kind: entry.kind,
                }
            })
            .collect()
    }

    pub(crate) fn table(
        &self,
        label: &str,
        group: u32,
        slots: &[(&'static str, SlotRef)],
    ) -> Vec<Binding> {
        self.matched(label, group, slots, |slot| slot.element())
            .into_iter()
            .map(|(declaration, slot)| Binding {
                index: declaration.binding,
                slot: *slot,
            })
            .collect()
    }

    pub fn group(
        &self,
        label: &str,
        handle: &PipelineHandle,
        device: &Device,
        group: u32,
        slots: &[(&'static str, TypedSlot<'_>)],
    ) -> BindGroup {
        let entries = self
            .matched(label, group, slots, |slot| slot.element())
            .into_iter()
            .map(|(declaration, slot)| BindGroupEntry {
                binding: declaration.binding,
                resource: slot.slot().as_binding(),
            })
            .collect::<Vec<_>>();
        handle.create_bind_group(device, group as usize, &entries)
    }

    fn matched<'a, 's, T>(
        &'s self,
        label: &str,
        group: u32,
        slots: &'a [(&'static str, T)],
        element: impl Fn(&T) -> StreamElement,
    ) -> Vec<(&'s ShaderBinding, &'a T)> {
        for (position, (name, _)) in slots.iter().enumerate() {
            assert!(
                slots[..position].iter().all(|(other, _)| other != name),
                "{label:?} binds {name:?} twice"
            );
        }
        let declared = self.ordered(group);
        assert!(
            declared.len() == slots.len(),
            "{label:?} declares {} bindings in group {group} but provides {}",
            declared.len(),
            slots.len()
        );
        declared
            .into_iter()
            .map(|declaration| {
                let slot = slots
                    .iter()
                    .find(|(name, _)| *name == declaration.name)
                    .unwrap_or_else(|| {
                        panic!("{label:?} declares no binding {:?}", declaration.name)
                    });
                let bound = element(&slot.1);
                assert!(
                    declaration.element == bound.wgsl(),
                    "stage {label:?} binds {:?} to the {} stream while its shader declares {:?}",
                    declaration.name,
                    bound.wgsl(),
                    declaration.element,
                );
                (declaration, &slot.1)
            })
            .collect()
    }

    fn ordered(&self, group: u32) -> Vec<&ShaderBinding> {
        let mut declared = self
            .declarations
            .iter()
            .filter(|entry| entry.group == group)
            .collect::<Vec<_>>();
        declared.sort_by_key(|entry| entry.binding);
        for (position, entry) in declared.iter().enumerate() {
            assert!(
                entry.binding == position as u32,
                "the shader binds group {group} index {} where {position} is required",
                entry.binding
            );
        }
        declared
    }
}
