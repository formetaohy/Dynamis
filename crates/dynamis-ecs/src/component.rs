use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

pub trait Component: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> Component for T {}

pub(crate) struct ComponentMeta {
    pub(crate) layout: Layout,
    pub(crate) drop: unsafe fn(*mut u8),
}

struct Registry {
    index_of: HashMap<TypeId, u32>,
    metas: Vec<ComponentMeta>,
}

impl Registry {
    fn descriptor<T: Component>() -> (u32, ComponentMeta) {
        let type_id = TypeId::of::<T>();
        let mut registry = REGISTRY.lock().expect("component registry mutex poisoned");
        if let Some(&index) = registry.index_of.get(&type_id) {
            return (index, registry.metas[index as usize].clone_meta());
        }
        let index = registry.next_index();
        let meta = ComponentMeta {
            layout: Layout::new::<T>(),
            drop: drop_typed::<T>,
        };
        registry.index_of.insert(type_id, index);
        registry.metas.push(meta.clone_meta());
        (index, meta)
    }

    fn next_index(&mut self) -> u32 {
        u32::try_from(self.metas.len()).expect("component type count overflow")
    }
}

impl ComponentMeta {
    fn clone_meta(&self) -> ComponentMeta {
        ComponentMeta {
            layout: self.layout,
            drop: self.drop,
        }
    }
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        index_of: HashMap::new(),
        metas: Vec::new(),
    })
});

unsafe fn drop_typed<T>(ptr: *mut u8) {
    unsafe {
        ptr.cast::<T>().drop_in_place();
    }
}

pub(crate) fn component_meta<T: Component>() -> (u32, ComponentMeta) {
    Registry::descriptor::<T>()
}

pub(crate) fn meta_by_index(index: u32) -> ComponentMeta {
    let registry = REGISTRY.lock().expect("component registry mutex poisoned");
    registry
        .metas
        .get(index as usize)
        .unwrap_or_else(|| panic!("no component registered under index {index}"))
        .clone_meta()
}
