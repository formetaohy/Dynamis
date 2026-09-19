use super::World;
use super::pool::Pool;
use dynamis_abi::FieldRecord;
use dynamis_model::{FieldDesc, FieldHandle};
use dynamis_state::StateStreams;

#[derive(Clone)]
pub(crate) struct FieldStore {
    pool: Pool<FieldHandle>,
    descs: Vec<FieldDesc>,
    records: Vec<FieldRecord>,
    dirty: bool,
}

impl FieldStore {
    pub(crate) const fn new() -> Self {
        Self {
            pool: Pool::compact("force field"),
            descs: Vec::new(),
            records: Vec::new(),
            dirty: false,
        }
    }

    pub(crate) fn len(&self) -> u32 {
        self.pool.len()
    }

    pub(crate) fn handles(&self) -> &[FieldHandle] {
        self.pool.alive()
    }

    pub(crate) fn add(&mut self, desc: FieldDesc) -> FieldHandle {
        desc.assert_valid();
        let handle = self.pool.acquire();
        self.pool.insert(handle);
        self.hold(handle.id);
        self.descs[handle.id as usize] = desc;
        self.derive();
        handle
    }

    pub(crate) fn update(&mut self, handle: FieldHandle, desc: FieldDesc) {
        desc.assert_valid();
        self.pool.validate(handle);
        self.descs[handle.id as usize] = desc;
        self.derive();
    }

    pub(crate) fn remove(&mut self, handle: FieldHandle) {
        self.pool.retire(handle);
        self.derive();
    }

    pub(crate) fn desc(&self, handle: FieldHandle) -> FieldDesc {
        self.pool.validate(handle);
        self.descs[handle.id as usize]
    }

    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, streams: &StateStreams) {
        if !std::mem::take(&mut self.dirty) {
            return;
        }
        streams
            .fields
            .write(queue, bytemuck::cast_slice(&self.records));
    }

    fn hold(&mut self, id: u32) {
        if self.descs.len() <= id as usize {
            self.descs.resize(id as usize + 1, FieldDesc::VACANT);
        }
    }

    fn derive(&mut self) {
        self.records.clear();
        self.records.extend(
            self.pool
                .alive()
                .iter()
                .map(|handle| FieldRecord::build(&self.descs[handle.id as usize])),
        );
        self.dirty = true;
    }
}

impl World {
    pub fn add_field(&mut self, desc: FieldDesc) -> FieldHandle {
        let handle = self.fields.add(desc);
        self.wake_all();
        handle
    }

    pub fn remove_field(&mut self, handle: FieldHandle) {
        self.fields.remove(handle);
        self.wake_all();
    }

    pub fn update_field(&mut self, handle: FieldHandle, desc: FieldDesc) {
        self.fields.update(handle, desc);
    }

    pub fn field_desc(&self, handle: FieldHandle) -> FieldDesc {
        self.fields.desc(handle)
    }

    pub fn fields(&self) -> &[FieldHandle] {
        self.fields.handles()
    }

    pub fn field_count(&self) -> usize {
        self.fields.len() as usize
    }
}
