use super::World;
use super::body::NEVER_REPORTED;
use super::constraint::Constraints;
use super::ids::IdSpace;
use super::readback::{ConstraintForce, constraint_force_of, joint_state_of};
use super::soft::SoftRuns;
use dynamis_abi::{
    BodyStateRecord, ConstraintReactionRecord, ConstraintRuntimeRecord, JointStateRecord, NO_SLOT,
    SoftElementRecord, SoftParticleRecord,
};
use dynamis_gpu::{Publication, Stream, SubmissionEncoder};
use dynamis_model::{
    BodyHandle, BodyState, ConstraintHandle, ConstraintKind, JointState, SoftBodyHandle,
    SoftElementState,
};
use std::collections::HashMap;
use std::mem::size_of;
use wgpu::{Device, Queue};

pub struct Observation<T> {
    pub step: u64,
    pub value: T,
}

pub(crate) const DEPTH: usize = Publication::<()>::DEPTH;
const SOFT_DEPTH: usize = 2;

#[derive(Clone, Copy)]
pub(crate) struct ConstraintRow {
    pub(crate) handle: ConstraintHandle,
    pub(crate) kind: ConstraintKind,
    pub(crate) first: BodyHandle,
    pub(crate) second: BodyHandle,
}

pub(crate) struct Observations {
    pub(crate) bodies: BodyMirror,
    pub(crate) joints: JointMirror,
    soft: SoftMirror,
}

impl Observations {
    pub(crate) fn new() -> Self {
        Self {
            bodies: BodyMirror::new(),
            joints: JointMirror::new(),
            soft: SoftMirror::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.bodies.reset();
        self.joints.reset();
        self.soft.reset();
    }
}

pub(crate) struct BodyMirror {
    observed: Vec<u32>,
    slot_of: Vec<u32>,
    required: Vec<u32>,
    dirty: bool,
    epoch: Option<u64>,
    publication: Publication<u64>,
}

impl BodyMirror {
    pub(crate) const fn new() -> Self {
        Self {
            observed: Vec::new(),
            slot_of: Vec::new(),
            required: Vec::new(),
            dirty: false,
            epoch: None,
            publication: Publication::new("body state observation", DEPTH),
        }
    }

    pub(crate) fn len(&self) -> u32 {
        self.observed.len() as u32
    }

    pub(crate) fn ids(&self) -> &[u32] {
        &self.observed
    }

    pub(crate) fn observe(&mut self, id: u32) {
        self.insert(id);
    }

    pub(crate) fn require(&mut self, id: u32) {
        if self.insert(id) {
            self.required.push(id);
        }
    }

    pub(crate) fn release_required(&mut self) {
        for id in std::mem::take(&mut self.required) {
            self.forget(id);
        }
    }

    fn insert(&mut self, id: u32) -> bool {
        if self.slot_of.len() <= id as usize {
            self.slot_of.resize(id as usize + 1, NO_SLOT);
        }
        if self.slot_of[id as usize] != NO_SLOT {
            return false;
        }
        self.slot_of[id as usize] = self.observed.len() as u32;
        self.observed.push(id);
        self.dirty = true;
        true
    }

    pub(crate) fn forget(&mut self, id: u32) {
        let Some(slot) = self
            .slot_of
            .get(id as usize)
            .copied()
            .filter(|slot| *slot != NO_SLOT)
        else {
            return;
        };
        self.slot_of[id as usize] = NO_SLOT;
        self.observed.swap_remove(slot as usize);
        if let Some(moved) = self.observed.get(slot as usize) {
            self.slot_of[*moved as usize] = slot;
        }
        self.dirty = true;
    }

    pub(crate) fn reset(&mut self) {
        self.observed.clear();
        self.slot_of.clear();
        self.required.clear();
        self.dirty = false;
        self.epoch = None;
        self.publication.clear();
    }

    fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}

struct JointWatch {
    handles: Vec<ConstraintHandle>,
}

impl JointWatch {
    const fn new() -> Self {
        Self {
            handles: Vec::new(),
        }
    }

    fn watch(&mut self, handle: ConstraintHandle) {
        if let Err(slot) = self
            .handles
            .binary_search_by_key(&handle.id, |candidate| candidate.id)
        {
            self.handles.insert(slot, handle);
        }
    }

    fn forget(&mut self, id: u32) {
        if let Ok(slot) = self
            .handles
            .binary_search_by_key(&id, |candidate| candidate.id)
        {
            self.handles.remove(slot);
        }
    }

    fn clear(&mut self) {
        self.handles.clear();
    }
}

struct JointManifest {
    step: u64,
    dt: f32,
    watched: Vec<ConstraintRow>,
}

struct PublishedJoints {
    step: u64,
    joints: Vec<(ConstraintHandle, JointState, ConstraintForce)>,
}

impl PublishedJoints {
    fn decode(manifest: JointManifest, bytes: &[u8]) -> Self {
        let count = manifest.watched.len();
        let (state_bytes, runtime_bytes) = bytes.split_at(count * size_of::<JointStateRecord>());
        let states = dynamis_abi::decode::<JointStateRecord>(state_bytes);
        let runtimes = dynamis_abi::decode::<ConstraintRuntimeRecord>(runtime_bytes);
        let mut joints = Vec::with_capacity(count);
        for (row, (state, runtime)) in manifest
            .watched
            .iter()
            .zip(states.iter().zip(runtimes.iter()))
        {
            assert_eq!(
                (runtime.constraint_id, runtime.generation),
                (row.handle.id, row.handle.generation),
                "a constraint runtime must answer the joint its publication declared"
            );
            let reaction: ConstraintReactionRecord = runtime.reaction;
            joints.push((
                row.handle,
                joint_state_of(row.kind, *state),
                constraint_force_of(row.handle, row.first, row.second, reaction, manifest.dt),
            ));
        }
        Self {
            step: manifest.step,
            joints,
        }
    }

    fn slot(&self, handle: ConstraintHandle) -> Option<usize> {
        self.joints
            .binary_search_by_key(&handle.id, |(candidate, _, _)| candidate.id)
            .ok()
            .filter(|slot| self.joints[*slot].0 == handle)
    }
}

pub(crate) struct JointMirror {
    watch: JointWatch,
    wrote: Vec<u32>,
    publication: Publication<JointManifest>,
    published: Option<PublishedJoints>,
}

impl JointMirror {
    const fn new() -> Self {
        Self {
            watch: JointWatch::new(),
            wrote: Vec::new(),
            publication: Publication::new("joint state observation", DEPTH),
            published: None,
        }
    }

    pub(crate) fn demand(&self) -> u32 {
        self.watch.handles.len().max(self.wrote.len()) as u32
    }

    pub(crate) fn watched(&self) -> u32 {
        self.watch.handles.len() as u32
    }

    fn reset(&mut self) {
        self.watch.clear();
        self.wrote.clear();
        self.published = None;
        self.publication.clear();
    }

    fn watch(&mut self, handle: ConstraintHandle) {
        self.watch.watch(handle);
    }

    pub(crate) fn watch_every(&mut self, constraints: &Constraints) {
        for handle in &constraints.alive {
            self.watch.watch(*handle);
        }
    }

    pub(crate) fn forget(&mut self, handle: ConstraintHandle) {
        self.watch.forget(handle.id);
    }

    fn stop(&mut self) {
        self.watch.clear();
        self.published = None;
    }

    fn declared(&self) -> Vec<u32> {
        self.watch.handles.iter().map(|handle| handle.id).collect()
    }

    fn refresh(&mut self, queue: &Queue, stream: &Stream, declared: Vec<u32>) {
        if declared.is_empty() {
            self.wrote.clear();
            return;
        }
        if self.wrote == declared {
            return;
        }
        stream.write(queue, bytemuck::cast_slice(&declared));
        self.wrote = declared;
    }

    fn rows(&self, constraints: &Constraints, bodies: &IdSpace) -> Vec<ConstraintRow> {
        self.wrote
            .iter()
            .map(|id| {
                let index = constraints.index_of[*id as usize] as usize;
                let record = constraints.records[index];
                ConstraintRow {
                    handle: ConstraintHandle {
                        id: *id,
                        generation: constraints.ids.generation(*id),
                    },
                    kind: record.constraint_kind(),
                    first: BodyHandle {
                        id: record.first_body_id,
                        generation: bodies.generation(record.first_body_id),
                    },
                    second: BodyHandle {
                        id: record.second_body_id,
                        generation: bodies.generation(record.second_body_id),
                    },
                }
            })
            .collect()
    }

    fn accept(&mut self, bytes: &[u8], manifest: JointManifest) {
        self.published = Some(PublishedJoints::decode(manifest, bytes));
    }

    fn published(&self) -> &PublishedJoints {
        self.published
            .as_ref()
            .expect("a joint inspection publishes before it reads")
    }

    fn states(&self, constraints: &Constraints) -> Vec<(ConstraintHandle, JointState)> {
        self.published()
            .joints
            .iter()
            .filter(|(handle, _, _)| constraints.is_alive(*handle))
            .map(|(handle, state, _)| (*handle, *state))
            .collect()
    }

    fn state(&self, handle: ConstraintHandle) -> JointState {
        self.published().joints[self.slot(handle)].1
    }

    fn forces(&self, constraints: &Constraints) -> Vec<ConstraintForce> {
        self.published()
            .joints
            .iter()
            .filter(|(handle, _, _)| constraints.is_alive(*handle))
            .map(|(_, _, force)| *force)
            .collect()
    }

    fn force(&self, handle: ConstraintHandle) -> ConstraintForce {
        self.published().joints[self.slot(handle)].2
    }

    fn slot(&self, handle: ConstraintHandle) -> usize {
        self.published().slot(handle).unwrap_or_else(|| {
            panic!("constraint handle {handle:?} is missing from its publication")
        })
    }
}

struct PublishedSoft {
    handle: SoftBodyHandle,
    step: u64,
    positions: Vec<[f32; 3]>,
    elements: Vec<SoftElementState>,
}

struct SoftManifest {
    step: u64,
    runs: Vec<(SoftBodyHandle, SoftRuns)>,
}

struct SoftMirror {
    watched: Vec<SoftBodyHandle>,
    slot_of: HashMap<u32, usize>,
    publication: Publication<SoftManifest>,
    published: HashMap<u32, PublishedSoft>,
}

impl SoftMirror {
    fn new() -> Self {
        Self {
            watched: Vec::new(),
            slot_of: HashMap::new(),
            publication: Publication::new("soft particle observation", SOFT_DEPTH),
            published: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.watched.clear();
        self.slot_of.clear();
        self.published.clear();
        self.publication.clear();
    }

    fn watch(&mut self, handle: SoftBodyHandle) {
        if self.slot_of.contains_key(&handle.id) {
            return;
        }
        self.slot_of.insert(handle.id, self.watched.len());
        self.watched.push(handle);
    }

    fn stop(&mut self, handle: SoftBodyHandle) {
        self.published.remove(&handle.id);
        let Some(slot) = self.slot_of.remove(&handle.id) else {
            return;
        };
        self.watched.swap_remove(slot);
        if let Some(moved) = self.watched.get(slot) {
            self.slot_of.insert(moved.id, slot);
        }
    }

    fn watched(&self) -> &[SoftBodyHandle] {
        &self.watched
    }

    fn accept(&mut self, bytes: &[u8], manifest: SoftManifest) {
        let mut at = 0usize;
        for (handle, run) in manifest.runs {
            let particle_bytes = run.particles.len as usize * size_of::<SoftParticleRecord>();
            let records =
                dynamis_abi::decode::<SoftParticleRecord>(&bytes[at..at + particle_bytes]);
            at += particle_bytes;
            assert!(
                records
                    .iter()
                    .all(|record| record.owner == handle.id
                        && record.generation == handle.generation),
                "a soft body observation must return only its own particles"
            );
            let positions = records
                .iter()
                .map(|record| [record.position[0], record.position[1], record.position[2]])
                .collect();
            let mut elements = Vec::new();
            if run.elements.len > 0 {
                let element_bytes = run.elements.len as usize * size_of::<SoftElementRecord>();
                let records =
                    dynamis_abi::decode::<SoftElementRecord>(&bytes[at..at + element_bytes]);
                at += element_bytes;
                elements = records.iter().map(SoftElementRecord::state).collect();
            }
            self.published.insert(
                handle.id,
                PublishedSoft {
                    handle,
                    step: manifest.step,
                    positions,
                    elements,
                },
            );
        }
        assert_eq!(
            at,
            bytes.len(),
            "an observation decodes every byte it copies"
        );
    }

    fn observation(&self, handle: SoftBodyHandle) -> Option<&PublishedSoft> {
        self.published
            .get(&handle.id)
            .filter(|fact| fact.handle.generation == handle.generation)
    }
}

impl World {
    pub fn poll(&mut self) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.backend.gpu.assert_alive();
        self.resolve_queries();
        self.drain_readbacks();
        let stale = self
            .bodies
            .alive
            .iter()
            .filter(|handle| !self.body_current(handle.id))
            .map(|handle| handle.id)
            .collect::<Vec<_>>();
        if !stale.is_empty() {
            for id in stale {
                self.observed.bodies.require(id);
            }
            self.execute(dynamis_pass::Run::Publish);
            self.drain_readbacks();
            self.observed.bodies.release_required();
        }
        assert!(
            self.states_current(),
            "a publication must cover every live body"
        );
    }

    pub fn try_state(&mut self, handle: BodyHandle) -> Option<BodyState> {
        self.validate(handle);
        self.observed.bodies.observe(handle.id);
        self.observed.bodies.epoch?;
        self.bodies.states[handle.id as usize]
    }

    pub fn stop_observing_body(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.observed.bodies.forget(handle.id);
        self.bodies.states[handle.id as usize] = None;
        self.bodies.covered[handle.id as usize] = NEVER_REPORTED;
    }

    pub fn try_joint_states(&mut self) -> Option<Observation<Vec<(ConstraintHandle, JointState)>>> {
        self.observed.joints.watch_every(&self.constraints);
        let published = self.observed.joints.published.as_ref()?;
        let step = published.step;
        let states = published
            .joints
            .iter()
            .filter(|(handle, _, _)| self.constraints.is_alive(*handle))
            .map(|(handle, state, _)| (*handle, *state))
            .collect();
        Some(Observation {
            step,
            value: states,
        })
    }

    pub fn try_joint_state(&mut self, handle: ConstraintHandle) -> Option<Observation<JointState>> {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle);
        let published = self.observed.joints.published.as_ref()?;
        let slot = published.slot(handle)?;
        Some(Observation {
            step: published.step,
            value: published.joints[slot].1,
        })
    }

    pub fn try_constraint_forces(&mut self) -> Option<Observation<Vec<ConstraintForce>>> {
        self.observed.joints.watch_every(&self.constraints);
        let published = self.observed.joints.published.as_ref()?;
        let forces = published
            .joints
            .iter()
            .filter(|(handle, _, _)| self.constraints.is_alive(*handle))
            .map(|(_, _, force)| *force)
            .collect();
        Some(Observation {
            step: published.step,
            value: forces,
        })
    }

    pub fn try_constraint_force(
        &mut self,
        handle: ConstraintHandle,
    ) -> Option<Observation<ConstraintForce>> {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle);
        let published = self.observed.joints.published.as_ref()?;
        let slot = published.slot(handle)?;
        Some(Observation {
            step: published.step,
            value: published.joints[slot].2,
        })
    }

    pub fn stop_observing_joints(&mut self) {
        self.observed.joints.stop();
    }

    pub fn inspect_joint_states(&mut self) -> Vec<(ConstraintHandle, JointState)> {
        self.observed.joints.watch_every(&self.constraints);
        self.inspect_joints();
        self.observed.joints.states(&self.constraints)
    }

    pub fn inspect_joint_state(&mut self, handle: ConstraintHandle) -> JointState {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle);
        self.inspect_joints();
        self.observed.joints.state(handle)
    }

    pub fn inspect_constraint_forces(&mut self) -> Vec<ConstraintForce> {
        self.observed.joints.watch_every(&self.constraints);
        self.inspect_joints();
        self.observed.joints.forces(&self.constraints)
    }

    pub fn inspect_constraint_force(&mut self, handle: ConstraintHandle) -> ConstraintForce {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle);
        self.inspect_joints();
        self.observed.joints.force(handle)
    }

    fn inspect_joints(&mut self) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        self.execute(dynamis_pass::Run::Publish);
        self.drain_readbacks();
    }

    pub fn try_soft_particles(
        &mut self,
        handle: SoftBodyHandle,
    ) -> Option<Observation<Vec<[f32; 3]>>> {
        self.soft.validate(handle);
        self.observed.soft.watch(handle);
        let fact = self.observed.soft.observation(handle)?;
        Some(Observation {
            step: fact.step,
            value: fact.positions.clone(),
        })
    }

    pub fn try_soft_elements(
        &mut self,
        handle: SoftBodyHandle,
    ) -> Option<Observation<Vec<SoftElementState>>> {
        self.soft.validate(handle);
        self.observed.soft.watch(handle);
        let fact = self.observed.soft.observation(handle)?;
        Some(Observation {
            step: fact.step,
            value: fact.elements.clone(),
        })
    }

    pub fn stop_observing_soft_body(&mut self, handle: SoftBodyHandle) {
        self.soft.validate(handle);
        self.observed.soft.stop(handle);
    }

    pub(crate) fn states_current(&self) -> bool {
        self.bodies
            .alive
            .iter()
            .all(|handle| self.body_current(handle.id))
    }

    pub(crate) fn body_current(&self, id: u32) -> bool {
        self.bodies.covered[id as usize] == self.clock.step
    }

    pub(crate) fn completed_step(&self) -> Option<u64> {
        self.clock.step.checked_sub(1)
    }

    pub(crate) fn flush_observed(&mut self) {
        let queue = self.backend.gpu.queue().clone();
        if self.observed.bodies.take_dirty() && !self.observed.bodies.ids().is_empty() {
            self.backend
                .streams
                .state
                .observed_ids
                .write(&queue, bytemuck::cast_slice(self.observed.bodies.ids()));
        }
        let declared = self.observed.joints.declared();
        self.observed.joints.refresh(
            &queue,
            &self.backend.streams.state.observed_joint_ids,
            declared,
        );
    }

    pub(crate) fn declare_observations(&mut self, encoder: &mut SubmissionEncoder, step: u64) {
        let device = self.backend.gpu.device().clone();
        let dt = self.clock.sub_dt;
        self.declare_joints(&device, encoder, step, dt);
        self.declare_soft(&device, encoder, step);
        self.declare_bodies(&device, encoder, step);
    }

    fn declare_joints(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        step: u64,
        dt: f32,
    ) {
        if self.observed.joints.wrote.is_empty() {
            return;
        }
        let watched = self
            .observed
            .joints
            .rows(&self.constraints, &self.bodies.ids);
        let states = &self.backend.streams.state.observed_joint_states;
        let runtimes = &self.backend.streams.state.observed_joint_runtimes;
        let count = watched.len() as u64;
        let regions = [
            (states.buffer(), 0, count * states.stride()),
            (runtimes.buffer(), 0, count * runtimes.stride()),
        ];
        let budget = states.size() + runtimes.size();
        self.observed.joints.publication.reserve(device, budget);
        let manifest = JointManifest { step, dt, watched };
        if let Some((manifest, bytes)) = self
            .observed
            .joints
            .publication
            .declare(encoder, &regions, step, manifest)
        {
            self.observed.joints.accept(&bytes, manifest);
        }
    }

    fn declare_soft(&mut self, device: &Device, encoder: &mut SubmissionEncoder, step: u64) {
        let watched = self.observed.soft.watched().to_vec();
        if watched.is_empty() {
            return;
        }
        let runs = watched
            .iter()
            .filter(|handle| self.soft.is_alive(**handle))
            .map(|handle| (*handle, self.soft.runs_of(*handle)))
            .collect::<Vec<_>>();
        let mut regions = Vec::with_capacity(runs.len() * 2);
        let particles = self.backend.streams.soft.particles.buffer();
        let elements = self.backend.streams.soft.elements.buffer();
        for (_, run) in &runs {
            regions.push((
                particles,
                u64::from(run.particles.offset) * size_of::<SoftParticleRecord>() as u64,
                u64::from(run.particles.len) * size_of::<SoftParticleRecord>() as u64,
            ));
            if run.elements.len > 0 {
                regions.push((
                    elements,
                    u64::from(run.elements.offset) * size_of::<SoftElementRecord>() as u64,
                    u64::from(run.elements.len) * size_of::<SoftElementRecord>() as u64,
                ));
            }
        }
        let budget =
            self.backend.streams.soft.particles.size() + self.backend.streams.soft.elements.size();
        self.observed.soft.publication.reserve(device, budget);
        let manifest = SoftManifest { step, runs };
        if let Some((manifest, bytes)) = self
            .observed
            .soft
            .publication
            .declare(encoder, &regions, step, manifest)
        {
            self.observed.soft.accept(&bytes, manifest);
        }
    }

    fn declare_bodies(&mut self, device: &Device, encoder: &mut SubmissionEncoder, step: u64) {
        let count = self.observed.bodies.len();
        if count == 0 {
            return;
        }
        let observed = &self.backend.streams.state.observed_states;
        let bytes = count as u64 * observed.stride();
        let regions = [(observed.buffer(), 0, bytes)];
        self.observed
            .bodies
            .publication
            .reserve(device, observed.size());
        if let Some((sequence, bytes)) = self
            .observed
            .bodies
            .publication
            .declare(encoder, &regions, step, step)
        {
            self.consume_observations(sequence, &bytes);
        }
    }

    pub(crate) fn collect_observations(&mut self) {
        for (sequence, bytes) in self.observed.bodies.publication.collect() {
            self.consume_observations(sequence, &bytes);
        }
        for (manifest, bytes) in self.observed.joints.publication.collect() {
            self.observed.joints.accept(&bytes, manifest);
        }
        for (manifest, bytes) in self.observed.soft.publication.collect() {
            self.observed.soft.accept(&bytes, manifest);
        }
    }

    pub(crate) fn drain_observations(&mut self) {
        for (sequence, bytes) in self.observed.bodies.publication.drain() {
            self.consume_observations(sequence, &bytes);
        }
        for (manifest, bytes) in self.observed.joints.publication.drain() {
            self.observed.joints.accept(&bytes, manifest);
        }
        for (manifest, bytes) in self.observed.soft.publication.drain() {
            self.observed.soft.accept(&bytes, manifest);
        }
    }

    pub(crate) fn consume_observations(&mut self, sequence: u64, bytes: &[u8]) {
        let (records, remainder) = bytes.as_chunks::<{ size_of::<BodyStateRecord>() }>();
        assert!(
            remainder.is_empty(),
            "an observation readback must be a whole number of records"
        );
        for chunk in records {
            self.accept_body(sequence, bytemuck::pod_read_unaligned(chunk));
        }
        self.observed.bodies.epoch = Some(sequence);
    }

    fn accept_body(&mut self, sequence: u64, record: BodyStateRecord) {
        let id = record.body_id as usize;
        assert!(
            id < self.bodies.ids.len(),
            "a state readback returned an out-of-range body id"
        );
        if record.generation != self.bodies.ids.generation(record.body_id)
            || self.bodies.index_of[id] == u32::MAX
        {
            return;
        }
        self.bodies.states[id] = Some(BodyState {
            position: record.position,
            prev_position: record.prev_position,
            orientation: record.orientation,
            velocity: record.velocity,
            angular_velocity: record.angular_velocity,
            inverse_mass: self.bodies.descriptors[id].inverse_mass,
            com: self.bodies.descriptors[id].com,
            sleeping: record.sleeping != 0,
            step: sequence,
        });
        self.bodies.covered[id] = sequence + 1;
    }
}
