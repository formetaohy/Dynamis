use super::World;
use super::body::NEVER_REPORTED;
use super::readback::{ConstraintForce, constraint_force_of, joint_state_of};
use super::soft::SoftRuns;
use dynamis_abi::{
    BodyStateRecord, ConstraintReactionRecord, ConstraintRuntimeRecord, JointStateRecord, NO_SLOT,
    SoftElementRecord, SoftParticleRecord,
};
use dynamis_gpu::{Readback, SubmissionEncoder};
use dynamis_model::{
    BodyHandle, BodyState, ConstraintHandle, ConstraintKind, JointState, SoftBodyHandle,
    SoftElementState,
};
use std::collections::{HashMap, VecDeque};
use std::mem::size_of;
use wgpu::{Buffer, Device};

pub struct Observation<T> {
    pub step: u64,
    pub value: T,
}

const DEPTH: usize = Readback::DEPTH;
const SOFT_DEPTH: usize = 2;

#[derive(Clone, Copy)]
pub(crate) struct ConstraintRow {
    pub(crate) handle: ConstraintHandle,
    pub(crate) kind: ConstraintKind,
    pub(crate) first: BodyHandle,
    pub(crate) second: BodyHandle,
}

struct Ring {
    label: &'static str,
    depth: usize,
    readback: Option<Readback>,
}

impl Ring {
    const fn new(label: &'static str, depth: usize) -> Self {
        Self {
            label,
            depth,
            readback: None,
        }
    }

    fn clear(&mut self) {
        self.readback = None;
    }

    fn publish(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        capacity: u64,
        regions: &[(&Buffer, u64, u64)],
        sequence: u64,
    ) -> Option<(u64, Vec<u8>)> {
        let bytes = regions.iter().map(|(_, _, bytes)| *bytes).sum::<u64>();
        assert!(
            bytes > 0 && bytes <= capacity,
            "an observation publishes {bytes} bytes within the {capacity} byte mirror it is sized for"
        );
        if self
            .readback
            .as_ref()
            .is_none_or(|ring| ring.size() < capacity)
        {
            if let Some(ring) = &self.readback {
                assert!(
                    ring.is_idle(),
                    "observation {:?} must drain before its ring reallocates",
                    self.label
                );
            }
            self.readback = Some(Readback::new(device, self.label, capacity, self.depth));
        }
        self.readback
            .as_mut()
            .expect("a publication opens its ring")
            .enqueue_regions(encoder, regions, sequence)
    }

    fn collect(&mut self) -> Vec<(u64, Vec<u8>)> {
        self.readback
            .as_mut()
            .map(Readback::collect)
            .unwrap_or_default()
    }

    fn drain(&mut self) -> Vec<(u64, Vec<u8>)> {
        self.readback
            .as_mut()
            .map(Readback::drain)
            .unwrap_or_default()
    }
}

pub(crate) struct Observations {
    pub(crate) bodies: BodyMirror,
    joints: JointMirror,
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
    ring: Ring,
}

impl BodyMirror {
    pub(crate) const fn new() -> Self {
        Self {
            observed: Vec::new(),
            slot_of: Vec::new(),
            required: Vec::new(),
            dirty: false,
            epoch: None,
            ring: Ring::new("body state observation", DEPTH),
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

    fn reset(&mut self) {
        self.observed.clear();
        self.slot_of.clear();
        self.required.clear();
        self.dirty = false;
        self.epoch = None;
        self.ring.clear();
    }

    fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}

struct JointManifest {
    step: u64,
    dt: f32,
    rows: Vec<ConstraintRow>,
}

struct PublishedJoints {
    step: u64,
    rows: Vec<ConstraintRow>,
    states: Vec<JointState>,
    forces: Vec<ConstraintForce>,
}

impl PublishedJoints {
    fn decode(manifest: JointManifest, bytes: &[u8]) -> Self {
        let count = manifest.rows.len();
        let (state_bytes, runtime_bytes) = bytes.split_at(count * size_of::<JointStateRecord>());
        let states = dynamis_abi::decode::<JointStateRecord>(state_bytes);
        let runtimes = dynamis_abi::decode::<ConstraintRuntimeRecord>(runtime_bytes);
        let mut decoded = Vec::with_capacity(count);
        let mut forces = Vec::with_capacity(count);
        for (row, (state, runtime)) in manifest.rows.iter().zip(states.iter().zip(runtimes.iter()))
        {
            assert_eq!(
                (runtime.constraint_id, runtime.generation),
                (row.handle.id, row.handle.generation),
                "a constraint runtime must answer the row its publication declared"
            );
            let reaction: ConstraintReactionRecord = runtime.reaction;
            decoded.push(joint_state_of(row.kind, *state));
            forces.push(constraint_force_of(
                row.handle,
                row.first,
                row.second,
                reaction,
                manifest.dt,
            ));
        }
        Self {
            step: manifest.step,
            rows: manifest.rows,
            states: decoded,
            forces,
        }
    }

    fn row_of(&self, handle: ConstraintHandle) -> Option<usize> {
        self.rows.iter().position(|row| row.handle == handle)
    }
}

struct JointMirror {
    enabled: bool,
    ring: Ring,
    pending: VecDeque<JointManifest>,
    published: Option<PublishedJoints>,
}

impl JointMirror {
    const fn new() -> Self {
        Self {
            enabled: false,
            ring: Ring::new("joint state observation", DEPTH),
            pending: VecDeque::new(),
            published: None,
        }
    }

    fn reset(&mut self) {
        self.enabled = false;
        self.pending.clear();
        self.published = None;
        self.ring.clear();
    }

    fn declare(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        streams: &super::backend::registry::Streams,
        rows: Vec<ConstraintRow>,
        step: u64,
        dt: f32,
    ) {
        if rows.is_empty() {
            self.published = Some(PublishedJoints {
                step,
                rows,
                states: Vec::new(),
                forces: Vec::new(),
            });
            return;
        }
        let states = &streams.rigid.joint_states;
        let runtimes = &streams.state.constraint_runtime;
        let count = rows.len() as u64;
        let regions = [
            (states.buffer(), 0, count * states.stride()),
            (runtimes.buffer(), 0, count * runtimes.stride()),
        ];
        let capacity = states.size() + runtimes.size();
        if let Some((sequence, bytes)) =
            self.ring.publish(device, encoder, capacity, &regions, step)
        {
            self.accept(sequence, bytes);
        }
        self.pending.push_back(JointManifest { step, dt, rows });
    }

    fn accept(&mut self, sequence: u64, bytes: Vec<u8>) {
        let manifest = self
            .pending
            .pop_front()
            .expect("an observation arrival retires the declaration it answers");
        assert_eq!(
            manifest.step, sequence,
            "observation arrivals must retire in declaration order"
        );
        self.published = Some(PublishedJoints::decode(manifest, &bytes));
    }

    fn collect(&mut self) {
        for (sequence, bytes) in self.ring.collect() {
            self.accept(sequence, bytes);
        }
    }

    fn drain(&mut self) {
        for (sequence, bytes) in self.ring.drain() {
            self.accept(sequence, bytes);
        }
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
    ring: Ring,
    pending: VecDeque<SoftManifest>,
    published: HashMap<u32, PublishedSoft>,
}

impl SoftMirror {
    fn new() -> Self {
        Self {
            watched: Vec::new(),
            slot_of: HashMap::new(),
            ring: Ring::new("soft particle observation", SOFT_DEPTH),
            pending: VecDeque::new(),
            published: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.watched.clear();
        self.slot_of.clear();
        self.pending.clear();
        self.published.clear();
        self.ring.clear();
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

    pub(crate) fn watched(&self) -> &[SoftBodyHandle] {
        &self.watched
    }

    fn declare(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        streams: &super::backend::registry::Streams,
        runs: Vec<(SoftBodyHandle, SoftRuns)>,
        step: u64,
    ) {
        let mut regions = Vec::with_capacity(runs.len() * 2);
        let particles = streams.soft.particles.buffer();
        let elements = streams.soft.elements.buffer();
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
        let capacity = streams.soft.particles.size() + streams.soft.elements.size();
        if let Some((sequence, bytes)) =
            self.ring.publish(device, encoder, capacity, &regions, step)
        {
            self.accept(sequence, bytes);
        }
        self.pending.push_back(SoftManifest { step, runs });
    }

    fn accept(&mut self, sequence: u64, bytes: Vec<u8>) {
        let manifest = self
            .pending
            .pop_front()
            .expect("an observation arrival retires the declaration it answers");
        assert_eq!(
            manifest.step, sequence,
            "observation arrivals must retire in declaration order"
        );
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

    fn collect(&mut self) {
        for (sequence, bytes) in self.ring.collect() {
            self.accept(sequence, bytes);
        }
    }

    fn drain(&mut self) {
        for (sequence, bytes) in self.ring.drain() {
            self.accept(sequence, bytes);
        }
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
        self.observed.joints.enabled = true;
        let published = self.observed.joints.published.as_ref()?;
        let step = published.step;
        let states = self
            .constraints
            .alive
            .iter()
            .filter_map(|handle| {
                let row = published.row_of(*handle)?;
                Some((*handle, published.states[row]))
            })
            .collect();
        Some(Observation {
            step,
            value: states,
        })
    }

    pub fn try_joint_state(&mut self, handle: ConstraintHandle) -> Option<Observation<JointState>> {
        self.validate_constraint(handle);
        self.observed.joints.enabled = true;
        let published = self.observed.joints.published.as_ref()?;
        let row = published.row_of(handle)?;
        Some(Observation {
            step: published.step,
            value: published.states[row],
        })
    }

    pub fn try_constraint_forces(&mut self) -> Option<Observation<Vec<ConstraintForce>>> {
        self.observed.joints.enabled = true;
        let published = self.observed.joints.published.as_ref()?;
        let forces = self
            .constraints
            .alive
            .iter()
            .filter_map(|handle| {
                let row = published.row_of(*handle)?;
                Some(published.forces[row])
            })
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
        self.observed.joints.enabled = true;
        let published = self.observed.joints.published.as_ref()?;
        let row = published.row_of(handle)?;
        Some(Observation {
            step: published.step,
            value: published.forces[row],
        })
    }

    pub fn stop_observing_joints(&mut self) {
        self.observed.joints.enabled = false;
        self.observed.joints.published = None;
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
        if !self.observed.bodies.take_dirty() {
            return;
        }
        let ids = self.observed.bodies.ids();
        if ids.is_empty() {
            return;
        }
        self.backend
            .streams
            .state
            .observed_ids
            .write(self.backend.gpu.queue(), bytemuck::cast_slice(ids));
    }

    pub(crate) fn declare_observations(
        &mut self,
        encoder: &mut SubmissionEncoder,
        step: u64,
    ) -> Option<(u64, Vec<u8>)> {
        let device = self.backend.gpu.device().clone();
        if self.observed.joints.enabled {
            let rows = self
                .constraints
                .alive
                .iter()
                .map(|handle| {
                    let index = self.constraints.index_of[handle.id as usize] as usize;
                    let record = self.constraints.records[index];
                    ConstraintRow {
                        handle: *handle,
                        kind: record.constraint_kind(),
                        first: BodyHandle {
                            id: record.first_body_id,
                            generation: self.bodies.ids.generation(record.first_body_id),
                        },
                        second: BodyHandle {
                            id: record.second_body_id,
                            generation: self.bodies.ids.generation(record.second_body_id),
                        },
                    }
                })
                .collect();
            let dt = self.clock.sub_dt;
            self.observed
                .joints
                .declare(&device, encoder, &self.backend.streams, rows, step, dt);
        }
        let watched = self.observed.soft.watched().to_vec();
        if !watched.is_empty() {
            let runs = watched
                .iter()
                .filter(|handle| self.soft.is_alive(**handle))
                .map(|handle| (*handle, self.soft.runs_of(*handle)))
                .collect();
            self.observed
                .soft
                .declare(&device, encoder, &self.backend.streams, runs, step);
        }
        self.declare_body_observations(encoder, step)
    }

    pub(crate) fn declare_body_observations(
        &mut self,
        encoder: &mut SubmissionEncoder,
        step: u64,
    ) -> Option<(u64, Vec<u8>)> {
        let count = self.observed.bodies.len();
        if count == 0 {
            return None;
        }
        let observed = &self.backend.streams.state.observed_states;
        let bytes = count as u64 * observed.stride();
        self.observed.bodies.ring.publish(
            &self.backend.gpu.device().clone(),
            encoder,
            observed.size(),
            &[(observed.buffer(), 0, bytes)],
            step,
        )
    }

    pub(crate) fn collect_observations(&mut self) {
        for (sequence, bytes) in self.observed.bodies.ring.collect() {
            self.consume_observations(sequence, &bytes);
        }
        self.observed.joints.collect();
        self.observed.soft.collect();
    }

    pub(crate) fn drain_observations(&mut self) {
        for (sequence, bytes) in self.observed.bodies.ring.drain() {
            self.consume_observations(sequence, &bytes);
        }
        self.observed.joints.drain();
        self.observed.soft.drain();
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
