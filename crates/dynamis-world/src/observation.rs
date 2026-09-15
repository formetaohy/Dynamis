use super::World;
use super::facts::{Facts, Kind};
use super::ids::IdSpace;
use super::readback::{ConstraintForce, constraint_force_of, joint_state_of};
use super::soft::SoftRuns;
use dynamis_abi::{
    BodyDescriptorRecord, BodyStateRecord, Census, CharacterStateRecord, ConstraintReactionRecord,
    ConstraintRuntimeRecord, JointStateRecord, SoftElementRecord, SoftParticleRecord,
    VehicleStateRecord,
};
use dynamis_gpu::{Publication, SubmissionEncoder};
use dynamis_model::{
    BodyHandle, BodyState, CharacterHandle, CharacterState, ConstraintHandle, ConstraintKind,
    JointState, SoftBodyHandle, SoftElementState, VehicleHandle, VehicleState,
};
use std::mem::size_of;
use wgpu::Device;

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

pub(crate) struct BodyFacts;

impl Kind for BodyFacts {
    type Manifest = u64;
    type Context<'a> = (&'a IdSpace, &'a [BodyDescriptorRecord]);
    type Value = BodyState;

    fn step(manifest: &Self::Manifest) -> u64 {
        *manifest
    }

    fn decode(
        (ids, descriptors): Self::Context<'_>,
        manifest: &Self::Manifest,
        bytes: &[u8],
    ) -> Vec<(u32, Self::Value)> {
        let (records, remainder) = bytes.as_chunks::<{ size_of::<BodyStateRecord>() }>();
        assert!(
            remainder.is_empty(),
            "an observation readback must be a whole number of records"
        );
        let mut states = Vec::with_capacity(records.len());
        for chunk in records {
            let record: BodyStateRecord = bytemuck::pod_read_unaligned(chunk);
            let id = record.body_id as usize;
            assert!(
                id < ids.len(),
                "a state readback returned an out-of-range body id"
            );
            if record.generation != ids.generation(record.body_id) {
                continue;
            }
            let descriptor = descriptors[id];
            states.push((
                record.body_id,
                BodyState {
                    position: record.position,
                    prev_position: record.prev_position,
                    orientation: record.orientation,
                    velocity: record.velocity,
                    angular_velocity: record.angular_velocity,
                    inverse_mass: descriptor.inverse_mass,
                    com: descriptor.com,
                    sleeping: record.sleeping != 0,
                    step: *manifest,
                },
            ));
        }
        states
    }
}

pub(crate) struct JointFacts;

pub(crate) struct JointManifest {
    pub(crate) step: u64,
    pub(crate) dt: f32,
    pub(crate) watched: Vec<ConstraintRow>,
}

pub(crate) struct JointFact {
    pub(crate) handle: ConstraintHandle,
    pub(crate) state: JointState,
    pub(crate) force: ConstraintForce,
}

impl Kind for JointFacts {
    type Manifest = JointManifest;
    type Context<'a> = ();
    type Value = JointFact;

    fn step(manifest: &Self::Manifest) -> u64 {
        manifest.step
    }

    fn decode(
        (): Self::Context<'_>,
        manifest: &Self::Manifest,
        bytes: &[u8],
    ) -> Vec<(u32, Self::Value)> {
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
                row.handle.id,
                JointFact {
                    handle: row.handle,
                    state: joint_state_of(row.kind, *state),
                    force: constraint_force_of(
                        row.handle,
                        row.first,
                        row.second,
                        reaction,
                        manifest.dt,
                    ),
                },
            ));
        }
        joints
    }
}

pub(crate) struct CharacterFacts;

pub(crate) struct CharacterManifest {
    pub(crate) step: u64,
    pub(crate) handles: Vec<(CharacterHandle, u32)>,
}

pub(crate) struct CharacterFact {
    pub(crate) handle: CharacterHandle,
    pub(crate) state: CharacterState,
}

impl Kind for CharacterFacts {
    type Manifest = CharacterManifest;
    type Context<'a> = ();
    type Value = CharacterFact;

    fn step(manifest: &Self::Manifest) -> u64 {
        manifest.step
    }

    fn decode(
        (): Self::Context<'_>,
        manifest: &Self::Manifest,
        bytes: &[u8],
    ) -> Vec<(u32, Self::Value)> {
        let records = dynamis_abi::decode::<CharacterStateRecord>(bytes);
        let mut characters = Vec::with_capacity(manifest.handles.len());
        for (handle, slot) in &manifest.handles {
            let record = records
                .get(*slot as usize)
                .expect("a character observation must cover every live slot");
            assert!(
                record.owns(handle.id, handle.generation),
                "a character observation must return only its own state"
            );
            characters.push((
                handle.id,
                CharacterFact {
                    handle: *handle,
                    state: record.state(),
                },
            ));
        }
        characters
    }
}

pub(crate) struct VehicleFacts;

pub(crate) struct VehicleManifest {
    pub(crate) step: u64,
    pub(crate) handles: Vec<(VehicleHandle, u32)>,
}

pub(crate) struct VehicleFact {
    pub(crate) handle: VehicleHandle,
    pub(crate) state: VehicleState,
}

impl Kind for VehicleFacts {
    type Manifest = VehicleManifest;
    type Context<'a> = ();
    type Value = VehicleFact;

    fn step(manifest: &Self::Manifest) -> u64 {
        manifest.step
    }

    fn decode(
        (): Self::Context<'_>,
        manifest: &Self::Manifest,
        bytes: &[u8],
    ) -> Vec<(u32, Self::Value)> {
        let records = dynamis_abi::decode::<VehicleStateRecord>(bytes);
        let mut vehicles = Vec::with_capacity(manifest.handles.len());
        for (handle, slot) in &manifest.handles {
            let record = records
                .get(*slot as usize)
                .expect("a vehicle observation must cover every live slot");
            assert!(
                record.owns(handle.id, handle.generation),
                "a vehicle observation must return only its own state"
            );
            vehicles.push((
                handle.id,
                VehicleFact {
                    handle: *handle,
                    state: record.state(),
                },
            ));
        }
        vehicles
    }
}

pub(crate) struct SoftFacts;

pub(crate) struct SoftManifest {
    pub(crate) step: u64,
    pub(crate) runs: Vec<(SoftBodyHandle, SoftRuns)>,
}

pub(crate) struct SoftFact {
    pub(crate) handle: SoftBodyHandle,
    pub(crate) positions: Vec<[f32; 3]>,
    pub(crate) elements: Vec<SoftElementState>,
}

impl Kind for SoftFacts {
    type Manifest = SoftManifest;
    type Context<'a> = ();
    type Value = SoftFact;

    fn step(manifest: &Self::Manifest) -> u64 {
        manifest.step
    }

    fn decode(
        (): Self::Context<'_>,
        manifest: &Self::Manifest,
        bytes: &[u8],
    ) -> Vec<(u32, Self::Value)> {
        let mut at = 0usize;
        let mut bodies = Vec::with_capacity(manifest.runs.len());
        for (handle, run) in &manifest.runs {
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
            bodies.push((
                handle.id,
                SoftFact {
                    handle: *handle,
                    positions,
                    elements,
                },
            ));
        }
        assert_eq!(
            at,
            bytes.len(),
            "an observation decodes every byte it copies"
        );
        bodies
    }
}

pub(crate) struct Observations {
    pub(crate) bodies: Facts<BodyFacts>,
    pub(crate) joints: Facts<JointFacts>,
    pub(crate) characters: Facts<CharacterFacts>,
    pub(crate) vehicles: Facts<VehicleFacts>,
    pub(crate) soft: Facts<SoftFacts>,
}

impl Observations {
    pub(crate) fn new() -> Self {
        Self {
            bodies: Facts::new("body state observation", DEPTH),
            joints: Facts::new("joint state observation", DEPTH),
            characters: Facts::new("character state observation", DEPTH),
            vehicles: Facts::new("vehicle state observation", DEPTH),
            soft: Facts::new("soft particle observation", SOFT_DEPTH),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.bodies.reset();
        self.joints.reset();
        self.characters.reset();
        self.vehicles.reset();
        self.soft.reset();
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
        self.retire_device_facts();
        let stale = self
            .bodies
            .alive
            .iter()
            .filter(|handle| !self.body_current(handle.id))
            .map(|handle| handle.id)
            .collect::<Vec<_>>();
        let characters = self.observed.characters.needs_publication(self.clock.step);
        let vehicles = self.observed.vehicles.needs_publication(self.clock.step);
        if !stale.is_empty() || characters || vehicles {
            let mut held = Vec::with_capacity(stale.len());
            for id in stale {
                if self.observed.bodies.watch(id) {
                    held.push(id);
                }
            }
            self.execute(dynamis_pass::Run::Publish);
            self.retire_device_facts();
            for id in held {
                self.observed.bodies.forget(id);
            }
        }
        assert!(
            self.states_current(),
            "a publication must cover every live body"
        );
    }

    pub fn try_state(&mut self, handle: BodyHandle) -> Option<BodyState> {
        self.validate(handle);
        self.observed.bodies.watch(handle.id);
        self.observed.bodies.age()?;
        self.observed.bodies.get(handle.id).copied()
    }

    pub fn stop_observing_body(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.observed.bodies.stop_watching(handle.id);
    }

    pub fn try_joint_states(&mut self) -> Option<Observation<Vec<(ConstraintHandle, JointState)>>> {
        let constraints = &self.constraints;
        self.observed
            .joints
            .watch_all(constraints.alive.iter().map(|handle| handle.id));
        let step = self.observed.joints.age()?;
        let states = self
            .observed
            .joints
            .entries()
            .filter(|(_, fact)| constraints.is_alive(fact.handle))
            .map(|(_, fact)| (fact.handle, fact.state))
            .collect();
        Some(Observation {
            step,
            value: states,
        })
    }

    pub fn try_joint_state(&mut self, handle: ConstraintHandle) -> Option<Observation<JointState>> {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle.id);
        let step = self.observed.joints.age()?;
        let state = self.joint_fact(handle).state;
        Some(Observation { step, value: state })
    }

    pub fn try_constraint_forces(&mut self) -> Option<Observation<Vec<ConstraintForce>>> {
        let constraints = &self.constraints;
        self.observed
            .joints
            .watch_all(constraints.alive.iter().map(|handle| handle.id));
        let step = self.observed.joints.age()?;
        let forces = self
            .observed
            .joints
            .entries()
            .filter(|(_, fact)| constraints.is_alive(fact.handle))
            .map(|(_, fact)| fact.force)
            .collect();
        Some(Observation {
            step,
            value: forces,
        })
    }

    pub fn try_constraint_force(
        &mut self,
        handle: ConstraintHandle,
    ) -> Option<Observation<ConstraintForce>> {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle.id);
        let step = self.observed.joints.age()?;
        let force = self.joint_fact(handle).force;
        Some(Observation { step, value: force })
    }

    pub fn stop_observing_joints(&mut self) {
        self.observed.joints.stop();
    }

    pub fn inspect_joint_states(&mut self) -> Vec<(ConstraintHandle, JointState)> {
        self.observe_every_joint();
        self.inspect_facts();
        let constraints = &self.constraints;
        self.observed
            .joints
            .entries()
            .filter(|(_, fact)| constraints.is_alive(fact.handle))
            .map(|(_, fact)| (fact.handle, fact.state))
            .collect()
    }

    pub fn inspect_joint_state(&mut self, handle: ConstraintHandle) -> JointState {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle.id);
        self.inspect_facts();
        self.joint_fact(handle).state
    }

    pub fn inspect_constraint_forces(&mut self) -> Vec<ConstraintForce> {
        self.observe_every_joint();
        self.inspect_facts();
        let constraints = &self.constraints;
        self.observed
            .joints
            .entries()
            .filter(|(_, fact)| constraints.is_alive(fact.handle))
            .map(|(_, fact)| fact.force)
            .collect()
    }

    pub fn inspect_constraint_force(&mut self, handle: ConstraintHandle) -> ConstraintForce {
        self.validate_constraint(handle);
        self.observed.joints.watch(handle.id);
        self.inspect_facts();
        self.joint_fact(handle).force
    }

    pub fn try_character_state(
        &mut self,
        handle: CharacterHandle,
    ) -> Option<Observation<CharacterState>> {
        self.characters.validate(handle);
        self.observed.characters.watch(handle.id);
        let step = self.observed.characters.age()?;
        let fact = self.character_fact(handle)?;
        Some(Observation {
            step,
            value: fact.state,
        })
    }

    pub fn inspect_character_state(&mut self, handle: CharacterHandle) -> CharacterState {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        let census = self.census();
        let live = self.live(&census);
        self.apply_plan(&live);
        self.flush_rows();
        let slot = self.characters.slot_of(handle);
        let states = &self.backend.streams.rigid.character_states;
        let stride = states.stride();
        let buffer = states.buffer().clone();
        let raw = self.read_regions(
            "character state",
            &[(buffer, u64::from(slot) * stride, stride)],
        );
        let record = dynamis_abi::decode::<CharacterStateRecord>(&raw)
            .first()
            .copied()
            .expect("a character readback retires exactly one state");
        assert!(
            record.owns(handle.id, handle.generation),
            "a character readback must return its own state"
        );
        record.state()
    }

    pub fn stop_observing_character(&mut self, handle: CharacterHandle) {
        self.characters.validate(handle);
        self.observed.characters.stop_watching(handle.id);
    }

    pub fn try_vehicle_state(
        &mut self,
        handle: VehicleHandle,
    ) -> Option<Observation<VehicleState>> {
        self.vehicles.validate(handle);
        self.observed.vehicles.watch(handle.id);
        let step = self.observed.vehicles.age()?;
        let fact = self.vehicle_fact(handle)?;
        Some(Observation {
            step,
            value: fact.state,
        })
    }

    pub fn inspect_vehicle_state(&mut self, handle: VehicleHandle) -> VehicleState {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        let census = self.census();
        let live = self.live(&census);
        self.apply_plan(&live);
        self.flush_rows();
        let slot = self.vehicles.slot_of(handle);
        let states = &self.backend.streams.rigid.vehicle_states;
        let stride = states.stride();
        let buffer = states.buffer().clone();
        let raw = self.read_regions(
            "vehicle state",
            &[(buffer, u64::from(slot) * stride, stride)],
        );
        let record = dynamis_abi::decode::<VehicleStateRecord>(&raw)
            .first()
            .copied()
            .expect("a vehicle readback retires exactly one state");
        assert!(
            record.owns(handle.id, handle.generation),
            "a vehicle readback must return its own state"
        );
        record.state()
    }

    pub fn stop_observing_vehicle(&mut self, handle: VehicleHandle) {
        self.vehicles.validate(handle);
        self.observed.vehicles.stop_watching(handle.id);
    }

    pub fn try_soft_particles(
        &mut self,
        handle: SoftBodyHandle,
    ) -> Option<Observation<Vec<[f32; 3]>>> {
        self.soft.validate(handle);
        self.observed.soft.watch(handle.id);
        let step = self.observed.soft.age()?;
        let fact = self.soft_fact(handle)?;
        Some(Observation {
            step,
            value: fact.positions.clone(),
        })
    }

    pub fn try_soft_elements(
        &mut self,
        handle: SoftBodyHandle,
    ) -> Option<Observation<Vec<SoftElementState>>> {
        self.soft.validate(handle);
        self.observed.soft.watch(handle.id);
        let step = self.observed.soft.age()?;
        let fact = self.soft_fact(handle)?;
        Some(Observation {
            step,
            value: fact.elements.clone(),
        })
    }

    pub fn stop_observing_soft_body(&mut self, handle: SoftBodyHandle) {
        self.soft.validate(handle);
        self.observed.soft.stop_watching(handle.id);
    }

    fn joint_fact(&self, handle: ConstraintHandle) -> &JointFact {
        self.observed
            .joints
            .get(handle.id)
            .filter(|fact| fact.handle == handle)
            .unwrap_or_else(|| {
                panic!("constraint handle {handle:?} is missing from its publication")
            })
    }

    fn character_fact(&self, handle: CharacterHandle) -> Option<&CharacterFact> {
        self.observed
            .characters
            .get(handle.id)
            .filter(|fact| fact.handle.generation == handle.generation)
    }

    fn vehicle_fact(&self, handle: VehicleHandle) -> Option<&VehicleFact> {
        self.observed
            .vehicles
            .get(handle.id)
            .filter(|fact| fact.handle.generation == handle.generation)
    }

    fn soft_fact(&self, handle: SoftBodyHandle) -> Option<&SoftFact> {
        self.observed
            .soft
            .get(handle.id)
            .filter(|fact| fact.handle.generation == handle.generation)
    }

    fn observe_every_joint(&mut self) {
        let ids = self
            .constraints
            .alive
            .iter()
            .map(|handle| handle.id)
            .collect::<Vec<_>>();
        self.observed.joints.watch_all(ids);
    }

    fn inspect_facts(&mut self) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        let census = self.census();
        let live = self.live(&census);
        self.apply_plan(&live);
        self.flush_rows();
        self.execute(dynamis_pass::Run::Publish);
        self.retire_device_facts();
    }

    pub(crate) fn states_current(&self) -> bool {
        self.bodies
            .alive
            .iter()
            .all(|handle| self.body_current(handle.id))
    }

    pub(crate) fn body_current(&self, id: u32) -> bool {
        self.observed.bodies.covered(id) == Some(self.clock.step)
    }

    pub(crate) fn completed_step(&self) -> Option<u64> {
        self.clock.step.checked_sub(1)
    }

    pub(crate) fn flush_observed(&mut self) {
        let queue = self.backend.gpu.queue().clone();
        if self.observed.bodies.take_dirty() && !self.observed.bodies.is_empty() {
            self.backend
                .streams
                .state
                .observed_ids
                .write(&queue, bytemuck::cast_slice(self.observed.bodies.keys()));
        }
        let declared = self.observed.joints.keys().to_vec();
        if declared.is_empty() {
            self.observed.joints.set_declared(Vec::new());
        } else if declared.as_slice() != self.observed.joints.declared() {
            self.backend
                .streams
                .state
                .observed_joint_ids
                .write(&queue, bytemuck::cast_slice(&declared));
            self.observed.joints.set_declared(declared);
        }
    }

    pub(crate) fn declare_observations(
        &mut self,
        encoder: &mut SubmissionEncoder,
        census: &Census,
        step: u64,
    ) {
        let device = self.backend.gpu.device().clone();
        let dt = self.clock.sub_dt;
        self.declare_joints(&device, encoder, step, dt);
        self.declare_soft(&device, encoder, step);
        self.declare_characters(&device, encoder, census, step);
        self.declare_vehicles(&device, encoder, census, step);
        self.declare_bodies(&device, encoder, census, step);
    }

    fn declare_bodies(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        census: &Census,
        step: u64,
    ) {
        let count = census.observed;
        if count == 0 {
            return;
        }
        let observed = &self.backend.streams.state.observed_states;
        let bytes = count as u64 * observed.stride();
        let regions = [(observed.buffer(), 0, bytes)];
        let context = (&self.bodies.ids, self.bodies.descriptors.as_slice());
        self.observed
            .bodies
            .publish(context, device, encoder, observed.size(), &regions, step);
    }

    fn declare_joints(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        step: u64,
        dt: f32,
    ) {
        if self.observed.joints.declared().is_empty() {
            return;
        }
        let watched = self.joint_rows();
        let states = &self.backend.streams.state.observed_joint_states;
        let runtimes = &self.backend.streams.state.observed_joint_runtimes;
        let count = watched.len() as u64;
        let regions = [
            (states.buffer(), 0, count * states.stride()),
            (runtimes.buffer(), 0, count * runtimes.stride()),
        ];
        let budget = states.size() + runtimes.size();
        let manifest = JointManifest { step, dt, watched };
        self.observed
            .joints
            .publish((), device, encoder, budget, &regions, manifest);
    }

    fn declare_characters(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        census: &Census,
        step: u64,
    ) {
        if self.observed.characters.is_empty() || census.live_characters == 0 {
            return;
        }
        let states = &self.backend.streams.rigid.character_states;
        let bytes = u64::from(census.characters) * states.stride();
        let regions = [(states.buffer(), 0, bytes)];
        let manifest = CharacterManifest {
            step,
            handles: self.characters.live_slots(),
        };
        self.observed
            .characters
            .publish((), device, encoder, states.size(), &regions, manifest);
    }

    fn declare_vehicles(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        census: &Census,
        step: u64,
    ) {
        if self.observed.vehicles.is_empty() || census.live_vehicles == 0 {
            return;
        }
        let states = &self.backend.streams.rigid.vehicle_states;
        let bytes = u64::from(census.vehicles) * states.stride();
        let regions = [(states.buffer(), 0, bytes)];
        let manifest = VehicleManifest {
            step,
            handles: self.vehicles.live_slots(),
        };
        self.observed
            .vehicles
            .publish((), device, encoder, states.size(), &regions, manifest);
    }

    fn declare_soft(&mut self, device: &Device, encoder: &mut SubmissionEncoder, step: u64) {
        if self.observed.soft.is_empty() {
            return;
        }
        let runs = self
            .observed
            .soft
            .keys()
            .iter()
            .filter_map(|id| self.soft.handle_of(*id))
            .map(|handle| (handle, self.soft.runs_of(handle)))
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
        let manifest = SoftManifest { step, runs };
        self.observed
            .soft
            .publish((), device, encoder, budget, &regions, manifest);
    }

    fn joint_rows(&self) -> Vec<ConstraintRow> {
        self.observed
            .joints
            .declared()
            .iter()
            .map(|id| {
                let index = self.constraints.index_of[*id as usize] as usize;
                let record = self.constraints.records[index];
                ConstraintRow {
                    handle: ConstraintHandle {
                        id: *id,
                        generation: self.constraints.ids.generation(*id),
                    },
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
            .collect()
    }

    pub(crate) fn collect_observations(&mut self) {
        let context = (&self.bodies.ids, self.bodies.descriptors.as_slice());
        self.observed.bodies.collect(context);
        self.observed.joints.collect(());
        self.observed.characters.collect(());
        self.observed.vehicles.collect(());
        self.observed.soft.collect(());
    }

    pub(crate) fn drain_observations(&mut self) {
        let context = (&self.bodies.ids, self.bodies.descriptors.as_slice());
        self.observed.bodies.drain(context);
        self.observed.joints.drain(());
        self.observed.characters.drain(());
        self.observed.vehicles.drain(());
        self.observed.soft.drain(());
    }
}
