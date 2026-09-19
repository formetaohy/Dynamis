use super::World;
use super::device::{Facts, Region};
use super::fact::{FactStore, Kind};
use super::id::IdSpace;
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
    const LABEL: &'static str = "body state observation";

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
            states.push((record.body_id, record.state(&descriptors[id], *manifest)));
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
    const LABEL: &'static str = "joint state observation";

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
    const LABEL: &'static str = "character state observation";

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
    const LABEL: &'static str = "vehicle state observation";

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
    const LABEL: &'static str = "soft particle observation";

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

/// One kind of device fact the host observes: the label it is declared under, the path that
/// declares its publication, and the paths that land its arrivals into the host mirror.
struct ObservedFact {
    label: &'static str,
    declare: fn(&mut World, &Device, &mut SubmissionEncoder, &Census, u64),
    collect: fn(&mut World),
    drain: fn(&mut World),
}

/// The observed facts the host mirrors from the device. One list declares every kind the world
/// observes: the store, the publication a wait owes, and every sweep over the kinds are derived
/// from it, so a kind cannot be declared without the path that lands it, and no sweep can name
/// fewer kinds than the world observes.
macro_rules! observed_facts {
    ( $( $field:ident: $kind:ty, depth $depth:expr,
            declared by $declare:expr, collected by $collect:expr, drained by $drain:expr; )* ) => {
        pub(crate) struct ObservationStore {
            $( pub(crate) $field: FactStore<$kind>, )*
            whole_body_set: bool,
        }

        const OBSERVED_FACTS: &[ObservedFact] = &[
            $( ObservedFact {
                label: <$kind as Kind>::LABEL,
                declare: $declare,
                collect: $collect,
                drain: $drain,
            }, )*
        ];

        const _: () = assert_each_observed_once(OBSERVED_FACTS);

        impl ObservationStore {
            pub(crate) fn new() -> Self {
                Self {
                    $( $field: FactStore::<$kind>::new($depth), )*
                    whole_body_set: false,
                }
            }

            pub(crate) fn reset(&mut self) {
                $( self.$field.reset(); )*
                self.whole_body_set = false;
            }

            /// Whether the host still waits for an observation of a declared kind: a kind owes a
            /// publication while it watches something and the step it last published is not the
            /// step the wait follows.
            pub(crate) fn owes_publication(&self, step: u64) -> bool {
                false $( || self.$field.needs_publication(step) )*
            }
        }

        impl World {
            pub(crate) fn declare_observations(
                &mut self,
                encoder: &mut SubmissionEncoder,
                census: &Census,
                step: u64,
            ) {
                let device = self.backend.gpu.device().clone();
                for fact in OBSERVED_FACTS {
                    (fact.declare)(self, &device, encoder, census, step);
                }
            }

            pub(crate) fn collect_observations(&mut self) {
                for fact in OBSERVED_FACTS {
                    (fact.collect)(self);
                }
            }

            pub(crate) fn drain_observations(&mut self) {
                for fact in OBSERVED_FACTS {
                    (fact.drain)(self);
                }
            }
        }
    };
}

observed_facts! {
    bodies: BodyFacts, depth DEPTH,
        declared by World::declare_bodies,
        collected by |world| {
            let context = (world.bodies.pool.identities(), world.bodies.records.as_slice());
            world.observed.bodies.collect(context);
        },
        drained by |world| {
            let context = (world.bodies.pool.identities(), world.bodies.records.as_slice());
            world.observed.bodies.drain(context);
        };
    joints: JointFacts, depth DEPTH,
        declared by World::declare_joints,
        collected by |world| world.observed.joints.collect(()),
        drained by |world| world.observed.joints.drain(());
    characters: CharacterFacts, depth DEPTH,
        declared by World::declare_characters,
        collected by |world| world.observed.characters.collect(()),
        drained by |world| world.observed.characters.drain(());
    vehicles: VehicleFacts, depth DEPTH,
        declared by World::declare_vehicles,
        collected by |world| world.observed.vehicles.collect(()),
        drained by |world| world.observed.vehicles.drain(());
    soft: SoftFacts, depth SOFT_DEPTH,
        declared by World::declare_soft,
        collected by |world| world.observed.soft.collect(()),
        drained by |world| world.observed.soft.drain(());
}

const fn assert_each_observed_once(facts: &[ObservedFact]) {
    assert!(
        !facts.is_empty(),
        "a world that observes device facts must declare the kinds it observes",
    );
    let mut index = 0;
    while index < facts.len() {
        let mut other = index + 1;
        while other < facts.len() {
            assert!(
                !same_label(facts[index].label, facts[other].label),
                "an observed fact kind must be declared once",
            );
            other += 1;
        }
        index += 1;
    }
}

const fn same_label(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

impl ObservationStore {
    pub(crate) fn observe_body(&mut self, id: u32) {
        self.bodies.watch(id);
    }

    pub(crate) fn observe_whole_body_set(&mut self, ids: impl IntoIterator<Item = u32>) {
        self.whole_body_set = true;
        self.bodies.watch_all(ids);
    }

    pub(crate) fn observes_whole_body_set(&self) -> bool {
        self.whole_body_set
    }

    pub(crate) fn stop_observing_whole_body_set(&mut self) {
        self.whole_body_set = false;
        self.bodies.stop();
    }
}

impl World {
    pub fn poll(&mut self) {
        self.sync(Facts::Arrived);
    }

    pub fn wait(&mut self) {
        self.sync(Facts::Landed);
    }

    pub fn try_state(&mut self, handle: BodyHandle) -> Option<BodyState> {
        self.validate(handle);
        self.observe_body(handle);
        self.observed.bodies.age()?;
        self.observed.bodies.get(handle.id).copied()
    }

    pub fn observe_bodies(&mut self, handles: &[BodyHandle]) {
        for handle in handles {
            self.observe_body(*handle);
        }
    }

    pub fn observe_all_bodies(&mut self) {
        let ids = self
            .bodies
            .pool
            .alive()
            .iter()
            .map(|handle| handle.id)
            .collect::<Vec<_>>();
        self.observed.observe_whole_body_set(ids);
    }

    pub fn stop_observing_all_bodies(&mut self) {
        self.observed.stop_observing_whole_body_set();
    }

    pub fn stop_observing_body(&mut self, handle: BodyHandle) {
        self.validate(handle);
        assert!(
            !self.observed.observes_whole_body_set(),
            "the whole live body set is observed; stop observing the whole set instead"
        );
        self.observed.bodies.stop_watching(handle.id);
    }

    pub(crate) fn observe_body(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.observed.observe_body(handle.id);
    }

    pub fn try_joint_states(&mut self) -> Option<Observation<Vec<(ConstraintHandle, JointState)>>> {
        let constraints = &self.constraints;
        self.observed
            .joints
            .watch_all(constraints.pool.alive().iter().map(|handle| handle.id));
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
            .watch_all(constraints.pool.alive().iter().map(|handle| handle.id));
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
        self.sync(Facts::Retired);
        let region = Region::of(
            &self.backend.streams.rigid.character_states,
            self.characters.slot_of(handle),
            1,
        );
        let record = self.read_one::<CharacterStateRecord>("character state", region);
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
        self.sync(Facts::Retired);
        let region = Region::of(
            &self.backend.streams.rigid.vehicle_states,
            self.vehicles.slot_of(handle),
            1,
        );
        let record = self.read_one::<VehicleStateRecord>("vehicle state", region);
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
            .pool
            .alive()
            .iter()
            .map(|handle| handle.id)
            .collect::<Vec<_>>();
        self.observed.joints.watch_all(ids);
    }

    fn inspect_facts(&mut self) {
        self.sync(Facts::Retired);
        self.execute(dynamis_pass::Run::Publish);
        self.retire_device_facts();
    }

    pub(crate) fn body_current(&self, id: u32) -> bool {
        self.observed.bodies.covered(id) == Some(self.clock.step)
    }

    pub(crate) fn completed_step(&self) -> Option<u64> {
        self.clock.step.checked_sub(1)
    }

    pub(crate) fn stage_observations(&mut self) {
        self.backend.staged.observations = true;
    }

    pub(crate) fn flush_observations(&mut self) {
        if !std::mem::take(&mut self.backend.staged.observations) {
            return;
        }
        let queue = self.backend.gpu.queue().clone();
        if self.observed.bodies.watch_moved() {
            self.backend
                .streams
                .state
                .observed_ids
                .write(&queue, bytemuck::cast_slice(self.observed.bodies.keys()));
        }
        if self.observed.joints.watch_moved() {
            self.backend
                .streams
                .state
                .observed_joint_ids
                .write(&queue, bytemuck::cast_slice(self.observed.joints.keys()));
        }
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
        let context = (
            self.bodies.pool.identities(),
            self.bodies.records.as_slice(),
        );
        self.observed
            .bodies
            .publish(context, device, encoder, observed.size(), &regions, step);
    }

    fn declare_joints(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        _: &Census,
        step: u64,
    ) {
        if self.observed.joints.is_empty() {
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
        let manifest = JointManifest {
            step,
            dt: self.clock.sub_dt,
            watched,
        };
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

    fn declare_soft(
        &mut self,
        device: &Device,
        encoder: &mut SubmissionEncoder,
        _: &Census,
        step: u64,
    ) {
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
            .keys()
            .iter()
            .map(|id| {
                let handle = ConstraintHandle {
                    id: *id,
                    generation: self.constraints.pool.generation(*id),
                };
                let joint = self.joint(handle);
                ConstraintRow {
                    handle,
                    kind: joint.desc.kind(),
                    first: BodyHandle {
                        id: joint.first,
                        generation: self.bodies.pool.generation(joint.first),
                    },
                    second: BodyHandle {
                        id: joint.second,
                        generation: self.bodies.pool.generation(joint.second),
                    },
                }
            })
            .collect()
    }
}
