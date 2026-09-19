use super::World;
use crate::backend::registry::STREAM_WRITERS;
use dynamis_domain::StreamWriters;
use dynamis_rigid::RigidStream;
use dynamis_scene::SceneStream;
use dynamis_soft::SoftStream;
use dynamis_state::StateStream;

pub(crate) type Flush = fn(&mut World);

/// Where a declared handover runs. A scene handover carries the records a scene mutation moved,
/// and runs with every mutation the world sweeps. A step handover carries the records the step
/// itself composes, and runs once the step has composed all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Scene,
    Step,
}

/// One declared handover of host records to the device: the streams it carries and the path that
/// performs it. Every stream the stream table declares the host writes owes exactly one entry
/// here, so a scene record cannot be added to a stream table without the path that uploads it,
/// and a stream no handover carries cannot be claimed by one.
pub(crate) struct Handover {
    pub(crate) streams: &'static [&'static str],
    pub(crate) stage: Stage,
    pub(crate) flush: Flush,
}

macro_rules! handovers {
    ( $( $stage:ident { $( $stream:expr ),* $(,)? } => $flush:path ),* $(,)? ) => {
        /// Every declared handover of host records to the device, in the order a sweep runs them:
        /// the streams each hands over, the stage that composes them, and the path that does it.
        pub(crate) const HANDOVERS: &[Handover] = &[
            $(
                Handover {
                    streams: &[ $( $stream.label() ),* ],
                    stage: Stage::$stage,
                    flush: $flush,
                },
            )*
        ];
    };
}

handovers! {
    Scene {
        StateStream::ShapeSources,
        StateStream::ShapeVertices,
        StateStream::ShapeTriangles,
        StateStream::ShapeNodes,
        StateStream::ShapeCells,
    } => World::flush_shapes,
    Scene {
        StateStream::BodyDescriptors,
        StateStream::Colliders,
        StateStream::ColliderOwners,
    } => World::flush_body_records,
    Scene {
        StateStream::ConstraintDescriptors,
    } => World::flush_constraint_records,
    Scene {
        StateStream::Fields,
    } => World::flush_field_records,
    Scene {
        SoftStream::Particles,
        SoftStream::Elements,
        SoftStream::Attachments,
        SoftStream::Adjacency,
        SoftStream::BodyStates,
    } => World::flush_soft_records,
    Scene {
        RigidStream::Characters,
        RigidStream::CharacterInputs,
        RigidStream::CharacterStates,
        RigidStream::CharacterSweeps,
        RigidStream::CharacterHits,
    } => World::flush_characters,
    Scene {
        RigidStream::Vehicles,
        RigidStream::VehicleWheels,
        RigidStream::VehicleInputs,
        RigidStream::VehicleStates,
        RigidStream::VehicleSweeps,
        RigidStream::VehicleHits,
    } => World::flush_vehicles,
    Scene {
        RigidStream::JointRows,
        RigidStream::JointLayers,
        RigidStream::JointComponents,
        RigidStream::JointBatches,
    } => World::flush_joint_order,
    Scene {
        StateStream::EntryBase,
    } => World::flush_entry_base,
    Step {
        StateStream::BodyEdits,
        StateStream::BodyEditRuns,
        StateStream::BodyRowMoves,
        StateStream::BodyFreshRows,
    } => World::flush_body_commands,
    Step {
        StateStream::ConstraintRowMoves,
        StateStream::ConstraintFreshRows,
    } => World::flush_constraint_commands,
    Step {
        SoftStream::Edits,
        SoftStream::BodyEdits,
    } => World::flush_soft_commands,
    Step {
        StateStream::Params,
        StateStream::RowStreams,
    } => World::flush_step_parameters,
    Step {
        StateStream::ObservedIds,
        StateStream::ObservedJointIds,
    } => World::flush_observations,
    Step {
        SceneStream::QueryRecords,
    } => World::flush_queries,
}

const _: () = assert_handovers(STREAM_WRITERS, HANDOVERS);

impl World {
    /// Hands the device every record a scene mutation moved. A mutation records the change it made
    /// in the stream table it owns, so the sweep only has to hand over what the tables hold.
    pub(crate) fn flush_scene_records(&mut self) {
        for handover in HANDOVERS {
            if handover.stage == Stage::Scene {
                (handover.flush)(self);
            }
        }
    }

    /// Hands the device every record the step composed, once the step has composed all of them.
    pub(crate) fn flush_step_records(&mut self) {
        for handover in HANDOVERS {
            if handover.stage == Stage::Step {
                (handover.flush)(self);
            }
        }
    }
}

const fn assert_handovers(declared: &[&[(&str, StreamWriters)]], handovers: &[Handover]) {
    let mut table = 0;
    while table < declared.len() {
        let streams = declared[table];
        let mut index = 0;
        while index < streams.len() {
            let (label, sides) = streams[index];
            let held = declared_by(handovers, label);
            match sides {
                StreamWriters::Host | StreamWriters::HostThenDevice => assert!(
                    held == 1,
                    "a device stream the host hands records to must have exactly one declared writer, and one that names only streams the host declares",
                ),
                StreamWriters::Device => assert!(
                    held == 0,
                    "a device stream no handover carries must not be claimed by one",
                ),
            }
            index += 1;
        }
        table += 1;
    }
    let mut handover = 0;
    while handover < handovers.len() {
        let streams = handovers[handover].streams;
        let mut index = 0;
        while index < streams.len() {
            assert!(
                declared_once(declared, streams[index]),
                "a declared writer must name a device stream the composition declares once",
            );
            index += 1;
        }
        handover += 1;
    }
}

const fn declared_once(streams: &[&[(&str, StreamWriters)]], label: &str) -> bool {
    let mut declared = 0;
    let mut table = 0;
    while table < streams.len() {
        let streams = streams[table];
        let mut index = 0;
        while index < streams.len() {
            if same(streams[index].0, label) {
                declared += 1;
            }
            index += 1;
        }
        table += 1;
    }
    declared == 1
}

const fn declared_by(handovers: &[Handover], label: &str) -> usize {
    let mut declared = 0;
    let mut handover = 0;
    while handover < handovers.len() {
        let streams = handovers[handover].streams;
        let mut index = 0;
        while index < streams.len() {
            if same(streams[index], label) {
                declared += 1;
            }
            index += 1;
        }
        handover += 1;
    }
    declared
}

const fn same(left: &str, right: &str) -> bool {
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
