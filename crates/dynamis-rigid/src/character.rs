use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::{CHARACTER_SWEEPS, Count};
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, GpuContext, Resources};
use dynamis_pass::Stage;
use dynamis_shader::{GEOMETRY_INDEX, rows, workgroups};
use dynamis_state::StateStream;

pub struct Characters {
    step: Stage,
    sweeps: Stage,
}

impl Characters {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            step: Stage::build(
                context,
                "character",
                rows(
                    context,
                    include_str!("../shaders/character.wgsl"),
                    &[],
                    Count::Characters.bound(),
                ),
                streams,
                &[
                    ("characters", RigidStream::Characters.whole()),
                    ("character_inputs", RigidStream::CharacterInputs.whole()),
                    ("character_states", RigidStream::CharacterStates.whole()),
                    ("character_sweeps", RigidStream::CharacterSweeps.whole()),
                    ("character_hits", RigidStream::CharacterHits.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("params", StateStream::Params.whole()),
                ],
                &[],
            ),
            sweeps: Stage::build(
                context,
                "character sweeps",
                workgroups(
                    context,
                    include_str!("../shaders/queries.wgsl"),
                    GEOMETRY_INDEX,
                ),
                streams,
                &[
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                    ("queries", RigidStream::CharacterSweeps.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("query_results", RigidStream::CharacterHits.whole()),
                    ("params", StateStream::Params.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
        }
    }

    pub fn record_step(
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.step.record_rows(
            recorder,
            streams,
            Count::Characters.rows(&frame.params, &frame.rows),
        );
    }

    pub fn record_sweeps(
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        let sweeps = Count::Characters
            .rows(&frame.params, &frame.rows)
            .saturating_mul(CHARACTER_SWEEPS);
        self.sweeps.record_workgroups(recorder, streams, sweeps);
    }
}
