use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{COUNTERS, GRID_INDEX, rows, workgroups};
use dynamis_state::StateStream;

const COLLIDER_EMIT: &str = include_str!("../shaders/collider_emit.wgsl");

fn collider_fragments() -> Vec<&'static str> {
    let mut fragments = GRID_INDEX.to_vec();
    fragments.push(COLLIDER_EMIT);
    fragments
}

pub struct EmitEntries {
    reset: Stage,
    immovable: Stage,
    gate: Stage,
    resting: Stage,
    commit: Stage,
    moving: Stage,
}

impl PassRuntime<RigidFrame> for EmitEntries {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        let bindings = [
            ("params", StateStream::Params.whole()),
            ("aabbs", RigidStream::ColliderAabbs.whole()),
            ("collider_owners", StateStream::ColliderOwners.whole()),
            ("body_activity", RigidStream::BodyActivity.whole()),
            ("wake_flags", StateStream::WakeFlags.whole()),
            ("body_descs", StateStream::BodyDescriptors.whole()),
            ("entry_keys", BroadphaseStream::EntryKeys.whole()),
            ("entry_order", BroadphaseStream::EntryOrder.whole()),
            ("entries", BroadphaseStream::Entries.whole()),
            ("counters", StateStream::Counters.whole()),
        ];
        let mut flagged = bindings.to_vec();
        flagged.push(("body_admitted", BroadphaseStream::BodyAdmitted.whole()));
        let fragments = collider_fragments();
        Self {
            reset: Stage::build(
                context,
                "immovable_reset",
                workgroups(
                    context,
                    include_str!("../shaders/immovable_reset.wgsl"),
                    COUNTERS,
                ),
                streams,
                &[
                    ("counters", StateStream::Counters.whole()),
                    ("entry_base", StateStream::EntryBase.whole()),
                ],
                &[],
            ),
            immovable: Stage::build(
                context,
                "immovable_entries",
                rows(
                    context,
                    include_str!("../shaders/immovable_entries.wgsl"),
                    &fragments,
                    Count::Colliders.bound(),
                ),
                streams,
                &bindings,
                &[],
            ),
            gate: Stage::build(
                context,
                "grid resting gate",
                workgroups(
                    context,
                    include_str!("../shaders/grid_resting_gate.wgsl"),
                    COUNTERS,
                ),
                streams,
                &[("counters", StateStream::Counters.whole())],
                &[],
            ),
            resting: Stage::build(
                context,
                "grid resting entries",
                rows(
                    context,
                    include_str!("../shaders/grid_resting_entries.wgsl"),
                    &fragments,
                    Count::Colliders.bound(),
                ),
                streams,
                &flagged,
                &[],
            ),
            commit: Stage::build(
                context,
                "grid resting commit",
                workgroups(
                    context,
                    include_str!("../shaders/grid_resting_commit.wgsl"),
                    COUNTERS,
                ),
                streams,
                &[("counters", StateStream::Counters.whole())],
                &[],
            ),
            moving: Stage::build(
                context,
                "collider_entries",
                rows(
                    context,
                    include_str!("../shaders/collider_entries.wgsl"),
                    &fragments,
                    Count::Colliders.bound(),
                ),
                streams,
                &flagged,
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        let colliders = Count::Colliders.rows(&frame.params, &frame.rows);
        if frame.immovable_rebuild {
            self.reset.record_workgroups(recorder, streams, 1);
            self.immovable.record_rows(recorder, streams, colliders);
        }
        if frame.resting_rebuild {
            self.gate.record_workgroups(recorder, streams, 1);
            self.resting.record_rows(recorder, streams, colliders);
            self.commit.record_workgroups(recorder, streams, 1);
        }
        self.moving.record_rows(recorder, streams, colliders);
    }
}
