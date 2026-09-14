use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_abi::{COUNTER_BLOCKS, COUNTER_CONTACTS, COUNTER_LIVE};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernel::{CORE, rows, stream, stream_warm, workgroups};
use dynamis_pass::Resources;
use dynamis_pass::Stage;
use dynamis_state::StateStream;

const BLOCKS: &[&str] = &[
    include_str!("../shaders/solver_contact_block.wgsl"),
    include_str!("../shaders/solver_constraint_block.wgsl"),
];

fn block_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_kernel::JOINTS.to_vec();
    fragments.extend_from_slice(BLOCKS);
    fragments
}
const POSITION_CORRECTION: &[&str] = &[include_str!("../shaders/position_correction.wgsl")];

fn position_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_kernel::JOINTS.to_vec();
    fragments.extend_from_slice(POSITION_CORRECTION);
    fragments
}

pub struct Solver {
    reset: Stage,
    total: Stage,
    blocks: Stage,
    block_solve: Stage,
    block_apply: Stage,
    position_block: Stage,
    position_apply: Stage,
}

impl Solver {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        let segments = RigidStream::SolverSegments.whole();
        let block_count = dynamis_state::counter(COUNTER_BLOCKS);
        let reset = Stage::build(
            context,
            "solver reset",
            rows(
                context,
                include_str!("../shaders/solver_reset.wgsl"),
                CORE,
                Count::Bodies.field(),
            ),
            streams,
            &[
                ("params", StateStream::Params.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("resolution", RigidStream::SolverResolution.whole()),
                ("contributions", RigidStream::SolverContributions.whole()),
                ("velocity_deltas", RigidStream::SolverVelocityDeltas.whole()),
                ("position_deltas", RigidStream::SolverPositionDeltas.whole()),
            ],
            &[],
        );
        let total = Stage::build(
            context,
            "solver total",
            workgroups(context, include_str!("../shaders/solver_total.wgsl"), CORE),
            streams,
            &[
                ("params", StateStream::Params.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                ("segments", segments),
                ("block_count", block_count),
            ],
            &[],
        );
        let blocks = Stage::build(
            context,
            "solver blocks",
            stream(
                context,
                include_str!("../shaders/solver_blocks.wgsl"),
                CORE,
                "work",
                RigidStream::SolverBlocks,
            ),
            streams,
            &[
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
                ("body_descs", StateStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                ("segments", segments),
                ("collider_owners", StateStream::ColliderOwners.whole()),
                ("constraint_rows", RigidStream::ConstraintRows.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("target_speeds", RigidStream::ContactTargetSpeeds.whole()),
                ("blocks", RigidStream::SolverBlocks.whole()),
            ],
            &[],
        );
        let block_solve = Stage::build(
            context,
            "solver_block_solve",
            stream_warm(
                context,
                include_str!("../shaders/solver_block_solve.wgsl"),
                &block_fragments(),
                "work",
                RigidStream::SolverBlocks,
            ),
            streams,
            &[
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
                ("body_descs", StateStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                (
                    "constraint_descs",
                    StateStream::ConstraintDescriptors.whole(),
                ),
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                ("segments", segments),
                ("blocks", RigidStream::SolverBlocks.whole()),
                ("velocity_deltas", RigidStream::SolverVelocityDeltas.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("block_count", dynamis_state::counter(COUNTER_BLOCKS)),
                ("target_speeds", RigidStream::ContactTargetSpeeds.whole()),
                ("constraint_rows", RigidStream::ConstraintRows.whole()),
            ],
            &[],
        );
        let block_apply = Stage::build(
            context,
            "solver block apply",
            stream(
                context,
                include_str!("../shaders/solver_block_apply.wgsl"),
                CORE,
                "work",
                RigidStream::LiveBodies,
            ),
            streams,
            &[
                ("body_states", StateStream::BodyStates.whole()),
                ("velocity_deltas", RigidStream::SolverVelocityDeltas.whole()),
                ("live_bodies", RigidStream::LiveBodies.whole()),
                ("live_count", dynamis_state::counter(COUNTER_LIVE)),
            ],
            &[],
        );
        let position_block = Stage::build(
            context,
            "position block",
            stream(
                context,
                include_str!("../shaders/position_block.wgsl"),
                &position_fragments(),
                "work",
                RigidStream::SolverBlocks,
            ),
            streams,
            &[
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
                ("body_descs", StateStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                (
                    "constraint_descs",
                    StateStream::ConstraintDescriptors.whole(),
                ),
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                ("segments", segments),
                ("blocks", RigidStream::SolverBlocks.whole()),
                ("position_deltas", RigidStream::SolverPositionDeltas.whole()),
                ("resolution", RigidStream::SolverResolution.whole()),
                ("contributions", RigidStream::SolverContributions.whole()),
                ("block_count", block_count),
                ("constraint_rows", RigidStream::ConstraintRows.whole()),
            ],
            &[],
        );
        let position_apply = Stage::build(
            context,
            "position apply",
            stream(
                context,
                include_str!("../shaders/position_apply.wgsl"),
                CORE,
                "work",
                RigidStream::LiveBodies,
            ),
            streams,
            &[
                ("body_states", StateStream::BodyStates.whole()),
                ("position_deltas", RigidStream::SolverPositionDeltas.whole()),
                ("contributions", RigidStream::SolverContributions.whole()),
                ("resolution", RigidStream::SolverResolution.whole()),
                ("live_bodies", RigidStream::LiveBodies.whole()),
                ("live_count", dynamis_state::counter(COUNTER_LIVE)),
            ],
            &[],
        );
        Self {
            reset,
            total,
            blocks,
            block_solve,
            block_apply,
            position_block,
            position_apply,
        }
    }

    pub fn record_prepare(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.reset
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
        self.total.record_workgroups(recorder, streams, 1);
    }

    pub fn record_topology(&self, recorder: &mut ComputeRecorder, streams: &impl Resources) {
        self.blocks.record_stream(recorder, streams);
    }

    pub fn record_warm(&self, recorder: &mut ComputeRecorder, streams: &impl Resources) {
        self.block_solve.record_warm(recorder, streams);
        self.block_apply.record_stream(recorder, streams);
    }

    pub fn record_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        for _ in 0..frame.params.solve_iterations {
            self.block_solve.record_stream(recorder, streams);
            self.block_apply.record_stream(recorder, streams);
        }
    }

    pub fn record_position_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        for _ in 0..frame.params.position_iterations {
            self.position_block.record_stream(recorder, streams);
            self.position_apply.record_stream(recorder, streams);
        }
    }
}
