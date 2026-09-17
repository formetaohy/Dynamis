use super::streams::{BLOCK_LANES, RigidStream};
use crate::RigidFrame;
use crate::integrate::SubstepIntegrate;
use dynamis_abi::Count;
use dynamis_abi::{COUNTER_BLOCKS, COUNTER_CONTACTS, COUNTER_SOLVER_ROWS};
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{CORE, Extent, rows, stream, stream_warm, workgroups};
use dynamis_state::StateStream;

const BLOCKS: &[&str] = &[
    include_str!("../shaders/solver_contact_block.wgsl"),
    include_str!("../shaders/solver_constraint_block.wgsl"),
];

fn block_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::JOINTS.to_vec();
    fragments.extend_from_slice(dynamis_shader::COUNTERS);
    fragments.extend_from_slice(BLOCKS);
    fragments
}
const POSITION_CORRECTION: &[&str] = &[include_str!("../shaders/position_correction.wgsl")];

fn position_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::JOINTS.to_vec();
    fragments.extend_from_slice(POSITION_CORRECTION);
    fragments
}

pub struct SolverPrepare {
    reset: Stage,
    total: Stage,
}

impl PassRuntime<RigidFrame> for SolverPrepare {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            reset: Stage::build(
                context,
                "solver reset",
                rows(
                    context,
                    include_str!("../shaders/solver_reset.wgsl"),
                    CORE,
                    Count::Bodies.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("block_counts", RigidStream::SolverBlockCounts.whole()),
                    ("resolution", RigidStream::SolverResolution.whole()),
                ],
                &[],
            ),
            total: Stage::build(
                context,
                "solver total",
                workgroups(context, include_str!("../shaders/solver_total.wgsl"), CORE),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("segments", RigidStream::SolverSegments.whole()),
                    ("block_count", dynamis_state::counter(COUNTER_BLOCKS)),
                    (
                        "solver_row_count",
                        dynamis_state::counter(COUNTER_SOLVER_ROWS),
                    ),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.reset.record_rows(
            recorder,
            streams,
            Count::Bodies.rows(&frame.params, &frame.rows),
        );
        self.total.record_workgroups(recorder, streams, 1);
    }
}

pub struct SolveSubsteps {
    integrate: SubstepIntegrate,
    blocks: Stage,
    block_solve: Stage,
    block_apply: Stage,
    position_block: Stage,
    position_apply: Stage,
}

impl PassRuntime<RigidFrame> for SolveSubsteps {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        let segments = RigidStream::SolverSegments.whole();
        let block_count = dynamis_state::counter(COUNTER_BLOCKS);
        Self {
            integrate: SubstepIntegrate::build(context, streams),
            blocks: Stage::build(
                context,
                "solver blocks",
                stream(
                    context,
                    include_str!("../shaders/solver_blocks.wgsl"),
                    CORE,
                    "work",
                    Extent::slot(COUNTER_BLOCKS, "block_count", "blocks").lanes(BLOCK_LANES),
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
                    ("block_count", dynamis_state::counter(COUNTER_BLOCKS)),
                    ("solver_rows", RigidStream::SolverRows.whole()),
                    (
                        "solver_row_count",
                        dynamis_state::counter(COUNTER_SOLVER_ROWS),
                    ),
                ],
                &[],
            ),
            block_solve: Stage::build(
                context,
                "solver_block_solve",
                stream_warm(
                    context,
                    include_str!("../shaders/solver_block_solve.wgsl"),
                    &block_fragments(),
                    "work",
                    Extent::slot(COUNTER_BLOCKS, "block_count", "blocks").lanes(BLOCK_LANES),
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
                    ("block_count", block_count),
                    ("target_speeds", RigidStream::ContactTargetSpeeds.whole()),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                    ("solver_rounds", RigidStream::SolverRounds.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            block_apply: Stage::build(
                context,
                "solver block apply",
                stream(
                    context,
                    include_str!("../shaders/solver_block_apply.wgsl"),
                    CORE,
                    "work",
                    Extent::slot(COUNTER_SOLVER_ROWS, "solver_row_count", "solver_rows"),
                ),
                streams,
                &[
                    ("body_states", StateStream::BodyStates.whole()),
                    ("velocity_deltas", RigidStream::SolverVelocityDeltas.whole()),
                    ("solver_rows", RigidStream::SolverRows.whole()),
                    (
                        "solver_row_count",
                        dynamis_state::counter(COUNTER_SOLVER_ROWS),
                    ),
                    ("solver_rounds", RigidStream::SolverRounds.whole()),
                ],
                &[],
            ),
            position_block: Stage::build(
                context,
                "position block",
                stream(
                    context,
                    include_str!("../shaders/position_block.wgsl"),
                    &position_fragments(),
                    "work",
                    Extent::slot(COUNTER_BLOCKS, "block_count", "blocks").lanes(BLOCK_LANES),
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
            ),
            position_apply: Stage::build(
                context,
                "position apply",
                stream(
                    context,
                    include_str!("../shaders/position_apply.wgsl"),
                    CORE,
                    "work",
                    Extent::slot(COUNTER_SOLVER_ROWS, "solver_row_count", "solver_rows"),
                ),
                streams,
                &[
                    ("body_states", StateStream::BodyStates.whole()),
                    ("position_deltas", RigidStream::SolverPositionDeltas.whole()),
                    ("contributions", RigidStream::SolverContributions.whole()),
                    ("resolution", RigidStream::SolverResolution.whole()),
                    ("solver_rows", RigidStream::SolverRows.whole()),
                    (
                        "solver_row_count",
                        dynamis_state::counter(COUNTER_SOLVER_ROWS),
                    ),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.blocks.record_stream(recorder, streams);
        for substep in 0..frame.params.substeps {
            self.integrate.record(recorder, streams);
            if substep == 0 {
                self.block_solve.record_warm(recorder, streams);
                self.block_apply.record_stream(recorder, streams);
            }
            for _ in 0..frame.params.solve_iterations {
                self.block_solve.record_stream(recorder, streams);
                self.block_apply.record_stream(recorder, streams);
            }
            self.integrate.record_advance(recorder, streams);
            for _ in 0..frame.params.position_iterations {
                self.position_block.record_stream(recorder, streams);
                self.position_apply.record_stream(recorder, streams);
            }
        }
    }
}
