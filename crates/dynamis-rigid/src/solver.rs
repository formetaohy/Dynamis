use super::streams::{BLOCK_LANES, RigidStream};
use crate::RigidFrame;
use crate::integrate::SubstepIntegrate;
use dynamis_abi::Count;
use dynamis_abi::{COUNTER_BLOCKS, COUNTER_CONTACTS, COUNTER_SOLVER_ROWS};
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{CORE, Extent, rows, stream, stream_warm, workgroups, workgroups_warm};
use dynamis_state::StateStream;

const CONTACT_BLOCK: &[&str] = &[include_str!("../shaders/solver_contact_block.wgsl")];
const JOINT_SOLVE: &[&str] = &[include_str!("../shaders/joint_solve.wgsl")];
const CORRECTION: &[&str] = &[include_str!("../shaders/correction.wgsl")];
const CONTACT_CORRECTION: &[&str] = &[include_str!("../shaders/position_correction.wgsl")];
const JOINT_CORRECTION: &[&str] = &[include_str!("../shaders/joint_correction.wgsl")];

fn block_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::COUNTERS.to_vec();
    fragments.extend_from_slice(CONTACT_BLOCK);
    fragments
}

fn joint_velocity_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::JOINTS.to_vec();
    fragments.extend_from_slice(dynamis_shader::COUNTERS);
    fragments.extend_from_slice(JOINT_SOLVE);
    fragments
}

fn contact_position_fragments() -> Vec<&'static str> {
    let mut fragments = CORRECTION.to_vec();
    fragments.extend_from_slice(CONTACT_CORRECTION);
    fragments
}

fn joint_position_fragments() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::JOINTS.to_vec();
    fragments.extend_from_slice(CORRECTION);
    fragments.extend_from_slice(JOINT_CORRECTION);
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
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
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
    joint_velocity: Stage,
    block_solve: Stage,
    block_apply: Stage,
    joint_position: Stage,
    position_block: Stage,
    position_apply: Stage,
}

impl PassRuntime<RigidFrame> for SolveSubsteps {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
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
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                    ("block_counts", RigidStream::SolverBlockCounts.whole()),
                    ("target_speeds", RigidStream::ContactTargetSpeeds.whole()),
                    ("blocks", RigidStream::SolverBlocks.whole()),
                    ("solver_rows", RigidStream::SolverRows.whole()),
                    (
                        "solver_row_count",
                        dynamis_state::counter(COUNTER_SOLVER_ROWS),
                    ),
                    ("block_count", block_count),
                ],
                &[],
            ),
            joint_velocity: Stage::build(
                context,
                "joint velocity",
                workgroups_warm(
                    context,
                    include_str!("../shaders/joint_velocity.wgsl"),
                    &joint_velocity_fragments(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    (
                        "constraint_descs",
                        StateStream::ConstraintDescriptors.whole(),
                    ),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                    ("joint_rows", RigidStream::JointRows.whole()),
                    ("joint_layers", RigidStream::JointLayers.whole()),
                    ("joint_islands", RigidStream::JointIslands.whole()),
                    ("counters", StateStream::Counters.whole()),
                    ("solver_rounds", RigidStream::SolverRounds.whole()),
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
                    ("blocks", RigidStream::SolverBlocks.whole()),
                    ("velocity_deltas", RigidStream::SolverVelocityDeltas.whole()),
                    ("block_counts", RigidStream::SolverBlockCounts.whole()),
                    ("block_count", block_count),
                    ("target_speeds", RigidStream::ContactTargetSpeeds.whole()),
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
            joint_position: Stage::build(
                context,
                "joint position",
                workgroups(
                    context,
                    include_str!("../shaders/joint_position.wgsl"),
                    &joint_position_fragments(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    (
                        "constraint_descs",
                        StateStream::ConstraintDescriptors.whole(),
                    ),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                    ("joint_rows", RigidStream::JointRows.whole()),
                    ("joint_layers", RigidStream::JointLayers.whole()),
                    ("joint_islands", RigidStream::JointIslands.whole()),
                    ("resolution", RigidStream::SolverResolution.whole()),
                ],
                &[],
            ),
            position_block: Stage::build(
                context,
                "position block",
                stream(
                    context,
                    include_str!("../shaders/position_block.wgsl"),
                    &contact_position_fragments(),
                    "work",
                    Extent::slot(COUNTER_BLOCKS, "block_count", "blocks").lanes(BLOCK_LANES),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("blocks", RigidStream::SolverBlocks.whole()),
                    ("position_deltas", RigidStream::SolverPositionDeltas.whole()),
                    ("resolution", RigidStream::SolverResolution.whole()),
                    ("contributions", RigidStream::SolverContributions.whole()),
                    ("block_count", block_count),
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
                self.joint_velocity
                    .record_workgroups_warm(recorder, streams, frame.joint_islands);
                self.block_solve.record_warm(recorder, streams);
                self.block_apply.record_stream(recorder, streams);
            }
            for _ in 0..frame.params.solve_iterations {
                self.joint_velocity
                    .record_workgroups(recorder, streams, frame.joint_islands);
                self.block_solve.record_stream(recorder, streams);
                self.block_apply.record_stream(recorder, streams);
            }
            self.integrate.record_advance(recorder, streams);
            for _ in 0..frame.params.position_iterations {
                self.joint_position
                    .record_workgroups(recorder, streams, frame.joint_islands);
                self.position_block.record_stream(recorder, streams);
                self.position_apply.record_stream(recorder, streams);
            }
        }
    }
}
