use crate::StepParamsRecord;
use dynamis_model::{MaterialCombine, PhysicsConfig};

impl StepParamsRecord {
    pub fn new(
        config: &PhysicsConfig,
        dt: f32,
        counts: FrameCounts,
        streams: RowStreams,
        event_slot: u32,
    ) -> Self {
        let FrameCounts {
            dynamic_bodies,
            bodies,
            body_ids: _,
            colliders,
            constraints,
            particles,
            elements,
            soft_bodies,
            attachments,
            characters,
        } = counts;
        Self {
            gravity: [config.gravity[0], config.gravity[1], config.gravity[2], 0.0],
            dt,
            substep_dt: dt / config.substeps as f32,
            damping: config.damping,
            angular_damping: config.angular_damping,
            body_count: bodies,
            substeps: config.substeps,
            solve_iterations: config.solve_iterations,
            position_iterations: config.position_iterations,
            soft_iterations: config.soft_iterations,
            soft_substeps: config.soft_substeps,
            constraint_count: constraints,
            relaxation: config.relaxation,
            slop: config.slop,
            contact_margin: config.contact_margin,
            restitution_threshold: config.restitution_threshold,
            max_velocity: config.max_velocity,
            max_angular_velocity: config.max_angular_velocity,
            dynamic_count: dynamic_bodies,
            collider_count: colliders,
            sleep_velocity: config.sleep_velocity,
            sleep_angular_velocity: config.sleep_angular_velocity,
            sleep_time: config.sleep_time,
            friction_combine: combine_code(config.friction_combine),
            restitution_combine: combine_code(config.restitution_combine),
            body_edit_run_count: streams.body_edit_runs,
            body_move_count: streams.body_moves,
            constraint_move_count: streams.constraint_moves,
            observed_count: streams.observed,
            event_slot,
            particle_count: particles,
            element_count: elements,
            soft_substep_dt: dt / config.soft_substeps as f32,
            soft_body_count: soft_bodies,
            attachment_count: attachments,
            settle_velocity: config.settle_velocity,
            soft_edit_count: streams.soft_edits,
            soft_body_edit_count: streams.soft_body_edits,
            observed_joint_count: streams.observed_joints,
            character_count: characters,
            _wgsl_pad0: [0; 4],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameCounts {
    pub dynamic_bodies: u32,
    pub bodies: u32,
    pub body_ids: u32,
    pub colliders: u32,
    pub constraints: u32,
    pub particles: u32,
    pub elements: u32,
    pub attachments: u32,
    pub soft_bodies: u32,
    pub characters: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowStreams {
    pub body_edit_runs: u32,
    pub soft_edits: u32,
    pub soft_body_edits: u32,
    pub body_moves: u32,
    pub constraint_moves: u32,
    pub observed: u32,
    pub observed_joints: u32,
}

#[derive(Clone, Copy)]
pub enum Count {
    Bodies,
    Dynamic,
    Colliders,
    Constraints,
    Particles,
    Elements,
    Attachments,
    SoftBodies,
    BodyEditRuns,
    SoftEdits,
    SoftBodyEdits,
    BodyMoves,
    ConstraintMoves,
    Observed,
    ObservedJoints,
    Characters,
}

impl Count {
    pub const fn field(self) -> &'static str {
        match self {
            Self::Bodies => "body_count",
            Self::Dynamic => "dynamic_count",
            Self::Colliders => "collider_count",
            Self::Constraints => "constraint_count",
            Self::Particles => "particle_count",
            Self::Elements => "element_count",
            Self::Attachments => "attachment_count",
            Self::SoftBodies => "soft_body_count",
            Self::BodyEditRuns => "body_edit_run_count",
            Self::SoftEdits => "soft_edit_count",
            Self::SoftBodyEdits => "soft_body_edit_count",
            Self::BodyMoves => "body_move_count",
            Self::ConstraintMoves => "constraint_move_count",
            Self::Observed => "observed_count",
            Self::ObservedJoints => "observed_joint_count",
            Self::Characters => "character_count",
        }
    }

    pub const fn rows(self, params: &StepParamsRecord) -> u32 {
        match self {
            Self::Bodies => params.body_count,
            Self::Dynamic => params.dynamic_count,
            Self::Colliders => params.collider_count,
            Self::Constraints => params.constraint_count,
            Self::Particles => params.particle_count,
            Self::Elements => params.element_count,
            Self::Attachments => params.attachment_count,
            Self::SoftBodies => params.soft_body_count,
            Self::BodyEditRuns => params.body_edit_run_count,
            Self::SoftEdits => params.soft_edit_count,
            Self::SoftBodyEdits => params.soft_body_edit_count,
            Self::BodyMoves => params.body_move_count,
            Self::ConstraintMoves => params.constraint_move_count,
            Self::Observed => params.observed_count,
            Self::ObservedJoints => params.observed_joint_count,
            Self::Characters => params.character_count,
        }
    }
}

fn combine_code(combine: MaterialCombine) -> u32 {
    match combine {
        MaterialCombine::Multiply => 0,
        MaterialCombine::Min => 1,
        MaterialCombine::Max => 2,
        MaterialCombine::Average => 3,
    }
}
