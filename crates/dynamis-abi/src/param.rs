use crate::{RowStreamsRecord, StepParamsRecord};
use dynamis_model::{MaterialCombine, PhysicsConfig};

impl StepParamsRecord {
    pub fn new(
        config: &PhysicsConfig,
        dt: f32,
        counts: FrameCounts,
        subscriptions: Subscriptions,
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
            vehicles,
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
            observed_count: subscriptions.observed,
            particle_count: particles,
            element_count: elements,
            soft_substep_dt: dt / config.soft_substeps as f32,
            soft_body_count: soft_bodies,
            attachment_count: attachments,
            settle_velocity: config.settle_velocity,
            observed_joint_count: subscriptions.observed_joints,
            character_count: characters,
            vehicle_count: vehicles,
            _wgsl_pad0: [0; 8],
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
    pub vehicles: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Subscriptions {
    pub observed: u32,
    pub observed_joints: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowStreams {
    pub body_edit_runs: u32,
    pub body_moves: u32,
    pub constraint_moves: u32,
    pub soft_edits: u32,
    pub soft_body_edits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Params(&'static str),
    Rows(&'static str),
}

impl Bound {
    pub fn expression(self) -> String {
        match self {
            Self::Params(field) => format!("params.{field}"),
            Self::Rows(field) => format!("row_streams.{field}"),
        }
    }
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
    Vehicles,
}

impl Count {
    pub const fn bound(self) -> Bound {
        match self {
            Self::BodyEditRuns => Bound::Rows("body_edit_runs"),
            Self::BodyMoves => Bound::Rows("body_moves"),
            Self::ConstraintMoves => Bound::Rows("constraint_moves"),
            Self::SoftEdits => Bound::Rows("soft_edits"),
            Self::SoftBodyEdits => Bound::Rows("soft_body_edits"),
            Self::Bodies => Bound::Params("body_count"),
            Self::Dynamic => Bound::Params("dynamic_count"),
            Self::Colliders => Bound::Params("collider_count"),
            Self::Constraints => Bound::Params("constraint_count"),
            Self::Particles => Bound::Params("particle_count"),
            Self::Elements => Bound::Params("element_count"),
            Self::Attachments => Bound::Params("attachment_count"),
            Self::SoftBodies => Bound::Params("soft_body_count"),
            Self::Observed => Bound::Params("observed_count"),
            Self::ObservedJoints => Bound::Params("observed_joint_count"),
            Self::Characters => Bound::Params("character_count"),
            Self::Vehicles => Bound::Params("vehicle_count"),
        }
    }

    pub const fn rows(self, params: &StepParamsRecord, rows: &RowStreams) -> u32 {
        match self {
            Self::BodyEditRuns => rows.body_edit_runs,
            Self::BodyMoves => rows.body_moves,
            Self::ConstraintMoves => rows.constraint_moves,
            Self::SoftEdits => rows.soft_edits,
            Self::SoftBodyEdits => rows.soft_body_edits,
            Self::Bodies => params.body_count,
            Self::Dynamic => params.dynamic_count,
            Self::Colliders => params.collider_count,
            Self::Constraints => params.constraint_count,
            Self::Particles => params.particle_count,
            Self::Elements => params.element_count,
            Self::Attachments => params.attachment_count,
            Self::SoftBodies => params.soft_body_count,
            Self::Observed => params.observed_count,
            Self::ObservedJoints => params.observed_joint_count,
            Self::Characters => params.character_count,
            Self::Vehicles => params.vehicle_count,
        }
    }
}

impl From<RowStreams> for RowStreamsRecord {
    fn from(rows: RowStreams) -> Self {
        Self {
            body_edit_runs: rows.body_edit_runs,
            body_moves: rows.body_moves,
            constraint_moves: rows.constraint_moves,
            soft_edits: rows.soft_edits,
            soft_body_edits: rows.soft_body_edits,
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
