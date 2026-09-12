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
            colliders,
            constraints,
        } = counts;
        Self {
            gravity: [config.gravity[0], config.gravity[1], config.gravity[2], 0.0],
            dt,
            damping: config.damping,
            angular_damping: config.angular_damping,
            body_count: bodies,
            solve_iterations: config.solve_iterations,
            position_iterations: config.position_iterations,
            soft_iterations: config.soft_iterations,
            soft_compliance: config.soft_compliance,
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
            wake_velocity: config.wake_velocity,
            friction_combine: combine_code(config.friction_combine),
            restitution_combine: combine_code(config.restitution_combine),
            edit_run_count: streams.edit_runs,
            body_move_count: streams.body_moves,
            constraint_move_count: streams.constraint_moves,
            event_slot,
            _pad3: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameCounts {
    pub dynamic_bodies: u32,
    pub bodies: u32,
    pub colliders: u32,
    pub constraints: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowStreams {
    pub edit_runs: u32,
    pub body_moves: u32,
    pub constraint_moves: u32,
}

fn combine_code(combine: MaterialCombine) -> u32 {
    match combine {
        MaterialCombine::Multiply => 0,
        MaterialCombine::Min => 1,
        MaterialCombine::Max => 2,
        MaterialCombine::Average => 3,
    }
}
