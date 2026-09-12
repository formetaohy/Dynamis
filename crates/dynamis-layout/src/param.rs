use crate::StepParamsRecord;
use crate::constant::MAX_CELLS_PER_COLLIDER;
use dynamis_model::{MaterialCombine, PhysicsConfig};

impl StepParamsRecord {
    pub fn new(
        config: &PhysicsConfig,
        dt: f32,
        dynamic_count: u32,
        body_count: u32,
        constraint_count: u32,
        streams: RowStreams,
        event_slot: u32,
    ) -> Self {
        Self {
            gravity: [config.gravity[0], config.gravity[1], config.gravity[2], 0.0],
            dt,
            damping: config.damping,
            angular_damping: config.angular_damping,
            body_count,
            solve_iterations: config.solve_iterations,
            constraint_count,
            relaxation: config.relaxation,
            slop: config.slop,
            restitution_threshold: config.restitution_threshold,
            max_velocity: config.max_velocity,
            max_angular_velocity: config.max_angular_velocity,
            grid_cell_size: config.broadphase_cell_size,
            max_cells_per_collider: MAX_CELLS_PER_COLLIDER,
            dynamic_count,
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
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
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
