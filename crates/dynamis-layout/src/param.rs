use crate::constant::MAX_CELLS_PER_COLLIDER;
use bytemuck::{Pod, Zeroable};
use dynamis_model::{MaterialCombine, PhysicsConfig};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<SimParamsRecord>() == 112);
    assert!(size_of::<DispatchArgs>() == 12);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimParamsRecord {
    pub gravity: [f32; 4],
    pub dt: f32,
    pub damping: f32,
    pub angular_damping: f32,
    pub body_count: u32,
    pub solve_iterations: u32,
    pub position_iterations: u32,
    pub constraint_count: u32,
    pub relaxation: f32,
    pub slop: f32,
    pub restitution_threshold: f32,
    pub max_velocity: f32,
    pub max_angular_velocity: f32,
    pub grid_cell_size: f32,
    pub max_cells_per_collider: u32,
    pub _pad0: u32,
    pub sleep_velocity: f32,
    pub sleep_angular_velocity: f32,
    pub sleep_time: f32,
    pub wake_velocity: f32,
    pub friction_combine: u32,
    pub restitution_combine: u32,
    pub _pad5: f32,
    pub _pad6: f32,
    pub _pad7: f32,
}

impl SimParamsRecord {
    pub fn new(config: &PhysicsConfig, dt: f32, body_count: u32, constraint_count: u32) -> Self {
        Self {
            gravity: [config.gravity[0], config.gravity[1], config.gravity[2], 0.0],
            dt,
            damping: config.damping,
            angular_damping: config.angular_damping,
            body_count,
            solve_iterations: config.solve_iterations,
            position_iterations: config.position_iterations,
            constraint_count,
            relaxation: config.relaxation,
            slop: config.slop,
            restitution_threshold: config.restitution_threshold,
            max_velocity: config.max_velocity,
            max_angular_velocity: config.max_angular_velocity,
            grid_cell_size: config.broadphase_cell_size,
            max_cells_per_collider: MAX_CELLS_PER_COLLIDER,
            _pad0: 0,
            sleep_velocity: config.sleep_velocity,
            sleep_angular_velocity: config.sleep_angular_velocity,
            sleep_time: config.sleep_time,
            wake_velocity: config.wake_velocity,
            friction_combine: combine_code(config.friction_combine),
            restitution_combine: combine_code(config.restitution_combine),
            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DispatchArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl DispatchArgs {
    pub const fn none() -> Self {
        Self { x: 0, y: 1, z: 1 }
    }

    pub const fn sized(elements: u32) -> Self {
        Self {
            x: elements,
            y: 1,
            z: 1,
        }
    }
}
