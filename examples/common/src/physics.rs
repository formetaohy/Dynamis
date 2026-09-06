use bevy::prelude::*;
use dynamis::{GpuContext, PhysicsConfig, Simulation};

const PHYSICS_STEP: f32 = 1.0 / 60.0;

#[derive(Resource)]
pub struct Dynamics {
    pub simulation: Simulation,
    accumulator: f32,
}

impl Dynamics {
    pub fn with_capacity(capacity: usize) -> Self {
        let gpu = pollster::block_on(GpuContext::new());
        Self {
            simulation: Simulation::new(gpu, capacity, PhysicsConfig::default()),
            accumulator: 0.0,
        }
    }
}

#[derive(Component)]
pub struct PhysicsBody(pub dynamis::BodyHandle);

pub fn advance_physics(time: Res<Time>, mut dynamics: ResMut<Dynamics>) {
    dynamics.accumulator = (dynamics.accumulator + time.delta_secs()).min(PHYSICS_STEP * 8.0);
    while dynamics.accumulator >= PHYSICS_STEP {
        dynamics.simulation.step(PHYSICS_STEP);
        dynamics.accumulator -= PHYSICS_STEP;
    }
}

pub fn sync_visuals(mut bodies: Query<(&PhysicsBody, &mut Transform)>, dynamics: Res<Dynamics>) {
    for (body, mut transform) in &mut bodies {
        let state = dynamics.simulation.read_state(body.0);
        transform.translation = Vec3::from(state.position);
        transform.rotation = Quat::from_xyzw(
            state.orientation[0],
            state.orientation[1],
            state.orientation[2],
            state.orientation[3],
        );
    }
}
