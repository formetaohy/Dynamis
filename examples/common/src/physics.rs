use dynamis::{BodyHandle, GpuContext, PhysicsConfig, Simulation, StreamBudget};
use dynamis_example_render::{AppContext, MeshId, Quat, Vec3};

const PHYSICS_STEP: f32 = 1.0 / 60.0;

pub struct Simulator {
    pub simulation: Simulation,
    accumulator: f32,
}

impl Simulator {
    pub fn with_capacity(capacity: usize) -> Self {
        let gpu = pollster::block_on(GpuContext::new());
        Self {
            simulation: Simulation::new(
                gpu,
                capacity,
                PhysicsConfig::default(),
                StreamBudget::default(),
            ),
            accumulator: 0.0,
        }
    }
}

pub fn advance_physics(ctx: &AppContext, simulator: &mut Simulator) {
    simulator.accumulator = (simulator.accumulator + ctx.time.delta_secs()).min(PHYSICS_STEP * 8.0);
    while simulator.accumulator >= PHYSICS_STEP {
        simulator.simulation.step(PHYSICS_STEP);
        simulator.accumulator -= PHYSICS_STEP;
    }
}

pub fn sync_visuals(ctx: &mut AppContext, simulator: &Simulator, bodies: &[(BodyHandle, MeshId)]) {
    for (body, mesh) in bodies {
        let state = simulator.simulation.read_state(*body);
        let transform = ctx.mesh_transform(*mesh);
        transform.translation = Vec3::from(state.position);
        transform.rotation = Quat::from_xyzw(
            state.orientation[0],
            state.orientation[1],
            state.orientation[2],
            state.orientation[3],
        );
        transform.scale = Vec3::ONE;
    }
}
