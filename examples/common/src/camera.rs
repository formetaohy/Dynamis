use dynamis_example_render::{AppContext, Vec3};

pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Orbit {
    pub const fn new(yaw: f32, pitch: f32, distance: f32, target: Vec3) -> Self {
        Self {
            yaw,
            pitch,
            distance,
            target,
        }
    }
}

pub fn orbit_camera(ctx: &mut AppContext, orbit: &mut Orbit) {
    if ctx.input.right_down {
        orbit.yaw -= ctx.input.cursor_delta.0 * 0.008;
        orbit.pitch = (orbit.pitch - ctx.input.cursor_delta.1 * 0.008).clamp(-1.45, 1.45);
    }
    if ctx.input.wheel != 0.0 {
        orbit.distance = (orbit.distance * (1.0 - ctx.input.wheel * 0.04)).clamp(4.0, 120.0);
    }
    let horizontal = orbit.yaw.cos();
    let offset = Vec3::new(
        horizontal * orbit.pitch.cos(),
        orbit.pitch.sin(),
        horizontal * orbit.pitch.sin(),
    );
    ctx.camera.eye = orbit.target + offset * orbit.distance;
    ctx.camera.target = orbit.target;
}
