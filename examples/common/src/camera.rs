use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

#[derive(Component)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

pub fn orbit_camera(
    mut cameras: Query<(&mut Transform, &mut Orbit)>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
) {
    let mut yaw = 0.0;
    let mut pitch = 0.0;
    let mut zoom = 0.0;
    for event in motion.read() {
        if buttons.pressed(MouseButton::Right) {
            yaw -= event.delta.x * 0.008;
            pitch -= event.delta.y * 0.008;
        }
    }
    for event in wheel.read() {
        zoom += event.y;
    }
    for (mut transform, mut orbit) in &mut cameras {
        orbit.yaw += yaw;
        orbit.pitch = (orbit.pitch + pitch).clamp(-1.45, 1.45);
        orbit.distance = (orbit.distance * (1.0 - zoom * 0.04)).clamp(4.0, 120.0);
        let horizontal = orbit.yaw.cos();
        let offset = Vec3::new(
            horizontal * orbit.pitch.cos(),
            orbit.pitch.sin(),
            horizontal * orbit.pitch.sin(),
        );
        transform.translation = orbit.target + offset * orbit.distance;
        transform.look_at(orbit.target, Vec3::Y);
    }
}
