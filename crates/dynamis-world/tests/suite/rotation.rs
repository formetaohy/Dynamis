use super::common::{DT, new_world, static_config};
use dynamis_math::{dot, length, normalize, quat_conjugate, quat_rotate};
use dynamis_model::{BodyDesc, BodyHandle, BodyState, ColliderDesc, MassProperties, Shape};
use dynamis_world::World;
use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

const HALF_EXTENTS: [f32; 3] = [0.5, 1.0, 2.0];
const NON_PRINCIPAL_SPIN: [f32; 3] = [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0];
const TILTED_AXIS_SPIN: [f32; 3] = [0.05, 1.0, 0.0];
const MEASURE_EVERY: usize = 30;
const FRAMES: usize = 600;
const MOMENTUM_DRIFT: f32 = 1e-3;
const MOMENTUM_TURN: f32 = 3e-2;
const ENERGY_DRIFT: f32 = 1e-3;

fn mass_properties() -> MassProperties {
    dynamis_model::mass_properties_of_intent(
        &[ColliderDesc::new(Shape::cuboid(HALF_EXTENTS))],
        1.0,
        None,
        None,
        |_| None,
    )
}

fn spin_body(world: &mut World, spin: [f32; 3]) -> BodyHandle {
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid(HALF_EXTENTS)))
            .position([0.0; 3])
            .angular_velocity(spin),
    )
}

fn body_momentum(state: &BodyState, mass: &MassProperties) -> [f32; 3] {
    quat_rotate(state.orientation, symmetric(mass.inertia, body_spin(state)))
}

fn kinetic_energy(state: &BodyState, mass: &MassProperties) -> f32 {
    0.5 * dot(body_spin(state), symmetric(mass.inertia, body_spin(state)))
}

fn body_spin(state: &BodyState) -> [f32; 3] {
    quat_rotate(quat_conjugate(state.orientation), state.angular_velocity)
}

fn symmetric(tensor: [f32; 6], v: [f32; 3]) -> [f32; 3] {
    [
        tensor[0] * v[0] + tensor[1] * v[1] + tensor[2] * v[2],
        tensor[1] * v[0] + tensor[3] * v[1] + tensor[4] * v[2],
        tensor[2] * v[0] + tensor[4] * v[1] + tensor[5] * v[2],
    ]
}

fn angle_between(first: [f32; 3], second: [f32; 3]) -> f32 {
    dot(normalize(first), normalize(second))
        .clamp(-1.0, 1.0)
        .acos()
}

fn rotation_of(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half = angle * 0.5;
    let sine = half.sin();
    [axis[0] * sine, axis[1] * sine, axis[2] * sine, half.cos()]
}

fn assert_momentum_held(
    state: &BodyState,
    mass: &MassProperties,
    momentum: [f32; 3],
    frame: usize,
) {
    let measured = body_momentum(state, mass);
    let drift = (length(measured) - length(momentum)).abs() / length(momentum);
    assert!(
        drift < MOMENTUM_DRIFT,
        "angular momentum magnitude drifted by {drift} at frame {frame}"
    );
    let turn = angle_between(measured, momentum);
    assert!(
        turn < MOMENTUM_TURN,
        "angular momentum turned {turn} radians at frame {frame}"
    );
}

#[test]
fn free_rotation_conserves_angular_momentum() {
    let mass = mass_properties();
    let mut world = new_world(static_config());
    let body = spin_body(&mut world, NON_PRINCIPAL_SPIN);
    let initial = world.read_state(body);
    let momentum = body_momentum(&initial, &mass);
    let mut precessed = 0.0f32;
    for frame in 1..=FRAMES {
        world.step(DT);
        if !frame.is_multiple_of(MEASURE_EVERY) {
            continue;
        }
        world.wait();
        let state = world.read_state(body);
        assert_momentum_held(&state, &mass, momentum, frame);
        precessed = precessed.max(angle_between(body_spin(&state), body_spin(&initial)));
    }
    assert!(
        precessed > 1.0,
        "a non principal spin must precess, the body spin only turned {precessed} radians"
    );
}

#[test]
fn intermediate_axis_spin_tumbles_and_keeps_its_energy() {
    let mass = mass_properties();
    let mut world = new_world(static_config());
    let body = spin_body(&mut world, TILTED_AXIS_SPIN);
    let initial = world.read_state(body);
    let momentum = body_momentum(&initial, &mass);
    let energy = kinetic_energy(&initial, &mass);
    let mut tumbled = 0.0f32;
    for frame in 1..=FRAMES {
        world.step(DT);
        if !frame.is_multiple_of(MEASURE_EVERY) {
            continue;
        }
        world.wait();
        let state = world.read_state(body);
        assert_momentum_held(&state, &mass, momentum, frame);
        let drift = (kinetic_energy(&state, &mass) - energy).abs() / energy;
        assert!(
            drift < ENERGY_DRIFT,
            "a torque free spin must keep its kinetic energy, drifted by {drift} at frame {frame}"
        );
        tumbled = tumbled.max(angle_between(body_spin(&state), body_spin(&initial)));
    }
    assert!(
        tumbled > FRAC_PI_2,
        "an intermediate axis spin must tumble, the body spin only turned {tumbled} radians"
    );
}

#[test]
fn principal_axis_spin_stays_on_its_axis() {
    let mut world = new_world(static_config());
    let body = spin_body(&mut world, [0.0, 0.0, 1.0]);
    for frame in 1..=FRAMES {
        world.step(DT);
        if !frame.is_multiple_of(MEASURE_EVERY) {
            continue;
        }
        world.wait();
        let state = world.read_state(body);
        assert_eq!(
            state.angular_velocity,
            [0.0, 0.0, 1.0],
            "a principal axis spin must keep its axis bit exactly at frame {frame}"
        );
        assert!(
            state.orientation[0].abs() < 1e-6 && state.orientation[1].abs() < 1e-6,
            "a principal axis spin must stay a rotation about that axis, got {:?} at frame {frame}",
            state.orientation
        );
    }
}

#[test]
fn isotropic_spin_is_exactly_untouched() {
    let mut world = new_world(static_config());
    let spin = [1.0, 1.0, 1.0];
    let body = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0; 3])
            .angular_velocity(spin),
    );
    for frame in 1..=FRAMES {
        world.step(DT);
        if !frame.is_multiple_of(MEASURE_EVERY) {
            continue;
        }
        world.wait();
        let state = world.read_state(body);
        assert_eq!(
            state.angular_velocity, spin,
            "an isotropic body carries no gyroscopic torque, it changed at frame {frame}"
        );
        let analytic = quat_rotate(
            rotation_of(normalize(spin), length(spin) * DT * frame as f32),
            [0.0, 0.0, 1.0],
        );
        assert!(
            angle_between(quat_rotate(state.orientation, [0.0, 0.0, 1.0]), analytic) < 1e-2,
            "an isotropic body must spin about its axis at its analytic rate, at frame {frame}"
        );
    }
}

#[test]
fn an_angular_impulse_adds_exactly_its_momentum() {
    let mass = mass_properties();
    let mut world = new_world(static_config());
    let body = spin_body(&mut world, NON_PRINCIPAL_SPIN);
    let impulse = [0.25, -0.5, 0.75];
    let base = body_momentum(&world.read_state(body), &mass);
    let target = [
        base[0] + impulse[0],
        base[1] + impulse[1],
        base[2] + impulse[2],
    ];
    world.apply_angular_impulse(body, impulse);
    for frame in 1..=FRAMES {
        world.step(DT);
        if !frame.is_multiple_of(MEASURE_EVERY) {
            continue;
        }
        world.wait();
        let state = world.read_state(body);
        assert_momentum_held(&state, &mass, target, frame);
        if frame == MEASURE_EVERY {
            let measured = body_momentum(&state, &mass);
            let error = (length(measured) - length(target)).abs() / length(target);
            assert!(
                error < 1e-4,
                "an angular impulse must add exactly its momentum, off by {error}"
            );
        }
    }
}
