use dynamis::Simulation;
use dynamis_example_render::{
    AppContext, Color, EulerRot, Geometry, Material, Quat, Transform, Vec3,
};

pub const GROUND_HALF: f32 = 30.0;

pub fn setup_scene(ctx: &mut AppContext, simulation: &mut Simulation) {
    simulation.spawn(
        dynamis::BodyDesc::cuboid([GROUND_HALF, 0.5, GROUND_HALF])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    ctx.spawn(
        &Geometry::Cuboid {
            half_extents: [GROUND_HALF, 0.5, GROUND_HALF],
        },
        Material {
            base_color: Color::srgb(0.2, 0.23, 0.26),
            roughness: 0.95,
            ..Default::default()
        },
        Transform::from_xyz(0.0, -0.5, 0.0),
    );
    ctx.lights.direction = Quat::from_euler(EulerRot::XYZ, -0.9, 0.7, 0.0) * Vec3::NEG_Z;
    ctx.lights.color = Color::srgb(1.0, 0.94, 0.84);
    ctx.lights.intensity = 1.6;
    ctx.lights.ambient = Color::srgb(0.72, 0.74, 0.78);
    ctx.lights.ambient_intensity = 0.35;
}

pub fn indexed_color(index: usize) -> Color {
    const PALETTE: [[u8; 3]; 8] = [
        [224, 108, 117],
        [97, 175, 239],
        [152, 195, 121],
        [229, 192, 123],
        [198, 120, 221],
        [86, 182, 194],
        [209, 154, 102],
        [92, 143, 253],
    ];
    let [red, green, blue] = PALETTE[index % PALETTE.len()];
    Color::srgb_u8(red, green, blue)
}

pub fn geometry_from_shape(shape: &dynamis::Shape) -> Geometry {
    match *shape {
        dynamis::Shape::Sphere { radius } => Geometry::Sphere { radius },
        dynamis::Shape::Cuboid { half_extents } => Geometry::Cuboid { half_extents },
        dynamis::Shape::Capsule {
            radius,
            half_height,
        } => Geometry::Capsule {
            radius,
            half_height,
        },
        dynamis::Shape::Cylinder {
            radius,
            half_height,
        } => Geometry::Cylinder {
            radius,
            half_height,
        },
        dynamis::Shape::Hull(_) | dynamis::Shape::Mesh(_) | dynamis::Shape::HeightField(_) => {
            panic!("vertex-sourced shapes have no render counterpart")
        }
    }
}

pub fn standard_material(color: Color) -> Material {
    Material {
        base_color: color,
        roughness: 0.55,
        ..Default::default()
    }
}
