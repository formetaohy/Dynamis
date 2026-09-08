use crate::collider::ColliderDesc;
use crate::shape::Shape;

pub struct MassProperties {
    pub com: [f32; 3],
    pub inverse_inertia: [f32; 6],
}

impl MassProperties {
    pub fn zeroed() -> Self {
        Self {
            com: [0.0; 3],
            inverse_inertia: [0.0; 6],
        }
    }
}

#[derive(Clone, Copy)]
pub enum MassSource {
    Fixed(f32),
    Density(f32),
}

pub fn compute_mass_properties(
    colliders: &[ColliderDesc],
    source: MassSource,
    com: Option<[f32; 3]>,
    bounds: impl Fn(&Shape) -> Option<([f32; 3], [f32; 3])>,
) -> MassProperties {
    let solid = colliders
        .iter()
        .filter(|collider| !collider.sensor && !matches!(collider.shape, Shape::Plane))
        .collect::<Vec<_>>();
    if solid.is_empty() {
        return MassProperties::zeroed();
    }
    let volumes = solid
        .iter()
        .map(|collider| shape_volume(&collider.shape, collider.scale, bounds(&collider.shape)))
        .collect::<Vec<_>>();
    let total_volume = volumes.iter().sum::<f32>();
    if total_volume <= 0.0 {
        return MassProperties::zeroed();
    }
    let mass = match source {
        MassSource::Fixed(mass) => mass,
        MassSource::Density(density) => density * total_volume,
    };
    if mass <= 0.0 {
        return MassProperties::zeroed();
    }
    let com = com.unwrap_or_else(|| {
        let mut sum = [0.0f32; 3];
        for (index, collider) in solid.iter().enumerate() {
            let weight = volumes[index];
            sum[0] += collider.offset[0] * weight;
            sum[1] += collider.offset[1] * weight;
            sum[2] += collider.offset[2] * weight;
        }
        [
            sum[0] / total_volume,
            sum[1] / total_volume,
            sum[2] / total_volume,
        ]
    });
    let mut inertia = [0.0f32; 6];
    for (index, collider) in solid.iter().enumerate() {
        let shape_mass = mass * volumes[index] / total_volume;
        let local = analytic_inertia(&collider.shape, shape_mass, bounds(&collider.shape));
        let scaled = inertia_scale(local, collider.scale);
        let rotated = inertia_rotate(scaled, collider.rotation);
        let offset = [
            collider.offset[0] - com[0],
            collider.offset[1] - com[1],
            collider.offset[2] - com[2],
        ];
        let translated = inertia_translate(rotated, offset, shape_mass);
        inertia[0] += translated[0];
        inertia[1] += translated[1];
        inertia[2] += translated[2];
        inertia[3] += translated[3];
        inertia[4] += translated[4];
        inertia[5] += translated[5];
    }
    MassProperties {
        com,
        inverse_inertia: inertia_inverse(inertia),
    }
}

pub fn solid_volume_of(
    colliders: &[ColliderDesc],
    bounds: &impl Fn(&Shape) -> Option<([f32; 3], [f32; 3])>,
) -> f32 {
    colliders
        .iter()
        .filter(|collider| !collider.sensor && !matches!(collider.shape, Shape::Plane))
        .map(|collider| shape_volume(&collider.shape, collider.scale, bounds(&collider.shape)))
        .sum()
}

pub fn shape_volume(shape: &Shape, scale: [f32; 3], bounds: Option<([f32; 3], [f32; 3])>) -> f32 {
    let base = match *shape {
        Shape::Sphere { radius } => 4.0 / 3.0 * std::f32::consts::PI * radius * radius * radius,
        Shape::Cuboid { half_extents } => 8.0 * half_extents[0] * half_extents[1] * half_extents[2],
        Shape::Capsule {
            radius,
            half_height,
        } => std::f32::consts::PI * radius * radius * (4.0 / 3.0 * radius + 2.0 * half_height),
        Shape::Cylinder {
            radius,
            half_height,
        } => std::f32::consts::PI * radius * radius * 2.0 * half_height,
        Shape::Hull(_) | Shape::Mesh(_) | Shape::HeightField(_) => {
            let (min, max) = bounds.expect("world geometry volume requires bounds");
            let extent = [
                (max[0] - min[0]).max(0.0),
                (max[1] - min[1]).max(0.0),
                (max[2] - min[2]).max(0.0),
            ];
            extent[0] * extent[1] * extent[2]
        }
        Shape::Plane => 0.0,
    };
    base * scale[0] * scale[1] * scale[2]
}

fn analytic_inertia(shape: &Shape, mass: f32, bounds: Option<([f32; 3], [f32; 3])>) -> [f32; 6] {
    match *shape {
        Shape::Sphere { radius } => {
            let i = 2.0 / 5.0 * mass * radius * radius;
            [i, 0.0, 0.0, i, 0.0, i]
        }
        Shape::Cuboid { half_extents } => {
            let ex = half_extents[0] * 2.0;
            let ey = half_extents[1] * 2.0;
            let ez = half_extents[2] * 2.0;
            let ix = mass / 12.0 * (ey * ey + ez * ez);
            let iy = mass / 12.0 * (ex * ex + ez * ez);
            let iz = mass / 12.0 * (ex * ex + ey * ey);
            [ix, 0.0, 0.0, iy, 0.0, iz]
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let r = radius;
            let h = half_height;
            let cylinder_volume = std::f32::consts::PI * r * r * 2.0 * h;
            let sphere_volume = 4.0 / 3.0 * std::f32::consts::PI * r * r * r;
            let total = cylinder_volume + sphere_volume;
            let cylinder_mass = mass * cylinder_volume / total;
            let sphere_mass = mass * sphere_volume / total;
            let ix = cylinder_mass / 12.0 * (3.0 * r * r + (2.0 * h) * (2.0 * h))
                + sphere_mass
                    * (2.0 / 5.0 * r * r + (h + 3.0 / 8.0 * r) * (h + 3.0 / 8.0 * r))
                    * 2.0;
            let iy = cylinder_mass / 2.0 * r * r + sphere_mass * 2.0 / 5.0 * r * r * 2.0;
            [ix, 0.0, 0.0, iy, 0.0, ix]
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let h = half_height * 2.0;
            let ix = mass / 12.0 * (3.0 * radius * radius + h * h);
            let iy = 0.5 * mass * radius * radius;
            let iz = ix;
            [ix, 0.0, 0.0, iy, 0.0, iz]
        }
        Shape::Hull(_) | Shape::Mesh(_) | Shape::HeightField(_) => {
            let (min, max) = bounds.expect("world geometry inertia requires bounds");
            let ex = max[0] - min[0];
            let ey = max[1] - min[1];
            let ez = max[2] - min[2];
            let ix = mass / 12.0 * (ey * ey + ez * ez);
            let iy = mass / 12.0 * (ex * ex + ez * ez);
            let iz = mass / 12.0 * (ex * ex + ey * ey);
            [ix, 0.0, 0.0, iy, 0.0, iz]
        }
        Shape::Plane => panic!("plane colliders carry no mass"),
    }
}

fn inertia_translate(inertia: [f32; 6], offset: [f32; 3], mass: f32) -> [f32; 6] {
    let d = offset;
    [
        inertia[0] + mass * (d[1] * d[1] + d[2] * d[2]),
        inertia[1] - mass * d[0] * d[1],
        inertia[2] - mass * d[0] * d[2],
        inertia[3] + mass * (d[0] * d[0] + d[2] * d[2]),
        inertia[4] - mass * d[1] * d[2],
        inertia[5] + mass * (d[0] * d[0] + d[1] * d[1]),
    ]
}

fn inertia_scale(inertia: [f32; 6], scale: [f32; 3]) -> [f32; 6] {
    let m = sym_to_mat(inertia);
    let trace = m[0][0] + m[1][1] + m[2][2];
    let second = [
        [trace * 0.5 - m[0][0], -m[0][1], -m[0][2]],
        [-m[1][0], trace * 0.5 - m[1][1], -m[1][2]],
        [-m[2][0], -m[2][1], trace * 0.5 - m[2][2]],
    ];
    let mut scaled = [[0.0f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            scaled[row][col] = scale[row] * second[row][col] * scale[col];
        }
    }
    let scaled_trace = scaled[0][0] + scaled[1][1] + scaled[2][2];
    let result = [
        [scaled_trace - scaled[0][0], -scaled[0][1], -scaled[0][2]],
        [-scaled[1][0], scaled_trace - scaled[1][1], -scaled[1][2]],
        [-scaled[2][0], -scaled[2][1], scaled_trace - scaled[2][2]],
    ];
    mat_to_sym(result)
}

fn inertia_rotate(inertia: [f32; 6], q: [f32; 4]) -> [f32; 6] {
    let r = mat_from_quat(q);
    let rotated = mat_mul(r, mat_mul(sym_to_mat(inertia), mat_transpose(r)));
    mat_to_sym(rotated)
}

pub(crate) fn inertia_inverse(inertia: [f32; 6]) -> [f32; 6] {
    let m = sym_to_mat(inertia);
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(det > 0.0, "inertia tensor must be positive definite");
    let inverse = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][1] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[0][1] * m[2][2] - m[0][2] * m[2][1]) / det,
            (m[0][2] * m[1][1] - m[0][1] * m[1][2]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ];
    mat_to_sym(inverse)
}

fn mat_from_quat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let x = q[0];
    let y = q[1];
    let z = q[2];
    let w = q[3];
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn sym_to_mat(sym: [f32; 6]) -> [[f32; 3]; 3] {
    [
        [sym[0], sym[1], sym[2]],
        [sym[1], sym[3], sym[4]],
        [sym[2], sym[4], sym[5]],
    ]
}

fn mat_to_sym(m: [[f32; 3]; 3]) -> [f32; 6] {
    [
        m[0][0],
        (m[0][1] + m[1][0]) * 0.5,
        (m[0][2] + m[2][0]) * 0.5,
        m[1][1],
        (m[1][2] + m[2][1]) * 0.5,
        m[2][2],
    ]
}

fn mat_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut result = [[0.0f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            result[row][col] =
                a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    result
}

fn mat_transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}
