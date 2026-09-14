use crate::collider::ColliderDesc;
use crate::shape::{Shape, SolidGeometry};

pub struct MassProperties {
    pub com: [f32; 3],
    pub inertia: [f32; 6],
    pub inverse_inertia: [f32; 6],
}

impl MassProperties {
    pub fn zeroed() -> Self {
        Self {
            com: [0.0; 3],
            inertia: [0.0; 6],
            inverse_inertia: [0.0; 6],
        }
    }

    fn of(com: [f32; 3], inertia: [f32; 6]) -> Self {
        Self {
            com,
            inverse_inertia: inertia_inverse(inertia),
            inertia,
        }
    }
}

#[derive(Clone, Copy)]
pub enum MassSource {
    Fixed(f32),
    Density(f32),
}

pub fn analytic_solid(shape: &Shape) -> Option<SolidGeometry> {
    Some(match *shape {
        Shape::Sphere { radius } => {
            let i = 2.0 / 5.0 * radius * radius;
            SolidGeometry {
                volume: 4.0 / 3.0 * std::f32::consts::PI * radius * radius * radius,
                centroid: [0.0; 3],
                unit_inertia: [i, 0.0, 0.0, i, 0.0, i],
            }
        }
        Shape::Cuboid { half_extents } => {
            let ex = half_extents[0] * 2.0;
            let ey = half_extents[1] * 2.0;
            let ez = half_extents[2] * 2.0;
            SolidGeometry {
                volume: ex * ey * ez,
                centroid: [0.0; 3],
                unit_inertia: [
                    (ey * ey + ez * ez) / 12.0,
                    0.0,
                    0.0,
                    (ex * ex + ez * ez) / 12.0,
                    0.0,
                    (ex * ex + ey * ey) / 12.0,
                ],
            }
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let cylinder = std::f32::consts::PI * radius * radius * 2.0 * half_height;
            let sphere = 4.0 / 3.0 * std::f32::consts::PI * radius * radius * radius;
            let total = cylinder + sphere;
            let height = half_height + 3.0 / 8.0 * radius;
            let axial = (cylinder * 0.5 * radius * radius + 0.4 * sphere * radius * radius) / total;
            let lateral = (cylinder / 12.0
                * (3.0 * radius * radius + 4.0 * half_height * half_height)
                + sphere * (83.0 / 320.0 * radius * radius + height * height))
                / total;
            SolidGeometry {
                volume: total,
                centroid: [0.0; 3],
                unit_inertia: [lateral, 0.0, 0.0, axial, 0.0, lateral],
            }
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let height = half_height * 2.0;
            let lateral = (3.0 * radius * radius + height * height) / 12.0;
            SolidGeometry {
                volume: std::f32::consts::PI * radius * radius * height,
                centroid: [0.0; 3],
                unit_inertia: [lateral, 0.0, 0.0, 0.5 * radius * radius, 0.0, lateral],
            }
        }
        Shape::Hull(_) | Shape::Mesh(_) | Shape::HeightField(_) | Shape::Plane => return None,
    })
}

pub fn shape_solid(
    shape: &Shape,
    source: impl Fn(&Shape) -> Option<SolidGeometry>,
) -> Option<SolidGeometry> {
    match shape {
        Shape::Hull(_) => {
            Some(source(shape).expect("a hull shape source must answer its solid geometry"))
        }
        Shape::Mesh(_) | Shape::HeightField(_) | Shape::Plane => None,
        _ => analytic_solid(shape),
    }
}

fn collider_solid(
    collider: &ColliderDesc,
    source: &impl Fn(&Shape) -> Option<SolidGeometry>,
) -> Option<SolidGeometry> {
    if collider.sensor {
        return None;
    }
    shape_solid(&collider.shape, source)
}

fn scale_volume(scale: [f32; 3]) -> f32 {
    scale[0] * scale[1] * scale[2]
}

fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    crate::math::quat_rotate(q, v)
}

fn centroid_of(collider: &ColliderDesc, solid: &SolidGeometry) -> [f32; 3] {
    let scaled = [
        collider.scale[0] * solid.centroid[0],
        collider.scale[1] * solid.centroid[1],
        collider.scale[2] * solid.centroid[2],
    ];
    let rotated = rotate(collider.rotation, scaled);
    [
        collider.offset[0] + rotated[0],
        collider.offset[1] + rotated[1],
        collider.offset[2] + rotated[2],
    ]
}

pub fn mass_properties_of_intent(
    colliders: &[ColliderDesc],
    mass: f32,
    com: Option<[f32; 3]>,
    inertia: Option<[f32; 6]>,
    source: impl Fn(&Shape) -> Option<SolidGeometry>,
) -> MassProperties {
    if let Some(inertia) = inertia {
        return MassProperties::of(com.unwrap_or([0.0; 3]), inertia);
    }
    compute_mass_properties(colliders, MassSource::Fixed(mass), com, source)
}

pub fn compute_mass_properties(
    colliders: &[ColliderDesc],
    source: MassSource,
    com: Option<[f32; 3]>,
    geometry: impl Fn(&Shape) -> Option<SolidGeometry>,
) -> MassProperties {
    let solids = colliders
        .iter()
        .filter_map(|collider| collider_solid(collider, &geometry).map(|solid| (collider, solid)))
        .collect::<Vec<_>>();
    if solids.is_empty() {
        return MassProperties::zeroed();
    }
    let volumes = solids
        .iter()
        .map(|(collider, solid)| scale_volume(collider.scale) * solid.volume)
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
        for ((collider, solid), volume) in solids.iter().zip(&volumes) {
            let at = centroid_of(collider, solid);
            sum[0] += at[0] * volume;
            sum[1] += at[1] * volume;
            sum[2] += at[2] * volume;
        }
        [
            sum[0] / total_volume,
            sum[1] / total_volume,
            sum[2] / total_volume,
        ]
    });
    let mut inertia = [0.0f32; 6];
    for ((collider, solid), volume) in solids.iter().zip(&volumes) {
        let shape_mass = mass * volume / total_volume;
        let local = inertia_scale(solid.unit_inertia, collider.scale);
        let scaled = [
            local[0] * shape_mass,
            local[1] * shape_mass,
            local[2] * shape_mass,
            local[3] * shape_mass,
            local[4] * shape_mass,
            local[5] * shape_mass,
        ];
        let rotated = inertia_rotate(scaled, collider.rotation);
        let at = centroid_of(collider, solid);
        let offset = [at[0] - com[0], at[1] - com[1], at[2] - com[2]];
        let translated = inertia_translate(rotated, offset, shape_mass);
        inertia[0] += translated[0];
        inertia[1] += translated[1];
        inertia[2] += translated[2];
        inertia[3] += translated[3];
        inertia[4] += translated[4];
        inertia[5] += translated[5];
    }
    MassProperties::of(com, inertia)
}

pub fn solid_volume_of(
    colliders: &[ColliderDesc],
    geometry: impl Fn(&Shape) -> Option<SolidGeometry>,
) -> f32 {
    colliders
        .iter()
        .filter_map(|collider| {
            collider_solid(collider, &geometry)
                .map(|solid| scale_volume(collider.scale) * solid.volume)
        })
        .sum()
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
    let [xx, xy, xz, yy, yz, zz] = inertia;
    let cofactor_xx = yy * zz - yz * yz;
    let cofactor_xy = xz * yz - xy * zz;
    let cofactor_xz = xy * yz - xz * yy;
    let cofactor_yy = xx * zz - xz * xz;
    let cofactor_yz = xy * xz - xx * yz;
    let cofactor_zz = xx * yy - xy * xy;
    let determinant = xx * cofactor_xx + xy * cofactor_xy + xz * cofactor_xz;
    assert!(
        determinant > 0.0,
        "inertia tensor must be positive definite"
    );
    [
        cofactor_xx / determinant,
        cofactor_xy / determinant,
        cofactor_xz / determinant,
        cofactor_yy / determinant,
        cofactor_yz / determinant,
        cofactor_zz / determinant,
    ]
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
