#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeSourceHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
    Capsule { radius: f32, half_height: f32 },
    Cylinder { radius: f32, half_height: f32 },
    Hull(ShapeSourceHandle),
    Mesh(ShapeSourceHandle),
    HeightField(ShapeSourceHandle),
}

impl Shape {
    pub fn sphere(radius: f32) -> Self {
        assert!(radius > 0.0, "shape radius must be strictly positive");
        Self::Sphere { radius }
    }

    pub fn cuboid(half_extents: [f32; 3]) -> Self {
        assert!(
            half_extents.iter().all(|extent| *extent > 0.0),
            "box half extents must be strictly positive"
        );
        Self::Box { half_extents }
    }

    pub fn capsule(radius: f32, half_height: f32) -> Self {
        assert!(radius > 0.0, "capsule radius must be strictly positive");
        assert!(
            half_height >= 0.0,
            "capsule half height must be non-negative"
        );
        Self::Capsule {
            radius,
            half_height,
        }
    }

    pub fn cylinder(radius: f32, half_height: f32) -> Self {
        assert!(radius > 0.0, "cylinder radius must be strictly positive");
        assert!(
            half_height >= 0.0,
            "cylinder half height must be non-negative"
        );
        Self::Cylinder {
            radius,
            half_height,
        }
    }

    pub fn hull(source: ShapeSourceHandle) -> Self {
        Self::Hull(source)
    }

    pub fn mesh(source: ShapeSourceHandle) -> Self {
        Self::Mesh(source)
    }

    pub fn height_field(source: ShapeSourceHandle) -> Self {
        Self::HeightField(source)
    }

    pub fn bounding_radius(&self) -> f32 {
        match *self {
            Self::Sphere { radius } => radius,
            Self::Box { half_extents } => {
                let x = half_extents[0];
                let y = half_extents[1];
                let z = half_extents[2];
                (x * x + y * y + z * z).sqrt()
            }
            Self::Capsule {
                radius,
                half_height,
            }
            | Self::Cylinder {
                radius,
                half_height,
            } => (half_height * half_height + radius * radius).sqrt(),
            Self::Hull(_) | Self::Mesh(_) | Self::HeightField(_) => 0.0,
        }
    }

    pub fn is_world_geometry(&self) -> bool {
        matches!(self, Self::Mesh(_) | Self::HeightField(_))
    }

    pub fn is_convex(&self) -> bool {
        !matches!(self, Self::Mesh(_) | Self::HeightField(_))
    }
}

pub fn inverse_inertia_diagonal(
    shape: &Shape,
    inverse_mass: f32,
    world_bounds: Option<([f32; 3], [f32; 3])>,
) -> [f32; 3] {
    if inverse_mass == 0.0 {
        return [0.0; 3];
    }
    let mass = 1.0 / inverse_mass;
    match *shape {
        Shape::Sphere { radius } => {
            let i = 2.0 / 5.0 * mass * radius * radius;
            [1.0 / i; 3]
        }
        Shape::Box { half_extents } => {
            let hx = half_extents[0];
            let hy = half_extents[1];
            let hz = half_extents[2];
            let ex = 2.0 * hx;
            let ey = 2.0 * hy;
            let ez = 2.0 * hz;
            let ix = mass / 12.0 * (ey * ey + ez * ez);
            let iy = mass / 12.0 * (ex * ex + ez * ez);
            let iz = mass / 12.0 * (ex * ex + ey * ey);
            [1.0 / ix, 1.0 / iy, 1.0 / iz]
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
            [1.0 / ix, 1.0 / iy, 1.0 / ix]
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let h = 2.0 * half_height;
            let ix = mass / 12.0 * (3.0 * radius * radius + h * h);
            let iy = 0.5 * mass * radius * radius;
            [1.0 / ix, 1.0 / iy, 1.0 / ix]
        }
        Shape::Hull(_) | Shape::Mesh(_) | Shape::HeightField(_) => {
            let (min, max) = world_bounds.expect("world geometry inertia requires bounds");
            let ex = max[0] - min[0];
            let ey = max[1] - min[1];
            let ez = max[2] - min[2];
            let ix = mass / 12.0 * (ey * ey + ez * ez);
            let iy = mass / 12.0 * (ex * ex + ez * ez);
            let iz = mass / 12.0 * (ex * ex + ey * ey);
            [1.0 / ix, 1.0 / iy, 1.0 / iz]
        }
    }
}
