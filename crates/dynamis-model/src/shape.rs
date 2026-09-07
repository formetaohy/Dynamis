#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeSourceHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Sphere { radius: f32 },
    Cuboid { half_extents: [f32; 3] },
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
            "cuboid half extents must be strictly positive"
        );
        Self::Cuboid { half_extents }
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
            Self::Cuboid { half_extents } => {
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
