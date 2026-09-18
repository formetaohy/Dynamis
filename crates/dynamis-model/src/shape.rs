use crate::domain;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeSourceHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidGeometry {
    pub volume: f32,
    pub centroid: [f32; 3],
    pub unit_second_moment: [f32; 6],
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
    Plane,
}

impl Shape {
    pub fn assert_valid(&self) {
        match *self {
            Self::Sphere { radius } => {
                domain::positive(radius, "a sphere radius");
            }
            Self::Cuboid { half_extents } => {
                domain::finite_vector(half_extents, "cuboid half extents");
                for extent in half_extents {
                    domain::positive(extent, "a cuboid half extent");
                }
            }
            Self::Capsule {
                radius,
                half_height,
            }
            | Self::Cylinder {
                radius,
                half_height,
            } => {
                domain::positive(radius, "a shape radius");
                domain::non_negative(half_height, "a shape half height");
            }
            Self::Hull(_) | Self::Mesh(_) | Self::HeightField(_) | Self::Plane => {}
        }
    }

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

    pub fn plane() -> Self {
        Self::Plane
    }
}
