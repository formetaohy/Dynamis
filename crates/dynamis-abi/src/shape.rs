use crate::constant::{
    SHAPE_CAPSULE, SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH,
    SHAPE_NONE, SHAPE_PLANE, SHAPE_SPHERE, SHAPE_TRIANGLE,
};
use dynamis_model::{Shape, ShapeSourceHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapePath {
    Convex,
    WorldGeometry,
    Vacant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeScale {
    InDimensions,
    UniformFolded,
    InRecord,
    NotScalable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeRole {
    pub code: u32,
    pub path: ShapePath,
    pub scale: ShapeScale,
    pub source: bool,
    pub analytic: bool,
}

pub const SHAPE_ROLE_COUNT: u32 = SHAPE_PLANE + 1;

pub const SHAPE_ROLES: [ShapeRole; SHAPE_ROLE_COUNT as usize] = [
    ShapeRole {
        code: SHAPE_NONE,
        path: ShapePath::Vacant,
        scale: ShapeScale::NotScalable,
        source: false,
        analytic: false,
    },
    ShapeRole {
        code: SHAPE_SPHERE,
        path: ShapePath::Convex,
        scale: ShapeScale::UniformFolded,
        source: false,
        analytic: true,
    },
    ShapeRole {
        code: SHAPE_CUBOID,
        path: ShapePath::Convex,
        scale: ShapeScale::InDimensions,
        source: false,
        analytic: true,
    },
    ShapeRole {
        code: SHAPE_CAPSULE,
        path: ShapePath::Convex,
        scale: ShapeScale::UniformFolded,
        source: false,
        analytic: true,
    },
    ShapeRole {
        code: SHAPE_CYLINDER,
        path: ShapePath::Convex,
        scale: ShapeScale::UniformFolded,
        source: false,
        analytic: false,
    },
    ShapeRole {
        code: SHAPE_HULL,
        path: ShapePath::Convex,
        scale: ShapeScale::InRecord,
        source: true,
        analytic: false,
    },
    ShapeRole {
        code: SHAPE_MESH,
        path: ShapePath::WorldGeometry,
        scale: ShapeScale::InRecord,
        source: true,
        analytic: false,
    },
    ShapeRole {
        code: SHAPE_HEIGHTFIELD,
        path: ShapePath::WorldGeometry,
        scale: ShapeScale::InRecord,
        source: true,
        analytic: false,
    },
    ShapeRole {
        code: SHAPE_TRIANGLE,
        path: ShapePath::Convex,
        scale: ShapeScale::InRecord,
        source: true,
        analytic: false,
    },
    ShapeRole {
        code: SHAPE_PLANE,
        path: ShapePath::WorldGeometry,
        scale: ShapeScale::NotScalable,
        source: false,
        analytic: false,
    },
];

const _: () = {
    let mut index = 0;
    let mut vacant = 0;
    while index < SHAPE_ROLE_COUNT as usize {
        let role = SHAPE_ROLES[index];
        assert!(
            role.code == index as u32,
            "a shape role must be declared for every device shape code"
        );
        if matches!(role.path, ShapePath::Vacant) {
            vacant += 1;
        }
        assert!(
            !role.source || matches!(role.scale, ShapeScale::InRecord),
            "a shape with a source must carry its scale"
        );
        assert!(
            !matches!(role.path, ShapePath::Convex)
                || !matches!(role.scale, ShapeScale::NotScalable),
            "a convex shape must have dimensions to scale"
        );
        assert!(
            !matches!(role.path, ShapePath::WorldGeometry)
                || !matches!(role.scale, ShapeScale::UniformFolded),
            "world geometry must not fold its scale"
        );
        assert!(
            !matches!(role.path, ShapePath::Convex)
                || !matches!(role.scale, ShapeScale::InDimensions)
                || !role.source,
            "a shape that scales its dimensions has no source to scale"
        );
        assert!(
            !role.analytic || matches!(role.path, ShapePath::Convex),
            "only a convex shape can answer an analytic kernel"
        );
        index += 1;
    }
    assert!(
        vacant == 1,
        "exactly one shape code may belong to no collision path"
    );
};

fn uniform(scale: [f32; 3]) -> bool {
    scale[0] == scale[1] && scale[1] == scale[2]
}

impl ShapeScale {
    pub fn dimension_factor(self, scale: [f32; 3]) -> [f32; 3] {
        match self {
            Self::InDimensions => scale,
            Self::UniformFolded if uniform(scale) => [scale[0]; 3],
            Self::UniformFolded | Self::InRecord | Self::NotScalable => [1.0; 3],
        }
    }

    pub fn record_scale(self, scale: [f32; 3]) -> [f32; 3] {
        match self {
            Self::InRecord => scale,
            Self::UniformFolded if !uniform(scale) => scale,
            Self::InDimensions | Self::UniformFolded | Self::NotScalable => [1.0; 3],
        }
    }
}

impl ShapeRole {
    pub const fn of(code: u32) -> Self {
        assert!(
            code < SHAPE_ROLE_COUNT,
            "a shape role must be declared for every device shape code"
        );
        SHAPE_ROLES[code as usize]
    }

    pub fn of_shape(shape: &Shape) -> Self {
        Self::of(shape_code(shape))
    }

    pub const fn convex(self) -> bool {
        matches!(self.path, ShapePath::Convex)
    }

    pub const fn world_geometry(self) -> bool {
        matches!(self.path, ShapePath::WorldGeometry)
    }

    pub const fn triangle_scene(self) -> bool {
        self.world_geometry() && self.source
    }
}

pub fn shape_code(shape: &Shape) -> u32 {
    match shape {
        Shape::Sphere { .. } => SHAPE_SPHERE,
        Shape::Cuboid { .. } => SHAPE_CUBOID,
        Shape::Capsule { .. } => SHAPE_CAPSULE,
        Shape::Cylinder { .. } => SHAPE_CYLINDER,
        Shape::Hull(_) => SHAPE_HULL,
        Shape::Mesh(_) => SHAPE_MESH,
        Shape::HeightField(_) => SHAPE_HEIGHTFIELD,
        Shape::Plane => SHAPE_PLANE,
    }
}

pub fn shape_source_handle(shape: &Shape) -> Option<ShapeSourceHandle> {
    match shape {
        Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) => Some(*handle),
        _ => None,
    }
}

fn predicate(out: &mut String, name: &str, holds: fn(&ShapeRole) -> bool) {
    out.push_str("fn ");
    out.push_str(name);
    out.push_str("(kind: u32) -> bool { return ");
    let mut named = false;
    for role in &SHAPE_ROLES {
        if !holds(role) {
            continue;
        }
        if named {
            out.push_str(" || ");
        }
        out.push_str(&format!("kind == {}u", role.code));
        named = true;
    }
    if !named {
        out.push_str("false");
    }
    out.push_str("; }\n");
}

pub(crate) fn emit_predicates(out: &mut String) {
    predicate(out, "shape_world_geometry", |role| role.world_geometry());
    predicate(out, "shape_triangle_scene", |role| role.triangle_scene());
    predicate(out, "shape_source", |role| role.source);
    predicate(out, "shape_analytic", |role| role.analytic);
}
