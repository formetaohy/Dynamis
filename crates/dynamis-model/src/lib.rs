mod body;
mod config;
mod constraint;
mod shape;

pub use body::{
    BodyDesc, BodyHandle, BodyState, DEFAULT_COLLISION_GROUP, DEFAULT_COLLISION_MASK,
};
pub use config::PhysicsConfig;
pub use constraint::{ConstraintDesc, ConstraintHandle, ConstraintKind};
pub use shape::{ShapeDesc, ShapeKind};
