mod body;
mod collider;
mod config;
mod constraint;
mod event;
mod mass;
mod query;
mod shape;

pub use body::{BodyDesc, BodyHandle, BodyState, MAX_COLLIDERS_PER_BODY};
pub use collider::ColliderDesc;
pub use config::PhysicsConfig;
pub use constraint::{
    ConstraintDesc, ConstraintHandle, ConstraintKind, ConstraintLimit, ConstraintMotor,
    ConstraintSpring,
};
pub use event::{ContactEvent, ContactEventKind};
pub use mass::{MassProperties, compute_mass_properties};
pub use query::QueryFilter;
pub use shape::{Shape, ShapeSourceHandle};
