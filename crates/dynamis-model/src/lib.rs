mod body;
mod collider;
mod config;
mod constraint;
mod event;
mod query;
mod shape;

pub use body::{BODY_DESC_COLLIDERS_MAX, BodyDesc, BodyHandle, BodyState};
pub use collider::ColliderDesc;
pub use config::PhysicsConfig;
pub use constraint::{
    ConstraintDesc, ConstraintHandle, ConstraintKind, ConstraintLimit, ConstraintMotor,
    ConstraintSpring,
};
pub use event::{ContactEvent, ContactEventKind};
pub use query::QueryFilter;
pub use shape::{Shape, ShapeSourceHandle, inverse_inertia_diagonal};
