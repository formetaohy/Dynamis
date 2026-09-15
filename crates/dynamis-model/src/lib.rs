mod body;
mod character;
mod collider;
mod collision;
mod config;
mod constraint;
mod event;
mod impact;
mod mass;
pub mod math;
mod query;
mod shape;
mod soft;
mod surface;
mod vehicle;

pub use body::{BodyDesc, BodyHandle, BodyState};
pub use character::{CharacterDesc, CharacterHandle, CharacterInput, CharacterState};
pub use collider::{ColliderDesc, ContactEventMode};
pub use collision::CollisionFilter;
pub use config::{MaterialCombine, PhysicsConfig};
pub use constraint::{
    ConstraintBreak, ConstraintDesc, ConstraintHandle, ConstraintKind, ConstraintLimit,
    ConstraintMotor, ConstraintSpring, ConstraintSwing, DofDesc, JointDof, JointState,
};
pub use event::{ContactEvent, ContactEventKind};
pub use impact::ImpactEvent;
pub use mass::{
    MassProperties, MassSource, analytic_solid, compute_mass_properties, mass_properties_of_intent,
    shape_solid, solid_volume_of,
};
pub use query::QueryFilter;
pub use shape::{Shape, ShapeSourceHandle, SolidGeometry};
pub use soft::{
    FluidMaterial, SoftAttachment, SoftBodyDesc, SoftBodyHandle, SoftElement, SoftElementKind,
    SoftElementState, SoftMaterial, SoftParticleState,
};
pub use surface::{SurfaceDesc, SurfaceTable};
pub use vehicle::{VehicleDesc, VehicleHandle, VehicleInput, VehicleState, WheelDesc};
