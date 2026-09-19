mod body;
mod character;
mod collider;
mod collision;
mod config;
mod constraint;
pub mod domain;
mod event;
mod field;
mod impact;
mod mass;
pub mod math;
mod query;
mod scene;
mod shape;
mod soft;
mod surface;
mod vehicle;

pub use body::{BodyDesc, BodyHandle, BodyKind, BodyState};
pub use character::{CharacterDesc, CharacterHandle, CharacterInput, CharacterState};
pub use collider::{ColliderDesc, ContactEventMode};
pub use collision::CollisionFilter;
pub use config::{MaterialCombine, PhysicsConfig};
pub use constraint::{
    ConstraintBreak, ConstraintData, ConstraintDesc, ConstraintHandle, ConstraintKind,
    ConstraintLimit, ConstraintMotor, ConstraintPositionTarget, ConstraintSpring, ConstraintSwing,
    DofDesc, JointDof, JointState,
};
pub use event::{ContactEvent, ContactEventKind};
pub use field::{FieldDesc, FieldHandle, FieldRegion};
pub use impact::ImpactEvent;
pub use mass::{
    MassProperties, MassSource, analytic_solid, compute_mass_properties, mass_properties_of_intent,
    shape_solid, solid_volume_of,
};
pub use query::{QueryFilter, QueryTargets};
pub use scene::SceneTarget;
pub use shape::{Shape, ShapeSourceHandle, SolidGeometry};
pub use soft::{
    FluidMaterial, SoftAttachment, SoftBodyDesc, SoftBodyHandle, SoftElement, SoftElementKind,
    SoftElementState, SoftMaterial, SoftParticleState,
};
pub use surface::{SurfaceDesc, SurfaceTable};
pub use vehicle::{VehicleDesc, VehicleHandle, VehicleInput, VehicleState, WheelDesc};
