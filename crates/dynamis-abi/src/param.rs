use crate::{RowStreamsRecord, StepParamsRecord};
use bytemuck::Zeroable;
use dynamis_model::domain;
use dynamis_model::{MaterialCombine, PhysicsConfig};

macro_rules! census {
    (
        device {
            $( $variant:ident: $field:ident = $param:ident, )*
        }
        host {
            $( $host:ident, )*
        }
        rows {
            $( $row:ident: $row_field:ident, )*
        }
    ) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct Census {
            $( pub $field: u32, )*
            $( pub $host: u32, )*
        }

        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct RowStreams {
            $( pub $row_field: u32, )*
        }

        #[derive(Clone, Copy)]
        pub enum Count {
            $( $variant, )*
            $( $row, )*
        }

        impl Count {
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )* $( Self::$row, )* ];

            pub const fn bound(self) -> Bound {
                match self {
                    $( Self::$variant => Bound::Params(stringify!($param)), )*
                    $( Self::$row => Bound::Rows(stringify!($row_field)), )*
                }
            }

            pub const fn rows(self, params: &StepParamsRecord, rows: &RowStreams) -> u32 {
                match self {
                    $( Self::$variant => params.$param, )*
                    $( Self::$row => rows.$row_field, )*
                }
            }
        }

        impl StepParamsRecord {
            fn absorb_census(&mut self, census: &Census) {
                $( self.$param = census.$field; )*
            }
        }

        impl From<RowStreams> for RowStreamsRecord {
            fn from(rows: RowStreams) -> Self {
                Self { $( $row_field: rows.$row_field, )* }
            }
        }
    };
}

census! {
    device {
        Bodies: bodies = body_count,
        Dynamic: dynamic_bodies = dynamic_count,
        Colliders: colliders = collider_count,
        Constraints: constraints = constraint_count,
        Particles: particles = particle_count,
        Elements: elements = element_count,
        Attachments: attachments = attachment_count,
        SoftBodies: soft_bodies = soft_body_count,
        Observed: observed = observed_count,
        ObservedJoints: observed_joints = observed_joint_count,
        Characters: characters = character_count,
        Vehicles: vehicles = vehicle_count,
        VehicleWheels: vehicle_wheels = vehicle_wheel_count,
    }
    host {
        body_ids,
        constraint_ids,
        live_colliders,
        movable_colliders,
        immovable_colliders,
        live_characters,
        live_vehicles,
        queries,
        query_hits,
        adjacency,
        body_commands,
        constraint_commands,
        pending_soft_edits,
        pending_soft_body_edits,
        observed_joint_demand,
    }
    rows {
        BodyEditRuns: body_edit_runs,
        BodyMoves: body_moves,
        ConstraintMoves: constraint_moves,
        SoftEdits: soft_edits,
        SoftBodyEdits: soft_body_edits,
    }
}

impl StepParamsRecord {
    pub fn new(config: &PhysicsConfig, dt: f32, census: Census, wake_all: bool) -> Self {
        config.assert_valid();
        domain::positive(dt, "a step timestep");
        let mut record = Self {
            gravity: [config.gravity[0], config.gravity[1], config.gravity[2], 0.0],
            dt,
            substep_dt: dt / config.substeps as f32,
            damping: config.damping,
            angular_damping: config.angular_damping,
            substeps: config.substeps,
            solve_iterations: config.solve_iterations,
            position_iterations: config.position_iterations,
            soft_iterations: config.soft_iterations,
            soft_substeps: config.soft_substeps,
            relaxation: config.relaxation,
            slop: config.slop,
            contact_margin: config.contact_margin,
            restitution_threshold: config.restitution_threshold,
            max_velocity: config.max_velocity,
            max_angular_velocity: config.max_angular_velocity,
            sleep_velocity: config.sleep_velocity,
            sleep_angular_velocity: config.sleep_angular_velocity,
            sleep_time: config.sleep_time,
            friction_combine: combine_code(config.friction_combine),
            restitution_combine: combine_code(config.restitution_combine),
            soft_substep_dt: dt / config.soft_substeps as f32,
            settle_velocity: config.settle_velocity,
            wake_all: u32::from(wake_all),
            ..Self::zeroed()
        };
        record.absorb_census(&census);
        record
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Params(&'static str),
    Rows(&'static str),
}

impl Bound {
    pub fn expression(self) -> String {
        match self {
            Self::Params(field) => format!("params.{field}"),
            Self::Rows(field) => format!("row_streams.{field}"),
        }
    }
}

fn combine_code(combine: MaterialCombine) -> u32 {
    match combine {
        MaterialCombine::Multiply => 0,
        MaterialCombine::Min => 1,
        MaterialCombine::Max => 2,
        MaterialCombine::Average => 3,
    }
}
