use crate::World;
use dynamis_abi::{COUNTER_ACTIVE, FrameCounts, RowStreams, StepParamsRecord};
use dynamis_broadphase::BroadphaseFrame;
use dynamis_rigid::{RigidFrame, RigidShape};
use dynamis_soft::SoftFrame;

pub(crate) struct StepFrames {
    pub(crate) params: StepParamsRecord,
    pub(crate) rigid: RigidFrame,
    pub(crate) broadphase: BroadphaseFrame,
    pub(crate) soft: SoftFrame,
}

impl World {
    pub(crate) fn frame_counts(&self) -> FrameCounts {
        FrameCounts {
            dynamic_bodies: self.bodies.dynamic_count as u32,
            bodies: self.bodies.alive.len() as u32,
            colliders: self.colliders.used(),
            constraints: self.constraints.alive.len() as u32,
            particles: self.soft.used().0,
            elements: self.soft.used().1,
        }
    }

    pub(crate) fn simulating(&self, query_count: u32) -> bool {
        let commands = self.bodies.last_edits > 0
            || self.bodies.last_moves > 0
            || self.constraints.last_commands > 0
            || self.constraints.last_moves > 0;
        self.backend.measured_step.is_none()
            || self.backend.measured[COUNTER_ACTIVE] != 0
            || commands
            || query_count > 0
            || self.shapes.uploaded
            || self.soft.count() > 0
    }

    pub(crate) fn frames(&self, dt: f32, query_count: u32) -> StepFrames {
        let counts = self.frame_counts();
        let simulating = self.simulating(query_count);
        let params = StepParamsRecord::new(
            &self.config,
            dt,
            counts,
            RowStreams {
                edit_runs: self.bodies.last_edits,
                body_moves: self.bodies.last_moves,
                constraint_moves: self.constraints.last_moves,
            },
            self.event_slot_of(self.clock.step),
        );
        StepFrames {
            params,
            rigid: RigidFrame {
                params,
                shape: RigidShape::of(&counts),
                query_count,
                simulating,
                ccd: self.ccd_active(),
            },
            broadphase: BroadphaseFrame {
                indexing: simulating,
            },
            soft: SoftFrame {
                params,
                simulating: simulating && self.soft.count() > 0,
                material: self.soft.carries_strength(),
            },
        }
    }
}
