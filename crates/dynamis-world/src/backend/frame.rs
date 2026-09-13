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

    pub(crate) fn frames(&self, dt: f32, query_count: u32) -> StepFrames {
        let counts = self.frame_counts();
        let synced = self.backend.measured_step.filter(|measured| {
            self.backend
                .commanded_step
                .is_none_or(|commanded| commanded <= *measured)
        });
        let awake = synced.map(|_| self.backend.measured[COUNTER_ACTIVE]);
        let simulating =
            awake != Some(0) || counts.constraints > 0 || query_count > 0 || self.soft_active();
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
                simulating: self.soft_active(),
                material: self.soft.carries_strength(),
            },
        }
    }
}
