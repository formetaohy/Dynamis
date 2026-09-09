//! The capacity controller: what the device measured becomes the next stream plan.
//!
//! Stream lanes are sized by demand, never declared by host code. A step that spilled
//! or ran near its ceiling latches pressure, and the next step boundary serves it with
//! a wider plan. Sustained idleness narrows a stream back toward what it actually
//! needs, so a one-frame pile-up does not bill its peak forever.

use crate::reservation::{Reservation, StreamDemand};
use dynamis_layout::{
    COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_PAIRS, COUNTER_SPILLOVER_ENTRIES,
    COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_PAIRS, Counters,
};

/// The fraction of lanes a stream must keep free; less latches pressure ahead of a
/// spill.
const PRESSURE_HEADROOM: u32 = 8;
/// A stream only counts as idle while demand stays under this fraction of its lanes.
const IDLE_FRACTION: u32 = 4;
/// Steps of idleness required before a stream narrows.
const IDLE_DELAY: u32 = 120;
/// Steps between two plans, so a spike cannot oscillate reallocations.
const PLAN_COOLDOWN: u32 = 10;
/// What a plan serves a stream: twice what the device measured it to need.
const STREAM_HEADROOM: u32 = 2;
/// Streams never drop below this lane count.
const MIN_STREAM_LANES: u32 = 256;

#[derive(Clone, Copy)]
struct StreamWatch {
    pressure: bool,
    idle_steps: u32,
}

impl StreamWatch {
    const IDLE: Self = Self {
        pressure: false,
        idle_steps: 0,
    };

    fn observe(&mut self, demand: u32, spilled: bool, lanes: u32) {
        let free = lanes.saturating_sub(demand);
        if spilled || free < lanes / PRESSURE_HEADROOM {
            self.pressure = true;
            self.idle_steps = 0;
        } else if demand * IDLE_FRACTION < lanes {
            self.idle_steps = self.idle_steps.saturating_add(1);
        } else {
            self.idle_steps = 0;
        }
    }
}

/// The control loop between what the device measured and the next plan.
pub(crate) struct Capacity {
    pairs: StreamWatch,
    entries: StreamWatch,
    events: StreamWatch,
    cooldown: u32,
}

impl Capacity {
    pub(crate) const fn new() -> Self {
        Self {
            pairs: StreamWatch::IDLE,
            entries: StreamWatch::IDLE,
            events: StreamWatch::IDLE,
            cooldown: 0,
        }
    }

    /// Takes what the device measured on a finished step and returns the demand the
    /// next plan must serve, or nothing while the current plan suffices.
    pub(crate) fn observe(
        &mut self,
        measured: &Counters,
        plan: &Reservation,
    ) -> Option<StreamDemand> {
        self.pairs.observe(
            measured[COUNTER_PAIRS],
            measured[COUNTER_SPILLOVER_PAIRS] > 0,
            plan.pairs,
        );
        self.entries.observe(
            measured[COUNTER_ENTRIES],
            measured[COUNTER_SPILLOVER_ENTRIES] > 0,
            plan.entries,
        );
        self.events.observe(
            measured[COUNTER_EVENTS],
            measured[COUNTER_SPILLOVER_EVENTS] > 0,
            plan.events,
        );
        self.cooldown = self.cooldown.saturating_sub(1);
        let pressed = (self.pairs.pressure || self.entries.pressure || self.events.pressure)
            && self.cooldown == 0;
        let idle = self.cooldown == 0
            && self.pairs.idle_steps >= IDLE_DELAY
            && self.entries.idle_steps >= IDLE_DELAY
            && self.events.idle_steps >= IDLE_DELAY;
        if !pressed && !idle {
            return None;
        }
        self.pairs = StreamWatch::IDLE;
        self.entries = StreamWatch::IDLE;
        self.events = StreamWatch::IDLE;
        self.cooldown = PLAN_COOLDOWN;
        Some(StreamDemand {
            pairs: serve(measured[COUNTER_PAIRS]),
            entries: serve(measured[COUNTER_ENTRIES]),
            events: serve(measured[COUNTER_EVENTS]),
        })
    }
}

fn serve(demand: u32) -> u32 {
    demand.saturating_mul(STREAM_HEADROOM).max(MIN_STREAM_LANES)
}
