//! How much device storage the world needs, planned from what is live and from what
//! the device measured on the previous step — never from the world's ceiling.

use dynamis_layout::{COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_PAIRS, Counters};
use dynamis_model::MAX_COLLIDERS_PER_BODY;

/// Headroom kept over a stream's last measured demand, so ordinary frame-to-frame
/// growth never costs a reallocation.
const MEASURED_HEADROOM: u32 = 2;
/// Growth applied to the body slot tables once live bodies outgrow them.
const SLOTS_HEADROOM: u32 = 2;
/// Grid cells guaranteed per live collider before the device has measured anything.
const ENTRIES_PER_COLLIDER: u32 = 2;
/// Contact events guaranteed per live body.
const EVENTS_PER_BODY: u32 = 4;
/// Body commands guaranteed per live body.
const COMMANDS_PER_BODY: u32 = 4;
/// Queries guaranteed per live body.
const QUERIES_PER_BODY: u32 = 2;
/// The smallest slot table worth building.
const MIN_SLOTS: u32 = 64;

/// What the host is about to hand the device this step.
pub(crate) struct Live {
    pub bodies: u32,
    pub constraints: u32,
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
}

/// The device footprint of one world, in elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reservation {
    pub bodies: u32,
    pub constraints: u32,
    pub entries: u32,
    pub pairs: u32,
    pub events: u32,
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
}

/// A plan only ever grows: shrinking would free storage the next step needs again,
/// and a reservation that oscillates reallocates the world every frame.
fn grown(current: u32, live: u32) -> u32 {
    current.max(live).max(MIN_SLOTS)
}

fn grown_from(current: u32, floor: u32, observed: u32) -> u32 {
    current
        .max(floor)
        .max(observed.saturating_mul(MEASURED_HEADROOM))
}

/// The reservation a stream takes once live bodies outgrow it.
fn stepped(current: u32, live: u32) -> u32 {
    if live <= current {
        return current.max(MIN_SLOTS);
    }
    live.max(current.saturating_mul(SLOTS_HEADROOM))
        .max(MIN_SLOTS)
}

impl Reservation {
    pub(crate) fn initial(bodies: u32, pairs_per_body: u32) -> Self {
        let bodies = bodies.max(MIN_SLOTS);
        Self::planned(
            &Self {
                bodies: 0,
                constraints: 0,
                entries: 0,
                pairs: 0,
                events: 0,
                body_commands: 0,
                constraint_commands: 0,
                queries: 0,
            },
            &Live {
                bodies,
                constraints: 0,
                body_commands: 0,
                constraint_commands: 0,
                queries: 0,
            },
            &[0; dynamis_layout::COUNTER_COUNT],
            pairs_per_body,
        )
    }

    /// The next plan, given what is live now and what the device measured last step.
    pub(crate) fn planned(
        current: &Self,
        live: &Live,
        observed: &Counters,
        pairs_per_body: u32,
    ) -> Self {
        let bodies = stepped(current.bodies, live.bodies);
        let constraints = stepped(current.constraints, live.constraints);
        let colliders = bodies * MAX_COLLIDERS_PER_BODY as u32;
        Self {
            bodies,
            constraints,
            entries: grown_from(
                current.entries,
                ENTRIES_PER_COLLIDER * colliders,
                observed[COUNTER_ENTRIES],
            ),
            pairs: grown_from(
                current.pairs,
                pairs_per_body * bodies,
                observed[COUNTER_PAIRS],
            ),
            events: grown_from(
                current.events,
                EVENTS_PER_BODY * bodies,
                observed[COUNTER_EVENTS],
            ),
            body_commands: grown_from(
                current.body_commands,
                COMMANDS_PER_BODY * bodies,
                live.body_commands,
            ),
            constraint_commands: grown_from(
                current.constraint_commands,
                COMMANDS_PER_BODY * constraints,
                live.constraint_commands,
            ),
            queries: grown_from(current.queries, QUERIES_PER_BODY * bodies, live.queries),
        }
    }

    pub(crate) fn colliders(&self) -> u32 {
        self.bodies * MAX_COLLIDERS_PER_BODY as u32
    }

    /// The widest lane set any sort in this world will face.
    pub(crate) fn sort(&self) -> u32 {
        self.entries.max(self.pairs)
    }
}

/// The device footprint of the shape store, in elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShapeReservation {
    pub sources: u32,
    pub vertices: u32,
    pub triangles: u32,
    pub nodes: u32,
}

impl ShapeReservation {
    pub(crate) const EMPTY: Self = Self {
        sources: 0,
        vertices: 0,
        triangles: 0,
        nodes: 0,
    };

    /// The next plan, given the packed lengths the shape store has actually reached.
    pub(crate) fn planned(current: &Self, used: &Self) -> Self {
        Self {
            sources: grown(current.sources, used.sources),
            vertices: grown(current.vertices, used.vertices),
            triangles: grown(current.triangles, used.triangles),
            nodes: grown(current.nodes, used.nodes),
        }
    }
}
