//! Stream planning: what the device measured turns into the lane counts the next
//! step allocates.

use dynamis_layout::MAX_CELLS_PER_COLLIDER;
use dynamis_model::MAX_COLLIDERS_PER_BODY;

const SLOTS_HEADROOM: u32 = 2;
const COMMANDS_PER_BODY: u32 = 4;
const QUERIES_PER_BODY: u32 = 2;
const MIN_SLOTS: u32 = 64;
const STREAM_DENSITY_PAIRS: u32 = 128;
const STREAM_DENSITY_EVENTS: u32 = 8;

/// A stream never serves below this many lanes, even when nothing measured wants
/// more: the first step of a plan must not spill before it has measured demand.
pub const STREAM_FLOOR: u32 = 256;

pub struct Live {
    pub bodies: u32,
    pub constraints: u32,
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
}

/// The lane counts the device measured a step to need; the next plan serves them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamDemand {
    pub pairs: u32,
    pub entries: u32,
    pub events: u32,
}

/// One re-planning direction. `Widen` serves a measured peak, `Narrow` releases the
/// headroom a stream has been idling on; both carry the demand the next plan must
/// serve, so a plan is always at least what the device asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamPlan {
    Widen(StreamDemand),
    Narrow(StreamDemand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub bodies: u32,
    pub constraints: u32,
    pub entries: u32,
    pub pairs: u32,
    pub events: u32,
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
}

fn product(left: u32, right: u32, name: &str) -> u32 {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("{name} capacity exceeds the device index space"))
}

/// Serves a live count: reach it now, and keep room for the next batch at the
/// boundary; a shrinking live count is served by the narrow path, never here.
fn grown(current: u32, live: u32) -> u32 {
    if live <= current {
        return current.max(MIN_SLOTS);
    }
    live.max(current.saturating_mul(SLOTS_HEADROOM))
        .max(MIN_SLOTS)
}

fn narrowed(current: u32, live: u32) -> u32 {
    let target = live.max(MIN_SLOTS).saturating_mul(SLOTS_HEADROOM);
    current.min(current.saturating_div(2).max(target))
}

/// Grows a stream to `required`, then to measured demand, never below the floor.
fn stream_grown(current: u32, required: u32, demand: u32) -> u32 {
    current.max(required).max(demand).max(STREAM_FLOOR)
}

/// Releases stream headroom toward `target`, one halving per plan at most, never
/// below the target or the floor.
fn stream_narrowed(current: u32, target: u32) -> u32 {
    let half = current.saturating_div(2);
    current.min(target.max(half).max(STREAM_FLOOR))
}

impl Reservation {
    /// The plan a world starts on: nothing live, nothing measured, minimum streams.
    pub fn initial() -> Self {
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
                bodies: 0,
                constraints: 0,
                body_commands: 0,
                constraint_commands: 0,
                queries: 0,
            },
            None,
        )
    }

    pub fn planned(current: &Self, live: &Live, plan: Option<StreamPlan>) -> Self {
        let widen = plan
            .as_ref()
            .is_none_or(|plan| matches!(plan, StreamPlan::Widen(_)));
        let bodies = if widen {
            grown(current.bodies, live.bodies)
        } else {
            narrowed(current.bodies, live.bodies)
        };
        let constraints = if widen {
            grown(current.constraints, live.constraints)
        } else {
            narrowed(current.constraints, live.constraints)
        };
        let colliders = product(bodies, MAX_COLLIDERS_PER_BODY as u32, "collider");
        let demand = match plan {
            Some(StreamPlan::Widen(demand)) | Some(StreamPlan::Narrow(demand)) => demand,
            None => StreamDemand {
                pairs: 0,
                entries: 0,
                events: 0,
            },
        };
        // Measured streams release headroom toward what the device actually asked
        // for; the density baselines only serve widening, so a measured peak never
        // bills its own forecast for the life of the world.
        let entries = if widen {
            stream_grown(
                current.entries,
                product(colliders, MAX_CELLS_PER_COLLIDER, "grid entry"),
                demand.entries,
            )
        } else {
            stream_narrowed(current.entries, demand.entries)
        };
        let pairs = if widen {
            stream_grown(
                current.pairs,
                product(bodies, STREAM_DENSITY_PAIRS, "pair"),
                demand.pairs,
            )
        } else {
            stream_narrowed(current.pairs, demand.pairs)
        };
        let events = if widen {
            stream_grown(
                current.events,
                product(bodies, STREAM_DENSITY_EVENTS, "event"),
                demand.events,
            )
        } else {
            stream_narrowed(current.events, demand.events)
        };
        let body_commands = if widen {
            stream_grown(
                current.body_commands,
                live.body_commands
                    .max(product(bodies, COMMANDS_PER_BODY, "body command")),
                0,
            )
        } else {
            stream_narrowed(
                current.body_commands,
                live.body_commands
                    .max(product(bodies, COMMANDS_PER_BODY, "body command")),
            )
        };
        let constraint_commands = if widen {
            stream_grown(
                current.constraint_commands,
                live.constraint_commands.max(product(
                    constraints,
                    COMMANDS_PER_BODY,
                    "constraint command",
                )),
                0,
            )
        } else {
            stream_narrowed(
                current.constraint_commands,
                live.constraint_commands.max(product(
                    constraints,
                    COMMANDS_PER_BODY,
                    "constraint command",
                )),
            )
        };
        let queries = if widen {
            stream_grown(
                current.queries,
                live.queries.max(product(bodies, QUERIES_PER_BODY, "query")),
                0,
            )
        } else {
            stream_narrowed(
                current.queries,
                live.queries.max(product(bodies, QUERIES_PER_BODY, "query")),
            )
        };
        Self {
            bodies,
            constraints,
            entries,
            pairs,
            events,
            body_commands,
            constraint_commands,
            queries,
        }
    }

    pub fn colliders(&self) -> u32 {
        product(self.bodies, MAX_COLLIDERS_PER_BODY as u32, "collider")
    }

    pub fn sort(&self) -> u32 {
        self.entries.max(self.pairs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeReservation {
    pub sources: u32,
    pub vertices: u32,
    pub triangles: u32,
    pub nodes: u32,
}

impl ShapeReservation {
    pub const EMPTY: Self = Self {
        sources: 0,
        vertices: 0,
        triangles: 0,
        nodes: 0,
    };

    pub fn planned(current: &Self, used: &Self) -> Self {
        Self {
            sources: grown(current.sources, used.sources),
            vertices: grown(current.vertices, used.vertices),
            triangles: grown(current.triangles, used.triangles),
            nodes: grown(current.nodes, used.nodes),
        }
    }
}
