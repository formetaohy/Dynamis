use dynamis_layout::{
    COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_PAIRS, COUNTER_SPILLOVER_ENTRIES,
    COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_PAIRS, COUNTER_SPILLOVER_RESTING, Counters,
    MAX_CELLS_PER_COLLIDER,
};
use dynamis_model::MAX_COLLIDERS_PER_BODY;

const SLOTS_HEADROOM: u32 = 2;
const COMMANDS_PER_BODY: u32 = 4;
const QUERIES_PER_BODY: u32 = 2;
const MIN_SLOTS: u32 = 64;
const STREAM_DENSITY_PAIRS: u32 = 128;
const STREAM_DENSITY_EVENTS: u32 = 8;

pub(crate) const STREAM_FLOOR: u32 = 256;

pub(crate) struct Live {
    pub(crate) bodies: u32,
    pub(crate) constraints: u32,
    pub(crate) body_commands: u32,
    pub(crate) constraint_commands: u32,
    pub(crate) queries: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StreamDemand {
    pub(crate) pairs: u32,
    pub(crate) entries: u32,
    pub(crate) events: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapacityPlan {
    Widen(StreamDemand),
    Narrow(StreamDemand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reservation {
    pub(crate) bodies: u32,
    pub(crate) constraints: u32,
    pub(crate) entries: u32,
    pub(crate) pairs: u32,
    pub(crate) events: u32,
    pub(crate) body_commands: u32,
    pub(crate) constraint_commands: u32,
    pub(crate) queries: u32,
}

fn product(left: u32, right: u32, name: &str) -> u32 {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("{name} capacity exceeds the device index space"))
}

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

fn stream_grown(current: u32, required: u32, demand: u32) -> u32 {
    current.max(required).max(demand).max(STREAM_FLOOR)
}

fn stream_narrowed(current: u32, target: u32) -> u32 {
    let half = current.saturating_div(2);
    current.min(target.max(half).max(STREAM_FLOOR))
}

impl Reservation {
    pub(crate) fn initial() -> Self {
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

    pub(crate) fn planned(current: &Self, live: &Live, plan: Option<CapacityPlan>) -> Self {
        let widen = plan
            .as_ref()
            .is_none_or(|plan| matches!(plan, CapacityPlan::Widen(_)));
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
            Some(CapacityPlan::Widen(demand)) | Some(CapacityPlan::Narrow(demand)) => demand,
            None => StreamDemand {
                pairs: 0,
                entries: 0,
                events: 0,
            },
        };

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

    pub(crate) fn streams(&self) -> StreamCapacity {
        StreamCapacity {
            entries: self.entries,
            pairs: self.pairs,
            events: self.events,
        }
    }

    pub(crate) fn colliders(&self) -> u32 {
        product(self.bodies, MAX_COLLIDERS_PER_BODY as u32, "collider")
    }

    pub(crate) fn sort(&self) -> u32 {
        self.entries.max(self.pairs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamCapacity {
    pub entries: u32,
    pub pairs: u32,
    pub events: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShapeReservation {
    pub(crate) sources: u32,
    pub(crate) vertices: u32,
    pub(crate) triangles: u32,
    pub(crate) nodes: u32,
}

impl ShapeReservation {
    pub(crate) const EMPTY: Self = Self {
        sources: 0,
        vertices: 0,
        triangles: 0,
        nodes: 0,
    };

    pub(crate) fn planned(current: &Self, used: &Self) -> Self {
        Self {
            sources: grown(current.sources, used.sources),
            vertices: grown(current.vertices, used.vertices),
            triangles: grown(current.triangles, used.triangles),
            nodes: grown(current.nodes, used.nodes),
        }
    }
}

const PRESSURE_HEADROOM: u32 = 8;

const IDLE_FRACTION: u32 = 4;

const IDLE_DELAY: u32 = 120;

const PLAN_COOLDOWN: u32 = 10;

const STREAM_HEADROOM: u32 = 2;

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

pub(crate) struct Capacity {
    pairs: StreamWatch,
    entries: StreamWatch,
    events: StreamWatch,
    cooldown: u32,
}

impl Default for Capacity {
    fn default() -> Self {
        Self::new()
    }
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

    pub(crate) fn observe(
        &mut self,
        measured: &Counters,
        plan: &Reservation,
    ) -> Option<CapacityPlan> {
        self.pairs.observe(
            measured[COUNTER_PAIRS],
            measured[COUNTER_SPILLOVER_PAIRS] > 0 || measured[COUNTER_SPILLOVER_RESTING] > 0,
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
        let under_pressure = (self.pairs.pressure || self.entries.pressure || self.events.pressure)
            && self.cooldown == 0;
        let idle = self.cooldown == 0
            && self.pairs.idle_steps >= IDLE_DELAY
            && self.entries.idle_steps >= IDLE_DELAY
            && self.events.idle_steps >= IDLE_DELAY;
        if !under_pressure && !idle {
            return None;
        }
        self.pairs = StreamWatch::IDLE;
        self.entries = StreamWatch::IDLE;
        self.events = StreamWatch::IDLE;
        self.cooldown = PLAN_COOLDOWN;
        let demand = StreamDemand {
            pairs: serve(measured[COUNTER_PAIRS]),
            entries: serve(measured[COUNTER_ENTRIES]),
            events: serve(measured[COUNTER_EVENTS]),
        };
        if under_pressure {
            Some(CapacityPlan::Widen(demand))
        } else {
            Some(CapacityPlan::Narrow(demand))
        }
    }
}

fn serve(demand: u32) -> u32 {
    demand.saturating_mul(STREAM_HEADROOM).max(STREAM_FLOOR)
}
