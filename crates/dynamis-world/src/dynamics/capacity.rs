use crate::dynamics::buffers::{Demand, ShapeUse, WorldBuffers};
use dynamis_layout::{
    COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_PAIRS, COUNTER_SPILLOVER_ENTRIES,
    COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_PAIRS, COUNTER_SPILLOVER_RESTING, Counters,
    MAX_CELLS_PER_COLLIDER,
};

const MIN_SLOTS: u32 = 64;
const STREAM_FLOOR: u32 = 256;
const SLOTS_HEADROOM: u32 = 2;
const COMMANDS_PER_BODY: u32 = 4;
const CONSTRAINT_COMMANDS_PER_CONSTRAINT: u32 = 4;
const QUERIES_PER_BODY: u32 = 2;
const STREAM_DENSITY_PAIRS: u32 = 16;
const STREAM_DENSITY_EVENTS: u32 = 8;
const STREAM_HEADROOM: u32 = 2;
const PRESSURE_HEADROOM: u32 = 8;
const IDLE_FRACTION: u32 = 4;
const IDLE_DELAY: u32 = 120;
const PLAN_COOLDOWN: u32 = 10;

pub(crate) struct Live {
    pub(crate) bodies: u32,
    pub(crate) colliders: u32,
    pub(crate) collider_pool: u32,
    pub(crate) body_ids: u32,
    pub(crate) constraints: u32,
    pub(crate) body_commands: u32,
    pub(crate) constraint_commands: u32,
    pub(crate) queries: u32,
    pub(crate) shapes: ShapeUse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamCapacity {
    pub entries: u32,
    pub pairs: u32,
    pub events: u32,
}

fn product(left: u32, right: u32, name: &str) -> u32 {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("{name} capacity exceeds the device index space"))
}

fn grown(capacity: u32, required: u32, floor: u32) -> u32 {
    if required <= capacity {
        capacity.max(floor)
    } else {
        required
            .max(capacity.saturating_mul(SLOTS_HEADROOM))
            .max(floor)
    }
}

fn narrowed(capacity: u32, required: u32, floor: u32) -> u32 {
    let target = required.max(floor);
    capacity.min(target.max(capacity / 2))
}

fn settled(idle: bool, capacity: u32, required: u32, floor: u32) -> u32 {
    if idle {
        narrowed(capacity, required, floor)
    } else {
        grown(capacity, required, floor)
    }
}

struct StreamWatch {
    served: u32,
    pressure: bool,
    idle_steps: u32,
}

impl StreamWatch {
    const IDLE: Self = Self {
        served: 0,
        pressure: false,
        idle_steps: 0,
    };

    fn observe(&mut self, observed: u32, spilled: bool, lanes: u32, open: bool) {
        let free = lanes.saturating_sub(observed);
        if spilled || free < lanes / PRESSURE_HEADROOM {
            if open {
                self.served = self.served.max(observed.saturating_mul(STREAM_HEADROOM));
            }
            self.pressure = true;
            self.idle_steps = 0;
        } else if observed.saturating_mul(IDLE_FRACTION) < lanes {
            self.idle_steps = self.idle_steps.saturating_add(1);
        } else {
            self.idle_steps = 0;
        }
    }

    fn released(&self, capacity: u32, budget: u32) -> u32 {
        narrowed(capacity, budget, STREAM_FLOOR)
    }

    fn widened(&self, capacity: u32, budget: u32) -> u32 {
        capacity.max(budget).max(self.served).max(STREAM_FLOOR)
    }

    fn is_idle(&self) -> bool {
        self.idle_steps >= IDLE_DELAY
    }

    fn settle(&mut self, release: bool) {
        if release {
            self.served = 0;
        }
        self.pressure = false;
        self.idle_steps = 0;
    }
}

pub(crate) struct Capacity {
    entries: StreamWatch,
    pairs: StreamWatch,
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
            entries: StreamWatch::IDLE,
            pairs: StreamWatch::IDLE,
            events: StreamWatch::IDLE,
            cooldown: 0,
        }
    }

    pub(crate) fn minimum() -> Demand {
        Demand {
            bodies: MIN_SLOTS,
            body_ids: MIN_SLOTS,
            colliders: MIN_SLOTS,
            constraints: MIN_SLOTS,
            body_commands: STREAM_FLOOR,
            constraint_commands: STREAM_FLOOR,
            queries: STREAM_FLOOR,
            entries: STREAM_FLOOR,
            pairs: STREAM_FLOOR,
            events: STREAM_FLOOR,
            shapes: ShapeUse {
                sources: MIN_SLOTS,
                vertices: MIN_SLOTS,
                triangles: MIN_SLOTS,
                nodes: MIN_SLOTS,
            },
            sort: Demand::sort_slots(STREAM_FLOOR, STREAM_FLOOR, MIN_SLOTS),
        }
    }

    pub(crate) fn demand(
        &mut self,
        measured: &Counters,
        live: &Live,
        current: &WorldBuffers,
    ) -> Demand {
        self.cooldown = self.cooldown.saturating_sub(1);
        let open = self.cooldown == 0;
        self.pairs.observe(
            measured[COUNTER_PAIRS],
            measured[COUNTER_SPILLOVER_PAIRS] > 0 || measured[COUNTER_SPILLOVER_RESTING] > 0,
            current.pair_capacity(),
            open,
        );
        self.entries.observe(
            measured[COUNTER_ENTRIES],
            measured[COUNTER_SPILLOVER_ENTRIES] > 0,
            current.entry_capacity(),
            open,
        );
        self.events.observe(
            measured[COUNTER_EVENTS],
            measured[COUNTER_SPILLOVER_EVENTS] > 0,
            current.event_capacity(),
            open,
        );
        let pressured = self.pairs.pressure || self.entries.pressure || self.events.pressure;
        let idle = open
            && !pressured
            && self.pairs.is_idle()
            && self.entries.is_idle()
            && self.events.is_idle()
            && live.body_commands == 0
            && live.constraint_commands == 0;
        if open && (pressured || idle) {
            self.cooldown = PLAN_COOLDOWN;
            self.entries.settle(idle);
            self.pairs.settle(idle);
            self.events.settle(idle);
        }
        let entry_budget = product(live.colliders, MAX_CELLS_PER_COLLIDER, "grid entry");
        let pair_budget = product(live.colliders, STREAM_DENSITY_PAIRS, "pair");
        let event_budget = product(live.colliders, STREAM_DENSITY_EVENTS, "event");
        let entries = if idle {
            self.entries
                .released(current.entry_capacity(), entry_budget)
        } else {
            self.entries.widened(current.entry_capacity(), entry_budget)
        };
        let pairs = if idle {
            self.pairs.released(current.pair_capacity(), pair_budget)
        } else {
            self.pairs.widened(current.pair_capacity(), pair_budget)
        };
        let events = if idle {
            self.events.released(current.event_capacity(), event_budget)
        } else {
            self.events.widened(current.event_capacity(), event_budget)
        };
        self.live_demand(live, current, idle, entries, pairs, events)
    }

    fn live_demand(
        &self,
        live: &Live,
        current: &WorldBuffers,
        idle: bool,
        entries: u32,
        pairs: u32,
        events: u32,
    ) -> Demand {
        let bodies = grown(current.body_row_count(), live.bodies, MIN_SLOTS);
        let body_ids = current
            .body_row_of_id
            .slots()
            .max(live.body_ids)
            .max(MIN_SLOTS);
        let colliders = current
            .collider_capacity()
            .max(live.collider_pool)
            .max(MIN_SLOTS);
        let constraints = settled(
            idle,
            current.constraint_capacity(),
            live.constraints,
            MIN_SLOTS,
        );
        let body_commands = settled(
            idle,
            current.body_edits.slots(),
            live.body_commands
                .max(product(bodies, COMMANDS_PER_BODY, "body command")),
            STREAM_FLOOR,
        );
        let constraint_commands = settled(
            idle,
            current.constraint_fresh_rows.slots(),
            live.constraint_commands.max(product(
                constraints,
                CONSTRAINT_COMMANDS_PER_CONSTRAINT,
                "constraint command",
            )),
            STREAM_FLOOR,
        );
        let queries = settled(
            idle,
            current.query_records.slots(),
            live.queries.max(product(bodies, QUERIES_PER_BODY, "query")),
            STREAM_FLOOR,
        );
        let shapes = ShapeUse {
            sources: grown(
                current.shape_sources.slots(),
                live.shapes.sources,
                MIN_SLOTS,
            ),
            vertices: grown(
                current.shape_vertices.slots(),
                live.shapes.vertices,
                MIN_SLOTS,
            ),
            triangles: grown(
                current.shape_triangles.slots(),
                live.shapes.triangles,
                MIN_SLOTS,
            ),
            nodes: grown(current.shape_nodes.slots(), live.shapes.nodes, MIN_SLOTS),
        };
        Demand {
            bodies,
            body_ids,
            colliders,
            constraints,
            body_commands,
            constraint_commands,
            queries,
            entries,
            pairs,
            events,
            shapes,
            sort: Demand::sort_slots(entries, pairs, constraints),
        }
    }
}
