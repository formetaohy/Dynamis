use dynamis_layout::MAX_CELLS_PER_COLLIDER;
use dynamis_model::MAX_COLLIDERS_PER_BODY;

const SLOTS_HEADROOM: u32 = 2;
const COMMANDS_PER_BODY: u32 = 4;
const QUERIES_PER_BODY: u32 = 2;
const MIN_SLOTS: u32 = 64;

pub(crate) struct Live {
    pub bodies: u32,
    pub constraints: u32,
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
}

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

fn product(left: u32, right: u32, name: &str) -> u32 {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("{name} capacity exceeds the device index space"))
}

fn grown(current: u32, required: u32) -> u32 {
    current.max(required).max(MIN_SLOTS)
}

fn stepped(current: u32, live: u32) -> u32 {
    if live <= current {
        return current.max(MIN_SLOTS);
    }
    live.max(current.saturating_mul(SLOTS_HEADROOM))
        .max(MIN_SLOTS)
}

impl Reservation {
    pub(crate) fn initial(pairs_per_body: u32, events_per_body: u32) -> Self {
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
            pairs_per_body,
            events_per_body,
        )
    }

    pub(crate) fn planned(
        current: &Self,
        live: &Live,
        pairs_per_body: u32,
        events_per_body: u32,
    ) -> Self {
        let bodies = stepped(current.bodies, live.bodies);
        let constraints = stepped(current.constraints, live.constraints);
        let colliders = product(bodies, MAX_COLLIDERS_PER_BODY as u32, "collider");
        let entries = product(colliders, MAX_CELLS_PER_COLLIDER, "grid entry");
        let pairs = product(bodies, pairs_per_body, "pair");
        let events = product(bodies, events_per_body, "event");
        Self {
            bodies,
            constraints,
            entries: grown(current.entries, entries),
            pairs: grown(current.pairs, pairs),
            events: grown(current.events, events),
            body_commands: grown(
                current.body_commands,
                live.body_commands
                    .max(product(bodies, COMMANDS_PER_BODY, "body command")),
            ),
            constraint_commands: grown(
                current.constraint_commands,
                live.constraint_commands.max(product(
                    constraints,
                    COMMANDS_PER_BODY,
                    "constraint command",
                )),
            ),
            queries: grown(
                current.queries,
                live.queries.max(product(bodies, QUERIES_PER_BODY, "query")),
            ),
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

    pub(crate) fn planned(current: &Self, used: &Self) -> Self {
        Self {
            sources: grown(current.sources, used.sources),
            vertices: grown(current.vertices, used.vertices),
            triangles: grown(current.triangles, used.triangles),
            nodes: grown(current.nodes, used.nodes),
        }
    }
}
