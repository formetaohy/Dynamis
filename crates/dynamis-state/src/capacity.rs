use super::streams::{StateDemand, StateStreams};
use dynamis_pass::{MIN_SLOTS, STREAM_FLOOR, grown, product, settled};

const COMMANDS_PER_BODY: u32 = 4;
const CONSTRAINT_COMMANDS_PER_CONSTRAINT: u32 = 4;
const QUERIES_PER_BODY: u32 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeCapacity {
    pub sources: u32,
    pub vertices: u32,
    pub triangles: u32,
    pub nodes: u32,
}

pub struct Live {
    pub bodies: u32,
    pub colliders: u32,
    pub collider_pool: u32,
    pub body_ids: u32,
    pub constraints: u32,
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
    pub shapes: ShapeCapacity,
    pub particles: u32,
    pub elements: u32,
    pub adjacency: u32,
}

pub fn floor() -> StateDemand {
    StateDemand {
        bodies: MIN_SLOTS,
        body_ids: MIN_SLOTS,
        colliders: MIN_SLOTS,
        constraints: MIN_SLOTS,
        body_commands: STREAM_FLOOR,
        constraint_commands: STREAM_FLOOR,
        queries: STREAM_FLOOR,
        shapes: ShapeCapacity {
            sources: MIN_SLOTS,
            vertices: MIN_SLOTS,
            triangles: MIN_SLOTS,
            nodes: MIN_SLOTS,
        },
    }
}

pub fn plan(live: &Live, idle: bool, current: &StateStreams) -> StateDemand {
    let bodies = grown(current.body_states.slots(), live.bodies, MIN_SLOTS);
    let body_ids = current
        .body_row_of_id
        .slots()
        .max(live.body_ids)
        .max(MIN_SLOTS);
    let colliders = current
        .collider_owners
        .slots()
        .max(live.collider_pool)
        .max(MIN_SLOTS);
    let constraints = settled(
        idle,
        current.constraint_runtime.slots(),
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
    StateDemand {
        bodies,
        body_ids,
        colliders,
        constraints,
        body_commands,
        constraint_commands,
        queries,
        shapes: ShapeCapacity {
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
        },
    }
}
