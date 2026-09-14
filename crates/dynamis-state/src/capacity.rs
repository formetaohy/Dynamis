use super::streams::{StateDemand, StateStreams};
use dynamis_domain::{MIN_SLOTS, STREAM_FLOOR, grown, settled};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeCapacity {
    pub sources: u32,
    pub vertices: u32,
    pub triangles: u32,
    pub nodes: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateCapacity {
    pub shapes: ShapeCapacity,
    pub observed: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct StateInputs {
    pub bodies: u32,
    pub body_ids: u32,
    pub collider_pool: u32,
    pub constraints: u32,
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub queries: u32,
    pub shapes: ShapeCapacity,
    pub observed: u32,
}

pub fn capacity(streams: &StateStreams) -> StateCapacity {
    StateCapacity {
        shapes: ShapeCapacity {
            sources: streams.shape_sources.slots(),
            vertices: streams.shape_vertices.slots(),
            triangles: streams.shape_triangles.slots(),
            nodes: streams.shape_nodes.slots(),
        },
        observed: streams.observed_ids.slots(),
    }
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
        observed: MIN_SLOTS,
    }
}

pub fn plan(inputs: &StateInputs, idle: bool, current: &StateStreams) -> StateDemand {
    let bodies = grown(current.body_states.slots(), inputs.bodies, MIN_SLOTS);
    let body_ids = current
        .body_row_of_id
        .slots()
        .max(inputs.body_ids)
        .max(MIN_SLOTS);
    let colliders = current
        .collider_owners
        .slots()
        .max(inputs.collider_pool)
        .max(MIN_SLOTS);
    let constraints = settled(
        idle,
        current.constraint_runtime.slots(),
        inputs.constraints,
        MIN_SLOTS,
    );
    let body_commands = settled(
        idle,
        current.body_edits.slots(),
        inputs.body_commands,
        STREAM_FLOOR,
    );
    let constraint_commands = settled(
        idle,
        current.constraint_fresh_rows.slots(),
        inputs.constraint_commands,
        STREAM_FLOOR,
    );
    let queries = settled(
        idle,
        current.query_records.slots(),
        inputs.queries,
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
                inputs.shapes.sources,
                MIN_SLOTS,
            ),
            vertices: grown(
                current.shape_vertices.slots(),
                inputs.shapes.vertices,
                MIN_SLOTS,
            ),
            triangles: grown(
                current.shape_triangles.slots(),
                inputs.shapes.triangles,
                MIN_SLOTS,
            ),
            nodes: grown(current.shape_nodes.slots(), inputs.shapes.nodes, MIN_SLOTS),
        },
        observed: grown(current.observed_ids.slots(), inputs.observed, MIN_SLOTS),
    }
}
