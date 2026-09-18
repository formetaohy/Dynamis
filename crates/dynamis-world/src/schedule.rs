use super::body::BodyStore;
use super::constraint::ConstraintStore;
use crate::derivation::{JointOrder, JointOrderStorage, SceneFacts};
use dynamis_rigid::JOINT_BATCH_LANES;
use std::cmp::Reverse;

const NO_ROW: u32 = u32::MAX;
const NO_DEPTH: u32 = u32::MAX;
const BATCH_ROOM_LIMIT: u32 = 16;

#[derive(Clone)]
pub(crate) struct JointSchedule {
    pub(crate) batches: u32,
    rows: Vec<u32>,
    layers: Vec<u32>,
    components: Vec<u32>,
    packing: Vec<u32>,
    derived: Option<JointOrder>,
    uploaded: Option<JointOrderStorage>,
}

impl JointSchedule {
    pub(crate) const fn new() -> Self {
        Self {
            batches: 0,
            rows: Vec::new(),
            layers: Vec::new(),
            components: Vec::new(),
            packing: Vec::new(),
            derived: None,
            uploaded: None,
        }
    }

    pub(crate) fn stale(&self, facts: &SceneFacts) -> bool {
        self.derived != Some(JointOrder::of(facts))
    }

    pub(crate) fn owes_upload(&self, storage: JointOrderStorage) -> bool {
        self.uploaded != Some(storage)
    }

    pub(crate) fn uploaded(&mut self, storage: JointOrderStorage) {
        self.uploaded = Some(storage);
    }

    pub(crate) fn rows(&self) -> &[u32] {
        &self.rows
    }

    pub(crate) fn layers(&self) -> &[u32] {
        &self.layers
    }

    pub(crate) fn components(&self) -> &[u32] {
        &self.components
    }

    pub(crate) fn packing(&self) -> &[u32] {
        &self.packing
    }

    pub(crate) fn rebuild(
        &mut self,
        constraints: &ConstraintStore,
        bodies: &BodyStore,
        facts: &SceneFacts,
    ) {
        let plan = solve_order(constraints, bodies);
        self.batches = plan.batches;
        self.rows = plan.rows;
        self.layers = plan.layers;
        self.components = plan.components;
        self.packing = plan.packing;
        self.derived = Some(JointOrder::of(facts));
        self.uploaded = None;
    }
}

struct Plan {
    batches: u32,
    rows: Vec<u32>,
    layers: Vec<u32>,
    components: Vec<u32>,
    packing: Vec<u32>,
}

struct Component {
    first_layer: u32,
    layers: u32,
    first_row: u32,
    head: u32,
    lanes: u32,
    room: u32,
}

/// The order the whole joint graph is solved in. A component is a connected component of the joint
/// graph, a layer holds joints that touch disjoint bodies, and a layer follows every layer it hangs
/// off, so one ordered sweep of a component carries an impulse the whole length of its chains while
/// the joints of one layer are solved together. Components are packed into workgroup batches by the
/// width of their widest layer and the depth of their layer stack, so that a lane owns one component
/// and a workgroup holds as much independent work as the machine can hide latency behind. The
/// schedule is a pure derivation of the authored topology: bodies and joints are its only inputs,
/// and it is derived again whenever either moves.
fn solve_order(constraints: &ConstraintStore, bodies: &BodyStore) -> Plan {
    let joints = constraints.pool.len();
    if joints == 0 {
        return Plan {
            batches: 0,
            rows: Vec::new(),
            layers: Vec::new(),
            components: Vec::new(),
            packing: Vec::new(),
        };
    }
    let body_rows = bodies.pool.len();
    let mut ends: Vec<(u32, u32)> = Vec::with_capacity(joints as usize);
    let mut anchor: Vec<bool> = vec![false; body_rows as usize];
    let mut roots: Vec<u32> = (0..body_rows).collect();
    for row in 0..joints {
        let handle = constraints.pool.handle_of_row(row);
        let joint = &constraints.joints[handle.id as usize];
        let first = bodies.pool.row_of_id(joint.first);
        let second = bodies.pool.row_of_id(joint.second);
        assert!(
            first != NO_ROW && second != NO_ROW,
            "a joint row must join two live bodies",
        );
        ends.push((first, second));
        anchor[first as usize] |= bodies.records[joint.first as usize].inverse_mass == 0.0;
        anchor[second as usize] |= bodies.records[joint.second as usize].inverse_mass == 0.0;
        let first_root = find(&mut roots, first);
        let second_root = find(&mut roots, second);
        let root = first_root.min(second_root);
        roots[first_root as usize] = root;
        roots[second_root as usize] = root;
    }
    let roots_of = ends
        .iter()
        .map(|(first, _)| find(&mut roots, *first))
        .collect::<Vec<_>>();
    let mut anchored: Vec<bool> = vec![false; body_rows as usize];
    let mut head_of: Vec<u32> = vec![NO_ROW; body_rows as usize];
    for (index, (first, second)) in ends.iter().enumerate() {
        let root = roots_of[index] as usize;
        anchored[root] |= anchor[*first as usize] || anchor[*second as usize];
        head_of[root] = head_of[root].min(*first).min(*second);
    }
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); body_rows as usize];
    for (first, second) in &ends {
        adjacency[*first as usize].push(*second);
        adjacency[*second as usize].push(*first);
    }
    let mut depth: Vec<u32> = vec![NO_DEPTH; body_rows as usize];
    let mut frontier: Vec<u32> = Vec::new();
    for body in 0..body_rows {
        if anchor[body as usize] {
            depth[body as usize] = 0;
            frontier.push(body);
        }
    }
    for root in 0..body_rows as usize {
        if head_of[root] != NO_ROW && !anchored[root] {
            depth[head_of[root] as usize] = 0;
            frontier.push(head_of[root]);
        }
    }
    let mut cursor = 0usize;
    while cursor < frontier.len() {
        let body = frontier[cursor];
        cursor += 1;
        for neighbor in &adjacency[body as usize] {
            if depth[*neighbor as usize] == NO_DEPTH {
                depth[*neighbor as usize] = depth[body as usize] + 1;
                frontier.push(*neighbor);
            }
        }
    }
    let mut settled: Vec<u32> = (0..joints).collect();
    settled.sort_by_key(|row| {
        let (first, second) = ends[*row as usize];
        let root = roots_of[*row as usize];
        (
            depth[first as usize].max(depth[second as usize]),
            root,
            *row,
        )
    });
    let mut held_layer: Vec<u32> = vec![0; body_rows as usize];
    let mut held: Vec<(u32, u32, u32)> = Vec::with_capacity(joints as usize);
    for row in settled {
        let (first, second) = ends[row as usize];
        let layer = held_layer[first as usize].max(held_layer[second as usize]) + 1;
        held_layer[first as usize] = layer;
        held_layer[second as usize] = layer;
        held.push((roots_of[row as usize], layer, row));
    }
    held.sort_unstable();
    let mut rows: Vec<u32> = Vec::with_capacity(joints as usize);
    let mut layers: Vec<u32> = Vec::new();
    let mut components: Vec<Component> = Vec::new();
    let mut index = 0usize;
    while index < held.len() {
        let root = held[index].0;
        let first_layer = (layers.len() / 2) as u32;
        let first_row = rows.len() as u32;
        let mut head = NO_ROW;
        let mut depth = 0u32;
        let mut lanes = 0u32;
        while index < held.len() && held[index].0 == root {
            let layer = held[index].1;
            let first = rows.len();
            while index < held.len() && held[index].0 == root && held[index].1 == layer {
                let row = held[index].2;
                let (first_body, second_body) = ends[row as usize];
                head = head.min(first_body).min(second_body);
                rows.push(row);
                index += 1;
            }
            let count = rows.len() as u32 - first as u32;
            lanes = lanes.max(count);
            layers.push(first as u32);
            layers.push(count);
            depth += 1;
        }
        let lanes = lanes.next_power_of_two().clamp(1, JOINT_BATCH_LANES);
        components.push(Component {
            first_layer,
            layers: depth,
            first_row,
            head,
            lanes,
            room: batch_room(lanes, depth),
        });
    }
    components.sort_by_key(|component| {
        (
            Reverse(component.lanes),
            Reverse(component.layers),
            component.head,
        )
    });
    let mut plan = Plan {
        batches: 0,
        rows: Vec::with_capacity(rows.len()),
        layers: Vec::with_capacity(layers.len()),
        components: Vec::with_capacity(components.len() * 2),
        packing: Vec::new(),
    };
    for component in &components {
        plan.components.push((plan.layers.len() / 2) as u32);
        plan.components.push(component.layers);
        let mut cursor = component.first_row as usize;
        for layer in component.first_layer..component.first_layer + component.layers {
            let count = layers[layer as usize * 2 + 1] as usize;
            plan.layers.push(plan.rows.len() as u32);
            plan.layers.push(count as u32);
            plan.rows.extend_from_slice(&rows[cursor..cursor + count]);
            cursor += count;
        }
    }
    let mut packed = 0usize;
    while packed < components.len() {
        let lanes = components[packed].lanes;
        let room = components[packed].room as usize;
        let mut taken = 1usize;
        let mut span = components[packed].layers;
        while taken + packed < components.len()
            && taken < room
            && components[packed + taken].lanes == lanes
            && components[packed + taken].room as usize == room
        {
            span = span.max(components[packed + taken].layers);
            taken += 1;
        }
        plan.packing.push(packed as u32);
        plan.packing.push(taken as u32);
        plan.packing.push(lanes);
        plan.packing.push(span);
        plan.batches += 1;
        packed += taken;
    }
    plan
}

/// How many components one workgroup solves at once. A component whose layer stack runs deep is a
/// chain of dependent joint solves, and a lane yields that latency only next to other chains, so a
/// deep component shares its workgroup with few peers while a shallow one packs it full. The room
/// never fills the whole workgroup either, so that a schedule keeps enough workgroups for the device
/// to interleave.
fn batch_room(lanes: u32, layers: u32) -> u32 {
    let dependency = lanes.saturating_mul(layers).max(1);
    (JOINT_BATCH_LANES / dependency)
        .min(BATCH_ROOM_LIMIT)
        .clamp(1, JOINT_BATCH_LANES / lanes)
}

fn find(roots: &mut [u32], mut row: u32) -> u32 {
    while roots[row as usize] != row {
        roots[row as usize] = roots[roots[row as usize] as usize];
        row = roots[row as usize];
    }
    row
}
