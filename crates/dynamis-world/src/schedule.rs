use super::body::BodyStore;
use super::constraint::ConstraintStore;

const NO_ROW: u32 = u32::MAX;
const NO_DEPTH: u32 = u32::MAX;

#[derive(Clone)]
pub(crate) struct JointSchedule {
    pub(crate) pending: bool,
    pub(crate) dirty: bool,
    pub(crate) groups: u32,
    rows: Vec<u32>,
    layers: Vec<u32>,
    group_records: Vec<u32>,
}

impl JointSchedule {
    pub(crate) const fn new() -> Self {
        Self {
            pending: true,
            dirty: true,
            groups: 0,
            rows: Vec::new(),
            layers: Vec::new(),
            group_records: Vec::new(),
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.pending = true;
    }

    pub(crate) fn publish(&mut self) {
        self.dirty = true;
    }

    pub(crate) fn published(&mut self) {
        self.dirty = false;
    }

    pub(crate) fn stale(&self) -> bool {
        self.pending
    }

    pub(crate) fn rows(&self) -> &[u32] {
        &self.rows
    }

    pub(crate) fn layers(&self) -> &[u32] {
        &self.layers
    }

    pub(crate) fn group_records(&self) -> &[u32] {
        &self.group_records
    }

    pub(crate) fn rebuild(&mut self, constraints: &ConstraintStore, bodies: &BodyStore) {
        let (groups, rows, layers, group_records) = solve_order(constraints, bodies);
        self.pending = false;
        self.dirty = true;
        self.groups = groups;
        self.rows = rows;
        self.layers = layers;
        self.group_records = group_records;
    }
}

/// The order the whole joint graph is solved in. A group is a connected component of the joint
/// graph, and a layer holds joints that touch disjoint bodies and that hang off the layers before
/// it, so one ordered sweep of a group carries an impulse the whole length of its chains while the
/// joints of one layer are solved together. The schedule is a pure derivation of the authored
/// topology: bodies and joints are its only inputs, and it is derived again whenever either moves.
fn solve_order(
    constraints: &ConstraintStore,
    bodies: &BodyStore,
) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    let joints = constraints.pool.len();
    if joints == 0 {
        return (0, Vec::new(), Vec::new(), Vec::new());
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
    let groups: Vec<u32> = ends
        .iter()
        .map(|(first, _)| find(&mut roots, *first))
        .collect();
    let mut group_holds_anchor: Vec<bool> = vec![false; body_rows as usize];
    let mut group_head: Vec<u32> = vec![NO_ROW; body_rows as usize];
    for (index, (first, second)) in ends.iter().enumerate() {
        let root = groups[index] as usize;
        group_holds_anchor[root] |= anchor[*first as usize] || anchor[*second as usize];
        let held = group_head[root];
        let lowest = (*first).min(*second).min(held);
        group_head[root] = lowest;
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
        if group_head[root] != NO_ROW && !group_holds_anchor[root] {
            depth[group_head[root] as usize] = 0;
            frontier.push(group_head[root]);
        }
    }
    let mut adjacency: Vec<Vec<(u32, u32)>> = vec![Vec::new(); body_rows as usize];
    for (row, (first, second)) in ends.iter().enumerate() {
        adjacency[*first as usize].push((*second, row as u32));
        adjacency[*second as usize].push((*first, row as u32));
    }
    let mut head = 0usize;
    while head < frontier.len() {
        let body = frontier[head];
        head += 1;
        for (neighbor, _) in &adjacency[body as usize] {
            if depth[*neighbor as usize] == NO_DEPTH {
                depth[*neighbor as usize] = depth[body as usize] + 1;
                frontier.push(*neighbor);
            }
        }
    }
    let mut settled: Vec<u32> = (0..joints).collect();
    settled.sort_by_key(|row| {
        let (first, second) = ends[*row as usize];
        (
            depth[first as usize].max(depth[second as usize]),
            groups[*row as usize],
            *row,
        )
    });
    let mut held_layer: Vec<u32> = vec![0; body_rows as usize];
    let mut order: Vec<(u32, u32, u32)> = Vec::with_capacity(joints as usize);
    for row in settled {
        let (first, second) = ends[row as usize];
        let layer = held_layer[first as usize].max(held_layer[second as usize]) + 1;
        held_layer[first as usize] = layer;
        held_layer[second as usize] = layer;
        order.push((groups[row as usize], layer, row));
    }
    order.sort_unstable();
    let mut rows: Vec<u32> = Vec::with_capacity(joints as usize);
    let mut layer_records: Vec<u32> = Vec::new();
    let mut group_records: Vec<u32> = Vec::new();
    let mut index = 0usize;
    while index < order.len() {
        let group = order[index].0;
        let group_first_layer = layer_records.len() as u32 / 2;
        while index < order.len() && order[index].0 == group {
            let layer = order[index].1;
            let first = rows.len() as u32;
            while index < order.len() && order[index].0 == group && order[index].1 == layer {
                rows.push(order[index].2);
                index += 1;
            }
            layer_records.push(first);
            layer_records.push(rows.len() as u32 - first);
        }
        group_records.push(group_first_layer);
        group_records.push(layer_records.len() as u32 / 2 - group_first_layer);
    }
    (
        group_records.len() as u32 / 2,
        rows,
        layer_records,
        group_records,
    )
}

fn find(roots: &mut [u32], mut row: u32) -> u32 {
    while roots[row as usize] != row {
        roots[row as usize] = roots[roots[row as usize] as usize];
        row = roots[row as usize];
    }
    row
}
