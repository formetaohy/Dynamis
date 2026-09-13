struct ReachCells {
    min: vec3i,
    max: vec3i,
}

const REACH_CELL_BUDGET: u32 = 4096u;

fn reach_box(center: vec3f, reach: f32) -> Aabb {
    var box: Aabb;
    box.min = center - vec3f(reach);
    box.max = center + vec3f(reach);
    return box;
}

fn reach_cells(box: Aabb, cell_size: f32) -> ReachCells {
    var cells: ReachCells;
    cells.min = vec3i(floor(box.min / cell_size));
    cells.max = vec3i(floor(box.max / cell_size));
    return cells;
}

fn reach_span(cells: ReachCells) -> u32 {
    let span = cells.max - cells.min + vec3i(1);
    return u32(span.x) * u32(span.y) * u32(span.z);
}

fn reach_overlap_cell(box: Aabb, other: Aabb, cell_size: f32) -> vec3i {
    return vec3i(floor(max(box.min, other.min) / cell_size));
}

fn reach_holds(node: u32, box: Aabb, cell_size: f32) -> bool {
    return all(entry_cell(node, cell_size) == reach_overlap_cell(box, entry_box(node), cell_size));
}
