fn reach_box(center: vec3f, reach: f32) -> Aabb {
    var box: Aabb;
    box.min = center - vec3f(reach);
    box.max = center + vec3f(reach);
    return box;
}

fn reach_holds(node: u32, box: Aabb, cell_size: f32) -> bool {
    return all(entry_cell(node, cell_size) == grid_overlap_cell(box, entry_box(node), cell_size));
}
