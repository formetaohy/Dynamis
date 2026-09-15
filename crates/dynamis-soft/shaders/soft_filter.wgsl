fn owner_filter(owner: u32) -> vec2u {
    return vec2u(bodies[owner].collision_group, bodies[owner].collision_mask);
}
