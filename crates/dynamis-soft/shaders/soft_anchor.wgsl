fn rigid_row(body_id: u32, generation: u32) -> u32 {
    if (body_id >= arrayLength(&row_of_body)) {
        return NO_BODY;
    }
    let row = row_of_body[body_id];
    if (row >= arrayLength(&body_states)
        || body_states[row].body_id != body_id
        || body_states[row].generation != generation) {
        return NO_BODY;
    }
    return row;
}

fn rigid_anchor(body: Body, local: vec3f) -> vec3f {
    return body.state.position + quat_rotate(body.state.orientation, local);
}
