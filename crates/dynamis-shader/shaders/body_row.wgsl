fn resolve_row(body_id: u32, generation: u32) -> u32 {
    if (body_id >= arrayLength(&row_of_body)) {
        return NO_BODY;
    }
    let row = row_of_body[body_id];
    if (row >= params.body_count) {
        return NO_BODY;
    }
    let state = body_states[row];
    if (state.body_id != body_id || state.generation != generation) {
        return NO_BODY;
    }
    return row;
}
