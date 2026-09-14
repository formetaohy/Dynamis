fn resolve_row(body_id: u32, generation: u32) -> u32 {
    if (body_id >= arrayLength(&row_of_body)) {
        return NO_BODY;
    }
    let row = row_of_body[body_id];
    if (row >= arrayLength(&body_states) || !contact_row_matches(body_states[row], body_id, generation)) {
        return NO_BODY;
    }
    return row;
}
