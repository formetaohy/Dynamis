fn accumulate_reaction(row: u32, shift: vec3f, spin: vec3f) {
    let base = row * REACTION_WORDS;
    atomicAdd(&reactions[base], reaction_word(shift.x));
    atomicAdd(&reactions[base + 1u], reaction_word(shift.y));
    atomicAdd(&reactions[base + 2u], reaction_word(shift.z));
    atomicAdd(&reactions[base + 3u], reaction_word(spin.x));
    atomicAdd(&reactions[base + 4u], reaction_word(spin.y));
    atomicAdd(&reactions[base + 5u], reaction_word(spin.z));
}
