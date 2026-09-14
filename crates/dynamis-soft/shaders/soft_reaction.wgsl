const REACTION_SCALE: f32 = 65536.0;
const REACTION_WORDS: u32 = 8u;

fn reaction_word(value: f32) -> u32 {
    return u32(i32(clamp(value * REACTION_SCALE, -2.0e9, 2.0e9)));
}

fn accumulate_reaction(row: u32, shift: vec3f, spin: vec3f) {
    let base = row * REACTION_WORDS;
    atomicAdd(&reactions[base], reaction_word(shift.x));
    atomicAdd(&reactions[base + 1u], reaction_word(shift.y));
    atomicAdd(&reactions[base + 2u], reaction_word(shift.z));
    atomicAdd(&reactions[base + 4u], reaction_word(spin.x));
    atomicAdd(&reactions[base + 5u], reaction_word(spin.y));
    atomicAdd(&reactions[base + 6u], reaction_word(spin.z));
}
