const CLASS_BITS: u32 = (1u << SOLVER_CLASS_TOKEN_SHIFT) - 1u;

fn class_token(round: u32, bucket: u32) -> u32 {
    return (round << SOLVER_CLASS_TOKEN_SHIFT) | (bucket + 1u);
}

fn token_class(token: u32) -> u32 {
    return (token & CLASS_BITS) - 1u;
}

fn token_round(token: u32) -> u32 {
    return token >> SOLVER_CLASS_TOKEN_SHIFT;
}

fn block_live() -> u32 {
    return segments[SOLVER_BLOCK_CONTACT] + segments[SOLVER_BLOCK_CONSTRAINT];
}
