const POLY6_NORMALIZATION: f32 = 315.0 / (64.0 * 3.141592653589793);
const POLY6_SLOPE: f32 = 945.0 / (32.0 * 3.141592653589793);

fn fixed(value: f32, scale: f32) -> i32 {
    return i32(round(value * scale));
}

fn poly6(r2: f32, h2: f32, h9: f32) -> f32 {
    let span = h2 - r2;
    return POLY6_NORMALIZATION * span * span * span / h9;
}

fn poly6_slope(r: f32, r2: f32, h2: f32, h9: f32) -> f32 {
    let span = h2 - r2;
    return -POLY6_SLOPE * r * span * span / h9;
}
