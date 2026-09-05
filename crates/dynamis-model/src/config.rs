pub struct PhysicsConfig {
    pub gravity: [f32; 3],
    pub damping: f32,
    pub angular_damping: f32,
    pub solve_iterations: u32,
    pub relaxation: f32,
    pub slop: f32,
    pub restitution_threshold: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            damping: 0.05,
            angular_damping: 0.05,
            solve_iterations: 12,
            relaxation: 0.8,
            slop: 0.005,
            restitution_threshold: 1.0,
        }
    }
}
