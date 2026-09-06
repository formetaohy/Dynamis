pub struct PhysicsConfig {
    pub gravity: [f32; 3],
    pub damping: f32,
    pub angular_damping: f32,
    pub solve_iterations: u32,
    pub position_iterations: u32,
    pub relaxation: f32,
    pub slop: f32,
    pub restitution_threshold: f32,
    pub max_velocity: f32,
    pub max_angular_velocity: f32,
    pub broadphase_cell_size: f32,
    pub sleep_velocity: f32,
    pub sleep_angular_velocity: f32,
    pub sleep_time: f32,
    pub wake_velocity: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            damping: 0.05,
            angular_damping: 0.05,
            solve_iterations: 12,
            position_iterations: 6,
            relaxation: 0.8,
            slop: 0.005,
            restitution_threshold: 1.0,
            max_velocity: 200.0,
            max_angular_velocity: 400.0,
            broadphase_cell_size: 2.0,
            sleep_velocity: 0.2,
            sleep_angular_velocity: 0.5,
            sleep_time: 0.5,
            wake_velocity: 0.4,
        }
    }
}
