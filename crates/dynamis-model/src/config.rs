#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialCombine {
    Multiply,
    Min,
    Max,
    Average,
}

#[derive(Clone, Copy)]
pub struct PhysicsConfig {
    pub gravity: [f32; 3],
    pub damping: f32,
    pub angular_damping: f32,
    pub substeps: u32,
    pub solve_iterations: u32,
    pub position_iterations: u32,
    pub soft_substeps: u32,
    pub soft_iterations: u32,
    pub relaxation: f32,
    pub slop: f32,
    pub contact_margin: f32,
    pub restitution_threshold: f32,
    pub max_velocity: f32,
    pub max_angular_velocity: f32,
    pub sleep_velocity: f32,
    pub sleep_angular_velocity: f32,
    pub sleep_time: f32,
    pub settle_velocity: f32,
    pub friction_combine: MaterialCombine,
    pub restitution_combine: MaterialCombine,
}

impl PhysicsConfig {
    pub fn assert_valid(&self) {
        assert!(
            self.contact_margin >= 0.0,
            "contact margin must be non-negative"
        );
        assert!(
            self.substeps > 0,
            "rigid substeps must be strictly positive"
        );
        assert!(
            self.substeps > 0,
            "rigid substeps must be strictly positive"
        );
        assert!(
            self.position_iterations > 0,
            "position iterations must be strictly positive"
        );
        assert!(
            self.soft_iterations > 0,
            "soft iterations must be strictly positive"
        );
        assert!(
            self.soft_substeps > 0,
            "soft substeps must be strictly positive"
        );
        assert!(
            self.settle_velocity >= 0.0,
            "the soft settle velocity must be non-negative"
        );
        assert!(
            (0.0..=1.0).contains(&self.relaxation),
            "position relaxation must be within (0, 1]"
        );
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            damping: 0.05,
            angular_damping: 0.05,
            substeps: 2,
            solve_iterations: 8,
            position_iterations: 10,
            soft_substeps: 4,
            soft_iterations: 1,
            relaxation: 0.8,
            slop: 0.005,
            contact_margin: 0.02,
            restitution_threshold: 1.0,
            max_velocity: 200.0,
            max_angular_velocity: 400.0,
            sleep_velocity: 0.2,
            sleep_angular_velocity: 0.5,
            sleep_time: 0.5,
            settle_velocity: 1.0e-4,
            friction_combine: MaterialCombine::Multiply,
            restitution_combine: MaterialCombine::Max,
        }
    }
}
