#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterDesc {
    pub radius: f32,
    pub half_height: f32,
    pub step_height: f32,
    pub slope_limit: f32,
    pub max_speed: f32,
    pub jump_speed: f32,
}

impl CharacterDesc {
    pub fn assert_valid(&self) {
        assert!(self.radius > 0.0, "character radius must be positive");
        assert!(
            self.half_height >= 0.0,
            "character half height must be non-negative"
        );
        assert!(
            self.step_height >= 0.0,
            "character step height must be non-negative"
        );
        assert!(
            (0.0..=std::f32::consts::FRAC_PI_2).contains(&self.slope_limit),
            "character slope limit must be within [0, pi/2]"
        );
        assert!(self.max_speed > 0.0, "character max speed must be positive");
        assert!(
            self.jump_speed >= 0.0,
            "character jump speed must be non-negative"
        );
    }

    pub fn slope_cosine(&self) -> f32 {
        self.slope_limit.cos()
    }
}

impl Default for CharacterDesc {
    fn default() -> Self {
        Self {
            radius: 0.4,
            half_height: 0.5,
            step_height: 0.3,
            slope_limit: 50.0_f32.to_radians(),
            max_speed: 4.0,
            jump_speed: 5.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CharacterHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterState {
    pub position: [f32; 3],
    pub vertical_speed: f32,
    pub grounded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterInput {
    pub direction: [f32; 3],
    pub jump: bool,
}

impl CharacterInput {
    pub fn moving(direction: [f32; 3]) -> Self {
        Self {
            direction,
            jump: false,
        }
    }
}
