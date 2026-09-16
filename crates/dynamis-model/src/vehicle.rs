use crate::body::BodyDesc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VehicleHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelDesc {
    pub anchor: [f32; 3],
    pub radius: f32,
    pub travel: f32,
    pub frequency: f32,
    pub damping_ratio: f32,
    pub friction: f32,
    pub steering: bool,
    pub driving: bool,
}

impl WheelDesc {
    pub fn new(anchor: [f32; 3], radius: f32) -> Self {
        assert!(radius > 0.0, "a wheel radius must be strictly positive");
        assert!(
            anchor.iter().all(|value| value.is_finite()),
            "a wheel anchor must be finite"
        );
        Self {
            anchor,
            radius,
            travel: radius,
            frequency: 2.0,
            damping_ratio: 0.7,
            friction: 1.0,
            steering: false,
            driving: false,
        }
    }

    pub fn suspension(mut self, travel: f32, frequency: f32, damping_ratio: f32) -> Self {
        assert!(travel > 0.0, "a suspension travel must be positive");
        assert!(
            frequency > 0.0,
            "a suspension frequency must be strictly positive"
        );
        assert!(
            damping_ratio >= 0.0,
            "a suspension damping ratio must be non-negative"
        );
        self.travel = travel;
        self.frequency = frequency;
        self.damping_ratio = damping_ratio;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        assert!(friction >= 0.0, "a wheel friction must be non-negative");
        self.friction = friction;
        self
    }

    pub fn steering(mut self) -> Self {
        self.steering = true;
        self
    }

    pub fn driving(mut self) -> Self {
        self.driving = true;
        self
    }
}

#[derive(Clone, Debug)]
pub struct VehicleDesc {
    pub chassis: BodyDesc,
    pub wheels: Vec<WheelDesc>,
    pub max_steer: f32,
    pub drive_force: f32,
    pub brake_force: f32,
}

impl VehicleDesc {
    pub fn new(chassis: BodyDesc, wheels: Vec<WheelDesc>) -> Self {
        assert!(
            !chassis.kinematic && chassis.mass > 0.0,
            "a vehicle chassis must be a dynamic body"
        );
        let thrust = chassis.mass * 9.81;
        Self {
            chassis,
            wheels,
            max_steer: 30.0_f32.to_radians(),
            drive_force: thrust * 0.5,
            brake_force: thrust,
        }
    }

    pub fn max_steer(mut self, max_steer: f32) -> Self {
        assert!(
            (0.0..std::f32::consts::FRAC_PI_2).contains(&max_steer),
            "a vehicle steering limit must be within [0, pi/2)"
        );
        self.max_steer = max_steer;
        self
    }

    pub fn drive_force(mut self, drive_force: f32) -> Self {
        assert!(
            drive_force >= 0.0,
            "a vehicle drive force must be non-negative"
        );
        self.drive_force = drive_force;
        self
    }

    pub fn brake_force(mut self, brake_force: f32) -> Self {
        assert!(
            brake_force >= 0.0,
            "a vehicle brake force must be non-negative"
        );
        self.brake_force = brake_force;
        self
    }

    pub fn assert_valid(&self) {
        assert!(
            !self.wheels.is_empty(),
            "a vehicle carries at least one wheel"
        );
        for wheel in &self.wheels {
            assert!(
                wheel.radius > 0.0,
                "a wheel radius must be strictly positive"
            );
            assert!(wheel.travel > 0.0, "a suspension travel must be positive");
            assert!(
                wheel.frequency > 0.0,
                "a suspension frequency must be strictly positive"
            );
            assert!(
                wheel.damping_ratio >= 0.0,
                "a suspension damping ratio must be non-negative"
            );
            assert!(
                wheel.friction >= 0.0,
                "a wheel friction must be non-negative"
            );
            assert!(
                wheel.anchor.iter().all(|value| value.is_finite()),
                "a wheel anchor must be finite"
            );
        }
        assert!(
            (0.0..std::f32::consts::FRAC_PI_2).contains(&self.max_steer),
            "a vehicle steering limit must be within [0, pi/2)"
        );
        assert!(
            self.drive_force >= 0.0 && self.brake_force >= 0.0,
            "a vehicle drive and brake force must be non-negative"
        );
        assert!(
            self.wheels.iter().any(|wheel| wheel.driving) || self.drive_force == 0.0,
            "a vehicle that drives wheels needs at least one driving wheel"
        );
    }

    pub fn driving_wheels(&self) -> u32 {
        self.wheels.iter().filter(|wheel| wheel.driving).count() as u32
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VehicleInput {
    pub throttle: f32,
    pub steering: f32,
    pub brake: f32,
}

impl VehicleInput {
    pub const IDLE: Self = Self {
        throttle: 0.0,
        steering: 0.0,
        brake: 0.0,
    };

    pub fn drive(throttle: f32, steering: f32) -> Self {
        let input = Self {
            throttle,
            steering,
            brake: 0.0,
        };
        input.assert_valid();
        input
    }

    pub fn assert_valid(&self) {
        for (name, value) in [
            ("throttle", self.throttle),
            ("steering", self.steering),
            ("brake", self.brake),
        ] {
            assert!(
                (-1.0..=1.0).contains(&value),
                "a vehicle {name} must be within [-1, 1]"
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleState {
    pub position: [f32; 3],
    pub forward_speed: f32,
    pub wheels_grounded: u32,
}

impl VehicleState {
    pub const fn grounded(&self) -> bool {
        self.wheels_grounded > 0
    }
}
