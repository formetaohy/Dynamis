use crate::VehicleInputRecord;
use crate::VehicleRecord;
use crate::VehicleStateRecord;
use crate::VehicleWheelRecord;
use crate::constant::NO_BODY;
use dynamis_model::{VehicleDesc, VehicleInput, VehicleState, WheelDesc};

impl VehicleRecord {
    pub fn build(desc: &VehicleDesc, body_id: u32, generation: u32, wheel_base: u32) -> Self {
        desc.assert_valid();
        Self {
            body_id,
            generation,
            wheel_base,
            wheel_count: desc.wheels.len() as u32,
            driving_count: desc.driving_wheels(),
            max_steer: desc.max_steer,
            drive_force: desc.drive_force,
            brake_force: desc.brake_force,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            body_id: NO_BODY,
            generation: 0,
            wheel_base: 0,
            wheel_count: 0,
            driving_count: 0,
            max_steer: 0.0,
            drive_force: 0.0,
            brake_force: 0.0,
        }
    }

    pub const fn is_live(&self) -> bool {
        self.body_id != NO_BODY
    }
}

impl VehicleWheelRecord {
    pub fn build(wheel: &WheelDesc) -> Self {
        wheel.assert_valid();
        Self {
            anchor: wheel.anchor,
            radius: wheel.radius,
            travel: wheel.travel,
            frequency: wheel.frequency,
            damping_ratio: wheel.damping_ratio,
            friction: wheel.friction,
            steering: u32::from(wheel.steering),
            driving: u32::from(wheel.driving),
            _wgsl_pad0: [0; 8],
        }
    }

    pub const fn cleared() -> Self {
        Self {
            anchor: [0.0; 3],
            radius: 0.0,
            travel: 0.0,
            frequency: 0.0,
            damping_ratio: 0.0,
            friction: 0.0,
            steering: 0,
            driving: 0,
            _wgsl_pad0: [0; 8],
        }
    }
}

impl VehicleInputRecord {
    pub fn of(input: VehicleInput) -> Self {
        input.assert_valid();
        Self {
            throttle: input.throttle,
            steering: input.steering,
            brake: input.brake,
            _pad0: 0.0,
        }
    }

    pub const fn idle() -> Self {
        Self {
            throttle: 0.0,
            steering: 0.0,
            brake: 0.0,
            _pad0: 0.0,
        }
    }
}

impl VehicleStateRecord {
    pub fn spawn(owner: u32, generation: u32, position: [f32; 3]) -> Self {
        Self {
            position,
            forward_speed: 0.0,
            ground_count: 0,
            owner,
            generation,
            _pad0: 0,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            position: [0.0; 3],
            forward_speed: 0.0,
            ground_count: 0,
            owner: NO_BODY,
            generation: 0,
            _pad0: 0,
        }
    }

    pub fn owns(&self, owner: u32, generation: u32) -> bool {
        self.owner == owner && self.generation == generation
    }

    pub fn state(&self) -> VehicleState {
        VehicleState {
            position: self.position,
            forward_speed: self.forward_speed,
            wheels_grounded: self.ground_count,
        }
    }
}
