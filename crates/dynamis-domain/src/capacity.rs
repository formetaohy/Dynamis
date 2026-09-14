pub const MIN_SLOTS: u32 = 64;
pub const STREAM_FLOOR: u32 = 256;
pub(crate) const SLOTS_HEADROOM: u32 = 2;
pub(crate) const IDLE_FRACTION: u32 = 4;
pub(crate) const IDLE_DELAY: u32 = 120;

pub fn product(left: u32, right: u32, name: &str) -> u32 {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("{name} capacity exceeds the device index space"))
}

pub fn unreported(live: u32, reported: u32) -> u32 {
    live.saturating_sub(reported)
}

pub fn grown(capacity: u32, required: u32, floor: u32) -> u32 {
    if required <= capacity {
        capacity.max(floor)
    } else {
        required
            .max(capacity.saturating_mul(SLOTS_HEADROOM))
            .max(floor)
    }
}

fn narrowed(capacity: u32, required: u32, floor: u32) -> u32 {
    let target = required.max(floor);
    capacity.min(target.max(capacity / 2)).max(target)
}

pub fn settled(idle: bool, capacity: u32, required: u32, floor: u32) -> u32 {
    if idle && required <= capacity {
        narrowed(capacity, required, floor)
    } else {
        grown(capacity, required, floor)
    }
}

pub struct StreamWatch {
    served: u32,
    pressure: bool,
    idle_steps: u32,
}

impl StreamWatch {
    pub const IDLE: Self = Self {
        served: 0,
        pressure: false,
        idle_steps: 0,
    };

    pub fn observe(&mut self, observed: u32, spilled: bool, lanes: u32) {
        if spilled || observed.saturating_mul(SLOTS_HEADROOM) > lanes {
            self.served = self.served.max(observed.saturating_mul(SLOTS_HEADROOM));
            self.pressure = true;
            self.idle_steps = 0;
        } else if observed.saturating_mul(IDLE_FRACTION) < lanes {
            self.idle_steps = self.idle_steps.saturating_add(1);
        } else {
            self.idle_steps = 0;
        }
    }

    pub fn released(&self, capacity: u32, budget: u32) -> u32 {
        narrowed(capacity, budget, STREAM_FLOOR)
    }

    pub fn widened(&self, capacity: u32, budget: u32) -> u32 {
        capacity.max(budget).max(self.served).max(STREAM_FLOOR)
    }

    pub fn doubled(&self, capacity: u32, budget: u32) -> u32 {
        grown(capacity, self.served.max(budget), STREAM_FLOOR)
    }

    pub fn pressured(&self) -> bool {
        self.pressure
    }

    pub fn is_idle(&self) -> bool {
        self.idle_steps >= IDLE_DELAY
    }

    pub fn settle(&mut self, release: bool) {
        if release {
            self.served = 0;
        }
        self.pressure = false;
        self.idle_steps = 0;
    }
}
