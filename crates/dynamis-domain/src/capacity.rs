pub const MIN_SLOTS: u32 = 64;
pub const STREAM_FLOOR: u32 = 256;
const SLOTS_HEADROOM: u32 = 2;
pub const SETTLE_STEPS: u32 = 120;

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

pub fn settled(capacity: u32, required: u32, floor: u32, release: bool) -> u32 {
    if release {
        required.max(floor)
    } else {
        grown(capacity, required, floor)
    }
}

pub struct Settling {
    observed: Option<u64>,
    quiet: u32,
}

impl Settling {
    pub const IDLE: Self = Self {
        observed: None,
        quiet: 0,
    };

    pub fn release(&mut self, step: u64, busy: bool) -> bool {
        if self.observed != Some(step) {
            self.observed = Some(step);
            self.quiet = if busy {
                0
            } else {
                self.quiet.saturating_add(1)
            };
        }
        self.quiet >= SETTLE_STEPS
    }
}
