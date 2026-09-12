pub(crate) const MIN_SLOTS: u32 = 64;
pub(crate) const STREAM_FLOOR: u32 = 256;
pub(crate) const MOVE_ENTRIES_PER_COMMAND: u32 = 2;
pub(crate) const SLOTS_HEADROOM: u32 = 2;
pub(crate) const PRESSURE_HEADROOM: u32 = 8;
pub(crate) const IDLE_FRACTION: u32 = 4;
pub(crate) const IDLE_DELAY: u32 = 120;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeCapacity {
    pub sources: u32,
    pub vertices: u32,
    pub triangles: u32,
    pub nodes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamCapacity {
    pub entries: u32,
    pub pairs: u32,
    pub events: u32,
    pub shapes: ShapeCapacity,
}

pub(crate) struct Live {
    pub(crate) bodies: u32,
    pub(crate) colliders: u32,
    pub(crate) collider_pool: u32,
    pub(crate) body_ids: u32,
    pub(crate) constraints: u32,
    pub(crate) body_commands: u32,
    pub(crate) constraint_commands: u32,
    pub(crate) queries: u32,
    pub(crate) shapes: ShapeCapacity,
}

pub(crate) fn product(left: u32, right: u32, name: &str) -> u32 {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("{name} capacity exceeds the device index space"))
}

pub(crate) fn grown(capacity: u32, required: u32, floor: u32) -> u32 {
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
    capacity.min(target.max(capacity / 2))
}

pub(crate) fn settled(idle: bool, capacity: u32, required: u32, floor: u32) -> u32 {
    if idle {
        narrowed(capacity, required, floor)
    } else {
        grown(capacity, required, floor)
    }
}

pub(crate) struct StreamWatch {
    served: u32,
    pressure: bool,
    idle_steps: u32,
}

impl StreamWatch {
    pub(crate) const IDLE: Self = Self {
        served: 0,
        pressure: false,
        idle_steps: 0,
    };

    pub(crate) fn observe(&mut self, observed: u32, spilled: bool, lanes: u32, open: bool) {
        let free = lanes.saturating_sub(observed);
        if spilled || free < lanes / PRESSURE_HEADROOM {
            if open {
                self.served = self.served.max(observed.saturating_mul(SLOTS_HEADROOM));
            }
            self.pressure = true;
            self.idle_steps = 0;
        } else if observed.saturating_mul(IDLE_FRACTION) < lanes {
            self.idle_steps = self.idle_steps.saturating_add(1);
        } else {
            self.idle_steps = 0;
        }
    }

    pub(crate) fn released(&self, capacity: u32, budget: u32) -> u32 {
        narrowed(capacity, budget, STREAM_FLOOR)
    }

    pub(crate) fn widened(&self, capacity: u32, budget: u32) -> u32 {
        capacity.max(budget).max(self.served).max(STREAM_FLOOR)
    }

    pub(crate) fn pressured(&self) -> bool {
        self.pressure
    }

    pub(crate) fn is_idle(&self) -> bool {
        self.idle_steps >= IDLE_DELAY
    }

    pub(crate) fn settle(&mut self, release: bool) {
        if release {
            self.served = 0;
        }
        self.pressure = false;
        self.idle_steps = 0;
    }
}
