#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterReset {
    PerStep,
    Maintained,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shortfall {
    None,
    Fatal,
    Physics,
    Report,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterSpec {
    pub name: &'static str,
    pub label: &'static str,
    pub reset: CounterReset,
    pub shortfall: Shortfall,
}

impl CounterSpec {
    pub const fn new(
        name: &'static str,
        label: &'static str,
        reset: CounterReset,
        shortfall: Shortfall,
    ) -> Self {
        Self {
            name,
            label,
            reset,
            shortfall,
        }
    }

    pub const fn clears_every_step(self) -> bool {
        matches!(self.reset, CounterReset::PerStep)
    }

    pub const fn refuses(self) -> bool {
        !matches!(self.shortfall, Shortfall::None)
    }

    pub const fn is_fatal(self) -> bool {
        matches!(self.shortfall, Shortfall::Fatal)
    }
}

macro_rules! declare_counters {
    ($( $name:ident => $label:literal, $reset:ident, $shortfall:ident; )*) => {
        declare_counters!(@slot 0usize; $( $name => $label, $reset, $shortfall; )*);

        pub const COUNTERS: &[CounterSpec] = &[
            $(
                CounterSpec::new(
                    stringify!($name),
                    $label,
                    CounterReset::$reset,
                    Shortfall::$shortfall,
                ),
            )*
        ];
    };
    (@slot $slot:expr; $name:ident => $label:literal, $reset:ident, $shortfall:ident; $($rest:tt)*) => {
        pub const $name: usize = $slot;
        declare_counters!(@slot $slot + 1usize; $($rest)*);
    };
    (@slot $slot:expr;) => {
        pub const COUNTER_DEVICE_COUNT: usize = $slot;
    };
}

declare_counters! {
    COUNTER_ENTRIES => "grid entries", PerStep, None;
    COUNTER_RESTING_ENTRIES => "grid entries of resting bodies", Maintained, None;
    COUNTER_RESTING_LEVELS => "resting grid levels", Maintained, None;
    COUNTER_RESTING_REBUILD => "resting grid entries derived this step", PerStep, None;
    COUNTER_RESTING_SORT => "resting grid entries to sort", PerStep, None;
    COUNTER_AWAKE_BASE => "grid entry base of awake bodies", Maintained, None;
    COUNTER_IMMOVABLE_ENTRIES => "immovable grid entries", Maintained, None;
    COUNTER_IMMOVABLE_EMITTED => "immovable grid entries derived this step", PerStep, None;
    COUNTER_IMMOVABLE_LEVELS => "immovable grid levels", Maintained, None;
    COUNTER_ENTRY_BASE => "grid entry base", Maintained, None;
    COUNTER_PAIRS => "candidate contact pairs", PerStep, None;
    COUNTER_GRID_LEVELS => "occupied grid levels", PerStep, None;
    COUNTER_CONTACTS => "contact manifolds", PerStep, None;
    COUNTER_ARCHIVED => "archived contact manifolds", Maintained, None;
    COUNTER_JOINTS => "joint candidate pairs", PerStep, None;
    COUNTER_EVENTS => "contact events", PerStep, None;
    COUNTER_REFUSED_PAIRS => "candidate pairs beyond the pair stream", PerStep, Physics;
    COUNTER_REFUSED_EVENTS => "contact events beyond the event stream", PerStep, Report;
    COUNTER_ENTRY_FAULTS => "grid entries beyond their budget", PerStep, Fatal;
    COUNTER_RESTING => "resting contact manifolds", Maintained, None;
    COUNTER_REFUSED_RESTING => "resting manifolds beyond the resting archive", PerStep, Physics;
    COUNTER_SLEPT => "bodies that fell asleep", PerStep, None;
    COUNTER_WOKE => "bodies that woke", PerStep, None;
    COUNTER_ACTIVE => "active bodies", PerStep, None;
    COUNTER_WOKE_DEFERRED => "wake requests deferred to the next step", PerStep, None;
    COUNTER_RESTING_GATHER => "resting manifolds gathered", PerStep, None;
    COUNTER_RESTING_INDEX => "resting archive index slots", Maintained, None;
    COUNTER_RESTING_PENDING => "resting manifolds pending reindex", Maintained, None;
    COUNTER_ISLANDS => "coupled body islands", PerStep, None;
    COUNTER_ISLAND_FAULTS => "island unions beyond their hook budget", PerStep, Fatal;
    COUNTER_BLOCKS => "solver blocks", Maintained, None;
    COUNTER_SOLVER_ROWS => "solver rows", Maintained, None;
    COUNTER_SOLVE_LINEAR_RESIDUAL => "velocity the solve applied in its last recorded iteration", PerStep, None;
    COUNTER_SOLVE_ANGULAR_RESIDUAL => "angular velocity the solve applied in its last recorded iteration", PerStep, None;
    COUNTER_COARSE_ACTIVE => "coarse level links considered", PerStep, None;
    COUNTER_GRID_SCALE => "grid scale", PerStep, None;
    COUNTER_GRID_EXTENT => "grid extent", PerStep, None;
    COUNTER_BREAKS => "constraint breaks", PerStep, None;
    COUNTER_PARTICLE_REACH => "particle reach", PerStep, None;
    COUNTER_COARSE_NEIGHBOURS => "coarse neighbours", PerStep, None;
    COUNTER_LIVE => "live body rows", PerStep, None;
    COUNTER_LIVE_FAULTS => "live body rows beyond the row stream", PerStep, Fatal;
    COUNTER_ROW_FAULTS => "row moves that answer no state their row could hold", PerStep, Fatal;
    COUNTER_SOFT_ACTIVE => "active soft bodies", PerStep, None;
    COUNTER_SOFT_SLEPT => "soft bodies that fell asleep", PerStep, None;
    COUNTER_SOFT_WOKE => "soft bodies that woke", PerStep, None;
    COUNTER_REFUSED_CONTACTS => "contact manifolds beyond the contact stream", PerStep, Physics;
    COUNTER_STEP => "step index", Maintained, None;
    COUNTER_IMPACTS => "contact impacts", PerStep, None;
    COUNTER_REFUSED_IMPACTS => "impacts beyond the impact stream", PerStep, Report;
    COUNTER_SOFT_EVENTS => "soft contact events", PerStep, None;
    COUNTER_REFUSED_SOFT_EVENTS => "soft contact events beyond the soft event stream", PerStep, Report;
}

pub fn spec(slot: usize) -> CounterSpec {
    *COUNTERS.get(slot).unwrap_or_else(|| {
        panic!("counter slot {slot} is outside the {COUNTER_DEVICE_COUNT} device counters")
    })
}

const fn count_step_reset() -> usize {
    let mut total = 0;
    let mut slot = 0;
    while slot < COUNTER_DEVICE_COUNT {
        if COUNTERS[slot].clears_every_step() {
            total += 1;
        }
        slot += 1;
    }
    total
}

pub const STEP_RESET_COUNT: usize = count_step_reset();

const fn step_reset_slots() -> [u32; STEP_RESET_COUNT] {
    let mut slots = [0u32; STEP_RESET_COUNT];
    let mut slot = 0;
    let mut next = 0;
    while slot < COUNTER_DEVICE_COUNT {
        if COUNTERS[slot].clears_every_step() {
            slots[next] = slot as u32;
            next += 1;
        }
        slot += 1;
    }
    slots
}

pub const STEP_RESET: [u32; STEP_RESET_COUNT] = step_reset_slots();

const _: () = {
    let mut slot = 0;
    while slot < COUNTER_DEVICE_COUNT {
        assert!(
            !COUNTERS[slot].name.is_empty() && !COUNTERS[slot].label.is_empty(),
            "a device counter must declare a name and a label",
        );
        let mut other = slot + 1;
        while other < COUNTER_DEVICE_COUNT {
            assert!(
                !equal_names(COUNTERS[slot].name, COUNTERS[other].name),
                "a device counter must be declared once",
            );
            other += 1;
        }
        slot += 1;
    }
};

const fn equal_names(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

pub(crate) fn constants_wgsl(out: &mut String) {
    for (slot, counter) in COUNTERS.iter().enumerate() {
        out.push_str(&format!("const {}: u32 = {}u;\n", counter.name, slot));
    }
}

pub(crate) fn step_reset_wgsl(out: &mut String) {
    let slots = STEP_RESET;
    out.push_str(&format!(
        "const STEP_RESET_COUNT: u32 = {}u;\n",
        slots.len()
    ));
    out.push_str(&format!(
        "const STEP_RESET_SLOTS: array<u32, {}> = array<u32, {}>(",
        slots.len(),
        slots.len(),
    ));
    for slot in slots {
        out.push_str(&format!("{slot}u,"));
    }
    out.push_str(");\n");
}

pub mod host {
    use super::COUNTER_DEVICE_COUNT;

    pub const COUNT: usize = 8;

    pub const COUNTER_BODIES: usize = COUNTER_DEVICE_COUNT;
    pub const COUNTER_COLLIDERS: usize = COUNTER_DEVICE_COUNT + 1;
    pub const COUNTER_CONSTRAINTS: usize = COUNTER_DEVICE_COUNT + 2;
    pub const COUNTER_BODY_EDITS: usize = COUNTER_DEVICE_COUNT + 3;
    pub const COUNTER_BODY_MOVES: usize = COUNTER_DEVICE_COUNT + 4;
    pub const COUNTER_CONSTRAINT_DECLARATIONS: usize = COUNTER_DEVICE_COUNT + 5;
    pub const COUNTER_CONSTRAINT_MOVES: usize = COUNTER_DEVICE_COUNT + 6;
    pub const COUNTER_MOVABLE_COLLIDERS: usize = COUNTER_DEVICE_COUNT + 7;
}

pub const COUNTER_COUNT: usize = COUNTER_DEVICE_COUNT + host::COUNT;

pub const COUNTER_STRIDE: u64 = 256;

pub type Counters = [u32; COUNTER_COUNT];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeclaredCounters {
    pub bodies: u32,
    pub colliders: u32,
    pub movable_colliders: u32,
    pub constraints: u32,
    pub body_edits: u32,
    pub body_moves: u32,
    pub constraint_declarations: u32,
    pub constraint_moves: u32,
}

impl DeclaredCounters {
    pub fn write_into(&self, counters: &mut Counters) {
        counters[host::COUNTER_BODIES] = self.bodies;
        counters[host::COUNTER_COLLIDERS] = self.colliders;
        counters[host::COUNTER_MOVABLE_COLLIDERS] = self.movable_colliders;
        counters[host::COUNTER_CONSTRAINTS] = self.constraints;
        counters[host::COUNTER_BODY_EDITS] = self.body_edits;
        counters[host::COUNTER_BODY_MOVES] = self.body_moves;
        counters[host::COUNTER_CONSTRAINT_DECLARATIONS] = self.constraint_declarations;
        counters[host::COUNTER_CONSTRAINT_MOVES] = self.constraint_moves;
    }
}
