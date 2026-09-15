use dynamis_domain::{SETTLE_STEPS, Settling, grown, settled, unreported};

const CAPACITIES: [u32; 6] = [0, 1, 64, 256, 1024, 1 << 20];
const REQUIRED: [u32; 6] = [0, 1, 63, 255, 4096, 1 << 21];
const FLOORS: [u32; 2] = [1, 256];

#[test]
fn a_plan_never_serves_less_than_its_requirement() {
    for capacity in CAPACITIES {
        for required in REQUIRED {
            for floor in FLOORS {
                for release in [false, true] {
                    let planned = settled(capacity, required, floor, release);
                    assert!(
                        planned >= required,
                        "planned {planned} must serve required {required}, capacity {capacity}, floor {floor}, release {release}"
                    );
                    assert!(
                        planned >= floor,
                        "planned {planned} must hold floor {floor}, capacity {capacity}, required {required}, release {release}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_held_plan_keeps_every_slot_it_already_served() {
    for capacity in CAPACITIES {
        for required in REQUIRED {
            for floor in FLOORS {
                let planned = settled(capacity, required, floor, false);
                let held = capacity.max(floor);
                if required <= capacity {
                    assert_eq!(
                        planned, held,
                        "demand {required} within capacity {capacity} must keep the plan still, floor {floor}"
                    );
                } else if required > held {
                    assert!(
                        planned > held,
                        "demand {required} beyond the served {held} must grow the plan, capacity {capacity}, floor {floor}"
                    );
                }
            }
        }
    }
}

#[test]
fn only_an_outgrown_held_plan_moves() {
    let capacity = 4096;
    let floor = 256;
    let mut plan = settled(capacity, 0, floor, false);
    assert_eq!(plan, capacity);
    for required in [0, 1, 255, 256, 4096] {
        let next = settled(plan, required, floor, false);
        assert_eq!(
            next, plan,
            "demand {required} must leave the settled plan {plan} untouched"
        );
        plan = next;
    }
    assert_eq!(
        settled(plan, 4097, floor, false),
        8192,
        "the first demand beyond the plan must double the served reservation"
    );
}

#[test]
fn a_released_plan_serves_exactly_the_demand_it_settled_for() {
    for capacity in CAPACITIES {
        for required in REQUIRED {
            for floor in FLOORS {
                assert_eq!(
                    settled(capacity, required, floor, true),
                    required.max(floor),
                    "a released plan must answer the demand {required}, capacity {capacity}, floor {floor}"
                );
            }
        }
    }
}

#[test]
fn a_plan_never_shrinks_below_a_grown_plan() {
    for capacity in CAPACITIES {
        for required in REQUIRED {
            for floor in FLOORS {
                let grown = grown(capacity, required, floor);
                for release in [false, true] {
                    let planned = settled(capacity, required, floor, release);
                    assert!(
                        planned <= grown,
                        "planned {planned} outran the grown plan {grown}, capacity {capacity}, required {required}, floor {floor}, release {release}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_plan_grows_with_the_demand_it_serves() {
    for capacity in CAPACITIES {
        for floor in FLOORS {
            for release in [false, true] {
                let mut previous = settled(capacity, 0, floor, release);
                for required in REQUIRED {
                    let planned = settled(capacity, required, floor, release);
                    assert!(
                        planned >= previous,
                        "demand {required} shrank the plan {previous} to {planned}, capacity {capacity}, floor {floor}, release {release}"
                    );
                    previous = planned;
                }
            }
        }
    }
}

#[test]
fn unreported_counts_the_sources_the_counters_cannot_know() {
    assert_eq!(unreported(0, 0), 0);
    assert_eq!(unreported(10, 4), 6);
    assert_eq!(unreported(4, 4), 0);
    assert_eq!(unreported(4, 10), 0);
}

#[test]
fn a_world_holds_its_reservation_until_a_quiet_window_passes() {
    let mut settling = Settling::IDLE;
    for step in 0..SETTLE_STEPS as u64 - 1 {
        assert!(
            !settling.release(step, false),
            "step {step} is still inside the quiet window"
        );
    }
    assert!(
        settling.release(SETTLE_STEPS as u64 - 1, false),
        "a full quiet window must settle the reservation"
    );
}

#[test]
fn an_active_step_restarts_the_quiet_window() {
    let mut settling = Settling::IDLE;
    for step in 0..SETTLE_STEPS as u64 - 1 {
        settling.release(step, false);
    }
    assert!(
        !settling.release(SETTLE_STEPS as u64 - 1, true),
        "an active step must hold the reservation"
    );
    for step in SETTLE_STEPS as u64..SETTLE_STEPS as u64 * 2 - 1 {
        assert!(
            !settling.release(step, false),
            "step {step} must count from the last active step"
        );
    }
    assert!(settling.release(SETTLE_STEPS as u64 * 2 - 1, false));
}

#[test]
fn every_run_of_one_step_settles_it_once() {
    let mut settling = Settling::IDLE;
    for step in 0..SETTLE_STEPS as u64 - 1 {
        for _ in 0..3 {
            settling.release(step, false);
        }
    }
    assert!(
        settling.release(SETTLE_STEPS as u64 - 1, false),
        "the runs of a step must not shorten the window that step holds"
    );
}
