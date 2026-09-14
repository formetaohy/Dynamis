use dynamis_domain::{grown, settled, unreported};

const CAPACITIES: [u32; 6] = [0, 1, 64, 256, 1024, 1 << 20];
const REQUIRED: [u32; 6] = [0, 1, 63, 255, 4096, 1 << 21];
const FLOORS: [u32; 2] = [1, 256];

#[test]
fn a_plan_never_serves_less_than_its_requirement() {
    for capacity in CAPACITIES {
        for required in REQUIRED {
            for floor in FLOORS {
                for idle in [false, true] {
                    let planned = settled(idle, capacity, required, floor);
                    assert!(
                        planned >= required,
                        "planned {planned} must serve required {required}, capacity {capacity}, floor {floor}, idle {idle}"
                    );
                    assert!(
                        planned >= floor,
                        "planned {planned} must hold floor {floor}, capacity {capacity}, required {required}, idle {idle}"
                    );
                }
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
                for idle in [false, true] {
                    let planned = settled(idle, capacity, required, floor);
                    assert!(
                        planned <= grown,
                        "planned {planned} outran the grown plan {grown}, capacity {capacity}, required {required}, floor {floor}, idle {idle}"
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
            for idle in [false, true] {
                let mut previous = settled(idle, capacity, 0, floor);
                for required in REQUIRED {
                    let planned = settled(idle, capacity, required, floor);
                    assert!(
                        planned >= previous,
                        "demand {required} shrank the plan {previous} to {planned}, capacity {capacity}, floor {floor}, idle {idle}"
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
