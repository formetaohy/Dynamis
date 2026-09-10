use dynamis_simulate::planning::{Capacity, Live, Reservation, StreamDemand, StreamPlan};

fn demand(pairs: u32, entries: u32, events: u32) -> StreamDemand {
    StreamDemand {
        pairs,
        entries,
        events,
    }
}

fn live(bodies: u32, constraints: u32) -> Live {
    Live {
        bodies,
        constraints,
        body_commands: 0,
        constraint_commands: 0,
        queries: 0,
    }
}

/// A plan with everything at the floor, as a world about to serve its first demand.
fn floor_plan() -> Reservation {
    Reservation {
        bodies: 64,
        constraints: 64,
        entries: 256,
        pairs: 256,
        events: 256,
        body_commands: 256,
        constraint_commands: 256,
        queries: 256,
    }
}

#[test]
fn widen_serves_measured_demand_and_live_rows() {
    let plan = Reservation::planned(
        &floor_plan(),
        &live(32, 0),
        Some(StreamPlan::Widen(demand(8192, 1024, 64))),
    );
    assert!(plan.bodies >= 32);
    assert!(plan.pairs >= 8192);
    assert!(plan.entries >= 1024);
    assert!(plan.events >= 64);
}

#[test]
fn widen_keeps_density_baseline_for_the_live_count() {
    let plan = Reservation::planned(
        &floor_plan(),
        &live(100, 0),
        Some(StreamPlan::Widen(demand(64, 16, 0))),
    );
    // 100 bodies want at least 100*128 pair lanes even when nothing measured much.
    assert!(plan.pairs >= 100 * 128);
    assert!(plan.entries >= 100 * 16 * 8);
}

#[test]
fn narrow_releases_stream_headroom_gradually() {
    let current = Reservation::planned(
        &floor_plan(),
        &live(32, 0),
        Some(StreamPlan::Widen(demand(8192, 1024, 64))),
    );
    assert_eq!(current.pairs, 8192);
    assert_eq!(current.entries, 8192);
    let idle = Reservation::planned(
        &current,
        &live(32, 0),
        Some(StreamPlan::Narrow(demand(64, 32, 0))),
    );
    // One halving per plan: 8192 -> 4096, never below the floor.
    assert_eq!(idle.pairs, 4096);
    assert_eq!(idle.entries, 4096);
    assert!(idle.events >= 256);
}

#[test]
fn narrow_never_goes_below_the_floor_or_live_counts() {
    let mut current = Reservation::planned(
        &floor_plan(),
        &live(100, 0),
        Some(StreamPlan::Widen(demand(8192, 2048, 256))),
    );
    for _ in 0..8 {
        let next = Reservation::planned(
            &current,
            &live(100, 0),
            Some(StreamPlan::Narrow(demand(0, 0, 0))),
        );
        assert!(next.pairs >= dynamis_simulate::planning::STREAM_FLOOR);
        assert!(next.entries >= dynamis_simulate::planning::STREAM_FLOOR);
        assert!(next.bodies >= 100);
        current = next;
    }
}

#[test]
fn body_rows_shrink_with_the_live_count() {
    let current = Reservation::planned(
        &floor_plan(),
        &live(1024, 0),
        Some(StreamPlan::Widen(demand(0, 0, 0))),
    );
    assert!(current.bodies >= 1024);
    let idle = Reservation::planned(
        &current,
        &live(16, 0),
        Some(StreamPlan::Narrow(demand(0, 0, 0))),
    );
    // Halving toward live*2, yet never below the live count.
    assert!(idle.bodies >= 16);
    assert!(idle.bodies < current.bodies);
    assert!(idle.bodies >= current.bodies / 2);
}

#[test]
fn observe_latches_pressure_and_then_narrows() {
    let mut capacity = Capacity::new();
    let mut plan = floor_plan();
    let mut counts = [0u32; dynamis_layout::COUNTER_COUNT];
    // A heavy step spills the pair stream.
    counts[dynamis_layout::COUNTER_PAIRS] = plan.pairs + 1;
    counts[dynamis_layout::COUNTER_SPILLOVER_PAIRS] = 4;
    let widened = capacity
        .observe(&counts, &plan)
        .expect("pressure must plan a widen");
    assert!(matches!(widened, StreamPlan::Widen(_)));
    plan = Reservation::planned(&plan, &live(64, 0), Some(widened));

    // Sustained idleness must eventually produce a narrowing plan.
    let mut narrowed = None;
    for _ in 0..400 {
        counts[dynamis_layout::COUNTER_PAIRS] = 16;
        counts[dynamis_layout::COUNTER_SPILLOVER_PAIRS] = 0;
        if let Some(next) = capacity.observe(&counts, &plan) {
            if matches!(next, StreamPlan::Narrow(_)) {
                narrowed = Some(next);
                break;
            }
            plan = Reservation::planned(&plan, &live(64, 0), Some(next));
        }
    }
    assert!(narrowed.is_some(), "idle demand must narrow streams");
}
