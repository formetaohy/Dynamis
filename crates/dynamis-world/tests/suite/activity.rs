use super::common::{
    DT, asleep, gravity_config, observed_world, settle, settle_until, static_config,
};
use dynamis_abi::{
    COUNTER_ACTIVE, COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_LIVE, COUNTER_PAIRS,
    COUNTER_RESTING, COUNTER_RESTING_GATHER, COUNTER_SLEPT, COUNTER_WOKE,
};
use dynamis_model::{
    BodyDesc, BodyHandle, ColliderDesc, ContactEventKind, ContactEventMode, PhysicsConfig,
    QueryFilter, Shape,
};

fn rest_scene() -> (dynamis_world::World, BodyHandle) {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let bodies = (0..16)
        .map(|index| {
            let x = (index % 4) as f32 * 0.85 - 1.3;
            let z = (index / 4) as f32 * 0.85 - 1.3;
            world.spawn(BodyDesc::sphere(0.4).position([x, 0.4, z]).friction(0.6))
        })
        .collect::<Vec<_>>();
    (world, bodies[0])
}

#[test]
fn a_pile_at_rest_leaves_the_simulation_domain() {
    let (mut world, _) = rest_scene();
    settle(&mut world, 20);
    assert!(
        world.measured()[COUNTER_PAIRS] > 0,
        "an unsettled pile must generate pairs"
    );
    settle_until(&mut world, 400, |world| {
        let measured = world.measured();
        measured[COUNTER_ACTIVE] == 0
            && measured[COUNTER_ENTRIES] == 0
            && measured[COUNTER_PAIRS] == 0
            && measured[COUNTER_CONTACTS] == 0
    });
    let settled = world.measured();
    assert_eq!(
        settled[COUNTER_ACTIVE], 0,
        "every body must be asleep once the pile settles"
    );
    assert_eq!(
        settled[COUNTER_ENTRIES], 0,
        "a sleeping world must not maintain a broadphase grid"
    );
    assert_eq!(
        settled[COUNTER_PAIRS], 0,
        "a sleeping world must not generate pairs"
    );
    assert_eq!(
        settled[COUNTER_CONTACTS], 0,
        "a sleeping world must not hold live contacts"
    );
    assert!(
        world.is_idle(),
        "a sleeping world must leave the simulation domain"
    );
}

#[test]
fn a_slept_constrained_island_leaves_the_simulation_domain() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let first = world.spawn(BodyDesc::cuboid([0.4; 3]).position([-0.5, 0.4, 0.0]));
    let second = world.spawn(BodyDesc::cuboid([0.4; 3]).position([0.5, 0.4, 0.0]));
    world.add_constraint(
        first,
        second,
        dynamis_model::ConstraintDesc::fixed([0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]),
    );
    settle(&mut world, 4);
    assert!(!world.is_idle(), "a fresh constrained island must simulate");
    world.sleep(first);
    world.sleep(second);
    settle(&mut world, 4);
    assert!(asleep(&world), "the constrained island must be asleep");
    assert!(
        world.is_idle(),
        "a sleeping island must leave the simulation domain even while its constraint lives"
    );
    settle(&mut world, 4);
    assert!(world.is_idle(), "an idle world must stay idle");
    world.wake(first);
    world.step(DT);
    world.wait();
    assert!(
        !world.is_idle(),
        "waking a body must ask the domain for work again"
    );
    world.step(DT);
    world.wait();
    assert!(
        !world.is_idle(),
        "an awake body must keep the domain simulating"
    );
}

#[test]
fn resting_contacts_survive_sleep_and_recycle_their_slots() {
    let (mut world, bottom) = rest_scene();
    settle_until(&mut world, 400, |world| asleep(world));
    let resting = world.measured()[COUNTER_RESTING];
    assert!(
        resting > 0,
        "a sleeping pile must keep its contacts in the resting store"
    );
    assert_eq!(
        world.inspect_contacts().len(),
        resting as usize,
        "the public contact list must report the resting contacts"
    );

    world.wake(bottom);
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_WOKE] > 0,
        "the wake must be counted"
    );
    assert!(
        world.measured()[COUNTER_PAIRS] > 0,
        "a woken world must generate pairs again"
    );
    for _ in 0..3 {
        for handle in world.bodies().to_vec() {
            world.wake(handle);
        }
        settle_until(&mut world, 400, |world| asleep(world));
    }
    assert!(
        world.measured()[COUNTER_RESTING] <= resting + 16,
        "sleeping again must recycle resting slots instead of appending"
    );
}

#[test]
fn sleeping_a_pair_never_ends_its_contact() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).events(ContactEventMode::BeginEnd))
            .position([0.0, 2.0, 0.0]),
    );
    let mut began = 0;
    for _ in 0..300 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            match event.kind {
                ContactEventKind::Begin => began += 1,
                ContactEventKind::End => {
                    panic!("a resting pair must not report the end of its contact")
                }
                ContactEventKind::Persist => {}
            }
        }
        if world.read_state(ball).sleeping {
            break;
        }
    }
    assert_eq!(began, 1, "the landing must report exactly one begin");
}

#[test]
fn sleep_and_wake_transitions_are_counted_once() {
    let mut world = observed_world(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    settle(&mut world, 40);
    assert!(world.read_state(ball).sleeping);
    assert!(
        world.measured()[COUNTER_SLEPT] == 0,
        "a body that is already asleep must not be counted again"
    );
    world.wake(ball);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_WOKE],
        1,
        "the wake command must count exactly one transition"
    );
}

#[test]
fn static_pairs_never_reach_the_pair_stream() {
    let mut world = observed_world(static_config());
    assert!(PhysicsConfig::default().gravity[1] < 0.0);
    for index in 0..32 {
        world.spawn(BodyDesc::static_sphere(6.0).position([index as f32 * 0.5, 0.0, 0.0]));
    }
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_PAIRS],
        0,
        "two static colliders must never enter the pair stream"
    );
}

#[derive(Default)]
struct ContactTally {
    begins: usize,
    persists: usize,
    ends: usize,
}

impl ContactTally {
    fn collect(&mut self, world: &mut dynamis_world::World) {
        for event in world.drain_events() {
            match event.kind {
                ContactEventKind::Begin => self.begins += 1,
                ContactEventKind::Persist => self.persists += 1,
                ContactEventKind::End => self.ends += 1,
            }
        }
    }

    fn drain(&mut self, world: &mut dynamis_world::World, steps: usize) {
        for _ in 0..steps {
            world.step(DT);
            world.wait();
            self.collect(world);
        }
    }
}

fn sleeping_ball_on_ground(collider: ColliderDesc) -> (dynamis_world::World, BodyHandle) {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(BodyDesc::new(collider).position([0.0, 2.0, 0.0]));
    settle_until(&mut world, 300, |world| asleep(world));
    assert!(
        world.read_state(ball).sleeping,
        "the landing ball must fall asleep"
    );
    (world, ball)
}

#[test]
fn reviving_a_resting_pair_reports_no_second_begin() {
    let (mut world, ball) = sleeping_ball_on_ground(ColliderDesc::new(Shape::sphere(0.5)));
    world.drain_events();

    world.wake(ball);
    let mut tally = ContactTally::default();
    tally.drain(&mut world, 4);
    assert_eq!(
        tally.begins, 0,
        "waking a resting pair must not report a begin"
    );
    assert_eq!(
        tally.ends, 0,
        "waking a resting pair must not report an end"
    );

    world.set_position(ball, [0.0, 8.0, 0.0]);
    world.set_velocity(ball, [0.0; 3]);
    tally.drain(&mut world, 4);
    assert_eq!(tally.begins, 0, "a separated pair must not report a begin");
    assert_eq!(
        tally.ends, 1,
        "a separated pair must report exactly one end"
    );
}

#[test]
fn reviving_a_resting_pair_reports_no_second_begin_once_the_id_space_outgrows_the_live_colliders() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let churn = (0..256)
        .map(|_| world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0])))
        .collect::<Vec<_>>();
    let resting = churn
        .iter()
        .copied()
        .filter(|body| body.id >= 255)
        .collect::<Vec<_>>();
    assert_eq!(
        resting.iter().map(|body| body.id).collect::<Vec<_>>(),
        [255, 256],
        "the id space must straddle a key byte while the live rows stay inside one"
    );
    for body in churn.iter().copied().filter(|body| body.id < 255) {
        world.remove(body);
    }
    for body in &resting {
        world.add_collider(*body, ColliderDesc::new(Shape::sphere(0.5)));
        world.remove_collider(*body, 1);
    }
    world.set_position(resting[1], [2.0, 0.5, 0.0]);
    let shape = world.rigid_shape();
    assert_eq!(
        (shape.body_row_words, shape.collider_slot_words),
        (1, 1),
        "the live rows and the collider slots must stay inside one key byte"
    );
    assert_eq!(
        shape.body_id_words, 2,
        "the id space must span two key bytes"
    );

    settle_until(&mut world, 300, |world| asleep(world));
    assert!(
        world.read_state(resting[0]).sleeping && world.read_state(resting[1]).sleeping,
        "both resting balls must fall asleep"
    );
    world.drain_events();

    world.wake(resting[0]);
    let mut tally = ContactTally::default();
    tally.drain(&mut world, 4);
    assert_eq!(
        tally.begins, 0,
        "waking a resting pair must not report a begin once the id space outgrows the live colliders"
    );
    assert_eq!(
        tally.ends, 0,
        "waking a resting pair must not report an end once the id space outgrows the live colliders"
    );
}

#[test]
fn reviving_a_resting_pair_resumes_its_persist_stream() {
    let persist = |shape| ColliderDesc::new(shape).events(ContactEventMode::Persist);
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::new(persist(Shape::cuboid([5.0, 0.5, 5.0])))
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(BodyDesc::new(persist(Shape::sphere(0.5))).position([0.0, 2.0, 0.0]));
    settle_until(&mut world, 300, |world| asleep(world));
    assert!(
        world.read_state(ball).sleeping,
        "the landing ball must fall asleep"
    );
    assert!(
        !world.inspect_contacts().is_empty(),
        "the sleeping ball must keep its resting contact"
    );
    world.drain_events();

    world.wake(ball);
    let mut tally = ContactTally::default();
    tally.drain(&mut world, 4);
    assert_eq!(tally.begins, 0, "reviving must not report a begin");
    assert_eq!(tally.ends, 0, "reviving must not report an end");
    assert!(
        tally.persists > 0,
        "the revived contact must continue its persist stream"
    );
}

#[test]
fn an_impact_revives_a_resting_pair_without_a_new_begin() {
    let mut world = observed_world(gravity_config());
    let ground = world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let sleeper = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    settle_until(&mut world, 300, |world| asleep(world));
    assert!(
        world.read_state(sleeper).sleeping,
        "the resting ball must fall asleep"
    );
    world.drain_events();

    let faller = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    let rests_on_ground = |event: &dynamis_model::ContactEvent| {
        (event.first.is_body(sleeper) || event.second.is_body(sleeper))
            && (event.first.is_body(ground) || event.second.is_body(ground))
    };
    let mut disturbed = 0;
    let mut impacts = 0;
    for _ in 0..60 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            let touches_faller = event.first.is_body(faller) || event.second.is_body(faller);
            match event.kind {
                ContactEventKind::Begin if rests_on_ground(&event) => disturbed += 1,
                ContactEventKind::End if rests_on_ground(&event) => disturbed += 1,
                ContactEventKind::Begin if touches_faller => impacts += 1,
                _ => {}
            }
        }
    }
    assert!(
        !world.read_state(sleeper).sleeping,
        "the impact must wake the sleeping ball"
    );
    assert_eq!(
        disturbed, 0,
        "an impact must not disturb the resting contact of the sleeper"
    );
    assert_eq!(impacts, 1, "the impact must report exactly one begin");
}

#[test]
fn removing_a_body_ends_only_its_own_resting_contacts() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    let victim = world.spawn(BodyDesc::sphere(0.5).position([3.0, 0.5, 0.0]));
    settle_until(&mut world, 300, |world| asleep(world));
    assert!(
        world.read_state(ball).sleeping && world.read_state(victim).sleeping,
        "both resting balls must fall asleep"
    );
    world.drain_events();

    world.remove(victim);
    let mut tally = ContactTally::default();
    tally.drain(&mut world, 6);
    assert_eq!(tally.begins, 0, "removing a body must not report a begin");
    assert_eq!(
        tally.ends, 1,
        "the removed body must end exactly one resting contact"
    );
    assert!(
        world.inspect_contacts().iter().any(|manifold| {
            (manifold.first.id == ball.id && manifold.second.id == 0)
                || (manifold.second.id == ball.id && manifold.first.id == 0)
        }),
        "the sleeping ball must keep the ground contact it slept on"
    );
}

#[test]
fn an_explicitly_slept_body_holds_the_pose_it_was_slept_at() {
    let mut world = observed_world(static_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.4)
            .position([0.0, 0.0, 0.0])
            .velocity([12.0, 0.0, 0.0]),
    );
    settle(&mut world, 4);
    let frozen = world.read_state(ball).position;
    assert!(
        frozen[0] > 0.0,
        "the ball must be travelling before it sleeps"
    );
    world.sleep(ball);
    settle(&mut world, 4);
    let held = world.read_state(ball);
    assert!(held.sleeping, "the slept ball must stay asleep");
    assert!(held.velocity == [0.0; 3], "a slept ball must hold still");
    assert_eq!(
        held.position, frozen,
        "sleeping a moving body must freeze the pose it sleeps at, {frozen:?} -> {:?}",
        held.position
    );
}

#[test]
fn falling_asleep_holds_the_pose_the_body_slept_at() {
    let config = PhysicsConfig {
        sleep_time: 0.1,
        ..static_config()
    };
    let mut world = observed_world(config);
    let ball = world.spawn(BodyDesc::sphere(0.4).velocity([0.15, 0.0, 0.0]));
    let mut frozen = None;
    for _ in 0..120 {
        world.step(DT);
        world.wait();
        let state = world.read_state(ball);
        if state.sleeping {
            frozen = Some(state.position);
            break;
        }
    }
    let frozen = frozen.expect("the drifting ball must fall asleep");
    assert!(
        frozen[0] > 0.0,
        "the sleeper must have travelled before it fell asleep"
    );
    world.step(DT);
    world.wait();
    let held = world.read_state(ball);
    assert!(held.sleeping, "the sleeper must stay asleep");
    assert_eq!(
        held.position, frozen,
        "falling asleep must not rewind the pose, {frozen:?} -> {:?}",
        held.position
    );
}

#[test]
fn the_live_set_holds_exactly_the_simulated_bodies() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let sleeper = world.spawn(BodyDesc::sphere(0.4).position([0.0, 0.4, 0.0]));
    let lonely = world.spawn(BodyDesc::sphere(0.4).position([4.0, 0.4, 4.0]));

    settle_until(&mut world, 400, |world| {
        world.measured()[COUNTER_ACTIVE] == 0
    });
    assert!(
        world.read_state(sleeper).sleeping && world.read_state(lonely).sleeping,
        "the scene must fall asleep before the live set empties"
    );
    assert_eq!(
        world.measured()[COUNTER_LIVE],
        0,
        "a sleeping world must hold no live body"
    );

    world.wake(lonely);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_LIVE],
        1,
        "only the woken body joins the live set"
    );

    world.spawn(
        BodyDesc::cuboid([0.5, 0.1, 0.5])
            .kinematic(true)
            .position([0.0, 3.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_LIVE],
        2,
        "a driven body stays live beside the awake body"
    );
}

#[test]
fn a_query_never_wakes_a_sleeping_simulation() {
    let (mut world, _) = rest_scene();
    settle_until(&mut world, 400, |world| world.is_idle());
    assert!(
        world.measured()[COUNTER_RESTING] > 0,
        "a settled pile must archive its resting contacts"
    );

    let filter = QueryFilter::default();
    for _ in 0..8 {
        let handle = world.ray_query([0.0, 4.0, 0.0], [0.0, -1.0, 0.0], 20.0, &filter);
        world.step(DT);
        world.wait();
        assert!(
            world.query_hit(handle).is_some(),
            "a sleeping world must still answer its queries"
        );
    }

    let measured = world.measured();
    assert_eq!(
        measured[COUNTER_ACTIVE], 0,
        "a query must not wake a sleeping body"
    );
    assert_eq!(
        measured[COUNTER_RESTING_GATHER], 0,
        "a query must not drag the resting contacts back through the projection passes"
    );
    assert_eq!(
        measured[COUNTER_CONTACTS], 0,
        "a query must not reopen the narrowphase"
    );
    assert_eq!(
        measured[COUNTER_LIVE], 0,
        "a query must not open the live set"
    );
}
