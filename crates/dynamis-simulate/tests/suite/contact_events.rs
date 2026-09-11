use super::common::{DT, sim, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, ContactEventKind, ContactEventMode, Shape};
use std::cell::Cell;
use std::rc::Rc;

fn falling_ball_scene() -> (
    dynamis_simulate::Simulation,
    dynamis_model::BodyHandle,
    dynamis_model::BodyHandle,
) {
    let mut world = sim(super::common::gravity_config());
    let ground = world.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::cuboid([20.0, 0.5, 20.0])).events(ContactEventMode::Persist),
        )
        .mass(0.0)
        .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).events(ContactEventMode::Persist))
            .position([0.0, 2.0, 0.0]),
    );
    (world, ground, ball)
}

#[test]
fn disabled_events_silence_contacts() {
    let mut world = sim(static_config());
    let ground = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(1.0)).events(ContactEventMode::None))
            .mass(0.0)
            .position([0.0, -1.0, 0.0]),
    );
    let ball = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.2)).events(ContactEventMode::BeginEnd))
            .position([0.0, 0.0, 0.0]),
    );
    for _ in 0..10 {
        world.step(DT);
        world.wait();
        assert!(
            world.drain_events().is_empty(),
            "a collider with events disabled must not emit"
        );
    }
    let _ = ground;
    let _ = ball;
}

#[test]
fn persist_mode_emits_per_frame_touch() {
    let (mut world, ground, ball) = falling_ball_scene();
    let mut persist_count = 0;
    let mut began = false;
    for _ in 0..40 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            match event.kind {
                ContactEventKind::Begin => began = true,
                ContactEventKind::Persist => {
                    assert!(event.first == ground || event.second == ground);
                    assert!(event.first == ball || event.second == ball);
                    persist_count += 1;
                }
                ContactEventKind::End => {}
            }
        }
    }
    assert!(began, "persist mode must still emit begin");
    assert!(
        persist_count >= 3,
        "persist mode must emit per-frame touch events, got {persist_count}"
    );
}

#[test]
fn persist_without_both_opt_in_stays_silent() {
    let (mut world, _ground, _ball) = falling_ball_scene();
    let ball = _ball;
    world.set_collider_events(ball, 0, ContactEventMode::Persist);
    let mut persist_count = 0;
    for _ in 0..30 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            if event.kind == ContactEventKind::Persist {
                persist_count += 1;
            }
        }
    }
    assert_eq!(
        persist_count, 0,
        "one-side opt-in must not emit persist events"
    );
}

#[test]
fn event_sink_receives_events() {
    let (mut world, ground, ball) = falling_ball_scene();
    let begin_count = Rc::new(Cell::new(0));
    let sink_count = begin_count.clone();
    let sink_ground = ground;
    let sink_ball = ball;
    world.set_event_sink(Some(Box::new(move |event| {
        if event.kind == ContactEventKind::Begin
            && (event.first == sink_ground || event.second == sink_ground)
            && (event.first == sink_ball || event.second == sink_ball)
        {
            sink_count.set(sink_count.get() + 1);
        }
    })));
    for _ in 0..40 {
        world.step(DT);
        world.wait();
    }
    assert!(
        begin_count.get() >= 1,
        "event sink must deliver contact events"
    );
}

#[test]
fn persisted_touch_envokes_sink_per_frame() {
    let mut world = super::common::sim(super::common::gravity_config());
    let ground = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(10.0)).events(ContactEventMode::Persist))
            .mass(0.0)
            .position([0.0, -2.0, 0.0]),
    );
    let ball = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).events(ContactEventMode::Persist))
            .position([0.0, 2.0, 0.0]),
    );
    let persist_count = Rc::new(Cell::new(0));
    let sink_count = persist_count.clone();
    let sink_ball = ball;
    world.set_event_sink(Some(Box::new(move |event| {
        if event.kind == ContactEventKind::Persist
            && (event.first == sink_ball || event.second == sink_ball)
        {
            sink_count.set(sink_count.get() + 1);
        }
    })));
    for _ in 0..40 {
        world.step(DT);
        world.wait();
    }
    assert!(
        persist_count.get() > 5,
        "sink must receive per-frame touch, got {}",
        persist_count.get()
    );
    let _ = ground;
}
