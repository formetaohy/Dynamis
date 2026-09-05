use dynamis_ecs::World;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

#[derive(Debug, PartialEq)]
struct Position(f32, f32);

#[derive(Debug, PartialEq)]
struct Velocity(f32, f32);

#[derive(Debug, PartialEq)]
struct Mass(f32);

#[derive(Clone)]
struct DropCounter(Arc<AtomicI32>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn spawn_and_get() {
    let mut world = World::new();
    let entity = world.spawn((Position(1.0, 2.0), Velocity(0.5, 0.0)));
    assert!(world.alive(entity));
    assert_eq!(world.len(), 1);
    assert_eq!(world.get::<Position>(entity), Some(&Position(1.0, 2.0)));
    assert!(world.get::<Mass>(entity).is_none());
}

#[test]
fn spawn_without_components() {
    let mut world = World::new();
    let entity = world.spawn(());
    assert!(world.alive(entity));
    assert!(!world.has::<Position>(entity));
}

#[test]
fn get_mut_updates_in_place() {
    let mut world = World::new();
    let entity = world.spawn((Position(0.0, 0.0),));
    world.get_mut::<Position>(entity).unwrap().0 = 42.0;
    assert_eq!(world.get::<Position>(entity), Some(&Position(42.0, 0.0)));
}

#[test]
fn stale_entity_is_not_alive() {
    let mut world = World::new();
    let entity = world.spawn((Position(0.0, 0.0),));
    world.despawn(entity);
    assert!(!world.alive(entity));
    assert!(world.get::<Position>(entity).is_none());
    assert_eq!(world.len(), 0);
}

#[test]
fn index_reuse_bumps_generation() {
    let mut world = World::new();
    let first = world.spawn((Position(0.0, 0.0),));
    world.despawn(first);
    let second = world.spawn((Position(1.0, 1.0),));
    assert_eq!(first.index(), second.index());
    assert_ne!(first.generation(), second.generation());
    assert!(!world.alive(first));
    assert!(world.alive(second));
}

#[test]
fn despawn_stale_panics() {
    let mut world = World::new();
    let entity = world.spawn((Position(0.0, 0.0),));
    world.despawn(entity);
    world.spawn((Position(1.0, 1.0),));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.despawn(entity);
        }))
        .is_err()
    );
}

#[test]
fn insert_moves_archetype_and_preserves_data() {
    let mut world = World::new();
    let entity = world.spawn((Position(3.0, 4.0),));
    world.insert(entity, (Velocity(1.0, 2.0),));
    assert_eq!(world.get::<Position>(entity), Some(&Position(3.0, 4.0)));
    assert_eq!(world.get::<Velocity>(entity), Some(&Velocity(1.0, 2.0)));
    assert!(world.has::<Velocity>(entity));
}

#[test]
fn insert_duplicate_panics() {
    let mut world = World::new();
    let entity = world.spawn((Position(0.0, 0.0),));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.insert(entity, (Position(9.0, 9.0),));
        }))
        .is_err()
    );
}

#[test]
fn remove_component_and_keep_others() {
    let mut world = World::new();
    let entity = world.spawn((Position(0.0, 0.0), Velocity(5.0, 5.0), Mass(2.0)));
    world.remove::<Velocity>(entity);
    assert!(!world.has::<Velocity>(entity));
    assert_eq!(world.get::<Position>(entity), Some(&Position(0.0, 0.0)));
    assert_eq!(world.get::<Mass>(entity), Some(&Mass(2.0)));
}

#[test]
fn remove_missing_panics() {
    let mut world = World::new();
    let entity = world.spawn((Position(0.0, 0.0),));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.remove::<Velocity>(entity);
        }))
        .is_err()
    );
}

#[test]
fn swap_removed_row_moves_last_entity_location() {
    let mut world = World::new();
    let a = world.spawn((Position(0.0, 0.0),));
    let b = world.spawn((Position(1.0, 1.0),));
    let c = world.spawn((Position(2.0, 2.0),));
    world.despawn(a);
    assert_eq!(world.get::<Position>(b), Some(&Position(1.0, 1.0)));
    assert_eq!(world.get::<Position>(c), Some(&Position(2.0, 2.0)));
    assert_eq!(world.len(), 2);
}

#[test]
fn despawn_unrelated_entities_survive() {
    let mut world = World::new();
    let a = world.spawn((Position(0.0, 0.0), Mass(1.0)));
    let b = world.spawn((Position(1.0, 1.0),));
    let c = world.spawn((Position(2.0, 2.0), Mass(2.0)));
    world.despawn(b);
    assert_eq!(world.get::<Position>(a), Some(&Position(0.0, 0.0)));
    assert_eq!(world.get::<Mass>(a), Some(&Mass(1.0)));
    assert_eq!(world.get::<Position>(c), Some(&Position(2.0, 2.0)));
    assert_eq!(world.get::<Mass>(c), Some(&Mass(2.0)));
}

#[test]
fn query_reads_pair() {
    let mut world = World::new();
    world.spawn((Position(0.0, 0.0), Velocity(1.0, 1.0)));
    world.spawn((Position(1.0, 1.0), Velocity(2.0, 2.0)));
    world.spawn((Position(2.0, 2.0),));
    let mut pairs = Vec::new();
    world
        .query::<(&Position, &Velocity)>()
        .for_each(|(position, velocity)| {
            pairs.push((position.0, velocity.0));
        });
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(pairs, vec![(0.0, 1.0), (1.0, 2.0)]);
}

#[test]
fn query_mutates_bodies() {
    let mut world = World::new();
    let a = world.spawn((Position(0.0, 0.0), Velocity(1.0, 1.0)));
    let b = world.spawn((Position(1.0, 1.0), Velocity(2.0, 2.0)));
    world
        .query::<(&mut Position, &Velocity)>()
        .for_each(|(position, velocity)| {
            position.0 += velocity.0;
            position.1 += velocity.1;
        });
    assert_eq!(world.get::<Position>(a), Some(&Position(1.0, 1.0)));
    assert_eq!(world.get::<Position>(b), Some(&Position(3.0, 3.0)));
}

#[test]
fn query_counts_only_matching() {
    let mut world = World::new();
    world.spawn((Position(0.0, 0.0), Velocity(1.0, 1.0)));
    world.spawn((Position(0.0, 0.0), Velocity(1.0, 1.0), Mass(2.0)));
    world.spawn((Position(0.0, 0.0),));
    assert_eq!(world.query::<(&Position, &Velocity)>().count(), 2);
    assert_eq!(world.query::<(&Position, &Velocity, &Mass)>().count(), 1);
}

#[test]
fn query_conflicting_access_panics() {
    let mut world = World::new();
    world.spawn((Position(0.0, 0.0),));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world
                .query::<(&Position, &mut Position)>()
                .for_each(|(_, _)| {});
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world
                .query::<(&mut Position, &mut Position)>()
                .for_each(|(_, _)| {});
        }))
        .is_err()
    );
}

#[test]
fn drop_runs_on_despawn() {
    let drops = Arc::new(AtomicI32::new(0));
    let mut world = World::new();
    let entity = world.spawn((Position(0.0, 0.0), DropCounter(drops.clone())));
    world.despawn(entity);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn drop_runs_when_world_drops() {
    let drops = Arc::new(AtomicI32::new(0));
    let mut world = World::new();
    world.spawn((Position(0.0, 0.0), DropCounter(drops.clone())));
    world.spawn((Position(1.0, 1.0), DropCounter(drops.clone())));
    drop(world);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}

#[test]
fn bundle_partial_tuples() {
    let mut world = World::new();
    let a = world.spawn((Position(1.0, 2.0), Mass(3.0)));
    assert_eq!(world.get::<Position>(a), Some(&Position(1.0, 2.0)));
    assert_eq!(world.get::<Mass>(a), Some(&Mass(3.0)));
    assert_eq!(world.query::<(&Position,)>().count(), 1);
    assert_eq!(world.query::<&Position>().count(), 1);
}

#[test]
fn spawn_duplicate_component_panics() {
    let mut world = World::new();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.spawn((Position(0.0, 0.0), Position(1.0, 1.0)));
        }))
        .is_err()
    );
}

#[test]
fn query_only_sees_alive_entities() {
    let mut world = World::new();
    let a = world.spawn((Position(0.0, 0.0),));
    let b = world.spawn((Position(1.0, 1.0),));
    world.despawn(a);
    let mut positions = Vec::new();
    world.query::<&Position>().for_each(|p| positions.push(p.0));
    assert_eq!(positions, vec![1.0]);
    assert!(world.alive(b));
}
