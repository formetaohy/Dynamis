use super::common::{DT, gpu, gravity_config, host};
use dynamis_abi::BodyStateRecord;
use dynamis_gpu::WarmupBudget;
use dynamis_model::BodyDesc;
use dynamis_world::World;
use std::mem::size_of;

#[test]
fn a_host_pipeline_reads_the_engine_state_on_its_own_device() {
    let host = host();
    let mut world = World::new(gpu(), gravity_config());
    world.warmup(WarmupBudget::All);
    let body = world.spawn(BodyDesc::sphere(0.5).position([0.0, 10.0, 0.0]));
    for _ in 0..30 {
        world.step(DT);
    }
    let bytes = dynamis_gpu::read_regions(
        &host.device,
        &host.queue,
        "host state read",
        &[(
            world.state_buffer().buffer(),
            0,
            size_of::<BodyStateRecord>() as u64,
        )],
    );
    let state = dynamis_abi::decode::<BodyStateRecord>(&bytes)
        .first()
        .copied()
        .expect("the state stream holds the spawned body");
    assert_eq!(
        (state.body_id, state.generation),
        (body.id, body.generation)
    );
    assert!(
        state.position[1] < 9.9,
        "the host must observe gravity through the engine's own state stream, {}",
        state.position[1]
    );
}
