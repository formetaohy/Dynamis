use dynamis_abi::{ENTRY_INDEX_MASK, ENTRY_KIND_COLLIDER, ENTRY_KIND_PARTICLE, ENTRY_KIND_SHIFT};
use dynamis_model::{BodyHandle, SceneTarget, SoftBodyHandle};

pub const fn scene_slot(packed: u32) -> u32 {
    packed & ENTRY_INDEX_MASK
}

pub fn scene_target(
    packed: u32,
    id: u32,
    generation: u32,
    collider_of: impl Fn(BodyHandle, u32) -> u32,
    particle_of: impl Fn(SoftBodyHandle, u32) -> u32,
) -> SceneTarget {
    let slot = scene_slot(packed);
    match packed >> ENTRY_KIND_SHIFT {
        ENTRY_KIND_COLLIDER => {
            let body = BodyHandle { id, generation };
            SceneTarget::Collider {
                body,
                collider: collider_of(body, slot),
            }
        }
        ENTRY_KIND_PARTICLE => {
            let body = SoftBodyHandle { id, generation };
            SceneTarget::Particle {
                body,
                particle: particle_of(body, slot),
            }
        }
        kind => panic!("a scene target reports an unknown kind {kind}"),
    }
}
