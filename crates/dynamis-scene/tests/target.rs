use dynamis_abi::{ENTRY_INDEX_MASK, ENTRY_KIND_COLLIDER, ENTRY_KIND_PARTICLE, ENTRY_KIND_SHIFT};
use dynamis_model::{BodyHandle, SceneTarget, SoftBodyHandle};
use dynamis_scene::{scene_slot, scene_target};

fn packed(kind: u32, slot: u32) -> u32 {
    (kind << ENTRY_KIND_SHIFT) | slot
}

#[test]
fn a_packed_collider_target_decodes_into_the_body_that_owns_it() {
    let body = BodyHandle {
        id: 7,
        generation: 3,
    };
    let target = scene_target(
        packed(ENTRY_KIND_COLLIDER, 5),
        body.id,
        body.generation,
        |owner, slot| {
            assert_eq!(owner, body);
            slot * 2
        },
        |_, _| panic!("a collider target must not resolve a particle"),
    );
    assert_eq!(
        target,
        SceneTarget::Collider {
            body,
            collider: 10,
        }
    );
}

#[test]
fn a_packed_particle_target_decodes_into_the_soft_body_that_owns_it() {
    let body = SoftBodyHandle {
        id: 11,
        generation: 4,
    };
    let target = scene_target(
        packed(ENTRY_KIND_PARTICLE, 9),
        body.id,
        body.generation,
        |_, _| panic!("a particle target must not resolve a collider"),
        |owner, slot| {
            assert_eq!(owner, body);
            slot * 3
        },
    );
    assert_eq!(
        target,
        SceneTarget::Particle {
            body,
            particle: 27,
        }
    );
}

#[test]
fn a_scene_slot_is_the_payload_below_the_kind_bits() {
    for kind in [ENTRY_KIND_COLLIDER, ENTRY_KIND_PARTICLE] {
        assert_eq!(scene_slot(packed(kind, 1234)), 1234 & ENTRY_INDEX_MASK);
    }
}
