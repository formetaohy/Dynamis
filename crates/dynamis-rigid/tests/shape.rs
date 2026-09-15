use dynamis_abi::Census;
use dynamis_rigid::RigidShape;

fn counts(bodies: u32, colliders: u32) -> Census {
    Census {
        dynamic_bodies: bodies,
        bodies,
        body_ids: bodies,
        colliders,
        ..Census::default()
    }
}

#[test]
fn island_rounds_cover_the_widest_component_a_live_row_can_form() {
    for bodies in [
        0u32,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        255,
        256,
        257,
        4096,
        65535,
        65536,
        1 << 20,
    ] {
        let shape = RigidShape::of(&counts(bodies, 1));
        let reach = 1u64 << shape.island_rounds;
        assert!(
            reach > u64::from(bodies),
            "one label hop must double per round, so {reach} has to outrun a {bodies} body component"
        );
    }
}

#[test]
fn key_words_cover_the_row_and_slot_spaces_a_step_sorts() {
    for rows in [1u32, 255, 256, 257, 65_535, 65_536, 16_777_215] {
        let shape = RigidShape::of(&counts(rows, rows));
        for words in [shape.body_row_words, shape.collider_slot_words] {
            let reach = 1u64 << (8 * words);
            assert!(
                reach >= u64::from(rows),
                "one digit per key byte must span every row of a {rows} row stream"
            );
        }
    }
}

#[test]
fn key_words_cover_the_body_id_space_a_resting_index_sorts() {
    for ids in [1u32, 255, 256, 257, 65_535, 65_536, 16_777_215] {
        let shape = RigidShape::of(&Census {
            body_ids: ids,
            ..counts(1, 1)
        });
        let reach = 1u64 << (8 * shape.body_id_words);
        assert!(
            reach >= u64::from(ids),
            "a resting index keyed by body ids must span every id of a {ids} id space"
        );
    }
}

#[test]
fn the_id_key_space_outgrows_the_row_key_space_it_no_longer_holds() {
    let shape = RigidShape::of(&Census {
        body_ids: 257,
        ..counts(2, 2)
    });
    assert_eq!(
        (shape.body_row_words, shape.collider_slot_words),
        (1, 1),
        "two live rows and two live collider slots fit one key byte"
    );
    assert_eq!(
        shape.body_id_words, 2,
        "the id space reaches past one key byte"
    );
}

#[test]
fn the_shape_ignores_every_count_it_does_not_schedule() {
    let scheduled = counts(9, 5);
    let shape = RigidShape::of(&scheduled);
    for irrelevant in [
        Census {
            constraints: 4096,
            ..scheduled
        },
        Census {
            particles: 65_536,
            elements: 4096,
            ..scheduled
        },
    ] {
        assert_eq!(
            RigidShape::of(&irrelevant),
            shape,
            "a count that carries no key space and no island may not shape the step"
        );
    }
}

#[test]
fn the_shape_grows_with_the_live_rows_alone() {
    let small = RigidShape::of(&counts(200, 200));
    let large = RigidShape::of(&counts(20_000, 20_000));
    assert!(small.body_row_words < large.body_row_words);
    assert!(small.collider_slot_words < large.collider_slot_words);
    assert_eq!(small.island_rounds, 8);
    assert_eq!(large.island_rounds, 15);
}
