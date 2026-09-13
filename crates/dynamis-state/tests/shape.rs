use dynamis_abi::FrameCounts;
use dynamis_state::StepShape;

fn counts(bodies: u32, colliders: u32) -> FrameCounts {
    FrameCounts {
        dynamic_bodies: bodies,
        bodies,
        colliders,
        ..FrameCounts::default()
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
        let shape = StepShape::of(&counts(bodies, 1));
        let reach = 1u64 << shape.island_rounds;
        assert!(
            reach > u64::from(bodies),
            "one label hop must double per round, so {reach} has to outrun a {bodies} body component"
        );
    }
}

#[test]
fn key_words_cover_the_row_index_space() {
    for rows in [1u32, 255, 256, 257, 65_535, 65_536, 16_777_215] {
        let shape = StepShape::of(&counts(rows, rows));
        for words in [shape.body_words, shape.collider_words] {
            let reach = 1u64 << (8 * words);
            assert!(
                reach >= u64::from(rows),
                "one digit per key byte must span every row of a {rows} row stream"
            );
        }
    }
}

#[test]
fn the_shape_ignores_every_count_it_does_not_schedule() {
    let scheduled = counts(9, 5);
    let shape = StepShape::of(&scheduled);
    for irrelevant in [
        FrameCounts {
            constraints: 4096,
            ..scheduled
        },
        FrameCounts {
            particles: 65_536,
            elements: 4096,
            ..scheduled
        },
    ] {
        assert_eq!(
            StepShape::of(&irrelevant),
            shape,
            "a count that carries no key space and no island may not shape the step"
        );
    }
}

#[test]
fn the_shape_grows_with_the_live_rows_alone() {
    let small = StepShape::of(&counts(200, 200));
    let large = StepShape::of(&counts(20_000, 20_000));
    assert!(small.body_words < large.body_words);
    assert!(small.collider_words < large.collider_words);
    assert_eq!(small.island_rounds, 8);
    assert_eq!(large.island_rounds, 15);
}
