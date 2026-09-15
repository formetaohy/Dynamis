use dynamis_model::CollisionFilter;

#[test]
fn a_filter_only_admits_mutual_group_and_mask_agreement() {
    let first = CollisionFilter::new(0b0010, 0b0100);
    let second = CollisionFilter::new(0b0100, 0b0010);
    assert!(
        first.intersects(second) && second.intersects(first),
        "both sides agreeing on one another's group must intersect"
    );

    let one_way = CollisionFilter::new(0b0010, 0b0100);
    let other = CollisionFilter::new(0b1000, 0b0010);
    assert!(
        !one_way.intersects(other) && !other.intersects(one_way),
        "a mask that names no counterpart group must exclude the pair"
    );
}

#[test]
fn a_zero_group_collides_with_nothing() {
    let void = CollisionFilter::new(0, u32::MAX);
    assert!(!void.intersects(CollisionFilter::DEFAULT));
    assert!(!void.intersects(void));
}

#[test]
fn the_default_filter_admits_every_default_participant() {
    let other = CollisionFilter::new(0x8000_0000, u32::MAX);
    assert!(CollisionFilter::DEFAULT.intersects(other));
    assert!(CollisionFilter::DEFAULT.intersects(CollisionFilter::DEFAULT));
}

#[test]
fn a_filter_rebuilds_one_half_without_disturbing_the_other() {
    let filter = CollisionFilter::new(3, 5).with_group(7).with_mask(9);
    assert_eq!(filter.group(), 7);
    assert_eq!(filter.mask(), 9);
}
