use dynamis_gpu::{BindingKind, parse_bindings};
use std::panic::catch_unwind;

#[test]
fn parses_every_binding_class() {
    let source = "
@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> bodies: array<BodyState>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(0) var<storage, read> shapes: array<ShapeSource>;
";
    let bindings = parse_bindings(source);
    assert_eq!(bindings.len(), 4);
    assert_eq!(bindings[0].name, "params");
    assert_eq!(bindings[0].kind, BindingKind::Uniform);
    assert_eq!(bindings[1].name, "bodies");
    assert_eq!(bindings[1].kind, BindingKind::ReadOnlyStorage);
    assert_eq!(bindings[2].name, "counts");
    assert_eq!(bindings[2].kind, BindingKind::ReadWriteStorage);
    assert_eq!(bindings[3].group, 1);
    assert_eq!(bindings[3].binding, 0);
}

#[test]
fn rejects_a_repeated_binding() {
    let source = "
@group(0) @binding(0) var<storage, read> first: array<u32>;
@group(0) @binding(0) var<storage, read> second: array<u32>;
";
    assert!(catch_unwind(|| parse_bindings(source)).is_err());
}

#[test]
fn rejects_an_unknown_binding_class() {
    let source = "@group(0) @binding(0) var<storage, read_write, extra> data: array<u32>;";
    assert!(catch_unwind(|| parse_bindings(source)).is_err());
    let source = "@group(0) @binding(0) var<immediate> data: array<u32>;";
    assert!(catch_unwind(|| parse_bindings(source)).is_err());
}

#[test]
fn rejects_a_truncated_binding() {
    assert!(catch_unwind(|| parse_bindings("@group(0) @binding(0) var<storage, read>")).is_err());
    assert!(catch_unwind(|| parse_bindings("@group(0) @binding(")).is_err());
}

#[test]
fn rejects_a_nameless_binding() {
    let source = "@group(0) @binding(0) var<storage, read> : array<u32>;";
    assert!(catch_unwind(|| parse_bindings(source)).is_err());
}
