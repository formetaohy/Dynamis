use dynamis_gpu::BindingKind;
use dynamis_shader::{Dispatch, Program, reflect};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn reflects_every_binding_class() {
    let source = "
struct StepParams {
    dt: f32,
}

struct BodyState {
    position: vec4f,
}

struct ShapeSource {
    kind: u32,
}

@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> bodies: array<BodyState>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(0) var<storage, read> shapes: array<ShapeSource>;
";
    let bindings = reflect(source);
    assert_eq!(bindings.len(), 4);
    assert_eq!(bindings[0].group, 0);
    assert_eq!(bindings[0].binding, 0);
    assert_eq!(bindings[0].name, "params");
    assert_eq!(bindings[0].kind, BindingKind::Uniform);
    assert_eq!(bindings[0].element, "StepParams");
    assert_eq!(bindings[1].name, "bodies");
    assert_eq!(bindings[1].kind, BindingKind::ReadOnlyStorage);
    assert_eq!(bindings[1].element, "BodyState");
    assert_eq!(bindings[2].name, "counts");
    assert_eq!(bindings[2].kind, BindingKind::ReadWriteStorage);
    assert_eq!(bindings[2].element, "u32");
    assert_eq!(bindings[3].group, 1);
    assert_eq!(bindings[3].binding, 0);
}

#[test]
fn reflects_the_element_a_binding_exposes() {
    let source = "
struct QueryResult {
    hit: u32,
}

@group(0) @binding(0) var<storage, read_write> sized: array<atomic<u32>, 2048>;
@group(0) @binding(1) var<storage, read_write> words: array<vec4f, 4>;
@group(0) @binding(2) var<storage, read> records: array<QueryResult>;
";
    let bindings = reflect(source);
    assert_eq!(bindings[0].element, "u32");
    assert_eq!(bindings[1].element, "vec4f");
    assert_eq!(bindings[2].element, "QueryResult");
}

#[test]
fn reflects_a_binding_whatever_the_attribute_order() {
    let source = "@binding(3) @group(2) var<storage, read> words: array<f32>;";
    let bindings = reflect(source);
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].group, 2);
    assert_eq!(bindings[0].binding, 3);
    assert_eq!(bindings[0].element, "f32");
}

#[test]
fn reflects_a_binding_past_a_comment() {
    let source = "
// @group(7) @binding(7) var<storage, read> commented: array<u32>;
@group(0) @binding(0) var<storage, read> words: array<u32>;
";
    let bindings = reflect(source);
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].name, "words");
}

#[test]
fn rejects_a_repeated_binding() {
    let source = "
@group(0) @binding(0) var<storage, read> first: array<u32>;
@group(0) @binding(0) var<storage, read> second: array<u32>;
";
    assert!(catch_unwind(AssertUnwindSafe(|| reflect(source))).is_err());
}

#[test]
fn rejects_a_repeated_name() {
    let source = "
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> words: array<u32>;
";
    assert!(catch_unwind(AssertUnwindSafe(|| reflect(source))).is_err());
}

#[test]
fn rejects_an_array_of_arrays() {
    let source = "@group(0) @binding(0) var<storage, read> nested: array<array<f32, 4>>;";
    assert!(catch_unwind(AssertUnwindSafe(|| reflect(source))).is_err());
}

#[test]
fn rejects_a_shader_that_is_not_valid_wgsl() {
    assert!(
        catch_unwind(AssertUnwindSafe(|| reflect(
            "@group(0) @binding(0) var<storage"
        )))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| reflect(
            "@group(0) @binding(0) var<storage, read> : array<u32>;"
        )))
        .is_err(),
        "every shader binding needs a name"
    );
}

#[test]
fn rejects_a_binding_outside_the_buffer_spaces() {
    let source = "@group(0) @binding(0) var<workgroup> words: array<u32>;";
    assert!(catch_unwind(AssertUnwindSafe(|| reflect(source))).is_err());
}

#[test]
fn a_program_carries_the_bindings_of_its_source() {
    let source = "@group(0) @binding(0) var<storage, read_write> words: array<u32>;";
    let program = Program::new(source.to_owned(), Dispatch::Rows, false);
    let (assembled, bindings, dispatch, warm) = program.into_parts();
    assert_eq!(assembled.as_ref(), source);
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].name, "words");
    assert_eq!(bindings[0].kind, BindingKind::ReadWriteStorage);
    assert_eq!(bindings[0].element, "u32");
    assert!(matches!(dispatch, Dispatch::Rows));
    assert!(!warm);
}
