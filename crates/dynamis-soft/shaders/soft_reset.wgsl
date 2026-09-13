@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> elements: array<SoftElement>;

fn work(index: u32) {
    elements[index].lambda = 0.0;
}
