@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(2) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(3) var<storage, read_write> constraint_rows: array<ConstraintRows>;

fn work(index: u32) {
    let descriptor = constraint_descs[index];
    constraint_rows[index].first_row = row_of_body[descriptor.first_body_id];
    constraint_rows[index].second_row = row_of_body[descriptor.second_body_id];
}
