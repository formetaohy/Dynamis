@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(4) var<storage, read> constraint_rows: array<ConstraintRows>;
@group(0) @binding(5) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(6) var<storage, read_write> joint_states: array<JointState>;

fn work(index: u32) {
    let constraint = constraint_descs[index];
    let rows = constraint_rows[index];
    let first = Body(body_states[rows.first_row], body_descs[rows.first_row]);
    let second = Body(body_states[rows.second_row], body_descs[rows.second_row]);
    let anchor_a = constraint_anchor(first, constraint.anchor_a);
    let anchor_b = constraint_anchor(second, constraint.anchor_b);
    let runtime = constraint_runtime[index];
    let dofs = joint_dof_count(constraint.kind);
    var state: JointState;
    state.dof_count = dofs;
    for (var dof = 0u; dof < JOINT_DOF; dof = dof + 1u) {
        if (dof < dofs) {
            state.coordinates[dof] = joint_coordinate(
                constraint,
                first,
                second,
                anchor_a,
                anchor_b,
                dof,
            );
            state.rates[dof] = joint_rate(constraint, first, second, anchor_a, anchor_b, dof);
            state.impulses[dof] = joint_impulse(constraint, runtime, dof);
        } else {
            state.coordinates[dof] = 0.0;
            state.rates[dof] = 0.0;
            state.impulses[dof] = 0.0;
        }
    }
    joint_states[index] = state;
}
