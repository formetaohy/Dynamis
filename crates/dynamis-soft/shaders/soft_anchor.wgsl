fn rigid_anchor(body: Body, local: vec3f) -> vec3f {
    return body.state.position + quat_rotate(body.state.orientation, local);
}
