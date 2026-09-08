# Examples

Run any of them with:

```bash
cargo run --example <name> --release
```

A GPU with wgpu support is required.

| Example | Description |
| --- | --- |
| [falling_shapes](falling_shapes.rs) | 128 rigid bodies of mixed shapes (sphere, cuboid, capsule, cylinder) fall onto a ground plane, demonstrating spawning bodies and basic dynamics. |
| [joints](joints.rs) | A ball chain, a distance-constrained pendulum, and a motor-driven revolute joint, demonstrating constraints. |
| [queries](queries.rs) | Interactive raycasts and sphere queries: left-click casts a ray and applies an impulse to the hit body, right-click highlights all bodies in a sphere. |
| [sensors](sensors.rs) | Sensor gates that count falling bodies passing through them via contact events. |
| [mechanisms](mechanisms.rs) | Six-dof servo arm, cone-limited pendulum and prismatic servo slider — left-click spawns bodies (grow reserves capacity without stalls) and right-click grows the reservation. |
