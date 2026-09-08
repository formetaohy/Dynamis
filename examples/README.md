# Examples

Run any of them with:

```bash
cargo run --example <name> --release
```

A GPU with wgpu support is required.

| Example | Description |
| --- | --- |
| [falling](falling.rs) | 128 rigid bodies of mixed shapes (sphere, cuboid, capsule, cylinder) fall onto a ground plane, demonstrating spawning bodies and basic dynamics. |
| [benchmark](benchmark.rs) | Headless GPU benchmark reporting per-step CPU submit and GPU drain timings plus body-step throughput. |
