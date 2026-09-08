# Examples

Run any of them with:

```bash
cargo run --example <name> --release
```

A GPU with wgpu support is required.

| Example | Description |
| --- | --- |
| [falling](falling.rs) | 128 rigid bodies of mixed shapes (sphere, cuboid, capsule, cylinder) fall onto a ground plane, demonstrating spawning bodies and basic dynamics. |
| [benchmark](benchmark.rs) | Headless GPU benchmark reporting per-step CPU submit, per-pass GPU durations and body-step throughput. |

`benchmark` exists only with the `profile` feature, which is what compiles the per-pass GPU timestamp instrumentation into the engine:

```bash
cargo run --example benchmark --release --features profile -- 128 600
```
