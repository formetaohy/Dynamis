//! Convex hulls and convex decomposition of triangle meshes.

mod decompose;
mod hull;

pub use decompose::{HullDecomposeSettings, HullMesh, decompose_mesh};
pub use hull::convex_hull_mesh;
