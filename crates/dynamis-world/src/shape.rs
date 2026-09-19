use super::World;
use crate::shape_pool::{ShapePool, height_field_vertices};
use dynamis_abi::{SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH};
use dynamis_hull::hull;
use dynamis_model::{Shape, ShapeSourceHandle, SolidGeometry, SurfaceTable};

#[derive(Clone)]
pub(crate) struct ShapeStore {
    pub(crate) pool: ShapePool,
    pub(crate) dirty: bool,
    pub(crate) uploaded: bool,
}

impl ShapeStore {
    pub(crate) fn new() -> Self {
        Self {
            pool: ShapePool::new(),
            dirty: false,
            uploaded: false,
        }
    }
}

impl World {
    pub fn add_hull(&mut self, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> ShapeSourceHandle {
        self.allocate_shape(SHAPE_HULL, vertices, triangles.to_vec(), None)
    }

    pub fn add_mesh(
        &mut self,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        surfaces: Option<SurfaceTable<'_>>,
    ) -> ShapeSourceHandle {
        self.allocate_shape(SHAPE_MESH, vertices, triangles.to_vec(), surfaces)
    }

    pub fn add_decomposed_mesh(
        &mut self,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        settings: dynamis_hull::DecomposeSettings,
    ) -> Vec<ShapeSourceHandle> {
        let parts = dynamis_hull::decompose(vertices, triangles, &settings);
        parts
            .iter()
            .map(|part| self.add_hull(&part.vertices, &part.triangles))
            .collect()
    }

    pub fn add_hull_from_mesh(
        &mut self,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
    ) -> ShapeSourceHandle {
        let (hull_vertices, hull_triangles) = hull(vertices, triangles);
        self.allocate_shape(SHAPE_HULL, &hull_vertices, hull_triangles, None)
    }

    pub fn add_height_field(
        &mut self,
        rows: u32,
        cols: u32,
        heights: &[f32],
        cell_size: [f32; 2],
        surfaces: Option<SurfaceTable<'_>>,
    ) -> ShapeSourceHandle {
        let vertices = height_field_vertices(rows, cols, heights, cell_size);
        let handle =
            self.shapes
                .pool
                .allocate_grid(SHAPE_HEIGHTFIELD, rows, cols, &vertices, surfaces);
        self.declare_shape_geometry();
        handle
    }

    pub fn remove_shape(&mut self, handle: ShapeSourceHandle) {
        self.shapes.pool.remove(handle);
        self.declare_shape_geometry();
    }

    pub fn update_mesh(
        &mut self,
        handle: ShapeSourceHandle,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        surfaces: Option<SurfaceTable<'_>>,
    ) {
        self.shapes
            .pool
            .update_mesh(handle, vertices, triangles, surfaces);
        self.declare_shape_geometry();
        self.encode_shape_readers(handle);
    }

    pub fn update_height_field(
        &mut self,
        handle: ShapeSourceHandle,
        rows: u32,
        cols: u32,
        heights: &[f32],
        cell_size: [f32; 2],
        surfaces: Option<SurfaceTable<'_>>,
    ) {
        let vertices = height_field_vertices(rows, cols, heights, cell_size);
        self.shapes
            .pool
            .update_grid(handle, rows, cols, &vertices, surfaces);
        self.declare_shape_geometry();
        self.encode_shape_readers(handle);
    }

    fn allocate_shape(
        &mut self,
        kind: u32,
        vertices: &[[f32; 3]],
        triangles: Vec<[u32; 3]>,
        surfaces: Option<SurfaceTable<'_>>,
    ) -> ShapeSourceHandle {
        let handle = self
            .shapes
            .pool
            .allocate(kind, vertices, &triangles, surfaces);
        self.declare_shape_geometry();
        handle
    }

    /// Declares that the geometry a shape source answers has moved. A grid entry is keyed at a
    /// resolution the bounds of every source reachable from a collider answer, so the source that
    /// moved owes the index a derivation whether or not a collider already reads it.
    fn declare_shape_geometry(&mut self) {
        self.shapes.dirty = true;
        self.facts.geometry += 1;
    }

    pub(super) fn shape_solid(&self, shape: &Shape) -> Option<SolidGeometry> {
        match shape {
            Shape::Hull(handle) => Some(self.shapes.pool.solid(*handle)),
            _ => None,
        }
    }
}
