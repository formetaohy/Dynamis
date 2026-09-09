use super::Simulation;
use crate::shape_pool::height_field_triangles;
use dynamis_layout::{SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH};
use dynamis_mesh::convex_hull_mesh;
use dynamis_model::{MAX_COLLIDERS_PER_BODY, Shape, ShapeSourceHandle};

impl Simulation {
    pub fn add_hull(&mut self, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> ShapeSourceHandle {
        self.allocate_shape(SHAPE_HULL, vertices, triangles.to_vec())
    }

    pub fn add_mesh(&mut self, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> ShapeSourceHandle {
        self.allocate_shape(SHAPE_MESH, vertices, triangles.to_vec())
    }

    pub fn add_decomposed_mesh(
        &mut self,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        settings: crate::HullDecomposeSettings,
    ) -> Vec<ShapeSourceHandle> {
        assert!(
            settings.max_parts as usize <= MAX_COLLIDERS_PER_BODY,
            "decomposition part limit must fit within one body"
        );
        let parts = dynamis_mesh::decompose_mesh(vertices, triangles, &settings);
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
        let (hull_vertices, hull_triangles) = convex_hull_mesh(vertices, triangles);
        self.allocate_shape(SHAPE_HULL, &hull_vertices, hull_triangles)
    }

    pub fn add_height_field(
        &mut self,
        rows: u32,
        cols: u32,
        heights: &[f32],
        cell_size: [f32; 2],
    ) -> ShapeSourceHandle {
        let (vertices, triangles) = height_field_triangles(rows, cols, heights, cell_size);
        self.allocate_shape(SHAPE_HEIGHTFIELD, &vertices, triangles)
    }

    pub fn remove_shape(&mut self, handle: ShapeSourceHandle) {
        self.shape_pool.remove(handle);
        self.shapes_dirty = true;
    }

    pub fn update_mesh(
        &mut self,
        handle: ShapeSourceHandle,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
    ) {
        self.shape_pool.update_mesh(handle, vertices, triangles);
        self.shape_pool.upload_update(
            self.gpu.queue(),
            &self.buffers.shapes,
            &self.buffers.shape_vertices,
            &self.buffers.shape_triangles,
            &self.buffers.shape_nodes,
            handle,
        );
    }

    pub fn update_height_field(
        &mut self,
        handle: ShapeSourceHandle,
        rows: u32,
        cols: u32,
        heights: &[f32],
        cell_size: [f32; 2],
    ) {
        let (vertices, triangles) = height_field_triangles(rows, cols, heights, cell_size);
        self.update_mesh(handle, &vertices, &triangles);
    }

    fn allocate_shape(
        &mut self,
        kind: u32,
        vertices: &[[f32; 3]],
        triangles: Vec<[u32; 3]>,
    ) -> ShapeSourceHandle {
        let handle = self.shape_pool.allocate(kind, vertices, &triangles);
        self.shapes_dirty = true;
        handle
    }

    pub(super) fn shape_bounds(&self, shape: &Shape) -> Option<([f32; 3], [f32; 3])> {
        match shape {
            Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) => {
                Some(self.shape_pool.record(*handle).bounds)
            }
            _ => None,
        }
    }
}
