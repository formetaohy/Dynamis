use super::World;
use crate::shape_pool::{ShapePool, height_field_triangles};
use dynamis_abi::{SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH};
use dynamis_hull::hull;
use dynamis_model::{Shape, ShapeSourceHandle, SolidGeometry, SurfaceDesc, SurfaceTable};

#[derive(Clone)]
pub(crate) struct Shapes {
    pub(crate) pool: ShapePool,
    pub(crate) dirty: bool,
    pub(crate) uploaded: bool,
}

impl Shapes {
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
        let (vertices, triangles) = height_field_triangles(rows, cols, heights, cell_size);
        let expanded = height_field_surfaces(surfaces, rows, cols);
        let table = expanded
            .as_ref()
            .map(|(palette, indices)| SurfaceTable::new(palette, indices));
        self.allocate_shape(SHAPE_HEIGHTFIELD, &vertices, triangles, table)
    }

    pub fn remove_shape(&mut self, handle: ShapeSourceHandle) {
        self.shapes.pool.remove(handle);
        self.shapes.dirty = true;
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
        self.shapes.dirty = true;
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
        let (vertices, triangles) = height_field_triangles(rows, cols, heights, cell_size);
        let expanded = height_field_surfaces(surfaces, rows, cols);
        let table = expanded
            .as_ref()
            .map(|(palette, indices)| SurfaceTable::new(palette, indices));
        self.update_mesh(handle, &vertices, &triangles, table);
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
        self.shapes.dirty = true;
        handle
    }

    pub(super) fn shape_solid(&self, shape: &Shape) -> Option<SolidGeometry> {
        match shape {
            Shape::Hull(handle) => Some(self.shapes.pool.solid(*handle)),
            _ => None,
        }
    }
}

fn height_field_surfaces(
    surfaces: Option<SurfaceTable<'_>>,
    rows: u32,
    cols: u32,
) -> Option<(Vec<SurfaceDesc>, Vec<u32>)> {
    let table = surfaces?;
    let cells = rows.saturating_sub(1) * cols.saturating_sub(1);
    assert_eq!(
        table.count(),
        cells as usize,
        "a height field carries one surface per cell"
    );
    let palette = table.palette().to_vec();
    let mut indices = Vec::with_capacity(table.count() * 2);
    for index in table.indices() {
        indices.push(*index);
        indices.push(*index);
    }
    Some((palette, indices))
}
