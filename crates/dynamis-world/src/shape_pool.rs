use super::arena::{Arena, Run, merged};
use super::pool::Pool;
use bytemuck::Zeroable;
use dynamis_abi::{BvhNodeRecord, CellRecord, SurfaceRecord, TriangleRecord};
use dynamis_model::{ShapeSourceHandle, SolidGeometry, SurfaceDesc, SurfaceTable};
use dynamis_state::ShapeCapacity;
use dynamis_state::{CELL_BYTES, TRIANGLE_BYTES, VERTEX_BYTES};
use std::mem::size_of;

#[derive(Clone)]
struct Table<T> {
    arena: Arena,
    rows: Vec<T>,
    dirty: Vec<Run>,
}

impl<T: Copy> Table<T> {
    const fn new() -> Self {
        Self {
            arena: Arena::new(),
            rows: Vec::new(),
            dirty: Vec::new(),
        }
    }

    fn used(&self) -> u32 {
        self.arena.used()
    }

    fn slice(&self, run: Run) -> &[T] {
        &self.rows[run.span()]
    }

    fn take(&mut self, blank: T, values: &[T]) -> Run {
        if values.is_empty() {
            return Run::EMPTY;
        }
        let run = self.arena.take(values.len() as u32);
        self.rows.resize(self.arena.used() as usize, blank);
        self.rows[run.span()].copy_from_slice(values);
        self.dirty.push(run);
        run
    }

    fn rewrite(&mut self, run: Run, blank: T, values: &[T]) -> Run {
        if values.is_empty() {
            self.release(run, blank);
            return Run::EMPTY;
        }
        if run.len as usize == values.len() {
            self.rows[run.span()].copy_from_slice(values);
            self.dirty.push(run);
            return run;
        }
        self.release(run, blank);
        self.take(blank, values)
    }

    fn release(&mut self, run: Run, blank: T) {
        if run.len == 0 {
            return;
        }
        self.rows[run.span()].fill(blank);
        self.arena.release(run);
        self.rows.truncate(self.arena.used() as usize);
    }

    fn take_dirty(&mut self) -> Vec<Run> {
        merged(std::mem::take(&mut self.dirty), self.arena.used())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Geometry {
    Vacant,
    Mesh { triangles: Run, nodes: Run },
    Grid { rows: u32, cols: u32 },
}

impl Geometry {
    const fn triangles(self) -> Run {
        match self {
            Self::Mesh { triangles, .. } => triangles,
            Self::Grid { .. } | Self::Vacant => Run::EMPTY,
        }
    }

    const fn nodes(self) -> Run {
        match self {
            Self::Mesh { nodes, .. } => nodes,
            Self::Grid { .. } | Self::Vacant => Run::EMPTY,
        }
    }

    const fn grid(self) -> Option<(u32, u32)> {
        match self {
            Self::Grid { rows, cols } => Some((rows, cols)),
            Self::Mesh { .. } | Self::Vacant => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Source {
    kind: u32,
    vertices: Run,
    geometry: Geometry,
    cells: Run,
    palette: Run,
    bounds: ([f32; 3], [f32; 3]),
    solid: Option<SolidGeometry>,
}

impl Source {
    const DEAD: Self = Self {
        kind: 0,
        vertices: Run::EMPTY,
        geometry: Geometry::Vacant,
        cells: Run::EMPTY,
        palette: Run::EMPTY,
        bounds: ([0.0; 3], [0.0; 3]),
        solid: None,
    };
}

#[derive(Clone)]
pub(crate) struct ShapePool {
    pool: Pool<ShapeSourceHandle>,
    refs: Vec<u32>,
    sources: Vec<Source>,
    dirty: Vec<u32>,
    vertices: Table<[f32; 4]>,
    triangles: Table<TriangleRecord>,
    nodes: Table<BvhNodeRecord>,
    cells: Table<CellRecord>,
    palettes: Table<SurfaceRecord>,
}

impl ShapePool {
    pub(crate) const fn new() -> Self {
        Self {
            pool: Pool::vacate("shape source"),
            refs: Vec::new(),
            sources: Vec::new(),
            dirty: Vec::new(),
            vertices: Table::new(),
            triangles: Table::new(),
            nodes: Table::new(),
            cells: Table::new(),
            palettes: Table::new(),
        }
    }

    pub(crate) fn used(&self) -> ShapeCapacity {
        ShapeCapacity {
            sources: self.sources.len() as u32,
            vertices: self.vertices.used(),
            triangles: self.triangles.used(),
            nodes: self.nodes.used(),
            cells: self.cells.used(),
        }
    }

    pub(crate) fn source_surface(&self, source: u32, index: u32) -> SurfaceDesc {
        let source = self
            .pool
            .handle_of(source)
            .map(|handle| &self.sources[handle.id as usize])
            .unwrap_or_else(|| panic!("shape source {source} is not alive"));
        assert!(
            index < source.palette.len,
            "surface index {index} is outside the shape source palette"
        );
        self.palettes.rows[source.palette.offset as usize + index as usize].desc()
    }

    pub(crate) fn solid(&self, handle: ShapeSourceHandle) -> SolidGeometry {
        self.source(handle)
            .solid
            .expect("a hull shape source carries its solid geometry")
    }

    fn source(&self, handle: ShapeSourceHandle) -> &Source {
        self.pool.validate(handle);
        &self.sources[handle.id as usize]
    }

    fn grow_to(&mut self, id: u32) {
        if self.sources.len() > id as usize {
            return;
        }
        self.refs.resize(id as usize + 1, 0);
        self.sources.resize(id as usize + 1, Source::DEAD);
    }

    pub(crate) fn retain(&mut self, handle: ShapeSourceHandle) {
        self.source(handle);
        self.refs[handle.id as usize] += 1;
    }

    pub(crate) fn release(&mut self, handle: ShapeSourceHandle) {
        self.source(handle);
        assert!(
            self.refs[handle.id as usize] > 0,
            "shape source refcount underflow"
        );
        self.refs[handle.id as usize] -= 1;
    }

    pub(crate) fn remove(&mut self, handle: ShapeSourceHandle) {
        let id = handle.id as usize;
        let source = *self.source(handle);
        assert!(
            self.refs[id] == 0,
            "shape source is still referenced by a live body"
        );
        self.vertices.release(source.vertices, [0.0; 4]);
        self.triangles
            .release(source.geometry.triangles(), TriangleRecord::zeroed());
        self.nodes
            .release(source.geometry.nodes(), BvhNodeRecord::zeroed());
        self.cells.release(source.cells, CellRecord::zeroed());
        self.palettes
            .release(source.palette, SurfaceRecord::zeroed());
        self.sources[id] = Source::DEAD;
        self.dirty.push(handle.id);
        self.pool.retire(handle);
    }

    pub(crate) fn allocate(
        &mut self,
        kind: u32,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        surfaces: Option<SurfaceTable<'_>>,
    ) -> ShapeSourceHandle {
        validate_geometry(kind, vertices, triangles, surfaces);
        let nodes = build_bvh(vertices, triangles);
        let bounds = bounds_of(vertices);
        let handle = self.pool.acquire();
        let id = handle.id;
        self.grow_to(id);
        let vertex_rows = vertex_rows(vertices);
        let triangle_rows = triangle_rows(triangles, surfaces);
        let palette_rows = palette_rows(surfaces);
        let vertex_run = self.vertices.take([0.0; 4], &vertex_rows);
        let triangle_run = self
            .triangles
            .take(TriangleRecord::zeroed(), &triangle_rows);
        let node_run = self.nodes.take(BvhNodeRecord::zeroed(), &nodes);
        let palette_run = self.palettes.take(SurfaceRecord::zeroed(), &palette_rows);
        self.sources[id as usize] = Source {
            kind,
            vertices: vertex_run,
            geometry: Geometry::Mesh {
                triangles: triangle_run,
                nodes: node_run,
            },
            cells: Run::EMPTY,
            palette: palette_run,
            bounds,
            solid: solid_of(kind, vertices, triangles),
        };
        self.dirty.push(id);
        self.pool.insert(handle);
        handle
    }

    pub(crate) fn allocate_grid(
        &mut self,
        kind: u32,
        rows: u32,
        cols: u32,
        vertices: &[[f32; 3]],
        surfaces: Option<SurfaceTable<'_>>,
    ) -> ShapeSourceHandle {
        validate_grid(rows, cols, vertices, surfaces);
        let bounds = bounds_of(vertices);
        let handle = self.pool.acquire();
        let id = handle.id;
        self.grow_to(id);
        let vertex_rows = vertex_rows(vertices);
        let cell_rows = cell_rows(surfaces);
        let palette_rows = palette_rows(surfaces);
        let vertex_run = self.vertices.take([0.0; 4], &vertex_rows);
        let cell_run = self.cells.take(CellRecord::zeroed(), &cell_rows);
        let palette_run = self.palettes.take(SurfaceRecord::zeroed(), &palette_rows);
        self.sources[id as usize] = Source {
            kind,
            vertices: vertex_run,
            geometry: Geometry::Grid { rows, cols },
            cells: cell_run,
            palette: palette_run,
            bounds,
            solid: None,
        };
        self.dirty.push(id);
        self.pool.insert(handle);
        handle
    }

    pub(crate) fn update_grid(
        &mut self,
        handle: ShapeSourceHandle,
        rows: u32,
        cols: u32,
        vertices: &[[f32; 3]],
        surfaces: Option<SurfaceTable<'_>>,
    ) {
        let source = *self.source(handle);
        assert!(
            matches!(source.geometry, Geometry::Grid { .. }),
            "a height field update requires a grid source"
        );
        validate_grid(rows, cols, vertices, surfaces);
        let bounds = bounds_of(vertices);
        let vertex_rows = vertex_rows(vertices);
        let cell_rows = cell_rows(surfaces);
        let palette_rows = palette_rows(surfaces);
        let vertex_run = self
            .vertices
            .rewrite(source.vertices, [0.0; 4], &vertex_rows);
        let cell_run = self
            .cells
            .rewrite(source.cells, CellRecord::zeroed(), &cell_rows);
        let palette_run =
            self.palettes
                .rewrite(source.palette, SurfaceRecord::zeroed(), &palette_rows);
        self.sources[handle.id as usize] = Source {
            kind: source.kind,
            vertices: vertex_run,
            geometry: Geometry::Grid { rows, cols },
            cells: cell_run,
            palette: palette_run,
            bounds,
            solid: None,
        };
        self.dirty.push(handle.id);
    }

    pub(crate) fn update_mesh(
        &mut self,
        handle: ShapeSourceHandle,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        surfaces: Option<SurfaceTable<'_>>,
    ) {
        let source = *self.source(handle);
        assert!(
            matches!(source.geometry, Geometry::Mesh { .. }),
            "a mesh update requires a mesh source"
        );
        validate_geometry(source.kind, vertices, triangles, surfaces);
        let nodes = build_bvh(vertices, triangles);
        let bounds = bounds_of(vertices);
        let vertex_rows = vertex_rows(vertices);
        let triangle_rows = triangle_rows(triangles, surfaces);
        let palette_rows = palette_rows(surfaces);
        let vertex_run = self
            .vertices
            .rewrite(source.vertices, [0.0; 4], &vertex_rows);
        let triangle_run = self.triangles.rewrite(
            source.geometry.triangles(),
            TriangleRecord::zeroed(),
            &triangle_rows,
        );
        let node_run = self
            .nodes
            .rewrite(source.geometry.nodes(), BvhNodeRecord::zeroed(), &nodes);
        let palette_run =
            self.palettes
                .rewrite(source.palette, SurfaceRecord::zeroed(), &palette_rows);
        self.sources[handle.id as usize] = Source {
            vertices: vertex_run,
            geometry: Geometry::Mesh {
                triangles: triangle_run,
                nodes: node_run,
            },
            cells: Run::EMPTY,
            palette: palette_run,
            bounds,
            solid: solid_of(source.kind, vertices, triangles),
            ..source
        };
        self.dirty.push(handle.id);
    }

    pub(crate) fn upload_pending(
        &mut self,
        queue: &wgpu::Queue,
        sources_buffer: &dynamis_gpu::Stream,
        vertices_buffer: &dynamis_gpu::Stream,
        triangles_buffer: &dynamis_gpu::Stream,
        nodes_buffer: &dynamis_gpu::Stream,
        cells_buffer: &dynamis_gpu::Stream,
    ) {
        let dirty = std::mem::take(&mut self.dirty);
        let runs = merged(
            dirty
                .into_iter()
                .map(|id| Run { offset: id, len: 1 })
                .collect(),
            self.sources.len() as u32,
        );
        for run in runs {
            let records = self.sources[run.span()]
                .iter()
                .map(ShapePool::layout_of)
                .collect::<Vec<_>>();
            sources_buffer.write_at(
                queue,
                run.offset as u64 * std::mem::size_of::<dynamis_abi::ShapeSourceRecord>() as u64,
                bytemuck::cast_slice(&records),
            );
        }
        for run in self.vertices.take_dirty() {
            vertices_buffer.write_at(
                queue,
                run.offset as u64 * VERTEX_BYTES,
                bytemuck::cast_slice(self.vertices.slice(run)),
            );
        }
        for run in self.triangles.take_dirty() {
            triangles_buffer.write_at(
                queue,
                run.offset as u64 * TRIANGLE_BYTES,
                bytemuck::cast_slice(self.triangles.slice(run)),
            );
        }
        for run in self.nodes.take_dirty() {
            nodes_buffer.write_at(
                queue,
                run.offset as u64 * size_of::<BvhNodeRecord>() as u64,
                bytemuck::cast_slice(self.nodes.slice(run)),
            );
        }
        for run in self.cells.take_dirty() {
            cells_buffer.write_at(
                queue,
                run.offset as u64 * CELL_BYTES,
                bytemuck::cast_slice(self.cells.slice(run)),
            );
        }
    }

    fn layout_of(source: &Source) -> dynamis_abi::ShapeSourceRecord {
        dynamis_abi::ShapeSourceRecord {
            kind: source.kind,
            vertex_offset: source.vertices.offset,
            vertex_count: source.vertices.len,
            triangle_offset: source.geometry.triangles().offset,
            node_offset: source.geometry.nodes().offset,
            node_count: source.geometry.nodes().len,
            cell_offset: source.cells.offset,
            cell_count: source.cells.len,
            grid_rows: source.geometry.grid().map_or(0, |grid| grid.0),
            grid_cols: source.geometry.grid().map_or(0, |grid| grid.1),
            _pad0: 0,
            _pad1: 0,
            local_min: source.bounds.0,
            _pad2: 0.0,
            local_max: source.bounds.1,
            _pad3: 0.0,
        }
    }
}

fn validate_geometry(
    kind: u32,
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
    surfaces: Option<SurfaceTable<'_>>,
) {
    assert!(
        kind != dynamis_abi::SHAPE_HULL || surfaces.is_none(),
        "a hull carries no addressable faces for surfaces"
    );
    if let Some(surfaces) = surfaces {
        assert_eq!(
            surfaces.count(),
            triangles.len(),
            "a surface table carries one surface per triangle"
        );
    }
    assert!(
        kind != dynamis_abi::SHAPE_HULL
            || vertices.len() <= dynamis_abi::FEATURE_INDEX_LIMIT as usize,
        "a hull must not exceed {} vertices so its features stay addressable",
        dynamis_abi::FEATURE_INDEX_LIMIT
    );
    assert!(
        triangles.len() <= dynamis_abi::FEATURE_TRIANGLE_MASK as usize,
        "a shape source must not exceed {} triangles so its features stay addressable",
        dynamis_abi::FEATURE_TRIANGLE_MASK
    );
}

fn validate_grid(rows: u32, cols: u32, vertices: &[[f32; 3]], surfaces: Option<SurfaceTable<'_>>) {
    assert!(
        rows >= 2 && cols >= 2,
        "a height field requires at least two rows and two columns"
    );
    assert_eq!(
        vertices.len() as u32,
        rows * cols,
        "a height field carries one vertex per sample"
    );
    assert!(
        surfaces.is_none_or(|table| table.count() == ((rows - 1) * (cols - 1)) as usize),
        "a height field carries one surface per cell"
    );
    assert!(
        (rows - 1) * (cols - 1) <= dynamis_abi::FEATURE_TRIANGLE_MASK / 2,
        "a height field must not exceed {} cells so its features stay addressable",
        dynamis_abi::FEATURE_TRIANGLE_MASK / 2
    );
}

fn vertex_rows(vertices: &[[f32; 3]]) -> Vec<[f32; 4]> {
    vertices
        .iter()
        .map(|vertex| [vertex[0], vertex[1], vertex[2], 0.0])
        .collect()
}

fn triangle_rows(
    triangles: &[[u32; 3]],
    surfaces: Option<SurfaceTable<'_>>,
) -> Vec<TriangleRecord> {
    triangles
        .iter()
        .enumerate()
        .map(|(slot, triangle)| {
            let (surface, material) = match surfaces {
                Some(table) => (
                    table.indices()[slot],
                    SurfaceRecord::build(&table.palette()[table.indices()[slot] as usize]),
                ),
                None => (dynamis_abi::NO_SURFACE, SurfaceRecord::zeroed()),
            };
            TriangleRecord {
                a: triangle[0],
                b: triangle[1],
                c: triangle[2],
                surface,
                material,
            }
        })
        .collect()
}

fn cell_rows(surfaces: Option<SurfaceTable<'_>>) -> Vec<CellRecord> {
    surfaces
        .map(|table| {
            table
                .indices()
                .iter()
                .map(|index| CellRecord {
                    surface: *index,
                    material: SurfaceRecord::build(&table.palette()[*index as usize]),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn palette_rows(surfaces: Option<SurfaceTable<'_>>) -> Vec<SurfaceRecord> {
    surfaces
        .map(|table| table.palette().iter().map(SurfaceRecord::build).collect())
        .unwrap_or_default()
}

fn solid_of(kind: u32, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Option<SolidGeometry> {
    (kind == dynamis_abi::SHAPE_HULL).then(|| dynamis_hull::solid(vertices, triangles))
}

fn bounds_of(vertices: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut bounds_min = [f32::MAX; 3];
    let mut bounds_max = [f32::MIN; 3];
    for vertex in vertices {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(vertex[axis]);
            bounds_max[axis] = bounds_max[axis].max(vertex[axis]);
        }
    }
    (bounds_min, bounds_max)
}

fn build_bvh(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<BvhNodeRecord> {
    assert!(
        !triangles.is_empty(),
        "a shape source requires at least one triangle"
    );
    let mut nodes = Vec::new();
    let mut order = (0..triangles.len() as u32).collect::<Vec<_>>();
    build_node(vertices, triangles, &mut order, &mut nodes);
    nodes
}

fn build_node(
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
    order: &mut [u32],
    nodes: &mut Vec<BvhNodeRecord>,
) {
    let count = order.len() as u32;
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for &triangle in order.iter() {
        let tri = triangles[triangle as usize];
        for corner in tri {
            let vertex = vertices[corner as usize];
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
    }
    let node_index = nodes.len() as u32;
    nodes.push(BvhNodeRecord {
        min,
        _pad0: 0.0,
        max,
        _pad1: 0.0,
        left: 0,
        right: 0,
        leaf: 0,
        _pad2: 0,
    });
    if count == 1 {
        nodes[node_index as usize].leaf = 1;
        nodes[node_index as usize].left = order[0];
        nodes[node_index as usize].right = 1;
        return;
    }
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    order.sort_unstable_by(|a, b| {
        let ca = centroid(triangles[*a as usize], vertices)[axis];
        let cb = centroid(triangles[*b as usize], vertices)[axis];
        ca.total_cmp(&cb)
    });
    let middle = (count / 2) as usize;
    let (left_order, right_order) = order.split_at_mut(middle);
    let left_index = nodes.len() as u32;
    build_node(vertices, triangles, left_order, nodes);
    let right_index = nodes.len() as u32;
    build_node(vertices, triangles, right_order, nodes);
    nodes[node_index as usize].left = left_index;
    nodes[node_index as usize].right = right_index;
}

fn centroid(triangle: [u32; 3], vertices: &[[f32; 3]]) -> [f32; 3] {
    let a = vertices[triangle[0] as usize];
    let b = vertices[triangle[1] as usize];
    let c = vertices[triangle[2] as usize];
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

pub fn height_field_vertices(
    rows: u32,
    cols: u32,
    heights: &[f32],
    cell_size: [f32; 2],
) -> Vec<[f32; 3]> {
    assert!(
        rows >= 2 && cols >= 2,
        "a height field requires at least two rows and two columns"
    );
    assert!(
        heights.len() as u32 == rows * cols,
        "height field sample count must match rows * cols"
    );
    assert!(
        cell_size.iter().all(|size| *size > 0.0),
        "height field cell size must be strictly positive"
    );
    let mut vertices = Vec::with_capacity((rows * cols) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let index = (row * cols + col) as usize;
            vertices.push([
                col as f32 * cell_size[0],
                heights[index],
                row as f32 * cell_size[1],
            ]);
        }
    }
    vertices
}
