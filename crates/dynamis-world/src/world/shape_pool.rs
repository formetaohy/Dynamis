use super::arena::{Arena, Run};
use super::ids::IdSpace;
use crate::dynamics::ShapeCapacity;
use crate::dynamics::scene::{TRIANGLE_BYTES, VERTEX_BYTES};
use bytemuck::Zeroable;
use dynamis_layout::{BvhNodeRecord, TriangleRecord};
use dynamis_model::ShapeSourceHandle;
use std::mem::size_of;

struct Geometry<T> {
    arena: Arena,
    rows: Vec<T>,
    dirty: Vec<Run>,
}

impl<T: Copy> Geometry<T> {
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
        let run = self.arena.take(values.len() as u32);
        self.rows.resize(self.arena.used() as usize, blank);
        self.rows[run.span()].copy_from_slice(values);
        self.dirty.push(run);
        run
    }

    fn rewrite(&mut self, run: Run, blank: T, values: &[T]) -> Run {
        if run.len as usize == values.len() {
            self.rows[run.span()].copy_from_slice(values);
            self.dirty.push(run);
            return run;
        }
        self.release(run, blank);
        self.take(blank, values)
    }

    fn release(&mut self, run: Run, blank: T) {
        self.rows[run.span()].fill(blank);
        self.arena.release(run);
        self.rows.truncate(self.arena.used() as usize);
    }

    fn take_dirty(&mut self) -> Vec<Run> {
        let live = self.arena.used();
        let mut runs = std::mem::take(&mut self.dirty);
        runs.retain_mut(|run| {
            if run.offset >= live {
                return false;
            }
            run.len = run.len.min(live - run.offset);
            true
        });
        runs.sort_unstable_by_key(|run| run.offset);
        let mut merged: Vec<Run> = Vec::with_capacity(runs.len());
        for run in runs {
            match merged.last_mut() {
                Some(previous) if previous.offset + previous.len >= run.offset => {
                    previous.len = previous.offset.max(run.offset + run.len) - previous.offset;
                }
                _ => merged.push(run),
            }
        }
        merged
    }
}

#[derive(Clone, Copy)]
struct Source {
    kind: u32,
    vertices: Run,
    triangles: Run,
    nodes: Run,
    bounds: ([f32; 3], [f32; 3]),
}

impl Source {
    const DEAD: Self = Self {
        kind: 0,
        vertices: Run::EMPTY,
        triangles: Run::EMPTY,
        nodes: Run::EMPTY,
        bounds: ([0.0; 3], [0.0; 3]),
    };

    fn alive(&self) -> bool {
        self.kind != 0
    }
}

pub(crate) struct ShapePool {
    ids: IdSpace,
    refs: Vec<u32>,
    sources: Vec<Source>,
    vertices: Geometry<[f32; 4]>,
    triangles: Geometry<TriangleRecord>,
    nodes: Geometry<BvhNodeRecord>,
}

impl ShapePool {
    pub(crate) const fn new() -> Self {
        Self {
            ids: IdSpace::new(),
            refs: Vec::new(),
            sources: Vec::new(),
            vertices: Geometry::new(),
            triangles: Geometry::new(),
            nodes: Geometry::new(),
        }
    }

    pub(crate) fn used(&self) -> ShapeCapacity {
        ShapeCapacity {
            sources: self.sources.len() as u32,
            vertices: self.vertices.used(),
            triangles: self.triangles.used(),
            nodes: self.nodes.used(),
        }
    }

    pub(crate) fn bounds(&self, handle: ShapeSourceHandle) -> ([f32; 3], [f32; 3]) {
        self.source(handle).bounds
    }

    fn source(&self, handle: ShapeSourceHandle) -> &Source {
        let source = self
            .sources
            .get(handle.id as usize)
            .unwrap_or_else(|| panic!("shape source handle {handle:?} is out of range"));
        assert!(
            self.ids.generation(handle.id) == handle.generation,
            "shape source handle {handle:?} is stale"
        );
        assert!(
            source.alive(),
            "shape source handle {handle:?} is not alive"
        );
        source
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
            .release(source.triangles, TriangleRecord::zeroed());
        self.nodes.release(source.nodes, BvhNodeRecord::zeroed());
        self.sources[id] = Source::DEAD;
        self.ids.release(handle.id);
    }

    pub(crate) fn allocate(
        &mut self,
        kind: u32,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
    ) -> ShapeSourceHandle {
        validate_geometry(kind, vertices, triangles);
        let nodes = build_bvh(vertices, triangles);
        let bounds = bounds_of(vertices);
        let (id, generation) = self.ids.acquire();
        self.grow_to(id);
        let vertex_rows = vertex_rows(vertices);
        let triangle_rows = triangle_rows(triangles);
        let vertex_run = self.vertices.take([0.0; 4], &vertex_rows);
        let triangle_run = self
            .triangles
            .take(TriangleRecord::zeroed(), &triangle_rows);
        let node_run = self.nodes.take(BvhNodeRecord::zeroed(), &nodes);
        self.sources[id as usize] = Source {
            kind,
            vertices: vertex_run,
            triangles: triangle_run,
            nodes: node_run,
            bounds,
        };
        ShapeSourceHandle { id, generation }
    }

    pub(crate) fn update_mesh(
        &mut self,
        handle: ShapeSourceHandle,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
    ) {
        let source = *self.source(handle);
        validate_geometry(source.kind, vertices, triangles);
        let nodes = build_bvh(vertices, triangles);
        let bounds = bounds_of(vertices);
        let vertex_rows = vertex_rows(vertices);
        let triangle_rows = triangle_rows(triangles);
        let vertex_run = self
            .vertices
            .rewrite(source.vertices, [0.0; 4], &vertex_rows);
        let triangle_run =
            self.triangles
                .rewrite(source.triangles, TriangleRecord::zeroed(), &triangle_rows);
        let node_run = self
            .nodes
            .rewrite(source.nodes, BvhNodeRecord::zeroed(), &nodes);
        self.sources[handle.id as usize] = Source {
            vertices: vertex_run,
            triangles: triangle_run,
            nodes: node_run,
            bounds,
            ..source
        };
    }

    pub(crate) fn upload_pending(
        &mut self,
        queue: &wgpu::Queue,
        sources_buffer: &dynamis_gpu::Stream,
        vertices_buffer: &dynamis_gpu::Stream,
        triangles_buffer: &dynamis_gpu::Stream,
        nodes_buffer: &dynamis_gpu::Stream,
    ) {
        sources_buffer.write(queue, bytemuck::cast_slice(&self.layouts()));
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
    }

    fn layouts(&self) -> Vec<dynamis_layout::ShapeSourceRecord> {
        self.sources
            .iter()
            .map(|source| dynamis_layout::ShapeSourceRecord {
                kind: source.kind,
                vertex_offset: source.vertices.offset,
                vertex_count: source.vertices.len,
                triangle_offset: source.triangles.offset,
                triangle_count: source.triangles.len,
                node_offset: source.nodes.offset,
                node_count: source.nodes.len,
                _pad0: 0,
                local_min: source.bounds.0,
                _pad1: 0.0,
                local_max: source.bounds.1,
                _pad2: 0.0,
            })
            .collect()
    }
}

fn validate_geometry(kind: u32, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) {
    assert!(
        kind != dynamis_layout::SHAPE_HULL
            || vertices.len() <= dynamis_layout::FEATURE_INDEX_LIMIT as usize,
        "a hull must not exceed {} vertices so its features stay addressable",
        dynamis_layout::FEATURE_INDEX_LIMIT
    );
    assert!(
        triangles.len() <= dynamis_layout::FEATURE_TRIANGLE_MASK as usize,
        "a shape source must not exceed {} triangles so its features stay addressable",
        dynamis_layout::FEATURE_TRIANGLE_MASK
    );
}

fn vertex_rows(vertices: &[[f32; 3]]) -> Vec<[f32; 4]> {
    vertices
        .iter()
        .map(|vertex| [vertex[0], vertex[1], vertex[2], 0.0])
        .collect()
}

fn triangle_rows(triangles: &[[u32; 3]]) -> Vec<TriangleRecord> {
    triangles
        .iter()
        .map(|triangle| TriangleRecord {
            a: triangle[0],
            b: triangle[1],
            c: triangle[2],
            _pad0: 0,
        })
        .collect()
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

pub fn height_field_triangles(
    rows: u32,
    cols: u32,
    heights: &[f32],
    cell_size: [f32; 2],
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    assert!(
        heights.len() as u32 == rows * cols,
        "height field sample count must match rows * cols"
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
    let mut triangles = Vec::with_capacity(((rows - 1) * (cols - 1) * 2) as usize);
    for row in 0..rows - 1 {
        for col in 0..cols - 1 {
            let a = row * cols + col;
            let b = (row + 1) * cols + col;
            let c = row * cols + col + 1;
            let d = (row + 1) * cols + col + 1;
            triangles.push([a, b, c]);
            triangles.push([b, d, c]);
        }
    }
    (vertices, triangles)
}
