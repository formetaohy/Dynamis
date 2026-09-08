use dynamis_layout::{
    BvhNodeRecord, SHAPE_SOURCE_HEIGHTFIELD, SHAPE_SOURCE_HULL, SHAPE_SOURCE_MESH,
};
use dynamis_model::ShapeSourceHandle;

#[derive(Clone, Copy)]
pub(crate) enum PoolKind {
    Hull,
    Mesh,
    HeightField,
}

pub(crate) struct ShapePool {
    capacity: usize,
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    refs: Vec<u32>,
    records: Vec<ShapeSourceRecordStorage>,
    pub(crate) vertices: Vec<[f32; 4]>,
    triangles: Vec<[u32; 4]>,
    nodes: Vec<BvhNodeRecord>,
    uploaded_vertices: usize,
    uploaded_triangles: usize,
    uploaded_nodes: usize,
}
#[derive(Clone, Copy)]
pub(crate) struct ShapeSourceRecordStorage {
    pub kind: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub triangle_offset: u32,
    pub triangle_count: u32,
    pub node_offset: u32,
    pub node_count: u32,
    pub bounds: ([f32; 3], [f32; 3]),
}

impl ShapePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            generations: vec![1; capacity],
            free_ids: (0..capacity as u32).rev().collect(),
            refs: vec![0; capacity],
            records: Vec::new(),
            vertices: Vec::new(),
            triangles: Vec::new(),
            nodes: Vec::new(),
            uploaded_vertices: 0,
            uploaded_triangles: 0,
            uploaded_nodes: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn record(&self, handle: ShapeSourceHandle) -> &ShapeSourceRecordStorage {
        assert!(
            (handle.id as usize) < self.records.len(),
            "shape source handle is out of range"
        );
        assert!(
            self.generations[handle.id as usize] == handle.generation,
            "shape source handle is stale"
        );
        assert!(
            self.records[handle.id as usize].kind != 0,
            "shape source handle is not alive"
        );
        &self.records[handle.id as usize]
    }

    pub(crate) fn record_by_index(&self, id: usize) -> &ShapeSourceRecordStorage {
        &self.records[id]
    }

    pub fn retain(&mut self, handle: ShapeSourceHandle) {
        let _ = self.record(handle);
        self.refs[handle.id as usize] += 1;
    }

    pub fn release(&mut self, handle: ShapeSourceHandle) {
        let _ = self.record(handle);
        assert!(
            self.refs[handle.id as usize] > 0,
            "shape source refcount underflow"
        );
        self.refs[handle.id as usize] -= 1;
    }

    pub fn remove(&mut self, handle: ShapeSourceHandle) {
        let _ = self.record(handle);
        assert!(
            self.refs[handle.id as usize] == 0,
            "shape source is still referenced by a live body"
        );
        let id = handle.id as usize;
        self.generations[id] += 1;
        self.records[id].kind = 0;
        self.free_ids.push(handle.id);
    }

    pub fn update_mesh(
        &mut self,
        handle: ShapeSourceHandle,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
    ) {
        let record_index = handle.id as usize;
        {
            let record = self.record(handle);
            assert!(
                vertices.len() as u32 == record.vertex_count
                    && triangles.len() as u32 == record.triangle_count,
                "shape update must keep the exact vertex and triangle counts"
            );
        }
        let node_offset = self.records[record_index].node_offset as usize;
        let node_capacity = self.records[record_index].node_count as usize;
        let nodes = build_bvh(vertices, triangles);
        assert!(
            nodes.len() <= node_capacity,
            "shape update BVH exceeds the original node capacity"
        );
        for (index, vertex) in vertices.iter().enumerate() {
            self.vertices[self.records[record_index].vertex_offset as usize + index] =
                [vertex[0], vertex[1], vertex[2], 0.0];
        }
        for (index, triangle) in triangles.iter().enumerate() {
            self.triangles[self.records[record_index].triangle_offset as usize + index] =
                [triangle[0], triangle[1], triangle[2], 0];
        }
        for (index, node) in nodes.iter().enumerate() {
            self.nodes[node_offset + index] = *node;
        }
        let (bounds_min, bounds_max) = bounds_of(vertices);
        let record = &mut self.records[record_index];
        record.node_count = nodes.len() as u32;
        record.bounds = (bounds_min, bounds_max);
    }

    pub fn allocate(
        &mut self,
        kind: PoolKind,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
    ) -> ShapeSourceHandle {
        let id = self
            .free_ids
            .pop()
            .expect("shape source capacity exhausted");
        self.generations[id as usize] += 1;
        let (bounds_min, bounds_max) = bounds_of(vertices);
        let vertex_offset = self.vertices.len() as u32;
        let triangle_offset = self.triangles.len() as u32;
        self.vertices
            .extend(vertices.iter().map(|v| [v[0], v[1], v[2], 0.0]));
        self.triangles
            .extend(triangles.iter().map(|t| [t[0], t[1], t[2], 0]));
        let node_offset = self.nodes.len() as u32;
        let nodes = build_bvh(vertices, triangles);
        let node_count = nodes.len() as u32;
        self.nodes.extend(nodes);
        let kind_code = match kind {
            PoolKind::Hull => SHAPE_SOURCE_HULL,
            PoolKind::Mesh => SHAPE_SOURCE_MESH,
            PoolKind::HeightField => SHAPE_SOURCE_HEIGHTFIELD,
        };
        while self.records.len() as u32 <= id {
            self.records.push(ShapeSourceRecordStorage {
                kind: 0,
                vertex_offset: 0,
                vertex_count: 0,
                triangle_offset: 0,
                triangle_count: 0,
                node_offset: 0,
                node_count: 0,
                bounds: ([0.0; 3], [0.0; 3]),
            });
        }
        self.records[id as usize] = ShapeSourceRecordStorage {
            kind: kind_code,
            vertex_offset,
            vertex_count: vertices.len() as u32,
            triangle_offset,
            triangle_count: triangles.len() as u32,
            node_offset,
            node_count,
            bounds: (bounds_min, bounds_max),
        };
        ShapeSourceHandle {
            id,
            generation: self.generations[id as usize],
        }
    }

    pub fn layouts(&self) -> Vec<dynamis_layout::ShapeSourceRecord> {
        self.records
            .iter()
            .map(|record| dynamis_layout::ShapeSourceRecord {
                kind: record.kind,
                vertex_offset: record.vertex_offset,
                vertex_count: record.vertex_count,
                triangle_offset: record.triangle_offset,
                triangle_count: record.triangle_count,
                node_offset: record.node_offset,
                node_count: record.node_count,
                _pad: 0,
            })
            .collect()
    }

    pub fn upload_pending(
        &mut self,
        queue: &wgpu::Queue,
        records_buffer: &dynamis_gpu::GpuBuffer,
        vertices_buffer: &dynamis_gpu::GpuBuffer,
        triangles_buffer: &dynamis_gpu::GpuBuffer,
        nodes_buffer: &dynamis_gpu::GpuBuffer,
    ) {
        records_buffer.write(queue, bytemuck::cast_slice(&self.layouts()));
        let pending_vertices = &self.vertices[self.uploaded_vertices..];
        if !pending_vertices.is_empty() {
            vertices_buffer.write_at(
                queue,
                (self.uploaded_vertices * 16) as u64,
                bytemuck::cast_slice(pending_vertices),
            );
            self.uploaded_vertices = self.vertices.len();
        }
        let pending_triangles = &self.triangles[self.uploaded_triangles..];
        if !pending_triangles.is_empty() {
            triangles_buffer.write_at(
                queue,
                (self.uploaded_triangles * 16) as u64,
                bytemuck::cast_slice(pending_triangles),
            );
            self.uploaded_triangles = self.triangles.len();
        }
        let pending_nodes = &self.nodes[self.uploaded_nodes..];
        if !pending_nodes.is_empty() {
            nodes_buffer.write_at(
                queue,
                (self.uploaded_nodes * std::mem::size_of::<BvhNodeRecord>()) as u64,
                bytemuck::cast_slice(pending_nodes),
            );
            self.uploaded_nodes = self.nodes.len();
        }
    }

    pub fn reset_upload_cursor(&mut self) {
        self.uploaded_vertices = 0;
        self.uploaded_triangles = 0;
        self.uploaded_nodes = 0;
    }

    pub fn upload_update(
        &mut self,
        queue: &wgpu::Queue,
        records_buffer: &dynamis_gpu::GpuBuffer,
        vertices_buffer: &dynamis_gpu::GpuBuffer,
        triangles_buffer: &dynamis_gpu::GpuBuffer,
        nodes_buffer: &dynamis_gpu::GpuBuffer,
        handle: dynamis_model::ShapeSourceHandle,
    ) {
        self.upload_pending(
            queue,
            records_buffer,
            vertices_buffer,
            triangles_buffer,
            nodes_buffer,
        );
        let record = self.records[handle.id as usize];
        if record.vertex_count > 0 {
            let start = record.vertex_offset as usize;
            let end = start + record.vertex_count as usize;
            vertices_buffer.write_at(
                queue,
                (record.vertex_offset as u64) * 16,
                bytemuck::cast_slice(&self.vertices[start..end]),
            );
        }
        if record.triangle_count > 0 {
            let start = record.triangle_offset as usize;
            let end = start + record.triangle_count as usize;
            triangles_buffer.write_at(
                queue,
                (record.triangle_offset as u64) * 16,
                bytemuck::cast_slice(&self.triangles[start..end]),
            );
        }
        if record.node_count > 0 {
            let start = record.node_offset as usize;
            let end = start + record.node_count as usize;
            nodes_buffer.write_at(
                queue,
                (record.node_offset as u64) * std::mem::size_of::<BvhNodeRecord>() as u64,
                bytemuck::cast_slice(&self.nodes[start..end]),
            );
        }
    }
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
