@group(1) @binding(0) var<storage, read> shape_sources: array<ShapeSource>;
@group(1) @binding(1) var<storage, read> shape_vertices: array<vec4f>;
@group(1) @binding(2) var<storage, read> shape_triangles: array<Triangle>;
@group(1) @binding(3) var<storage, read> shape_nodes: array<BvhNode>;
