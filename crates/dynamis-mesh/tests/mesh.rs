use dynamis_mesh::{HullDecomposeSettings, convex_hull_mesh, decompose_mesh};

fn box_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
    ];
    (vertices, triangles)
}

fn l_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ];
    let triangles = vec![
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [6, 8, 7],
        [6, 9, 8],
        [6, 10, 9],
        [6, 11, 10],
        [0, 6, 7],
        [0, 7, 1],
        [1, 7, 8],
        [1, 8, 2],
        [2, 8, 9],
        [2, 9, 3],
        [3, 9, 10],
        [3, 10, 4],
        [4, 10, 11],
        [4, 11, 5],
        [5, 11, 6],
        [5, 6, 0],
    ];
    (vertices, triangles)
}

#[test]
fn convex_hull_reduces_box_mesh_to_eight_vertices() {
    let (vertices, triangles) = box_mesh();
    let (hull_vertices, hull_triangles) = convex_hull_mesh(&vertices, &triangles);
    assert_eq!(hull_vertices.len(), 8);
    assert_eq!(hull_triangles.len(), 12);
    for triangle in &hull_triangles {
        assert!(
            triangle
                .iter()
                .all(|&index| (index as usize) < hull_vertices.len()),
            "hull triangles must reference the hull vertices"
        );
    }
}

#[test]
fn convex_hull_handles_repeated_vertices() {
    let mut vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let triangles = vec![[0u32, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]];
    vertices.push([0.0, 0.0, 0.0]);
    let (hull_vertices, _) = convex_hull_mesh(&vertices, &triangles);
    assert_eq!(hull_vertices.len(), 4);
}

#[test]
fn decompose_keeps_convex_mesh_single() {
    let (vertices, triangles) = box_mesh();
    let parts = decompose_mesh(&vertices, &triangles, &HullDecomposeSettings::default());
    assert_eq!(parts.len(), 1, "a convex mesh must remain a single part");
}

#[test]
fn decompose_splits_concave_mesh() {
    let (vertices, triangles) = l_mesh();
    let parts = decompose_mesh(
        &vertices,
        &triangles,
        &HullDecomposeSettings {
            max_parts: 4,
            ..HullDecomposeSettings::default()
        },
    );
    assert!(parts.len() >= 2, "a concave mesh must split into parts");
    assert!(
        parts.len() <= 4,
        "decomposition must respect the part limit"
    );
    for (index, part) in parts.iter().enumerate() {
        assert!(
            part.triangles.len() >= 4,
            "part {index} must be a closed hull, got {} triangles",
            part.triangles.len()
        );
        assert!(
            part.vertices
                .iter()
                .all(|vertex| vertex.iter().all(|coordinate| coordinate.is_finite())),
            "part {index} must have finite vertices"
        );
    }
}
