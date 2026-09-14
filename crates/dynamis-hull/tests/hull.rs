use dynamis_hull::{DecomposeSettings, decompose, hull, solid};

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
    let (hull_vertices, hull_triangles) = hull(&vertices, &triangles);
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
    let (hull_vertices, _) = hull(&vertices, &triangles);
    assert_eq!(hull_vertices.len(), 4);
}

#[test]
fn decompose_keeps_convex_mesh_single() {
    let (vertices, triangles) = box_mesh();
    let parts = decompose(&vertices, &triangles, &DecomposeSettings::default());
    assert_eq!(parts.len(), 1, "a convex mesh must remain a single part");
}

#[test]
fn decompose_splits_concave_mesh() {
    let (vertices, triangles) = l_mesh();
    let parts = decompose(
        &vertices,
        &triangles,
        &DecomposeSettings {
            max_parts: 4,
            ..DecomposeSettings::default()
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

fn tetrahedron() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    (vertices, triangles)
}

fn octahedron(half: f32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [half, 0.0, 0.0],
        [-half, 0.0, 0.0],
        [0.0, half, 0.0],
        [0.0, -half, 0.0],
        [0.0, 0.0, half],
        [0.0, 0.0, -half],
    ];
    let triangles = vec![
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 2],
        [1, 3, 4],
        [1, 5, 3],
        [1, 2, 5],
    ];
    (vertices, triangles)
}

fn assert_close(actual: f32, expected: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= expected.abs().max(1.0) * 1e-5,
        "{what} must be {expected}, got {actual}"
    );
}

#[test]
fn a_unit_tetrahedron_reports_its_closed_form_solid() {
    let (vertices, triangles) = tetrahedron();
    let geometry = solid(&vertices, &triangles);
    assert_close(geometry.volume, 1.0 / 6.0, "tetrahedron volume");
    assert_close(geometry.centroid[0], 0.25, "tetrahedron centroid x");
    assert_close(geometry.centroid[1], 0.25, "tetrahedron centroid y");
    assert_close(geometry.centroid[2], 0.25, "tetrahedron centroid z");
    assert_close(
        geometry.unit_inertia[0],
        3.0 / 40.0,
        "tetrahedron inertia xx",
    );
    assert_close(
        geometry.unit_inertia[3],
        3.0 / 40.0,
        "tetrahedron inertia yy",
    );
    assert_close(
        geometry.unit_inertia[5],
        3.0 / 40.0,
        "tetrahedron inertia zz",
    );
    assert_close(
        geometry.unit_inertia[1],
        1.0 / 80.0,
        "tetrahedron inertia xy",
    );
    assert_close(
        geometry.unit_inertia[2],
        1.0 / 80.0,
        "tetrahedron inertia xz",
    );
    assert_close(
        geometry.unit_inertia[4],
        1.0 / 80.0,
        "tetrahedron inertia yz",
    );
}

#[test]
fn a_tetrahedron_reports_the_same_solid_for_either_winding() {
    let (vertices, triangles) = tetrahedron();
    let flipped = triangles
        .iter()
        .map(|tri| [tri[0], tri[2], tri[1]])
        .collect::<Vec<_>>();
    assert_eq!(solid(&vertices, &triangles), solid(&vertices, &flipped));
}

#[test]
fn an_octahedron_reports_an_isotropic_solid() {
    let (vertices, triangles) = octahedron(1.0);
    let geometry = solid(&vertices, &triangles);
    assert_close(geometry.volume, 4.0 / 3.0, "octahedron volume");
    assert_close(geometry.centroid[0], 0.0, "octahedron centroid x");
    assert_close(geometry.centroid[1], 0.0, "octahedron centroid y");
    assert_close(geometry.centroid[2], 0.0, "octahedron centroid z");
    for (index, axis) in [(0, "xx"), (3, "yy"), (5, "zz")]
        .iter()
        .map(|(i, n)| (*i, *n))
    {
        assert_close(
            geometry.unit_inertia[index],
            0.2,
            &format!("octahedron inertia {axis}"),
        );
    }
    assert_close(geometry.unit_inertia[1], 0.0, "octahedron inertia xy");
    assert_close(geometry.unit_inertia[2], 0.0, "octahedron inertia xz");
    assert_close(geometry.unit_inertia[4], 0.0, "octahedron inertia yz");
}

#[test]
fn a_translated_hull_keeps_its_solid_and_moves_its_centroid() {
    let (vertices, triangles) = octahedron(1.0);
    let moved = vertices
        .iter()
        .map(|vertex| [vertex[0] + 10.0, vertex[1] - 4.0, vertex[2] + 0.5])
        .collect::<Vec<_>>();
    let geometry = solid(&moved, &triangles);
    assert_close(geometry.volume, 4.0 / 3.0, "moved octahedron volume");
    assert_close(geometry.centroid[0], 10.0, "moved octahedron centroid x");
    assert_close(geometry.centroid[1], -4.0, "moved octahedron centroid y");
    assert_close(geometry.centroid[2], 0.5, "moved octahedron centroid z");
    assert_close(geometry.unit_inertia[0], 0.2, "moved octahedron inertia xx");
}

#[test]
#[should_panic(expected = "positive volume")]
fn a_flat_hull_is_refused() {
    let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let triangles = vec![[0, 1, 2], [0, 2, 1]];
    solid(&vertices, &triangles);
}
