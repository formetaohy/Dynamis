use crate::hull::convex_hull_mesh;
use std::collections::HashSet;

pub struct HullDecomposeSettings {
    pub max_parts: u32,
    pub concavity: f32,
    pub depth: u32,
}

impl Default for HullDecomposeSettings {
    fn default() -> Self {
        Self {
            max_parts: 16,
            concavity: 0.05,
            depth: 6,
        }
    }
}

pub struct HullMesh {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

pub fn decompose_mesh(
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
    settings: &HullDecomposeSettings,
) -> Vec<HullMesh> {
    validate_mesh(vertices, triangles);
    assert!(
        !planar(triangles, vertices, 0..triangles.len()),
        "decomposition requires a volumetric mesh"
    );
    let mut concavity = settings.concavity;
    for _ in 0..4 {
        let parts = split(
            vertices,
            triangles,
            (0..triangles.len() as u32).collect(),
            settings.max_parts as usize,
            settings.depth,
            concavity,
        );
        if parts.len() <= settings.max_parts as usize {
            return parts
                .into_iter()
                .map(|part| {
                    let (vertices, triangles) = convex_hull_mesh(vertices, &part);
                    HullMesh {
                        vertices,
                        triangles,
                    }
                })
                .collect();
        }
        concavity *= 1.5;
    }
    panic!(
        "decomposition exceeds the {} part limit; raise the concavity threshold",
        settings.max_parts
    );
}

fn validate_mesh(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) {
    assert!(!triangles.is_empty(), "decomposition requires triangles");
    for (index, triangle) in triangles.iter().enumerate() {
        assert!(
            triangle
                .iter()
                .all(|&vertex| (vertex as usize) < vertices.len()),
            "decomposition triangle {index} references an out-of-range vertex"
        );
    }
}

fn split(
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
    indices: Vec<u32>,
    parts_left: usize,
    depth: u32,
    concavity: f32,
) -> Vec<Vec<[u32; 3]>> {
    if parts_left <= 1 || indices.len() <= 4 || depth == 0 {
        return vec![part_triangles(triangles, &indices)];
    }
    if planar_indices(triangles, vertices, &indices) {
        return vec![part_triangles(triangles, &indices)];
    }
    let part = part_triangles(triangles, &indices);
    let (hull_vertices, hull_triangles) = convex_hull_mesh(vertices, &part);
    let radius = hull_radius(&hull_vertices);
    let depth_value = vertex_depth(
        vertices,
        triangles,
        &indices,
        &hull_vertices,
        &hull_triangles,
    );
    if depth_value <= concavity * radius.max(1e-6) {
        return vec![part];
    }
    let axis = split_axis(vertices, triangles, &indices);
    let mut projected = indices
        .iter()
        .map(|&index| {
            (
                triangle_centroid(vertices, triangles[index as usize]),
                index,
            )
        })
        .collect::<Vec<_>>();
    projected.sort_by(|a, b| dot(a.0, axis).total_cmp(&dot(b.0, axis)));
    let total = projected.len();
    let mut best: Option<(Vec<u32>, Vec<u32>, f32)> = None;
    for &fraction in &[0.25, 0.5, 0.75] {
        let cut = (total as f32 * fraction).round() as usize;
        if cut == 0 || cut >= total {
            continue;
        }
        let left = projected[..cut]
            .iter()
            .map(|(_, index)| *index)
            .collect::<Vec<_>>();
        let right = projected[cut..]
            .iter()
            .map(|(_, index)| *index)
            .collect::<Vec<_>>();
        if planar_indices(triangles, vertices, &left) || planar_indices(triangles, vertices, &right)
        {
            continue;
        }
        let score = child_score(vertices, triangles, &left, &right);
        match best {
            Some((_, _, best_score)) if score >= best_score => {}
            _ => best = Some((left, right, score)),
        }
    }
    let Some((left, right, _)) = best else {
        return vec![part];
    };
    let left_parts = parts_left.div_ceil(2);
    let right_parts = parts_left / 2;
    split(vertices, triangles, left, left_parts, depth - 1, concavity)
        .into_iter()
        .chain(split(
            vertices,
            triangles,
            right,
            right_parts,
            depth - 1,
            concavity,
        ))
        .collect()
}

fn child_score(vertices: &[[f32; 3]], triangles: &[[u32; 3]], left: &[u32], right: &[u32]) -> f32 {
    let score_of = |indices: &[u32]| {
        let part = part_triangles(triangles, indices);
        let (hull_vertices, hull_triangles) = convex_hull_mesh(vertices, &part);
        let radius = hull_radius(&hull_vertices);
        vertex_depth(
            vertices,
            triangles,
            indices,
            &hull_vertices,
            &hull_triangles,
        ) / radius.max(1e-6)
    };
    score_of(left).max(score_of(right))
}

fn part_triangles(triangles: &[[u32; 3]], indices: &[u32]) -> Vec<[u32; 3]> {
    indices
        .iter()
        .map(|&index| triangles[index as usize])
        .collect()
}

fn vertex_depth(
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
    indices: &[u32],
    hull_vertices: &[[f32; 3]],
    hull_triangles: &[[u32; 3]],
) -> f32 {
    let planes = hull_planes(hull_vertices, hull_triangles);
    let mut visited = HashSet::new();
    let mut max_depth = 0.0f32;
    for &index in indices {
        for &vertex in &triangles[index as usize] {
            if !visited.insert(vertex) {
                continue;
            }
            let point = vertices[vertex as usize];
            let depth = planes
                .iter()
                .map(|(normal, offset)| offset - dot(*normal, point))
                .filter(|signed| *signed > 0.0)
                .fold(f32::MAX, f32::min);
            if depth.is_finite() {
                max_depth = max_depth.max(depth);
            }
        }
    }
    max_depth
}

fn hull_planes(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<([f32; 3], f32)> {
    let mut planes = Vec::new();
    for triangle in triangles {
        let a = vertices[triangle[0] as usize];
        let b = vertices[triangle[1] as usize];
        let c = vertices[triangle[2] as usize];
        let normal = normalized(cross(sub(b, a), sub(c, a)));
        let offset = dot(normal, a);
        if !planes.iter().any(|(n, _)| dot(*n, normal) > 1.0 - 1e-4) {
            planes.push((normal, offset));
        }
    }
    planes
}

fn hull_radius(vertices: &[[f32; 3]]) -> f32 {
    let centroid = vertices
        .iter()
        .fold([0.0; 3], |acc, vertex| add(acc, *vertex));
    let centroid = mul(centroid, 1.0 / vertices.len().max(1) as f32);
    vertices
        .iter()
        .map(|vertex| length(sub(*vertex, centroid)))
        .fold(0.0, f32::max)
}

fn split_axis(vertices: &[[f32; 3]], triangles: &[[u32; 3]], indices: &[u32]) -> [f32; 3] {
    let mut covariance = [[0.0f32; 3]; 3];
    let mut mean = [0.0f32; 3];
    let count = indices.len() as f32;
    for &index in indices {
        mean = add(mean, triangle_centroid(vertices, triangles[index as usize]));
    }
    mean = mul(mean, 1.0 / count);
    for &index in indices {
        let offset = sub(triangle_centroid(vertices, triangles[index as usize]), mean);
        for i in 0..3 {
            for j in 0..3 {
                covariance[i][j] += offset[i] * offset[j];
            }
        }
    }
    for row in &mut covariance {
        for cell in row {
            *cell /= count;
        }
    }
    dominant_eigen(covariance).1
}

fn dominant_eigen(mut a: [[f32; 3]; 3]) -> (f32, [f32; 3]) {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..16 {
        let (p, q) = max_off_diagonal(&a);
        if a[p][q].abs() < 1e-12 {
            break;
        }
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let c = theta.cos();
        let s = theta.sin();
        let rotate_rows = |matrix: &mut [[f32; 3]; 3]| {
            for row in matrix.iter_mut() {
                let ap = row[p];
                let aq = row[q];
                row[p] = c * ap - s * aq;
                row[q] = s * ap + c * aq;
            }
        };
        let rotate_cols = |matrix: &mut [[f32; 3]; 3]| {
            for col in matrix.iter_mut() {
                let apk = col[p];
                let aqk = col[q];
                col[p] = c * apk - s * aqk;
                col[q] = s * apk + c * aqk;
            }
        };
        rotate_rows(&mut a);
        rotate_cols(&mut a);
        rotate_rows(&mut vectors);
    }
    let mut dominant = 0;
    for (i, row) in a.iter().enumerate().skip(1) {
        if row[i] > a[dominant][dominant] {
            dominant = i;
        }
    }
    (a[dominant][dominant], vectors[dominant])
}

fn max_off_diagonal(a: &[[f32; 3]; 3]) -> (usize, usize) {
    let mut best = (0, 1);
    let mut best_value = f32::MIN;
    for (i, row) in a.iter().enumerate() {
        for (j, cell) in row.iter().enumerate().skip(i + 1) {
            if cell.abs() > best_value {
                best_value = cell.abs();
                best = (i, j);
            }
        }
    }
    best
}

fn planar(triangles: &[[u32; 3]], vertices: &[[f32; 3]], range: std::ops::Range<usize>) -> bool {
    let indices = range.map(|index| index as u32).collect::<Vec<_>>();
    planar_indices(triangles, vertices, &indices)
}

fn planar_indices(triangles: &[[u32; 3]], vertices: &[[f32; 3]], indices: &[u32]) -> bool {
    let mut first = None;
    for &index in indices {
        let normal = face_normal(vertices, triangles[index as usize]);
        match first {
            None => first = Some(normal),
            Some(reference) => {
                if dot(reference, normal).abs() < 1.0 - 1e-4 {
                    return false;
                }
            }
        }
    }
    true
}

fn face_normal(vertices: &[[f32; 3]], triangle: [u32; 3]) -> [f32; 3] {
    let a = vertices[triangle[0] as usize];
    let b = vertices[triangle[1] as usize];
    let c = vertices[triangle[2] as usize];
    normalized(cross(sub(b, a), sub(c, a)))
}

fn triangle_centroid(vertices: &[[f32; 3]], triangle: [u32; 3]) -> [f32; 3] {
    let a = vertices[triangle[0] as usize];
    let b = vertices[triangle[1] as usize];
    let c = vertices[triangle[2] as usize];
    mul(add(add(a, b), c), 1.0 / 3.0)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul(a: [f32; 3], scalar: f32) -> [f32; 3] {
    [a[0] * scalar, a[1] * scalar, a[2] * scalar]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn normalized(a: [f32; 3]) -> [f32; 3] {
    let len = length(a);
    if len < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        mul(a, 1.0 / len)
    }
}
