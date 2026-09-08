use std::collections::{HashMap, HashSet};

struct Face {
    a: usize,
    b: usize,
    c: usize,
    normal: [f32; 3],
    offset: f32,
    outside: Vec<usize>,
    dead: bool,
}

impl Face {
    fn new(a: usize, b: usize, c: usize, points: &[[f32; 3]]) -> Self {
        let normal = face_normal(points, a, b, c);
        let offset = dot(normal, points[a]);
        Self {
            a,
            b,
            c,
            normal,
            offset,
            outside: Vec::new(),
            dead: false,
        }
    }

    fn distance(&self, point: [f32; 3]) -> f32 {
        dot(self.normal, point) - self.offset
    }
}

pub fn convex_hull_mesh(
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let used = triangles
        .iter()
        .flat_map(|triangle| triangle.iter())
        .copied()
        .collect::<HashSet<_>>();
    assert!(!used.is_empty(), "hull mesh must contain triangles");
    let mut point_indices = used.into_iter().collect::<Vec<_>>();
    point_indices.sort_unstable();
    let points = point_indices
        .iter()
        .map(|index| vertices[*index as usize])
        .collect::<Vec<_>>();
    let (hull_indices, hull_triangles) = quickhull(&points);
    let mut unique_points = Vec::new();
    let mut remap = HashMap::new();
    for &index in &hull_indices {
        remap.entry(index).or_insert_with(|| {
            unique_points.push(points[index]);
            unique_points.len() - 1
        });
    }
    let hull_triangles = hull_triangles
        .into_iter()
        .map(|triangle| {
            [
                remap[&triangle[0]] as u32,
                remap[&triangle[1]] as u32,
                remap[&triangle[2]] as u32,
            ]
        })
        .collect();
    (unique_points, hull_triangles)
}

fn quickhull(points: &[[f32; 3]]) -> (Vec<usize>, Vec<[usize; 3]>) {
    let (p0, p1) = extreme_pair(points);
    let p2 = farthest_from_line(points, p0, p1);
    let p3 = farthest_from_plane(points, p0, p1, p2);
    let mut faces = initial_tetra(points, p0, p1, p2, p3);
    assign_outside(points, &mut faces, None);
    let mut indices = (0..faces.len()).collect::<Vec<_>>();
    while !indices.is_empty() {
        indices.retain(|index| !faces[*index].dead && !faces[*index].outside.is_empty());
        let Some(&seed) = indices.iter().max_by(|&&a, &&b| {
            faces[a]
                .outside
                .first()
                .map(|&i| faces[a].distance(points[i]))
                .partial_cmp(
                    &faces[b]
                        .outside
                        .first()
                        .map(|&i| faces[b].distance(points[i])),
                )
                .unwrap()
        }) else {
            break;
        };
        let apex = faces[seed]
            .outside
            .iter()
            .copied()
            .max_by(|&a, &b| {
                faces[seed]
                    .distance(points[a])
                    .partial_cmp(&faces[seed].distance(points[b]))
                    .unwrap()
            })
            .unwrap();
        let visible = faces
            .iter()
            .enumerate()
            .filter(|(index, face)| {
                !face.dead && *index != seed && face.distance(points[apex]) > 1e-6
            })
            .map(|(index, _)| index)
            .chain(std::iter::once(seed))
            .collect::<Vec<_>>();
        let mut edge_set = HashSet::new();
        for &index in &visible {
            let face = &faces[index];
            for (start, end) in [(face.a, face.b), (face.b, face.c), (face.c, face.a)] {
                edge_set.insert((start, end));
            }
        }
        let mut horizon = Vec::new();
        for &index in &visible {
            let face = &faces[index];
            for (start, end) in [(face.a, face.b), (face.b, face.c), (face.c, face.a)] {
                if !edge_set.contains(&(end, start)) {
                    horizon.push((start, end));
                }
            }
        }
        assert!(!horizon.is_empty(), "hull horizon must not be empty");
        // 分配点前收集旧 outside
        let mut pending = Vec::new();
        for &index in &visible {
            pending.extend(faces[index].outside.iter().copied());
            faces[index].dead = true;
        }
        pending.push(apex);
        for (start, end) in horizon {
            let face = Face::new(start, end, apex, points);
            faces.push(face);
        }
        assign_outside(points, &mut faces, Some(&pending));
        indices = (0..faces.len()).collect();
    }
    let output_faces = faces
        .iter()
        .filter(|face| !face.dead)
        .map(|face| [face.a, face.b, face.c])
        .collect::<Vec<_>>();
    let mut output_vertices = Vec::new();
    for face in &output_faces {
        for &vertex in face {
            if !output_vertices.contains(&vertex) {
                output_vertices.push(vertex);
            }
        }
    }
    (output_vertices, output_faces)
}

fn assign_outside(points: &[[f32; 3]], faces: &mut [Face], restrict: Option<&[usize]>) {
    let candidates = match restrict {
        Some(list) => list.to_vec(),
        None => (0..points.len()).collect(),
    };
    for &point_index in &candidates {
        for face in faces.iter_mut().filter(|face| !face.dead) {
            if face.distance(points[point_index]) > 1e-6 {
                face.outside.push(point_index);
                break;
            }
        }
    }
}

fn initial_tetra(points: &[[f32; 3]], i0: usize, i1: usize, i2: usize, i3: usize) -> Vec<Face> {
    let centroid = [
        (points[i0][0] + points[i1][0] + points[i2][0] + points[i3][0]) * 0.25,
        (points[i0][1] + points[i1][1] + points[i2][1] + points[i3][1]) * 0.25,
        (points[i0][2] + points[i1][2] + points[i2][2] + points[i3][2]) * 0.25,
    ];
    let mut faces = vec![
        Face::new(i0, i1, i2, points),
        Face::new(i0, i1, i3, points),
        Face::new(i0, i2, i3, points),
        Face::new(i1, i2, i3, points),
    ];
    for face in &mut faces {
        let normal_centroid = dot(face.normal, centroid) - face.offset;
        if normal_centroid > 0.0 {
            face.normal = [-face.normal[0], -face.normal[1], -face.normal[2]];
            face.offset = -face.offset;
            std::mem::swap(&mut face.b, &mut face.c);
        }
    }
    faces
}

fn extreme_pair(points: &[[f32; 3]]) -> (usize, usize) {
    let mut x_min = 0;
    let mut x_max = 0;
    for (index, point) in points.iter().enumerate() {
        if point[0] < points[x_min][0] {
            x_min = index;
        }
        if point[0] > points[x_max][0] {
            x_max = index;
        }
    }
    if x_min == x_max {
        (0, points.len() - 1)
    } else {
        (x_min, x_max)
    }
}

fn farthest_from_line(points: &[[f32; 3]], a: usize, b: usize) -> usize {
    let dir = sub(points[b], points[a]);
    let length_sq = dot(dir, dir);
    let mut best = 0;
    let mut best_dist = -1.0;
    for (index, point) in points.iter().enumerate() {
        let delta = sub(*point, points[a]);
        let projection = if length_sq > 0.0 {
            dot(delta, dir) / length_sq
        } else {
            0.0
        };
        let closest = add(points[a], mul(dir, projection));
        let dist = length(sub(*point, closest));
        if dist > best_dist {
            best_dist = dist;
            best = index;
        }
    }
    assert!(best_dist > 1e-6, "hull mesh points must not be collinear");
    best
}

fn farthest_from_plane(points: &[[f32; 3]], a: usize, b: usize, c: usize) -> usize {
    let normal = face_normal(points, a, b, c);
    let offset = dot(normal, points[a]);
    let mut best = 0;
    let mut best_dist = 0.0;
    for (index, point) in points.iter().enumerate() {
        let dist = (dot(normal, *point) - offset).abs();
        if dist > best_dist {
            best_dist = dist;
            best = index;
        }
    }
    assert!(best_dist > 1e-6, "hull mesh points must not be coplanar");
    best
}

fn face_normal(points: &[[f32; 3]], a: usize, b: usize, c: usize) -> [f32; 3] {
    let ab = sub(points[b], points[a]);
    let ac = sub(points[c], points[a]);
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = length(cross);
    if len < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [cross[0] / len, cross[1] / len, cross[2] / len]
    }
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

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}
