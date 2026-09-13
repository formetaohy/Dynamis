#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullSolid {
    pub volume: f32,
    pub centroid: [f32; 3],
    pub unit_inertia: [f32; 6],
}

fn interior(vertices: &[[f32; 3]]) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    for vertex in vertices {
        for axis in 0..3 {
            sum[axis] += vertex[axis] as f64;
        }
    }
    let count = vertices.len() as f64;
    [sum[0] / count, sum[1] / count, sum[2] / count]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn shifted(point: [f32; 3], apex: [f64; 3]) -> [f64; 3] {
    [
        point[0] as f64 - apex[0],
        point[1] as f64 - apex[1],
        point[2] as f64 - apex[2],
    ]
}

fn outward(apex: [f64; 3], points: [[f32; 3]; 3]) -> [[f64; 3]; 3] {
    let [a, b, c] = points.map(|point| shifted(point, apex));
    if dot(cross(a, b), c) < 0.0 {
        [a, c, b]
    } else {
        [a, b, c]
    }
}

fn squared(ax: f64, bx: f64, cx: f64) -> f64 {
    (ax * ax + bx * bx + cx * cx + ax * bx + ax * cx + bx * cx) / 10.0
}

fn mixed(ax: f64, bx: f64, cx: f64, ay: f64, by: f64, cy: f64) -> f64 {
    (2.0 * (ax * ay + bx * by + cx * cy)
        + ax * by
        + ay * bx
        + ax * cy
        + ay * cx
        + bx * cy
        + by * cx)
        / 20.0
}

fn per_volume(value: f64, volume: f64) -> f32 {
    (value / volume) as f32
}

pub fn hull_solid(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> HullSolid {
    assert!(!vertices.is_empty(), "a hull must contain vertices");
    assert!(!triangles.is_empty(), "a hull must contain triangles");
    let apex = interior(vertices);
    let mut volume = 0.0f64;
    let mut first = [0.0f64; 3];
    let mut second = [0.0f64; 6];
    for triangle in triangles {
        let points = triangle.map(|corner| vertices[corner as usize]);
        let [a, b, c] = outward(apex, points);
        let signed = dot(a, cross(b, c)) / 6.0;
        volume += signed;
        for axis in 0..3 {
            first[axis] += signed * (a[axis] + b[axis] + c[axis]) / 4.0;
        }
        second[0] += signed * squared(a[0], b[0], c[0]);
        second[3] += signed * squared(a[1], b[1], c[1]);
        second[5] += signed * squared(a[2], b[2], c[2]);
        second[1] += signed * mixed(a[0], b[0], c[0], a[1], b[1], c[1]);
        second[2] += signed * mixed(a[0], b[0], c[0], a[2], b[2], c[2]);
        second[4] += signed * mixed(a[1], b[1], c[1], a[2], b[2], c[2]);
    }
    assert!(
        volume > 0.0 && volume.is_finite(),
        "a hull must enclose a positive volume"
    );
    let centroid = [first[0] / volume, first[1] / volume, first[2] / volume];
    let trace = second[0] + second[3] + second[5];
    let shift = [
        volume * (centroid[1] * centroid[1] + centroid[2] * centroid[2]),
        volume * (centroid[0] * centroid[0] + centroid[2] * centroid[2]),
        volume * (centroid[0] * centroid[0] + centroid[1] * centroid[1]),
    ];
    HullSolid {
        volume: volume as f32,
        centroid: [
            (apex[0] + centroid[0]) as f32,
            (apex[1] + centroid[1]) as f32,
            (apex[2] + centroid[2]) as f32,
        ],
        unit_inertia: [
            per_volume(trace - second[0] - shift[0], volume),
            per_volume(-second[1] + volume * centroid[0] * centroid[1], volume),
            per_volume(-second[2] + volume * centroid[0] * centroid[2], volume),
            per_volume(trace - second[3] - shift[1], volume),
            per_volume(-second[4] + volume * centroid[1] * centroid[2], volume),
            per_volume(trace - second[5] - shift[2], volume),
        ],
    }
}
