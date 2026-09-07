use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::PI;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub enum Geometry {
    Sphere { radius: f32 },
    Cuboid { half_extents: [f32; 3] },
    Capsule { radius: f32, half_height: f32 },
    Cylinder { radius: f32, half_height: f32 },
}

impl Geometry {
    pub fn build(&self) -> MeshData {
        match *self {
            Self::Sphere { radius } => sphere(radius, 16, 24),
            Self::Cuboid { half_extents } => cuboid(half_extents),
            Self::Capsule {
                radius,
                half_height,
            } => capsule(radius, half_height, 20, 8),
            Self::Cylinder {
                radius,
                half_height,
            } => cylinder(radius, half_height, 24),
        }
    }
}

fn sphere(radius: f32, stacks: usize, slices: usize) -> MeshData {
    let mut vertices = Vec::with_capacity((stacks + 1) * (slices + 1));
    for stack in 0..=stacks {
        let theta = PI * stack as f32 / stacks as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for slice in 0..=slices {
            let phi = 2.0 * PI * slice as f32 / slices as f32;
            let normal = [sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin()];
            vertices.push(Vertex {
                position: scale(normal, radius),
                normal,
            });
        }
    }
    let mut indices = Vec::with_capacity(stacks * slices * 6);
    let stride = slices + 1;
    for stack in 0..stacks {
        for slice in 0..slices {
            let a = (stack * stride + slice) as u32;
            let b = a + 1;
            let c = a + stride as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    MeshData { vertices, indices }
}

fn cuboid(half_extents: [f32; 3]) -> MeshData {
    const AXES: [(usize, [f32; 3]); 6] = [
        (0, [1.0, 0.0, 0.0]),
        (0, [-1.0, 0.0, 0.0]),
        (1, [0.0, 1.0, 0.0]),
        (1, [0.0, -1.0, 0.0]),
        (2, [0.0, 0.0, 1.0]),
        (2, [0.0, 0.0, -1.0]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (axis, normal) in AXES {
        let tangent = [(axis + 1) % 3, (axis + 2) % 3];
        for (sign_a, sign_b) in [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)] {
            let mut position = [0.0; 3];
            position[axis] = normal[axis] * half_extents[axis];
            position[tangent[0]] = sign_a * half_extents[tangent[0]];
            position[tangent[1]] = sign_b * half_extents[tangent[1]];
            vertices.push(Vertex { position, normal });
        }
        let base = (vertices.len() - 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshData { vertices, indices }
}

fn cylinder(radius: f32, half_height: f32, radial: usize) -> MeshData {
    let ring = |y: f32, normal_y: f32| -> Vec<Vertex> {
        let mut ring = Vec::with_capacity(radial + 1);
        for index in 0..=radial {
            let angle = 2.0 * PI * index as f32 / radial as f32;
            let normal = [angle.cos(), normal_y, angle.sin()];
            ring.push(Vertex {
                position: [normal[0] * radius, y, normal[2] * radius],
                normal,
            });
        }
        ring
    };
    let mut vertices = Vec::with_capacity(2 * (radial + 1) + 2 + 2 * radial);
    let bottom_ring = ring(-half_height, 0.0);
    let top_ring = ring(half_height, 0.0);
    vertices.extend_from_slice(&bottom_ring);
    vertices.extend_from_slice(&top_ring);
    let cap = |y: f32, normal: [f32; 3]| -> (u32, Vec<Vertex>) {
        let ring = (0..radial)
            .map(|index| {
                let angle = 2.0 * PI * index as f32 / radial as f32;
                Vertex {
                    position: [angle.cos() * radius, y, angle.sin() * radius],
                    normal,
                }
            })
            .collect::<Vec<_>>();
        let center = Vertex {
            position: [0.0, y, 0.0],
            normal,
        };
        (
            vertices.len() as u32,
            std::iter::once(center).chain(ring).collect(),
        )
    };
    let (bottom_center, bottom_ring_cap) = cap(-half_height, [0.0, -1.0, 0.0]);
    let (top_center, top_ring_cap) = cap(half_height, [0.0, 1.0, 0.0]);
    vertices.extend_from_slice(&bottom_ring_cap);
    vertices.extend_from_slice(&top_ring_cap);

    let mut indices = Vec::with_capacity(radial * 6 + radial * 6);
    let bottom = 0u32;
    let top = (radial + 1) as u32;
    for index in 0..radial {
        let a = bottom + index as u32;
        let b = a + 1;
        let c = top + index as u32;
        let d = c + 1;
        indices.extend_from_slice(&[a, b, c, b, d, c]);
        indices.extend_from_slice(&[bottom_center, a, a + 1, top_center, c + 1, c]);
    }
    MeshData { vertices, indices }
}

fn capsule(radius: f32, half_height: f32, radial: usize, dome_steps: usize) -> MeshData {
    let side_ring = |y: f32| -> Vec<Vertex> {
        let mut ring = Vec::with_capacity(radial + 1);
        for index in 0..=radial {
            let angle = 2.0 * PI * index as f32 / radial as f32;
            let normal = [angle.cos(), 0.0, angle.sin()];
            ring.push(Vertex {
                position: [normal[0] * radius, y, normal[2] * radius],
                normal,
            });
        }
        ring
    };
    let dome_ring = |y: f32, ring_radius: f32, sin_theta: f32| -> Vec<Vertex> {
        let mut ring = Vec::with_capacity(radial + 1);
        for index in 0..=radial {
            let angle = 2.0 * PI * index as f32 / radial as f32;
            let normal = [
                angle.cos() * ring_radius / radius,
                -sin_theta,
                angle.sin() * ring_radius / radius,
            ];
            ring.push(Vertex {
                position: [angle.cos() * ring_radius, y, angle.sin() * ring_radius],
                normal,
            });
        }
        ring
    };
    let mut vertices = Vec::new();
    let bottom_ring = side_ring(-half_height);
    let top_ring = side_ring(half_height);
    vertices.extend_from_slice(&bottom_ring);
    vertices.extend_from_slice(&top_ring);

    let mut bottom_dome: Vec<Vec<Vertex>> = Vec::new();
    for step in 1..dome_steps {
        let theta = (PI / 2.0) * step as f32 / dome_steps as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        bottom_dome.push(dome_ring(
            -half_height - radius * sin_theta,
            radius * cos_theta,
            sin_theta,
        ));
    }
    let bottom_pole = Vertex {
        position: [0.0, -half_height - radius, 0.0],
        normal: [0.0, -1.0, 0.0],
    };
    let mut top_dome: Vec<Vec<Vertex>> = Vec::new();
    for step in 1..dome_steps {
        let theta = (PI / 2.0) * step as f32 / dome_steps as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        top_dome.push(dome_ring(
            half_height + radius * sin_theta,
            radius * cos_theta,
            -sin_theta,
        ));
    }
    let top_pole = Vertex {
        position: [0.0, half_height + radius, 0.0],
        normal: [0.0, 1.0, 0.0],
    };
    for ring in &bottom_dome {
        vertices.extend_from_slice(ring);
    }
    vertices.push(bottom_pole);
    for ring in &top_dome {
        vertices.extend_from_slice(ring);
    }
    vertices.push(top_pole);

    let stride = radial + 1;
    let mut indices = Vec::new();
    for index in 0..radial {
        let a = index as u32;
        let b = a + 1;
        let c = (stride + index) as u32;
        let d = c + 1;
        indices.extend_from_slice(&[a, b, c, b, d, c]);
    }
    let bottom_dome_start = 2 * stride;
    let top_dome_start = bottom_dome_start + (dome_steps - 1) * stride + 1;
    for ring_index in 0..bottom_dome.len() {
        let lower_base = if ring_index == 0 {
            0
        } else {
            bottom_dome_start + (ring_index - 1) * stride
        };
        let ring_base = bottom_dome_start + ring_index * stride;
        for index in 0..radial {
            let a = (lower_base + index) as u32;
            let b = (lower_base + index + 1) as u32;
            let c = (ring_base + index) as u32;
            let d = (ring_base + index + 1) as u32;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    let bottom_pole_index = (bottom_dome_start + (dome_steps - 1) * stride) as u32;
    let bottom_last_base = bottom_dome_start + (bottom_dome.len() - 1) * stride;
    for index in 0..radial {
        indices.extend_from_slice(&[
            (bottom_last_base + index) as u32,
            (bottom_last_base + index + 1) as u32,
            bottom_pole_index,
        ]);
    }
    for ring_index in 0..top_dome.len() {
        let lower_base = if ring_index == 0 {
            stride
        } else {
            top_dome_start + (ring_index - 1) * stride
        };
        let ring_base = top_dome_start + ring_index * stride;
        for index in 0..radial {
            let a = (lower_base + index) as u32;
            let b = (lower_base + index + 1) as u32;
            let c = (ring_base + index) as u32;
            let d = (ring_base + index + 1) as u32;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    let top_pole_index = (top_dome_start + (dome_steps - 1) * stride) as u32;
    let top_last_base = top_dome_start + (top_dome.len() - 1) * stride;
    for index in 0..radial {
        indices.extend_from_slice(&[
            (top_last_base + index) as u32,
            (top_last_base + index + 1) as u32,
            top_pole_index,
        ]);
    }
    MeshData { vertices, indices }
}

pub fn sphere_wireframe(
    center: Vec3,
    radius: f32,
    rings: usize,
    segments: usize,
) -> Vec<(Vec3, Vec3)> {
    let mut lines = Vec::new();
    for ring in 1..rings {
        let phi = PI * ring as f32 / rings as f32;
        let (sin_phi, cos_phi) = phi.sin_cos();
        for segment in 0..segments {
            let a0 = 2.0 * PI * segment as f32 / segments as f32;
            let a1 = 2.0 * PI * (segment + 1) as f32 / segments as f32;
            lines.push((
                center + Vec3::new(sin_phi * a0.cos(), cos_phi, sin_phi * a0.sin()) * radius,
                center + Vec3::new(sin_phi * a1.cos(), cos_phi, sin_phi * a1.sin()) * radius,
            ));
        }
    }
    for segment in 0..segments {
        let angle = 2.0 * PI * segment as f32 / segments as f32;
        for ring in 0..rings {
            let phi0 = PI * ring as f32 / rings as f32;
            let phi1 = PI * (ring + 1) as f32 / rings as f32;
            let dir0 = Vec3::new(
                phi0.sin() * angle.cos(),
                phi0.cos(),
                phi0.sin() * angle.sin(),
            );
            let dir1 = Vec3::new(
                phi1.sin() * angle.cos(),
                phi1.cos(),
                phi1.sin() * angle.sin(),
            );
            lines.push((center + dir0 * radius, center + dir1 * radius));
        }
    }
    lines
}

fn scale(vector: [f32; 3], factor: f32) -> [f32; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}
