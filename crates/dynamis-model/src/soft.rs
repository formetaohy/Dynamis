#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftBodyHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftElement {
    pub particles: [u32; 2],
    pub rest: f32,
    pub compliance: f32,
}

impl SoftElement {
    pub fn distance(first: u32, second: u32, rest: f32) -> Self {
        assert!(first != second, "a soft element needs distinct particles");
        assert!(rest > 0.0, "a soft element needs a positive rest length");
        Self {
            particles: [first, second],
            rest,
            compliance: 0.0,
        }
    }

    pub fn compliance(mut self, compliance: f32) -> Self {
        assert!(
            compliance >= 0.0,
            "a soft element compliance must be non-negative"
        );
        self.compliance = compliance;
        self
    }

    pub fn participants(&self) -> impl Iterator<Item = u32> + '_ {
        self.particles.iter().copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftMaterial {
    pub stretch: f32,
    pub shear: f32,
    pub bend: f32,
    pub volume: f32,
}

impl SoftMaterial {
    pub const fn rigid() -> Self {
        Self {
            stretch: 0.0,
            shear: 0.0,
            bend: 0.0,
            volume: 0.0,
        }
    }

    pub fn new(stretch: f32, shear: f32, bend: f32, volume: f32) -> Self {
        let material = Self {
            stretch,
            shear,
            bend,
            volume,
        };
        assert!(
            material.stretch >= 0.0
                && material.shear >= 0.0
                && material.bend >= 0.0
                && material.volume >= 0.0,
            "soft material compliances must be non-negative"
        );
        material
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoftBodyDesc {
    pub particles: Vec<[f32; 3]>,
    pub inverse_masses: Vec<f32>,
    pub elements: Vec<SoftElement>,
    pub radius: f32,
    pub friction: f32,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
}

impl SoftBodyDesc {
    pub fn new(particles: Vec<[f32; 3]>, elements: Vec<SoftElement>) -> Self {
        assert!(
            !particles.is_empty(),
            "a soft body requires at least one particle"
        );
        assert!(
            particles
                .iter()
                .all(|particle| particle.iter().all(|value| value.is_finite())),
            "soft body particles must be finite"
        );
        for element in &elements {
            for particle in element.participants() {
                assert!(
                    (particle as usize) < particles.len(),
                    "a soft element must reference live particles"
                );
            }
        }
        let inverse_masses = vec![1.0; particles.len()];
        Self {
            particles,
            inverse_masses,
            elements,
            radius: 0.0,
            friction: 0.5,
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
        }
    }

    pub fn net(particles: Vec<[f32; 3]>, links: Vec<[u32; 2]>) -> Self {
        let elements = links
            .iter()
            .map(|link| {
                SoftElement::distance(
                    link[0],
                    link[1],
                    particle_distance(&particles[link[0] as usize], &particles[link[1] as usize]),
                )
            })
            .collect();
        Self::new(particles, elements)
    }

    pub fn cloth(extent: [u32; 2], spacing: f32, material: SoftMaterial) -> Self {
        let [rows, columns] = extent;
        assert!(
            rows > 1 && columns > 1,
            "a cloth needs at least two rows and two columns"
        );
        assert!(
            rows.max(columns) <= 64,
            "a cloth axis must not exceed 64 particles"
        );
        assert!(spacing > 0.0, "cloth spacing must be positive");
        let index = |row: u32, column: u32| column * rows + row;
        let mut particles = Vec::with_capacity((rows * columns) as usize);
        for column in 0..columns {
            for row in 0..rows {
                particles.push([row as f32 * spacing, 0.0, column as f32 * spacing]);
            }
        }
        let mut elements = Vec::new();
        let mut distance = |first: u32, second: u32, compliance: f32| {
            elements.push(
                SoftElement::distance(
                    first,
                    second,
                    particle_distance(&particles[first as usize], &particles[second as usize]),
                )
                .compliance(compliance),
            );
        };
        for column in 0..columns {
            for row in 0..rows {
                if row + 1 < rows {
                    distance(index(row, column), index(row + 1, column), material.stretch);
                }
                if column + 1 < columns {
                    distance(index(row, column), index(row, column + 1), material.stretch);
                }
                if row + 1 < rows && column + 1 < columns {
                    distance(
                        index(row, column),
                        index(row + 1, column + 1),
                        material.shear,
                    );
                    distance(
                        index(row + 1, column),
                        index(row, column + 1),
                        material.shear,
                    );
                }
                if row + 2 < rows {
                    distance(index(row, column), index(row + 2, column), material.bend);
                }
                if column + 2 < columns {
                    distance(index(row, column), index(row, column + 2), material.bend);
                }
            }
        }
        Self::new(particles, elements)
    }

    pub fn lattice(extent: [u32; 3], spacing: f32, material: SoftMaterial) -> Self {
        let [rows, columns, layers] = extent;
        assert!(
            rows > 0 && columns > 0 && layers > 0,
            "a lattice needs a positive extent on every axis"
        );
        assert!(
            rows.max(columns).max(layers) <= 64,
            "a lattice axis must not exceed 64 particles"
        );
        assert!(spacing > 0.0, "lattice spacing must be positive");
        let index = |row: u32, column: u32, layer: u32| (layer * columns + column) * rows + row;
        let mut particles = Vec::with_capacity((rows * columns * layers) as usize);
        for layer in 0..layers {
            for column in 0..columns {
                for row in 0..rows {
                    particles.push([
                        row as f32 * spacing,
                        layer as f32 * spacing,
                        column as f32 * spacing,
                    ]);
                }
            }
        }
        let mut edges: Vec<[u32; 2]> = Vec::new();
        for layer in 0..layers.saturating_sub(1) {
            for column in 0..columns.saturating_sub(1) {
                for row in 0..rows.saturating_sub(1) {
                    let corners = [
                        index(row, column, layer),
                        index(row + 1, column, layer),
                        index(row, column + 1, layer),
                        index(row + 1, column + 1, layer),
                        index(row, column, layer + 1),
                        index(row + 1, column, layer + 1),
                        index(row, column + 1, layer + 1),
                        index(row + 1, column + 1, layer + 1),
                    ];
                    for tet in cube_tetrahedra(corners) {
                        for first in 0..3 {
                            for second in first + 1..4 {
                                let edge = [tet[first], tet[second]];
                                if !edges.contains(&edge) {
                                    edges.push(edge);
                                }
                            }
                        }
                    }
                }
            }
        }
        let elements = edges
            .into_iter()
            .map(|edge| {
                let rest =
                    particle_distance(&particles[edge[0] as usize], &particles[edge[1] as usize]);
                let compliance = if axis_aligned(&particles, edge) {
                    material.stretch
                } else {
                    material.shear
                };
                SoftElement::distance(edge[0], edge[1], rest).compliance(compliance)
            })
            .collect();
        Self::new(particles, elements)
    }

    pub fn inverse_masses(mut self, inverse_masses: Vec<f32>) -> Self {
        assert_eq!(
            self.particles.len(),
            inverse_masses.len(),
            "a soft body requires one inverse mass per particle"
        );
        assert!(
            inverse_masses.iter().all(|mass| *mass >= 0.0),
            "soft body inverse masses must be non-negative"
        );
        self.inverse_masses = inverse_masses;
        self
    }

    pub fn compliance(mut self, compliance: f32) -> Self {
        assert!(
            compliance >= 0.0,
            "soft body compliance must be non-negative"
        );
        for element in &mut self.elements {
            element.compliance = compliance;
        }
        self
    }

    pub fn pinned(mut self, indices: &[u32]) -> Self {
        for index in indices {
            let slot = *index as usize;
            assert!(
                slot < self.particles.len(),
                "a pinned particle must be part of the soft body"
            );
            self.inverse_masses[slot] = 0.0;
        }
        self
    }

    pub fn radius(mut self, radius: f32) -> Self {
        assert!(radius >= 0.0, "particle radius must be non-negative");
        self.radius = radius;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        assert!(friction >= 0.0, "soft body friction must be non-negative");
        self.friction = friction;
        self
    }

    pub fn position(mut self, position: [f32; 3]) -> Self {
        self.position = position;
        self
    }

    pub fn orientation(mut self, orientation: [f32; 4]) -> Self {
        assert!(
            (orientation[0] * orientation[0]
                + orientation[1] * orientation[1]
                + orientation[2] * orientation[2]
                + orientation[3] * orientation[3]
                - 1.0)
                .abs()
                < 1e-4,
            "orientation must be a unit quaternion"
        );
        self.orientation = orientation;
        self
    }

    pub fn velocity(mut self, velocity: [f32; 3]) -> Self {
        self.velocity = velocity;
        self
    }
}

fn particle_distance(first: &[f32; 3], second: &[f32; 3]) -> f32 {
    let dx = second[0] - first[0];
    let dy = second[1] - first[1];
    let dz = second[2] - first[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn axis_aligned(particles: &[[f32; 3]], edge: [u32; 2]) -> bool {
    let first = particles[edge[0] as usize];
    let second = particles[edge[1] as usize];
    let differing = (0..3)
        .filter(|axis| (first[*axis] - second[*axis]).abs() > 1e-6)
        .count();
    differing == 1
}

fn cube_tetrahedra(corners: [u32; 8]) -> [[u32; 4]; 6] {
    [
        [corners[0], corners[1], corners[3], corners[7]],
        [corners[0], corners[1], corners[5], corners[7]],
        [corners[0], corners[2], corners[3], corners[7]],
        [corners[0], corners[2], corners[6], corners[7]],
        [corners[0], corners[4], corners[5], corners[7]],
        [corners[0], corners[4], corners[6], corners[7]],
    ]
}
