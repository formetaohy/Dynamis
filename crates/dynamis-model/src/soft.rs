#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftBodyHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoftBodyDesc {
    pub particles: Vec<[f32; 3]>,
    pub inverse_masses: Vec<f32>,
    pub links: Vec<[u32; 2]>,
    pub radius: f32,
    pub friction: f32,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
}

impl SoftBodyDesc {
    pub fn new(particles: Vec<[f32; 3]>, links: Vec<[u32; 2]>) -> Self {
        let inverse_masses = vec![1.0; particles.len()];
        Self::with_masses(particles, inverse_masses, links)
    }

    pub fn with_masses(
        particles: Vec<[f32; 3]>,
        inverse_masses: Vec<f32>,
        links: Vec<[u32; 2]>,
    ) -> Self {
        assert!(
            !particles.is_empty(),
            "a soft body requires at least one particle"
        );
        assert_eq!(
            particles.len(),
            inverse_masses.len(),
            "a soft body requires one inverse mass per particle"
        );
        assert!(
            inverse_masses.iter().all(|mass| *mass >= 0.0),
            "soft body inverse masses must be non-negative"
        );
        assert!(
            particles
                .iter()
                .all(|particle| particle.iter().all(|value| value.is_finite())),
            "soft body particles must be finite"
        );
        for link in &links {
            let first = link[0] as usize;
            let second = link[1] as usize;
            assert!(
                first != second,
                "a soft body link must join two distinct particles"
            );
            assert!(
                first < particles.len() && second < particles.len(),
                "a soft body link must reference live particles"
            );
        }
        Self {
            particles,
            inverse_masses,
            links,
            radius: 0.0,
            friction: 0.5,
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
        }
    }

    pub fn lattice(extent: [u32; 3], spacing: f32, radius: f32) -> Self {
        let [rows, columns, layers] = extent;
        assert!(
            rows > 0 && columns > 0 && layers > 0,
            "a lattice needs a positive extent on every axis"
        );
        assert!(spacing > 0.0, "lattice spacing must be positive");
        assert!(
            extent.iter().all(|count| *count <= 64),
            "a lattice axis must not exceed 64 particles"
        );
        let index = |row: u32, column: u32, layer: u32| (layer * columns + column) * rows + row;
        let mut particles = Vec::with_capacity((rows * columns * layers) as usize);
        for layer in 0..layers {
            for column in 0..columns {
                for row in 0..rows {
                    particles.push([
                        (row as f32 - (rows - 1) as f32 * 0.5) * spacing,
                        (layer as f32 - (layers - 1) as f32 * 0.5) * spacing,
                        (column as f32 - (columns - 1) as f32 * 0.5) * spacing,
                    ]);
                }
            }
        }
        let mut links = Vec::new();
        for layer in 0..layers {
            for column in 0..columns {
                for row in 0..rows {
                    if row + 1 < rows {
                        links.push([index(row, column, layer), index(row + 1, column, layer)]);
                    }
                    if column + 1 < columns {
                        links.push([index(row, column, layer), index(row, column + 1, layer)]);
                    }
                    if layer + 1 < layers {
                        links.push([index(row, column, layer), index(row, column, layer + 1)]);
                    }
                }
            }
        }
        Self {
            radius,
            ..Self::new(particles, links)
        }
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
