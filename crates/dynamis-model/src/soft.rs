use crate::BodyHandle;
use crate::collision::CollisionFilter;
use std::collections::{BTreeMap, BTreeSet};

const ELASTIC_STRAIN: f32 = f32::INFINITY;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftBodyHandle {
    pub id: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftElementKind {
    Distance = 0,
    Area = 1,
    Bend = 2,
    Volume = 3,
}

impl SoftElementKind {
    pub const fn arity(self) -> usize {
        match self {
            Self::Distance => 2,
            Self::Area => 3,
            Self::Bend | Self::Volume => SoftElement::PARTICLES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftElement {
    kind: SoftElementKind,
    particles: [u32; SoftElement::PARTICLES],
    rest: f32,
    compliance: f32,
    yield_strain: f32,
    break_strain: f32,
    plastic_flow: f32,
}

impl SoftElement {
    pub const PARTICLES: usize = 4;
    pub const UNUSED: u32 = u32::MAX;

    fn build(kind: SoftElementKind, particles: [u32; Self::PARTICLES], rest: f32) -> Self {
        let participants = &particles[..kind.arity()];
        for (slot, particle) in participants.iter().enumerate() {
            assert!(
                participants[..slot].iter().all(|other| other != particle),
                "a soft element needs distinct particles"
            );
        }
        Self {
            kind,
            particles,
            rest,
            compliance: 0.0,
            yield_strain: ELASTIC_STRAIN,
            break_strain: ELASTIC_STRAIN,
            plastic_flow: 0.0,
        }
    }

    pub fn distance(first: u32, second: u32, rest: f32) -> Self {
        assert!(
            rest > 0.0,
            "a distance element needs a positive rest length"
        );
        let mut particles = [Self::UNUSED; Self::PARTICLES];
        particles[..2].copy_from_slice(&[first, second]);
        Self::build(SoftElementKind::Distance, particles, rest)
    }

    pub fn area(first: u32, second: u32, third: u32, rest: f32) -> Self {
        assert!(rest > 0.0, "an area element needs a positive rest area");
        let mut particles = [Self::UNUSED; Self::PARTICLES];
        particles[..3].copy_from_slice(&[first, second, third]);
        Self::build(SoftElementKind::Area, particles, rest)
    }

    pub fn bend(apex_a: u32, apex_b: u32, edge_a: u32, edge_b: u32, rest: f32) -> Self {
        assert!(
            (0.0..=std::f32::consts::PI).contains(&rest),
            "a bend element needs a rest angle within [0, pi]"
        );
        Self::build(
            SoftElementKind::Bend,
            [apex_a, apex_b, edge_a, edge_b],
            rest,
        )
    }

    pub fn volume(first: u32, second: u32, third: u32, fourth: u32, rest: f32) -> Self {
        assert!(rest > 0.0, "a volume element needs a positive rest volume");
        Self::build(
            SoftElementKind::Volume,
            [first, second, third, fourth],
            rest,
        )
    }

    pub fn compliance(mut self, compliance: f32) -> Self {
        assert!(
            compliance >= 0.0,
            "a soft element compliance must be non-negative"
        );
        self.compliance = compliance;
        self
    }

    pub fn yielding(mut self, yield_strain: f32, plastic_flow: f32) -> Self {
        self.yield_strain = assert_strain(yield_strain, "yield");
        self.plastic_flow = assert_flow(plastic_flow);
        self
    }

    pub fn fracturing(mut self, break_strain: f32) -> Self {
        self.break_strain = assert_strain(break_strain, "break");
        self
    }

    pub const fn kind(&self) -> SoftElementKind {
        self.kind
    }

    pub const fn rest(&self) -> f32 {
        self.rest
    }

    pub const fn compliance_of(&self) -> f32 {
        self.compliance
    }

    pub const fn yield_strain_of(&self) -> f32 {
        self.yield_strain
    }

    pub const fn break_strain_of(&self) -> f32 {
        self.break_strain
    }

    pub const fn plastic_flow_of(&self) -> f32 {
        self.plastic_flow
    }

    pub fn carries_strength(&self) -> bool {
        self.yield_strain.is_finite() || self.break_strain.is_finite()
    }

    pub const fn particles(&self) -> [u32; Self::PARTICLES] {
        self.particles
    }

    pub fn participants(&self) -> impl Iterator<Item = u32> + '_ {
        self.particles[..self.kind.arity()].iter().copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftMaterial {
    pub stretch: f32,
    pub shear: f32,
    pub bend: f32,
    pub volume: f32,
    pub yield_strain: f32,
    pub break_strain: f32,
    pub plastic_flow: f32,
}

impl SoftMaterial {
    pub const fn rigid() -> Self {
        Self {
            stretch: 0.0,
            shear: 0.0,
            bend: 0.0,
            volume: 0.0,
            yield_strain: ELASTIC_STRAIN,
            break_strain: ELASTIC_STRAIN,
            plastic_flow: 0.0,
        }
    }

    pub fn new(stretch: f32, shear: f32, bend: f32, volume: f32) -> Self {
        let material = Self {
            stretch,
            shear,
            bend,
            volume,
            ..Self::rigid()
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

    pub fn yielding(mut self, yield_strain: f32, plastic_flow: f32) -> Self {
        self.yield_strain = assert_strain(yield_strain, "yield");
        self.plastic_flow = assert_flow(plastic_flow);
        self
    }

    pub fn fracturing(mut self, break_strain: f32) -> Self {
        self.break_strain = assert_strain(break_strain, "break");
        self
    }
}

fn assert_strain(strain: f32, role: &str) -> f32 {
    assert!(strain >= 0.0, "a soft {role} strain must be non-negative");
    strain
}

fn assert_flow(plastic_flow: f32) -> f32 {
    assert!(
        (0.0..=1.0).contains(&plastic_flow),
        "a soft plastic flow must be within [0, 1]"
    );
    plastic_flow
}

fn apply_strength(material: &SoftMaterial, element: SoftElement) -> SoftElement {
    element
        .yielding(material.yield_strain, material.plastic_flow)
        .fracturing(material.break_strain)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftAttachment {
    particle: u32,
    body: BodyHandle,
    local: [f32; 3],
}

impl SoftAttachment {
    pub fn new(particle: u32, body: BodyHandle, local: [f32; 3]) -> Self {
        assert!(
            local.iter().all(|value| value.is_finite()),
            "a soft attachment anchor must be finite"
        );
        Self {
            particle,
            body,
            local,
        }
    }

    pub const fn particle(&self) -> u32 {
        self.particle
    }

    pub const fn body(&self) -> BodyHandle {
        self.body
    }

    pub const fn local(&self) -> [f32; 3] {
        self.local
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftElementState {
    kind: SoftElementKind,
    particles: [u32; SoftElement::PARTICLES],
    rest: f32,
    broken: bool,
}

impl SoftElementState {
    pub const fn new(
        kind: SoftElementKind,
        particles: [u32; SoftElement::PARTICLES],
        rest: f32,
        broken: bool,
    ) -> Self {
        Self {
            kind,
            particles,
            rest,
            broken,
        }
    }

    pub const fn kind(&self) -> SoftElementKind {
        self.kind
    }

    pub const fn rest(&self) -> f32 {
        self.rest
    }

    pub const fn broken(&self) -> bool {
        self.broken
    }

    pub fn participants(&self) -> impl Iterator<Item = u32> + '_ {
        self.particles[..self.kind.arity()].iter().copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidMaterial {
    spacing: f32,
    support: f32,
}

impl FluidMaterial {
    pub fn new(spacing: f32, support: f32) -> Self {
        assert!(
            spacing > 0.0,
            "a fluid rest spacing must be strictly positive"
        );
        assert!(
            support > spacing,
            "a fluid support radius must exceed its rest spacing"
        );
        Self { spacing, support }
    }

    pub const fn spacing(&self) -> f32 {
        self.spacing
    }

    pub const fn support(&self) -> f32 {
        self.support
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoftBodyDesc {
    pub particles: Vec<[f32; 3]>,
    pub inverse_masses: Vec<f32>,
    pub elements: Vec<SoftElement>,
    pub attachments: Vec<SoftAttachment>,
    pub fluid: Option<FluidMaterial>,
    pub radius: f32,
    pub friction: f32,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub filter: CollisionFilter,
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
            attachments: Vec::new(),
            fluid: None,
            radius: 0.0,
            friction: 0.5,
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            filter: CollisionFilter::DEFAULT,
        }
    }

    pub fn attach(mut self, attachment: SoftAttachment) -> Self {
        self.attachments.push(attachment);
        self.assert_attachments();
        self
    }

    pub fn assert_attachments(&self) {
        for (slot, attachment) in self.attachments.iter().enumerate() {
            assert!(
                (attachment.particle() as usize) < self.particles.len(),
                "a soft attachment must reference a live particle"
            );
            assert!(
                self.attachments[..slot]
                    .iter()
                    .all(|other| other.particle() != attachment.particle()),
                "a soft particle carries at most one attachment"
            );
        }
    }

    pub fn fluid(particles: Vec<[f32; 3]>, radius: f32, material: FluidMaterial) -> Self {
        assert!(
            radius > 0.0,
            "a fluid particle radius must be strictly positive"
        );
        assert!(
            2.0 * radius <= material.spacing(),
            "a fluid particle must be narrower than its rest spacing"
        );
        let mut body = Self::new(particles, Vec::new());
        body.fluid = Some(material);
        body.radius = radius;
        body
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
        let mut triangles = Vec::new();
        for column in 0..columns - 1 {
            for row in 0..rows - 1 {
                let first = index(row, column);
                let second = index(row + 1, column);
                let third = index(row, column + 1);
                let fourth = index(row + 1, column + 1);
                triangles.push([first, second, fourth]);
                triangles.push([first, fourth, third]);
            }
        }
        let mut elements = Vec::new();
        for column in 0..columns {
            for row in 0..rows {
                if row + 1 < rows {
                    elements.push(apply_strength(
                        &material,
                        edge_element(
                            &particles,
                            index(row, column),
                            index(row + 1, column),
                            material.stretch,
                        ),
                    ));
                }
                if column + 1 < columns {
                    elements.push(apply_strength(
                        &material,
                        edge_element(
                            &particles,
                            index(row, column),
                            index(row, column + 1),
                            material.stretch,
                        ),
                    ));
                }
            }
        }
        for triangle in &triangles {
            elements.push(apply_strength(
                &material,
                SoftElement::area(
                    triangle[0],
                    triangle[1],
                    triangle[2],
                    triangle_area(&particles, *triangle),
                )
                .compliance(material.shear),
            ));
        }
        for [edge_a, edge_b, apex_a, apex_b] in shared_edges(&triangles) {
            elements.push(apply_strength(
                &material,
                SoftElement::bend(
                    apex_a,
                    apex_b,
                    edge_a,
                    edge_b,
                    dihedral_angle(&particles, apex_a, apex_b, edge_a, edge_b),
                )
                .compliance(material.bend),
            ));
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
        let mut edges: BTreeSet<[u32; 2]> = BTreeSet::new();
        let mut tets: Vec<[u32; 4]> = Vec::new();
        for layer in 0..layers.saturating_sub(1) {
            for column in 0..columns.saturating_sub(1) {
                for row in 0..rows.saturating_sub(1) {
                    let corners = [
                        index(row, column, layer),
                        index(row + 1, column, layer),
                        index(row, column, layer + 1),
                        index(row + 1, column, layer + 1),
                        index(row, column + 1, layer),
                        index(row + 1, column + 1, layer),
                        index(row, column + 1, layer + 1),
                        index(row + 1, column + 1, layer + 1),
                    ];
                    for tet in cube_tetrahedra(corners) {
                        for first in 0..3 {
                            for second in first + 1..4 {
                                let edge = if tet[first] < tet[second] {
                                    [tet[first], tet[second]]
                                } else {
                                    [tet[second], tet[first]]
                                };
                                edges.insert(edge);
                            }
                        }
                        tets.push(tet);
                    }
                }
            }
        }
        let mut elements = Vec::new();
        for edge in &edges {
            let compliance = if axis_aligned(&particles, *edge) {
                material.stretch
            } else {
                material.shear
            };
            elements.push(apply_strength(
                &material,
                edge_element(&particles, edge[0], edge[1], compliance),
            ));
        }
        for tet in &tets {
            elements.push(apply_strength(
                &material,
                SoftElement::volume(
                    tet[0],
                    tet[1],
                    tet[2],
                    tet[3],
                    tetrahedron_volume(&particles, *tet),
                )
                .compliance(material.volume),
            ));
        }
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
            *element = element.compliance(compliance);
        }
        self
    }

    pub fn yielding(mut self, yield_strain: f32, plastic_flow: f32) -> Self {
        for element in &mut self.elements {
            *element = element.yielding(yield_strain, plastic_flow);
        }
        self
    }

    pub fn fracturing(mut self, break_strain: f32) -> Self {
        for element in &mut self.elements {
            *element = element.fracturing(break_strain);
        }
        self
    }

    pub fn carries_strength(&self) -> bool {
        self.elements.iter().any(SoftElement::carries_strength)
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

fn offset(particles: &[[f32; 3]], first: u32, second: u32) -> [f32; 3] {
    let from = particles[first as usize];
    let to = particles[second as usize];
    [to[0] - from[0], to[1] - from[1], to[2] - from[2]]
}

fn cross(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn dot(first: [f32; 3], second: [f32; 3]) -> f32 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn length(vector: [f32; 3]) -> f32 {
    dot(vector, vector).sqrt()
}

fn edge_element(particles: &[[f32; 3]], first: u32, second: u32, compliance: f32) -> SoftElement {
    SoftElement::distance(
        first,
        second,
        particle_distance(&particles[first as usize], &particles[second as usize]),
    )
    .compliance(compliance)
}

fn triangle_area(particles: &[[f32; 3]], triangle: [u32; 3]) -> f32 {
    0.5 * length(cross(
        offset(particles, triangle[0], triangle[1]),
        offset(particles, triangle[0], triangle[2]),
    ))
}

fn tetrahedron_volume(particles: &[[f32; 3]], tet: [u32; 4]) -> f32 {
    dot(
        cross(
            offset(particles, tet[0], tet[1]),
            offset(particles, tet[0], tet[2]),
        ),
        offset(particles, tet[0], tet[3]),
    )
    .abs()
        / 6.0
}

fn dihedral_angle(
    particles: &[[f32; 3]],
    apex_a: u32,
    apex_b: u32,
    edge_a: u32,
    edge_b: u32,
) -> f32 {
    let first = cross(
        offset(particles, apex_a, edge_a),
        offset(particles, apex_a, edge_b),
    );
    let second = cross(
        offset(particles, apex_b, edge_b),
        offset(particles, apex_b, edge_a),
    );
    let first_length = length(first);
    let second_length = length(second);
    assert!(
        first_length > 0.0 && second_length > 0.0,
        "a bend element needs two non-degenerate triangles"
    );
    (dot(first, second) / (first_length * second_length))
        .clamp(-1.0, 1.0)
        .acos()
}

fn shared_edges(triangles: &[[u32; 3]]) -> Vec<[u32; 4]> {
    let mut apexes: BTreeMap<[u32; 2], Vec<u32>> = BTreeMap::new();
    for triangle in triangles {
        for role in 0..3 {
            let edge_a = triangle[role];
            let edge_b = triangle[(role + 1) % 3];
            let apex = triangle[(role + 2) % 3];
            let key = if edge_a < edge_b {
                [edge_a, edge_b]
            } else {
                [edge_b, edge_a]
            };
            let shared = apexes.entry(key).or_default();
            assert!(
                shared.len() < 2,
                "a cloth edge must belong to at most two triangles"
            );
            shared.push(apex);
        }
    }
    apexes
        .into_iter()
        .filter_map(|([edge_a, edge_b], shared)| {
            let [apex_a, apex_b] = shared[..] else {
                return None;
            };
            Some([edge_a, edge_b, apex_a, apex_b])
        })
        .collect()
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
        [corners[0], corners[3], corners[2], corners[7]],
        [corners[0], corners[2], corners[6], corners[7]],
        [corners[0], corners[6], corners[4], corners[7]],
        [corners[0], corners[4], corners[5], corners[7]],
        [corners[0], corners[5], corners[1], corners[7]],
    ]
}
