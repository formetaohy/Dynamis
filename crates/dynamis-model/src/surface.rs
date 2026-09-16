#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceDesc {
    pub friction: f32,
    pub restitution: f32,
    pub rolling_friction: f32,
    pub spin_friction: f32,
    pub contact_frequency: f32,
    pub contact_damping_ratio: f32,
}

impl SurfaceDesc {
    pub fn new() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.0,
            rolling_friction: 0.0,
            spin_friction: 0.0,
            contact_frequency: f32::INFINITY,
            contact_damping_ratio: 1.0,
        }
    }

    pub fn friction(mut self, friction: f32) -> Self {
        assert!(friction >= 0.0, "surface friction must be non-negative");
        self.friction = friction;
        self
    }

    pub fn restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    pub fn rolling_friction(mut self, rolling_friction: f32) -> Self {
        assert!(
            rolling_friction >= 0.0,
            "surface rolling friction must be non-negative"
        );
        self.rolling_friction = rolling_friction;
        self
    }

    pub fn spin_friction(mut self, spin_friction: f32) -> Self {
        assert!(
            spin_friction >= 0.0,
            "surface spin friction must be non-negative"
        );
        self.spin_friction = spin_friction;
        self
    }

    pub fn contact_frequency(mut self, contact_frequency: f32) -> Self {
        assert!(
            contact_frequency > 0.0,
            "a contact frequency must be strictly positive"
        );
        self.contact_frequency = contact_frequency;
        self
    }

    pub fn contact_damping_ratio(mut self, contact_damping_ratio: f32) -> Self {
        assert!(
            contact_damping_ratio >= 0.0,
            "a contact damping ratio must be non-negative"
        );
        self.contact_damping_ratio = contact_damping_ratio;
        self
    }
}

impl Default for SurfaceDesc {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceTable<'a> {
    palette: &'a [SurfaceDesc],
    indices: &'a [u32],
}

impl<'a> SurfaceTable<'a> {
    pub fn new(palette: &'a [SurfaceDesc], indices: &'a [u32]) -> Self {
        assert!(
            !palette.is_empty(),
            "a surface table carries at least one surface"
        );
        assert!(
            indices
                .iter()
                .all(|index| (*index as usize) < palette.len()),
            "a surface index must address its own palette"
        );
        assert!(
            palette
                .iter()
                .all(|surface| !surface.contact_frequency.is_finite()),
            "a surface palette carries no contact softness; a collider does"
        );
        Self { palette, indices }
    }

    pub fn palette(self) -> &'a [SurfaceDesc] {
        self.palette
    }

    pub fn indices(self) -> &'a [u32] {
        self.indices
    }

    pub fn count(self) -> usize {
        self.indices.len()
    }
}
