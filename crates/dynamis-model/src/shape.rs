#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    Sphere,
    Box,
    Capsule,
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeDesc {
    pub kind: ShapeKind,
    pub radius: f32,
    pub half_extents: [f32; 3],
    pub half_height: f32,
}

impl ShapeDesc {
    pub fn sphere(radius: f32) -> Self {
        Self {
            kind: ShapeKind::Sphere,
            radius,
            half_extents: [0.0; 3],
            half_height: 0.0,
        }
    }

    pub fn cuboid(half_extents: [f32; 3]) -> Self {
        Self {
            kind: ShapeKind::Box,
            radius: 0.0,
            half_extents,
            half_height: 0.0,
        }
    }

    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self {
            kind: ShapeKind::Capsule,
            radius,
            half_extents: [0.0; 3],
            half_height,
        }
    }

    pub fn bounding_radius(&self) -> f32 {
        match self.kind {
            ShapeKind::Sphere => self.radius,
            ShapeKind::Box => {
                let x = self.half_extents[0];
                let y = self.half_extents[1];
                let z = self.half_extents[2];
                (x * x + y * y + z * z).sqrt()
            }
            ShapeKind::Capsule => self.half_height + self.radius,
        }
    }

    pub fn inverse_inertia_diagonal(&self, inverse_mass: f32) -> [f32; 3] {
        if inverse_mass == 0.0 {
            return [0.0; 3];
        }
        let mass = 1.0 / inverse_mass;
        match self.kind {
            ShapeKind::Sphere => {
                let i = 2.0 / 5.0 * mass * self.radius * self.radius;
                [1.0 / i; 3]
            }
            ShapeKind::Box => {
                let hx = self.half_extents[0];
                let hy = self.half_extents[1];
                let hz = self.half_extents[2];
                let ex = 2.0 * hx;
                let ey = 2.0 * hy;
                let ez = 2.0 * hz;
                let ix = mass / 12.0 * (ey * ey + ez * ez);
                let iy = mass / 12.0 * (ex * ex + ez * ez);
                let iz = mass / 12.0 * (ex * ex + ey * ey);
                [1.0 / ix, 1.0 / iy, 1.0 / iz]
            }
            ShapeKind::Capsule => {
                let r = self.radius;
                let h = self.half_height;
                let cylinder_volume = std::f32::consts::PI * r * r * 2.0 * h;
                let sphere_volume = 4.0 / 3.0 * std::f32::consts::PI * r * r * r;
                let total = cylinder_volume + sphere_volume;
                let cylinder_mass = mass * cylinder_volume / total;
                let sphere_mass = mass * sphere_volume / total;
                let ix = cylinder_mass / 12.0 * (3.0 * r * r + (2.0 * h) * (2.0 * h))
                    + sphere_mass * (2.0 / 5.0 * r * r + (h + 3.0 / 8.0 * r) * (h + 3.0 / 8.0 * r))
                        * 2.0;
                let iy = cylinder_mass / 2.0 * r * r + sphere_mass * 2.0 / 5.0 * r * r * 2.0;
                [1.0 / ix, 1.0 / iy, 1.0 / ix]
            }
        }
    }
}
