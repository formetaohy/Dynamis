use dynamis_math::{add, dot, length, mul, negate, normalize, sub};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, QueryFilter, Shape};
use dynamis_world::{QueryHandle, QueryHit, World};

const SKIN: f32 = 0.05;

pub struct CharacterDesc {
    pub radius: f32,
    pub half_height: f32,
    pub step_height: f32,
    pub slope_limit: f32,
    pub max_speed: f32,
    pub jump_speed: f32,
}

impl Default for CharacterDesc {
    fn default() -> Self {
        Self {
            radius: 0.4,
            half_height: 0.5,
            step_height: 0.3,
            slope_limit: 50.0_f32.to_radians(),
            max_speed: 4.0,
            jump_speed: 5.0,
        }
    }
}

struct Cast {
    forward: Option<QueryHandle>,
    forward_low: Option<QueryHandle>,
    lifted_forward: Option<QueryHandle>,
    down: Option<QueryHandle>,
    ceiling: Option<QueryHandle>,
    down_length: f32,
}

impl Cast {
    fn handles(&self) -> [Option<QueryHandle>; 5] {
        [
            self.forward,
            self.forward_low,
            self.lifted_forward,
            self.down,
            self.ceiling,
        ]
    }
}

struct Sweeps {
    forward: Option<QueryHit>,
    lifted_forward: Option<QueryHit>,
    down: Option<QueryHit>,
    ceiling: Option<QueryHit>,
    down_length: f32,
}

pub struct Character {
    body: BodyHandle,
    desc: CharacterDesc,
    vertical: f32,
    grounded: bool,
    position: [f32; 3],
    cast: Option<Cast>,
}

impl Character {
    pub fn spawn(world: &mut World, position: [f32; 3], desc: CharacterDesc) -> Self {
        assert!(desc.radius > 0.0, "character radius must be positive");
        assert!(
            desc.half_height >= 0.0,
            "character half height must be non-negative"
        );
        assert!(
            desc.step_height >= 0.0,
            "character step height must be non-negative"
        );
        assert!(desc.max_speed > 0.0, "character max speed must be positive");
        assert!(
            desc.jump_speed >= 0.0,
            "character jump speed must be non-negative"
        );
        assert!(
            (0.0..=std::f32::consts::FRAC_PI_2).contains(&desc.slope_limit),
            "character slope limit must be within [0, pi/2]"
        );
        let body = world.spawn(
            BodyDesc::new(ColliderDesc::new(Shape::capsule(
                desc.radius,
                desc.half_height,
            )))
            .position(position)
            .kinematic(true),
        );
        Self {
            body,
            desc,
            vertical: 0.0,
            grounded: true,
            position,
            cast: None,
        }
    }

    pub fn step(&mut self, world: &mut World, dt: f32, move_dir: [f32; 3], jump: bool) {
        assert!(dt > 0.0, "character dt must be positive");
        let up = up_of(world.config().gravity);
        let gravity_magnitude = length(world.config().gravity);
        let horizontal = if length(move_dir) > 0.0 {
            mul(normalize(move_dir), self.desc.max_speed)
        } else {
            [0.0; 3]
        };
        let sweeps = self.take_sweeps(world, up);
        let mut jumped = false;
        if self.grounded {
            self.vertical = 0.0;
            if jump && gravity_magnitude > 0.0 {
                self.vertical = self.desc.jump_speed;
                jumped = true;
            }
        } else {
            self.vertical -= gravity_magnitude * dt;
        }
        let vertical = self.vertical;
        let mut target = self.position;
        let mut grounded = false;
        let mut vertical_after = vertical;
        let mut cpu_horizontal = horizontal;
        let mut press_dir = [0.0; 3];
        let slope_cos = self.desc.slope_limit.cos();
        if let Some(sweeps) = sweeps {
            if !jumped && let Some(landing) = sweeps.down {
                target = add(
                    target,
                    mul(
                        up,
                        -(landing.distance - SKIN).clamp(0.0, sweeps.down_length),
                    ),
                );
                let surface = negate(landing.normal);
                grounded = dot(surface, up) > slope_cos;
                vertical_after = if grounded { 0.0 } else { vertical };
            }
            if sweeps.forward.is_some() {
                match sweeps.lifted_forward {
                    None => {
                        target = add(self.position, mul(up, self.desc.step_height));
                        grounded = false;
                        vertical_after = vertical;
                    }
                    Some(_) => {
                        cpu_horizontal = [0.0; 3];
                        press_dir = normalize(horizontal);
                    }
                }
            }
            if sweeps.ceiling.is_some() && vertical_after > 0.0 {
                vertical_after = 0.0;
            }
        }
        let velocity = add(horizontal, mul(up, vertical_after));
        let moved = add(mul(cpu_horizontal, dt), mul(up, vertical_after * dt));
        self.position = add(target, moved);
        self.grounded = grounded;
        self.vertical = vertical_after;
        let press = mul(press_dir, SKIN * 0.5);
        let pre_move = sub(add(self.position, press), mul(velocity, dt));
        world.set_position(self.body, pre_move);
        world.set_velocity(self.body, velocity);
        self.cast = Some(self.cast_sweeps(world, dt, up, horizontal, vertical_after, jumped));
    }

    pub fn body(&self) -> BodyHandle {
        self.body
    }

    pub fn position(&self) -> [f32; 3] {
        self.position
    }

    pub fn grounded(&self) -> bool {
        self.grounded
    }

    pub fn vertical_speed(&self) -> f32 {
        self.vertical
    }

    fn take_sweeps(&mut self, world: &mut World, up: [f32; 3]) -> Option<Sweeps> {
        let cast = self.cast.take()?;
        world.poll();
        if cast
            .handles()
            .into_iter()
            .flatten()
            .any(|handle| !world.query_ready(handle))
        {
            world.resolve_queries();
            for handle in cast.handles().into_iter().flatten() {
                world.wait_query(handle);
            }
        }
        let forward = cast.forward.and_then(|handle| world.query_hit(handle));
        let forward_low = cast.forward_low.and_then(|handle| world.query_hit(handle));
        Some(Sweeps {
            forward: pick_forward(forward, forward_low, up),
            lifted_forward: cast
                .lifted_forward
                .and_then(|handle| world.query_hit(handle)),
            down: cast.down.and_then(|handle| world.query_hit(handle)),
            ceiling: cast.ceiling.and_then(|handle| world.query_hit(handle)),
            down_length: cast.down_length,
        })
    }

    fn cast_sweeps(
        &self,
        world: &mut World,
        dt: f32,
        up: [f32; 3],
        horizontal: [f32; 3],
        vertical: f32,
        jumped: bool,
    ) -> Cast {
        let position = self.position;
        let probe = Shape::sphere(self.desc.radius);
        let identity = [0.0, 0.0, 0.0, 1.0];
        let bottom = add(position, mul(up, -self.desc.half_height));
        let top = add(position, mul(up, self.desc.half_height));
        let filter = QueryFilter {
            exclude: Some(self.body),
            max_hits: 1,
            ..QueryFilter::default()
        };
        let horizontal_speed = length(horizontal);
        let forward_length = horizontal_speed * dt + SKIN;
        let down_length = SKIN + (vertical.abs() + horizontal_speed) * dt;
        let up_length = SKIN + vertical.max(0.0) * dt;
        let forward_dir = if horizontal_speed > 0.0 {
            Some(normalize(horizontal))
        } else {
            None
        };
        let forward = forward_dir.map(|direction| {
            world.sweep_query(
                &probe,
                identity,
                position,
                direction,
                forward_length,
                &filter,
            )
        });
        let forward_low = forward_dir.map(|direction| {
            world.sweep_query(&probe, identity, bottom, direction, forward_length, &filter)
        });
        let lifted = add(position, mul(up, self.desc.step_height));
        let lifted_forward = forward_dir.map(|direction| {
            world.sweep_query(&probe, identity, lifted, direction, forward_length, &filter)
        });
        let down = if !jumped {
            Some(world.sweep_query(&probe, identity, bottom, negate(up), down_length, &filter))
        } else {
            None
        };
        let ceiling = if vertical > 0.0 {
            Some(world.sweep_query(&probe, identity, top, up, up_length, &filter))
        } else {
            None
        };
        Cast {
            forward,
            forward_low,
            lifted_forward,
            down,
            ceiling,
            down_length,
        }
    }
}

fn up_of(gravity: [f32; 3]) -> [f32; 3] {
    let magnitude = length(gravity);
    if magnitude > 0.0 {
        mul(gravity, -1.0 / magnitude)
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn pick_forward(
    first: Option<QueryHit>,
    second: Option<QueryHit>,
    up: [f32; 3],
) -> Option<QueryHit> {
    let floor_like = |hit: &QueryHit| dot(hit.normal, up) < -0.5;
    match (first, second) {
        (Some(a), Some(b)) => {
            if floor_like(&a) && floor_like(&b) {
                None
            } else if floor_like(&a) {
                Some(b)
            } else if floor_like(&b) {
                Some(a)
            } else {
                Some(if a.distance <= b.distance { a } else { b })
            }
        }
        (Some(a), None) => {
            if floor_like(&a) {
                None
            } else {
                Some(a)
            }
        }
        (None, Some(b)) => {
            if floor_like(&b) {
                None
            } else {
                Some(b)
            }
        }
        (None, None) => None,
    }
}
