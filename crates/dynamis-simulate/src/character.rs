use crate::simulation::Simulation;
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, QueryFilter, Shape};
use dynamis_query::QueryHit;

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

pub struct Character {
    body: BodyHandle,
    desc: CharacterDesc,
    up: [f32; 3],
    vertical: f32,
    grounded: bool,
    position: [f32; 3],
}

impl Character {
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
}

impl Simulation {
    pub fn spawn_character(&mut self, position: [f32; 3], desc: CharacterDesc) -> Character {
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
        let body = self.spawn(
            BodyDesc::new(ColliderDesc::new(Shape::capsule(
                desc.radius,
                desc.half_height,
            )))
            .position(position)
            .kinematic(true),
        );
        let up = self.character_up();
        Character {
            body,
            desc,
            up,
            vertical: 0.0,
            grounded: true,
            position,
        }
    }

    pub fn character_step(
        &mut self,
        character: &mut Character,
        dt: f32,
        move_dir: [f32; 3],
        jump: bool,
    ) {
        assert!(dt > 0.0, "character dt must be positive");
        let up = self.character_up();
        character.up = up;
        let gravity = self.config().gravity;
        let gravity_magnitude = vec_len(gravity);
        let horizontal = if vec_len(move_dir) > 0.0 {
            scale(normalize(move_dir), character.desc.max_speed)
        } else {
            [0.0; 3]
        };
        let mut jumped = false;
        if character.grounded {
            character.vertical = 0.0;
            if jump && gravity_magnitude > 0.0 {
                character.vertical = character.desc.jump_speed;
                character.grounded = false;
                jumped = true;
            }
        } else {
            character.vertical -= gravity_magnitude * dt;
        }
        let vertical = character.vertical;
        let position = character.position;
        let probe = Shape::sphere(character.desc.radius);
        let identity = [0.0, 0.0, 0.0, 1.0];
        let bottom = add(position, scale(up, -character.desc.half_height));
        let top = add(position, scale(up, character.desc.half_height));
        let filter = QueryFilter {
            exclude: Some(character.body),
            max_hits: 1,
            ..QueryFilter::default()
        };
        let horizontal_speed = vec_len(horizontal);
        let forward_length = horizontal_speed * dt + SKIN;
        let down_length = SKIN + (vertical.abs() + horizontal_speed) * dt;
        let up_length = SKIN + vertical.max(0.0) * dt;
        let forward_dir = if horizontal_speed > 0.0 {
            Some(normalize(horizontal))
        } else {
            None
        };
        let forward = forward_dir.map(|direction| {
            self.sweep_query(
                &probe,
                identity,
                position,
                direction,
                forward_length,
                &filter,
            )
        });
        let forward_low = forward_dir.map(|direction| {
            self.sweep_query(&probe, identity, bottom, direction, forward_length, &filter)
        });
        let lifted = add(position, scale(up, character.desc.step_height));
        let lifted_forward = forward_dir.map(|direction| {
            self.sweep_query(&probe, identity, lifted, direction, forward_length, &filter)
        });
        let down = if !jumped {
            Some(self.sweep_query(&probe, identity, bottom, negate(up), down_length, &filter))
        } else {
            None
        };
        let up_hit = if vertical > 0.0 {
            Some(self.sweep_query(&probe, identity, top, up, up_length, &filter))
        } else {
            None
        };
        self.flush_queries();

        let mut target = position;
        let mut grounded = false;
        let mut vertical_after = vertical;
        let mut cpu_horizontal = horizontal;
        let mut press_dir = [0.0; 3];
        let slope_cos = character.desc.slope_limit.cos();
        let forward_hit = pick_forward(
            forward.and_then(|handle| self.query_hit(handle)),
            forward_low.and_then(|handle| self.query_hit(handle)),
            up,
        );
        let lifted_hit = lifted_forward.and_then(|handle| self.query_hit(handle));
        let down_hit = down.and_then(|handle| self.query_hit(handle));
        let ceiling_hit = up_hit.and_then(|handle| self.query_hit(handle));

        if !jumped && let Some(landing) = down_hit {
            target = add(
                target,
                scale(up, -(landing.distance - SKIN).clamp(0.0, down_length)),
            );
            let surface = negate(landing.normal);
            grounded = dot(surface, up) > slope_cos;
            vertical_after = if grounded { 0.0 } else { vertical };
        }
        if let Some(_hit) = forward_hit {
            match lifted_hit {
                None => {
                    target = lifted;
                    grounded = false;
                    vertical_after = vertical;
                }
                Some(_) => {
                    cpu_horizontal = [0.0; 3];
                    press_dir = normalize(horizontal);
                }
            }
        }

        if let Some(_ceiling) = ceiling_hit
            && vertical_after > 0.0
        {
            vertical_after = 0.0;
        }

        let velocity = add(horizontal, scale(up, vertical_after));
        let moved = add(scale(cpu_horizontal, dt), scale(up, vertical_after * dt));
        character.position = add(target, moved);
        character.grounded = grounded;
        character.vertical = vertical_after;
        let press = scale(press_dir, SKIN * 0.5);
        let pre_move = sub(add(character.position, press), scale(velocity, dt));
        self.set_position(character.body, pre_move);
        self.set_velocity(character.body, velocity);
    }

    fn character_up(&self) -> [f32; 3] {
        let gravity = self.config().gravity;
        let magnitude = vec_len(gravity);
        if magnitude > 0.0 {
            scale(gravity, -1.0 / magnitude)
        } else {
            [0.0, 1.0, 0.0]
        }
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

fn normalize(v: [f32; 3]) -> [f32; 3] {
    scale(v, 1.0 / vec_len(v))
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec_len(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

fn negate(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}
