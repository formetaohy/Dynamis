//! 极简 wgpu 渲染器：窗口、网格、材质、光照、相机、gizmo 与 HUD 文本。

mod geometry;
mod gpu;
mod text;

pub use geometry::Geometry;
pub use glam::{EulerRot, Mat4, Quat, Vec3};

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Clone, Copy)]
pub struct MeshId(pub(crate) usize);

#[derive(Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    pub const fn srgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const fn srgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn srgb_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: 1.0,
        }
    }

    pub const fn linear(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

impl Color {
    pub(crate) fn linear_channels(self) -> [f32; 4] {
        [
            srgb_to_linear(self.r),
            srgb_to_linear(self.g),
            srgb_to_linear(self.b),
            self.a,
        ]
    }
}

fn srgb_to_linear(channel: f32) -> f32 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

#[derive(Clone, Copy)]
pub struct Material {
    pub base_color: Color,
    pub roughness: f32,
    pub emissive: Color,
    pub unlit: bool,
}

impl From<Color> for Material {
    fn from(base_color: Color) -> Self {
        Self {
            base_color,
            roughness: 0.55,
            emissive: Color::BLACK,
            unlit: false,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Color::WHITE,
            roughness: 0.55,
            emissive: Color::BLACK,
            unlit: false,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self {
            translation: Vec3::new(x, y, z),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation,
            scale: Vec3::ONE,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let projection = Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far);
        projection * view
    }

    pub fn ray_from_screen(&self, x: f32, y: f32, width: f32, height: f32) -> (Vec3, Vec3) {
        let ndc_x = x / width * 2.0 - 1.0;
        let ndc_y = 1.0 - y / height * 2.0;
        let projection = Mat4::perspective_rh(self.fov_y, width / height, self.near, self.far);
        let inverse = (projection * Mat4::look_at_rh(self.eye, self.target, self.up)).inverse();
        let far_point = inverse * Vec3::new(ndc_x, ndc_y, 1.0).extend(1.0);
        let far_point = far_point.truncate() / far_point.w;
        (self.eye, (far_point - self.eye).normalize())
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 3.0, 12.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 45.0_f32.to_radians(),
            near: 0.1,
            far: 300.0,
        }
    }
}

pub struct Lights {
    pub direction: Vec3,
    pub color: Color,
    pub intensity: f32,
    pub ambient: Color,
    pub ambient_intensity: f32,
}

impl Default for Lights {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.5, -0.8, -0.3),
            color: Color::srgb(1.0, 0.94, 0.84),
            intensity: 1.8,
            ambient: Color::srgb(0.6, 0.62, 0.68),
            ambient_intensity: 0.3,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Input {
    pub cursor: Option<(f32, f32)>,
    pub cursor_delta: (f32, f32),
    pub left_down: bool,
    pub left_pressed: bool,
    pub right_down: bool,
    pub right_pressed: bool,
    pub wheel: f32,
}

impl Input {
    pub(crate) fn end_frame(&mut self) {
        self.cursor_delta = (0.0, 0.0);
        self.left_pressed = false;
        self.right_pressed = false;
        self.wheel = 0.0;
    }
}

pub struct Time {
    started: Instant,
    previous: Instant,
    current: Instant,
}

impl Time {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            previous: now,
            current: now,
        }
    }

    pub fn delta_secs(&self) -> f32 {
        (self.current - self.previous).as_secs_f32()
    }

    pub fn elapsed_secs(&self) -> f32 {
        (self.current - self.started).as_secs_f32()
    }

    fn tick(&mut self, now: Instant) {
        self.previous = self.current;
        self.current = now;
    }
}

pub struct AppContext {
    pub input: Input,
    pub time: Time,
    pub camera: Camera,
    pub lights: Lights,
    pub hud: String,
    pub(crate) gpu: gpu::Gpu,
    pub(crate) meshes: Vec<gpu::MeshSlot>,
    pub(crate) lines: Vec<geometry::LineVertex>,
}

impl AppContext {
    pub fn spawn(
        &mut self,
        geometry: &Geometry,
        material: Material,
        transform: Transform,
    ) -> MeshId {
        let mesh = self.gpu.mesh_slot(geometry, material, transform);
        self.meshes.push(mesh);
        MeshId(self.meshes.len() - 1)
    }

    pub fn mesh_transform(&mut self, mesh: MeshId) -> &mut Transform {
        &mut self.meshes[mesh.0].transform
    }

    pub fn mesh_material(&mut self, mesh: MeshId) -> &mut Material {
        &mut self.meshes[mesh.0].material
    }

    pub fn ray_from_screen(&self, x: f32, y: f32) -> (Vec3, Vec3) {
        let (width, height) = self.gpu.surface_size();
        self.camera
            .ray_from_screen(x, y, width as f32, height as f32)
    }

    pub fn gizmo_line(&mut self, from: Vec3, to: Vec3, color: Color) {
        let color = color.linear_channels();
        self.lines.push(geometry::LineVertex {
            position: from.to_array(),
            color,
        });
        self.lines.push(geometry::LineVertex {
            position: to.to_array(),
            color,
        });
    }

    pub fn gizmo_ray(&mut self, origin: Vec3, direction: Vec3, color: Color) {
        self.gizmo_line(origin, origin + direction, color);
    }

    pub fn gizmo_sphere(&mut self, center: Vec3, radius: f32, color: Color) {
        for (from, to) in geometry::sphere_wireframe(center, radius, 12, 16) {
            self.gizmo_line(from, to, color);
        }
    }
}

type Startup<T> = Box<dyn FnOnce(&mut AppContext, &mut T) + 'static>;
type Update<T> = Box<dyn FnMut(&mut AppContext, &mut T) + 'static>;

pub struct App<T> {
    title: String,
    user: T,
    startup: Option<Startup<T>>,
    update: Option<Update<T>>,
    window: Option<Arc<Window>>,
    context: Option<AppContext>,
}

impl<T> App<T> {
    pub fn new(title: impl Into<String>, user: T) -> Self {
        Self {
            title: title.into(),
            user,
            startup: None,
            update: None,
            window: None,
            context: None,
        }
    }

    pub fn on_startup(mut self, startup: impl FnOnce(&mut AppContext, &mut T) + 'static) -> Self {
        self.startup = Some(Box::new(startup));
        self
    }

    pub fn on_update(mut self, update: impl FnMut(&mut AppContext, &mut T) + 'static) -> Self {
        self.update = Some(Box::new(update));
        self
    }

    pub fn run(mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let event_loop = EventLoop::new()?;
        event_loop.run_app(&mut self)?;
        Ok(())
    }
}

impl<T> ApplicationHandler for App<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(self.title.clone())
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
                )
                .expect("failed to create window"),
        );
        let gpu = gpu::Gpu::new(window.clone());
        let context = AppContext {
            input: Input::default(),
            time: Time::new(),
            camera: Camera::default(),
            lights: Lights::default(),
            hud: String::new(),
            gpu,
            meshes: Vec::new(),
            lines: Vec::new(),
        };
        self.window = Some(window);
        self.context = Some(context);
        if let Some(startup) = self.startup.take() {
            startup(
                self.context.as_mut().expect("context exists"),
                &mut self.user,
            );
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(context) = self.context.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => context.gpu.resize(size.width, size.height),
            WindowEvent::MouseInput { state, button, .. } => match (state, button) {
                (ElementState::Pressed, MouseButton::Left) => {
                    context.input.left_down = true;
                    context.input.left_pressed = true;
                }
                (ElementState::Released, MouseButton::Left) => context.input.left_down = false,
                (ElementState::Pressed, MouseButton::Right) => {
                    context.input.right_down = true;
                    context.input.right_pressed = true;
                }
                (ElementState::Released, MouseButton::Right) => context.input.right_down = false,
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;
                if let Some((last_x, last_y)) = context.input.cursor {
                    context.input.cursor_delta.0 += x - last_x;
                    context.input.cursor_delta.1 += y - last_y;
                }
                context.input.cursor = Some((x, y));
            }
            WindowEvent::CursorLeft { .. } => context.input.cursor = None,
            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::LineDelta(_, y) => context.input.wheel += y,
                MouseScrollDelta::PixelDelta(position) => {
                    context.input.wheel += position.y as f32 / 120.0;
                }
            },
            WindowEvent::RedrawRequested => {
                context.time.tick(Instant::now());
                if let Some(update) = self.update.as_mut() {
                    update(context, &mut self.user);
                }
                self.render_frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl<T> App<T> {
    fn render_frame(&mut self) {
        let context = self.context.as_mut().expect("context exists");
        let AppContext {
            gpu,
            meshes,
            lines,
            camera,
            lights,
            hud,
            input,
            ..
        } = context;
        gpu.render(meshes, lines, camera, lights, hud);
        input.end_frame();
    }
}
