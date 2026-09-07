use crate::geometry::{LineVertex, Vertex};
use crate::text;
use crate::{Camera, Geometry, Lights, Material, Transform};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use winit::window::Window;

const INSTANCE_STRIDE: usize = 256;
const INITIAL_INSTANCES: usize = 256;
const INITIAL_LINES: usize = 4096;
const TEXT_VERTICES: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightUniform {
    direction: [f32; 4],
    color: [f32; 4],
    ambient: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceUniform {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

pub(crate) struct MeshSlot {
    pub material: Material,
    pub transform: Transform,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

pub(crate) struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    camera_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    instances_buffer: wgpu::Buffer,
    instance_capacity: usize,
    global_bind_group_layout: wgpu::BindGroupLayout,
    global_bind_group: wgpu::BindGroup,
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_transparent_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    line_bind_group: wgpu::BindGroup,
    lines_buffer: wgpu::Buffer,
    line_capacity: usize,
    text_pipeline: wgpu::RenderPipeline,
    text_bind_group: wgpu::BindGroup,
    text_texture: wgpu::Texture,
    text_vertex_buffer: wgpu::Buffer,
}

impl Gpu {
    pub fn new(window: Arc<Window>) -> Self {
        let mut instance = wgpu::InstanceDescriptor::new_without_display_handle();
        instance.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(instance);
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no compatible GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("dynamis example device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::default(),
            ..Default::default()
        }))
        .expect("failed to request GPU device");
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        config.present_mode = wgpu::PresentMode::AutoVsync;
        config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
        surface.configure(&device, &config);

        let camera_buffer = create_uniform_buffer(&device, "camera", 80);
        let lights_buffer = create_uniform_buffer(&device, "lights", 48);
        let instance_capacity = INITIAL_INSTANCES;
        let instances_buffer = create_uniform_buffer(
            &device,
            "instances",
            (instance_capacity * INSTANCE_STRIDE) as u64,
        );
        let line_capacity = INITIAL_LINES;
        let lines_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo lines"),
            size: (line_capacity * size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_view = create_depth_view(&device, config.width, config.height);
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("line.wgsl").into()),
        });
        let text_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("text.wgsl").into()),
        });

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera lights instances"),
                entries: &[
                    uniform_binding(0),
                    uniform_binding(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: std::num::NonZeroU64::new(
                                size_of::<InstanceUniform>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });
        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera lights instances"),
            layout: &global_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &instances_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        let global_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mesh pipeline layout"),
                bind_group_layouts: &[Some(&global_bind_group_layout)],
                immediate_size: 0,
            });
        let mesh_pipeline = create_mesh_pipeline(
            &device,
            config.format,
            &global_pipeline_layout,
            &mesh_shader,
            None,
        );
        let mesh_transparent_pipeline = create_mesh_pipeline(
            &device,
            config.format,
            &global_pipeline_layout,
            &mesh_shader,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        let line_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[uniform_binding(0)],
        });
        let line_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &line_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let line_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("line pipeline layout"),
            bind_group_layouts: &[Some(&line_layout)],
            immediate_size: 0,
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line pipeline"),
            layout: Some(&line_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let text_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("font texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let text_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("font atlas"),
            size: wgpu::Extent3d {
                width: text::TEXTURE_WIDTH,
                height: text::TEXTURE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let text_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("font sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let text_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("font atlas"),
            layout: &text_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &text_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&text_sampler),
                },
            ],
        });
        let text_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text pipeline layout"),
            bind_group_layouts: &[Some(&text_layout)],
            immediate_size: 0,
        });
        let text_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("text pipeline"),
            layout: Some(&text_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &text_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<TextVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &text_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let text_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text vertices"),
            size: (TEXT_VERTICES * size_of::<TextVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            surface,
            device,
            queue,
            config,
            depth_view,
            camera_buffer,
            lights_buffer,
            instances_buffer,
            instance_capacity,
            global_bind_group_layout,
            global_bind_group,
            mesh_pipeline,
            mesh_transparent_pipeline,
            line_pipeline,
            line_bind_group,
            lines_buffer,
            line_capacity,
            text_pipeline,
            text_bind_group,
            text_texture,
            text_vertex_buffer,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.reconfigure_surface();
        self.depth_view = create_depth_view(&self.device, width, height);
    }

    fn reconfigure_surface(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn mesh_slot(
        &self,
        geometry: &Geometry,
        material: Material,
        transform: Transform,
    ) -> MeshSlot {
        let data = geometry.build();
        let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh vertices"),
            size: (data.vertices.len() * size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&data.vertices));
        let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh indices"),
            size: (data.indices.len() * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&index_buffer, 0, bytemuck::cast_slice(&data.indices));
        MeshSlot {
            material,
            transform,
            vertex_buffer,
            index_buffer,
            index_count: data.indices.len() as u32,
        }
    }

    pub fn render(
        &mut self,
        meshes: &[MeshSlot],
        lines: &[LineVertex],
        camera: &Camera,
        lights: &Lights,
        hud: &str,
    ) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view_proj = camera.view_projection(aspect);
        let eye = camera.eye;
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_proj: view_proj.to_cols_array_2d(),
                eye: [eye.x, eye.y, eye.z, 1.0],
            }),
        );
        let light_color = lights.color.linear_channels();
        let ambient = lights.ambient.linear_channels();
        let direction = lights.direction.normalize();
        self.queue.write_buffer(
            &self.lights_buffer,
            0,
            bytemuck::bytes_of(&LightUniform {
                direction: [direction.x, direction.y, direction.z, 0.0],
                color: [
                    light_color[0] * lights.intensity,
                    light_color[1] * lights.intensity,
                    light_color[2] * lights.intensity,
                    1.0,
                ],
                ambient: [
                    ambient[0] * lights.ambient_intensity,
                    ambient[1] * lights.ambient_intensity,
                    ambient[2] * lights.ambient_intensity,
                    1.0,
                ],
            }),
        );

        let uniforms = build_instance_uniforms(meshes);
        self.ensure_instance_capacity(meshes.len());
        self.queue
            .write_buffer(&self.instances_buffer, 0, &uniforms.bytes);

        self.ensure_line_capacity(lines.len());
        if !lines.is_empty() {
            self.queue
                .write_buffer(&self.lines_buffer, 0, bytemuck::cast_slice(lines));
        }

        let raster = text::raster(hud);
        if raster.height > 0 {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.text_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &raster.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(text::TEXTURE_WIDTH),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: text::TEXTURE_WIDTH,
                    height: raster.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.reconfigure_surface();
                frame
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure_surface();
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame) => frame,
                    other => panic!("surface acquisition failed after reconfigure: {other:?}"),
                }
            }
            other => panic!("surface acquisition failed: {other:?}"),
        };
        let text_vertices = if raster.height > 0 {
            build_text_vertices(&raster, self.config.width, self.config.height)
        } else {
            Vec::new()
        };
        if !text_vertices.is_empty() {
            self.queue.write_buffer(
                &self.text_vertex_buffer,
                0,
                bytemuck::cast_slice(&text_vertices),
            );
        }
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.035,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !uniforms.opaque.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline);
                for index in &uniforms.opaque {
                    draw_mesh(&mut pass, &self.global_bind_group, &meshes[*index], *index);
                }
            }
            if !uniforms.transparent.is_empty() {
                pass.set_pipeline(&self.mesh_transparent_pipeline);
                for index in &uniforms.transparent {
                    draw_mesh(&mut pass, &self.global_bind_group, &meshes[*index], *index);
                }
            }
            if !lines.is_empty() {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.line_bind_group, &[]);
                pass.set_vertex_buffer(0, self.lines_buffer.slice(..));
                pass.draw(0..lines.len() as u32, 0..1);
            }
            if !text_vertices.is_empty() {
                pass.set_pipeline(&self.text_pipeline);
                pass.set_bind_group(0, &self.text_bind_group, &[]);
                pass.set_vertex_buffer(0, self.text_vertex_buffer.slice(..));
                pass.draw(0..text_vertices.len() as u32, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }

    fn ensure_instance_capacity(&mut self, count: usize) {
        if count <= self.instance_capacity {
            return;
        }
        self.instance_capacity = (self.instance_capacity * 2).max(count);
        let size = (self.instance_capacity * INSTANCE_STRIDE) as u64;
        self.instances_buffer = create_uniform_buffer(&self.device, "instances", size);
        self.recreate_global_bind_group();
    }

    fn ensure_line_capacity(&mut self, count: usize) {
        if count <= self.line_capacity {
            return;
        }
        self.line_capacity = (self.line_capacity * 2).max(count);
        let size = (self.line_capacity * size_of::<LineVertex>()) as u64;
        self.lines_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo lines"),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    fn recreate_global_bind_group(&mut self) {
        self.global_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera lights instances"),
            layout: &self.global_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.instances_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
    }
}

struct InstanceBatch {
    bytes: Vec<u8>,
    opaque: Vec<usize>,
    transparent: Vec<usize>,
}

fn build_instance_uniforms(meshes: &[MeshSlot]) -> InstanceBatch {
    let mut bytes = Vec::with_capacity(meshes.len() * INSTANCE_STRIDE);
    let mut opaque = Vec::new();
    let mut transparent = Vec::new();
    for (index, mesh) in meshes.iter().enumerate() {
        let model = mesh.transform.matrix();
        let normal = model.inverse().transpose();
        let base_color = mesh.material.base_color.linear_channels();
        let emissive = mesh.material.emissive.linear_channels();
        let uniform = InstanceUniform {
            model: model.to_cols_array_2d(),
            normal: normal.to_cols_array_2d(),
            base_color,
            emissive: [
                emissive[0],
                emissive[1],
                emissive[2],
                if mesh.material.unlit { 1.0 } else { 0.0 },
            ],
            params: [
                mesh.material.roughness.clamp(0.05, 1.0).powi(2),
                0.0,
                0.0,
                0.0,
            ],
        };
        bytes.extend_from_slice(bytemuck::bytes_of(&uniform));
        bytes.resize(
            bytes.len() + INSTANCE_STRIDE - size_of::<InstanceUniform>(),
            0,
        );
        if mesh.material.base_color.a < 1.0 {
            transparent.push(index);
        } else {
            opaque.push(index);
        }
    }
    InstanceBatch {
        bytes,
        opaque,
        transparent,
    }
}

fn draw_mesh(
    pass: &mut wgpu::RenderPass<'_>,
    bind_group: &wgpu::BindGroup,
    mesh: &MeshSlot,
    index: usize,
) {
    pass.set_bind_group(0, bind_group, &[(index * INSTANCE_STRIDE) as u32]);
    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

fn build_text_vertices(raster: &text::Raster, width: u32, height: u32) -> Vec<TextVertex> {
    let (origin_x, origin_y) = text::TEXT_ORIGIN;
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    let x0 = -1.0 + 2.0 * origin_x / width;
    let y0 = 1.0 - 2.0 * origin_y / height;
    let x1 = -1.0 + 2.0 * (origin_x + raster.width as f32) / width;
    let y1 = 1.0 - 2.0 * (origin_y + raster.height as f32) / height;
    let uv_x = raster.width as f32 / text::TEXTURE_WIDTH as f32;
    let uv_y = raster.height as f32 / text::TEXTURE_HEIGHT as f32;
    vec![
        TextVertex {
            position: [x0, y0, 0.0],
            uv: [0.0, 0.0],
        },
        TextVertex {
            position: [x1, y0, 0.0],
            uv: [uv_x, 0.0],
        },
        TextVertex {
            position: [x1, y1, 0.0],
            uv: [uv_x, uv_y],
        },
        TextVertex {
            position: [x0, y0, 0.0],
            uv: [0.0, 0.0],
        },
        TextVertex {
            position: [x1, y1, 0.0],
            uv: [uv_x, uv_y],
        },
        TextVertex {
            position: [x0, y1, 0.0],
            uv: [0.0, uv_y],
        },
    ]
}

fn uniform_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_mesh_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            })],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
