use wgpu::{BufferAddress, BufferUsages, CommandEncoder, Device, MapMode, PollType, Queue};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageId {
    ApplyCommands,
    JointFilter,
    Integrate,
    Broadphase,
    Narrowphase,
    ContactEvents,
    Islands,
    Gather,
    Solve,
    Position,
    Queries,
}

pub const STAGE_COUNT: usize = 11;

impl StageId {
    pub fn name(self) -> &'static str {
        match self {
            StageId::ApplyCommands => "apply_commands",
            StageId::JointFilter => "joint_filter",
            StageId::Integrate => "integrate",
            StageId::Broadphase => "broadphase",
            StageId::Narrowphase => "narrowphase",
            StageId::ContactEvents => "contact_events",
            StageId::Islands => "islands",
            StageId::Gather => "gather",
            StageId::Solve => "solve",
            StageId::Position => "position",
            StageId::Queries => "queries",
        }
    }

    pub const ALL: [StageId; STAGE_COUNT] = [
        StageId::ApplyCommands,
        StageId::JointFilter,
        StageId::Integrate,
        StageId::Broadphase,
        StageId::Narrowphase,
        StageId::ContactEvents,
        StageId::Islands,
        StageId::Gather,
        StageId::Solve,
        StageId::Position,
        StageId::Queries,
    ];
}

pub struct StageTimings {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback: [wgpu::Buffer; 2],
    samples: [f64; STAGE_COUNT],
    next: usize,
    period_ns: f64,
}

impl StageTimings {
    pub fn new(device: &Device, queue: &Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("dynamis stage timings"),
            ty: wgpu::QueryType::Timestamp,
            count: (STAGE_COUNT * 2) as u32,
        });
        let bytes = (STAGE_COUNT * 2 * 8) as BufferAddress;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamis stage timings resolve"),
            size: bytes,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = std::array::from_fn(|index| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dynamis stage timings readback {index}")),
                size: bytes,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        Self {
            query_set,
            resolve_buffer,
            readback,
            samples: [0.0; STAGE_COUNT],
            next: 0,
            period_ns: queue.get_timestamp_period() as f64,
        }
    }

    pub fn stage_begin(&self, encoder: &mut CommandEncoder, stage: StageId) {
        encoder.write_timestamp(&self.query_set, stage as u32 * 2);
    }

    pub fn stage_end(&self, encoder: &mut CommandEncoder, stage: StageId) {
        encoder.write_timestamp(&self.query_set, stage as u32 * 2 + 1);
    }

    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..(STAGE_COUNT * 2) as u32,
            &self.resolve_buffer,
            0,
        );
        let slot = self.next;
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback[slot],
            0,
            self.resolve_buffer.size(),
        );
        self.next = (self.next + 1) % 2;
    }

    pub fn poll(&mut self, device: &Device) -> Option<[f64; STAGE_COUNT]> {
        let slot = (self.next + 1) % 2;
        let slice = self.readback[slot].slice(..);
        slice.map_async(MapMode::Read, |_| {});
        let data = loop {
            match slice.get_mapped_range() {
                Ok(data) => break data,
                Err(_) => {
                    device.poll(PollType::wait_indefinitely()).unwrap();
                }
            }
        };
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        for stage in 0..STAGE_COUNT {
            if timestamps[stage * 2 + 1] > timestamps[stage * 2] {
                self.samples[stage] =
                    (timestamps[stage * 2 + 1] - timestamps[stage * 2]) as f64 * self.period_ns;
            }
        }
        drop(data);
        self.readback[slot].unmap();
        Some(self.samples)
    }
}
