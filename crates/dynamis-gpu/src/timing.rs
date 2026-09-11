use crate::{ReadbackRing, SubmissionEncoder};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, ComputePassTimestampWrites, Device, QUERY_SIZE,
    QuerySet, QuerySetDescriptor, QueryType,
};

#[derive(Clone, Debug, PartialEq)]
pub struct GpuPassTiming {
    pub label: &'static str,
    pub nanoseconds: f64,
}

impl GpuPassTiming {
    pub fn milliseconds(&self) -> f64 {
        self.nanoseconds / 1e6
    }
}

pub struct GpuTimer {
    query_set: QuerySet,
    resolved: Buffer,
    readback: ReadbackRing,
    labels: &'static [&'static str],
    period_ns: f32,
}

impl GpuTimer {
    pub fn new(
        device: &Device,
        labels: &'static [&'static str],
        period_ns: f32,
        label_prefix: &str,
    ) -> Self {
        assert!(!labels.is_empty(), "a gpu timer needs at least one pass");
        assert!(
            period_ns > 0.0,
            "timestamp period must be strictly positive"
        );
        let resolved_bytes = Self::resolved_bytes(labels);
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some(&format!("{label_prefix} timestamps")),
            ty: QueryType::Timestamp,
            count: (labels.len() * 2) as u32,
        });
        let resolved = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{label_prefix} resolved timestamps")),
            size: resolved_bytes,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ReadbackRing::new(
            device,
            &format!("{label_prefix} timestamp readback"),
            resolved_bytes,
        );
        Self {
            query_set,
            resolved,
            readback,
            labels,
            period_ns,
        }
    }

    pub fn pass_count(&self) -> usize {
        self.labels.len()
    }

    fn resolved_bytes(labels: &[&'static str]) -> u64 {
        u64::from((labels.len() * 2) as u32 * QUERY_SIZE)
    }

    pub fn writes(&self, slot: usize) -> ComputePassTimestampWrites<'_> {
        assert!(
            slot < self.labels.len(),
            "timing slot {slot} exceeds the {n} recorded passes",
            n = self.labels.len()
        );
        ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some((slot * 2) as u32),
            end_of_pass_write_index: Some((slot * 2 + 1) as u32),
        }
    }

    pub fn capture(
        &mut self,
        encoder: &mut SubmissionEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<GpuPassTiming>)> {
        let count = (self.labels.len() * 2) as u32;
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolved, 0);
        let displaced =
            self.readback
                .enqueue(encoder, &self.resolved, 0, self.resolved.size(), sequence);
        displaced.map(|(frame, bytes)| (frame, self.decode(&bytes)))
    }

    pub fn collect(&mut self) -> Vec<(u64, Vec<GpuPassTiming>)> {
        self.readback
            .collect()
            .into_iter()
            .map(|(frame, bytes)| (frame, self.decode(&bytes)))
            .collect()
    }

    fn decode(&self, bytes: &[u8]) -> Vec<GpuPassTiming> {
        let mut ticks = Vec::with_capacity(self.labels.len() * 2);
        for chunk in bytes.chunks_exact(QUERY_SIZE as usize) {
            ticks.push(u64::from_le_bytes(
                chunk.try_into().expect("timestamp chunk is 8 bytes"),
            ));
        }
        assert_eq!(
            ticks.len(),
            self.labels.len() * 2,
            "resolved timestamp count disagrees with the recorded passes"
        );
        self.labels
            .iter()
            .enumerate()
            .map(|(slot, label)| GpuPassTiming {
                label,
                nanoseconds: ticks[slot * 2 + 1].wrapping_sub(ticks[slot * 2]) as f64
                    * self.period_ns as f64,
            })
            .collect()
    }
}
