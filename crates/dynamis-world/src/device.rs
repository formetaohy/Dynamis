use super::World;
use bytemuck::Pod;
use dynamis_abi::StreamRecord;
use dynamis_gpu::Stream;
use wgpu::Buffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Facts {
    Arrived,
    Retired,
    Landed,
}

pub(crate) struct Region {
    element: &'static str,
    buffer: Buffer,
    stride: u64,
    offset: u32,
    count: u32,
}

impl Region {
    pub(crate) fn of(stream: &Stream, offset: u32, count: u32) -> Self {
        assert!(count > 0, "a device read must cover at least one record");
        let end = offset
            .checked_add(count)
            .expect("a device read must fit the device index space");
        assert!(
            end <= stream.slots(),
            "a device read of {count} records at {offset} must stay inside the {} slots of its stream",
            stream.slots()
        );
        Self {
            element: stream.element().wgsl(),
            buffer: stream.buffer().clone(),
            stride: stream.stride(),
            offset,
            count,
        }
    }

    fn at(&self) -> u64 {
        u64::from(self.offset) * self.stride
    }

    fn bytes(&self) -> u64 {
        u64::from(self.count) * self.stride
    }
}

impl World {
    pub(crate) fn sync(&mut self, level: Facts) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        if level == Facts::Arrived {
            return;
        }
        if level >= Facts::Landed {
            self.resolve_queries();
            self.publish_observations();
        }
        self.retire_device_facts();
        self.settle();
        if level >= Facts::Landed {
            self.mirror_body_states();
            assert!(
                self.states_current(),
                "a landing must land the state of every live body"
            );
        }
    }

    fn settle(&mut self) {
        let census = self.census();
        let live = self.live(&census);
        self.apply_plan(&live);
        self.flush_rows();
    }

    fn publish_observations(&mut self) {
        if self.observed.characters.needs_publication(self.clock.step)
            || self.observed.vehicles.needs_publication(self.clock.step)
        {
            self.execute(dynamis_pass::Run::Publish);
        }
    }

    pub(crate) fn read<T: Pod + StreamRecord>(&mut self, label: &str, region: Region) -> Vec<T> {
        assert!(
            T::WGSL == region.element,
            "{label:?} must decode the {} records its stream carries",
            region.element
        );
        let at = region.at();
        let bytes = region.bytes();
        let raw = self.read_regions(label, &[(region.buffer, at, bytes)]);
        dynamis_abi::decode::<T>(&raw)
    }

    pub(crate) fn read_one<T: Pod + StreamRecord>(&mut self, label: &str, region: Region) -> T {
        assert_eq!(region.count, 1, "{label:?} reads exactly one record");
        self.read::<T>(label, region)
            .pop()
            .expect("a one record read returns exactly one record")
    }

    pub(crate) fn read_regions(&mut self, label: &str, regions: &[(Buffer, u64, u64)]) -> Vec<u8> {
        let bytes: u64 = regions.iter().map(|region| region.2).sum();
        assert!(
            bytes > 0 && bytes.is_multiple_of(4),
            "a device read must cover a positive word aligned length"
        );
        let device = self.backend.gpu.device().clone();
        let mut readback = match self.backend.inspect.take() {
            Some(readback) if readback.size() >= bytes => readback,
            _ => dynamis_gpu::Readback::new(&device, "world inspection readback", bytes, 1),
        };
        let borrowed = regions
            .iter()
            .map(|(buffer, offset, bytes)| (buffer, *offset, *bytes))
            .collect::<Vec<_>>();
        let mut encoder = dynamis_gpu::SubmissionEncoder::new(&device, label);
        assert!(
            readback
                .enqueue_regions(&mut encoder, &borrowed, 0)
                .is_none(),
            "a device read requires an idle readback"
        );
        self.submit(encoder);
        let entry = readback
            .drain()
            .pop()
            .expect("a device read retires exactly once");
        self.backend.inspect = Some(readback);
        entry.1
    }
}
