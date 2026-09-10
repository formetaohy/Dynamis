use dynamis_gpu::GpuBuffer;
use dynamis_sort::{SortChannels, key_words};

impl crate::buffers::WorldBuffers {
    pub(crate) fn body_words(&self) -> u32 {
        key_words(self.body_rows().max(1))
    }

    pub(crate) fn collider_words(&self) -> u32 {
        key_words(self.collider_rows().max(1))
    }

    pub(crate) fn sort_lanes<'a>(
        &'a self,
        count: dynamis_gpu::GpuSlot<'a>,
        major: &'a GpuBuffer,
        payload: &'a GpuBuffer,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major,
            minor: payload,
            payload,
            scratch_major: &self.sort.scratch.major,
            scratch_minor: &self.sort.scratch.payload,
            scratch_payload: &self.sort.scratch.payload,
        }
    }

    pub(crate) fn sort_keyed<'a>(
        &'a self,
        count: dynamis_gpu::GpuSlot<'a>,
        major: &'a GpuBuffer,
        minor: &'a GpuBuffer,
        payload: &'a GpuBuffer,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major,
            minor,
            payload,
            scratch_major: &self.sort.scratch.major,
            scratch_minor: &self.sort.scratch.minor,
            scratch_payload: &self.sort.scratch.payload,
        }
    }

    pub(crate) fn sort_lanes_dual<'a>(
        &'a self,
        count: dynamis_gpu::GpuSlot<'a>,
        major: &'a GpuBuffer,
        minor: &'a GpuBuffer,
    ) -> SortChannels<'a> {
        SortChannels {
            count,
            major,
            minor,
            payload: &self.sort.dummy.payload,
            scratch_major: &self.sort.scratch.major,
            scratch_minor: &self.sort.scratch.minor,
            scratch_payload: &self.sort.scratch.payload,
        }
    }
}
