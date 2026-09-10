use super::Pipeline;
use wgpu::CommandEncoder;

impl Pipeline {
    pub(crate) fn capture_timings(
        &mut self,
        encoder: &mut CommandEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(encoder, sequence))
    }

    pub(crate) fn poll_timings(&mut self) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        match &mut self.timer {
            Some(timer) => timer.poll(),
            None => Vec::new(),
        }
    }

    pub(crate) fn arm_timings(&mut self) {
        if let Some(timer) = &mut self.timer {
            timer.arm();
        }
    }
}
