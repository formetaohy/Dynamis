use crate::buffer::GpuBuffer;
use crate::{BindingKind, BindingSpec, ComputePipeline, ComputeRecorder, GpuContext};
use wgpu::{BindGroup, BindGroupEntry, Device};

pub struct GpuCountArgs {
    pipeline: ComputePipeline,
    bindings: std::sync::Mutex<Option<(u64, u64, BindGroup)>>,
}

impl GpuCountArgs {
    pub fn new(context: &GpuContext, label: &str) -> Self {
        let shader = include_str!("shaders/count_args.wgsl");
        let specs = [
            BindingSpec {
                binding: 0,
                kind: BindingKind::ReadOnlyStorage,
            },
            BindingSpec {
                binding: 1,
                kind: BindingKind::ReadWriteStorage,
            },
        ];
        Self {
            pipeline: context.compute_pipeline(label, shader, "main", &[&specs[..]], 1),
            bindings: std::sync::Mutex::new(None),
        }
    }

    fn bindings(&self, device: &Device, count: &GpuBuffer, args: &GpuBuffer) -> BindGroup {
        let mut guard = self.bindings.lock().unwrap();
        let key = (count.token(), args.token());
        if guard
            .as_ref()
            .is_none_or(|(count_key, args_key, _)| (*count_key, *args_key) != key)
        {
            let group = self.pipeline.create_bind_group(
                device,
                0,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: count.as_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: args.as_binding(),
                    },
                ],
            );
            *guard = Some((count.token(), args.token(), group));
        }
        guard
            .as_ref()
            .expect("bindings ensured just above")
            .2
            .clone()
    }

    pub fn encode(
        &self,
        device: &Device,
        recorder: &mut ComputeRecorder,
        count: &GpuBuffer,
        args: &GpuBuffer,
    ) {
        let group = self.bindings(device, count, args);
        recorder.record(&self.pipeline, &[&group], 1);
    }
}
