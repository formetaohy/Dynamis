use crate::buffer::GpuBuffer;
use crate::{BindingKind, BindingSpec, ComputePipeline, ComputeRecorder};
use wgpu::{BindGroup, BindGroupEntry, Device};

/// Converts a GPU-written element count into `dispatch_workgroups_indirect`
/// arguments for both the 64-thread and 256-thread workgroup sizes.
///
/// Every simulation stage launches workgroups from a runtime count rather
/// than from its pre-allocated capacity; without this conversion each
/// indirect dispatch would schedule one workgroup per element, idling
/// all but the first lane of every workgroup.
pub struct GpuCountArgs {
    pipeline: ComputePipeline,
    bindings: std::sync::Mutex<Option<(u64, u64, BindGroup)>>,
}

impl GpuCountArgs {
    pub fn new(device: &Device, label: &str) -> Self {
        let shader = "@group(0) @binding(0) var<storage, read> count: array<u32>;\n\
                      @group(0) @binding(1) var<storage, read_write> args: array<u32>;\n\n\
                      @compute @workgroup_size(1u)\n\
                      fn main() {\n\
                          let elements = count[0];\n\
                          args[0] = (elements + 63u) / 64u;\n\
                          args[1] = 1u;\n\
                          args[2] = 1u;\n\
                          args[3] = 0u;\n\
                          args[4] = (elements + 255u) / 256u;\n\
                          args[5] = 1u;\n\
                          args[6] = 1u;\n\
                          args[7] = 0u;\n\
                      }\n";
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
            pipeline: ComputePipeline::new(device, label, shader, "main", &[&specs[..]], 1),
            bindings: std::sync::Mutex::new(None),
        }
    }

    fn bindings(&self, device: &Device, count: &GpuBuffer, args: &GpuBuffer) -> BindGroup {
        let mut guard = self.bindings.lock().unwrap();
        let key = (count.token(), args.token());
        if guard.as_ref().is_none_or(|(count_key, args_key, _)| (*count_key, *args_key) != key) {
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
        guard.as_ref().expect("bindings ensured just above").2.clone()
    }

    pub fn encode(&self, device: &Device, recorder: &mut ComputeRecorder, count: &GpuBuffer, args: &GpuBuffer) {
        let group = self.bindings(device, count, args);
        recorder.record(&self.pipeline, &[&group], 1);
    }
}
