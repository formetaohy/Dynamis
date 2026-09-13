#[macro_export]
macro_rules! domains {
    (
        $( $field:ident: $domain:ty, )*
    ) => {
        const _: () = {
            let mut seen: u64 = 0;
            $(
                let id = <$domain as $crate::Domain>::ID as u64;
                assert!(id < 64, "a domain id must fit the registry bitmap");
                assert!(seen & (1 << id) == 0, "two registered domains share one id");
                seen |= 1 << id;
            )*
        };

        #[derive(Clone, Copy)]
        pub(crate) struct Plan {
            $( pub(crate) $field: <$domain as $crate::Domain>::Demand, )*
        }

        impl Plan {
            pub(crate) fn minimum() -> Self {
                Self { $( $field: <$domain as $crate::Domain>::minimum(), )* }
            }
        }

        #[derive(Clone, Copy)]
        pub(crate) struct Live {
            $( pub(crate) $field: <$domain as $crate::Domain>::Inputs, )*
        }

        #[derive(Clone, Copy)]
        pub(crate) struct HostWork {
            $( pub(crate) $field: <$domain as $crate::Domain>::Work, )*
        }

        #[derive(Clone, Copy)]
        pub(crate) struct StepFrames {
            $( pub(crate) $field: <$domain as $crate::Domain>::Frame, )*
        }

        #[derive(Default)]
        pub(crate) struct Planning {
            $( pub(crate) $field: <$domain as $crate::Domain>::Planner, )*
        }

        impl Planning {
            pub(crate) fn new() -> Self {
                Self::default()
            }
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct StreamCapacity {
            $( pub $field: <$domain as $crate::Domain>::Capacity, )*
        }

        pub(crate) struct Streams {
            $( pub(crate) $field: <$domain as $crate::Domain>::Streams, )*
            generation: u64,
        }

        impl Streams {
            pub(crate) fn new(
                device: &wgpu::Device,
                queue: &wgpu::Queue,
                plan: &Plan,
            ) -> Self {
                Self {
                    $(
                        $field: <$domain as $crate::Domain>::Streams::new(
                            device,
                            queue,
                            &plan.$field,
                        ),
                    )*
                    generation: 0,
                }
            }

            pub(crate) fn matches(&self, plan: &Plan) -> bool {
                true $( && self.$field.matches(&plan.$field) )*
            }

            pub(crate) fn reserve(
                &mut self,
                device: &wgpu::Device,
                encoder: &mut wgpu::CommandEncoder,
                plan: &Plan,
            ) -> bool {
                let mut changed = false;
                $( changed |= self.$field.reserve(device, encoder, &plan.$field); )*
                if changed {
                    self.generation = self
                        .generation
                        .checked_add(1)
                        .expect("a storage generation must not overflow");
                }
                changed
            }

            pub(crate) fn durable(&self) -> Vec<(&'static str, &dynamis_gpu::Stream)> {
                let mut streams = Vec::new();
                $( streams.extend(self.$field.durable()); )*
                streams
            }

            pub(crate) fn require<F: Fn(&'static str) -> Option<u32>>(
                &mut self,
                device: &wgpu::Device,
                encoder: &mut wgpu::CommandEncoder,
                floors: F,
            ) -> bool {
                let mut changed = false;
                $( changed |= self.$field.require(device, encoder, &floors); )*
                changed
            }

            pub(crate) fn capacity(&self) -> StreamCapacity {
                StreamCapacity {
                    $( $field: <$domain as $crate::Domain>::capacity(&self.$field), )*
                }
            }
        }

        impl dynamis_pass::Resources for Streams {
            fn generation(&self) -> u64 {
                self.generation
            }

            fn slots(&self, resource: dynamis_pass::ResourceId) -> u32 {
                $(
                    if resource.domain() == <$domain as $crate::Domain>::ID {
                        return self.$field.slots(resource.local());
                    }
                )*
                panic!(
                    "resource domain {} is outside the stream composition",
                    resource.domain(),
                )
            }

            fn whole(
                &self,
                resource: dynamis_pass::ResourceId,
            ) -> dynamis_gpu::GpuSlot<'_> {
                $(
                    if resource.domain() == <$domain as $crate::Domain>::ID {
                        return self.$field.whole(resource.local());
                    }
                )*
                panic!(
                    "resource domain {} is outside the stream composition",
                    resource.domain(),
                )
            }

            fn range(
                &self,
                resource: dynamis_pass::ResourceId,
                offset: u64,
                size: u64,
            ) -> dynamis_gpu::GpuSlot<'_> {
                $(
                    if resource.domain() == <$domain as $crate::Domain>::ID {
                        return self.$field.range(resource.local(), offset, size);
                    }
                )*
                panic!(
                    "resource domain {} is outside the stream composition",
                    resource.domain(),
                )
            }
        }

        pub(crate) struct PassIds {
            $( pub(crate) $field: <$domain as $crate::Domain>::Passes, )*
        }

        impl PassIds {
            pub(crate) fn declare(builder: &mut dynamis_pass::PipelineBuilder) {
                $(
                    for group in <$domain as $crate::Domain>::pass_groups() {
                        builder.declare(<$domain as $crate::Domain>::ID, *group);
                    }
                )*
            }

            pub(crate) fn resolve(pipeline: &dynamis_pass::Pipeline) -> Self {
                Self {
                    $( $field: <$domain as $crate::Domain>::resolve(pipeline), )*
                }
            }
        }

        pub(crate) struct PassRuntimes {
            $( pub(crate) $field: <$domain as $crate::Domain>::Runtime, )*
        }

        impl PassRuntimes {
            pub(crate) fn build(
                context: &dynamis_gpu::GpuContext,
                resources: &impl dynamis_pass::Resources,
                ids: PassIds,
            ) -> Self {
                let PassIds { $( $field, )* } = ids;
                Self {
                    $(
                        $field: <$domain as $crate::Domain>::build(
                            context,
                            resources,
                            $field,
                        ),
                    )*
                }
            }

            pub(crate) fn record(
                &self,
                pass: dynamis_pass::Pass,
                index: u32,
                schedule: &mut dynamis_pass::Schedule,
                encoder: &mut wgpu::CommandEncoder,
                resources: &impl dynamis_pass::Resources,
                frames: &StepFrames,
            ) {
                $(
                    if pass.domain == <$domain as $crate::Domain>::ID {
                        <$domain as $crate::Domain>::record(
                            &self.$field,
                            index,
                            schedule,
                            encoder,
                            resources,
                            &frames.$field,
                        );
                        return;
                    }
                )*
                panic!(
                    "pass {:?} names no registered domain",
                    pass.label,
                );
            }
        }
    };
}
