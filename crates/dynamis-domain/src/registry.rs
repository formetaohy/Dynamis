#[macro_export]
macro_rules! domains {
    (
        $( $field:ident: $domain:ty, )*
        [ $( $source:ident <-> $target:ident ),* $(,)? ]
    ) => {
        const _: () = {
            let mut seen: u64 = 0;
            $(
                let id = <$domain as $crate::Domain>::ID as u64;
                assert!(id < 64, "a domain id must fit the registry bitmap");
                assert!(seen & (1 << id) == 0, "two registered domains share one id");
                seen |= 1 << id;
            )*
            ::dynamis_pass::assert_declared(&[ $( <$domain as $crate::Domain>::PASS_EDGES, )* ]);
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

        impl HostWork {
            pub(crate) fn pending(&self) -> bool {
                false $( || <$domain as $crate::Domain>::pending(&self.$field) )*
            }
        }

        const DOMAIN_COUNT: usize = [ $( stringify!($field), )* ].len();

        const SIMULATES: [bool; DOMAIN_COUNT] = [ $( <$domain as $crate::Domain>::SIMULATES, )* ];

        #[derive(Clone, Copy)]
        struct Occupied {
            $( $field: bool, )*
        }

        impl Occupied {
            fn runs(self) -> [bool; DOMAIN_COUNT] {
                [ $( self.$field, )* ]
            }
        }

        #[derive(Clone, Copy)]
        pub(crate) struct Activity {
            $( pub(crate) $field: bool, )*
            occupied: Occupied,
        }

        impl Activity {
            pub(crate) const BUSY: Self = Self {
                $( $field: true, )*
                occupied: Occupied { $( $field: true, )* },
            };

            pub(crate) fn of(measured: &::dynamis_abi::Counters, live: &Live, work: &HostWork) -> Self {
                let occupied = Occupied {
                    $( $field: <$domain as $crate::Domain>::occupied(&live.$field), )*
                };
                let mut runs = [
                    $(
                        <$domain as $crate::Domain>::pending(&work.$field)
                            || <$domain as $crate::Domain>::active(measured),
                    )*
                ];
                for (running, occupied) in runs.iter_mut().zip(occupied.runs()) {
                    *running &= occupied;
                }
                let mut activity = Self { $( $field: false, )* occupied };
                activity.resume(runs);
                activity.close();
                activity
            }

            pub(crate) fn held(mut self, hold: bool) -> Self {
                if hold {
                    let occupied = self.occupied.runs();
                    let mut runs = self.runs();
                    for (running, (occupied, simulates)) in
                        runs.iter_mut().zip(occupied.into_iter().zip(SIMULATES))
                    {
                        *running |= occupied && simulates;
                    }
                    self.resume(runs);
                    self.close();
                }
                self
            }

            pub(crate) fn simulating(self) -> bool {
                self.runs()
                    .into_iter()
                    .zip(SIMULATES)
                    .any(|(running, simulates)| running && simulates)
            }

            pub(crate) fn busy(self) -> bool {
                self.runs().into_iter().any(|running| running)
            }

            fn close(&mut self) {
                while self.close_once() {}
            }

            fn close_once(&mut self) -> bool {
                let before = self.runs();
                $(
                    self.$target = self.$target || (self.$source && self.occupied.$target);
                )*
                self.runs() != before
            }

            fn runs(&self) -> [bool; DOMAIN_COUNT] {
                [ $( self.$field, )* ]
            }

            fn resume(&mut self, runs: [bool; DOMAIN_COUNT]) {
                let [ $( $field, )* ] = runs;
                $( self.$field = $field; )*
            }
        }

        #[derive(Clone, Copy)]
        pub(crate) struct StepFrames {
            $( pub(crate) $field: <$domain as $crate::Domain>::Frame, )*
            indexing: bool,
            awake: Awake,
        }

        #[derive(Clone, Copy)]
        struct Awake {
            $( $field: bool, )*
        }

        impl Awake {
            fn any(self) -> bool {
                false $( || self.$field )*
            }
        }

        impl StepFrames {
            pub(crate) fn of(facts: &$crate::StepFacts, live: &Live, liveness: Activity) -> Self {
                let awake = Awake { $( $field: liveness.$field, )* };
                Self {
                    $( $field: <$domain as $crate::Domain>::frame(facts, &live.$field), )*
                    indexing: awake.any(),
                    awake,
                }
            }
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
                use $crate::DomainStreams as _;
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
                use $crate::DomainStreams as _;
                true $( && self.$field.matches(&plan.$field) )*
            }

            pub(crate) fn reserve(
                &mut self,
                device: &wgpu::Device,
                encoder: &mut wgpu::CommandEncoder,
                plan: &Plan,
            ) -> bool {
                use $crate::DomainStreams as _;
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
                use $crate::DomainStreams as _;
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
                use $crate::DomainStreams as _;
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

        impl dynamis_gpu::Resources for Streams {
            fn generation(&self) -> u64 {
                self.generation
            }

            fn slots(&self, resource: dynamis_gpu::ResourceId) -> u32 {
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
                resource: dynamis_gpu::ResourceId,
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
                resource: dynamis_gpu::ResourceId,
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
                resources: &impl dynamis_gpu::Resources,
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
                resources: &impl dynamis_gpu::Resources,
                frames: &StepFrames,
            ) {
                $(
                    if pass.domain == <$domain as $crate::Domain>::ID {
                        let frame = &frames.$field;
                        let facts = dynamis_pass::Execution::facts(
                            frames.indexing,
                            frames.awake.$field,
                            <$domain as $crate::Domain>::gates(frame),
                        );
                        if pass.execution.held(facts) {
                            <$domain as $crate::Domain>::record(
                                &self.$field,
                                index,
                                schedule,
                                encoder,
                                resources,
                                frame,
                            );
                            assert!(
                                schedule.ran(index),
                                "pass {:?} of the {} domain declares {:?} yet records nothing",
                                pass.label,
                                stringify!($field),
                                pass.execution,
                            );
                        }
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
