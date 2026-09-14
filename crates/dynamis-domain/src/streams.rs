use dynamis_gpu::Stream;
use wgpu::{CommandEncoder, Device, Queue};

pub trait DomainStreams: Sized {
    type Demand;

    fn new(device: &Device, queue: &Queue, demand: &Self::Demand) -> Self;

    fn matches(&self, demand: &Self::Demand) -> bool;

    fn reserve(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        demand: &Self::Demand,
    ) -> bool;

    fn require<F: Fn(&'static str) -> Option<u32>>(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        floors: F,
    ) -> bool;

    fn durable(&self) -> Vec<(&'static str, &Stream)>;
}

#[macro_export]
macro_rules! stream_usage {
    () => {
        ::dynamis_gpu::STREAM
    };
    ($usage:expr) => {
        $usage
    };
}

#[macro_export]
macro_rules! streams {
    (
        $(#[$meta:meta])*
        $table:ident, $id:ident, $demand:ident, $domain:expr, $locale:ident,
        demand { $( $field:ident: $ty:ty, )* }
        streams {
            $(
                $name:ident, $variant:ident: $label:literal, $element:ty, $per_slot:expr, $contents:expr, $slots:expr $(, $usage:expr)?;
            )*
        }
    ) => {
        $(#[$meta])*
        pub struct $table {
            $( pub $name: ::dynamis_gpu::Stream, )*
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub enum $id {
            $( $variant, )*
        }

        impl $id {
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )* ];

            pub const CONTENTS: &'static [::dynamis_gpu::Contents] = &[ $( $contents ),* ];

            pub const fn contents(self) -> ::dynamis_gpu::Contents {
                Self::CONTENTS[self as usize]
            }

            pub const fn label(self) -> &'static str {
                match self {
                    $( Self::$variant => $label, )*
                }
            }

            pub const fn element(self) -> ::dynamis_gpu::StreamElement {
                match self {
                    $(
                        Self::$variant => ::dynamis_gpu::StreamElement::new(
                            <$element as ::dynamis_abi::StreamRecord>::WGSL,
                            ::core::mem::size_of::<$element>() as u64,
                        ),
                    )*
                }
            }

            pub const fn durable(self) -> bool {
                self.contents().durable()
            }

            pub const fn whole(self) -> ::dynamis_gpu::SlotRef {
                ::dynamis_gpu::SlotRef::whole(
                    ::dynamis_gpu::ResourceId::new($domain, self as u32),
                    self.element(),
                )
            }

            pub fn of(local: u32) -> Self {
                *Self::ALL.get(local as usize).unwrap_or_else(|| {
                    panic!(
                        "stream {local} is outside the {} table",
                        stringify!($table)
                    )
                })
            }

            pub fn stream(self, table: &$table) -> &::dynamis_gpu::Stream {
                match self {
                    $( Self::$variant => &table.$name, )*
                }
            }
        }

        impl From<$id> for ::dynamis_gpu::ResourceId {
            fn from(id: $id) -> Self {
                ::dynamis_gpu::ResourceId::new($domain, id as u32)
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub struct $demand {
            $( pub $field: $ty, )*
        }

        impl $crate::DomainStreams for $table {
            type Demand = $demand;

            fn new(device: &::wgpu::Device, queue: &::wgpu::Queue, $locale: &$demand) -> Self {
                Self {
                    $(
                        $name: ::dynamis_gpu::Stream::new(
                            device,
                            queue,
                            ::dynamis_gpu::StreamDesc {
                                label: $label,
                                slots: $slots,
                                element: $id::$variant.element(),
                                elements_per_slot: $per_slot as u64,
                                usage: $crate::stream_usage!($($usage)?),
                                contents: $contents,
                            },
                        ),
                    )*
                }
            }

            fn matches(&self, $locale: &$demand) -> bool {
                true $( && self.$name.slots() == $slots )*
            }

            fn reserve(
                &mut self,
                device: &::wgpu::Device,
                encoder: &mut ::wgpu::CommandEncoder,
                $locale: &$demand,
            ) -> bool {
                let mut changed = false;
                $( changed |= self.$name.reserve(device, encoder, $slots); )*
                changed
            }

            fn require<F: Fn(&'static str) -> Option<u32>>(
                &mut self,
                device: &::wgpu::Device,
                encoder: &mut ::wgpu::CommandEncoder,
                floors: F,
            ) -> bool {
                let mut changed = false;
                $(
                    if $contents.durable() {
                        let slots = floors($label).unwrap_or_else(|| {
                            panic!("a snapshot must answer the durable stream {:?}", $label)
                        });
                        assert!(
                            slots >= self.$name.slots(),
                            "durable stream {:?} cannot be restored below its snapshot size",
                            $label,
                        );
                        changed |= self.$name.reserve(device, encoder, slots);
                    }
                )*
                changed
            }

            fn durable(&self) -> Vec<(&'static str, &::dynamis_gpu::Stream)> {
                $id::ALL
                    .iter()
                    .copied()
                    .filter(|id| id.durable())
                    .map(|id| (id.label(), id.stream(self)))
                    .collect()
            }
        }

        impl $table {
            pub fn slots(&self, local: u32) -> u32 {
                $id::of(local).stream(self).slots()
            }

            pub fn whole(&self, local: u32) -> ::dynamis_gpu::GpuSlot<'_> {
                $id::of(local).stream(self).slot()
            }

            pub fn range(&self, local: u32, offset: u64, size: u64) -> ::dynamis_gpu::GpuSlot<'_> {
                ::dynamis_gpu::GpuSlot::range($id::of(local).stream(self).gpu(), offset, size)
            }
        }
    };
}
