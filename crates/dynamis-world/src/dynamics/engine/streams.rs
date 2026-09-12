macro_rules! stream_usage {
    () => {
        $crate::dynamics::engine::STREAM
    };
    ($usage:expr) => {
        $usage
    };
}

pub(crate) use stream_usage;

macro_rules! streams {
    (
        $(#[$meta:meta])*
        $table:ident, $id:ident, $demand:ident, $domain:expr, $locale:ident,
        demand { $( $field:ident: $ty:ty, )* }
        streams {
            $(
                $name:ident, $variant:ident: $label:literal, $stride:expr, $contents:expr, $slots:expr $(, $usage:expr)?;
            )*
        }
    ) => {
        $(#[$meta])*
        pub(crate) struct $table {
            $( pub(crate) $name: ::dynamis_gpu::Stream, )*
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(crate) enum $id {
            $( $variant, )*
        }

        impl $id {
            pub(crate) const ALL: &'static [Self] = &[ $( Self::$variant, )* ];

            pub(crate) const fn whole(self) -> $crate::dynamics::engine::SlotRef {
                $crate::dynamics::engine::SlotRef::whole($crate::dynamics::engine::ResourceId::new(
                    $domain,
                    self as u32,
                ))
            }

            pub(crate) fn of(local: u32) -> Self {
                *Self::ALL.get(local as usize).unwrap_or_else(|| {
                    panic!(
                        "stream {local} is outside the {} table",
                        stringify!($table)
                    )
                })
            }

            pub(crate) fn stream(self, table: &$table) -> &::dynamis_gpu::Stream {
                match self {
                    $( Self::$variant => &table.$name, )*
                }
            }
        }

        impl From<$id> for $crate::dynamics::engine::ResourceId {
            fn from(id: $id) -> Self {
                $crate::dynamics::engine::ResourceId::new($domain, id as u32)
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(crate) struct $demand {
            $( pub(crate) $field: $ty, )*
        }

        impl $table {
            pub(crate) fn new(
                device: &::wgpu::Device,
                queue: &::wgpu::Queue,
                $locale: &$demand,
            ) -> Self {
                Self {
                    $(
                        $name: ::dynamis_gpu::Stream::new(
                            device,
                            queue,
                            $label,
                            $slots,
                            $stride,
                            $crate::dynamics::engine::stream_usage!($($usage)?),
                            $contents,
                        ),
                    )*
                }
            }

            pub(crate) fn matches(&self, $locale: &$demand) -> bool {
                true $( && self.$name.slots() == $slots )*
            }

            pub(crate) fn reserve(
                &mut self,
                device: &::wgpu::Device,
                encoder: &mut ::wgpu::CommandEncoder,
                $locale: &$demand,
            ) -> bool {
                let mut changed = false;
                $( changed |= self.$name.reserve(device, encoder, $slots); )*
                changed
            }

            pub(crate) fn slots(&self, local: u32) -> u32 {
                $id::of(local).stream(self).slots()
            }

            pub(crate) fn whole(&self, local: u32) -> ::dynamis_gpu::GpuSlot<'_> {
                $id::of(local).stream(self).slot()
            }

            pub(crate) fn range(&self, local: u32, offset: u64, size: u64) -> ::dynamis_gpu::GpuSlot<'_> {
                ::dynamis_gpu::GpuSlot::range($id::of(local).stream(self).gpu(), offset, size)
            }
        }
    };
}

pub(crate) use streams;
