use wgpu::BufferUsages;

pub const EVENT_SLOTS: u32 = dynamis_gpu::Readback::DEPTH as u32 + 2;

pub const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
pub const UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);
pub const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

#[macro_export]
macro_rules! stream_usage {
    () => {
        $crate::STREAM
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
                $name:ident, $variant:ident: $label:literal, $stride:expr, $contents:expr, $slots:expr $(, $usage:expr)?;
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

            pub const fn whole(self) -> $crate::SlotRef {
                $crate::SlotRef::whole($crate::ResourceId::new(
                    $domain,
                    self as u32,
                ))
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

        impl From<$id> for $crate::ResourceId {
            fn from(id: $id) -> Self {
                $crate::ResourceId::new($domain, id as u32)
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub struct $demand {
            $( pub $field: $ty, )*
        }

        impl $table {
            pub fn new(
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
                            $crate::stream_usage!($($usage)?),
                            $contents,
                        ),
                    )*
                }
            }

            pub fn matches(&self, $locale: &$demand) -> bool {
                true $( && self.$name.slots() == $slots )*
            }

            pub fn reserve(
                &mut self,
                device: &::wgpu::Device,
                encoder: &mut ::wgpu::CommandEncoder,
                $locale: &$demand,
            ) -> bool {
                let mut changed = false;
                $( changed |= self.$name.reserve(device, encoder, $slots); )*
                changed
            }

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
