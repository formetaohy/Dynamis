use bytemuck::{Pod, pod_read_unaligned};
use std::mem::size_of;

pub fn decode<T: Pod>(bytes: &[u8]) -> Vec<T> {
    let width = size_of::<T>();
    assert!(
        bytes.len().is_multiple_of(width),
        "readback is not a whole number of records"
    );
    bytes
        .chunks_exact(width)
        .map(|chunk| pod_read_unaligned(chunk))
        .collect()
}
