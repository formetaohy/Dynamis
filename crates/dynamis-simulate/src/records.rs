//! Reinterprets a byte readback as records without assuming the byte buffer is aligned.

use bytemuck::{Pod, pod_read_unaligned};
use std::mem::size_of;

pub(crate) fn decode<T: Pod>(bytes: &[u8]) -> Vec<T> {
    let width = size_of::<T>();
    assert_eq!(
        bytes.len() % width,
        0,
        "readback is not a whole number of records"
    );
    bytes
        .chunks_exact(width)
        .map(|chunk| pod_read_unaligned(chunk))
        .collect()
}
