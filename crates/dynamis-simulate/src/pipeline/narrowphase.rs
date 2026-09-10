use super::dispatch::{CCD_SWEEP, COMPACT_SCAN, COMPACT_SCATTER, NARROWPHASE, SORT_PAIRS};
use super::stage::{CORE, GEOMETRY, RO, RW, Stage, UNIFORM, shape_resources, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_CONTACTS, COUNTER_JOINTS, COUNTER_PAIRS};
use dynamis_sort::RadixSort;

pub(super) struct Narrowphase {
    narrowphase: Stage,
    compact_scan: Stage,
    compact_offsets: Stage,
    compact_scatter: Stage,
    ccd_sweep: Stage,
}

impl Narrowphase {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            narrowphase: Stage::build(
                context,
                "narrowphase",
                include_str!("../shaders/narrowphase.wgsl"),
                per_row,
                GEOMETRY,
                &[
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.bodies.colliders)),
                    (RO, whole(&buffers.contacts.pairs.major)),
                    (RO, whole(&buffers.contacts.pairs.minor)),
                    (RW, whole(&buffers.contacts.raw)),
                    (RW, whole(&buffers.contacts.valid)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                    (RO, whole(&buffers.constraints.joint_major)),
                    (RO, whole(&buffers.constraints.joint_minor)),
                    (RW, buffers.counter(COUNTER_JOINTS)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &shape_resources(buffers),
            ),
            compact_scan: Stage::build(
                context,
                "compact_scan",
                include_str!("../shaders/compact_scan.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.contacts.valid)),
                    (RW, whole(&buffers.contacts.compact_ranks)),
                    (RW, whole(&buffers.contacts.compact_sums)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_offsets: Stage::build(
                context,
                "compact_offsets",
                include_str!("../shaders/compact_offsets.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.contacts.compact_sums)),
                    (RW, whole(&buffers.contacts.compact_offsets)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_scatter: Stage::build(
                context,
                "compact_scatter",
                include_str!("../shaders/compact_scatter.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.contacts.raw)),
                    (RO, whole(&buffers.contacts.valid)),
                    (RO, whole(&buffers.contacts.compact_ranks)),
                    (RO, whole(&buffers.contacts.compact_offsets)),
                    (RW, whole(&buffers.contacts.manifolds)),
                    (RW, whole(&buffers.contacts.a_body)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            ccd_sweep: Stage::build(
                context,
                "ccd_sweep",
                include_str!("../shaders/ccd_sweep.wgsl"),
                per_row,
                GEOMETRY,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.bodies.colliders)),
                    (RO, whole(&buffers.contacts.pairs.major)),
                    (RO, whole(&buffers.contacts.pairs.minor)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                ],
                &shape_resources(buffers),
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        sort: &RadixSort,
    ) {
        let words = buffers.collider_words();
        let channels = buffers.sort_lanes_dual(
            buffers.counter(COUNTER_PAIRS),
            &buffers.contacts.pairs.major,
            &buffers.contacts.pairs.minor,
        );
        sort.sort(
            recorder,
            &channels,
            words,
            words,
            &buffers.dispatch,
            SORT_PAIRS,
        );
        self.ccd_sweep
            .record_indirect(recorder, &buffers.dispatch, CCD_SWEEP);
        self.narrowphase
            .record_indirect(recorder, &buffers.dispatch, NARROWPHASE);
        self.compact_scan
            .record_indirect(recorder, &buffers.dispatch, COMPACT_SCAN);
        self.compact_offsets.record_workgroups(recorder, 1);
        self.compact_scatter
            .record_indirect(recorder, &buffers.dispatch, COMPACT_SCATTER);
    }
}
