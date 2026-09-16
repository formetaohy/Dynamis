use dynamis_abi::{NO_SURFACE, NO_TRIANGLE, QueryHitRecord, QueryRecord};
use dynamis_model::{BodyHandle, SceneTarget, SoftBodyHandle, SurfaceDesc};
use dynamis_scene::{scene_slot, scene_target};
use std::collections::VecDeque;
use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryHandle {
    pub batch: u64,
    pub index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryState {
    Pending,
    Retired,
    Lapsed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHit {
    pub target: SceneTarget,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub triangle: Option<u32>,
    pub surface: Option<SurfaceDesc>,
    pub step: u64,
}

impl QueryHit {
    pub fn body(&self) -> BodyHandle {
        self.target.collider().map_or_else(
            || panic!("a query hit against a soft particle carries no rigid body"),
            |(body, _)| body,
        )
    }

    pub fn collider(&self) -> u32 {
        self.target.collider().map_or_else(
            || panic!("a query hit against a soft particle carries no collider"),
            |(_, collider)| collider,
        )
    }

    pub fn triangle(&self) -> Option<u32> {
        assert!(
            self.target.collider().is_some(),
            "a query hit against a soft particle carries no triangle"
        );
        self.triangle
    }

    pub fn surface(&self) -> Option<SurfaceDesc> {
        assert!(
            self.target.collider().is_some(),
            "a query hit against a soft particle carries no surface"
        );
        self.surface
    }

    pub fn soft(&self) -> SoftBodyHandle {
        self.target.particle().map_or_else(
            || panic!("a query hit against a rigid collider carries no soft body"),
            |(body, _)| body,
        )
    }

    pub fn particle(&self) -> u32 {
        self.target.particle().map_or_else(
            || panic!("a query hit against a rigid collider carries no particle"),
            |(_, particle)| particle,
        )
    }
}

pub(crate) struct QueryOutcome {
    pub hits: Vec<Vec<QueryHit>>,
    pub overflow: Vec<bool>,
}

struct QueryBatch {
    batch: u64,
    step: u64,
    width: usize,
    outcome: Option<QueryOutcome>,
}

pub(crate) struct QueryPool {
    batches: VecDeque<QueryBatch>,
}

impl QueryPool {
    pub(crate) fn new() -> Self {
        Self {
            batches: VecDeque::new(),
        }
    }

    pub(crate) fn submit(&mut self, batch: u64, step: u64, width: usize) {
        self.batches.push_back(QueryBatch {
            batch,
            step,
            width,
            outcome: None,
        });
    }

    pub(crate) fn hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        self.hits(handle).first().copied()
    }

    pub(crate) fn hits(&self, handle: QueryHandle) -> &[QueryHit] {
        &self.outcome(handle).hits[handle.index as usize]
    }

    pub(crate) fn overflow(&self, handle: QueryHandle) -> bool {
        self.outcome(handle).overflow[handle.index as usize]
    }

    fn outcome(&self, handle: QueryHandle) -> &QueryOutcome {
        self.batch(handle)
            .and_then(|batch| batch.outcome.as_ref())
            .expect("query outcome read before it arrived")
    }

    pub(crate) fn is_current(&self, handle: QueryHandle) -> bool {
        self.batch(handle).is_some()
    }

    pub(crate) fn is_ready(&self, handle: QueryHandle) -> bool {
        self.batch(handle)
            .is_some_and(|batch| batch.outcome.is_some())
    }

    fn batch(&self, handle: QueryHandle) -> Option<&QueryBatch> {
        self.batches
            .iter()
            .find(|batch| batch.batch == handle.batch)
    }

    pub(crate) fn collect(
        &mut self,
        batch_id: u64,
        bytes: &[u8],
        surface: impl Fn(u32, u32) -> SurfaceDesc,
        collider_of: impl Fn(BodyHandle, u32) -> u32,
        particle_of: impl Fn(SoftBodyHandle, u32) -> u32,
    ) {
        let batch = self
            .batches
            .iter_mut()
            .find(|batch| batch.batch == batch_id)
            .unwrap_or_else(|| panic!("no query batch is registered for batch {batch_id}"));
        let step = batch.step;
        let width = batch.width;
        let record_bytes = width * size_of::<QueryRecord>();
        let records = dynamis_abi::decode::<QueryRecord>(&bytes[..record_bytes]);
        let hits = dynamis_abi::decode::<QueryHitRecord>(&bytes[record_bytes..]);
        let mut span = vec![Vec::new(); width];
        let mut overflow = vec![false; width];
        let mut base = 0usize;
        for (index, record) in records.iter().enumerate() {
            let count = record.count as usize;
            assert!(
                count <= record.hit_bound() as usize,
                "a query outcome exceeds the hit span it declared"
            );
            span[index] = hits[record.hit_base as usize..record.hit_base as usize + count]
                .iter()
                .map(|record| QueryHit {
                    target: scene_target(
                        record.scene_target,
                        record.body_id,
                        record.body_generation,
                        &collider_of,
                        &particle_of,
                    ),
                    distance: record.distance,
                    point: record.point,
                    normal: record.normal,
                    triangle: (record.triangle != NO_TRIANGLE).then_some(record.triangle),
                    surface: (record.surface != NO_SURFACE)
                        .then(|| surface(scene_slot(record.scene_target), record.surface)),
                    step,
                })
                .collect();
            overflow[index] = record.overflow != 0;
            base += record.hit_bound() as usize;
        }
        assert_eq!(
            base,
            hits.len(),
            "a query batch must cover every declared hit span"
        );
        batch.outcome = Some(QueryOutcome {
            hits: span,
            overflow,
        });
        while self.batches.len() > dynamis_gpu::FACT_LAG {
            self.batches.pop_front();
        }
    }
}
