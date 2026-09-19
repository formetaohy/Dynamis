use super::World;
use super::device::Facts;
use crate::backend::StepFrames;
use crate::backend::registry::{HostWork, Live};
use crate::command::Consumption;
use dynamis_abi::DeclaredCounters;
use dynamis_abi::QueryHitRecord;
use dynamis_broadphase::BroadphaseWork;
use dynamis_domain::StepFacts;
use dynamis_gpu::SubmissionEncoder;
use dynamis_pass::Run;
use std::mem::size_of;

impl World {
    pub(crate) fn flush_queries(&mut self) {
        let Some(records) = self.backend.staged.queries.take() else {
            return;
        };
        self.backend
            .streams
            .scene
            .query_records
            .write(self.backend.gpu.queue(), bytemuck::cast_slice(&records));
    }
}

#[derive(Clone, Copy)]
struct Batch {
    id: u64,
    queries: u32,
    hits: u32,
}

struct Declarations {
    counters: Option<(u64, DeclaredCounters, Vec<u8>)>,
    queries: Option<(u64, Vec<u8>)>,
}

impl World {
    pub(crate) fn execute(&mut self, run: Run) {
        self.sync(Facts::Arrived);
        self.rebuild_joint_schedule();
        let census = self.census();
        let mut live = self.live(&census);
        self.apply_plan(&live);
        self.stage_observations();
        self.prepare(run);
        let rebuild = self.reconcile_derivations(&mut live);
        let work = self.host_work();
        if run == Run::Step {
            self.backend.published = work.pending();
            self.soft.uploaded = false;
        }
        let facts = StepFacts::of(
            &self.config,
            self.clock.sub_dt,
            census,
            self.row_streams(),
            self.wake_all,
        );
        self.stage_step_records(facts.params);
        let frames = self.frames_of(&live, &work, &facts, run);
        let device = self.backend.gpu.device().clone();
        let mut encoder = SubmissionEncoder::new(&device, run.label());
        let segments = self.copy_segments(&mut encoder);
        let batch = self.declare(run);
        self.flush_step_records();
        self.backend.streams.measure(&self.backend.measured);
        let indexing = frames.indexing();
        self.backend
            .passes
            .record(&mut encoder, &self.backend.streams, &frames, run);
        #[cfg(feature = "profile")]
        let timings = self.backend.passes.capture_timings(&mut encoder);
        let declarations = self.publish(&mut encoder, run, batch, &census, self.clock.step);
        self.submit(encoder);
        #[cfg(feature = "profile")]
        if let Some(timings) = timings {
            self.backend.pass_timings = timings;
        }
        self.consume_segments(segments);
        self.consume(declarations);
        self.finish(run, rebuild, indexing);
    }

    fn prepare(&mut self, run: Run) {
        match run {
            Run::Step => {
                self.flush_scene_records();
                self.apply_pending_commands(Consumption::Step);
            }
            Run::Query => {
                self.flush_scene_records();
                self.apply_pending_commands(Consumption::Preview);
            }
            Run::Publish => {}
        }
    }

    fn frames_of(
        &mut self,
        live: &Live,
        work: &HostWork,
        facts: &StepFacts,
        run: Run,
    ) -> StepFrames {
        match run {
            Run::Step => self.frames(live, work, facts),
            Run::Query => self.query_frames(live, facts),
            Run::Publish => self.publish_frames(live, facts),
        }
    }

    fn declare(&mut self, run: Run) -> Option<Batch> {
        match run {
            Run::Step | Run::Query => self.stage_queries(),
            Run::Publish => None,
        }
    }

    fn publish(
        &mut self,
        encoder: &mut SubmissionEncoder,
        run: Run,
        batch: Option<Batch>,
        census: &dynamis_abi::Census,
        step: u64,
    ) -> Declarations {
        match run {
            Run::Step => {
                self.declare_observations(encoder, census, step);
                Declarations {
                    counters: self.enqueue_counters(encoder, census, step),
                    queries: batch.and_then(|batch| self.enqueue_queries(encoder, batch)),
                }
            }
            Run::Query => Declarations {
                counters: None,
                queries: batch.and_then(|batch| self.enqueue_queries(encoder, batch)),
            },
            Run::Publish => {
                let step = self
                    .completed_step()
                    .expect("a publication requires a completed step");
                self.declare_observations(encoder, census, step);
                Declarations {
                    counters: None,
                    queries: None,
                }
            }
        }
    }

    fn stage_queries(&mut self) -> Option<Batch> {
        let width = self.queries.pending.len();
        if width == 0 {
            return None;
        }
        let id = self.queries.next_batch;
        let hits = self.queries.pending_hits;
        self.backend.staged.queries = Some(std::mem::take(&mut self.queries.pending));
        self.queries.pool.submit(id, self.clock.step, width);
        self.queries.next_batch += 1;
        self.queries.pending_hits = 0;
        Some(Batch {
            id,
            queries: width as u32,
            hits,
        })
    }

    fn enqueue_counters(
        &mut self,
        encoder: &mut SubmissionEncoder,
        census: &dynamis_abi::Census,
        step: u64,
    ) -> Option<(u64, DeclaredCounters, Vec<u8>)> {
        let bytes = self.pack_step(encoder);
        let declared = self.declared_counters(census, &self.row_streams());
        let regions = [(self.backend.readback.pack.buffer(), 0, bytes)];
        self.backend
            .readback
            .counters
            .declare(encoder, &regions, step, (step, declared))
            .map(|((step, declared), bytes)| (step, declared, bytes))
    }

    fn enqueue_queries(
        &mut self,
        encoder: &mut SubmissionEncoder,
        batch: Batch,
    ) -> Option<(u64, Vec<u8>)> {
        let records = &self.backend.streams.scene.query_records;
        let regions = [
            (records.buffer(), 0, batch.queries as u64 * records.stride()),
            (
                self.backend.streams.scene.query_hits.buffer(),
                0,
                batch.hits as u64 * size_of::<QueryHitRecord>() as u64,
            ),
        ];
        self.backend
            .readback
            .queries
            .declare(encoder, &regions, batch.id, batch.id)
    }

    fn consume(&mut self, declarations: Declarations) {
        if let Some((step, declared, bytes)) = declarations.counters {
            self.consume_pack(step, declared, &bytes);
        }
        if let Some((batch, bytes)) = declarations.queries {
            self.collect_query_batch(batch, &bytes);
        }
    }

    /// Records the derivations a run actually made, so the work the spatial index owed this step
    /// stops being owed. A run that indexes the scene derived every index it owed, so the facts at
    /// hand become the facts the device holds; a run that did not index leaves the difference
    /// standing for the next run that does.
    fn finish(&mut self, run: Run, rebuild: BroadphaseWork, indexing: bool) {
        if indexing {
            if rebuild.immovable {
                self.backend.immovable.derived();
            }
            if rebuild.resting {
                self.backend.resting.derived(self.clock.step);
            }
            self.backend.owed = BroadphaseWork::NONE;
        }
        match run {
            Run::Step => self.wake_all = false,
            Run::Query => {
                let census = self.census();
                let live = self.live(&census);
                self.apply_plan(&live);
            }
            Run::Publish => {}
        }
    }
}
