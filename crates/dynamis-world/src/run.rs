use super::World;
use crate::backend::registry::HostWork;
use crate::command::Consumption;
use dynamis_abi::DeclaredCounters;
use dynamis_abi::QueryHitRecord;
use dynamis_domain::StepFacts;
use dynamis_gpu::SubmissionEncoder;
use dynamis_pass::Run;
use std::mem::size_of;

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
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        let census = self.census();
        let live = self.live(&census);
        self.apply_plan(&live);
        self.flush_observed();
        let work = self.prepare(run);
        let facts = StepFacts::of(&self.config, self.clock.sub_dt, census, self.row_streams());
        self.write_step_records(facts.params);
        let frames = self.frames_of(&live, work.as_ref(), &facts, run);
        let device = self.backend.gpu.device().clone();
        let mut encoder = SubmissionEncoder::new(&device, run.label());
        let segments = self.copy_segments(&mut encoder);
        let batch = self.declare(run);
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
        self.finish(run);
    }

    fn prepare(&mut self, run: Run) -> Option<HostWork> {
        match run {
            Run::Step => {
                self.flush_rows();
                self.apply_pending_commands(Consumption::Step);
                let work = self.host_work();
                self.backend.published = work.pending();
                self.shapes.uploaded = false;
                self.soft.uploaded = false;
                Some(work)
            }
            Run::Query => {
                self.flush_rows();
                self.apply_pending_commands(Consumption::Preview);
                None
            }
            Run::Publish => None,
        }
    }

    fn frames_of(
        &mut self,
        live: &crate::backend::registry::Live,
        work: Option<&HostWork>,
        facts: &StepFacts,
        run: Run,
    ) -> crate::backend::StepFrames {
        match run {
            Run::Step => {
                let work = work.expect("a step frames its graph from the host work it prepared");
                self.frames(live, work, facts)
            }
            Run::Query => self.query_frames(live, facts),
            Run::Publish => self.publish_frames(live, facts),
        }
    }

    fn declare(&mut self, run: Run) -> Option<Batch> {
        match run {
            Run::Step | Run::Query => self.register_queries(),
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

    fn register_queries(&mut self) -> Option<Batch> {
        let width = self.queries.pending.len();
        if width == 0 {
            return None;
        }
        let id = self.queries.next_batch;
        let hits = self.queries.pending_hits;
        self.backend.streams.scene.query_records.write(
            self.backend.gpu.queue(),
            bytemuck::cast_slice(&self.queries.pending),
        );
        self.queries.pool.submit(id, self.clock.step, width);
        self.queries.next_batch += 1;
        self.queries.pending.clear();
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

    fn finish(&mut self, run: Run) {
        if run == Run::Query {
            let census = self.census();
            let live = self.live(&census);
            self.apply_plan(&live);
        }
    }
}
