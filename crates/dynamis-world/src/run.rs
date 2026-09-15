use super::World;
use crate::backend::registry::HostWork;
use crate::commands::Consumption;
use dynamis_abi::DeclaredCounters;
use dynamis_abi::QueryResultRecord;
use dynamis_gpu::SubmissionEncoder;
use dynamis_pass::Run;
use std::mem::size_of;

#[derive(Clone, Copy)]
struct Batch {
    id: u64,
    bytes: u64,
}

struct Declarations {
    counters: Option<(u64, DeclaredCounters, Vec<u8>)>,
    queries: Option<(u64, Vec<u8>)>,
}

impl World {
    pub(crate) fn execute(&mut self, run: Run) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
        let live = self.live();
        self.apply_plan(&live);
        self.flush_observed();
        let work = self.prepare(run);
        let params = self.step_params(self.clock.sub_dt);
        self.write_step_records(params);
        let frames = self.frames_of(&live, work.as_ref(), params, run);
        let device = self.backend.gpu.device().clone();
        let mut encoder = SubmissionEncoder::new(&device, run.label());
        let segments = self.copy_segments(&mut encoder);
        let batch = self.declare(run);
        self.backend
            .passes
            .record(&mut encoder, &self.backend.streams, &frames, run);
        #[cfg(feature = "profile")]
        let timings = self.backend.passes.capture_timings(&mut encoder);
        let declarations = self.publish(&mut encoder, run, batch, self.clock.step);
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
        params: dynamis_abi::StepParamsRecord,
        run: Run,
    ) -> crate::backend::StepFrames {
        match run {
            Run::Step => {
                let work = work.expect("a step frames its graph from the host work it prepared");
                self.frames(live, work, params)
            }
            Run::Query => self.query_frames(live, params),
            Run::Publish => self.publish_frames(live, params),
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
        step: u64,
    ) -> Declarations {
        match run {
            Run::Step => {
                self.declare_observations(encoder, step);
                Declarations {
                    counters: self.enqueue_counters(encoder, step),
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
                self.declare_observations(encoder, step);
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
        self.backend.streams.state.query_records.write(
            self.backend.gpu.queue(),
            bytemuck::cast_slice(&self.queries.pending),
        );
        self.queries.pool.submit(id, self.clock.step, width);
        self.queries.next_batch += 1;
        self.queries.pending.clear();
        Some(Batch {
            id,
            bytes: width as u64 * size_of::<QueryResultRecord>() as u64,
        })
    }

    fn enqueue_counters(
        &mut self,
        encoder: &mut SubmissionEncoder,
        step: u64,
    ) -> Option<(u64, DeclaredCounters, Vec<u8>)> {
        let bytes = self.pack_step(encoder);
        let declared = self.declared_counters();
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
        let regions = [(
            self.backend.streams.state.query_results.buffer(),
            0,
            batch.bytes,
        )];
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
            let live = self.live();
            self.apply_plan(&live);
        }
    }
}
