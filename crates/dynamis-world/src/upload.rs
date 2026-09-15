use dynamis_abi::ColliderRecord;
use dynamis_state::StateStreams;

use crate::World;

fn flush_pool_range(
    state: &StateStreams,
    queue: &wgpu::Queue,
    head: Option<(u32, u32)>,
    records: &[ColliderRecord],
    owners: &[u32],
) {
    let Some((start, end)) = head else {
        return;
    };
    debug_assert_eq!((end - start) as usize, records.len());
    if records.is_empty() {
        return;
    }
    state.colliders.write_at(
        queue,
        start as u64 * std::mem::size_of::<ColliderRecord>() as u64,
        bytemuck::cast_slice(records),
    );
    state
        .collider_owners
        .write_at(queue, start as u64 * 4, bytemuck::cast_slice(owners));
}

fn contiguous_runs(slots: &[u32]) -> Vec<&[u32]> {
    let mut runs = Vec::new();
    let mut start = 0usize;
    for index in 1..=slots.len() {
        let breaks = index == slots.len() || slots[index] != slots[index - 1] + 1;
        if breaks {
            if index > start {
                runs.push(&slots[start..index]);
            }
            start = index;
        }
    }
    runs
}

impl World {
    pub(crate) fn flush_rows(&mut self) {
        let queue = self.backend.gpu.queue().clone();
        if self.shapes.dirty {
            self.upload_shapes(&queue);
        }
        self.soft.upload(&queue, &self.backend.streams.soft);
        self.characters.upload(&queue, &self.backend.streams.rigid);
        self.bodies.dirty.sort_unstable();
        self.bodies.dirty.dedup();
        let dirty = std::mem::take(&mut self.bodies.dirty);
        for run in contiguous_runs(&dirty) {
            let descriptors = run
                .iter()
                .map(|slot| self.bodies.descriptors[self.bodies.alive[*slot as usize].id as usize])
                .collect::<Vec<_>>();
            self.backend.streams.state.body_descriptors.write_at(
                &queue,
                run[0] as u64 * self.backend.streams.state.body_descriptors.stride(),
                bytemuck::cast_slice(&descriptors),
            );
        }
        self.upload_colliders(&queue, &dirty);
        self.constraints.dirty.sort_unstable();
        self.constraints.dirty.dedup();
        for run in contiguous_runs(&std::mem::take(&mut self.constraints.dirty)) {
            let first = run[0];
            let records = run
                .iter()
                .map(|slot| self.constraints.records[*slot as usize])
                .collect::<Vec<_>>();
            self.backend.streams.state.constraint_descriptors.write_at(
                &queue,
                first as u64 * self.backend.streams.state.constraint_descriptors.stride(),
                bytemuck::cast_slice(&records),
            );
        }
    }

    fn upload_shapes(&mut self, queue: &wgpu::Queue) {
        let state = &self.backend.streams.state;
        self.shapes.pool.upload_pending(
            queue,
            &state.shape_sources,
            &state.shape_vertices,
            &state.shape_triangles,
            &state.shape_nodes,
        );
        self.shapes.dirty = false;
        self.shapes.uploaded = true;
    }

    fn upload_colliders(&mut self, queue: &wgpu::Queue, dirty: &[u32]) {
        for cleared in self.colliders.take_cleared() {
            let records = vec![ColliderRecord::cleared(); cleared.len as usize];
            let owners = vec![dynamis_abi::NO_BODY; cleared.len as usize];
            flush_pool_range(
                &self.backend.streams.state,
                queue,
                Some((cleared.offset, cleared.offset + cleared.len)),
                &records,
                &owners,
            );
        }
        let mut placed = Vec::with_capacity(dirty.len());
        for slot in dirty {
            let id = self.bodies.alive[*slot as usize].id;
            let run = self
                .colliders
                .run_of(id)
                .unwrap_or_else(|| panic!("row {slot} of body {id} has no collider run"));
            placed.push((run, *slot));
        }
        placed.sort_unstable_by_key(|(run, _)| run.offset);
        let mut records = Vec::new();
        let mut owners = Vec::new();
        let mut head: Option<(u32, u32)> = None;
        for (run, row) in placed {
            let contiguous = head.is_some_and(|(_, end)| end == run.offset);
            if !contiguous {
                flush_pool_range(&self.backend.streams.state, queue, head, &records, &owners);
                head = Some((run.offset, run.offset));
                records.clear();
                owners.clear();
            }
            let (_, end) = head.expect("a pool range is open");
            head = Some((head.expect("a pool range is open").0, end + run.len));
            let range = run.offset as usize..(run.offset + run.len) as usize;
            records.extend_from_slice(&self.colliders.records()[range]);
            owners.extend(std::iter::repeat_n(row, run.len as usize));
        }
        flush_pool_range(&self.backend.streams.state, queue, head, &records, &owners);
    }
}
