use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::{Count, VEHICLE_WHEELS};
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, GpuContext, Resources};
use dynamis_pass::Stage;
use dynamis_shader::{GEOMETRY_INDEX, rows, workgroups};
use dynamis_state::StateStream;

pub struct Vehicles {
    step: Stage,
    sweeps: Stage,
}

impl Vehicles {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            step: Stage::build(
                context,
                "vehicle",
                rows(
                    context,
                    include_str!("../shaders/vehicle.wgsl"),
                    &[],
                    Count::Vehicles.field(),
                ),
                streams,
                &[
                    ("vehicles", RigidStream::Vehicles.whole()),
                    ("vehicle_wheels", RigidStream::VehicleWheels.whole()),
                    ("vehicle_inputs", RigidStream::VehicleInputs.whole()),
                    ("vehicle_states", RigidStream::VehicleStates.whole()),
                    ("vehicle_sweeps", RigidStream::VehicleSweeps.whole()),
                    ("vehicle_hits", RigidStream::VehicleHits.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("params", StateStream::Params.whole()),
                ],
                &[],
            ),
            sweeps: Stage::build(
                context,
                "vehicle sweeps",
                workgroups(
                    context,
                    include_str!("../shaders/queries.wgsl"),
                    GEOMETRY_INDEX,
                ),
                streams,
                &[
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                    ("queries", RigidStream::VehicleSweeps.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("query_results", RigidStream::VehicleHits.whole()),
                    ("params", StateStream::Params.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
        }
    }

    pub fn record_step(
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.step
            .record_rows(recorder, streams, Count::Vehicles.rows(&frame.params));
    }

    pub fn record_sweeps(
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        let sweeps = Count::Vehicles
            .rows(&frame.params)
            .saturating_mul(VEHICLE_WHEELS);
        self.sweeps.record_workgroups(recorder, streams, sweeps);
    }
}
