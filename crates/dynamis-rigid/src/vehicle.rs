use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::{Count, VEHICLE_WHEELS};
use dynamis_gpu::{ComputeRecorder, GpuContext, Resources};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_scene::SceneCast;
use dynamis_shader::rows;
use dynamis_state::StateStream;

pub struct Vehicle {
    step: Stage,
}

pub struct VehicleSweeps {
    cast: SceneCast,
}

impl PassRuntime<RigidFrame> for Vehicle {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            step: Stage::build(
                context,
                "vehicle",
                rows(
                    context,
                    include_str!("../shaders/vehicle.wgsl"),
                    &[],
                    Count::Vehicles.bound(),
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
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.step.record_rows(
            recorder,
            streams,
            Count::Vehicles.rows(&frame.params, &frame.rows),
        );
    }
}

impl PassRuntime<RigidFrame> for VehicleSweeps {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            cast: SceneCast::build(
                context,
                streams,
                RigidStream::VehicleSweeps.whole(),
                RigidStream::VehicleHits.whole(),
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        let sweeps = Count::Vehicles
            .rows(&frame.params, &frame.rows)
            .saturating_mul(VEHICLE_WHEELS);
        self.cast.record(recorder, streams, sweeps);
    }
}
