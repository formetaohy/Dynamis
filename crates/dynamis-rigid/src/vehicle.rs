use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_scene::SceneCast;
use dynamis_shader::rows;
use dynamis_state::StateStream;

pub struct Vehicle {
    step: Stage,
}

pub struct SweepVehicles {
    cast: SceneCast,
}

impl PassRuntime<RigidFrame> for Vehicle {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
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
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.step.record_rows(
            recorder,
            streams,
            Count::Vehicles.rows(&frame.params, &frame.rows),
        );
    }
}

impl PassRuntime<RigidFrame> for SweepVehicles {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
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
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        let sweeps = Count::VehicleWheels.rows(&frame.params, &frame.rows);
        self.cast.record(recorder, streams, sweeps);
    }
}
