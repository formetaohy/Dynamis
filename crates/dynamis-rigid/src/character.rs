use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::{CHARACTER_SWEEPS, Count};
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_scene::SceneCast;
use dynamis_shader::rows;
use dynamis_state::StateStream;

pub struct Character {
    step: Stage,
}

pub struct SweepCharacters {
    cast: SceneCast,
}

impl PassRuntime<RigidFrame> for Character {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            step: Stage::build(
                context,
                "character",
                rows(
                    context,
                    include_str!("../shaders/character.wgsl"),
                    &[],
                    Count::Characters.bound(),
                ),
                streams,
                &[
                    ("characters", RigidStream::Characters.whole()),
                    ("character_inputs", RigidStream::CharacterInputs.whole()),
                    ("character_states", RigidStream::CharacterStates.whole()),
                    ("character_sweeps", RigidStream::CharacterSweeps.whole()),
                    ("character_hits", RigidStream::CharacterHits.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
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
            Count::Characters.rows(&frame.params, &frame.rows),
        );
    }
}

impl PassRuntime<RigidFrame> for SweepCharacters {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            cast: SceneCast::build(
                context,
                streams,
                RigidStream::CharacterSweeps.whole(),
                RigidStream::CharacterHits.whole(),
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        let sweeps = Count::Characters
            .rows(&frame.params, &frame.rows)
            .saturating_mul(CHARACTER_SWEEPS);
        self.cast.record(recorder, streams, sweeps);
    }
}
