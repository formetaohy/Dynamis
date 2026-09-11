use std::ops::{Deref, DerefMut};
use std::sync::{Arc, OnceLock};
use wgpu::{CommandEncoder, CommandEncoderDescriptor, Device, Queue, SubmissionIndex};

pub struct SubmissionEncoder {
    encoder: CommandEncoder,
    submission: Arc<OnceLock<SubmissionIndex>>,
}

impl SubmissionEncoder {
    pub fn new(device: &Device, label: &str) -> Self {
        Self {
            encoder: device
                .create_command_encoder(&CommandEncoderDescriptor { label: Some(label) }),
            submission: Arc::new(OnceLock::new()),
        }
    }

    pub fn submit(self, queue: &Queue) -> SubmissionIndex {
        let index = queue.submit([self.encoder.finish()]);
        self.submission
            .set(index.clone())
            .expect("encoder submitted twice");
        index
    }

    pub(crate) fn submission(&self) -> Arc<OnceLock<SubmissionIndex>> {
        self.submission.clone()
    }
}

impl Deref for SubmissionEncoder {
    type Target = CommandEncoder;

    fn deref(&self) -> &Self::Target {
        &self.encoder
    }
}

impl DerefMut for SubmissionEncoder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.encoder
    }
}
