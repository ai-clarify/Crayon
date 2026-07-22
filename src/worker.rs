use std::{future::Future, sync::Arc};

use crate::{
    error::Error,
    operation::{OperationDescriptor, OperationKey},
    protocol::TaskAssignment,
};
use futures::future::BoxFuture;

pub type OperationFn =
    Arc<dyn Fn(Vec<Vec<u8>>) -> BoxFuture<'static, Result<Vec<u8>, Error>> + Send + Sync>;

#[derive(Default)]
pub struct OperationRegistry {
    entries: std::collections::HashMap<OperationKey, (OperationDescriptor, OperationFn)>,
}
impl OperationRegistry {
    pub fn register<F, Fut>(
        &mut self,
        descriptor: OperationDescriptor,
        operation: F,
    ) -> Result<(), Error>
    where
        F: Fn(Vec<Vec<u8>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<u8>, Error>> + Send + 'static,
    {
        descriptor.validate()?;
        if self.entries.contains_key(&descriptor.key) {
            return Err(Error::OperationConflict(descriptor.key.to_string()));
        }
        self.entries.insert(
            descriptor.key.clone(),
            (descriptor, Arc::new(move |args| Box::pin(operation(args)))),
        );
        Ok(())
    }
    pub fn descriptors(&self) -> Vec<OperationDescriptor> {
        self.entries.values().map(|(d, _)| d.clone()).collect()
    }
    pub async fn execute(
        &self,
        assignment: &TaskAssignment,
        inputs: Vec<Vec<u8>>,
    ) -> Result<Vec<u8>, Error> {
        let (_, op) = self
            .entries
            .get(&assignment.operation)
            .ok_or_else(|| Error::OperationUnavailable(assignment.operation.to_string()))?;
        op(inputs).await
    }
}
