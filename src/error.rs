use crate::ids::{ObjectId, TaskId};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum Error {
    Protocol(String),
    InvalidAddress(String),
    InvalidResource(String),
    OperationUnavailable(String),
    OperationConflict(String),
    IllegalTransition(String),
    StaleFence,
    TaskNotFound(TaskId),
    ObjectNotFound(ObjectId),
    ObjectLost(ObjectId),
    ObjectConflict(ObjectId),
    DependencyFailed(ObjectId),
    DeadlineExceeded(String),
    Serialization(String),
    Io(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
impl From<Box<bincode::ErrorKind>> for Error {
    fn from(value: Box<bincode::ErrorKind>) -> Self {
        Self::Serialization(value.to_string())
    }
}
