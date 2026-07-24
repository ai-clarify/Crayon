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
    StaleEpoch,
    CapacityExceeded(String),
    TaskNotFound(TaskId),
    TaskFailed(TaskId, String),
    TaskCancelled(TaskId),
    ObjectNotFound(ObjectId),
    ObjectPending(ObjectId),
    ObjectInUse(ObjectId),
    ObjectLost(ObjectId),
    ObjectConflict(ObjectId),
    DependencyFailed(ObjectId),
    DeadlineExceeded,
    Serialization(String),
    Io(String),
}
impl Error {
    /// Retry only what a retry can plausibly fix: transport faults, deadlines,
    /// fencing races, and inputs that are not ready *yet*. Everything else —
    /// lost objects (no lineage, gone forever), codec/serialization mismatches,
    /// bad arguments — is deterministic and retrying just burns attempts.
    pub fn failure_class(&self) -> crate::protocol::FailureClass {
        use crate::protocol::FailureClass;
        // Exhaustive on purpose: Permanent is the destructive direction (no
        // retry), so every new variant must pick its class here explicitly.
        match self {
            Error::Io(_)
            | Error::DeadlineExceeded
            | Error::StaleEpoch
            | Error::StaleFence
            | Error::ObjectNotFound(_)
            | Error::ObjectPending(_)
            | Error::ObjectInUse(_)
            | Error::CapacityExceeded(_) => FailureClass::Transient,
            Error::Protocol(_)
            | Error::InvalidAddress(_)
            | Error::InvalidResource(_)
            | Error::OperationUnavailable(_)
            | Error::OperationConflict(_)
            | Error::IllegalTransition(_)
            | Error::TaskNotFound(_)
            | Error::TaskFailed(_, _)
            | Error::TaskCancelled(_)
            | Error::ObjectLost(_)
            | Error::ObjectConflict(_)
            | Error::DependencyFailed(_)
            | Error::Serialization(_) => FailureClass::Permanent,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FailureClass;

    #[test]
    fn failure_class_maps_lost_and_deterministic_to_permanent() {
        assert_eq!(
            Error::ObjectLost(ObjectId::new()).failure_class(),
            FailureClass::Permanent
        );
        assert_eq!(
            Error::Serialization("bad".into()).failure_class(),
            FailureClass::Permanent
        );
        assert_eq!(
            Error::Protocol("bad arg".into()).failure_class(),
            FailureClass::Permanent
        );
    }

    #[test]
    fn failure_class_maps_infra_to_transient() {
        assert_eq!(
            Error::Io("reset".into()).failure_class(),
            FailureClass::Transient
        );
        assert_eq!(
            Error::DeadlineExceeded.failure_class(),
            FailureClass::Transient
        );
        assert_eq!(
            Error::ObjectNotFound(ObjectId::new()).failure_class(),
            FailureClass::Transient
        );
    }
}
