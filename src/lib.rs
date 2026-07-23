//! Crayon is a small distributed task runtime with one authoritative coordinator.
//!
//! Cluster tasks invoke versioned registered operations. Crayon does not ship
//! Rust closures or Python callables across process boundaries.

pub mod client;
pub mod cluster;
pub mod coordinator;
pub mod data_plane;
pub mod arena;
pub mod error;
pub mod ids;
pub mod operation;
pub mod protocol;
pub mod resources;
pub mod worker;

pub use client::{ClusterClient, ObjectRef, TaskHandle};
pub use error::Error;
pub use operation::{Codec, Operation, OperationDescriptor, OperationKey, TaskArg};
pub use resources::{ResourceQuantity, ResourceSet};
