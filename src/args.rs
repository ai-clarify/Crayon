//! Task argument resolution — Crayon's analog of Ray's automatic ObjectRef
//! dereferencing in `@ray.remote` function calls.
//!
//! When you pass an [`ObjectRef`] as a task argument, Ray transparently
//! fetches the object before the task runs. We model this with the
//! [`ResolveArg`] trait, implemented for [`ObjectRef<T>`]. Plain values
//! should be captured by the closure instead of passed as arguments.
//!
//! [`ResolveArgs`] is implemented for tuples up to 6 elements so `spawn` can
//! accept a typed argument list:
//!
//! ```ignore
//! let a = ray.put(1);
//! let b = ray.put(2);
//! let r = ray.spawn((a, b), |(a, b): (i32, i32)| a + b);
//! ```

use async_trait::async_trait;

use crate::common::{CrayonError, ObjectRef};
use crate::object_store::ObjectStore;

/// A value that can be resolved into a task argument.
#[async_trait]
pub trait ResolveArg: Send {
    type Output: Send;
    async fn resolve(self, store: &ObjectStore) -> Result<Self::Output, CrayonError>;
}

/// ObjectRefs are fetched from the store before the task runs.
#[async_trait]
impl<T: serde::de::DeserializeOwned + Send + 'static> ResolveArg for ObjectRef<T> {
    type Output = T;
    async fn resolve(self, store: &ObjectStore) -> Result<T, CrayonError> {
        store.get::<T>(self.id).await
    }
}

/// A tuple of [`ResolveArg`]s that resolves to a tuple of outputs.
#[async_trait]
pub trait ResolveArgs: Send {
    type Output: Send;
    async fn resolve(self, store: &ObjectStore) -> Result<Self::Output, CrayonError>;
}

macro_rules! impl_resolve_args {
    () => {
        #[async_trait]
        impl ResolveArgs for () {
            type Output = ();
            async fn resolve(self, _store: &ObjectStore) -> Result<(), CrayonError> {
                Ok(())
            }
        }
    };
    ($($T:ident),+) => {
        #[async_trait]
        #[allow(non_snake_case)]
        impl<$($T: ResolveArg),+> ResolveArgs for ($($T,)+) {
            type Output = ($($T::Output,)+);
            async fn resolve(self, store: &ObjectStore) -> Result<Self::Output, CrayonError> {
                let ($($T,)+) = self;
                let ($($T,)+) = (
                    $($T.resolve(store).await?,)+
                );
                Ok(($($T,)+))
            }
        }
    };
}

impl_resolve_args!();
impl_resolve_args!(A);
impl_resolve_args!(A, B);
impl_resolve_args!(A, B, C);
impl_resolve_args!(A, B, C, D);
impl_resolve_args!(A, B, C, D, E);
impl_resolve_args!(A, B, C, D, E, F);

/// A `Vec` of [`ResolveArg`]s resolves to a `Vec` of outputs, allowing a
/// dynamic number of task arguments (used by the Python bindings).
#[async_trait]
impl<A: ResolveArg> ResolveArgs for Vec<A> {
    type Output = Vec<A::Output>;
    async fn resolve(self, store: &ObjectStore) -> Result<Self::Output, CrayonError> {
        let mut out = Vec::with_capacity(self.len());
        for arg in self {
            out.push(arg.resolve(store).await?);
        }
        Ok(out)
    }
}
