use crate::{error::Error, ids::ObjectId};
use serde::{Deserialize, Serialize};
use std::{fmt, marker::PhantomData};

pub const MAX_OPERATION_COMPONENT_BYTES: usize = 128;
pub const MAX_INLINE_ARG_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Codec {
    BincodeV1,
    PythonPickleV1,
    JsonV1,
    RawBytes,
}
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct OperationKey {
    pub namespace: String,
    pub name: String,
    pub version: u32,
}
impl OperationKey {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>, version: u32) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            version,
        }
    }
    pub fn validate(&self) -> Result<(), Error> {
        for (label, value) in [("namespace", &self.namespace), ("name", &self.name)] {
            if value.is_empty()
                || value.len() > MAX_OPERATION_COMPONENT_BYTES
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            {
                return Err(Error::Protocol(format!("invalid operation {label}")));
            }
        }
        if self.version == 0 {
            return Err(Error::Protocol("operation version must be positive".into()));
        }
        Ok(())
    }
}
impl fmt::Display for OperationKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}@{}", self.namespace, self.name, self.version)
    }
}
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct OperationDescriptor {
    pub key: OperationKey,
    pub input_codec: Codec,
    pub output_codec: Codec,
    pub max_inline_arg_bytes: u64,
}
impl OperationDescriptor {
    pub fn validate(&self) -> Result<(), Error> {
        self.key.validate()?;
        if self.max_inline_arg_bytes > MAX_INLINE_ARG_BYTES {
            return Err(Error::Protocol(format!(
                "max inline argument size exceeds {MAX_INLINE_ARG_BYTES} bytes"
            )));
        }
        Ok(())
    }

    pub fn validate_args(&self, args: &[TaskArg]) -> Result<(), Error> {
        for arg in args {
            if let TaskArg::Inline { codec, bytes } = arg {
                if codec != &self.input_codec {
                    return Err(Error::Protocol("inline argument codec mismatch".into()));
                }
                if bytes.len() as u64 > self.max_inline_arg_bytes {
                    return Err(Error::Protocol(
                        "inline argument exceeds operation limit".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum TaskArg {
    Inline { codec: Codec, bytes: Vec<u8> },
    Object(ObjectId),
}
#[derive(Debug, Clone)]
pub struct Operation<A, O> {
    descriptor: OperationDescriptor,
    marker: PhantomData<fn(A) -> O>,
}
impl<A, O> Operation<A, O> {
    pub fn new(descriptor: OperationDescriptor) -> Result<Self, Error> {
        descriptor.validate()?;
        Ok(Self {
            descriptor,
            marker: PhantomData,
        })
    }
    pub fn descriptor(&self) -> &OperationDescriptor {
        &self.descriptor
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> OperationDescriptor {
        OperationDescriptor {
            key: OperationKey::new("test", "op", 1),
            input_codec: Codec::BincodeV1,
            output_codec: Codec::RawBytes,
            max_inline_arg_bytes: 4,
        }
    }

    #[test]
    fn validates_exact_codec_and_inline_limit() {
        let descriptor = descriptor();
        assert!(descriptor
            .validate_args(&[TaskArg::Inline {
                codec: Codec::BincodeV1,
                bytes: vec![0; 4],
            }])
            .is_ok());
        assert!(descriptor
            .validate_args(&[TaskArg::Inline {
                codec: Codec::JsonV1,
                bytes: vec![0; 4],
            }])
            .is_err());
        assert!(descriptor
            .validate_args(&[TaskArg::Inline {
                codec: Codec::BincodeV1,
                bytes: vec![0; 5],
            }])
            .is_err());
    }

    #[test]
    fn rejects_invalid_descriptor() {
        let mut value = descriptor();
        value.key.version = 0;
        assert!(value.validate().is_err());
        value.key.version = 1;
        value.max_inline_arg_bytes = MAX_INLINE_ARG_BYTES + 1;
        assert!(value.validate().is_err());
    }
}
