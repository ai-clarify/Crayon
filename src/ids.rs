use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use uuid::Uuid;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize,
        )]
        pub struct $name(pub [u8; 16]);
        impl $name {
            pub fn new() -> Self {
                Self(*Uuid::new_v4().as_bytes())
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", Uuid::from_bytes(self.0).simple())
            }
        }
        impl FromStr for $name {
            type Err = uuid::Error;
            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Ok(Self(*Uuid::parse_str(value)?.as_bytes()))
            }
        }
    };
}

id_type!(ClusterId);
id_type!(CoordinatorEpoch);
id_type!(NodeId);
id_type!(WorkerEpoch);
id_type!(WorkerSessionId);
id_type!(TaskId);

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ObjectId(pub [u8; 32]);
impl ObjectId {
    pub fn new() -> Self {
        Self(*blake3::hash(Uuid::new_v4().as_bytes()).as_bytes())
    }
    pub const fn from_checksum(checksum: [u8; 32]) -> Self {
        Self(checksum)
    }
}
impl Default for ObjectId {
    fn default() -> Self {
        Self::new()
    }
}
impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}
impl FromStr for ObjectId {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 64 {
            return Err("object id must contain 64 hexadecimal characters".into());
        }
        let mut bytes = [0; 32];
        for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
            let text = std::str::from_utf8(chunk).map_err(|error| error.to_string())?;
            bytes[index] = u8::from_str_radix(text, 16).map_err(|error| error.to_string())?;
        }
        Ok(Self(bytes))
    }
}

id_type!(RequestId);
id_type!(LeaseId);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Revision(pub u64);
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Attempt(pub u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_id_is_the_exact_blake3_digest() {
        let bytes = b"crayon";
        let checksum = *blake3::hash(bytes).as_bytes();
        let id = ObjectId::from_checksum(checksum);
        assert_eq!(id.0, checksum);
        assert_eq!(id.to_string().parse::<ObjectId>().unwrap(), id);
    }
}
