use std::{collections::HashMap, sync::Arc};

use crate::{cluster::checksum, error::Error, ids::ObjectId, operation::Codec};
use parking_lot::Mutex;

#[derive(Clone, Default)]
pub struct LocalObjectStore(Arc<Mutex<HashMap<ObjectId, LocalObject>>>);
#[derive(Clone)]
pub struct LocalObject {
    pub codec: Codec,
    pub bytes: Arc<[u8]>,
    pub checksum: [u8; 32],
}
impl LocalObjectStore {
    pub fn put(&self, id: ObjectId, codec: Codec, bytes: Vec<u8>) -> Result<LocalObject, Error> {
        let checksum = checksum(&bytes);
        let object = LocalObject {
            codec,
            bytes: bytes.into(),
            checksum,
        };
        let mut objects = self.0.lock();
        if let Some(existing) = objects.get(&id) {
            // Content is fenced by the blake3 checksum: equal checksum => equal
            // bytes (2^-256 collision), so no full-payload compare is needed.
            return if existing.checksum == object.checksum && existing.codec == object.codec {
                Ok(existing.clone())
            } else {
                Err(Error::ObjectConflict(id))
            };
        }
        objects.insert(id, object.clone());
        Ok(object)
    }
    pub fn get(&self, id: ObjectId) -> Result<LocalObject, Error> {
        self.0
            .lock()
            .get(&id)
            .cloned()
            .ok_or(Error::ObjectNotFound(id))
    }
    pub fn delete(&self, id: ObjectId) {
        self.0.lock().remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_id_is_fenced_against_bytes_and_codec() {
        let store = LocalObjectStore::default();
        let id = ObjectId::new();
        store.put(id, Codec::RawBytes, vec![1]).unwrap();
        assert!(matches!(
            store.put(id, Codec::RawBytes, vec![2]),
            Err(Error::ObjectConflict(_))
        ));
        assert!(matches!(
            store.put(id, Codec::JsonV1, vec![1]),
            Err(Error::ObjectConflict(_))
        ));
    }
}
