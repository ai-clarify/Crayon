//! Plasma-style shared-memory object arena — Crayon's same-host fast path.
//!
//! One memory-mapped file holds every payload; objects live at byte offsets in
//! it. A reader maps the arena **once** and then reads any object as a plain
//! memory slice — there is no per-object `open`/`mmap`, which is what makes a
//! file-per-object sidecar slow (each get pays a syscall + fresh mapping). This
//! mirrors Ray's plasma store: map the arena, read at offsets, RAM speed.
//!
//! Co-location needs no handshake: the arena file is host-local, so a reader
//! that maps it is on the same host by construction; one that cannot falls back
//! to the network. Objects are immutable and content-addressed, so a written
//! region is never rewritten while readable.

use std::{collections::HashMap, fs, io, path::PathBuf};

use memmap2::{Mmap, MmapMut};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::ids::ObjectId;

/// Sparse virtual size of the arena. mmap reserves the address range but pages
/// are backed only when written, so this costs nothing until used.
const ARENA_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MIN_CLASS: u64 = 64;

/// Where a reader finds an object: which arena, and at what offset. Length and
/// checksum travel in the surrounding `ObjectPayload`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaRef {
    pub token: String,
    pub offset: u64,
}

struct Alloc {
    top: u64,
    /// size-class -> reusable offsets, so released slots are recycled instead of
    /// growing the arena for a put/get/release loop.
    free: HashMap<u64, Vec<u64>>,
    /// live object -> (offset, class), so release knows which slot to recycle.
    live: HashMap<ObjectId, (u64, u64)>,
}

/// Writer side: owns the arena file, the read-write mapping, and the allocator.
pub struct ArenaStore {
    token: String,
    map: MmapMut,
    alloc: Mutex<Alloc>,
    path: PathBuf,
}

impl ArenaStore {
    pub fn new() -> io::Result<Self> {
        let token = uuid::Uuid::new_v4().simple().to_string();
        let dir = base_dir();
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("arena-{token}"));
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.set_len(ARENA_BYTES)?; // sparse: no disk committed until written
        // SAFETY: freshly created file sized to ARENA_BYTES; we hold the only
        // writer mapping and only ever write disjoint, allocator-owned regions.
        let map = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self {
            token,
            map,
            alloc: Mutex::new(Alloc {
                top: 0,
                free: HashMap::new(),
                live: HashMap::new(),
            }),
            path,
        })
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    /// Copies `bytes` into the arena and records the object. Returns its offset,
    /// or `None` if the arena is exhausted (caller keeps the network path).
    pub fn put(&self, id: ObjectId, bytes: &[u8]) -> Option<u64> {
        let class = class_of(bytes.len());
        let offset = {
            let mut a = self.alloc.lock();
            if let Some((off, _)) = a.live.get(&id) {
                return Some(*off); // content-addressed: already present, idempotent
            }
            let off = match a.free.get_mut(&class).and_then(Vec::pop) {
                Some(off) => off,
                None => {
                    let off = a.top;
                    let next = off.checked_add(class)?;
                    if next > ARENA_BYTES {
                        return None;
                    }
                    a.top = next;
                    off
                }
            };
            a.live.insert(id, (off, class));
            off
        };
        // SAFETY: [offset, offset+len) is an allocator-owned region disjoint from
        // every other live object, so concurrent writes never overlap. The base
        // pointer is stable for the mapping's lifetime.
        unsafe {
            let dst = (self.map.as_ptr() as *mut u8).add(offset as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        Some(offset)
    }

    /// Offset of a live object, for building a get reply. `None` if not here.
    pub fn locate(&self, id: ObjectId) -> Option<u64> {
        self.alloc.lock().live.get(&id).map(|(off, _)| *off)
    }

    /// Recycles an object's slot back into its size-class free list.
    pub fn release(&self, id: ObjectId) {
        let mut a = self.alloc.lock();
        if let Some((off, class)) = a.live.remove(&id) {
            a.free.entry(class).or_default().push(off);
        }
    }
}

impl Drop for ArenaStore {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Reader side: map an arena by token. The caller caches the returned mapping
/// and reads objects as slices, so this open+map happens once per arena, not
/// once per object. `None` when the file is absent (a different host).
pub fn map_arena(token: &str) -> Option<Mmap> {
    let path = base_dir().join(format!("arena-{token}"));
    let file = fs::File::open(path).ok()?;
    // SAFETY: arena regions are written once (content-addressed, immutable) and
    // slots are only recycled after release, so mapped pages stay valid.
    unsafe { Mmap::map(&file).ok() }
}

fn class_of(len: usize) -> u64 {
    (len as u64).max(1).next_power_of_two().max(MIN_CLASS)
}

fn base_dir() -> PathBuf {
    let shm = PathBuf::from("/dev/shm");
    if shm.is_dir() {
        shm.join("crayon")
    } else {
        std::env::temp_dir().join("crayon")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_maps_once_and_recycles_slots() {
        let store = ArenaStore::new().unwrap();
        let id = ObjectId::new();
        let payload = vec![9u8; 4096];
        let off = store.put(id, &payload).unwrap();
        assert_eq!(store.locate(id), Some(off));

        let map = map_arena(store.token()).expect("same-host arena");
        assert_eq!(&map[off as usize..off as usize + payload.len()], &payload[..]);

        // Release recycles the exact slot for a same-class object.
        store.release(id);
        assert_eq!(store.locate(id), None);
        let id2 = ObjectId::new();
        assert_eq!(store.put(id2, &vec![1u8; 4096]).unwrap(), off);

        assert!(map_arena("deadbeef").is_none());
    }
}
