//! Plasma-style shared-memory object arena — Crayon's same-host fast path.
//!
//! One memory-mapped file holds every same-host payload; objects live at byte
//! offsets. A reader maps the arena **once** and reads any object as a memory
//! slice — no per-object `open`/`mmap`. A writer (the client) writes its bytes
//! straight into its reserved region, so a `put` never ships the payload over a
//! socket and is not bounded by the RPC frame size: objects scale to gigabytes.
//! This mirrors Ray's plasma store.
//!
//! Protocol: the client `reserve`s a slot (the coordinator allocates an offset
//! and records size/codec/checksum), writes its bytes into the mapping at that
//! offset, then `commit`s. A `get` returns the offset; the reader maps the arena
//! and reads the slice. Objects are immutable and content-addressed, so a
//! committed region is never rewritten while readable.
//!
//! Same-host is proven structurally: the arena file is host-local, so a process
//! that can map it is co-located. Cross-host peers cannot map it and fall back
//! to the network (bounded by the frame size, hence to <= one frame).

use std::{collections::HashMap, fs, io, path::PathBuf};

use memmap2::MmapMut;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{ids::ObjectId, operation::Codec};

/// Sparse virtual size of the arena. mmap reserves the range; pages are backed
/// only when written, so this is address space, not committed memory.
const ARENA_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// Where a reader finds an object: which arena, and at what offset. Size and
/// checksum travel in the surrounding `ObjectPayload`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaRef {
    pub token: String,
    pub offset: u64,
}

#[derive(Clone)]
struct Entry {
    offset: u64,
    size: u64,
    slot: u64,
    codec: Codec,
    checksum: [u8; 32],
    committed: bool,
}

/// Metadata a get reply needs for a committed arena object.
pub struct Meta {
    pub offset: u64,
    pub size: u64,
    pub codec: Codec,
    pub checksum: [u8; 32],
}

struct Alloc {
    top: u64,
    /// exact-size slot -> reusable offsets. Exact-fit avoids the up-to-2x waste
    /// of power-of-two classes on gigabyte payloads; a churn of same-size puts
    /// (the common case) recycles perfectly. Mixed sizes fragment — acceptable,
    /// compact later if it ever matters.
    free: HashMap<u64, Vec<u64>>,
    live: HashMap<ObjectId, Entry>,
}

/// Writer/owner side: the arena file, its read-write mapping, and the allocator.
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
        file.set_len(ARENA_BYTES)?;
        // SAFETY: freshly created file sized to ARENA_BYTES; this is the only
        // writer mapping and writes only disjoint, allocator-owned regions.
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

    /// Allocates a slot for `id` and records its metadata as uncommitted. The
    /// client then writes its bytes at the returned offset and calls `commit`.
    /// Returns `(offset, already_committed)`; content addressing makes a repeat
    /// reserve idempotent. `None` if the arena is exhausted.
    pub fn reserve(
        &self,
        id: ObjectId,
        size: u64,
        codec: Codec,
        checksum: [u8; 32],
    ) -> Option<(u64, bool)> {
        let slot = size.max(1).div_ceil(8) * 8; // 8-byte aligned exact size
        let mut a = self.alloc.lock();
        if let Some(entry) = a.live.get(&id) {
            return Some((entry.offset, entry.committed));
        }
        let offset = match a.free.get_mut(&slot).and_then(Vec::pop) {
            Some(off) => off,
            None => {
                let off = a.top;
                let next = off.checked_add(slot)?;
                if next > ARENA_BYTES {
                    return None;
                }
                a.top = next;
                off
            }
        };
        a.live.insert(
            id,
            Entry {
                offset,
                size,
                slot,
                codec,
                checksum,
                committed: false,
            },
        );
        Some((offset, false))
    }

    /// Marks a reserved object as fully written and readable.
    pub fn commit(&self, id: ObjectId) {
        if let Some(entry) = self.alloc.lock().live.get_mut(&id) {
            entry.committed = true;
        }
    }

    /// Coordinator-side write: reserve + copy `bytes` in + commit, for payloads
    /// the coordinator already holds (e.g. a same-host object arriving inline).
    pub fn put(&self, id: ObjectId, bytes: &[u8], codec: Codec, checksum: [u8; 32]) -> Option<u64> {
        let (offset, already) = self.reserve(id, bytes.len() as u64, codec, checksum)?;
        if !already {
            self.write_at(offset, bytes);
            self.commit(id);
        }
        Some(offset)
    }

    /// Copies `bytes` into an allocator-owned region. Callers must only write a
    /// range they reserved.
    ///
    /// SAFETY invariant: `offset..offset+len` is disjoint from every other live
    /// object, so concurrent writes never overlap; the base pointer is stable
    /// for the mapping's lifetime.
    pub fn write_at(&self, offset: u64, bytes: &[u8]) {
        unsafe {
            let dst = (self.map.as_ptr() as *mut u8).add(offset as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
    }

    /// Metadata for a committed object, for building a get reply.
    pub fn meta(&self, id: ObjectId) -> Option<Meta> {
        let a = self.alloc.lock();
        let e = a.live.get(&id)?;
        e.committed.then(|| Meta {
            offset: e.offset,
            size: e.size,
            codec: e.codec.clone(),
            checksum: e.checksum,
        })
    }

    /// Copies a committed object out, for serving a cross-host `GetLocal` (which
    /// cannot map the arena). Bounded by the RPC frame on the wire.
    pub fn read(&self, id: ObjectId) -> Option<Vec<u8>> {
        let (offset, size) = {
            let a = self.alloc.lock();
            let e = a.live.get(&id)?;
            if !e.committed {
                return None;
            }
            (e.offset as usize, e.size as usize)
        };
        // SAFETY: committed region, immutable until released; bounds from Entry.
        let slice =
            unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(offset), size) };
        Some(slice.to_vec())
    }

    /// Recycles an object's slot back into its exact-size free list.
    pub fn release(&self, id: ObjectId) {
        let mut a = self.alloc.lock();
        if let Some(entry) = a.live.remove(&id) {
            a.free.entry(entry.slot).or_default().push(entry.offset);
        }
    }
}

impl Drop for ArenaStore {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Reader side: map an arena by token, read-only. The caller caches the mapping
/// and reads objects as slices, so this happens once per arena, not per object.
/// `None` when the file is absent (a different host).
pub fn map_arena(token: &str) -> Option<memmap2::Mmap> {
    let file = fs::File::open(base_dir().join(format!("arena-{token}"))).ok()?;
    // SAFETY: committed regions are immutable; slots are recycled only after the
    // owner releases, so mapped pages stay valid for the mapping's lifetime.
    unsafe { memmap2::Mmap::map(&file).ok() }
}

/// Writer side for a client: map an arena read-write so the client writes its
/// reserved regions directly. `None` when the arena is not on this host.
pub fn map_arena_mut(token: &str) -> Option<MmapMut> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(base_dir().join(format!("arena-{token}")))
        .ok()?;
    // SAFETY: the client writes only offsets the coordinator reserved for it,
    // which are disjoint from every other writer's regions.
    unsafe { MmapMut::map_mut(&file).ok() }
}

/// Multithreaded memcpy: a gigabyte copy is memory-bandwidth work one core
/// can't saturate. Below 8MB the spawn cost beats the win, so copy plainly.
pub fn copy_wide(dst: &mut [u8], src: &[u8]) {
    const PAR_MIN: usize = 8 * 1024 * 1024;
    if src.len() < PAR_MIN {
        dst.copy_from_slice(src);
        return;
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    let chunk = src.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (d, s) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || d.copy_from_slice(s));
        }
    });
}

/// `src.to_vec()` with the copy parallelized (and the redundant zero-fill of a
/// `vec![0; n]` skipped) — the read half of the same bandwidth problem.
pub fn to_vec_wide(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    // SAFETY: `copy_wide` overwrites every byte before the Vec is used; u8 has
    // no validity invariant.
    unsafe { out.set_len(src.len()) };
    copy_wide(&mut out, src);
    out
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
    fn reserve_write_commit_get_and_recycle() {
        let store = ArenaStore::new().unwrap();
        let id = ObjectId::new();
        let payload = vec![9u8; 40_000];
        let checksum = crate::cluster::checksum(&payload);
        let (offset, already) = store
            .reserve(id, payload.len() as u64, Codec::RawBytes, checksum)
            .unwrap();
        assert!(!already);
        assert!(store.meta(id).is_none()); // not yet committed
        store.write_at(offset, &payload);
        store.commit(id);

        let meta = store.meta(id).expect("committed");
        assert_eq!(meta.offset, offset);
        assert_eq!(meta.size, payload.len() as u64);
        let map = map_arena(store.token()).unwrap();
        assert_eq!(&map[offset as usize..offset as usize + payload.len()], &payload[..]);
        assert_eq!(store.read(id).unwrap(), payload);

        store.release(id);
        assert!(store.meta(id).is_none());
        let id2 = ObjectId::new();
        // Same exact size recycles the same slot.
        assert_eq!(
            store
                .reserve(id2, payload.len() as u64, Codec::RawBytes, checksum)
                .unwrap()
                .0,
            offset
        );
        assert!(map_arena("deadbeef").is_none());
    }
}
