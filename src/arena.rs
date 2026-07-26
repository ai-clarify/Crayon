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

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use memmap2::MmapMut;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{ids::ObjectId, operation::Codec};

/// Sparse virtual size of the arena. mmap reserves the range; pages are backed
/// only when written, so this is address space, not committed memory.
const ARENA_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// A reservation the client never committed (crash / lost RPC between
/// `reserve` and `commit`) is reaped after this long, so its slot cannot leak
/// forever and wedge the large-object put path. Well above any real write.
const RESERVE_TTL: Duration = Duration::from_secs(60);

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
    /// When the slot was reserved; an uncommitted entry past `RESERVE_TTL` is a
    /// crashed writer and gets reaped. `None` once committed (never expires).
    reserved_at: Option<Instant>,
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

impl Alloc {
    /// Returns crashed uncommitted reservations to the free list. Driven lazily
    /// from `reserve`; no background thread. O(live), bounded by in-flight puts.
    fn reap(&mut self, now: Instant) {
        let expired: Vec<ObjectId> = self
            .live
            .iter()
            .filter(|(_, e)| {
                e.reserved_at
                    .is_some_and(|t| now.duration_since(t) >= RESERVE_TTL)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            if let Some(e) = self.live.remove(&id) {
                self.free.entry(e.slot).or_default().push(e.offset);
            }
        }
    }
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
        let dir = base_dir();
        fs::create_dir_all(&dir)?;
        sweep_dead(&dir);
        // Tag the token with our pid so `sweep_dead` can reap arenas orphaned by a
        // SIGKILLed coordinator (which skips `Drop`). The token stays opaque to
        // readers, which only ever map `arena-{token}`.
        let token = format!("{}-{}", std::process::id(), uuid::Uuid::new_v4().simple());
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

    /// Filesystem path of the arena backing file, for the startup log.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// High-water mark of arena bytes handed out (allocator `top`). The number
    /// that matters for the leak check: it only grows when a new slot is cut, so
    /// healthy churn recycles slots and it plateaus. A live-bytes sum would hide
    /// the reservation-leak / quarantine regression we hunt.
    pub fn used_bytes(&self) -> u64 {
        self.alloc.lock().top
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
        a.reap(Instant::now());
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
                reserved_at: Some(Instant::now()),
            },
        );
        Some((offset, false))
    }

    /// Marks a reserved object as fully written and readable.
    pub fn commit(&self, id: ObjectId) {
        if let Some(entry) = self.alloc.lock().live.get_mut(&id) {
            entry.committed = true;
            entry.reserved_at = None; // committed objects never expire
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
        let slice = unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(offset), size) };
        Some(slice.to_vec())
    }

    /// Writes a byte range into a *reserved* (not yet committed) object's slot,
    /// for a cross-host client streaming a large put in chunks. `rel_offset` is
    /// relative to the object's start. Unlike `write_at`, the range is untrusted
    /// (it came off the wire), so it is bounds-checked against the reserved size;
    /// out-of-range or already-committed writes return `false` and copy nothing.
    pub fn write_chunk(&self, id: ObjectId, rel_offset: u64, bytes: &[u8]) -> bool {
        let base = {
            let a = self.alloc.lock();
            let e = match a.live.get(&id) {
                Some(e) if !e.committed => e,
                _ => return false,
            };
            match rel_offset.checked_add(bytes.len() as u64) {
                Some(end) if end <= e.size => e.offset + rel_offset,
                _ => return false,
            }
        };
        self.write_at(base, bytes);
        true
    }

    /// Reads a byte range out of a committed object, for a cross-host client
    /// streaming a large get in chunks. The range is untrusted, so it is
    /// bounds-checked against the object size; out-of-range or uncommitted reads
    /// return `None`. The caller verifies the whole-object checksum after
    /// reassembly, so no per-chunk hashing here.
    pub fn read_chunk(&self, id: ObjectId, rel_offset: u64, len: u64) -> Option<Vec<u8>> {
        let base = {
            let a = self.alloc.lock();
            let e = a.live.get(&id)?;
            if !e.committed {
                return None;
            }
            match rel_offset.checked_add(len) {
                Some(end) if end <= e.size => e.offset + rel_offset,
                _ => return None,
            }
        };
        // SAFETY: committed region within bounds checked above; immutable until release.
        let slice = unsafe {
            std::slice::from_raw_parts(self.map.as_ptr().add(base as usize), len as usize)
        };
        Some(slice.to_vec())
    }

    /// Recycles a slot into its exact-size free list, LIFO — the next same-size
    /// reserve reuses the warmest offset, so the put/read/release hot path never
    /// walks into cold arena pages. Release asserts no reader still holds the
    /// offset; releasing mid-read is an application use-after-free (a single-node
    /// arena does no reader refcounting).
    pub fn release(&self, id: ObjectId) {
        let mut a = self.alloc.lock();
        if let Some(entry) = a.live.remove(&id) {
            a.free.entry(entry.slot).or_default().push(entry.offset);
        }
    }

    /// Test hook: drive the lazy reaper as if `d` had elapsed, without sleeping.
    #[cfg(test)]
    fn reap_after(&self, d: Duration) {
        let mut a = self.alloc.lock();
        a.reap(Instant::now() + d);
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
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8);
    let chunk = src.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (d, s) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || d.copy_from_slice(s));
        }
    });
}

/// `src.to_vec()` with the copy parallelized (and the redundant zero-fill of a
/// `vec![0; n]` skipped) — the read half of the same bandwidth problem.
#[allow(clippy::uninit_vec)] // copy_wide fully overwrites out before any read; u8 has no invalid bit pattern
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

/// Removes arena files whose owning process is gone, so a coordinator killed by
/// SIGKILL (bypassing `Drop`) does not leak its /dev/shm backing forever. Only
/// runs where `/proc` exists (Linux, where the arena dir is /dev/shm); a live —
/// or pid-reused — file is left alone, so it never deletes an in-use arena.
fn sweep_dead(dir: &Path) {
    if !Path::new("/proc").is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        // arena-{pid}-{uuid}
        let Some(pid) = name
            .strip_prefix("arena-")
            .and_then(|rest| rest.split('-').next())
        else {
            continue;
        };
        if pid.parse::<u32>().is_ok() && !Path::new(&format!("/proc/{pid}")).exists() {
            let _ = fs::remove_file(entry.path());
        }
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
        assert_eq!(
            &map[offset as usize..offset as usize + payload.len()],
            &payload[..]
        );
        assert_eq!(store.read(id).unwrap(), payload);

        store.release(id);
        assert!(store.meta(id).is_none());
        // Same exact size recycles the released offset immediately (LIFO), so the
        // hot path reuses warm pages rather than bumping `top` into cold arena.
        let id2 = ObjectId::new();
        assert_eq!(
            store
                .reserve(id2, payload.len() as u64, Codec::RawBytes, checksum)
                .unwrap()
                .0,
            offset
        );
        assert!(map_arena("deadbeef").is_none());
    }

    #[test]
    fn uncommitted_reservation_is_reaped_after_ttl() {
        let store = ArenaStore::new().unwrap();
        let id = ObjectId::new();
        let (offset, _) = store.reserve(id, 4096, Codec::RawBytes, [0; 32]).unwrap();
        // Client crashes before commit: never readable, and the slot must not
        // leak. After the TTL its offset returns to the free list.
        assert!(store.meta(id).is_none());
        store.reap_after(RESERVE_TTL);
        let id2 = ObjectId::new();
        assert_eq!(
            store
                .reserve(id2, 4096, Codec::RawBytes, [0; 32])
                .unwrap()
                .0,
            offset
        );
        // A committed reservation is never reaped as stale.
        store.commit(id2);
        store.reap_after(RESERVE_TTL);
        assert!(store.meta(id2).is_some());
    }

    // Regression guard for the 5x same-host put regression (112830a): a serial
    // put/read/release churn of one size must reuse its slot, so `top` plateaus
    // at one slot regardless of loop count. Under the old release-quarantine
    // nothing recycled and `top` grew to iters*slot, cold-faulting every put.
    // Machine-speed-independent: asserts the allocator invariant, not a latency.
    #[test]
    fn same_size_churn_reuses_one_slot() {
        let store = ArenaStore::new().unwrap();
        let size = 1 << 20;
        let payload = vec![7u8; size];
        let checksum = crate::cluster::checksum(&payload);
        for _ in 0..1000 {
            let id = ObjectId::new();
            let (offset, _) = store
                .reserve(id, size as u64, Codec::RawBytes, checksum)
                .unwrap();
            store.write_at(offset, &payload);
            store.commit(id);
            assert_eq!(store.read(id).unwrap().len(), size);
            store.release(id);
        }
        // One 8-byte-aligned slot, not 1000. `top` is the high-water mark.
        assert_eq!(store.used_bytes(), size as u64);
    }

    #[test]
    fn chunked_write_and_read_round_trip_with_bounds() {
        let store = ArenaStore::new().unwrap();
        let id = ObjectId::new();
        // Object larger than one frame, written in three ranges.
        let size = 20_000_000u64;
        store.reserve(id, size, Codec::RawBytes, [0; 32]).unwrap();
        let chunk = 8_000_000usize;
        let mut expected = vec![0u8; size as usize];
        for (i, off) in (0..size).step_by(chunk).enumerate() {
            let len = chunk.min((size - off) as usize);
            let data = vec![i as u8 + 1; len];
            expected[off as usize..off as usize + len].copy_from_slice(&data);
            assert!(store.write_chunk(id, off, &data), "in-bounds write");
        }
        // Out-of-bounds write is rejected, copies nothing.
        assert!(!store.write_chunk(id, size - 10, &[7u8; 100]));
        // Cannot read chunks before commit.
        assert!(store.read_chunk(id, 0, 10).is_none());
        store.commit(id);

        // Reassemble via read_chunk in a different stride; must match.
        let mut got = Vec::with_capacity(size as usize);
        let mut off = 0u64;
        while off < size {
            let len = (size - off).min(7_000_000);
            got.extend_from_slice(&store.read_chunk(id, off, len).unwrap());
            off += len;
        }
        assert_eq!(got, expected);
        // Out-of-bounds read is rejected.
        assert!(store.read_chunk(id, size - 5, 10).is_none());
    }
}
