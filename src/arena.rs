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

use crate::{
    error::Error,
    ids::{ObjectId, RequestId},
    operation::Codec,
    protocol::ArenaWriteMode,
};

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
    reservation: RequestId,
    mode: ArenaWriteMode,
    written_until: u64,
    committed: bool,
    reserved_at: Option<Instant>,
}

pub struct Reservation {
    pub offset: u64,
    pub reservation: RequestId,
    pub committed: bool,
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
        mode: ArenaWriteMode,
    ) -> Result<Reservation, Error> {
        let slot = size.max(1).div_ceil(8) * 8;
        let mut a = self.alloc.lock();
        a.reap(Instant::now());
        if let Some(entry) = a.live.get(&id) {
            if entry.size != size
                || entry.codec != codec
                || entry.checksum != checksum
                || entry.mode != mode
            {
                return Err(Error::ObjectConflict(id));
            }
            return Ok(Reservation {
                offset: entry.offset,
                reservation: entry.reservation,
                committed: entry.committed,
            });
        }
        let offset = match a.free.get_mut(&slot).and_then(Vec::pop) {
            Some(off) => off,
            None => {
                let off = a.top;
                let next = off
                    .checked_add(slot)
                    .ok_or_else(|| Error::CapacityExceeded("arena exhausted".into()))?;
                if next > ARENA_BYTES {
                    return Err(Error::CapacityExceeded("arena exhausted".into()));
                }
                a.top = next;
                off
            }
        };
        let reservation = RequestId::new();
        a.live.insert(
            id,
            Entry {
                offset,
                size,
                slot,
                codec,
                checksum,
                reservation,
                mode,
                written_until: 0,
                committed: false,
                reserved_at: Some(Instant::now()),
            },
        );
        Ok(Reservation {
            offset,
            reservation,
            committed: false,
        })
    }

    pub fn commit(&self, id: ObjectId, reservation: RequestId) -> Result<Meta, Error> {
        let mut a = self.alloc.lock();
        let entry = a.live.get_mut(&id).ok_or(Error::ObjectNotFound(id))?;
        if entry.reservation != reservation {
            return Err(Error::StaleFence);
        }
        if entry.mode == ArenaWriteMode::Streamed && entry.written_until != entry.size {
            return Err(Error::Protocol("arena object incomplete".into()));
        }
        entry.committed = true;
        entry.reserved_at = None;
        Ok(Meta {
            offset: entry.offset,
            size: entry.size,
            codec: entry.codec.clone(),
            checksum: entry.checksum,
        })
    }

    pub fn rollback(&self, id: ObjectId, reservation: RequestId) -> Result<(), Error> {
        let mut a = self.alloc.lock();
        let entry = a.live.get(&id).ok_or(Error::ObjectNotFound(id))?;
        if entry.reservation != reservation || !entry.committed {
            return Err(Error::StaleFence);
        }
        let entry = a.live.remove(&id).unwrap();
        a.free.entry(entry.slot).or_default().push(entry.offset);
        Ok(())
    }

    /// Coordinator-side write: reserve + copy `bytes` in + commit, for payloads
    /// the coordinator already holds (e.g. a same-host object arriving inline).
    pub fn put(&self, id: ObjectId, bytes: &[u8], codec: Codec, checksum: [u8; 32]) -> Option<u64> {
        let reservation = self
            .reserve(
                id,
                bytes.len() as u64,
                codec,
                checksum,
                ArenaWriteMode::Direct,
            )
            .ok()?;
        if !reservation.committed {
            self.write_at(reservation.offset, bytes);
            self.commit(id, reservation.reservation).ok()?;
        }
        Some(reservation.offset)
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
    pub fn write_chunk(
        &self,
        id: ObjectId,
        reservation: RequestId,
        rel_offset: u64,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let mut a = self.alloc.lock();
        let entry = a.live.get_mut(&id).ok_or(Error::ObjectNotFound(id))?;
        if entry.reservation != reservation || entry.committed {
            return Err(Error::StaleFence);
        }
        if entry.mode != ArenaWriteMode::Streamed {
            return Err(Error::Protocol("arena reservation is not streamed".into()));
        }
        let end = rel_offset
            .checked_add(bytes.len() as u64)
            .filter(|&end| end <= entry.size)
            .ok_or_else(|| Error::Protocol("put chunk out of bounds".into()))?;
        let base = entry.offset + rel_offset;
        if rel_offset == entry.written_until {
            self.write_at(base, bytes);
            entry.written_until = end;
            entry.reserved_at = Some(Instant::now());
            return Ok(());
        }
        if end <= entry.written_until {
            let existing = unsafe {
                std::slice::from_raw_parts(self.map.as_ptr().add(base as usize), bytes.len())
            };
            return if existing == bytes {
                Ok(())
            } else {
                Err(Error::ObjectConflict(id))
            };
        }
        Err(Error::Protocol("put chunk must be contiguous".into()))
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
        let reserved = store
            .reserve(
                id,
                payload.len() as u64,
                Codec::RawBytes,
                checksum,
                ArenaWriteMode::Direct,
            )
            .unwrap();
        assert!(!reserved.committed);
        assert!(store.meta(id).is_none());
        store.write_at(reserved.offset, &payload);
        store.commit(id, reserved.reservation).unwrap();

        let meta = store.meta(id).unwrap();
        assert_eq!(meta.offset, reserved.offset);
        assert_eq!(meta.size, payload.len() as u64);
        let map = map_arena(store.token()).unwrap();
        assert_eq!(
            &map[reserved.offset as usize..reserved.offset as usize + payload.len()],
            &payload[..]
        );
        assert_eq!(store.read(id).unwrap(), payload);

        store.release(id);
        let next = store
            .reserve(
                ObjectId::new(),
                payload.len() as u64,
                Codec::RawBytes,
                checksum,
                ArenaWriteMode::Direct,
            )
            .unwrap();
        assert_eq!(next.offset, reserved.offset);
        assert!(map_arena("deadbeef").is_none());
    }

    #[test]
    fn reservation_is_token_fenced_and_streamed_commit_requires_coverage() {
        let store = ArenaStore::new().unwrap();
        let id = ObjectId::new();
        let reserved = store
            .reserve(id, 6, Codec::RawBytes, [0; 32], ArenaWriteMode::Streamed)
            .unwrap();
        assert_eq!(
            store.write_chunk(id, RequestId::new(), 0, b"abc"),
            Err(Error::StaleFence)
        );
        store
            .write_chunk(id, reserved.reservation, 0, b"abc")
            .unwrap();
        assert!(matches!(
            store.commit(id, reserved.reservation),
            Err(Error::Protocol(_))
        ));
        assert!(matches!(
            store.write_chunk(id, reserved.reservation, 4, b"ef"),
            Err(Error::Protocol(_))
        ));
        store
            .write_chunk(id, reserved.reservation, 0, b"abc")
            .unwrap();
        assert_eq!(
            store.write_chunk(id, reserved.reservation, 0, b"abd"),
            Err(Error::ObjectConflict(id))
        );
        store
            .write_chunk(id, reserved.reservation, 3, b"def")
            .unwrap();
        store.commit(id, reserved.reservation).unwrap();
        assert_eq!(store.read(id).unwrap(), b"abcdef");
    }

    #[test]
    fn stale_reservation_cannot_mutate_recycled_slot() {
        let store = ArenaStore::new().unwrap();
        let id = ObjectId::new();
        let old = store
            .reserve(id, 4096, Codec::RawBytes, [0; 32], ArenaWriteMode::Direct)
            .unwrap();
        store.reap_after(RESERVE_TTL);
        let id2 = ObjectId::new();
        let new = store
            .reserve(id2, 4096, Codec::RawBytes, [0; 32], ArenaWriteMode::Direct)
            .unwrap();
        assert_eq!(old.offset, new.offset);
        assert!(matches!(
            store.commit(id2, old.reservation),
            Err(Error::StaleFence)
        ));
        store.commit(id2, new.reservation).unwrap();
        store.reap_after(RESERVE_TTL);
        assert!(store.meta(id2).is_some());
    }

    #[test]
    fn rollback_recycles_immediately() {
        let store = ArenaStore::new().unwrap();
        let second_id = ObjectId::new();
        let second = store
            .reserve(
                second_id,
                1024,
                Codec::RawBytes,
                [0; 32],
                ArenaWriteMode::Direct,
            )
            .unwrap();
        store.commit(second_id, second.reservation).unwrap();
        store.rollback(second_id, second.reservation).unwrap();
        let third = store
            .reserve(
                ObjectId::new(),
                1024,
                Codec::RawBytes,
                [0; 32],
                ArenaWriteMode::Direct,
            )
            .unwrap();
        assert_eq!(third.offset, second.offset);
    }

    #[test]
    fn same_size_churn_reuses_one_slot() {
        let store = ArenaStore::new().unwrap();
        let size = 1 << 20;
        let payload = vec![7u8; size];
        let checksum = crate::cluster::checksum(&payload);
        for _ in 0..1000 {
            let id = ObjectId::new();
            let reserved = store
                .reserve(
                    id,
                    size as u64,
                    Codec::RawBytes,
                    checksum,
                    ArenaWriteMode::Direct,
                )
                .unwrap();
            store.write_at(reserved.offset, &payload);
            store.commit(id, reserved.reservation).unwrap();
            assert_eq!(store.read(id).unwrap().len(), size);
            store.release(id);
        }
        assert_eq!(store.used_bytes(), size as u64);
    }

    #[test]
    fn chunked_write_and_read_round_trip_with_bounds() {
        let store = ArenaStore::new().unwrap();
        let id = ObjectId::new();
        let size = 20_000_000u64;
        let reserved = store
            .reserve(id, size, Codec::RawBytes, [0; 32], ArenaWriteMode::Streamed)
            .unwrap();
        let chunk = 8_000_000usize;
        let mut expected = vec![0u8; size as usize];
        for (i, off) in (0..size).step_by(chunk).enumerate() {
            let len = chunk.min((size - off) as usize);
            let data = vec![i as u8 + 1; len];
            expected[off as usize..off as usize + len].copy_from_slice(&data);
            store
                .write_chunk(id, reserved.reservation, off, &data)
                .unwrap();
        }
        assert!(store
            .write_chunk(id, reserved.reservation, size - 10, &[7u8; 100])
            .is_err());
        assert!(store.read_chunk(id, 0, 10).is_none());
        store.commit(id, reserved.reservation).unwrap();

        let mut got = Vec::with_capacity(size as usize);
        let mut off = 0u64;
        while off < size {
            let len = (size - off).min(7_000_000);
            got.extend_from_slice(&store.read_chunk(id, off, len).unwrap());
            off += len;
        }
        assert_eq!(got, expected);
        assert!(store.read_chunk(id, size - 5, 10).is_none());
    }
}
