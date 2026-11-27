//! Saved states: states written to the SSD. This module saves states and loads
//! them back. Their files lie in one folder, set by state_dir.
//!
//! A saved state `ID` has an index file, `ID.state`, and the files that the
//! index names. Each part file is `OWNER-N.rows`, numbered from 0 within its
//! creating saved state `OWNER`; branches may reference prefix part files owned
//! by an earlier saved state. Its snapshots are `ID-V-POS.snap`. V is the save
//! that wrote the snapshot. POS is its position.
//! The index holds the model's identity and the tokens. For each part it holds
//! the positions, bytes, owner and file number. For each snapshot it holds the
//! position, the kind, the stale rows and the bytes. A SHA-256 sum in the index
//! covers every byte after it.
//!
//! Each file is written under a temporary name that ends in `.tmp`. Then it is
//! synced and renamed. The index is written last. So after a crash, each saved
//! state is as it was before the save, or as the save left it. The folder may
//! also hold files that no index names. The next start deletes them. The store
//! never deletes a file that is not its own.
//!
//! The files have mode 0600 and the folder 0700: only their owner may use them.
//! One server at a time uses the folder. It holds a lock on the folder's `lock`
//! file. Another server runs without saved states.

use crate::engine::{self, Model};
use crate::state_cache::{
    Cache, Entry, Part, SaveDecision, SavedSnapshot, SavedState, SavedStateId, Snapshot,
    SnapshotKind, count_unused_prefix_positions, prefix_parts,
};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::c_int;
use std::fs::{self, File};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

/// The first bytes of every index.
const INDEX_MAGIC: &[u8; 8] = b"EREMITEI";
/// The index format's version. A server loads only indexes of its own format
/// version.
const INDEX_FORMAT_VERSION: u32 = 2;
/// The bytes of an index's head, before its parts. The head holds the magic,
/// the format version, a reserved word, the SHA-256 sum, the identity, the
/// save's version, the replaced positions, and the counts of tokens, parts and
/// snapshots. It is 8 + 4 + 4 + 32 + 32 + 8 + 8 + 8 + 4 + 4, in the order of
/// the head's fields.
const INDEX_HEAD_BYTES: usize = 112;

// Where each field of an index's head starts. The sum covers the bytes from
// IDENTITY_AT on.
const FORMAT_VERSION_AT: usize = 8;
const SUM_AT: usize = 16;
const IDENTITY_AT: usize = 48;
const SAVE_VERSION_AT: usize = 80;
const REPLACED_AT: usize = 88;
const TOKEN_COUNT_AT: usize = 96;
const PART_COUNT_AT: usize = 104;
const SNAPSHOT_COUNT_AT: usize = 108;

/// The bytes of a model's identity, a SHA-256 sum.
const IDENTITY_BYTES: usize = 32;
/// The bytes of the index's SHA-256 sum.
const SUM_BYTES: usize = 32;
/// A part's record: from, to, bytes, owner, file number and a reserved word.
const PART_RECORD_BYTES: usize = 32;
/// A snapshot's record: position, kind, 3 reserved bytes, stale rows, a
/// reserved word, bytes and version.
const SNAPSHOT_RECORD_BYTES: usize = 32;
/// A token: one i32.
const TOKEN_BYTES: usize = 4;
/// The most digits of a number in a file name. Every 19-digit number fits in a
/// u64.
const MAX_NUMBER_DIGITS: usize = 19;
/// The bytes of one GiB.
const GIB_BYTES: u64 = 1 << 30;
/// The bytes that a save leaves free on the folder's volume.
const VOLUME_RESERVE_BYTES: u64 = 10 * GIB_BYTES;
/// Only the owner may list, read and write the folder.
const FOLDER_MODE: u32 = 0o700;
/// Only the owner may read and write the files.
const FILE_MODE: u32 = 0o600;
/// A file descriptor that makes a write return its bytes without writing.
const COUNT_ONLY_FD: c_int = -1;

/// The writes and reads of one state's rows and snapshots, for saving and
/// loading. An engine state implements it. So does the tests' fake state.
pub trait StateIo {
    /// What the state's snapshots are known by.
    type SnapshotHandle: Copy;

    /// Writes the rows that positions `from..to` wrote to `fd`. Returns the
    /// bytes written, or None when the write fails or a second stop request
    /// ends it. With fd -1, it writes nothing and returns the bytes it would
    /// write.
    fn write_rows(&self, from: usize, to: usize, fd: c_int) -> Option<u64>;

    /// Writes `snapshot` to `fd`, without its logits. With no `snapshot`, it
    /// writes the state's rolling rows at its end, `pos`. Returns the bytes
    /// written, or None when the write fails. With fd -1, it writes nothing and
    /// returns the bytes it would write.
    fn write_snapshot(
        &self,
        snapshot: Option<Self::SnapshotHandle>,
        pos: usize,
        fd: c_int,
    ) -> Option<u64>;

    /// Reads the model's identity: what its saved states depend on.
    fn read_model_identity(&self) -> [u8; IDENTITY_BYTES];

    /// Reads the rows before `positions` into the state from the part files
    /// `fds`, in the order the parts were written. Returns false when a part is
    /// short, damaged or of another shape.
    fn read_rows(&self, positions: usize, fds: &[c_int]) -> bool;

    /// Reads a snapshot from `fd`. Returns None when the file is short, damaged
    /// or of another shape.
    fn read_snapshot(&self, fd: c_int) -> Option<Self::SnapshotHandle>;
}

/// A state of an engine model, for saving and loading.
pub struct EngineState<'a> {
    /// The model that owns the state.
    pub model: &'a Model,
    /// The engine's state.
    pub state: engine::StateHandle,
}

impl StateIo for EngineState<'_> {
    type SnapshotHandle = engine::SnapshotHandle;

    fn write_rows(&self, from: usize, to: usize, fd: c_int) -> Option<u64> {
        self.model.state_write(self.state, from, to, fd)
    }

    fn write_snapshot(
        &self,
        snapshot: Option<engine::SnapshotHandle>,
        pos: usize,
        fd: c_int,
    ) -> Option<u64> {
        self.model.snapshot_write(snapshot, self.state, pos, fd)
    }

    fn read_model_identity(&self) -> [u8; IDENTITY_BYTES] {
        self.model.state_identity()
    }

    fn read_rows(&self, positions: usize, fds: &[c_int]) -> bool {
        self.model.state_read(self.state, positions, fds)
    }

    fn read_snapshot(&self, fd: c_int) -> Option<engine::SnapshotHandle> {
        self.model.snapshot_read(fd)
    }
}

/// Why a saved state was not loaded.
#[derive(Debug, PartialEq)]
pub enum LoadError {
    /// A file is missing, short, damaged or of another shape. The caller
    /// deletes the saved state. The text says what failed.
    Damaged(String),
    /// A file exists but cannot be opened now. The saved state stays. The text
    /// says what failed.
    Unavailable(String),
}

/// The saved states' folder, locked for this server, and counts of this run's
/// loads, saves and bytes written.
pub struct Store {
    /// The folder that holds the saved states' files.
    folder: PathBuf,
    /// The open `lock` file. The lock lasts while the store exists.
    _lock: File,
    /// The id that the next new saved state gets.
    next_id: SavedStateId,
    /// The saved states loaded in this run.
    pub load_count: u64,
    /// The states saved in this run.
    pub save_count: u64,
    /// The bytes written in this run.
    pub bytes_written: u64,
}

/// A saved state found at the start.
pub struct FoundSavedState {
    /// The saved state's id.
    pub id: SavedStateId,
    /// The identity of the model that saved it.
    pub identity: [u8; IDENTITY_BYTES],
    /// The saved state. Its model is None until the caller matches the
    /// identity.
    pub saved_state: SavedState,
    /// Its last use: the modification time of its index.
    pub last_used_time: SystemTime,
}

/// A file name of the store, parsed.
#[derive(Debug, PartialEq)]
enum FileName {
    /// `ID.state`: the index of saved state ID.
    Index(SavedStateId),
    /// `ID-N.rows`: a part of saved state ID.
    Part(SavedStateId),
    /// `ID-V-POS.snap`: a snapshot of saved state ID.
    Snapshot(SavedStateId),
}

/// Parses a file name of the store. Returns what the file is, and whether its
/// name is temporary. Returns None for any other name.
fn parse_file_name(name: &str) -> Option<(FileName, bool)> {
    let (name, is_temporary) = match name.strip_suffix(".tmp") {
        Some(base) => (base, true),
        None => (name, false),
    };
    // a number as the store writes it: digits, no leading zero
    let parse_number = |text: &str| {
        let plain = !text.is_empty()
            && text.len() <= MAX_NUMBER_DIGITS
            && text.bytes().all(|byte| byte.is_ascii_digit())
            && (text == "0" || !text.starts_with('0'));
        plain.then(|| text.parse::<u64>().ok()).flatten()
    };
    let (stem, extension) = name.rsplit_once('.')?;
    let fields: Vec<&str> = stem.split('-').collect();
    let id = parse_number(fields[0])?;
    // the name's fields: ID, ID-N or ID-V-POS
    let parsed = match (extension, fields.len()) {
        ("state", 1) => FileName::Index(id),
        ("rows", 2) if parse_number(fields[1]).is_some() => FileName::Part(id),
        ("snap", 3) if parse_number(fields[1]).is_some() && parse_number(fields[2]).is_some() => {
            FileName::Snapshot(id)
        }
        _ => return None,
    };
    Some((parsed, is_temporary))
}

/// Formats the file name of the index of saved state `id`: `ID.state`.
fn format_index_file_name(id: SavedStateId) -> String {
    format!("{id}.state")
}

/// Formats the file name of part `part`: `OWNER-N.rows`.
fn format_part_file_name(part: &Part) -> String {
    format!("{}-{}.rows", part.owner, part.file_number)
}

/// Formats the file name of snapshot `snapshot` of saved state `id`:
/// `ID-V-POS.snap`.
fn format_snapshot_file_name(id: SavedStateId, snapshot: &SavedSnapshot) -> String {
    format!("{id}-{}-{}.snap", snapshot.version, snapshot.snapshot.pos)
}

/// Encodes a snapshot kind as its code in the index. The codes are part of the
/// format.
fn encode_snapshot_kind(kind: SnapshotKind) -> u8 {
    kind as u8
}

/// Decodes a snapshot kind from its code in the index. Returns None for any
/// other code.
fn decode_snapshot_kind(code: u8) -> Option<SnapshotKind> {
    [
        SnapshotKind::PromptEnd,
        SnapshotKind::GenerationEnd,
        SnapshotKind::Branch,
    ]
    .into_iter()
    .find(|&kind| encode_snapshot_kind(kind) == code)
}

/// Encodes the index of `saved_state`, with the model's `identity`.
pub fn encode_index(identity: &[u8; IDENTITY_BYTES], saved_state: &SavedState) -> Vec<u8> {
    let mut index = Vec::with_capacity(
        INDEX_HEAD_BYTES
            + PART_RECORD_BYTES * saved_state.parts.len()
            + SNAPSHOT_RECORD_BYTES * saved_state.snapshots.len()
            + TOKEN_BYTES * saved_state.tokens.len(),
    );
    index.extend_from_slice(INDEX_MAGIC);
    index.extend_from_slice(&INDEX_FORMAT_VERSION.to_le_bytes());
    index.extend_from_slice(&0u32.to_le_bytes()); // a reserved word, 0
    // the sum: written as zeros, then filled in
    index.extend_from_slice(&[0u8; SUM_BYTES]);
    index.extend_from_slice(identity);
    index.extend_from_slice(&saved_state.version.to_le_bytes());
    index.extend_from_slice(&(saved_state.replaced_positions as u64).to_le_bytes());
    index.extend_from_slice(&(saved_state.tokens.len() as u64).to_le_bytes());
    index.extend_from_slice(&(saved_state.parts.len() as u32).to_le_bytes());
    index.extend_from_slice(&(saved_state.snapshots.len() as u32).to_le_bytes());
    for part in &saved_state.parts {
        index.extend_from_slice(&(part.from as u32).to_le_bytes());
        index.extend_from_slice(&(part.to as u32).to_le_bytes());
        index.extend_from_slice(&part.bytes.to_le_bytes());
        index.extend_from_slice(&part.owner.to_le_bytes());
        index.extend_from_slice(&part.file_number.to_le_bytes());
        index.extend_from_slice(&0u32.to_le_bytes()); // a reserved word, 0
    }
    for saved_snapshot in &saved_state.snapshots {
        index.extend_from_slice(&(saved_snapshot.snapshot.pos as u32).to_le_bytes());
        // the kind's code and 3 reserved bytes
        index.extend_from_slice(&[encode_snapshot_kind(saved_snapshot.snapshot.kind), 0, 0, 0]);
        index.extend_from_slice(&(saved_snapshot.snapshot.stale_rows as u32).to_le_bytes());
        index.extend_from_slice(&0u32.to_le_bytes()); // a reserved word, 0
        index.extend_from_slice(&saved_snapshot.snapshot.bytes.to_le_bytes());
        index.extend_from_slice(&saved_snapshot.version.to_le_bytes());
    }
    for token in &saved_state.tokens {
        index.extend_from_slice(&token.to_le_bytes());
    }
    let sum = Sha256::digest(&index[IDENTITY_AT..]);
    index[SUM_AT..SUM_AT + SUM_BYTES].copy_from_slice(&sum);
    index
}

/// Returns whether `index` is an index of another format version. A server of
/// that version wrote it. Only such a server loads it.
fn is_other_format_version(index: &[u8]) -> bool {
    let version_end = FORMAT_VERSION_AT + size_of::<u32>();
    index.len() >= version_end
        && &index[..INDEX_MAGIC.len()] == INDEX_MAGIC
        && index[FORMAT_VERSION_AT..version_end] != INDEX_FORMAT_VERSION.to_le_bytes()
}

/// Decodes an index into the model's identity and the saved state. The saved
/// state's model is None until the caller matches the identity. Returns None
/// when the index is damaged, of another format version, or inconsistent.
pub fn decode_index(index: &[u8]) -> Option<([u8; IDENTITY_BYTES], SavedState)> {
    let read_u32_at = |at: usize| {
        let bytes = index.get(at..at + size_of::<u32>())?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    };
    let read_u64_at = |at: usize| {
        let bytes = index.get(at..at + size_of::<u64>())?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    };
    if index.len() < INDEX_HEAD_BYTES
        || &index[..INDEX_MAGIC.len()] != INDEX_MAGIC
        || read_u32_at(FORMAT_VERSION_AT)? != INDEX_FORMAT_VERSION
        || Sha256::digest(&index[IDENTITY_AT..])[..] != index[SUM_AT..SUM_AT + SUM_BYTES]
    {
        return None;
    }
    let identity: [u8; IDENTITY_BYTES] = index[IDENTITY_AT..IDENTITY_AT + IDENTITY_BYTES]
        .try_into()
        .ok()?;
    let version = read_u64_at(SAVE_VERSION_AT)?;
    let replaced_positions = read_u64_at(REPLACED_AT)? as usize;
    let token_count = read_u64_at(TOKEN_COUNT_AT)? as usize;
    let part_count = read_u32_at(PART_COUNT_AT)? as usize;
    let snapshot_count = read_u32_at(SNAPSHOT_COUNT_AT)? as usize;
    if index.len()
        != INDEX_HEAD_BYTES
            + PART_RECORD_BYTES * part_count
            + SNAPSHOT_RECORD_BYTES * snapshot_count
            + TOKEN_BYTES * token_count
    {
        return None;
    }
    let mut at = INDEX_HEAD_BYTES;
    let mut saved_state = SavedState {
        model: None,
        tokens: Vec::with_capacity(token_count),
        parts: Vec::with_capacity(part_count),
        snapshots: Vec::with_capacity(snapshot_count),
        replaced_positions,
        // the cache sets it when it adds the saved state
        last_used: 0,
        version,
    };
    // a part's record: from at 0, to at 4, bytes at 8, owner at 16, file
    // number at 24, a reserved word at 28
    for _ in 0..part_count {
        saved_state.parts.push(Part {
            from: read_u32_at(at)? as usize,
            to: read_u32_at(at + 4)? as usize,
            bytes: read_u64_at(at + 8)?,
            owner: read_u64_at(at + 16)?,
            file_number: read_u32_at(at + 24)?,
        });
        at += PART_RECORD_BYTES;
    }
    // a snapshot's record: position at 0, kind at 4, stale rows at 8, a
    // reserved word at 12, bytes at 16, version at 24
    for _ in 0..snapshot_count {
        let kind = decode_snapshot_kind(index[at + 4])?;
        saved_state.snapshots.push(SavedSnapshot {
            snapshot: Snapshot {
                pos: read_u32_at(at)? as usize,
                kind,
                bytes: read_u64_at(at + 16)?,
                stale_rows: read_u32_at(at + 8)? as usize,
            },
            version: read_u64_at(at + 24)?,
        });
        at += SNAPSHOT_RECORD_BYTES;
    }
    for _ in 0..token_count {
        saved_state.tokens.push(read_u32_at(at)? as i32);
        at += TOKEN_BYTES;
    }
    // The parts must continue each other up to the state's end. The snapshots
    // must lie within it, at most one at each position.
    let parts_continue = saved_state.parts.first().is_some_and(|part| part.from == 0)
        && saved_state
            .parts
            .windows(2)
            .all(|pair| pair[1].from <= pair[0].to)
        && saved_state.parts.iter().all(|part| part.from <= part.to)
        && saved_state
            .parts
            .last()
            .is_some_and(|part| part.to == token_count);
    let mut positions = HashSet::new();
    let snapshots_fit = saved_state.snapshots.iter().all(|saved_snapshot| {
        saved_snapshot.snapshot.pos <= token_count && positions.insert(saved_snapshot.snapshot.pos)
    });
    (parts_continue && snapshots_fit).then_some((identity, saved_state))
}

impl Store {
    /// Opens the store in folder `dir`. It makes the folder when it is missing.
    /// It locks the folder for this server. It deletes damaged indexes, and
    /// indexes whose files are missing or of another size. It deletes the
    /// store's files that no index names. It keeps indexes of another format
    /// version, with their files. Returns the store and the saved states found,
    /// the least recently used first.
    pub fn open(dir: &Path) -> Result<(Store, Vec<FoundSavedState>), String> {
        let shown = dir.display();
        fs::DirBuilder::new()
            .recursive(true)
            .mode(FOLDER_MODE)
            .create(dir)
            .map_err(|e| format!("cannot make {shown}: {e}"))?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .mode(FILE_MODE)
            .open(dir.join("lock"))
            .map_err(|e| format!("cannot open {shown}/lock: {e}"))?;
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(format!("{shown} is in use by another server"));
        }
        let mut indexes = Vec::new();
        let mut files = Vec::new();
        let mut next_id = 1;
        for folder_entry in fs::read_dir(dir)
            .map_err(|e| format!("cannot read {shown}: {e}"))?
            .flatten()
        {
            let name = folder_entry.file_name().to_string_lossy().into_owned();
            match parse_file_name(&name) {
                Some((FileName::Index(id), false)) => indexes.push(id),
                Some((FileName::Index(id) | FileName::Part(id) | FileName::Snapshot(id), _)) => {
                    next_id = next_id.max(id + 1);
                    files.push((name, id));
                }
                None => {}
            }
        }
        let mut found = Vec::new();
        let mut named = HashSet::new();
        // saved states of another format version: their files stay
        let mut other_versions = HashSet::new();
        for id in indexes {
            next_id = next_id.max(id + 1);
            let path = dir.join(format_index_file_name(id));
            let last_used_time = fs::metadata(&path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let index = fs::read(&path).unwrap_or_default();
            if is_other_format_version(&index) {
                engine::log(&format!(
                    "states: {shown}/{id}.state is of another format version: kept, not loaded"
                ));
                other_versions.insert(id);
                continue;
            }
            let has_whole_files = |saved_state: &SavedState| {
                let file_size = |name: String| {
                    fs::metadata(dir.join(name))
                        .map(|metadata| metadata.len())
                        .ok()
                };
                saved_state
                    .parts
                    .iter()
                    .all(|part| file_size(format_part_file_name(part)) == Some(part.bytes))
                    && saved_state.snapshots.iter().all(|saved_snapshot| {
                        file_size(format_snapshot_file_name(id, saved_snapshot))
                            == Some(saved_snapshot.snapshot.bytes)
                    })
            };
            match decode_index(&index) {
                Some((identity, saved_state)) if has_whole_files(&saved_state) => {
                    named.extend(saved_state.parts.iter().map(format_part_file_name));
                    named.extend(
                        saved_state
                            .snapshots
                            .iter()
                            .map(|saved_snapshot| format_snapshot_file_name(id, saved_snapshot)),
                    );
                    found.push(FoundSavedState {
                        id,
                        identity,
                        saved_state,
                        last_used_time,
                    });
                }
                _ => {
                    engine::log(&format!(
                        "states: {shown}/{id}.state is damaged or misses files; deleted"
                    ));
                    let _ = fs::remove_file(&path);
                }
            }
        }
        for (name, id) in files {
            if !named.contains(&name) && !other_versions.contains(&id) {
                let _ = fs::remove_file(dir.join(&name));
            }
        }
        found.sort_by_key(|found_state| found_state.last_used_time);
        let store = Store {
            folder: dir.to_path_buf(),
            _lock: lock,
            next_id,
            load_count: 0,
            save_count: 0,
            bytes_written: 0,
        };
        Ok((store, found))
    }

    /// Gets the store's folder.
    pub fn get_folder(&self) -> &Path {
        &self.folder
    }

    /// Counts the bytes free for this user on the folder's volume. Returns 0
    /// when it cannot tell.
    pub fn count_free_bytes(&self) -> u64 {
        let Ok(path) = std::ffi::CString::new(self.folder.as_os_str().as_encoded_bytes()) else {
            return 0;
        };
        let mut stats: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(path.as_ptr(), &mut stats) } != 0 {
            return 0;
        }
        stats.f_bavail as u64 * stats.f_bsize as u64
    }

    /// Creates file `name` under its temporary name, `name.tmp`. It first
    /// deletes an old file of that temporary name. Returns the open file, its
    /// temporary path and its final path.
    pub(crate) fn create_temporary_file(
        folder: &Path,
        name: &str,
    ) -> Result<(File, PathBuf, PathBuf), String> {
        let (final_path, temporary_path) = (folder.join(name), folder.join(format!("{name}.tmp")));
        let _ = fs::remove_file(&temporary_path);
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(FILE_MODE)
            .open(&temporary_path)
            .map_err(|e| format!("cannot create {}: {e}", temporary_path.display()))?;
        Ok((file, temporary_path, final_path))
    }

    /// Syncs `file` and renames it from `temporary_path` to `final_path`.
    fn sync_and_rename(file: File, temporary_path: &Path, final_path: &Path) -> Result<(), String> {
        file.sync_all()
            .map_err(|e| format!("cannot sync {}: {e}", temporary_path.display()))?;
        fs::rename(temporary_path, final_path)
            .map_err(|e| format!("cannot rename {}: {e}", temporary_path.display()))
    }

    /// Writes file `name` under its temporary name. Then it syncs and renames
    /// the file. `write_contents` writes to a file descriptor and returns the
    /// bytes it wrote, or None. Returns those bytes. After a failure it deletes
    /// the temporary file.
    fn write_file(
        &self,
        name: &str,
        write_contents: impl FnOnce(c_int) -> Option<u64>,
    ) -> Result<u64, String> {
        let (file, temporary_path, final_path) = Self::create_temporary_file(&self.folder, name)?;
        let Some(bytes) = write_contents(file.as_raw_fd()) else {
            drop(file);
            let _ = fs::remove_file(&temporary_path);
            return Err(format!(
                "cannot write {name}, or a second stop request ended the write"
            ));
        };
        Self::sync_and_rename(file, &temporary_path, &final_path).inspect_err(|_| {
            let _ = fs::remove_file(&temporary_path);
        })?;
        Ok(bytes)
    }

    /// Writes the index of saved state `id` under its temporary name. Then it
    /// syncs and renames the index. Last it syncs the folder. The rename then
    /// reaches the SSD before any old file is deleted.
    fn write_index(
        &self,
        id: SavedStateId,
        identity: &[u8; IDENTITY_BYTES],
        saved_state: &SavedState,
    ) -> Result<(), String> {
        let temporary = Self::create_temporary_file(&self.folder, &format_index_file_name(id))?;
        let written = Self::replace_file(temporary, &encode_index(identity, saved_state));
        if let Ok(folder) = File::open(&self.folder) {
            unsafe { libc::fsync(folder.as_raw_fd()) };
        }
        written
    }

    /// Replaces the destination with a synced copy of the bytes.
    pub(crate) fn replace_file(
        temporary: (File, PathBuf, PathBuf),
        bytes: &[u8],
    ) -> Result<(), String> {
        let (mut file, temporary_path, final_path) = temporary;
        let written = std::io::Write::write_all(&mut file, bytes)
            .map_err(|e| format!("cannot write {}: {e}", temporary_path.display()))
            .and_then(|_| Self::sync_and_rename(file, &temporary_path, &final_path));
        if written.is_err() {
            let _ = fs::remove_file(&temporary_path);
        }
        written
    }

    /// Sets the last use of saved state `id` to `when`. Its index's
    /// modification time keeps it for the next start.
    fn set_last_used(&self, id: SavedStateId, when: SystemTime) {
        if let Ok(index) = fs::OpenOptions::new()
            .write(true)
            .open(self.folder.join(format_index_file_name(id)))
        {
            let _ = index.set_modified(when);
        }
    }

    /// Deletes the files of parts `parts` and snapshots `snapshots` of saved
    /// state `id`.
    fn delete_files(&self, id: SavedStateId, parts: &[Part], snapshots: &[SavedSnapshot]) {
        for part in parts {
            let _ = fs::remove_file(self.folder.join(format_part_file_name(part)));
        }
        for snapshot in snapshots {
            let _ = fs::remove_file(self.folder.join(format_snapshot_file_name(id, snapshot)));
        }
    }

    /// Deletes saved state `id`: its index first, then its snapshots and the
    /// part files that no remaining saved state references. Removes it from
    /// `cache`.
    pub fn delete_saved_state(&mut self, cache: &mut Cache, id: SavedStateId) {
        if let Some(saved_state) = cache.saved_states.remove(&id) {
            let _ = fs::remove_file(self.folder.join(format_index_file_name(id)));
            let referenced: HashSet<(SavedStateId, u32)> = cache
                .saved_states
                .values()
                .flat_map(|other| other.parts.iter().map(Part::file_key))
                .collect();
            let unshared_parts: Vec<Part> = saved_state
                .parts
                .into_iter()
                .filter(|part| !referenced.contains(&part.file_key()))
                .collect();
            self.delete_files(id, &unshared_parts, &saved_state.snapshots);
        }
    }

    /// Saves the evicted state `entry` to the SSD, as the cache decides. At
    /// quit, `entry` may be an idle state instead. `io` writes the state's
    /// rows. `find_snapshot` returns its snapshot at a position, or None.
    /// `others_stay` tells whether the states in the cache stay in memory. The
    /// saved state keeps the state's last use, `last_used_time`, also as its
    /// index's modification time. So when room is needed, the saved state used
    /// longest ago is deleted first, also among the states saved at quit. The
    /// save then deletes the saved states that the new one replaces or covers,
    /// and those it takes room from. Records the saved state in `cache`.
    /// Returns the log's line, if any.
    pub fn save_state<S: StateIo>(
        &mut self,
        cache: &mut Cache,
        io: &S,
        find_snapshot: &dyn Fn(usize) -> Option<S::SnapshotHandle>,
        entry: &Entry,
        last_used_time: SystemTime,
        others_stay: bool,
    ) -> Option<String> {
        let len = entry.tokens.len();
        let number = entry.number;
        let decision = cache.decide_save(entry, others_stay);
        let (target, from, version, mut parts, replaced_positions, old_snapshots) = match decision {
            SaveDecision::Skip => {
                return (len >= cache.min_unshared_tokens && cache.saved_byte_limit > 0).then(|| {
                    format!(
                        "state {number} ({len} tokens) not saved: it has fewer than {} unshared tokens",
                        cache.min_unshared_tokens
                    )
                });
            }
            SaveDecision::Append { saved_id, from } => {
                let saved_state = &cache.saved_states[&saved_id];
                let replaced_positions =
                    saved_state.replaced_positions + saved_state.tokens.len().saturating_sub(from);
                (
                    saved_id,
                    from,
                    saved_state.version + 1,
                    saved_state.parts.clone(),
                    replaced_positions,
                    saved_state.snapshots.clone(),
                )
            }
            SaveDecision::Branch { parent, from } => {
                let id = self.next_id;
                self.next_id += 1;
                let kept = prefix_parts(&cache.saved_states[&parent].parts, from);
                let replaced_positions = count_unused_prefix_positions(&kept, from);
                (id, from, 1, kept, replaced_positions, Vec::new())
            }
            SaveDecision::New { .. } => {
                let id = self.next_id;
                self.next_id += 1;
                // a new saved state: its rows from position 0, version 1, no
                // replaced positions
                (id, 0, 1, Vec::new(), 0, Vec::new())
            }
        };
        // It writes the rows from `from`, the newest prompt-end and branch
        // snapshots, and a snapshot at the state's end.
        let find_newest_snapshot = |kind: SnapshotKind| {
            entry
                .snapshots
                .iter()
                .filter(|snapshot| snapshot.kind == kind)
                .max_by_key(|snapshot| snapshot.pos)
                .copied()
        };
        let mut snapshots: Vec<(Snapshot, Option<S::SnapshotHandle>)> = [
            find_newest_snapshot(SnapshotKind::PromptEnd),
            find_newest_snapshot(SnapshotKind::Branch),
        ]
        .into_iter()
        .flatten()
        .filter_map(|snapshot| find_snapshot(snapshot.pos).map(|handle| (snapshot, Some(handle))))
        .collect();
        if !snapshots.iter().any(|(snapshot, _)| snapshot.pos == len) {
            let end = Snapshot {
                pos: len,
                kind: SnapshotKind::GenerationEnd,
                // the write gives its bytes
                bytes: 0,
                stale_rows: entry.stale_rows,
            };
            snapshots.push((end, None));
        }
        let snapshot_bytes: Option<Vec<u64>> = snapshots
            .iter()
            .map(|(snapshot, handle)| io.write_snapshot(*handle, snapshot.pos, COUNT_ONLY_FD))
            .collect();
        let (Some(row_bytes), Some(snapshot_bytes)) =
            (io.write_rows(from, len, COUNT_ONLY_FD), snapshot_bytes)
        else {
            return Some(format!(
                "state {number} not saved: the engine cannot write its state"
            ));
        };
        // The room the save needs: its files' bytes, less the bytes of the
        // files it replaces. An append replaces the saved state's old
        // snapshots. A new saved state may replace a whole saved state.
        let need_bytes = row_bytes + snapshot_bytes.iter().sum::<u64>();
        let replaced_bytes = match decision {
            SaveDecision::New {
                replaces: Some(replaced_id),
            } => cache.count_reclaimable_bytes(replaced_id),
            _ => old_snapshots
                .iter()
                .map(|saved_snapshot| saved_snapshot.snapshot.bytes)
                .sum(),
        };
        let keep = match decision {
            SaveDecision::New {
                replaces: Some(replaced_id),
            } => Some(replaced_id),
            SaveDecision::Branch { parent, .. } => Some(parent),
            _ => Some(target),
        };
        let Some(to_delete) = cache.find_saved_states_to_delete(
            need_bytes.saturating_sub(replaced_bytes),
            keep,
            Some(entry.last_used),
        ) else {
            return Some(format!(
                "state {number} ({len} tokens, {:.2} GiB) not saved: state_disk ({:.1} GiB) holds states used more recently",
                need_bytes as f64 / GIB_BYTES as f64,
                cache.saved_byte_limit as f64 / GIB_BYTES as f64
            ));
        };
        if self.count_free_bytes() < need_bytes + VOLUME_RESERVE_BYTES {
            return Some(format!(
                "state {number} ({len} tokens, {:.2} GiB) not saved: it would leave less than {} GiB free on the volume",
                need_bytes as f64 / GIB_BYTES as f64,
                VOLUME_RESERVE_BYTES / GIB_BYTES
            ));
        }
        let started = Instant::now();
        let next_file_number = parts
            .iter()
            .filter(|part| part.owner == target)
            .map(|part| part.file_number + 1)
            .max()
            .unwrap_or(0);
        let mut part = Part {
            from,
            to: len,
            // the write gives its bytes
            bytes: 0,
            owner: target,
            file_number: next_file_number,
        };
        let written = self.write_file(&format_part_file_name(&part), |fd| {
            io.write_rows(from, len, fd)
        });
        match written {
            Ok(bytes) => part.bytes = bytes,
            Err(message) => return Some(format!("state {number} not saved: {message}")),
        }
        let mut saved_snapshots = Vec::new();
        for (snapshot, handle) in &snapshots {
            let mut saved_snapshot = SavedSnapshot {
                snapshot: *snapshot,
                version,
            };
            let name = format_snapshot_file_name(target, &saved_snapshot);
            match self.write_file(&name, |fd| io.write_snapshot(*handle, snapshot.pos, fd)) {
                Ok(bytes) => {
                    saved_snapshot.snapshot.bytes = bytes;
                    saved_snapshots.push(saved_snapshot);
                }
                Err(message) => {
                    self.delete_files(target, &[part], &saved_snapshots);
                    return Some(format!("state {number} not saved: {message}"));
                }
            }
        }
        parts.push(part);
        let saved_state = SavedState {
            model: Some(entry.model),
            tokens: entry.tokens.clone(),
            parts,
            snapshots: saved_snapshots,
            replaced_positions,
            // the cache sets it when it adds the saved state
            last_used: 0,
            version,
        };
        if let Err(message) = self.write_index(target, &io.read_model_identity(), &saved_state) {
            self.delete_files(target, &[part], &saved_state.snapshots);
            return Some(format!("state {number} not saved: {message}"));
        }
        self.set_last_used(target, last_used_time);
        // Once the index is written, it deletes the old snapshots, the saved
        // state it replaces, and the saved states it takes room from.
        self.delete_files(target, &[], &old_snapshots);
        if let SaveDecision::New {
            replaces: Some(replaced_id),
        } = decision
        {
            self.delete_saved_state(cache, replaced_id);
        }
        for id in to_delete {
            self.delete_saved_state(cache, id);
        }
        let bytes = part.bytes
            + saved_state
                .snapshots
                .iter()
                .map(|saved_snapshot| saved_snapshot.snapshot.bytes)
                .sum::<u64>();
        cache.add_saved_state(target, saved_state);
        if let Some(saved_state) = cache.saved_states.get_mut(&target) {
            // the saved state keeps the state's last use on the cache's use
            // clock
            saved_state.last_used = entry.last_used;
        }
        for id in cache.find_covered_saved_states(target) {
            self.delete_saved_state(cache, id);
        }
        for id in cache
            .find_saved_states_to_delete(0, Some(target), None)
            .unwrap_or_default()
        {
            self.delete_saved_state(cache, id);
        }
        self.save_count += 1;
        self.bytes_written += bytes;
        Some(format!(
            "state {number} saved to the SSD as {target}: {len} tokens{}, {:.2} GiB written in {:.2} s; saved states take {:.1} of {:.1} GiB",
            if from > 0 {
                format!(" (from position {from})")
            } else {
                String::new()
            },
            bytes as f64 / GIB_BYTES as f64,
            started.elapsed().as_secs_f64(),
            cache.count_saved_bytes() as f64 / GIB_BYTES as f64,
            cache.saved_byte_limit as f64 / GIB_BYTES as f64
        ))
    }

    /// Loads saved state `id` into a new state through `io`. It reads the saved
    /// rows before `from`. With `snapshot_pos`, it also reads the snapshot at
    /// that position. It returns the snapshot without restoring it, or None
    /// without one. Marks the saved state used now.
    pub fn load_saved_state<S: StateIo>(
        &mut self,
        cache: &mut Cache,
        io: &S,
        id: SavedStateId,
        snapshot_pos: Option<usize>,
        from: usize,
    ) -> Result<Option<S::SnapshotHandle>, LoadError> {
        let damage = |message: String| LoadError::Damaged(message);
        let saved_state = cache
            .saved_states
            .get(&id)
            .ok_or(damage("it is gone".into()))?;
        // A missing file is damage. Other open errors may pass, such as
        // permissions or too many open files.
        let open_file = |name: String| {
            File::open(self.folder.join(&name)).map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => damage(format!("{name}: {e}")),
                _ => LoadError::Unavailable(format!("{name}: {e}")),
            })
        };
        let files = saved_state
            .parts
            .iter()
            .map(|part| open_file(format_part_file_name(part)))
            .collect::<Result<Vec<File>, LoadError>>()?;
        let fds: Vec<c_int> = files.iter().map(|file| file.as_raw_fd()).collect();
        if !io.read_rows(from, &fds) {
            return Err(damage("its rows are damaged or of another shape".into()));
        }
        let mut snapshot = None;
        if let Some(pos) = snapshot_pos {
            let saved_snapshot = saved_state
                .snapshots
                .iter()
                .find(|candidate| candidate.snapshot.pos == pos)
                .ok_or(damage("its snapshot is gone".into()))?;
            snapshot = io.read_snapshot(
                open_file(format_snapshot_file_name(id, saved_snapshot))?.as_raw_fd(),
            );
            if snapshot.is_none() {
                return Err(damage("its snapshot is damaged or of another shape".into()));
            }
        }
        self.set_last_used(id, SystemTime::now());
        cache.mark_saved_state_used(id);
        self.load_count += 1;
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_cache::SavedLink;

    /// Makes a saved state of `tokens` tokens in two parts that overlap by one
    /// position. It has one prompt-end snapshot at its end.
    fn make_saved_state(tokens: usize) -> SavedState {
        SavedState {
            model: None,
            tokens: (0..tokens as i32)
                .map(|position| position * 7 - 3)
                .collect(),
            parts: vec![
                Part {
                    from: 0,
                    to: tokens / 2,
                    bytes: 1000,
                    owner: 5,
                    file_number: 0,
                },
                Part {
                    from: tokens / 2 - 1,
                    to: tokens,
                    bytes: 500,
                    owner: 5,
                    file_number: 1,
                },
            ],
            snapshots: vec![SavedSnapshot {
                snapshot: Snapshot {
                    pos: tokens,
                    kind: SnapshotKind::PromptEnd,
                    bytes: 77,
                    stale_rows: 2,
                },
                version: 3,
            }],
            replaced_positions: 1,
            last_used: 0,
            version: 3,
        }
    }

    #[test]
    fn an_index_reads_back_as_written_and_a_changed_byte_is_refused() {
        let saved_state = make_saved_state(100);
        let identity = [9u8; IDENTITY_BYTES];
        let index = encode_index(&identity, &saved_state);
        let (decoded_identity, decoded) = decode_index(&index).expect("an index");
        assert_eq!(decoded_identity, identity);
        assert_eq!(
            (
                decoded.tokens,
                decoded.parts,
                decoded.snapshots,
                decoded.replaced_positions,
                decoded.version
            ),
            (
                saved_state.tokens,
                saved_state.parts,
                saved_state.snapshots,
                1,
                3
            )
        );
        // one byte each of the magic, the format version, the sum, the
        // identity, the counts and the last token
        for at in [0, 9, 20, 60, 110, index.len() - 1] {
            let mut bad = index.clone();
            bad[at] ^= 1;
            assert!(decode_index(&bad).is_none(), "a changed byte at {at}");
        }
        assert!(
            decode_index(&index[..index.len() - 4]).is_none(),
            "a short index"
        );
    }

    #[test]
    fn an_index_whose_parts_do_not_continue_each_other_is_refused() {
        let mut saved_state = make_saved_state(100);
        saved_state.parts[1].from = 60; // a gap after the first part's end
        assert!(decode_index(&encode_index(&[0; IDENTITY_BYTES], &saved_state)).is_none());
        let mut saved_state = make_saved_state(100);
        saved_state.parts[1].to = 99; // short of the state's end
        assert!(decode_index(&encode_index(&[0; IDENTITY_BYTES], &saved_state)).is_none());
    }

    #[test]
    fn only_the_stores_names_are_its_files() {
        assert_eq!(
            parse_file_name("12.state"),
            Some((FileName::Index(12), false))
        );
        assert_eq!(
            parse_file_name("12-3.rows"),
            Some((FileName::Part(12), false))
        );
        assert_eq!(
            parse_file_name("12-3-4096.snap.tmp"),
            Some((FileName::Snapshot(12), true))
        );
        for other in [
            "lock",
            "notes.txt",
            "12.state.bak",
            "a-3.rows",
            "12-3.snap",
            "12-x-4.snap",
            ".DS_Store",
            "12.STATE",
        ] {
            assert_eq!(parse_file_name(other), None, "{other}");
        }
        // numbers the store never writes: with leading zeros
        for other in [
            "05.state",
            "05-0.rows",
            "1-00.rows",
            "1-1-007.snap",
            "007-1-2.snap.tmp",
        ] {
            assert_eq!(parse_file_name(other), None, "{other}");
        }
    }

    /// A test state whose writes put zeros in the files.
    struct FakeState {
        /// The bytes it writes for its rows.
        row_bytes: u64,
        /// The bytes it writes for a snapshot.
        snapshot_bytes: u64,
        /// Whether every write to a file fails.
        fails_writes: bool,
    }

    impl FakeState {
        /// Writes `bytes` zero bytes to `fd`. With fd -1, it returns `bytes`
        /// without writing.
        fn write_zeros(&self, bytes: u64, fd: c_int) -> Option<u64> {
            if fd < 0 {
                return Some(bytes);
            }
            if self.fails_writes {
                return None;
            }
            let mut file = std::mem::ManuallyDrop::new(unsafe {
                <File as std::os::fd::FromRawFd>::from_raw_fd(fd)
            });
            std::io::Write::write_all(&mut *file, &vec![0u8; bytes as usize])
                .ok()
                .map(|_| bytes)
        }
    }

    impl StateIo for FakeState {
        type SnapshotHandle = ();

        fn write_rows(&self, _from: usize, _to: usize, fd: c_int) -> Option<u64> {
            self.write_zeros(self.row_bytes, fd)
        }

        fn write_snapshot(&self, _snapshot: Option<()>, _pos: usize, fd: c_int) -> Option<u64> {
            self.write_zeros(self.snapshot_bytes, fd)
        }

        fn read_model_identity(&self) -> [u8; IDENTITY_BYTES] {
            [3; IDENTITY_BYTES]
        }

        fn read_rows(&self, _positions: usize, _fds: &[c_int]) -> bool {
            true
        }

        fn read_snapshot(&self, _fd: c_int) -> Option<()> {
            Some(())
        }
    }

    /// Opens a store in a new folder named after `name`. Returns the folder,
    /// the store, and a cache that saves states of 100 unshared tokens or more.
    fn open_test_store(name: &str) -> (PathBuf, Store, Cache) {
        let dir = std::env::temp_dir().join(format!("eremite-disk-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let (store, _) = Store::open(&dir).expect("the store");
        let mut cache = Cache::new(vec![crate::state_cache::ModelLimits {
            bytes: 1 << 30,
            states: 16,
        }]);
        cache.min_unshared_tokens = 100;
        cache.saved_byte_limit = 1 << 30;
        (dir, store, cache)
    }

    /// Writes saved state `id` of model 0 to `dir`, and adds it to `cache`. It
    /// holds `tokens` in one part of `row_bytes` bytes. It has a generation-end
    /// snapshot of `snapshot_bytes` bytes.
    fn write_saved_state(
        dir: &Path,
        cache: &mut Cache,
        id: SavedStateId,
        tokens: std::ops::Range<i32>,
        row_bytes: u64,
        snapshot_bytes: u64,
    ) {
        let len = tokens.len();
        let saved_state = SavedState {
            model: Some(0),
            tokens: tokens.collect(),
            parts: vec![Part {
                from: 0,
                to: len,
                bytes: row_bytes,
                owner: id,
                file_number: 0,
            }],
            snapshots: vec![SavedSnapshot {
                snapshot: Snapshot {
                    pos: len,
                    kind: SnapshotKind::GenerationEnd,
                    bytes: snapshot_bytes,
                    stale_rows: 0,
                },
                version: 1,
            }],
            replaced_positions: 0,
            last_used: 0,
            version: 1,
        };
        fs::write(
            dir.join(format_index_file_name(id)),
            encode_index(&[3; IDENTITY_BYTES], &saved_state),
        )
        .unwrap();
        fs::write(
            dir.join(format_part_file_name(&saved_state.parts[0])),
            vec![0u8; row_bytes as usize],
        )
        .unwrap();
        fs::write(
            dir.join(format_snapshot_file_name(id, &saved_state.snapshots[0])),
            vec![0u8; snapshot_bytes as usize],
        )
        .unwrap();
        cache.add_saved_state(id, saved_state);
    }

    /// Makes an idle entry of model 0 that holds `tokens` and was last used at
    /// `last_used`.
    fn make_idle_entry(tokens: std::ops::Range<i32>, last_used: u64) -> Entry {
        Entry {
            model: 0,
            tokens: tokens.collect(),
            busy: false,
            last_used,
            bytes: 0,
            stale_rows: 0,
            snapshots: Vec::new(),
            reserved_snapshots: Vec::new(),
            number: 1,
            saved_link: None,
        }
    }

    #[test]
    fn a_saved_state_deletes_the_one_it_covers_and_is_found_at_the_next_start() {
        let (dir, mut store, mut cache) = open_test_store("save");
        let io = FakeState {
            row_bytes: 300,
            snapshot_bytes: 500,
            fails_writes: false,
        };
        let entry = make_idle_entry(0..500, 100);
        let line = store.save_state(&mut cache, &io, &|_| None, &entry, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("saved to the SSD as 1:"));
        // the same tokens and 300 more: it branches from saved state 1's part
        // file and covers saved state 1
        let entry = make_idle_entry(0..800, 200);
        // whole seconds, so that the file system keeps the time exactly
        let last_used_time = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);
        let line = store.save_state(&mut cache, &io, &|_| None, &entry, last_used_time, false);
        assert!(
            line.unwrap_or_default()
                .contains("saved to the SSD as 2: 800 tokens (from position 500)")
        );
        assert!(
            !dir.join("1.state").exists()
                && !dir.join("1-1-500.snap").exists()
                && dir.join("1-0.rows").exists(),
            "the covered state's index and snapshot are deleted while the shared prefix part stays"
        );
        assert_eq!(
            (store.save_count, store.bytes_written),
            (2, 2 * (300 + 500))
        );
        drop(store);
        let (_, found) = Store::open(&dir).expect("the store");
        assert_eq!(found.len(), 1);
        assert_eq!(
            (
                found[0].id,
                found[0].identity,
                found[0].saved_state.tokens.len(),
                found[0].saved_state.parts.len(),
                found[0].saved_state.count_bytes(),
                found[0].last_used_time
            ),
            (2, [3; IDENTITY_BYTES], 800, 2, 600 + 500, last_used_time)
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sibling_branches_share_prefix_part_files_across_deletions_and_restarts() {
        /// A test state whose written row bytes scale with `(to - from)`.
        struct ProportionalState {
            /// Bytes written per token.
            bytes_per_token: u64,
            /// Bytes written per snapshot.
            snapshot_bytes: u64,
        }

        impl StateIo for ProportionalState {
            type SnapshotHandle = ();

            fn write_rows(&self, from: usize, to: usize, fd: c_int) -> Option<u64> {
                let bytes = (to - from) as u64 * self.bytes_per_token;
                if fd < 0 {
                    return Some(bytes);
                }
                let mut file = std::mem::ManuallyDrop::new(unsafe {
                    <File as std::os::fd::FromRawFd>::from_raw_fd(fd)
                });
                std::io::Write::write_all(&mut *file, &vec![0u8; bytes as usize])
                    .ok()
                    .map(|_| bytes)
            }

            fn write_snapshot(&self, _snapshot: Option<()>, _pos: usize, fd: c_int) -> Option<u64> {
                if fd < 0 {
                    return Some(self.snapshot_bytes);
                }
                let mut file = std::mem::ManuallyDrop::new(unsafe {
                    <File as std::os::fd::FromRawFd>::from_raw_fd(fd)
                });
                std::io::Write::write_all(&mut *file, &vec![0u8; self.snapshot_bytes as usize])
                    .ok()
                    .map(|_| self.snapshot_bytes)
            }

            fn read_model_identity(&self) -> [u8; IDENTITY_BYTES] {
                [5; IDENTITY_BYTES]
            }

            fn read_rows(&self, _positions: usize, fds: &[c_int]) -> bool {
                !fds.is_empty()
            }

            fn read_snapshot(&self, _fd: c_int) -> Option<()> {
                Some(())
            }
        }

        let (dir, mut store, mut cache) = open_test_store("branch-share");
        let io = ProportionalState {
            bytes_per_token: 4,
            snapshot_bytes: 50,
        };
        // branch A: 1000 shared prefix tokens + 200 branch-A tokens
        let mut branch_a: Vec<i32> = (0..1000).collect();
        branch_a.extend(10_000..10_200);
        let entry_a = Entry {
            tokens: branch_a,
            ..make_idle_entry(0..0, 100)
        };
        let line_a = store.save_state(
            &mut cache,
            &io,
            &|_| None,
            &entry_a,
            SystemTime::now(),
            false,
        );
        assert!(
            line_a
                .unwrap_or_default()
                .contains("saved to the SSD as 1:")
        );
        assert_eq!(store.bytes_written, 1200 * 4 + 50);

        // branch B: same 1000 prefix tokens + 300 branch-B tokens
        let mut branch_b: Vec<i32> = (0..1000).collect();
        branch_b.extend(20_000..20_300);
        let entry_b = Entry {
            tokens: branch_b,
            ..make_idle_entry(0..0, 200)
        };
        let line_b = store.save_state(
            &mut cache,
            &io,
            &|_| None,
            &entry_b,
            SystemTime::now(),
            false,
        );
        assert!(
            line_b
                .unwrap_or_default()
                .contains("saved to the SSD as 2: 1300 tokens (from position 1000)")
        );
        // branch B wrote only its 300-token suffix and its snapshot
        assert_eq!(store.bytes_written, (1200 * 4 + 50) + (300 * 4 + 50));
        // deduplicated total counts 1-0.rows once
        assert_eq!(cache.count_saved_bytes(), 1200 * 4 + 300 * 4 + 2 * 50);
        assert!(dir.join("1-0.rows").exists() && dir.join("2-0.rows").exists());

        // deleting branch A keeps 1-0.rows alive because branch B still uses it
        store.delete_saved_state(&mut cache, 1);
        assert!(!dir.join("1.state").exists());
        assert!(dir.join("1-0.rows").exists());
        assert_eq!(cache.count_saved_bytes(), 1200 * 4 + 300 * 4 + 50);

        // restarting keeps 1-0.rows and reloads branch B cleanly
        drop(store);
        let (mut store, found) = Store::open(&dir).expect("the store");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, 2);
        assert_eq!(found[0].saved_state.parts.len(), 2);
        assert_eq!(found[0].saved_state.parts[0].owner, 1);
        assert_eq!(found[0].saved_state.parts[1].owner, 2);
        let mut cache2 = Cache::new(vec![crate::state_cache::ModelLimits {
            bytes: 1 << 30,
            states: 16,
        }]);
        cache2.add_saved_state(found[0].id, found[0].saved_state.clone());
        assert!(
            store
                .load_saved_state(&mut cache2, &io, 2, Some(1300), 1300)
                .is_ok()
        );

        // deleting branch B now removes both 1-0.rows and 2-0.rows
        store.delete_saved_state(&mut cache2, 2);
        assert!(!dir.join("1-0.rows").exists() && !dir.join("2-0.rows").exists());
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_save_takes_no_room_from_other_saved_states() {
        let (dir, mut store, mut cache) = open_test_store("failed-save");
        // used before the state
        write_saved_state(&dir, &mut cache, 7, 5000..5400, 800, 200);
        cache.saved_byte_limit = 1100;
        let entry = make_idle_entry(0..500, 100);
        let failing = FakeState {
            row_bytes: 300,
            snapshot_bytes: 500,
            fails_writes: true,
        };
        let line = store.save_state(
            &mut cache,
            &failing,
            &|_| None,
            &entry,
            SystemTime::now(),
            false,
        );
        assert!(line.unwrap_or_default().contains("not saved"));
        assert!(
            cache.saved_states.contains_key(&7) && dir.join("7.state").exists(),
            "the older saved state stays"
        );
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_append_and_a_replacement_count_what_they_replace_as_room() {
        let (dir, mut store, mut cache) = open_test_store("room");
        write_saved_state(&dir, &mut cache, 4, 0..1000, 1000, 500);
        // used before the states saved below
        write_saved_state(&dir, &mut cache, 5, 5000..5400, 800, 200);
        // the new part fits once the saved state's old end snapshot is deleted
        cache.saved_byte_limit = 1500 + 1000 + 300;
        let mut entry = make_idle_entry(0..1300, 100);
        entry.saved_link = Some(SavedLink {
            saved_id: 4,
            version: 1,
            shared_positions: 1000,
        });
        let io = FakeState {
            row_bytes: 300,
            snapshot_bytes: 500,
            fails_writes: false,
        };
        let line = store.save_state(&mut cache, &io, &|_| None, &entry, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("saved to the SSD as 4"));
        assert!(
            cache.saved_states.contains_key(&5),
            "the other saved state stays"
        );
        assert_eq!(cache.count_saved_bytes(), 1000 + 300 + 500 + 1000);
        assert!(
            !dir.join("4-1-1000.snap").exists(),
            "the old end snapshot is deleted"
        );
        // its replaced positions now outnumber the state's tokens. A new saved
        // state replaces it, in its room.
        cache.saved_states.get_mut(&4).unwrap().replaced_positions = 5000;
        let mut entry = make_idle_entry(0..1500, 200);
        entry.saved_link = Some(SavedLink {
            saved_id: 4,
            version: 2,
            shared_positions: 1300,
        });
        let line = store.save_state(&mut cache, &io, &|_| None, &entry, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("saved to the SSD"));
        assert!(
            cache.saved_states.contains_key(&5) && !cache.saved_states.contains_key(&4),
            "it replaced 4, not 5"
        );
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_save_deletes_the_least_recently_used_saved_state_to_make_room() {
        let (dir, mut store, mut cache) = open_test_store("make-room");
        write_saved_state(&dir, &mut cache, 7, 5000..5400, 800, 200);
        write_saved_state(&dir, &mut cache, 8, 6000..6400, 800, 200);
        // room for one of them and the new saved state
        cache.saved_byte_limit = 1000 + 300 + 500;
        let entry = make_idle_entry(0..500, 100);
        let io = FakeState {
            row_bytes: 300,
            snapshot_bytes: 500,
            fails_writes: false,
        };
        let line = store.save_state(&mut cache, &io, &|_| None, &entry, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("saved to the SSD as 1:"));
        assert!(
            !cache.saved_states.contains_key(&7) && !dir.join("7.state").exists(),
            "saved state 7, used first, is deleted"
        );
        assert!(cache.saved_states.contains_key(&8), "saved state 8 stays");
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_save_is_refused_when_newer_saved_states_fill_the_limit() {
        let (dir, mut store, mut cache) = open_test_store("no-room");
        write_saved_state(&dir, &mut cache, 7, 5000..5400, 800, 200);
        // room for the new saved state alone
        cache.saved_byte_limit = 300 + 500;
        // used before saved state 7
        let entry = make_idle_entry(0..500, 0);
        let io = FakeState {
            row_bytes: 300,
            snapshot_bytes: 500,
            fails_writes: false,
        };
        let line = store.save_state(&mut cache, &io, &|_| None, &entry, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("used more recently"));
        assert!(
            cache.saved_states.contains_key(&7) && !dir.join("1.state").exists(),
            "saved state 7 stays, and nothing is written"
        );
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_saved_state_that_cannot_be_opened_now_is_not_damaged() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, mut store, mut cache) = open_test_store("load-errors");
        write_saved_state(&dir, &mut cache, 4, 0..1000, 1000, 500);
        let io = FakeState {
            row_bytes: 0,
            snapshot_bytes: 0,
            fails_writes: false,
        };
        let part = dir.join("4-0.rows");
        fs::set_permissions(&part, fs::Permissions::from_mode(0o000)).unwrap();
        let read = store.load_saved_state(&mut cache, &io, 4, Some(1000), 1000);
        fs::set_permissions(&part, fs::Permissions::from_mode(FILE_MODE)).unwrap();
        assert!(matches!(read, Err(LoadError::Unavailable(_))), "{read:?}");
        assert!(
            store
                .load_saved_state(&mut cache, &io, 4, Some(1000), 1000)
                .is_ok()
        );
        fs::remove_file(&part).unwrap();
        let read = store.load_saved_state(&mut cache, &io, 4, Some(1000), 1000);
        assert!(
            matches!(read, Err(LoadError::Damaged(_))),
            "a missing part: {read:?}"
        );
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_index_of_another_format_version_is_kept_with_its_files() {
        let dir = std::env::temp_dir().join(format!("eremite-disk-version-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut index = encode_index(&[1; IDENTITY_BYTES], &make_saved_state(100));
        // a later server's
        index[FORMAT_VERSION_AT..FORMAT_VERSION_AT + size_of::<u32>()]
            .copy_from_slice(&(INDEX_FORMAT_VERSION + 1).to_le_bytes());
        fs::write(dir.join("9.state"), &index).unwrap();
        fs::write(dir.join("9-0.rows"), vec![0u8; 1000]).unwrap();
        fs::write(dir.join("9-7-100.snap"), vec![0u8; 77]).unwrap();
        let (store, found) = Store::open(&dir).expect("the store");
        assert!(found.is_empty(), "this server does not read it");
        assert!(store.next_id >= 10, "nor reuses its number");
        for name in ["9.state", "9-0.rows", "9-7-100.snap"] {
            assert!(dir.join(name).exists(), "{name} is kept");
        }
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn opening_keeps_what_an_index_names_and_deletes_only_the_store_s_other_files() {
        let dir = std::env::temp_dir().join(format!("eremite-disk-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let saved_state = make_saved_state(100);
        fs::write(
            dir.join("5.state"),
            encode_index(&[1; IDENTITY_BYTES], &saved_state),
        )
        .unwrap();
        fs::write(dir.join("5-0.rows"), vec![0u8; 1000]).unwrap();
        fs::write(dir.join("5-1.rows"), vec![0u8; 500]).unwrap();
        fs::write(dir.join("5-3-100.snap"), vec![0u8; 77]).unwrap();
        // named by no index
        fs::write(dir.join("5-2-100.snap"), b"an older save's").unwrap();
        fs::write(dir.join("6-0.rows.tmp"), b"a crash's").unwrap();
        fs::write(dir.join("7.state"), b"damaged").unwrap();
        fs::write(dir.join("notes.txt"), b"the user's").unwrap();
        {
            let (store, found) = Store::open(&dir).expect("the store");
            assert_eq!(store.get_folder(), dir);
            assert_eq!(found.len(), 1);
            assert_eq!((found[0].id, found[0].identity), (5, [1; IDENTITY_BYTES]));
            assert!(store.next_id >= 8);
            assert!(
                Store::open(&dir).is_err(),
                "a second server is refused while the first holds the lock"
            );
        }
        let mut names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|folder_entry| folder_entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "5-0.rows",
                "5-1.rows",
                "5-3-100.snap",
                "5.state",
                "lock",
                "notes.txt"
            ]
        );
        // a part shorter than its index says: the saved state is deleted
        fs::write(dir.join("5-1.rows"), vec![0u8; 499]).unwrap();
        let (_, found) = Store::open(&dir).expect("the store");
        assert!(found.is_empty());
        assert!(!dir.join("5.state").exists() && !dir.join("5-0.rows").exists());
        assert!(dir.join("notes.txt").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
