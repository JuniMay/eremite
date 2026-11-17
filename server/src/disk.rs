//! Saved states: states written to the SSD. This module saves states and loads
//! them back. Their files lie in one folder, set by state_dir.
//!
//! A saved state `ID` has an index file, `ID.state`, and the files that the
//! index names. Its parts are `ID-N.rows`, numbered from 0. Its snapshots are
//! `ID-V-POS.snap`. V is the save that wrote the snapshot. POS is its position.
//! The index holds the model's identity and the tokens. For each part it holds
//! the positions and the bytes. For each snapshot it holds the position, the
//! kind, the stale rows and the bytes. A SHA-256 sum in the index covers every
//! byte after it.
//!
//! The files have mode 0600 and the folder 0700: only their owner may use them.
//! One server at a time uses the folder. It holds a lock on the folder's `lock`
//! file. Another server runs without saved states.

use crate::engine;
use crate::state_cache::{Part, SavedSnapshot, SavedState, SavedStateId, Snapshot, SnapshotKind};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// The first bytes of every index.
const INDEX_MAGIC: &[u8; 8] = b"EREMITEI";
/// The index format's version. A server loads only indexes of its own format
/// version.
const INDEX_FORMAT_VERSION: u32 = 1;
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
/// A part's record: from, to, bytes, file number and a reserved word.
const PART_RECORD_BYTES: usize = 24;
/// A snapshot's record: position, kind, 3 reserved bytes, stale rows, a
/// reserved word, bytes and version.
const SNAPSHOT_RECORD_BYTES: usize = 32;
/// A token: one i32.
const TOKEN_BYTES: usize = 4;
/// The most digits of a number in a file name. Every 19-digit number fits in a
/// u64.
const MAX_NUMBER_DIGITS: usize = 19;
/// Only the owner may list, read and write the folder.
const FOLDER_MODE: u32 = 0o700;
/// Only the owner may read and write the files.
const FILE_MODE: u32 = 0o600;

/// The saved states' folder, locked for this server, and counts of this run's
/// loads, saves and bytes written.
pub struct Store {
    /// The folder that holds the saved states' files.
    folder: PathBuf,
    /// The open `lock` file. The lock lasts while the store exists.
    _lock: File,
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

/// Formats the file name of part `part` of saved state `id`: `ID-N.rows`.
fn format_part_file_name(id: SavedStateId, part: &Part) -> String {
    format!("{id}-{}.rows", part.file_number)
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
    // a part's record: from at 0, to at 4, bytes at 8, file number at 16, a
    // reserved word at 20
    for _ in 0..part_count {
        saved_state.parts.push(Part {
            from: read_u32_at(at)? as usize,
            to: read_u32_at(at + 4)? as usize,
            bytes: read_u64_at(at + 8)?,
            file_number: read_u32_at(at + 16)?,
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
        for folder_entry in fs::read_dir(dir)
            .map_err(|e| format!("cannot read {shown}: {e}"))?
            .flatten()
        {
            let name = folder_entry.file_name().to_string_lossy().into_owned();
            match parse_file_name(&name) {
                Some((FileName::Index(id), false)) => indexes.push(id),
                Some((FileName::Index(id) | FileName::Part(id) | FileName::Snapshot(id), _)) => {
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
                    .all(|part| file_size(format_part_file_name(id, part)) == Some(part.bytes))
                    && saved_state.snapshots.iter().all(|saved_snapshot| {
                        file_size(format_snapshot_file_name(id, saved_snapshot))
                            == Some(saved_snapshot.snapshot.bytes)
                    })
            };
            match decode_index(&index) {
                Some((identity, saved_state)) if has_whole_files(&saved_state) => {
                    named.extend(
                        saved_state
                            .parts
                            .iter()
                            .map(|part| format_part_file_name(id, part)),
                    );
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    file_number: 0,
                },
                Part {
                    from: tokens / 2 - 1,
                    to: tokens,
                    bytes: 500,
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
