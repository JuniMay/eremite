//! The next web turn of each session.
use std::collections::BTreeMap;
use std::fs::{self, DirBuilder, File};
use std::io::{self, ErrorKind};
use std::os::unix::fs::DirBuilderExt;
use std::path::PathBuf;

use crate::disk::Store;

/// The file of session IDs and their next turn numbers.
const COUNTER_FILE: &str = "turns.json";
/// Only the owner may list or change the search folder.
const FOLDER_MODE: u32 = 0o700;

/// The next turn of each session, kept after its pages expire.
#[derive(Default)]
pub(super) struct TurnCounters {
    /// The next unused turn by session ID.
    next: BTreeMap<String, u64>,
    /// The folder for counters that must survive a restart.
    folder: Option<PathBuf>,
    /// The open folder. Its lock keeps other writers out.
    lock: Option<File>,
}

impl TurnCounters {
    /// Keeps the counters in the given folder.
    pub(super) fn in_folder(folder: PathBuf) -> Self {
        Self {
            folder: Some(folder),
            ..Self::default()
        }
    }

    /// Reserves one turn, past any turns copied from the parent.
    pub(super) fn reserve(&mut self, id: &str, parent: Option<&str>) -> io::Result<u64> {
        if let Some(folder) = &self.folder
            && self.lock.is_none()
        {
            DirBuilder::new()
                .recursive(true)
                .mode(FOLDER_MODE)
                .create(folder)?;
            let lock = File::open(folder)?;
            lock.try_lock().map_err(|error| match error {
                fs::TryLockError::WouldBlock => io::Error::new(
                    ErrorKind::WouldBlock,
                    format!("another server uses search_dir {}", folder.display()),
                ),
                fs::TryLockError::Error(error) => error,
            })?;
            self.next = match fs::read(folder.join(COUNTER_FILE)) {
                Ok(bytes) => serde_json::from_slice(&bytes)?,
                Err(e) if e.kind() == ErrorKind::NotFound => BTreeMap::new(),
                Err(e) => return Err(e),
            };
            self.lock = Some(lock);
        }
        // A fork with an unseen parent may have copied any earlier web turn.
        let turn = self
            .next
            .get(id)
            .or_else(|| {
                let parent = parent?;
                self.next.get(parent).or_else(|| self.next.values().max())
            })
            .copied()
            .unwrap_or(0);
        let next = turn
            .checked_add(1)
            .ok_or_else(|| io::Error::other("web turn counter is full"))?;
        self.next.insert(id.to_owned(), next);
        if let (Some(folder), Some(lock)) = (&self.folder, &self.lock) {
            let bytes = serde_json::to_vec(&self.next)?;
            let temporary =
                Store::create_temporary_file(folder, COUNTER_FILE).map_err(io::Error::other)?;
            Store::replace_file(temporary, &bytes).map_err(io::Error::other)?;
            lock.sync_all()?;
        }
        Ok(turn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TestDir;

    /// Session and fork turns survive a reload.
    #[test]
    fn reserves_session_and_fork_turns_across_restarts() -> io::Result<()> {
        let dir = TestDir::new("fork-turns");
        let open = || TurnCounters::in_folder(dir.0.clone());
        let mut turns = open();
        assert_eq!(turns.reserve("parent", None)?, 0);
        assert_eq!(turns.reserve("child", Some("parent"))?, 1);
        assert_eq!(turns.reserve("child", Some("parent"))?, 2);
        drop(turns);
        let mut turns = open();
        assert_eq!(turns.reserve("sibling", Some("parent"))?, 1);
        assert_eq!(turns.reserve("nested", Some("unseen"))?, 3);
        assert_eq!(turns.reserve("independent", None)?, 0);
        assert_eq!(turns.reserve("independent", Some("nested"))?, 1);
        assert_eq!(turns.reserve("child", None)?, 3);
        let error = open().reserve("other writer", None).unwrap_err();
        assert!(
            error.to_string().contains("another server uses search_dir"),
            "{error}"
        );
        Ok(())
    }
}
