//! The state cache of all models: which states and snapshots stay in memory,
//! and how a prompt reuses them.
//!
//! A state holds what a model keeps of a conversation. For DeepSeek, it holds
//! the rows of each layer's attention keys and values for each position. The
//! rows stay in the engine's states. The cache holds only data about them. For
//! each state it holds the model, the token ids of the positions whose rows the
//! state holds, and the state's bytes. It also holds the state's snapshots,
//! whether a request runs in the state, and the state's last use.
//!
//! All models share one byte limit and one use clock.
//!
//! A state's rolling rows are the rows that later positions overwrite: each
//! layer's window ring. A snapshot holds a state's rolling rows at one
//! position. After a snapshot is restored into a state, the state continues at
//! the snapshot's position.

use std::collections::BTreeMap;

/// The id of a state in memory. The worker maps it to the engine's state.
pub type StateId = u64;

/// Where a snapshot was taken. Saved states' index files store these numbers.
/// They never change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SnapshotKind {
    /// At a prompt's end, with the logits its next token is sampled from.
    PromptEnd = 0,
    /// At a generation's end.
    GenerationEnd = 1,
    /// At a branch point, taken during a prefill. The branch point is where a
    /// replay starts, or the end of the longest prefix that the prompt shares
    /// with any state.
    Branch = 2,
}

/// A snapshot as the cache sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapshot {
    /// The position whose rolling rows the snapshot holds.
    pub pos: usize,
    /// Where the snapshot was taken.
    pub kind: SnapshotKind,
    /// The bytes the snapshot takes.
    pub bytes: u64,
    /// The stale rows in the snapshot's rings at `pos`.
    pub stale_rows: usize,
}

/// One attention state as the cache sees it.
#[derive(Debug)]
pub struct Entry {
    /// The model that owns the state. Only this model's prompts reuse it.
    pub model: usize,
    /// The tokens whose rows the state holds, from position 0.
    pub tokens: Vec<i32>,
    /// Whether a request runs in the state. Other requests may copy its rows
    /// before its end. No other request continues it in place.
    pub busy: bool,
    /// The state's last use, on the cache's use clock. Eviction takes the least
    /// recently used first.
    pub last_used: u64,
    /// The bytes the state's rows take, without its snapshots.
    pub bytes: u64,
    /// The stale rows in the state's rings. The state can continue at most the
    /// rewind margin less this many positions before its end.
    pub stale_rows: usize,
    /// The state's snapshots, in position order.
    pub snapshots: Vec<Snapshot>,
    /// The snapshots that the request's prefill will take. The cache counts
    /// their bytes before the engine allocates them.
    pub reserved_snapshots: Vec<Snapshot>,
    /// Its number in the log and the status.
    pub number: u64,
}

/// The state cache of all models: their states and snapshots.
#[derive(Debug)]
pub struct Cache {
    /// The states in memory, by id.
    pub entries: BTreeMap<StateId, Entry>,
    /// The bytes that states and snapshots may take (--state-cache).
    pub byte_limit: u64,
    /// The most states that the cache keeps (--max-cached-states).
    pub max_states: usize,
    /// The use clock. Each use of a state advances it by one.
    clock: u64,
    /// The id that the next new state gets.
    next_id: StateId,
}

impl Cache {
    /// Returns a new, empty cache with these limits.
    pub fn new(byte_limit: u64, max_states: usize) -> Self {
        Cache {
            entries: BTreeMap::new(),
            byte_limit,
            max_states,
            clock: 0,
            // Ids start at 1. They are the state numbers that the log and the
            // status show.
            next_id: 1,
        }
    }

    /// Counts the bytes of all states and snapshots, reserved snapshots
    /// included.
    pub fn count_bytes(&self) -> u64 {
        self.entries
            .values()
            .map(|entry| {
                entry.bytes
                    + entry
                        .snapshots
                        .iter()
                        .chain(&entry.reserved_snapshots)
                        .map(|snapshot| snapshot.bytes)
                        .sum::<u64>()
            })
            .sum()
    }

    /// Advances the use clock by one and returns its new time.
    fn advance_clock(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Adds an empty, busy state of model `model` that takes `bytes`. Returns
    /// its id.
    pub fn add_state(&mut self, model: usize, bytes: u64) -> StateId {
        let id = self.next_id;
        self.next_id += 1;
        let now = self.advance_clock();
        self.entries.insert(
            id,
            Entry {
                model,
                tokens: Vec::new(),
                busy: true,
                last_used: now,
                bytes,
                stale_rows: 0,
                snapshots: Vec::new(),
                reserved_snapshots: Vec::new(),
                number: id,
            },
        );
        id
    }

    /// Adds `snapshot` to state `id`. It replaces any snapshot at the same
    /// position. It removes the reserved snapshot of the same position and
    /// kind.
    pub fn add_snapshot(&mut self, id: StateId, snapshot: Snapshot) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry
                .reserved_snapshots
                .retain(|reserved| reserved.pos != snapshot.pos || reserved.kind != snapshot.kind);
            entry.snapshots.retain(|kept| kept.pos != snapshot.pos);
            entry.snapshots.push(snapshot);
            entry.snapshots.sort_by_key(|kept| kept.pos);
        }
    }

    /// Sets the reserved snapshots of state `id`: the snapshots that its
    /// request's prefill will take.
    pub fn set_reserved_snapshots(&mut self, id: StateId, snapshots: Vec<Snapshot>) {
        self.entries
            .get_mut(&id)
            .expect("an admitted state")
            .reserved_snapshots = snapshots;
    }

    /// Clears the reserved snapshots of state `id`.
    pub fn clear_reserved_snapshots(&mut self, id: StateId) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.reserved_snapshots.clear();
        }
    }

    /// Sets state `id` idle and used now. Sets its bytes to `bytes` and clears
    /// its reserved snapshots.
    pub fn set_idle(&mut self, id: StateId, bytes: u64) {
        let now = self.advance_clock();
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.busy = false;
            entry.last_used = now;
            entry.bytes = bytes;
            entry.reserved_snapshots.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_snapshots_count_once_toward_the_shared_limit() {
        let mut cache = Cache::new(8, 16);
        let admitted = cache.add_state(0, 5);
        let prompt_end = Snapshot {
            pos: 100,
            kind: SnapshotKind::PromptEnd,
            bytes: 2,
            stale_rows: 0,
        };
        cache.set_reserved_snapshots(admitted, vec![prompt_end]);
        assert_eq!(cache.count_bytes(), 7);
        cache.add_snapshot(admitted, prompt_end);
        assert_eq!(
            cache.count_bytes(),
            7,
            "an added reserved snapshot counts once"
        );
        assert!(cache.entries[&admitted].reserved_snapshots.is_empty());
        cache.set_reserved_snapshots(
            admitted,
            vec![Snapshot {
                pos: 50,
                kind: SnapshotKind::Branch,
                bytes: 1,
                stale_rows: 0,
            }],
        );
        cache.set_idle(admitted, 5);
        assert_eq!(
            cache.count_bytes(),
            7,
            "an idle state has no reserved snapshots"
        );
    }

    #[test]
    fn cleared_reserved_snapshots_stop_counting_and_the_state_stays_busy() {
        let mut cache = Cache::new(8, 16);
        let admitted = cache.add_state(0, 5);
        cache.set_reserved_snapshots(
            admitted,
            vec![Snapshot {
                pos: 100,
                kind: SnapshotKind::PromptEnd,
                bytes: 2,
                stale_rows: 0,
            }],
        );
        cache.clear_reserved_snapshots(admitted);
        assert_eq!(cache.count_bytes(), 5);
        assert!(cache.entries[&admitted].busy);
    }
}
