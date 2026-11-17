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
//! All models share one byte limit and one use clock. A prompt reuses only
//! states of its own model. Eviction picks from all models' states, the least
//! recently used first.
//!
//! A state's rolling rows are the rows that later positions overwrite: each
//! layer's window ring. A snapshot holds a state's rolling rows at one
//! position. After a snapshot is restored into a state, the state continues at
//! the snapshot's position.
//!
//! [`Cache::list_plans`] compares a prompt with every state, busy or idle. For
//! each state, `k` is the length of the prefix that the prompt shares with it.
//! The branch point is the end of the longest prefix that the prompt shares
//! with any state. The plans are:
//!
//! - **In place**: the prompt continues an idle state without a copy. The
//!   state's end lies at most its rewind room after `k`.
//! - **Snapshot**: a new state copies a source state's rows before a position
//!   `b`, at most `k`. It restores a snapshot at `b`, or one at most the
//!   snapshot's rewind room after `b`. Its prefill starts at `b`. When the
//!   branch point lies after `b` and before the prompt's end, the prefill takes
//!   a branch snapshot there. A later fork there then resumes at once. The
//!   rewind room after `b` serves the turn after a thinking prompt. That prompt
//!   ends with `<think>`. The next turn has `</think>` at that position. So the
//!   next turn differs one position before the prompt-end snapshot. It still
//!   resumes from that snapshot.
//! - **Replay**: a new state copies a source state's rows before `k`. Its
//!   prefill starts at `k` with the engine's replay. The replay rebuilds the
//!   rolling rows at `k`. The prefill takes them as a branch snapshot.
//! - **Cold start**: the prompt shares nothing. A new state prefills it from
//!   position 0.
//!
//! A snapshot plan and a replay plan fork the source state. Every plan gives
//! exact rows, except a replay under --ced. The --ced option is DeepSeek's
//! approximate prefill. Under it, a replay recomputes only the last ring of
//! positions.
//!
//! **Rewind room.** Before its window, each layer's ring also holds the rows of
//! a few more positions: the rewind margin. So a state can continue up to that
//! many positions before its end. A row written past where the state then
//! continues is stale. It takes the place of one of the margin's oldest rows. A
//! verify pass writes stale rows after the tokens it accepts. A prompt that
//! continues a state before its end also leaves stale rows: the state's old
//! rows after the prompt's end. The cache counts the stale rows of each state
//! and snapshot. Each row written at a state's end replaces one stale row. The
//! rewind room is the rewind margin less the stale rows. A plan resumes at most
//! that many positions before a state's end or a snapshot's position.
//!
//! The worker carries out a plan with engine calls. Then it tells the cache
//! what changed: the states, snapshots and tokens it added, and the rows that a
//! pass rejected. The cache holds no engine handles. So its tests need no
//! model.
//!
//! **Saved states** are states written to the SSD by server/src/disk.rs. The
//! cache holds their tokens, their parts' positions and their snapshots. A load
//! plan takes a saved state as its source, as the other plans take a state in
//! memory. The new state loads the saved rows before the plan's start. Then a
//! snapshot is restored into it, or a replay rebuilds the rolling rows. At
//! equal cost, a state in memory and a cold start go before the SSD.
//!
//! A state's unshared tokens are those after the longest prefix that it shares
//! with another state in memory or a saved state of its model. Saving an
//! evicted state writes it only when it has at least
//! [`Cache::min_unshared_tokens`] unshared tokens. A state may have a saved
//! link: its rows before some position equal a saved state's rows. Then the
//! state was loaded from that saved state, or copied from a state that was.
//! Saving such a state writes only its rows from that position. They become one
//! more part of the saved state.

use std::collections::BTreeMap;

/// How many tokens one slice comparison takes. Slices compare faster than
/// single tokens.
const SHARED_PREFIX_BLOCK: usize = 1024;
/// The default of --state-disk-tokens. The worker sets the configured value.
const DEFAULT_MIN_UNSHARED_TOKENS: usize = 8192;
/// The most parts a saved state has. A load opens one file for each part.
const MAX_PARTS: usize = 64;

// The order of plans at equal cost. The cost counts only prefilled tokens. A
// copy takes time it does not count. So in place comes first. An SSD read also
// takes time it does not count. So the SSD comes last.
const RANK_IN_PLACE: usize = 0;
const RANK_SNAPSHOT: usize = 1;
const RANK_REPLAY: usize = 2;
const RANK_COLD: usize = 3;
const RANK_LOAD_SNAPSHOT: usize = 4;
const RANK_LOAD_REPLAY: usize = 5;

/// The id of a state in memory. The worker maps it to the engine's state.
pub type StateId = u64;

/// The id of a saved state. Its files' names start with it.
pub type SavedStateId = u64;

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

impl Snapshot {
    /// Returns whether the snapshot has logits. Only a prompt-end snapshot has
    /// them.
    pub fn has_logits(&self) -> bool {
        self.kind == SnapshotKind::PromptEnd
    }
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
    /// The link to a saved state whose rows equal this state's rows before some
    /// position.
    pub saved_link: Option<SavedLink>,
}

/// A link from a state to a saved state. The state was loaded from the saved
/// state, or copied from a state that was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SavedLink {
    /// The saved state.
    pub saved_id: SavedStateId,
    /// The saved state's version when the link was made.
    pub version: u64,
    /// The positions from 0 whose rows the state and the saved state share.
    pub shared_positions: usize,
}

/// One part of a saved state: one file of rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    /// The first position whose rows the part holds.
    pub from: usize,
    /// The position after the last one.
    pub to: usize,
    /// The bytes of the part's file.
    pub bytes: u64,
    /// The N in the part's file name, `ID-N.rows`.
    pub file_number: u32,
}

/// A snapshot of a saved state, in its own file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavedSnapshot {
    /// The snapshot. Its bytes are its file's bytes.
    pub snapshot: Snapshot,
    /// The save that wrote the snapshot's file: the V in `ID-V-POS.snap`.
    pub version: u64,
}

/// A state on the SSD, as the cache sees it.
#[derive(Clone, Debug)]
pub struct SavedState {
    /// The model whose identity the saved state has. It is None when no model
    /// here has that identity. Such a saved state is never loaded. It is
    /// deleted first when room is needed.
    pub model: Option<usize>,
    /// The tokens whose rows it holds, from position 0.
    pub tokens: Vec<i32>,
    /// The saved state's parts, in the order they were written. The last part
    /// ends at the saved state's end.
    pub parts: Vec<Part>,
    /// The saved state's snapshots, at most one at each position.
    pub snapshots: Vec<SavedSnapshot>,
    /// How many positions a later part wrote again. Their old rows still take
    /// room in earlier parts.
    pub replaced_positions: usize,
    /// The saved state's last load or save, on the cache's use clock. When room
    /// is needed, the least recently used saved state is deleted first.
    pub last_used: u64,
    /// The saved state's version: 1 after its first save, and one more after
    /// each append.
    pub version: u64,
}

impl SavedState {
    /// Counts the bytes of the saved state's files.
    pub fn count_bytes(&self) -> u64 {
        self.parts.iter().map(|part| part.bytes).sum::<u64>()
            + self
                .snapshots
                .iter()
                .map(|saved_snapshot| saved_snapshot.snapshot.bytes)
                .sum::<u64>()
    }
}

/// What saving an evicted state writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveDecision {
    /// Writes nothing. Saving is off, or the state has too few unshared tokens.
    Skip,
    /// Writes one more part of saved state `saved_id`: the state's rows from
    /// position `from` to its end.
    Append { saved_id: SavedStateId, from: usize },
    /// Writes a new saved state. Once it is written, saved state `replaces` is
    /// deleted.
    New { replaces: Option<SavedStateId> },
}

/// How a prompt resumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// A cold start: a new state prefills the prompt from position 0.
    Cold,
    /// The idle state `state` continues at position `from`. When `from` is the
    /// prompt's length, its prompt-end snapshot there is restored. No prefill
    /// runs then.
    InPlace { state: StateId, from: usize },
    /// A new state from a snapshot of state `source`. It copies the source's
    /// rows before `from`. It restores the snapshot at `snapshot_pos`: `from`,
    /// or at most the snapshot's rewind room after it. Its prefill then starts
    /// at `from`. No prefill runs when `from` is the prompt's length: the
    /// snapshot's logits are the prompt's. `branch_point` is the end of the
    /// longest prefix that the prompt shares with any state. When it lies after
    /// `from` and before the prompt's end, the prefill takes a branch snapshot
    /// there. A later fork there then resumes at once.
    Snapshot {
        source: StateId,
        snapshot_pos: usize,
        from: usize,
        branch_point: usize,
    },
    /// A new state that replays from state `source`. It copies the source's
    /// rows before `from`. Its prefill starts with a replay at `from`. The
    /// replay rebuilds the rolling rows there. The prefill takes them as a
    /// branch snapshot.
    Replay { source: StateId, from: usize },
    /// A new state from saved state `saved_id`. It loads the saved rows before
    /// `from`. With `snapshot_pos`, it then restores that snapshot, as a
    /// snapshot plan does. Without one, its prefill replays at `from`, as a
    /// replay plan does. `branch_point` is as in a snapshot plan.
    Load {
        saved_id: SavedStateId,
        snapshot_pos: Option<usize>,
        from: usize,
        branch_point: usize,
    },
}

impl Plan {
    /// Counts the prompt tokens that the plan reuses. The prefill starts after
    /// them.
    pub fn count_reused_tokens(&self) -> usize {
        match *self {
            Plan::Cold => 0,
            Plan::InPlace { from, .. } => from,
            Plan::Snapshot { from, .. } | Plan::Replay { from, .. } | Plan::Load { from, .. } => {
                from
            }
        }
    }

    /// Returns whether the plan's prefill starts with a replay.
    pub fn starts_with_replay(&self) -> bool {
        matches!(
            self,
            Plan::Replay { .. }
                | Plan::Load {
                    snapshot_pos: None,
                    ..
                }
        )
    }

    /// Finds the state in memory whose rows the plan uses, if any.
    pub fn find_source_state(&self) -> Option<StateId> {
        match *self {
            Plan::Cold | Plan::Load { .. } => None,
            Plan::InPlace { state, .. } => Some(state),
            Plan::Snapshot { source, .. } | Plan::Replay { source, .. } => Some(source),
        }
    }

    /// Describes the plan for the log in one word. "cold" is a cold start.
    /// "continued" continues a state in place. "repeated" is the same prompt
    /// again, from its prompt-end snapshot. "forked" copies another state.
    /// "loaded" loads a saved state.
    pub fn describe_for_log(&self, prompt_len: usize) -> &'static str {
        match *self {
            Plan::Cold => "cold",
            Plan::InPlace { from, .. } if from == prompt_len => "repeated",
            Plan::InPlace { .. } => "continued",
            Plan::Snapshot { from, .. } if from == prompt_len => "repeated",
            Plan::Snapshot { .. } | Plan::Replay { .. } => "forked",
            Plan::Load { .. } => "loaded",
        }
    }

    /// Finds where the plan's prefill takes a branch snapshot, if anywhere. A
    /// replay takes it where the replay starts. A snapshot plan takes it at the
    /// branch point. It does so only when the branch point lies after `from`
    /// and before the prompt's end. A later fork there then resumes at once. A
    /// load with a snapshot takes it as a snapshot plan does. A load without
    /// one takes it as a replay does.
    pub fn find_branch_snapshot_pos(&self, prompt_len: usize) -> Option<usize> {
        match *self {
            Plan::Replay { from, .. }
            | Plan::Load {
                snapshot_pos: None,
                from,
                ..
            } => Some(from),
            Plan::Snapshot {
                from, branch_point, ..
            }
            | Plan::Load {
                from, branch_point, ..
            } if branch_point > from && branch_point < prompt_len => Some(branch_point),
            _ => None,
        }
    }

    /// Counts the snapshot bytes that a request on this plan adds before it
    /// decodes. A plan that prefills adds a prompt-end snapshot of
    /// `prompt_end_bytes`. A plan without a prefill uses the snapshot at the
    /// prompt's end. A plan with a branch snapshot adds `branch_bytes` more.
    pub fn count_snapshot_bytes(
        &self,
        prompt_len: usize,
        branch_bytes: u64,
        prompt_end_bytes: u64,
    ) -> u64 {
        let prompt_end = if self.count_reused_tokens() < prompt_len {
            prompt_end_bytes
        } else {
            0
        };
        prompt_end
            + if self.find_branch_snapshot_pos(prompt_len).is_some() {
                branch_bytes
            } else {
                0
            }
    }
}

/// The engine's limits and costs that plans depend on, for one model.
#[derive(Clone, Copy)]
pub struct PlanLimits<'a> {
    /// The rewind margin: the positions whose rows each ring holds before its
    /// window. A state can continue this many positions before its end, less
    /// its stale rows. The same holds before a snapshot's position.
    pub rewind_margin: usize,
    /// Whether the model rebuilds its rolling rows from the rows it keeps. A
    /// model that does not resumes only from snapshots, and computes the
    /// positions after the snapshot again.
    pub can_replay: bool,
    /// Returns what a replay at a position costs beyond the prompt's new
    /// tokens, counted in prefilled tokens. The cost is the rows before that
    /// position that each layer computes again, on average. It is at most the
    /// position.
    pub replay_cost: &'a dyn Fn(usize) -> usize,
}

/// Where the rows of a prefill continue from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Origin {
    /// The position that the rows continue from.
    pub pos: usize,
    /// The stale rows at `pos`.
    pub stale_rows: usize,
}

/// Counts the stale rows at `pos` in rows that continue from `origin`. Each row
/// written after the origin replaces one stale row. Each of the origin's old
/// rows after `pos` is one more stale row.
pub fn count_stale_rows_at(origin: Origin, pos: usize) -> usize {
    (origin.stale_rows + origin.pos).saturating_sub(pos)
}

/// Counts the positions that two token sequences share from the start.
pub fn count_shared_prefix(tokens: &[i32], other: &[i32]) -> usize {
    let len = tokens.len().min(other.len());
    let mut shared = 0;
    while shared + SHARED_PREFIX_BLOCK <= len
        && tokens[shared..shared + SHARED_PREFIX_BLOCK]
            == other[shared..shared + SHARED_PREFIX_BLOCK]
    {
        shared += SHARED_PREFIX_BLOCK;
    }
    while shared < len && tokens[shared] == other[shared] {
        shared += 1;
    }
    shared
}

/// What to evict to make room: snapshots first, then whole states.
#[derive(Debug, Default, PartialEq)]
pub struct Eviction {
    /// The snapshots to evict, each as its state and position.
    pub snapshots: Vec<(StateId, usize)>,
    /// The states to evict whole.
    pub states: Vec<StateId>,
}

/// What the cache evicted, for the worker to free: snapshots with their states'
/// ids, and whole states with their entries. The worker may save a whole state
/// before it frees it.
#[derive(Debug, Default)]
pub struct Evicted {
    /// The evicted snapshots, each with its state's id.
    pub snapshots: Vec<(StateId, Snapshot)>,
    /// The evicted whole states, each with its id.
    pub states: Vec<(StateId, Entry)>,
}

/// The state cache of all models: their states and snapshots. It also holds
/// their saved states.
#[derive(Debug)]
pub struct Cache {
    /// The states in memory, by id.
    pub entries: BTreeMap<StateId, Entry>,
    /// The bytes that states and snapshots may take (--state-cache).
    pub byte_limit: u64,
    /// The most states that the cache keeps (--max-cached-states).
    pub max_states: usize,
    /// The saved states, by id.
    pub saved_states: BTreeMap<SavedStateId, SavedState>,
    /// The bytes that saved states may take on the SSD (--state-disk). With 0,
    /// no state is saved.
    pub saved_byte_limit: u64,
    /// The fewest unshared tokens that make a state worth keeping on the SSD
    /// (--state-disk-tokens).
    pub min_unshared_tokens: usize,
    /// The saved state that an admitted request loads next. Until then, no save
    /// appends to it, replaces it or deletes it.
    pub pinned: Option<SavedStateId>,
    /// The saved state that cannot be loaded now. Planning skips it for the
    /// request being planned again.
    pub skipped: Option<SavedStateId>,
    /// The use clock. Each use of a state advances it by one. So does each use
    /// of a saved state.
    clock: u64,
    /// The id that the next new state gets.
    next_id: StateId,
}

impl Cache {
    /// Returns a new, empty cache with these limits. Saving to the SSD is off
    /// until `saved_byte_limit` is set.
    pub fn new(byte_limit: u64, max_states: usize) -> Self {
        Cache {
            entries: BTreeMap::new(),
            byte_limit,
            max_states,
            saved_states: BTreeMap::new(),
            saved_byte_limit: 0,
            min_unshared_tokens: DEFAULT_MIN_UNSHARED_TOKENS,
            pinned: None,
            skipped: None,
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

    /// Counts the bytes of model `model`'s states and snapshots, reserved
    /// snapshots included.
    pub fn count_model_bytes(&self, model: usize) -> u64 {
        self.entries
            .values()
            .filter(|entry| entry.model == model)
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

    /// Finds the cheapest plan that resumes `prompt` for model `model`.
    #[cfg(test)]
    pub fn find_cheapest_plan(&self, model: usize, prompt: &[i32], limits: PlanLimits<'_>) -> Plan {
        self.list_plans(model, prompt, limits)[0]
    }

    /// Lists the plans that resume `prompt` for model `model`, the cheapest
    /// first. The list ends at the cold start. A plan that reuses a state in
    /// memory keeps that state. That state may not fit beside the prompt's new
    /// state. Every plan gives exact rows, except a replay under --ced.
    pub fn list_plans(&self, model: usize, prompt: &[i32], limits: PlanLimits<'_>) -> Vec<Plan> {
        let prompt_len = prompt.len();
        // each plan with its cost and rank
        let mut plans = vec![(prompt_len, RANK_COLD, Plan::Cold)];
        // the branch point: the end of the longest prefix that the prompt
        // shares with any state
        let mut branch_point = 0;
        let mut add_plan = |cost: usize, rank: usize, plan: Plan| plans.push((cost, rank, plan));
        let count_rewind_room = |stale_rows: usize| limits.rewind_margin.saturating_sub(stale_rows);
        // Finds the prefix that a state's tokens share with the prompt, and
        // where the prompt resumes. It resumes at the prefix's end. At the
        // prompt's end it resumes only with logits there. Otherwise the
        // prompt's last token runs again.
        let find_resume_point = |tokens: &[i32], has_logits_at_end: bool| {
            let shared = count_shared_prefix(prompt, tokens);
            (
                shared,
                if shared >= prompt_len && !has_logits_at_end {
                    prompt_len - 1
                } else {
                    shared.min(prompt_len)
                },
            )
        };
        for (&id, saved_state) in &self.saved_states {
            if saved_state.model != Some(model) || Some(id) == self.skipped {
                continue;
            }
            let (shared, resume_at) = find_resume_point(
                &saved_state.tokens,
                saved_state.snapshots.iter().any(|saved_snapshot| {
                    saved_snapshot.snapshot.pos == prompt_len
                        && saved_snapshot.snapshot.has_logits()
                }),
            );
            branch_point = branch_point.max(shared.min(prompt_len));
            if resume_at == 0 {
                continue;
            }
            for snapshot in saved_state
                .snapshots
                .iter()
                .map(|saved_snapshot| saved_snapshot.snapshot)
            {
                let from = snapshot.pos.min(resume_at);
                if snapshot.pos <= resume_at + count_rewind_room(snapshot.stale_rows)
                    && (from < prompt_len || (snapshot.pos == prompt_len && snapshot.has_logits()))
                {
                    let plan = Plan::Load {
                        saved_id: id,
                        snapshot_pos: Some(snapshot.pos),
                        from,
                        branch_point: 0,
                    };
                    add_plan(prompt_len - from, RANK_LOAD_SNAPSHOT, plan);
                }
            }
            if limits.can_replay && resume_at < prompt_len {
                let plan = Plan::Load {
                    saved_id: id,
                    snapshot_pos: None,
                    from: resume_at,
                    branch_point: 0,
                };
                let cost = prompt_len - resume_at + (limits.replay_cost)(resume_at);
                add_plan(cost, RANK_LOAD_REPLAY, plan);
            }
        }
        for (&id, entry) in &self.entries {
            if entry.model != model {
                continue;
            }
            let (shared, resume_at) = find_resume_point(
                &entry.tokens,
                entry
                    .snapshots
                    .iter()
                    .any(|snapshot| snapshot.pos == prompt_len && snapshot.has_logits()),
            );
            branch_point = branch_point.max(shared.min(prompt_len));
            if resume_at == 0 {
                continue;
            }
            if !entry.busy && entry.tokens.len() <= resume_at + count_rewind_room(entry.stale_rows)
            {
                let plan = Plan::InPlace {
                    state: id,
                    from: resume_at,
                };
                add_plan(prompt_len - resume_at, RANK_IN_PLACE, plan);
            }
            for snapshot in &entry.snapshots {
                // A snapshot serves a resume at its position, or up to its
                // rewind room before it. Without a prefill, it serves only at
                // the prompt's end, with logits.
                let from = snapshot.pos.min(resume_at);
                if snapshot.pos <= resume_at + count_rewind_room(snapshot.stale_rows)
                    && (from < prompt_len || (snapshot.pos == prompt_len && snapshot.has_logits()))
                {
                    add_plan(
                        prompt_len - from,
                        RANK_SNAPSHOT,
                        Plan::Snapshot {
                            source: id,
                            snapshot_pos: snapshot.pos,
                            from,
                            branch_point: 0,
                        },
                    );
                }
            }
            if limits.can_replay && resume_at < prompt_len {
                add_plan(
                    prompt_len - resume_at + (limits.replay_cost)(resume_at),
                    RANK_REPLAY,
                    Plan::Replay {
                        source: id,
                        from: resume_at,
                    },
                );
            }
        }
        plans.sort_by_key(|&(cost, rank, _)| (cost, rank));
        plans.truncate(
            plans
                .iter()
                .position(|&(_, _, plan)| plan == Plan::Cold)
                .expect("the cold start")
                + 1,
        );
        plans
            .into_iter()
            .map(|(_, _, plan)| match plan {
                Plan::Snapshot {
                    source,
                    snapshot_pos,
                    from,
                    ..
                } => Plan::Snapshot {
                    source,
                    snapshot_pos,
                    from,
                    branch_point: branch_point.max(from),
                },
                Plan::Load {
                    saved_id,
                    snapshot_pos: Some(snapshot_pos),
                    from,
                    ..
                } => Plan::Load {
                    saved_id,
                    snapshot_pos: Some(snapshot_pos),
                    from,
                    branch_point: branch_point.max(from),
                },
                plan => plan,
            })
            .collect()
    }

    /// Finds the origin of a prefill on `plan` for a prompt of `prompt_len`
    /// tokens. The origin is the position that the prefill's rows continue
    /// from, with its stale rows. It is a state's end, a snapshot's position,
    /// where a replay starts, or 0. A replay rebuilds its rows. They have no
    /// stale rows.
    pub fn find_origin(&self, plan: &Plan, prompt_len: usize) -> Origin {
        let snapshot_stale_rows = |id: StateId, pos: usize| {
            self.find_snapshot(id, pos)
                .expect("the plan's snapshot")
                .stale_rows
        };
        match *plan {
            Plan::Cold => Origin {
                pos: 0,
                stale_rows: 0,
            },
            Plan::Replay { from, .. }
            | Plan::Load {
                snapshot_pos: None,
                from,
                ..
            } => Origin {
                pos: from,
                stale_rows: 0,
            },
            Plan::Snapshot {
                source,
                snapshot_pos,
                ..
            } => Origin {
                pos: snapshot_pos,
                stale_rows: snapshot_stale_rows(source, snapshot_pos),
            },
            Plan::Load {
                saved_id,
                snapshot_pos: Some(snapshot_pos),
                ..
            } => {
                let saved_snapshot = self.saved_states[&saved_id]
                    .snapshots
                    .iter()
                    .find(|candidate| candidate.snapshot.pos == snapshot_pos);
                Origin {
                    pos: snapshot_pos,
                    stale_rows: saved_snapshot
                        .expect("the plan's saved snapshot")
                        .snapshot
                        .stale_rows,
                }
            }
            // the same prompt again: its prompt-end snapshot is restored into
            // it
            Plan::InPlace { state, from } if from == prompt_len => Origin {
                pos: prompt_len,
                stale_rows: snapshot_stale_rows(state, prompt_len),
            },
            Plan::InPlace { state, .. } => Origin {
                pos: self.entries[&state].tokens.len(),
                stale_rows: self.entries[&state].stale_rows,
            },
        }
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
                saved_link: None,
            },
        );
        id
    }

    /// Starts a request in state `id` at position `from`. The state is new, or
    /// it continues in place. Its tokens become those of `prompt` before
    /// `from`, with `stale_rows` stale rows there. The request writes the
    /// state's rows from `from` on. So the cache removes the state's snapshots
    /// after `from`. It returns them for the worker to free. The state's saved
    /// link then covers at most the positions before `from`. The state becomes
    /// busy and used now. It takes `bytes`.
    pub fn start_request(
        &mut self,
        id: StateId,
        prompt: &[i32],
        from: usize,
        bytes: u64,
        stale_rows: usize,
    ) -> Vec<Snapshot> {
        let now = self.advance_clock();
        let entry = self.entries.get_mut(&id).expect("a live state");
        entry.busy = true;
        entry.last_used = now;
        entry.bytes = bytes;
        entry.stale_rows = stale_rows;
        entry.reserved_snapshots.clear();
        entry.tokens.clear();
        entry.tokens.extend_from_slice(&prompt[..from]);
        if let Some(link) = &mut entry.saved_link {
            // its rows from `from` on are written again
            link.shared_positions = link.shared_positions.min(from);
        }
        let (kept, dropped): (Vec<Snapshot>, Vec<Snapshot>) = entry
            .snapshots
            .drain(..)
            .partition(|snapshot| snapshot.pos <= from);
        entry.snapshots = kept;
        dropped
    }

    /// Adds the tokens that a prefill call computed to state `id`. `prompt` is
    /// the prompt up to where the call ended. Later requests may reuse these
    /// tokens before this request ends. `stale_rows` is the state's stale rows
    /// at the end of `prompt`.
    pub fn add_prefilled_tokens(&mut self, id: StateId, prompt: &[i32], stale_rows: usize) {
        let entry = self.entries.get_mut(&id).expect("a live prefilling state");
        assert!(entry.tokens.len() <= prompt.len());
        debug_assert_eq!(entry.tokens, prompt[..entry.tokens.len()]);
        entry
            .tokens
            .extend_from_slice(&prompt[entry.tokens.len()..]);
        entry.stale_rows = stale_rows;
    }

    /// Records that a request reuses state `id`. The state is used now.
    pub fn record_reuse(&mut self, id: StateId) {
        let now = self.advance_clock();
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.last_used = now;
        }
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

    /// Finds the snapshot of state `id` at `pos`.
    pub fn find_snapshot(&self, id: StateId, pos: usize) -> Option<Snapshot> {
        self.entries
            .get(&id)?
            .snapshots
            .iter()
            .find(|snapshot| snapshot.pos == pos)
            .copied()
    }

    /// Adds a decoded token at the end of state `id`. Its row replaces one
    /// stale row.
    pub fn add_decoded_token(&mut self, id: StateId, token: i32) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.tokens.push(token);
            entry.stale_rows = entry.stale_rows.saturating_sub(1);
        }
    }

    /// Records that a pass wrote `rejected_rows` rows past the end of state
    /// `id`. They are the rows after the tokens the pass accepted: rejected
    /// drafts, or rows after a stop token. They took the places of the margin's
    /// oldest rows. The stale rows were already there. So the state's stale
    /// rows become the larger of the two counts.
    pub fn record_rejected_rows(&mut self, id: StateId, rejected_rows: usize) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.stale_rows = entry.stale_rows.max(rejected_rows);
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

    /// Counts the bytes by which the states and snapshots exceed the limit.
    pub fn count_bytes_over_limit(&self) -> u64 {
        self.count_bytes().saturating_sub(self.byte_limit)
    }

    /// Grows the bytes counted for busy state `id` to `bytes`, before the pass
    /// that grows the state. First it evicts spare snapshots and idle states to
    /// stay within the limit. When that is not enough, it evicts all the other
    /// states' spare snapshots and idle states. The cache then exceeds its
    /// limit. It does nothing when `bytes` is not more than the state takes.
    /// Returns what it evicted.
    pub fn grow_state(&mut self, id: StateId, bytes: u64) -> Evicted {
        let Some(old_bytes) = self.entries.get(&id).map(|entry| entry.bytes) else {
            return Evicted::default();
        };
        if bytes <= old_bytes {
            return Evicted::default();
        }
        let (eviction, _) = self.list_evictions(bytes - old_bytes, false, &[id]);
        let evicted = self.evict(&eviction);
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.bytes = bytes;
        }
        evicted
    }

    /// Finishes the request in state `id`. The state becomes idle and used now.
    /// It takes `bytes`. `generation_end` is the snapshot at the generation's
    /// end, when the state has none there. The cache evicts what makes room for
    /// it within the limit. Then it adds the snapshot. When that room cannot be
    /// made, the cache adds no snapshot. It then evicts what brings it back
    /// within its limit, if it can. Returns whether the worker takes the
    /// snapshot, and what the cache evicted.
    pub fn finish_request(
        &mut self,
        id: StateId,
        bytes: u64,
        generation_end: Option<Snapshot>,
    ) -> (bool, Evicted) {
        self.set_idle(id, bytes);
        if let Some(eviction) = self.find_eviction(
            generation_end.map_or(0, |snapshot| snapshot.bytes),
            false,
            &[id],
        ) {
            let evicted = self.evict(&eviction);
            if let Some(snapshot) = generation_end {
                self.add_snapshot(id, snapshot);
            }
            return (generation_end.is_some(), evicted);
        }
        let evicted = self
            .find_eviction(0, false, &[id])
            .map(|eviction| self.evict(&eviction))
            .unwrap_or_default();
        (false, evicted)
    }

    /// Removes state `id`. Returns its snapshots for the worker to free.
    pub fn remove_state(&mut self, id: StateId) -> Vec<Snapshot> {
        self.entries
            .remove(&id)
            .map(|entry| entry.snapshots)
            .unwrap_or_default()
    }

    /// Lists the spare snapshots of `entry`: all but its last prompt-end
    /// snapshot and its last branch snapshot. They come in position order.
    fn list_spare_snapshots(entry: &Entry) -> Vec<Snapshot> {
        let find_last_pos = |kind: SnapshotKind| {
            entry
                .snapshots
                .iter()
                .filter(|snapshot| snapshot.kind == kind)
                .map(|snapshot| snapshot.pos)
                .max()
        };
        let (last_prompt_end, last_branch) = (
            find_last_pos(SnapshotKind::PromptEnd),
            find_last_pos(SnapshotKind::Branch),
        );
        entry
            .snapshots
            .iter()
            .filter(|snapshot| {
                !(snapshot.kind == SnapshotKind::PromptEnd && Some(snapshot.pos) == last_prompt_end)
                    && !(snapshot.kind == SnapshotKind::Branch && Some(snapshot.pos) == last_branch)
            })
            .copied()
            .collect()
    }

    /// Finds what to evict so that `need` more bytes fit within the limit. With
    /// `adds_state`, one more state must fit too. It picks spare snapshots
    /// first, from the least recently used states. Then it picks whole idle
    /// states, the least recently used first. It picks nothing of the states in
    /// `keep`. Returns None when no eviction makes the room.
    pub fn find_eviction(&self, need: u64, adds_state: bool, keep: &[StateId]) -> Option<Eviction> {
        let (eviction, fits) = self.list_evictions(need, adds_state, keep);
        fits.then_some(eviction)
    }

    /// Lists what to evict for `need` more bytes within the limit, and one more
    /// state with `adds_state`. The list stops when the room is there or
    /// nothing more can be evicted. Returns the list and whether the room is
    /// there.
    fn list_evictions(&self, need: u64, adds_state: bool, keep: &[StateId]) -> (Eviction, bool) {
        self.list_evictions_under(self.byte_limit, need, adds_state, keep)
    }

    /// Evicts spare snapshots and idle states until the cache takes `bytes`
    /// fewer bytes, or nothing more can be evicted. Returns what it evicted.
    pub fn evict_bytes(&mut self, bytes: u64) -> Evicted {
        let target = self.count_bytes().saturating_sub(bytes);
        let (eviction, _) = self.list_evictions_under(target, 0, false, &[]);
        self.evict(&eviction)
    }

    /// Lists what to evict for `need` more bytes within `limit`, and one more
    /// state with `adds_state`. It picks spare snapshots first, from the least
    /// recently used states. Then it picks whole idle states, the least
    /// recently used first. It picks nothing of the states in `keep`. The list
    /// stops when the room is there or nothing more can be evicted. Returns the
    /// list and whether the room is there.
    fn list_evictions_under(
        &self,
        limit: u64,
        need: u64,
        adds_state: bool,
        keep: &[StateId],
    ) -> (Eviction, bool) {
        let mut by_use: Vec<(&StateId, &Entry)> = self.entries.iter().collect();
        by_use.sort_by_key(|(_, entry)| entry.last_used);
        let mut total = self.count_bytes() + need;
        let mut state_count = self.entries.len() + usize::from(adds_state);
        let mut eviction = Eviction::default();
        for &(&id, entry) in &by_use {
            if total <= limit {
                break;
            }
            if keep.contains(&id) {
                continue;
            }
            for snapshot in Self::list_spare_snapshots(entry) {
                if total <= limit {
                    break;
                }
                total -= snapshot.bytes;
                eviction.snapshots.push((id, snapshot.pos));
            }
        }
        for &(&id, entry) in &by_use {
            if total <= limit && state_count <= self.max_states {
                break;
            }
            if keep.contains(&id) || entry.busy {
                continue;
            }
            let snapshot_bytes: u64 = entry
                .snapshots
                .iter()
                .filter(|snapshot| !eviction.snapshots.contains(&(id, snapshot.pos)))
                .chain(&entry.reserved_snapshots)
                .map(|snapshot| snapshot.bytes)
                .sum();
            total -= entry.bytes + snapshot_bytes;
            state_count -= 1;
            eviction.snapshots.retain(|&(state, _)| state != id);
            eviction.states.push(id);
        }
        let fits = total <= limit && state_count <= self.max_states;
        (eviction, fits)
    }

    /// Evicts the snapshots and states in `eviction`. Returns them for the
    /// worker to free. The worker may save the states first.
    pub fn evict(&mut self, eviction: &Eviction) -> Evicted {
        let mut snapshots = Vec::new();
        for &(id, pos) in &eviction.snapshots {
            if let Some(entry) = self.entries.get_mut(&id)
                && let Some(index) = entry
                    .snapshots
                    .iter()
                    .position(|snapshot| snapshot.pos == pos)
            {
                snapshots.push((id, entry.snapshots.remove(index)));
            }
        }
        let states = eviction
            .states
            .iter()
            .filter_map(|&id| self.entries.remove(&id).map(|entry| (id, entry)))
            .collect();
        Evicted { snapshots, states }
    }
}

// ---- saved states ----

impl Cache {
    /// Counts the bytes that saved states take on the SSD.
    pub fn count_saved_bytes(&self) -> u64 {
        self.saved_states
            .values()
            .map(SavedState::count_bytes)
            .sum()
    }

    /// Adds saved state `saved_state` with id `id`, and marks it used now. At
    /// startup, the caller adds the saved states it found, the least recently
    /// used first. So they keep their use order, before every state of this
    /// run.
    pub fn add_saved_state(&mut self, id: SavedStateId, mut saved_state: SavedState) {
        saved_state.last_used = self.advance_clock();
        self.saved_states.insert(id, saved_state);
    }

    /// Marks saved state `id` as used now.
    pub fn mark_saved_state_used(&mut self, id: SavedStateId) {
        let now = self.advance_clock();
        if let Some(saved_state) = self.saved_states.get_mut(&id) {
            saved_state.last_used = now;
        }
    }

    /// Links state `id` to saved state `saved_id`. State `id` loaded its rows
    /// before `shared_positions` from it. The link records the saved state's
    /// version now.
    pub fn link_to_saved_state(
        &mut self,
        id: StateId,
        saved_id: SavedStateId,
        shared_positions: usize,
    ) {
        let link = self
            .saved_states
            .get(&saved_id)
            .map(|saved_state| SavedLink {
                saved_id,
                version: saved_state.version,
                shared_positions,
            });
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.saved_link = link;
        }
    }

    /// Copies the saved link of state `source` to state `id`. State `id` copied
    /// the source's rows before `copied_positions`. The copy covers at most
    /// those positions.
    pub fn copy_saved_link(&mut self, id: StateId, source: StateId, copied_positions: usize) {
        let link = self
            .entries
            .get(&source)
            .and_then(|entry| entry.saved_link)
            .map(|source_link| SavedLink {
                shared_positions: source_link.shared_positions.min(copied_positions),
                ..source_link
            });
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.saved_link = link;
        }
    }

    /// Decides what saving the evicted state `entry` writes. `others_stay`
    /// tells whether the states in the cache stay in memory. Saving writes
    /// nothing when saving is off, or when the state has fewer than
    /// `min_unshared_tokens` unshared tokens. Otherwise it appends one part to
    /// the linked saved state, if that saved state has not changed since the
    /// link. It writes a new saved state instead when the saved state's
    /// replaced positions would then outnumber the state's tokens. It does the
    /// same when the saved state has `MAX_PARTS` parts. That new saved state
    /// replaces the linked one. An append or a replacement drops the linked
    /// saved state's tail: its tokens after the prefix it shares with the
    /// state. So neither happens when that tail has `min_unshared_tokens`
    /// tokens or more. Neither happens while an admitted request loads the
    /// linked saved state next. In every other case, saving writes a new saved
    /// state.
    pub fn decide_save(&self, entry: &Entry, others_stay: bool) -> SaveDecision {
        let len = entry.tokens.len();
        if self.saved_byte_limit == 0 || len == 0 {
            return SaveDecision::Skip;
        }
        let in_memory = self
            .entries
            .values()
            .filter(|other| others_stay && other.model == entry.model)
            .map(|other| &other.tokens[..]);
        let in_saved_states = self
            .saved_states
            .values()
            .filter(|saved_state| saved_state.model == Some(entry.model))
            .map(|saved_state| &saved_state.tokens[..]);
        let shared = in_memory
            .chain(in_saved_states)
            .map(|tokens| count_shared_prefix(&entry.tokens, tokens))
            .max()
            .unwrap_or(0);
        if len < shared + self.min_unshared_tokens {
            return SaveDecision::Skip;
        }
        let has_short_tail = |saved_state: &SavedState| {
            saved_state.tokens.len()
                < count_shared_prefix(&entry.tokens, &saved_state.tokens) + self.min_unshared_tokens
        };
        match entry.saved_link {
            Some(link)
                if Some(link.saved_id) != self.pinned
                    && self
                        .saved_states
                        .get(&link.saved_id)
                        .is_some_and(|saved_state| {
                            saved_state.version == link.version && has_short_tail(saved_state)
                        }) =>
            {
                let saved_state = &self.saved_states[&link.saved_id];
                let replaced_positions = saved_state.replaced_positions
                    + saved_state
                        .tokens
                        .len()
                        .saturating_sub(link.shared_positions);
                if replaced_positions > len || saved_state.parts.len() >= MAX_PARTS {
                    SaveDecision::New {
                        replaces: Some(link.saved_id),
                    }
                } else {
                    SaveDecision::Append {
                        saved_id: link.saved_id,
                        from: link.shared_positions,
                    }
                }
            }
            _ => SaveDecision::New { replaces: None },
        }
    }

    /// Finds the saved states that saved state `id` covers. A saved state of
    /// the same model is covered when it has fewer than `min_unshared_tokens`
    /// tokens after the prefix it shares with `id`. The pinned saved state is
    /// never covered.
    pub fn find_covered_saved_states(&self, id: SavedStateId) -> Vec<SavedStateId> {
        let Some(covering) = self.saved_states.get(&id) else {
            return Vec::new();
        };
        self.saved_states
            .iter()
            .filter(|&(&other, saved_state)| {
                other != id
                    && Some(other) != self.pinned
                    && saved_state.model.is_some()
                    && saved_state.model == covering.model
                    && saved_state.tokens.len()
                        < count_shared_prefix(&saved_state.tokens, &covering.tokens)
                            + self.min_unshared_tokens
            })
            .map(|(&other, _)| other)
            .collect()
    }

    /// Finds the saved states to delete so that `need` more bytes fit within
    /// `saved_byte_limit`. It never picks `keep` or the pinned saved state. It
    /// picks saved states of no model here first, then the least recently used.
    /// With `used_before`, it picks only those used before that time, and those
    /// of no model here. So a save deletes only saved states older than the
    /// state it saves. Returns None when deleting cannot make the room.
    pub fn find_saved_states_to_delete(
        &self,
        need: u64,
        keep: Option<SavedStateId>,
        used_before: Option<u64>,
    ) -> Option<Vec<SavedStateId>> {
        let mut total = self.count_saved_bytes() + need;
        let mut order: Vec<(&SavedStateId, &SavedState)> = self
            .saved_states
            .iter()
            .filter(|(id, saved_state)| {
                Some(**id) != keep
                    && Some(**id) != self.pinned
                    && (saved_state.model.is_none()
                        || used_before.is_none_or(|before| saved_state.last_used < before))
            })
            .collect();
        order.sort_by_key(|(_, saved_state)| (saved_state.model.is_some(), saved_state.last_used));
        let mut to_delete = Vec::new();
        for (&id, saved_state) in order {
            if total <= self.saved_byte_limit {
                break;
            }
            total -= saved_state.count_bytes();
            to_delete.push(id);
        }
        (total <= self.saved_byte_limit).then_some(to_delete)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek's layers.
    const LAYERS: usize = 40;
    /// DeepSeek's attention window, in positions.
    const WINDOW: usize = 128;
    /// The test model's limits: a rewind margin of 5 positions and a replay in
    /// DeepSeek's shape.
    const PLAN_LIMITS: PlanLimits<'static> = PlanLimits {
        rewind_margin: 5,
        can_replay: true,
        replay_cost: &count_replay_rows,
    };
    /// A layer's ring: its window and the rewind margin.
    const RING_ROWS: usize = WINDOW + PLAN_LIMITS.rewind_margin;
    /// A test state's bytes, 1 MiB.
    const STATE_BYTES: u64 = 1 << 20;
    /// A test snapshot's bytes, 6 MiB.
    const SNAPSHOT_BYTES: u64 = 6 << 20;

    /// Counts the rows a replay at `pos` computes again, in the shape of
    /// DeepSeek's replay. The rewind margin is 5, as in PLAN_LIMITS. The top
    /// layer computes again the last RING_ROWS positions before `pos`. Each
    /// layer below starts WINDOW - 1 positions earlier. The count is the
    /// average over the LAYERS layers, rounded up.
    fn count_replay_rows(pos: usize) -> usize {
        (0..LAYERS)
            .map(|depth| pos.min(RING_ROWS + (WINDOW - 1) * depth))
            .sum::<usize>()
            .div_ceil(LAYERS)
    }

    /// Collects the tokens of `range`.
    fn collect_tokens(range: std::ops::Range<i32>) -> Vec<i32> {
        range.collect()
    }

    /// Adds an idle state of model 0 that holds `tokens`. It has prompt-end
    /// snapshots at `prompt_ends`. Returns its id.
    fn add_idle_state(cache: &mut Cache, tokens: &[i32], prompt_ends: &[usize]) -> StateId {
        let id = cache.add_state(0, STATE_BYTES);
        cache.start_request(id, tokens, 0, STATE_BYTES, 0);
        cache.add_prefilled_tokens(id, tokens, 0);
        for &pos in prompt_ends {
            cache.add_snapshot(
                id,
                Snapshot {
                    pos,
                    kind: SnapshotKind::PromptEnd,
                    bytes: SNAPSHOT_BYTES,
                    stale_rows: 0,
                },
            );
        }
        cache.set_idle(id, STATE_BYTES);
        id
    }

    /// The plans for a prompt go from the cheapest to a cold start. The cold
    /// start can take the place of the state that the other plans keep.
    #[test]
    fn plans_end_with_a_cold_start_that_needs_no_state_kept() {
        let mut cache = Cache::new(3 << 20, 16);
        let source = add_idle_state(&mut cache, &collect_tokens(0..1000), &[]);
        let mut prompt = collect_tokens(0..8);
        prompt.extend(collect_tokens(5000..5900));
        let plans = cache.list_plans(0, &prompt, PLAN_LIMITS);
        assert_eq!(plans.first(), Some(&Plan::Replay { source, from: 8 }));
        assert_eq!(plans.last(), Some(&Plan::Cold));
        assert!(
            cache.find_eviction(5 << 20 >> 1, true, &[source]).is_none(),
            "beside the state it reuses"
        );
        let eviction = cache
            .find_eviction(5 << 20 >> 1, true, &[])
            .expect("in the state's place");
        assert_eq!(eviction.states, vec![source]);
    }

    #[test]
    fn models_share_the_limit_but_never_reuse_each_others_prefixes() {
        let mut cache = Cache::new(10, 16);
        let idle = cache.add_state(0, 6);
        cache.start_request(idle, &collect_tokens(0..100), 100, 6, 0);
        cache.set_idle(idle, 6);
        assert_eq!(
            cache.find_cheapest_plan(1, &collect_tokens(0..120), PLAN_LIMITS),
            Plan::Cold
        );
        let busy = cache.add_state(1, 3);
        cache.start_request(busy, &collect_tokens(0..100), 100, 3, 0);
        assert_eq!(cache.count_bytes(), 9);
        assert_eq!(
            (cache.count_model_bytes(0), cache.count_model_bytes(1)),
            (6, 3)
        );
        let eviction = cache
            .find_eviction(2, true, &[busy])
            .expect("room within the shared limit");
        assert_eq!(eviction.states, vec![idle]);
        cache.evict(&eviction);
        assert_eq!(cache.count_bytes(), 3);
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(0..120), PLAN_LIMITS),
            Plan::Cold
        );
    }

    #[test]
    fn growth_evicts_idle_states_of_other_models() {
        let mut cache = Cache::new(10, 16);
        let idle = cache.add_state(0, 3);
        cache.set_idle(idle, 3);
        let busy = cache.add_state(1, 6);
        let evicted = cache.grow_state(busy, 9);
        assert_eq!(
            evicted.states.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            vec![idle]
        );
        assert_eq!(cache.count_bytes(), 9);
        assert!(cache.entries[&busy].busy);
    }

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
        assert!(cache.find_eviction(2, true, &[]).is_none());
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

    #[test]
    fn evict_bytes_evicts_idle_states_of_any_model_and_keeps_busy_ones() {
        let mut cache = Cache::new(100, 16);
        let idle = cache.add_state(0, 4);
        cache.set_idle(idle, 4);
        let busy = cache.add_state(1, 3);
        let evicted = cache.evict_bytes(2);
        assert_eq!(
            evicted.states.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            vec![idle]
        );
        assert_eq!(cache.count_bytes(), 3);
        assert!(cache.entries[&busy].busy);
    }

    #[test]
    fn a_prefilling_state_is_reused_only_up_to_its_computed_tokens() {
        let mut cache = Cache::new(u64::MAX, 256);
        let source = add_idle_state(&mut cache, &collect_tokens(0..13772), &[13772]);
        cache.entries.get_mut(&source).unwrap().busy = true;
        let prefilling = cache.add_state(0, STATE_BYTES);
        cache.start_request(prefilling, &collect_tokens(0..19129), 13772, STATE_BYTES, 0);
        let plan = cache.find_cheapest_plan(0, &collect_tokens(0..22481), PLAN_LIMITS);
        assert!(
            plan.count_reused_tokens() <= 13772,
            "an admitted prompt has not computed its suffix: {plan:?}"
        );
        assert_eq!(cache.entries[&prefilling].tokens.len(), 13772);
        cache.add_prefilled_tokens(prefilling, &collect_tokens(0..16384), 0);
        assert!(
            cache
                .find_cheapest_plan(0, &collect_tokens(0..22481), PLAN_LIMITS)
                .count_reused_tokens()
                <= 16384
        );
        cache.add_prefilled_tokens(prefilling, &collect_tokens(0..19129), 0);
        assert_eq!(
            cache
                .find_cheapest_plan(0, &collect_tokens(0..22481), PLAN_LIMITS)
                .count_reused_tokens(),
            19129
        );
    }

    #[test]
    fn a_prompt_that_extends_an_idle_state_continues_it_in_place() {
        let mut cache = Cache::new(u64::MAX, 256);
        let idle = add_idle_state(&mut cache, &collect_tokens(0..1000), &[1000]);
        let prompt = collect_tokens(0..1200);
        assert_eq!(
            cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::InPlace {
                state: idle,
                from: 1000
            }
        );
        // within the rewind margin before its end: still in place
        let mut next_prompt = collect_tokens(0..997);
        next_prompt.extend(5000..5100);
        assert_eq!(
            cache.find_cheapest_plan(0, &next_prompt, PLAN_LIMITS),
            Plan::InPlace {
                state: idle,
                from: 997
            }
        );
    }

    #[test]
    fn the_longest_shared_prefix_wins_in_a_busy_or_idle_state() {
        let mut cache = Cache::new(u64::MAX, 256);
        let _short = add_idle_state(&mut cache, &collect_tokens(0..3000), &[3000]);
        let long = add_idle_state(&mut cache, &collect_tokens(0..20000), &[10000, 20000]);
        cache.entries.get_mut(&long).unwrap().busy = true;
        let mut prompt = collect_tokens(0..15000);
        prompt.extend(90000..90010);
        // a busy state is a source like any other, but never continued in place
        assert_eq!(
            cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::Replay {
                source: long,
                from: 15000
            }
        );
    }

    #[test]
    fn a_nearby_snapshot_beats_a_replay_and_a_far_one_loses_to_it() {
        let mut cache = Cache::new(u64::MAX, 256);
        let source = add_idle_state(&mut cache, &collect_tokens(0..20000), &[9000, 19000]);
        let mut near = collect_tokens(0..19500);
        near.push(-1);
        // snapshot at 19000: 501 tokens; replay at 19500: 1 + 2610
        assert_eq!(
            cache.find_cheapest_plan(0, &near, PLAN_LIMITS),
            Plan::Snapshot {
                source,
                snapshot_pos: 19000,
                from: 19000,
                branch_point: 19500
            }
        );
        let mut far = collect_tokens(0..15000);
        far.push(-1);
        // snapshot at 9000: 6001 tokens; replay at 15000: 1 + 2610
        assert_eq!(
            cache.find_cheapest_plan(0, &far, PLAN_LIMITS),
            Plan::Replay {
                source,
                from: 15000
            }
        );
        // for a model that resumes only from snapshots, the snapshot
        assert_eq!(
            cache.find_cheapest_plan(
                0,
                &far,
                PlanLimits {
                    can_replay: false,
                    ..PLAN_LIMITS
                }
            ),
            Plan::Snapshot {
                source,
                snapshot_pos: 9000,
                from: 9000,
                branch_point: 15000
            }
        );
    }

    #[test]
    fn the_same_prompt_again_resumes_from_its_prompt_end_without_a_prefill() {
        let mut cache = Cache::new(u64::MAX, 256);
        let prompt = collect_tokens(0..4000);
        let source = add_idle_state(&mut cache, &prompt, &[4000]);
        // its reply, decoded past the rewind margin
        for offset in 0..300 {
            cache.add_decoded_token(source, 100000 + offset);
        }
        let plan = cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS);
        assert_eq!(
            plan,
            Plan::Snapshot {
                source,
                snapshot_pos: 4000,
                from: 4000,
                branch_point: 4000
            }
        );
        assert_eq!(plan.count_reused_tokens(), 4000);
        assert_eq!(plan.describe_for_log(prompt.len()), "repeated");
        // without logits there, its last token runs again
        let mut other_cache = Cache::new(u64::MAX, 256);
        let idle = add_idle_state(&mut other_cache, &prompt, &[]);
        assert_eq!(
            other_cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::InPlace {
                state: idle,
                from: 3999
            }
        );
    }

    #[test]
    fn a_retry_after_a_short_reply_and_a_changed_last_token_stay_in_place() {
        let mut cache = Cache::new(u64::MAX, 256);
        let prompt = collect_tokens(0..4000);
        let idle = add_idle_state(&mut cache, &prompt, &[4000]);
        for offset in 0..3 {
            // a reply within the rewind margin
            cache.add_decoded_token(idle, 100000 + offset);
        }
        assert_eq!(
            cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::InPlace {
                state: idle,
                from: 4000
            }
        );
        assert_eq!(
            cache
                .find_cheapest_plan(0, &prompt, PLAN_LIMITS)
                .describe_for_log(prompt.len()),
            "repeated"
        );
        let mut changed = prompt.clone();
        *changed.last_mut().unwrap() = -7;
        assert_eq!(
            cache.find_cheapest_plan(0, &changed, PLAN_LIMITS),
            Plan::InPlace {
                state: idle,
                from: 3999
            }
        );
    }

    #[test]
    fn a_prompt_that_differs_just_before_a_snapshot_resumes_from_it() {
        // A thinking prompt ends with <think>. The next turn's history has
        // </think> there. So the next turn differs from the state one position
        // before its prompt-end snapshot. The snapshot still holds that
        // position's rows.
        let mut cache = Cache::new(u64::MAX, 256);
        let prompt = collect_tokens(0..4000);
        let source = add_idle_state(&mut cache, &prompt, &[4000]);
        for offset in 0..300 {
            // its reply, past the rewind margin
            cache.add_decoded_token(source, 100000 + offset);
        }
        let next_prompt = [&prompt[..3999], &collect_tokens(-60..0)[..]].concat();
        let plan = cache.find_cheapest_plan(0, &next_prompt, PLAN_LIMITS);
        assert_eq!(
            plan,
            Plan::Snapshot {
                source,
                snapshot_pos: 4000,
                from: 3999,
                branch_point: 3999
            }
        );
        assert_eq!(
            (
                plan.count_reused_tokens(),
                plan.find_branch_snapshot_pos(next_prompt.len()),
                plan.describe_for_log(next_prompt.len())
            ),
            (3999, None, "forked")
        );
        // the new state's rings hold all of the margin at its prompt's end
        assert_eq!(
            count_stale_rows_at(
                cache.find_origin(&plan, next_prompt.len()),
                next_prompt.len()
            ),
            0
        );
        // also with the bounded replay (--ced): it recomputes a ring
        let ced = PlanLimits {
            replay_cost: &|pos| pos.min(RING_ROWS),
            ..PLAN_LIMITS
        };
        assert_eq!(cache.find_cheapest_plan(0, &next_prompt, ced), plan);
        // the rewind margin's edge: 5 positions before it, not 6
        let edge = [&prompt[..3995], &collect_tokens(-60..0)[..]].concat();
        assert_eq!(
            cache.find_cheapest_plan(0, &edge, PLAN_LIMITS),
            Plan::Snapshot {
                source,
                snapshot_pos: 4000,
                from: 3995,
                branch_point: 3995
            }
        );
        let past = [&prompt[..3994], &collect_tokens(-60..0)[..]].concat();
        assert_eq!(
            cache.find_cheapest_plan(0, &past, PLAN_LIMITS),
            Plan::Replay { source, from: 3994 }
        );
        // a prompt that ends before the snapshot: its rows after the prompt's
        // end stay in the new state's rings, 2 stale rows
        let short = [&prompt[..3997], &[-1][..]].concat();
        let plan = cache.find_cheapest_plan(0, &short, PLAN_LIMITS);
        assert_eq!(
            plan,
            Plan::Snapshot {
                source,
                snapshot_pos: 4000,
                from: 3997,
                branch_point: 3997
            }
        );
        assert_eq!(
            count_stale_rows_at(cache.find_origin(&plan, short.len()), short.len()),
            2
        );
        // the same prompt again resumes at its prompt end, not before it
        assert_eq!(
            cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::Snapshot {
                source,
                snapshot_pos: 4000,
                from: 4000,
                branch_point: 4000
            }
        );
    }

    #[test]
    fn rows_a_verify_pass_writes_past_its_accepted_tokens_shrink_the_rewind_room() {
        let mut cache = Cache::new(u64::MAX, 256);
        let prompt = collect_tokens(0..4000);
        let state = add_idle_state(&mut cache, &prompt, &[]);
        // a pass of 6 rows that accepts 2 tokens: its 4 rows after them took
        // the places of the margin's 4 oldest rows. That leaves 1 position of
        // rewind room.
        cache.add_decoded_token(state, 100000);
        cache.add_decoded_token(state, 100001);
        cache.record_rejected_rows(state, 4);
        assert_eq!(cache.entries[&state].stale_rows, 4);
        // its end is 4002: a resume 2 positions before it would read a stale
        // row
        let two = [&prompt[..], &collect_tokens(-90..-60)[..]].concat();
        assert_eq!(
            cache.find_cheapest_plan(0, &two, PLAN_LIMITS),
            Plan::Replay {
                source: state,
                from: 4000
            }
        );
        let one = [&prompt[..], &[100000][..], &collect_tokens(-90..-60)[..]].concat();
        assert_eq!(
            cache.find_cheapest_plan(0, &one, PLAN_LIMITS),
            Plan::InPlace { state, from: 4001 }
        );
        // the generation-end snapshot carries them; the same snapshot without
        // stale rows would serve the resume 2 positions before it
        cache.add_snapshot(
            state,
            Snapshot {
                pos: 4002,
                kind: SnapshotKind::GenerationEnd,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 4,
            },
        );
        assert_eq!(
            cache.find_cheapest_plan(0, &two, PLAN_LIMITS),
            Plan::Replay {
                source: state,
                from: 4000
            }
        );
        cache.add_snapshot(
            state,
            Snapshot {
                pos: 4002,
                kind: SnapshotKind::GenerationEnd,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 0,
            },
        );
        assert_eq!(
            cache.find_cheapest_plan(0, &two, PLAN_LIMITS),
            Plan::Snapshot {
                source: state,
                snapshot_pos: 4002,
                from: 4000,
                branch_point: 4000
            }
        );
        // a second pass that accepts 1 token and writes 1 row after it: the
        // stale rows already hold the oldest places
        cache.add_decoded_token(state, 100002);
        cache.record_rejected_rows(state, 1);
        assert_eq!(cache.entries[&state].stale_rows, 3);
        // tokens written at its end give the rewind room back
        for offset in 0..3 {
            cache.add_decoded_token(state, 100003 + offset);
        }
        assert_eq!(cache.entries[&state].stale_rows, 0);
        let again = [
            &prompt[..],
            &[100000, 100001, 100002, 100003][..],
            &collect_tokens(-90..-60)[..],
        ]
        .concat();
        assert_eq!(
            cache.find_cheapest_plan(0, &again, PLAN_LIMITS),
            Plan::InPlace { state, from: 4004 }
        );
    }

    #[test]
    fn a_prompt_that_ends_before_a_continued_state_s_old_end_shrinks_its_rewind_room() {
        // A prompt continues a state before its end. The prompt also ends
        // before that end. The state's rings then keep its old rows after the
        // prompt's end. The rows of 4000..4002 took the places of the margin's
        // oldest positions, 3867..3869. A resume 5 positions before the new end
        // would read them.
        let mut cache = Cache::new(u64::MAX, 256);
        let prompt = collect_tokens(0..4000);
        let state = add_idle_state(&mut cache, &prompt, &[4000]);
        for offset in 0..3 {
            cache.add_decoded_token(state, 100000 + offset); // ends at 4003
        }
        let mut retry = prompt.clone();
        *retry.last_mut().unwrap() = -7; // differs at 3999, ends at 4000
        let plan = cache.find_cheapest_plan(0, &retry, PLAN_LIMITS);
        assert_eq!(plan, Plan::InPlace { state, from: 3999 });
        let stale_rows = count_stale_rows_at(cache.find_origin(&plan, retry.len()), retry.len());
        assert_eq!(stale_rows, 3);
        cache.start_request(state, &retry, 3999, STATE_BYTES, stale_rows);
        cache.add_prefilled_tokens(state, &retry, stale_rows);
        cache.set_idle(state, STATE_BYTES);
        let again = [&retry[..3995], &collect_tokens(-90..-60)[..]].concat();
        assert_ne!(
            cache.find_cheapest_plan(0, &again, PLAN_LIMITS),
            Plan::InPlace { state, from: 3995 }
        );
        // 2 positions before its end, within what its rings still hold
        let near = [&retry[..3998], &collect_tokens(-90..-60)[..]].concat();
        assert_eq!(
            cache.find_cheapest_plan(0, &near, PLAN_LIMITS),
            Plan::InPlace { state, from: 3998 }
        );
    }

    #[test]
    fn a_prompt_inside_a_longer_state_reuses_it_at_any_length() {
        assert_eq!(count_replay_rows(20000), 2610);
        assert_eq!(
            (count_replay_rows(1999), count_replay_rows(100)),
            (1633, 100)
        );
        let mut cache = Cache::new(u64::MAX, 256);
        let source = add_idle_state(&mut cache, &collect_tokens(0..20000), &[4000]);
        // 12,000 of its tokens: the replay at 11,999 (1 + 2,610) beats the
        // snapshot at 4,000
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(0..12000), PLAN_LIMITS),
            Plan::Replay {
                source,
                from: 11999
            }
        );
        // 2,000 of them: a replay recomputes 1,633 rows per layer, still less
        // than a cold start
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(0..2000), PLAN_LIMITS),
            Plan::Replay { source, from: 1999 }
        );
        // 100: a replay recomputes them all, as a cold start would, and reports
        // them reused
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(0..100), PLAN_LIMITS),
            Plan::Replay { source, from: 99 }
        );
    }

    #[test]
    fn a_fork_resumed_from_an_earlier_snapshot_carries_its_own_branch_point() {
        let mut cache = Cache::new(u64::MAX, 256);
        let source = add_idle_state(&mut cache, &collect_tokens(0..6000), &[]);
        cache.add_snapshot(
            source,
            Snapshot {
                pos: 4125,
                kind: SnapshotKind::Branch,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 0,
            },
        ); // an earlier fork's
        let mut prompt = collect_tokens(0..4262);
        prompt.extend(-20..0);
        // 137 tokens from the snapshot cost less than a replay at 4,262 (~2,610
        // rows)
        assert_eq!(
            cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::Snapshot {
                source,
                snapshot_pos: 4125,
                from: 4125,
                branch_point: 4262
            }
        );
    }

    #[test]
    fn under_ced_a_bounded_replay_beats_a_snapshot_more_than_a_ring_back() {
        // the bounded replay's cost under --ced with PLAN_LIMITS' margin of 5:
        // the ring, 133 positions
        let ced = PlanLimits {
            replay_cost: &|pos| pos.min(RING_ROWS),
            ..PLAN_LIMITS
        };
        let mut cache = Cache::new(u64::MAX, 256);
        let source = add_idle_state(&mut cache, &collect_tokens(0..6000), &[]);
        cache.add_snapshot(
            source,
            Snapshot {
                pos: 4125,
                kind: SnapshotKind::Branch,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 0,
            },
        );
        let mut far = collect_tokens(0..4262);
        far.extend(-20..0);
        assert_eq!(
            cache.find_cheapest_plan(0, &far, ced),
            Plan::Replay { source, from: 4262 }
        );
        let mut near = collect_tokens(0..4200);
        near.extend(-20..0);
        assert_eq!(
            cache.find_cheapest_plan(0, &near, ced),
            Plan::Snapshot {
                source,
                snapshot_pos: 4125,
                from: 4125,
                branch_point: 4200
            }
        );
    }

    #[test]
    fn a_prompt_that_shares_nothing_gets_a_cold_start() {
        let mut cache = Cache::new(u64::MAX, 256);
        add_idle_state(&mut cache, &collect_tokens(0..100), &[100]);
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(500..700), PLAN_LIMITS),
            Plan::Cold
        );
        assert_eq!(
            Cache::new(u64::MAX, 256).find_cheapest_plan(0, &collect_tokens(0..10), PLAN_LIMITS),
            Plan::Cold
        );
    }

    #[test]
    fn continuing_in_place_truncates_the_state_and_its_later_snapshots() {
        let mut cache = Cache::new(u64::MAX, 256);
        let state = add_idle_state(&mut cache, &collect_tokens(0..1000), &[400, 998, 1000]);
        let mut prompt = collect_tokens(0..998);
        prompt.extend(7000..7050);
        assert_eq!(
            cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::InPlace { state, from: 998 }
        );
        let dropped = cache.start_request(state, &prompt, 998, STATE_BYTES, 0);
        assert_eq!(
            dropped
                .iter()
                .map(|snapshot| snapshot.pos)
                .collect::<Vec<_>>(),
            vec![1000]
        );
        assert_eq!(cache.entries[&state].tokens, prompt[..998]);
        cache.add_prefilled_tokens(state, &prompt, 0);
        let entry = &cache.entries[&state];
        assert_eq!(entry.tokens, prompt);
        assert_eq!(
            entry
                .snapshots
                .iter()
                .map(|snapshot| snapshot.pos)
                .collect::<Vec<_>>(),
            vec![400, 998]
        );
        assert!(entry.busy);
        // a later prompt on the old tail shares only 998 positions with it
        assert_eq!(
            count_shared_prefix(&collect_tokens(0..1000), &entry.tokens),
            998
        );
    }

    #[test]
    fn a_removed_state_returns_its_snapshots_and_takes_no_bytes() {
        let mut cache = Cache::new(u64::MAX, 256);
        let state = add_idle_state(&mut cache, &collect_tokens(0..100), &[50, 100]);
        let snapshots = cache.remove_state(state);
        assert_eq!(
            snapshots
                .iter()
                .map(|snapshot| snapshot.pos)
                .collect::<Vec<_>>(),
            vec![50, 100]
        );
        assert!(!cache.entries.contains_key(&state));
        assert_eq!(cache.count_bytes(), 0);
    }

    #[test]
    fn a_replay_leaves_its_branch_point_as_a_snapshot_of_the_new_state() {
        let mut cache = Cache::new(u64::MAX, 256);
        let long = add_idle_state(&mut cache, &collect_tokens(0..8000), &[8000]);
        let mut prompt = collect_tokens(0..5000);
        prompt.extend(-50..0);
        let Plan::Replay { source, from } = cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS)
        else {
            panic!("a replay")
        };
        assert_eq!((source, from), (long, 5000));
        let new = cache.add_state(0, STATE_BYTES);
        cache.start_request(new, &prompt, from, STATE_BYTES, 0);
        cache.add_prefilled_tokens(new, &prompt, 0);
        cache.add_snapshot(
            new,
            Snapshot {
                pos: from,
                kind: SnapshotKind::Branch,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 0,
            },
        );
        cache.record_reuse(source);
        cache.set_idle(new, STATE_BYTES);
        // the next fork at the same point resumes from that snapshot
        let mut next_prompt = collect_tokens(0..5000);
        next_prompt.extend(-90..-60);
        assert_eq!(
            cache.find_cheapest_plan(0, &next_prompt, PLAN_LIMITS),
            Plan::Snapshot {
                source: new,
                snapshot_pos: 5000,
                from: 5000,
                branch_point: 5000
            }
        );
    }

    #[test]
    fn a_reused_state_becomes_the_most_recently_used() {
        let mut cache = Cache::new(u64::MAX, 256);
        let reused = add_idle_state(&mut cache, &collect_tokens(0..10), &[]);
        let later = add_idle_state(&mut cache, &collect_tokens(20..30), &[]);
        cache.record_reuse(reused);
        assert!(cache.entries[&reused].last_used > cache.entries[&later].last_used);
    }

    #[test]
    fn a_request_that_cannot_fit_is_refused_before_anything_is_evicted() {
        let mut cache = Cache::new(10 << 20, 256);
        let busy = add_idle_state(&mut cache, &collect_tokens(0..10), &[]);
        let idle = add_idle_state(&mut cache, &collect_tokens(20..30), &[]);
        assert_eq!(cache.count_bytes(), 2 << 20);
        cache.entries.get_mut(&busy).unwrap().busy = true;
        // 10 MiB more: only the idle state (1 MiB) can be evicted. That is not
        // enough.
        assert_eq!(cache.find_eviction(10 << 20, true, &[]), None);
        assert_eq!(cache.entries.len(), 2);
        // 9 MiB more fits once the idle state is evicted
        assert_eq!(
            cache.find_eviction(9 << 20, true, &[]),
            Some(Eviction {
                snapshots: vec![],
                states: vec![idle]
            })
        );
    }

    #[test]
    fn spare_snapshots_are_evicted_before_states_and_old_states_before_new_ones() {
        let mut cache = Cache::new(u64::MAX, 256);
        // 50 is spare: 100 is its last prompt end
        let old = add_idle_state(&mut cache, &collect_tokens(0..100), &[50, 100]);
        let new = add_idle_state(&mut cache, &collect_tokens(200..300), &[300 - 200]);
        cache
            .entries
            .get_mut(&new)
            .unwrap()
            .snapshots
            .push(Snapshot {
                pos: 40,
                kind: SnapshotKind::GenerationEnd,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 0,
            });
        let total = cache.count_bytes();
        cache.byte_limit = total;
        // one snapshot's room: the oldest state's spare snapshot
        assert_eq!(
            cache.find_eviction(SNAPSHOT_BYTES, false, &[]),
            Some(Eviction {
                snapshots: vec![(old, 50)],
                states: vec![]
            })
        );
        // a source in use is kept: the other state's spare snapshot instead
        assert_eq!(
            cache.find_eviction(SNAPSHOT_BYTES, false, &[old]),
            Some(Eviction {
                snapshots: vec![(new, 40)],
                states: vec![]
            })
        );
        // more than the spare snapshots: the least recently used state is
        // evicted whole
        let eviction = cache
            .find_eviction(2 * SNAPSHOT_BYTES + STATE_BYTES, false, &[])
            .unwrap();
        assert_eq!(eviction.states, vec![old]);
        let evicted = cache.evict(&eviction);
        assert_eq!(evicted.states.len(), 1);
        assert!(evicted.snapshots.iter().all(|&(id, _)| id == new));
        assert!(!cache.entries.contains_key(&old));
    }

    #[test]
    fn the_same_prompt_again_takes_no_new_snapshot_and_fits_a_full_cache() {
        let mut cache = Cache::new(u64::MAX, 256);
        let prompt = collect_tokens(0..4000);
        let source = add_idle_state(&mut cache, &prompt, &[4000]);
        // the same prompt again resumes at its prompt end, without a prefill
        let plan = cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS);
        assert_eq!(plan.count_reused_tokens(), prompt.len());
        assert_eq!(
            plan.count_snapshot_bytes(prompt.len(), SNAPSHOT_BYTES, SNAPSHOT_BYTES + 1),
            0
        );
        // so with the cache full it fits, and nothing is evicted
        cache.byte_limit = cache.count_bytes();
        assert_eq!(
            cache.find_eviction(
                plan.count_snapshot_bytes(prompt.len(), SNAPSHOT_BYTES, SNAPSHOT_BYTES + 1),
                false,
                &[source]
            ),
            Some(Eviction::default())
        );
        // a fork prefills: its prompt end (with logits) and its branch point
        let next_prompt = [&prompt[..3000], &collect_tokens(9000..9100)[..]].concat();
        let fork = cache.find_cheapest_plan(0, &next_prompt, PLAN_LIMITS);
        assert_eq!(fork.find_branch_snapshot_pos(next_prompt.len()), Some(3000));
        assert_eq!(
            fork.count_snapshot_bytes(next_prompt.len(), SNAPSHOT_BYTES, SNAPSHOT_BYTES + 1),
            2 * SNAPSHOT_BYTES + 1
        );
    }

    #[test]
    fn a_decoding_state_s_growth_is_counted_before_it_grows_and_others_are_evicted_first() {
        let mut cache = Cache::new(u64::MAX, 256);
        let old = add_idle_state(&mut cache, &collect_tokens(0..10), &[10]);
        cache.add_snapshot(
            old,
            Snapshot {
                pos: 5,
                kind: SnapshotKind::GenerationEnd,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 0,
            },
        ); // spare
        let busy = add_idle_state(&mut cache, &collect_tokens(20..30), &[]);
        // decoding
        cache.start_request(busy, &collect_tokens(20..30), 10, STATE_BYTES, 0);
        cache.byte_limit = cache.count_bytes();
        // one snapshot's growth: the spare snapshot is evicted before the
        // growth
        let evicted = cache.grow_state(busy, STATE_BYTES + SNAPSHOT_BYTES);
        assert_eq!(
            evicted
                .snapshots
                .iter()
                .map(|&(id, snapshot)| (id, snapshot.pos))
                .collect::<Vec<_>>(),
            vec![(old, 5)]
        );
        assert!(evicted.states.is_empty());
        assert_eq!(cache.entries[&busy].bytes, STATE_BYTES + SNAPSHOT_BYTES);
        assert_eq!(cache.count_bytes_over_limit(), 0);
        // more: the idle state is evicted whole
        let evicted = cache.grow_state(busy, (3 << 20) + SNAPSHOT_BYTES);
        assert_eq!(
            evicted.states.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            vec![old]
        );
        assert_eq!(cache.entries[&busy].bytes, (3 << 20) + SNAPSHOT_BYTES);
        assert_eq!(cache.count_bytes_over_limit(), 0);
        // no growth: nothing happens
        let evicted = cache.grow_state(busy, STATE_BYTES);
        assert!(evicted.snapshots.is_empty() && evicted.states.is_empty());
        assert_eq!(cache.entries[&busy].bytes, (3 << 20) + SNAPSHOT_BYTES);
    }

    #[test]
    fn growth_that_cannot_fit_evicts_every_idle_state_and_then_exceeds_the_limit() {
        let mut cache = Cache::new(u64::MAX, 256);
        let idle = add_idle_state(&mut cache, &collect_tokens(0..10), &[]);
        let busy = add_idle_state(&mut cache, &collect_tokens(20..30), &[]);
        let other_busy = add_idle_state(&mut cache, &collect_tokens(40..50), &[]);
        cache.start_request(busy, &collect_tokens(20..30), 10, STATE_BYTES, 0);
        cache.start_request(other_busy, &collect_tokens(40..50), 10, STATE_BYTES, 0);
        cache.byte_limit = cache.count_bytes(); // 3 MiB
        // 3 MiB more: only the idle state (1 MiB) can be evicted. The busy ones
        // stay.
        let evicted = cache.grow_state(busy, 4 << 20);
        assert_eq!(
            evicted.states.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            vec![idle]
        );
        assert_eq!(cache.entries[&busy].bytes, 4 << 20);
        assert_eq!(cache.count_bytes_over_limit(), 2 << 20);
        assert!(cache.entries.contains_key(&other_busy));
    }

    #[test]
    fn a_generation_end_snapshot_is_kept_only_within_the_limit() {
        let generation_end = Snapshot {
            pos: 10,
            kind: SnapshotKind::GenerationEnd,
            bytes: SNAPSHOT_BYTES,
            stale_rows: 0,
        };
        // the cache full, nothing else to evict: the snapshot is not taken
        let mut cache = Cache::new(u64::MAX, 256);
        let state = add_idle_state(&mut cache, &collect_tokens(0..10), &[]);
        cache.start_request(state, &collect_tokens(0..10), 10, STATE_BYTES, 0);
        cache.byte_limit = cache.count_bytes();
        let (take, _) = cache.finish_request(state, STATE_BYTES, Some(generation_end));
        assert!(!take);
        assert!(cache.find_snapshot(state, 10).is_none());
        assert_eq!(cache.count_bytes_over_limit(), 0);
        // an idle state to spare: it is evicted first, and the snapshot is kept
        let mut other_cache = Cache::new(u64::MAX, 256);
        let old = add_idle_state(&mut other_cache, &collect_tokens(100..110), &[]);
        let other_state = add_idle_state(&mut other_cache, &collect_tokens(0..10), &[]);
        other_cache.start_request(other_state, &collect_tokens(0..10), 10, STATE_BYTES, 0);
        other_cache.byte_limit = other_cache.count_bytes() + SNAPSHOT_BYTES - STATE_BYTES;
        let (take, evicted) =
            other_cache.finish_request(other_state, STATE_BYTES, Some(generation_end));
        assert!(take);
        assert_eq!(
            evicted.states.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            vec![old]
        );
        assert!(other_cache.find_snapshot(other_state, 10).is_some());
        assert_eq!(other_cache.count_bytes_over_limit(), 0);
    }

    #[test]
    fn the_state_count_is_bounded_too() {
        let mut cache = Cache::new(u64::MAX, 2);
        let old = add_idle_state(&mut cache, &collect_tokens(0..10), &[]);
        let _new = add_idle_state(&mut cache, &collect_tokens(20..30), &[]);
        assert_eq!(
            cache.find_eviction(0, true, &[]),
            Some(Eviction {
                snapshots: vec![],
                states: vec![old]
            })
        );
        assert_eq!(
            cache.find_eviction(0, false, &[]),
            Some(Eviction::default())
        );
    }

    // ---- saved states ----

    /// Makes a saved snapshot at `pos` of kind `kind`, written by save 1.
    fn make_saved_snapshot(pos: usize, kind: SnapshotKind) -> SavedSnapshot {
        SavedSnapshot {
            snapshot: Snapshot {
                pos,
                kind,
                bytes: SNAPSHOT_BYTES,
                stale_rows: 0,
            },
            version: 1,
        }
    }

    /// Adds saved state `id` of model 0 to `cache`. It holds `tokens` in one
    /// part, with `snapshots`.
    fn add_saved_state_in_one_part(
        cache: &mut Cache,
        id: SavedStateId,
        tokens: &[i32],
        snapshots: &[SavedSnapshot],
    ) {
        cache.add_saved_state(
            id,
            SavedState {
                model: Some(0),
                tokens: tokens.to_vec(),
                parts: vec![Part {
                    from: 0,
                    to: tokens.len(),
                    bytes: 100 * tokens.len() as u64,
                    file_number: 0,
                }],
                snapshots: snapshots.to_vec(),
                replaced_positions: 0,
                last_used: 0,
                version: 1,
            },
        );
    }

    /// Makes a cache that saves states of 100 unshared tokens or more.
    fn make_saving_cache() -> Cache {
        let mut cache = Cache::new(1 << 30, 16);
        cache.saved_byte_limit = 1 << 30;
        cache.min_unshared_tokens = 100;
        cache
    }

    #[test]
    fn a_saved_state_resumes_a_prompt_that_nothing_in_memory_holds() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(
            &mut cache,
            7,
            &collect_tokens(0..1000),
            &[
                make_saved_snapshot(1000, SnapshotKind::GenerationEnd),
                make_saved_snapshot(640, SnapshotKind::Branch),
            ],
        );
        let plan = cache.find_cheapest_plan(0, &collect_tokens(0..1500), PLAN_LIMITS);
        assert_eq!(
            plan,
            Plan::Load {
                saved_id: 7,
                snapshot_pos: Some(1000),
                from: 1000,
                branch_point: 1000
            }
        );
        assert_eq!(
            (
                plan.count_reused_tokens(),
                plan.describe_for_log(1500),
                plan.find_source_state(),
                plan.starts_with_replay()
            ),
            (1000, "loaded", None, false)
        );
        assert_eq!(
            cache.find_origin(&plan, 1500),
            Origin {
                pos: 1000,
                stale_rows: 0
            }
        );
        // differing before its end, past the rewind room: its branch snapshot
        let mut prompt = collect_tokens(0..700);
        prompt.extend(collect_tokens(5000..5100));
        assert_eq!(
            cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS),
            Plan::Load {
                saved_id: 7,
                snapshot_pos: Some(640),
                from: 640,
                branch_point: 700
            }
        );
        assert_eq!(
            cache
                .find_cheapest_plan(0, &prompt, PLAN_LIMITS)
                .find_branch_snapshot_pos(800),
            Some(700),
            "the branch snapshot is taken as in memory"
        );
    }

    #[test]
    fn a_saved_state_replays_where_no_snapshot_is_near() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(
            &mut cache,
            3,
            &collect_tokens(0..5000),
            &[make_saved_snapshot(200, SnapshotKind::Branch)],
        );
        let mut prompt = collect_tokens(0..4000);
        prompt.extend(collect_tokens(9000..9010));
        let plan = cache.find_cheapest_plan(0, &prompt, PLAN_LIMITS);
        assert_eq!(
            plan,
            Plan::Load {
                saved_id: 3,
                snapshot_pos: None,
                from: 4000,
                branch_point: 0
            }
        );
        assert!(plan.starts_with_replay());
        assert_eq!(
            (
                plan.find_branch_snapshot_pos(prompt.len()),
                cache.find_origin(&plan, prompt.len())
            ),
            (
                Some(4000),
                Origin {
                    pos: 4000,
                    stale_rows: 0
                }
            )
        );
    }

    #[test]
    fn a_saved_prompt_end_answers_the_same_prompt_again() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(
            &mut cache,
            1,
            &collect_tokens(0..300),
            &[make_saved_snapshot(300, SnapshotKind::PromptEnd)],
        );
        let plan = cache.find_cheapest_plan(0, &collect_tokens(0..300), PLAN_LIMITS);
        assert_eq!(
            plan,
            Plan::Load {
                saved_id: 1,
                snapshot_pos: Some(300),
                from: 300,
                branch_point: 300
            }
        );
        assert_eq!(plan.find_branch_snapshot_pos(300), None);
        // without logits there, its last token runs again
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(
            &mut cache,
            1,
            &collect_tokens(0..300),
            &[make_saved_snapshot(299, SnapshotKind::Branch)],
        );
        assert_eq!(
            cache
                .find_cheapest_plan(0, &collect_tokens(0..300), PLAN_LIMITS)
                .count_reused_tokens(),
            299
        );
    }

    #[test]
    fn at_equal_cost_memory_and_a_cold_start_go_before_the_ssd() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(
            &mut cache,
            1,
            &collect_tokens(0..1000),
            &[make_saved_snapshot(1000, SnapshotKind::GenerationEnd)],
        );
        let idle = add_idle_state(&mut cache, &collect_tokens(0..1000), &[1000]);
        let plan = cache.find_cheapest_plan(0, &collect_tokens(0..1200), PLAN_LIMITS);
        assert!(matches!(plan, Plan::InPlace { state, from: 1000 } if state == idle));
        // a saved state of another model, or of no model here, is never loaded
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(
            &mut cache,
            1,
            &collect_tokens(0..1000),
            &[make_saved_snapshot(1000, SnapshotKind::GenerationEnd)],
        );
        cache.saved_states.get_mut(&1).unwrap().model = None;
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(0..1200), PLAN_LIMITS),
            Plan::Cold
        );
        cache.saved_states.get_mut(&1).unwrap().model = Some(1);
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(0..1200), PLAN_LIMITS),
            Plan::Cold
        );
        // a replay that costs as much as a cold start: the cold start
        let mut cache = make_saving_cache();
        let limits = PlanLimits {
            replay_cost: &|pos| 2 * pos,
            ..PLAN_LIMITS
        };
        add_saved_state_in_one_part(&mut cache, 1, &collect_tokens(0..1000), &[]);
        assert_eq!(
            cache.find_cheapest_plan(0, &collect_tokens(0..1200), limits),
            Plan::Cold
        );
    }

    #[test]
    fn a_state_is_saved_when_it_holds_enough_unshared_tokens() {
        let mut cache = make_saving_cache();
        let state = add_idle_state(&mut cache, &collect_tokens(0..1000), &[1000]);
        let entry = cache.entries.remove(&state).unwrap();
        assert_eq!(
            cache.decide_save(&entry, true),
            SaveDecision::New { replaces: None }
        );
        cache.saved_byte_limit = 0;
        assert_eq!(
            cache.decide_save(&entry, true),
            SaveDecision::Skip,
            "no room on the SSD: nothing saved"
        );
        cache.saved_byte_limit = 1 << 30;
        // a state kept in memory holds all but 60 of its tokens
        let mut longer = collect_tokens(0..940);
        longer.extend(collect_tokens(7000..7500));
        add_idle_state(&mut cache, &longer, &[]);
        assert_eq!(cache.decide_save(&entry, true), SaveDecision::Skip);
        assert_eq!(
            cache.decide_save(&entry, false),
            SaveDecision::New { replaces: None },
            "at quit no state stays in memory"
        );
        // a saved state holds all but 60
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(&mut cache, 9, &longer, &[]);
        assert_eq!(cache.decide_save(&entry, false), SaveDecision::Skip);
        // too short to be worth it
        let mut cache = make_saving_cache();
        let short = add_idle_state(&mut cache, &collect_tokens(0..99), &[99]);
        let entry = cache.entries.remove(&short).unwrap();
        assert_eq!(cache.decide_save(&entry, false), SaveDecision::Skip);
    }

    #[test]
    fn a_state_loaded_from_a_saved_one_adds_a_part_to_it() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(
            &mut cache,
            4,
            &collect_tokens(0..1000),
            &[make_saved_snapshot(1000, SnapshotKind::GenerationEnd)],
        );
        let id = add_idle_state(&mut cache, &collect_tokens(0..1000), &[]);
        cache.link_to_saved_state(id, 4, 1000);
        // it continued past the saved state's end
        cache.start_request(id, &collect_tokens(0..1400), 1000, STATE_BYTES, 0);
        cache.add_prefilled_tokens(id, &collect_tokens(0..1400), 0);
        cache.set_idle(id, STATE_BYTES);
        let entry = cache.entries.remove(&id).unwrap();
        assert_eq!(
            cache.decide_save(&entry, true),
            SaveDecision::Append {
                saved_id: 4,
                from: 1000
            }
        );
        // a save into the saved state since: its link is old, a new saved state
        cache.saved_states.get_mut(&4).unwrap().version = 2;
        assert_eq!(
            cache.decide_save(&entry, true),
            SaveDecision::New { replaces: None }
        );
        cache.saved_states.get_mut(&4).unwrap().version = 1;
        // its replaced positions would outnumber the state's tokens: a new
        // saved state replaces it
        cache.saved_states.get_mut(&4).unwrap().replaced_positions = 1500;
        assert_eq!(
            cache.decide_save(&entry, true),
            SaveDecision::New { replaces: Some(4) }
        );
    }

    #[test]
    fn saved_links_follow_copies_and_continuations() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(&mut cache, 4, &collect_tokens(0..1000), &[]);
        let loaded = add_idle_state(&mut cache, &collect_tokens(0..1000), &[]);
        cache.link_to_saved_state(loaded, 4, 1000);
        assert_eq!(
            cache.entries[&loaded].saved_link,
            Some(SavedLink {
                saved_id: 4,
                version: 1,
                shared_positions: 1000
            })
        );
        // a copy of its rows before 600 shares them up to there
        let copy = cache.add_state(0, STATE_BYTES);
        cache.copy_saved_link(copy, loaded, 600);
        assert_eq!(
            cache.entries[&copy]
                .saved_link
                .map(|link| link.shared_positions),
            Some(600)
        );
        // continuing before its end: its rows from there are written again
        cache.start_request(loaded, &collect_tokens(0..1000), 950, STATE_BYTES, 0);
        assert_eq!(
            cache.entries[&loaded]
                .saved_link
                .map(|link| link.shared_positions),
            Some(950)
        );
        // a part from there replaces the saved state's 50 rows after it. They
        // are fewer than min_unshared_tokens. So the saved state's tail is not
        // worth keeping.
        let mut prompt = collect_tokens(0..950);
        prompt.extend(collect_tokens(3000..3200));
        cache.add_prefilled_tokens(loaded, &prompt, 0);
        cache.set_idle(loaded, STATE_BYTES);
        let entry = cache.entries.remove(&loaded).unwrap();
        assert_eq!(
            cache.decide_save(&entry, false),
            SaveDecision::Append {
                saved_id: 4,
                from: 950
            }
        );
    }

    #[test]
    fn a_save_leaves_the_saved_state_a_request_loads_next_as_it_is() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(&mut cache, 4, &collect_tokens(0..1000), &[]);
        let mut prompt = collect_tokens(0..990);
        prompt.extend(collect_tokens(8000..8300));
        let id = add_idle_state(&mut cache, &prompt, &[]);
        cache.link_to_saved_state(id, 4, 990);
        let entry = cache.entries.remove(&id).unwrap();
        assert_eq!(
            cache.decide_save(&entry, false),
            SaveDecision::Append {
                saved_id: 4,
                from: 990
            }
        );
        cache.pinned = Some(4);
        assert_eq!(
            cache.decide_save(&entry, false),
            SaveDecision::New { replaces: None },
            "not appended to"
        );
        cache.saved_states.get_mut(&4).unwrap().replaced_positions = 5000;
        assert_eq!(
            cache.decide_save(&entry, false),
            SaveDecision::New { replaces: None },
            "nor replaced"
        );
    }

    #[test]
    fn a_state_that_differs_early_does_not_cut_off_a_saved_state_s_tail() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(&mut cache, 4, &collect_tokens(0..1000), &[]);
        // a state loaded from it that differs at 500, with 200 unshared tokens.
        // The saved state's 500 tokens after position 500 stay.
        let mut prompt = collect_tokens(0..500);
        prompt.extend(collect_tokens(9000..9200));
        let id = add_idle_state(&mut cache, &prompt, &[]);
        cache.link_to_saved_state(id, 4, 500);
        let entry = cache.entries.remove(&id).unwrap();
        assert_eq!(
            cache.decide_save(&entry, false),
            SaveDecision::New { replaces: None },
            "not appended to"
        );
        cache.saved_states.get_mut(&4).unwrap().replaced_positions = 5000;
        assert_eq!(
            cache.decide_save(&entry, false),
            SaveDecision::New { replaces: None },
            "nor replaced"
        );
    }

    #[test]
    fn a_new_saved_state_covers_older_ones_and_the_oldest_are_deleted_first_for_room() {
        let mut cache = make_saving_cache();
        // 60 tokens after the prefix it shares with 2
        add_saved_state_in_one_part(&mut cache, 1, &collect_tokens(0..1000), &[]);
        let mut other = collect_tokens(0..940);
        other.extend(collect_tokens(5000..5300)); // 300 unshared tokens
        add_saved_state_in_one_part(&mut cache, 2, &other, &[]);
        let mut newer = collect_tokens(0..940);
        newer.extend(collect_tokens(6000..6500));
        add_saved_state_in_one_part(&mut cache, 3, &newer, &[]);
        assert_eq!(cache.find_covered_saved_states(3), vec![1]);
        cache.pinned = Some(1);
        assert!(
            cache.find_covered_saved_states(3).is_empty(),
            "the saved state a request loads next stays"
        );
        cache.pinned = None;
        // room: one of no model here first, then the least recently used
        add_saved_state_in_one_part(&mut cache, 4, &collect_tokens(0..10), &[]);
        cache.saved_states.get_mut(&4).unwrap().model = None;
        let all = cache.count_saved_bytes();
        cache.saved_byte_limit = all;
        assert_eq!(
            cache.find_saved_states_to_delete(0, None, None),
            Some(vec![])
        );
        assert_eq!(
            cache.find_saved_states_to_delete(1, None, None),
            Some(vec![4])
        );
        assert_eq!(
            cache.find_saved_states_to_delete(1001, None, None),
            Some(vec![4, 1])
        );
        assert_eq!(
            cache.find_saved_states_to_delete(1001, Some(1), None),
            Some(vec![4, 2])
        );
        assert_eq!(
            cache.find_saved_states_to_delete(all + 1, None, None),
            None,
            "more than the limit: no room"
        );
        // a state used before every saved one takes no room from them
        assert_eq!(cache.find_saved_states_to_delete(1001, None, Some(1)), None);
        assert_eq!(
            cache.find_saved_states_to_delete(1001, None, Some(3)),
            Some(vec![4, 1])
        );
    }

    #[test]
    fn a_saved_state_used_again_is_deleted_after_newer_ones() {
        let mut cache = make_saving_cache();
        add_saved_state_in_one_part(&mut cache, 1, &collect_tokens(0..10), &[]);
        add_saved_state_in_one_part(&mut cache, 2, &collect_tokens(20..30), &[]);
        cache.mark_saved_state_used(1);
        cache.saved_byte_limit = cache.count_saved_bytes();
        assert_eq!(
            cache.find_saved_states_to_delete(1, None, None),
            Some(vec![2])
        );
    }
}
