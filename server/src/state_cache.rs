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
//! states of its own model.
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

use std::collections::BTreeMap;

/// How many tokens one slice comparison takes. Slices compare faster than
/// single tokens.
const SHARED_PREFIX_BLOCK: usize = 1024;

// The order of plans at equal cost. The cost counts only prefilled tokens. A
// copy takes time it does not count. So in place comes first.
const RANK_IN_PLACE: usize = 0;
const RANK_SNAPSHOT: usize = 1;
const RANK_REPLAY: usize = 2;
const RANK_COLD: usize = 3;

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
}

impl Plan {
    /// Counts the prompt tokens that the plan reuses. The prefill starts after
    /// them.
    pub fn count_reused_tokens(&self) -> usize {
        match *self {
            Plan::Cold => 0,
            Plan::InPlace { from, .. } => from,
            Plan::Snapshot { from, .. } | Plan::Replay { from, .. } => from,
        }
    }

    /// Describes the plan for the log in one word. "cold" is a cold start.
    /// "continued" continues a state in place. "repeated" is the same prompt
    /// again, from its prompt-end snapshot. "forked" copies another state.
    pub fn describe_for_log(&self, prompt_len: usize) -> &'static str {
        match *self {
            Plan::Cold => "cold",
            Plan::InPlace { from, .. } if from == prompt_len => "repeated",
            Plan::InPlace { .. } => "continued",
            Plan::Snapshot { from, .. } if from == prompt_len => "repeated",
            Plan::Snapshot { .. } | Plan::Replay { .. } => "forked",
        }
    }

    /// Finds where the plan's prefill takes a branch snapshot, if anywhere. A
    /// replay takes it where the replay starts. A snapshot plan takes it at the
    /// branch point. It does so only when the branch point lies after `from`
    /// and before the prompt's end. A later fork there then resumes at once.
    pub fn find_branch_snapshot_pos(&self, prompt_len: usize) -> Option<usize> {
        match *self {
            Plan::Replay { from, .. } => Some(from),
            Plan::Snapshot {
                from, branch_point, ..
            } if branch_point > from && branch_point < prompt_len => Some(branch_point),
            _ => None,
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
            Plan::Replay { from, .. } => Origin {
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
            },
        );
        id
    }

    /// Starts a request in state `id` at position `from`. The state is new, or
    /// it continues in place. Its tokens become those of `prompt` before
    /// `from`, with `stale_rows` stale rows there. The request writes the
    /// state's rows from `from` on. So the cache removes the state's snapshots
    /// after `from`. It returns them for the worker to free. The state becomes
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

    /// Removes state `id`. Returns its snapshots for the worker to free.
    pub fn remove_state(&mut self, id: StateId) -> Vec<Snapshot> {
        self.entries
            .remove(&id)
            .map(|entry| entry.snapshots)
            .unwrap_or_default()
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
}
