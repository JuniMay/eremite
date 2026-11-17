//! An LLM's sub-view States: its sessions as a tree of their shared prefixes
//! along the token positions.

use std::ops::Range;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use unicode_width::UnicodeWidthStr;

use crate::ui::draw::{fit_line, format_percent, label, plain, strong};
use crate::ui::frame::{Choices, Screen};
use crate::ui::status::Session;
use crate::ui::theme::{Rgb, mix};

/// The text panel takes three eighths of the columns, and the tree the rest.
const PANEL_SHARE: (u16, u16) = (3, 8);
/// The columns at the right of the tree for the label after a session's end.
const LABEL_COLUMNS: u16 = 43;
/// The fewest columns of the tree's positions.
const LEAST_TREE_COLUMNS: u16 = 12;
/// The rows above the tree: the first line, an empty row, the axis and its
/// rule, and an empty row.
const ROWS_ABOVE_TREE: u16 = 5;
/// The steps between two marks of the positions' axis. The axis takes the
/// first that leaves this many columns between two marks.
const AXIS_STEPS: [u64; 10] = [
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000,
];
/// The fewest columns between two marks of the axis.
const AXIS_MARK_COLUMNS: f64 = 10.0;
/// The words of the plans of a request's start, in the order of the first
/// line.
const PLAN_WORDS: [&str; 5] = ["cold", "continued", "repeated", "forked", "loaded"];

/// A node of the sessions' tree: a run of positions that its sessions share.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeNode {
    /// The run's first position.
    pub start: u64,
    /// The position after the run.
    pub end: u64,
    /// The sessions through the node, by their indices in the model's
    /// sessions.
    pub sessions: Range<usize>,
    /// The sessions that end at the run's end.
    pub ending: Vec<usize>,
    /// The node before this one on its sessions' paths. None for a root.
    pub parent: Option<usize>,
    /// The nodes after it, in the order of their sessions' tokens.
    pub children: Vec<usize>,
    /// The tree's row of the node's line.
    pub row: usize,
}

/// Builds the tree of the shared prefixes of `sessions`. The sessions come in
/// the order of their tokens, each with the tokens that it shares with the
/// next. Returns the nodes, each after its parent, and the tree's rows. A
/// node's first child goes on in the node's row. Each other child takes the
/// rows under the subtrees before it.
pub fn build_session_tree(sessions: &[Session]) -> (Vec<TreeNode>, usize) {
    let mut nodes = Vec::new();
    for group in split_groups(sessions, 0..sessions.len(), 0) {
        add_group(sessions, group, 0, None, &mut nodes);
    }
    let mut rows = 0;
    for i in 0..nodes.len() {
        if nodes[i].parent.is_none() {
            place_rows(&mut nodes, i, &mut rows);
        }
    }
    (nodes, rows)
}

/// Splits the sessions of `range` into groups that share more than `shared`
/// tokens: each group's sessions share more with the next one of the group.
fn split_groups(sessions: &[Session], range: Range<usize>, shared: u64) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut start = range.start;
    for i in range.clone() {
        let goes_on = i + 1 < range.end && sessions[i].shared_with_next > shared;
        if !goes_on {
            groups.push(start..i + 1);
            start = i + 1;
        }
    }
    groups
}

/// Adds the node of `group`, the sessions that share more than `start`
/// tokens, with `parent`, then the nodes of its subgroups. The node runs from
/// `start` to the end of the prefix that the group's sessions share.
fn add_group(
    sessions: &[Session],
    group: Range<usize>,
    start: u64,
    parent: Option<usize>,
    nodes: &mut Vec<TreeNode>,
) {
    let end = if group.len() == 1 {
        sessions[group.start].tokens
    } else {
        (group.start..group.end - 1)
            .map(|i| sessions[i].shared_with_next)
            .min()
            .unwrap_or(start)
    };
    let index = nodes.len();
    let ending = group
        .clone()
        .filter(|&i| sessions[i].tokens == end)
        .collect();
    nodes.push(TreeNode {
        start,
        end,
        sessions: group.clone(),
        ending,
        parent,
        ..TreeNode::default()
    });
    if let Some(parent) = parent {
        nodes[parent].children.push(index);
    }
    for subgroup in split_groups(sessions, group, end) {
        // A session that ends at the node has no run after it.
        if subgroup.len() > 1 || sessions[subgroup.start].tokens > end {
            add_group(sessions, subgroup, end, Some(index), nodes);
        }
    }
}

/// Places the rows of node `i` and of its subtree, from row `rows` on. It
/// counts the rows used in `rows`. Returns the node's row.
fn place_rows(nodes: &mut [TreeNode], i: usize, rows: &mut usize) -> usize {
    let children = nodes[i].children.clone();
    let row = match children.split_first() {
        None => {
            *rows += 1;
            *rows - 1
        }
        Some((&first, rest)) => {
            let row = place_rows(nodes, first, rows);
            for &child in rest {
                place_rows(nodes, child, rows);
            }
            row
        }
    };
    nodes[i].row = row;
    row
}

/// Formats a token position briefly, such as 950, 12K or 1.2M.
fn format_position(position: u64) -> String {
    const THOUSAND: u64 = 1_000;
    const MILLION: u64 = 1_000_000;
    if position < THOUSAND {
        position.to_string()
    } else if position < MILLION {
        format!("{}K", position / THOUSAND)
    } else {
        format!("{:.1}M", position as f64 / MILLION as f64)
    }
}

/// Draws the States of the model chosen in `choices` in `area`: the first
/// line, then the axis of positions and the tree of the sessions. The columns
/// at the right stay for the text panel.
pub fn draw_states(buf: &mut Buffer, area: Rect, screen: &Screen, choices: &Choices) {
    let theme = screen.theme;
    let Some(status) = screen.statuses.get(choices.model) else {
        return;
    };
    let width = area.width as usize;
    let sessions = &status.sessions;
    let count = |word: &str| status.plans.get(word).copied().unwrap_or(0);
    let totals = &status.totals;
    let mut first = vec![
        plain(" "),
        strong(match sessions.len() {
            1 => "1 session".to_string(),
            n => format!("{n} sessions"),
        }),
        plain(format!("   this model {:.2} GiB", status.state_cache_gib)),
        label(theme, "   reused "),
        plain(format_percent(
            totals.reused_tokens as f64,
            totals.prompt_tokens as f64,
        )),
        label(theme, " of the prompt tokens   "),
    ];
    for (i, word) in PLAN_WORDS.iter().enumerate() {
        let gap = if i == 0 { "" } else { " · " };
        first.push(label(theme, format!("{gap}{word} ")));
        first.push(plain(count(word).to_string()));
    }
    if area.height == 0 {
        return;
    }
    buf.set_line(
        area.x,
        area.y,
        &fit_line(theme, Line::from(first), width),
        area.width,
    );
    let panel = area.width * PANEL_SHARE.0 / PANEL_SHARE.1;
    let tree_width = area.width - panel;
    let x0 = area.x + 2;
    let x1 = (area.x + tree_width)
        .saturating_sub(LABEL_COLUMNS)
        .max(x0 + LEAST_TREE_COLUMNS);
    if sessions.is_empty() || area.height < ROWS_ABOVE_TREE + 2 || x1 + 2 >= area.x + tree_width {
        let note = Line::from(vec![plain(" "), label(theme, "no sessions yet")]);
        buf.set_line(area.x, area.y + 2.min(area.height - 1), &note, area.width);
        return;
    }
    let (nodes, rows) = build_session_tree(sessions);
    let span = nodes.iter().map(|node| node.end).max().unwrap_or(1).max(1) as f64;
    let column =
        |position: u64| x0 + ((position as f64 / span).min(1.0) * (x1 - x0) as f64).round() as u16;
    // The positions' axis.
    let axis_y = area.y + 2;
    let rule = Style::new().fg(theme.color(theme.rule));
    let label_style = Style::new().fg(theme.color(theme.label));
    buf.set_string(x0, axis_y + 1, "─".repeat((x1 - x0 + 1) as usize), rule);
    let step = AXIS_STEPS
        .into_iter()
        .find(|&step| step as f64 / span * (x1 - x0) as f64 >= AXIS_MARK_COLUMNS)
        .unwrap_or(AXIS_STEPS[AXIS_STEPS.len() - 1]);
    let mut mark = 0;
    while mark as f64 <= span {
        let x = column(mark);
        buf[(x, axis_y + 1)].set_symbol("┬");
        let name = format_position(mark);
        let name_x = x.saturating_sub(name.width() as u16 / 2).max(area.x);
        buf.set_string(name_x, axis_y, &name, label_style);
        mark += step;
    }
    buf.set_string(x1 + 2, axis_y, "tokens", label_style);
    // Each node's line: whether a session through it decodes, and the
    // least idle time of its sessions.
    let busy: Vec<bool> = nodes
        .iter()
        .map(|node| node.sessions.clone().any(|i| sessions[i].busy))
        .collect();
    let idle: Vec<f64> = nodes
        .iter()
        .map(|node| {
            node.sessions
                .clone()
                .map(|i| sessions[i].idle_seconds)
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let longest_idle = sessions
        .iter()
        .map(|session| session.idle_seconds)
        .fold(0.0, f64::max);
    let tone = |busy: bool, idle: f64| -> Rgb {
        if busy {
            theme.accent
        } else {
            let share = if longest_idle > 0.0 {
                idle / longest_idle
            } else {
                0.0
            };
            mix(theme.idle_recent, theme.idle_oldest, share)
        }
    };
    let style_of = |i: usize| Style::new().fg(theme.color(tone(busy[i], idle[i])));
    let tree_y = area.y + ROWS_ABOVE_TREE;
    // Each row takes two lines: its line, and the one under it for the
    // forks' verticals.
    let fit = (area.bottom() - tree_y).div_ceil(2) as usize;
    let row_y = |row: usize| tree_y + 2 * row as u16;
    let start_column = |i: usize| {
        let x = column(nodes[i].start);
        if nodes[i].parent.is_some() { x + 1 } else { x }
    };
    for (i, node) in nodes.iter().enumerate() {
        if node.row >= fit {
            continue;
        }
        let (from, to) = (start_column(i), column(node.end).max(start_column(i)));
        let symbol = if busy[i] { "━" } else { "─" };
        for x in from..=to {
            buf[(x, row_y(node.row))]
                .set_symbol(symbol)
                .set_style(style_of(i));
        }
    }
    // The forks: a junction on the parent's row, a corner on each later
    // child's row, and the vertical between them.
    for (parent, node) in nodes.iter().enumerate() {
        let children = &node.children;
        if children.len() < 2 || node.row >= fit {
            continue;
        }
        let x = column(node.end);
        let later_busy = children[1..].iter().any(|&child| busy[child]);
        let junction = match (busy[parent], later_busy) {
            (true, true) => "┳",
            (true, false) => "┯",
            (false, true) => "┰",
            (false, false) => "┬",
        };
        buf[(x, row_y(node.row))]
            .set_symbol(junction)
            .set_style(style_of(parent));
        let last = children[children.len() - 1];
        for (j, &child) in children.iter().enumerate().skip(1) {
            let rest = &children[j..];
            let rest_busy = rest.iter().any(|&other| busy[other]);
            let rest_idle = rest
                .iter()
                .map(|&other| idle[other])
                .fold(f64::INFINITY, f64::min);
            let above = if j == 1 {
                node.row
            } else {
                nodes[children[j - 1]].row
            };
            let vertical = Style::new().fg(theme.color(tone(rest_busy, rest_idle)));
            for y in row_y(above) + 1..row_y(nodes[child].row).min(row_y(fit)) {
                let cell = &mut buf[(x, y)];
                let crossing = matches!(cell.symbol(), "─" | "━");
                let symbol = match (crossing, rest_busy) {
                    (true, true) => "╂",
                    (true, false) => "┼",
                    (false, true) => "┃",
                    (false, false) => "│",
                };
                cell.set_symbol(symbol).set_style(vertical);
            }
            if nodes[child].row < fit {
                let corner = match (child == last, busy[child]) {
                    (true, true) => "┗",
                    (true, false) => "└",
                    (false, true) => "┣",
                    (false, false) => "├",
                };
                buf[(x, row_y(nodes[child].row))]
                    .set_symbol(corner)
                    .set_style(style_of(child));
            }
        }
    }
    // Each session's end and its prompt's end.
    for (i, node) in nodes.iter().enumerate() {
        if node.ending.is_empty() || node.row >= fit {
            continue;
        }
        let y = row_y(node.row);
        let end = column(node.end);
        buf[(end, y)].set_symbol("●").set_style(style_of(i));
        for &index in &node.ending {
            if let Some(position) = sessions[index].prompt_end {
                // The node on the session's path that holds the position.
                let mut at = i;
                while nodes[at].start > position {
                    match nodes[at].parent {
                        Some(parent) => at = parent,
                        None => break,
                    }
                }
                if nodes[at].row < fit {
                    buf[(column(position), row_y(nodes[at].row))]
                        .set_symbol("◆")
                        .set_style(style_of(at).add_modifier(Modifier::BOLD));
                }
            }
        }
    }
    if rows > fit {
        let more = format!("… {} more rows", rows - fit);
        buf.set_string(x0, area.bottom() - 1, more, label_style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{Kind, ModelInfo};
    use crate::ui::samples::Samples;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, find_in_row, read_row, read_screen};
    use crate::ui::theme::DARK;

    /// deepseek-v4.1-flash's status as the worker builds it: five sessions in
    /// the order of their tokens. Session 5 decodes request 5. Sessions 2, 3,
    /// 4 and 1 share their first 7 tokens, and all share 2.
    const STATUS: &str = r#"{
        "loaded": true, "phase": "decode", "state_cache_gib": 0.33,
        "requests": [{"request": 5, "status": "decoding", "session": 5,
                      "prompt_tokens": 23, "generated_tokens": 581}],
        "sessions": [
            {"session": 5, "tokens": 604, "idle_seconds": 0, "busy": true,
             "bytes": 53687091, "prompt_end": 23, "shared_with_next": 2},
            {"session": 2, "tokens": 550, "idle_seconds": 23, "busy": false,
             "bytes": 53687091, "prompt_end": 518, "shared_with_next": 7},
            {"session": 3, "tokens": 469, "idle_seconds": 30, "busy": false,
             "bytes": 53687091, "prompt_end": 437, "shared_with_next": 7},
            {"session": 4, "tokens": 434, "idle_seconds": 23, "busy": false,
             "bytes": 53687091, "prompt_end": 402, "shared_with_next": 7},
            {"session": 1, "tokens": 673, "idle_seconds": 44, "busy": false,
             "bytes": 53687091, "prompt_end": 625, "shared_with_next": 0}],
        "totals": {"requests": 4, "prompt_tokens": 1938, "reused_tokens": 12},
        "plans": {"cold": 1, "forked": 4}
    }"#;

    /// Draws States at `width` x `height` with `choices`, and returns the
    /// cells.
    fn draw(width: u16, height: u16, status: &str, choices: &Choices) -> Buffer {
        let models = [ModelInfo {
            name: "deepseek-v4.1-flash".into(),
            kind: Kind::Llm,
        }];
        let statuses = [serde_json::from_str(status).expect("the worker's form")];
        let samples = Samples::new(WINDOWS);
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 60.0,
            models: &models,
            statuses: &statuses,
            samples: &samples,
            home: "",
        };
        draw_on_terminal(width, height, |buf, area| {
            draw_states(buf, area, &screen, choices);
        })
    }

    #[test]
    fn the_tree_draws_each_sessions_line_its_forks_and_its_ends() {
        let buf = draw(160, 40, STATUS, &Choices::default());
        assert_eq!(
            read_row(&buf, 0).trim_end(),
            " 5 sessions   this model 0.33 GiB   reused 0.6% of the prompt tokens   cold 1 \
             · continued 0 · repeated 0 · forked 4 · loaded 0"
        );
        assert_eq!(find_in_row(&buf, 2, "0"), Some(2));
        assert!(read_row(&buf, 2).contains("tokens"));
        assert!(read_row(&buf, 3).starts_with("  ┬────"));
        // the root's fork, then the decoding session's heavy line in the
        // accent, its prompt's end and its end
        let busy = read_row(&buf, 5);
        assert!(busy.starts_with("  ┯━◆━━━"), "{busy}");
        assert!(busy.contains("━━━●"), "{busy}");
        assert_eq!(buf[(3, 5)].fg, DARK.color(DARK.accent));
        // the shared prefix of the four others, with a fork for each
        assert!(read_row(&buf, 6).starts_with("  │"));
        assert!(
            read_row(&buf, 7).starts_with("  └┬───"),
            "{}",
            read_row(&buf, 7)
        );
        assert!(read_row(&buf, 9).starts_with("   ├───"));
        assert!(read_row(&buf, 13).starts_with("   └───"));
        // the session idle longest in the palest gray
        let longest = find_in_row(&buf, 13, "●").unwrap();
        assert_eq!(buf[(longest, 13)].fg, DARK.color(DARK.idle_oldest));
        assert_eq!(
            read_screen(&buf).matches('◆').count(),
            5,
            "the prompts' ends"
        );
    }

    #[test]
    fn without_sessions_or_room_the_view_says_so() {
        let choices = Choices::default();
        let buf = draw(160, 40, r#"{"loaded": true}"#, &choices);
        assert_eq!(read_row(&buf, 2).trim_end(), " no sessions yet");
        let buf = draw(100, 30, STATUS, &choices);
        // the tree in the 17 columns left beside the panel and the labels
        assert!(
            read_row(&buf, 5).starts_with("  ┯◆━━"),
            "{}",
            read_screen(&buf)
        );
        draw(20, 6, STATUS, &choices);
        draw(1, 1, STATUS, &choices);
        assert_eq!(format_position(12_345), "12K");
        assert_eq!(format_position(1_234_567), "1.2M");
    }

    /// A session numbered `number` with `tokens` tokens that shares `shared`
    /// tokens with the next.
    fn session(number: u64, tokens: u64, shared: u64) -> Session {
        Session {
            number,
            tokens,
            shared_with_next: shared,
            ..Session::default()
        }
    }

    #[test]
    fn the_tree_branches_where_the_sessions_part() {
        // Sessions 4 and 1 share 401 tokens; 3 and 2 share 400 with them and
        // 518 with each other; 5 shares nothing.
        let sessions = [
            session(4, 434, 401),
            session(1, 673, 400),
            session(3, 550, 518),
            session(2, 600, 0),
            session(5, 23, 0),
        ];
        let (nodes, rows) = build_session_tree(&sessions);
        let runs: Vec<(u64, u64, Vec<usize>, Option<usize>)> = nodes
            .iter()
            .map(|node| (node.start, node.end, node.ending.clone(), node.parent))
            .collect();
        assert_eq!(
            runs,
            [
                (0, 400, vec![], None),
                (400, 401, vec![], Some(0)),
                (401, 434, vec![0], Some(1)),
                (401, 673, vec![1], Some(1)),
                (400, 518, vec![], Some(0)),
                (518, 550, vec![2], Some(4)),
                (518, 600, vec![3], Some(4)),
                (0, 23, vec![4], None),
            ]
        );
        assert_eq!(rows, 5);
        // a first child goes on in its parent's row
        let rows: Vec<usize> = nodes.iter().map(|node| node.row).collect();
        assert_eq!(rows, [0, 0, 0, 1, 2, 2, 3, 4]);
        assert_eq!(nodes[0].sessions, 0..4);
    }

    #[test]
    fn a_session_that_ends_inside_another_ends_at_a_node_of_its_path() {
        // session 1 holds [1, 2, 3], session 2 goes on to five tokens
        let (nodes, rows) = build_session_tree(&[session(1, 3, 3), session(2, 5, 0)]);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            (nodes[0].start, nodes[0].end, nodes[0].ending.clone()),
            (0, 3, vec![0])
        );
        assert_eq!(
            (nodes[1].start, nodes[1].end, nodes[1].ending.clone()),
            (3, 5, vec![1])
        );
        assert_eq!(rows, 1);
        assert_eq!(build_session_tree(&[]), (Vec::new(), 0));
        // one session alone
        let (nodes, _) = build_session_tree(&[session(7, 9, 0)]);
        assert_eq!((nodes[0].end, nodes[0].ending.clone()), (9, vec![0]));
    }
}
