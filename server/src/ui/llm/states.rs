//! An LLM's sub-view States: its sessions as a tree of their shared prefixes
//! along the token positions.

use std::ops::Range;

use crate::ui::status::Session;

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

#[cfg(test)]
mod tests {
    use super::*;

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
