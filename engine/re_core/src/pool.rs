//! `pool` — the pipe-tag → owning-node WORD POOL table.
//!
//! ## What problem this solves
//!
//! A puzzle node authors a `pool` of memory-words (and a glitched `pool_alt`, and
//! an `anchor` index naming the one word it contributes to the level's assembled
//! phrase). Those words are the level's actual text content. The renderer streams
//! readable words along a POWERED pipe — but a pipe cell carries only a 15-bit
//! pipe tag (see [`crate::encode_pipe_side_data`]), and there is no path from that
//! tag back to the node that owns the pipe. So the pipe surface has, until now,
//! streamed a hardcoded built-in word list.
//!
//! This table is that missing path, distilled to plain data:
//! `pipe tag → owning node → its pool`. It is built ONCE at level load (by the
//! game layer, from the parsed level) and read by the shade path.
//!
//! ## Why it is shaped as two flat `Vec`s
//!
//! The lookup happens inside the per-cell shade path, so it must be an index
//! read, not a hash. `pipe_to_node[tag]` then `nodes[node]` is two bounds-checked
//! slice reads and no allocation — and the pools are stored ONCE per node rather
//! than cloned per pipe (a node commonly owns several pipes).
//!
//! A missing entry is normal, not an error: demo levels have pipes with no
//! authored pool, and the caller falls back to its built-in list. Every lookup
//! therefore returns `Option`, and an EMPTY table returns `None` for every tag —
//! which is exactly the pre-table behaviour.

/// The word pool of ONE puzzle node: the memory-words it carries.
///
/// `pool` is the clean register; `pool_alt` is the corrupted/multi-language
/// variant a glitched level shows instead. `anchor` indexes `pool` and names the
/// single word this node contributes to the level's assembled readout phrase — so
/// a surface that wants "the node's own word first" reads `anchor` first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodePool {
    /// The clean word list, in authored order. May be empty.
    pub pool: Vec<String>,
    /// The corrupted variant, in authored order. May be empty even when `pool`
    /// is not (most nodes author only the clean pool).
    pub pool_alt: Vec<String>,
    /// Index into [`pool`](Self::pool) of the node's ANCHOR word. Out-of-range
    /// values are tolerated (they behave as `0`) because the authored `.lvl`
    /// `anchor = N` is not range-checked at parse time.
    pub anchor: usize,
}

impl NodePool {
    /// A pool with no words — what an unauthored node yields. Const so a table
    /// can be built without allocating a placeholder.
    pub const EMPTY: NodePool = NodePool {
        pool: Vec::new(),
        pool_alt: Vec::new(),
        anchor: 0,
    };

    /// Number of words in the clean pool.
    #[inline]
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Whether the CLEAN pool carries no words. (`pool_alt` is a variant of the
    /// same content, so a node with only an alt pool still reads as empty here —
    /// there is nothing to stream in the clean register.)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// The clamped anchor index — always a valid index into `pool` when the pool
    /// is non-empty, `0` otherwise. This is the "out-of-range falls back to 0"
    /// rule, in one place.
    #[inline]
    pub fn anchor_index(&self) -> usize {
        if self.anchor < self.pool.len() {
            self.anchor
        } else {
            0
        }
    }

    /// The `i`-th word in ANCHOR-FIRST order: index 0 is the anchor word, then the
    /// remaining pool words in authored order (skipping the anchor). This is the
    /// reading order a word-stream surface wants — the node's own contribution
    /// leads, the rest follows. `None` past the end / for an empty pool.
    #[inline]
    pub fn ordered_word(&self, i: usize) -> Option<&str> {
        if self.pool.is_empty() {
            return None;
        }
        let a = self.anchor_index();
        if i == 0 {
            return Some(self.pool[a].as_str());
        }
        // Walk the authored order, skipping the anchor (already emitted).
        let mut seen = 0usize;
        for (k, w) in self.pool.iter().enumerate() {
            if k == a {
                continue;
            }
            seen += 1;
            if seen == i {
                return Some(w.as_str());
            }
        }
        None
    }
}

/// `pipe tag → owning node → its [`NodePool`]`, flat and index-addressed.
///
/// Built at level load; read (never written) by the shade path. An empty table is
/// the valid "no pool data" state and answers `None` for every tag.
#[derive(Clone, Debug, Default)]
pub struct PoolTable {
    /// Per-node pools, indexed by node index.
    nodes: Vec<NodePool>,
    /// Per-pipe owning-node index, indexed by PIPE TAG (the decoded low bits of a
    /// pipe cell's `side_data`). `u32::MAX` marks "no owner recorded".
    pipe_to_node: Vec<u32>,
}

/// Sentinel in [`PoolTable`]'s pipe→node map meaning "this pipe has no recorded
/// owning node" (so the surface falls back to its built-in words).
pub const NO_NODE: u32 = u32::MAX;

impl PoolTable {
    /// The empty table — every lookup is `None`. `const` so it can back a static
    /// default with no allocation.
    pub const fn new() -> PoolTable {
        PoolTable {
            nodes: Vec::new(),
            pipe_to_node: Vec::new(),
        }
    }

    /// Build a table from per-node pools plus a per-pipe owning-node index.
    ///
    /// `pipe_owner[t]` is the node index that owns pipe tag `t`; an entry that is
    /// out of range for `nodes` is stored as [`NO_NODE`] (so a malformed input
    /// degrades to "no pool", never to a wrong node or a panic).
    pub fn from_parts(nodes: Vec<NodePool>, pipe_owner: &[usize]) -> PoolTable {
        let n = nodes.len();
        let pipe_to_node = pipe_owner
            .iter()
            .map(|&owner| if owner < n { owner as u32 } else { NO_NODE })
            .collect();
        PoolTable {
            nodes,
            pipe_to_node,
        }
    }

    /// Number of pipes the table describes.
    #[inline]
    pub fn pipe_count(&self) -> usize {
        self.pipe_to_node.len()
    }

    /// Number of nodes the table describes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the table carries nothing (the default / no-level state).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.pipe_to_node.is_empty()
    }

    /// The pool of the node owning `pipe_tag`, or `None` when the tag is unknown,
    /// has no recorded owner, or that node authored no words.
    ///
    /// THE HOT-PATH LOOKUP: two slice index reads, no hashing, no allocation.
    #[inline]
    pub fn pool_for_pipe(&self, pipe_tag: u16) -> Option<&NodePool> {
        let owner = *self.pipe_to_node.get(pipe_tag as usize)?;
        if owner == NO_NODE {
            return None;
        }
        let pool = self.nodes.get(owner as usize)?;
        (!pool.is_empty()).then_some(pool)
    }

    /// The pool of node `index` directly (diagnostics + the phrase layer).
    #[inline]
    pub fn pool_for_node(&self, index: usize) -> Option<&NodePool> {
        self.nodes.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(words: &[&str], anchor: usize) -> NodePool {
        NodePool {
            pool: words.iter().copied().map(String::from).collect(),
            pool_alt: Vec::new(),
            anchor,
        }
    }

    #[test]
    fn empty_table_answers_none_for_every_tag() {
        let t = PoolTable::new();
        assert!(t.is_empty());
        for tag in [0u16, 1, 7, 0x7FFF] {
            assert!(t.pool_for_pipe(tag).is_none());
        }
    }

    #[test]
    fn pipe_tag_resolves_to_its_owning_nodes_pool() {
        // node 0 = {a,b}, node 1 = {c}; pipe 0 owned by node 1, pipe 1 by node 0.
        let t = PoolTable::from_parts(vec![pool(&["a", "b"], 0), pool(&["c"], 0)], &[1, 0]);
        assert_eq!(t.pipe_count(), 2);
        assert_eq!(t.pool_for_pipe(0).unwrap().pool, vec!["c".to_string()]);
        assert_eq!(t.pool_for_pipe(1).unwrap().len(), 2);
        // A tag past the end is simply unknown.
        assert!(t.pool_for_pipe(2).is_none());
    }

    #[test]
    fn out_of_range_owner_degrades_to_no_pool() {
        let t = PoolTable::from_parts(vec![pool(&["a"], 0)], &[9]);
        assert!(
            t.pool_for_pipe(0).is_none(),
            "an owner index past the node list must read as no-pool, not as node 0"
        );
    }

    #[test]
    fn an_empty_pool_reads_as_no_pool() {
        let t = PoolTable::from_parts(vec![NodePool::EMPTY], &[0]);
        assert!(t.pool_for_pipe(0).is_none());
    }

    #[test]
    fn anchor_leads_the_ordered_reading() {
        let p = pool(&["learn", "edge", "seen"], 1);
        assert_eq!(p.anchor_index(), 1);
        assert_eq!(p.ordered_word(0), Some("edge"), "anchor word leads");
        assert_eq!(p.ordered_word(1), Some("learn"));
        assert_eq!(p.ordered_word(2), Some("seen"));
        assert_eq!(p.ordered_word(3), None);
    }

    #[test]
    fn out_of_range_anchor_falls_back_to_zero() {
        let p = pool(&["one", "two"], 99);
        assert_eq!(p.anchor_index(), 0);
        assert_eq!(p.ordered_word(0), Some("one"));
        assert_eq!(p.ordered_word(1), Some("two"));
    }

    #[test]
    fn empty_pool_has_no_ordered_words() {
        assert_eq!(NodePool::EMPTY.ordered_word(0), None);
        assert!(NodePool::EMPTY.is_empty());
    }
}
