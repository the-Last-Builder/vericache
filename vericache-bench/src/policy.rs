use std::collections::HashMap;
use std::hash::Hash;

struct Node<K, V> {
    key: K,
    val: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Unverified O(1) intrusive doubly-linked-list LRU eviction tracker.
pub struct LruPolicy<K, V> {
    cap: usize,
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    map: HashMap<K, usize>,
    head: Option<usize>, // Most recently used
    tail: Option<usize>, // Least recently used
}

impl<K: Eq + Hash + Copy, V: Copy> LruPolicy<K, V> {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            nodes: Vec::with_capacity(cap.max(1)),
            free: Vec::new(),
            map: HashMap::with_capacity(cap.max(1)),
            head: None,
            tail: None,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn is_full(&self) -> bool {
        self.map.len() >= self.cap
    }

    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let &idx = self.map.get(key)?;
        self.nodes[idx].as_ref().map(|n| n.val)
    }

    /// Touch `key`, promoting it to MRU head in O(1), and return its associated value.
    pub fn touch(&mut self, key: &K) -> Option<V> {
        let &idx = self.map.get(key)?;
        self.detach(idx);
        self.push_front(idx);
        self.nodes[idx].as_ref().map(|n| n.val)
    }

    /// Insert or update `key -> val` at MRU head. If capacity is exceeded, evicts and returns
    /// the LRU `(key, val)` pair.
    pub fn insert(&mut self, key: K, val: V) -> Option<(K, V)> {
        if let Some(&idx) = self.map.get(&key) {
            if let Some(node) = self.nodes[idx].as_mut() {
                node.val = val;
            }
            self.detach(idx);
            self.push_front(idx);
            return None;
        }
        let evicted = if self.map.len() >= self.cap {
            self.pop_lru()
        } else {
            None
        };
        let idx = self.alloc_node(Node {
            key,
            val,
            prev: None,
            next: None,
        });
        self.push_front(idx);
        self.map.insert(key, idx);
        evicted
    }

    pub fn peek_lru(&self) -> Option<(K, V)> {
        let idx = self.tail?;
        self.nodes[idx].as_ref().map(|n| (n.key, n.val))
    }

    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        let idx = self.tail?;
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.free.push(idx);
        self.map.remove(&node.key);
        Some((node.key, node.val))
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.map.remove(key)?;
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.free.push(idx);
        Some(node.val)
    }

    fn alloc_node(&mut self, node: Node<K, V>) -> usize {
        if let Some(idx) = self.free.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(node));
            idx
        }
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let n = self.nodes[idx].as_mut().expect("valid node");
            let p = n.prev.take();
            let nx = n.next.take();
            (p, nx)
        };
        match prev {
            Some(p) => self.nodes[p].as_mut().expect("prev").next = next,
            None => self.head = next,
        }
        match next {
            Some(nx) => self.nodes[nx].as_mut().expect("next").prev = prev,
            None => self.tail = prev,
        }
    }

    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        {
            let n = self.nodes[idx].as_mut().expect("valid node");
            n.prev = None;
            n.next = old_head;
        }
        if let Some(h) = old_head {
            self.nodes[h].as_mut().expect("head").prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::LruPolicy;

    #[test]
    fn lru_eviction_order() {
        let mut lru = LruPolicy::new(2);
        assert_eq!(lru.insert(1u64, 10u32), None);
        assert_eq!(lru.insert(2u64, 20u32), None);
        assert_eq!(lru.touch(&1), Some(10));
        assert_eq!(lru.insert(3u64, 30u32), Some((2, 20)));
        assert_eq!(lru.peek_lru(), Some((1, 10)));
        assert_eq!(lru.remove(&1), Some(10));
        assert_eq!(lru.len(), 1);
    }
}
