//! A fixed-size cache for search results, as used in chess and backgammon engines.
//!
//! Unlike a hash map it never grows: the memory is allocated once, so a long game can't use more
//! than the configured size, and there is no resizing (which briefly needs the old and the new
//! table at once). When a slot is needed for a new entry, an old one is replaced.
//!
//! Layout: an array of buckets, indexed by the key's hash, each with two slots:
//! - slot 0 is *depth-preferred*: it is only taken by an entry searched at least as deep, because
//!   deep results are the most expensive to recompute. The entry it held moves to slot 1.
//! - slot 1 *always* takes the newest entry, so recent shallow results are cached too.
//!
//! Each slot stores the full key, so a lookup can never return the value of a different position
//! (an entry can only be missing, never wrong).
use crate::fast_hash::FxHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy)]
struct Slot<K, V> {
    key: K,
    value: V,
    depth: u8,
}

type Bucket<K, V> = [Option<Slot<K, V>>; 2];

/// Keys `K`, values `V` (an equity by default; the analyzer stores an equity with the outcome
/// probabilities).
pub struct TranspositionTable<K, V = f64> {
    buckets: Vec<Bucket<K, V>>,
}

impl<K: Copy + Eq + Hash, V: Copy> TranspositionTable<K, V> {
    /// A table that uses at most `bytes` of memory (as many buckets as fit, at least one).
    pub fn with_size_bytes(bytes: usize) -> Self {
        let count = (bytes / std::mem::size_of::<Bucket<K, V>>()).max(1);
        TranspositionTable { buckets: vec![[None, None]; count] }
    }

    /// The bucket for a key: the hash scaled onto 0..count with a multiply and shift, which works
    /// for any bucket count (so the whole memory budget is used) and depends on the hash's high
    /// bits, the best mixed ones for FxHash.
    #[inline]
    fn index(&self, key: &K) -> usize {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        ((hasher.finish() as u128 * self.buckets.len() as u128) >> 64) as usize
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        self.buckets[self.index(key)]
            .iter()
            .flatten()
            .find(|slot| slot.key == *key)
            .map(|slot| slot.value)
    }

    /// Store a result; `depth` is how deep it was searched (deeper entries are kept preferably).
    #[inline]
    pub fn store(&mut self, key: K, depth: u8, value: V) {
        let index = self.index(&key);
        let bucket = &mut self.buckets[index];
        let new = Some(Slot { key, value, depth });
        if let Some(slot) = bucket.iter_mut().find(|s| s.map_or(false, |s| s.key == key)) {
            *slot = new;
        } else if bucket[0].map_or(true, |s| depth >= s.depth) {
            bucket[1] = bucket[0];
            bucket[0] = new;
        } else {
            bucket[1] = new;
        }
    }

    pub fn clear(&mut self) {
        self.buckets.fill([None, None]);
    }

    /// Number of stored entries (scans the whole table).
    pub fn len(&self) -> usize {
        self.buckets.iter().flatten().filter(|s| s.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum number of entries.
    pub fn capacity(&self) -> usize {
        self.buckets.len() * 2
    }

    /// Memory used by the table, in bytes.
    pub fn size_bytes(&self) -> usize {
        self.buckets.len() * std::mem::size_of::<Bucket<K, V>>()
    }
}

/// A `TranspositionTable` that several threads use at once (the analyzer's parallel search): split
/// into shards, each behind its own lock, chosen by bits of the key's hash that the shards' own
/// bucket index doesn't depend on, so threads rarely wait for each other. Also counts lookups
/// and hits.
pub struct SharedTable<K, V = f64> {
    shards: Vec<std::sync::Mutex<TranspositionTable<K, V>>>,
    lookups: std::sync::atomic::AtomicU64,
    hits: std::sync::atomic::AtomicU64,
}

const SHARDS: usize = 64;

impl<K: Copy + Eq + Hash, V: Copy> SharedTable<K, V> {
    /// A table that uses at most `bytes` of memory in all.
    pub fn with_size_bytes(bytes: usize) -> Self {
        SharedTable {
            shards: (0..SHARDS).map(|_| std::sync::Mutex::new(TranspositionTable::with_size_bytes(bytes / SHARDS))).collect(),
            lookups: Default::default(),
            hits: Default::default(),
        }
    }

    /// The shard of a key: the low bits of a second hash (the shards index buckets by the high bits).
    #[inline]
    fn shard(&self, key: &K) -> &std::sync::Mutex<TranspositionTable<K, V>> {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        let h = hasher.finish();
        &self.shards[((h ^ (h >> 29)) as usize) % SHARDS]
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        use std::sync::atomic::Ordering::Relaxed;
        self.lookups.fetch_add(1, Relaxed);
        let value = self.shard(key).lock().unwrap().get(key);
        if value.is_some() {
            self.hits.fetch_add(1, Relaxed);
        }
        value
    }

    #[inline]
    pub fn store(&self, key: K, depth: u8, value: V) {
        self.shard(&key).lock().unwrap().store(key, depth, value);
    }

    pub fn clear(&self) {
        for shard in &self.shards {
            shard.lock().unwrap().clear();
        }
        self.reset_stats();
    }

    /// (lookups, hits) since the last `clear` or `reset_stats`.
    pub fn stats(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.lookups.load(Relaxed), self.hits.load(Relaxed))
    }

    pub fn reset_stats(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.lookups.store(0, Relaxed);
        self.hits.store(0, Relaxed);
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().capacity()).sum()
    }

    pub fn size_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().size_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_returns_stored_value() {
        let mut tt = TranspositionTable::<u64>::with_size_bytes(1 << 16);
        tt.store(42, 1, 0.5);
        assert_eq!(tt.get(&42), Some(0.5));
        assert_eq!(tt.get(&43), None);
        tt.store(42, 1, 0.25);
        assert_eq!(tt.get(&42), Some(0.25));
        assert_eq!(tt.len(), 1);
    }

    #[test]
    fn test_size_limit_is_respected() {
        let mut tt = TranspositionTable::<u64>::with_size_bytes(1 << 16);
        let capacity = tt.capacity();
        assert!(tt.size_bytes() <= 1 << 16);
        for key in 0..100_000u64 {
            tt.store(key, 1, key as f64);
        }
        assert!(tt.len() <= capacity);
        // Whatever survived is correct.
        for key in 0..100_000u64 {
            if let Some(v) = tt.get(&key) {
                assert_eq!(v, key as f64);
            }
        }
    }

    #[test]
    fn test_deep_entries_survive_shallow_ones() {
        // A single bucket: every key competes for the same two slots.
        let mut tt = TranspositionTable::<u64>::with_size_bytes(1);
        assert_eq!(tt.capacity(), 2);
        tt.store(1, 3, 1.0);           // deep entry
        for key in 2..100 {
            tt.store(key, 1, 0.0);     // many shallow entries
        }
        assert_eq!(tt.get(&1), Some(1.0), "the deep entry must be kept");
        assert_eq!(tt.get(&99), Some(0.0), "the newest shallow entry is kept too");
        tt.store(100, 3, 2.0);         // an equally deep entry takes the depth-preferred slot
        assert_eq!(tt.get(&100), Some(2.0));
        assert_eq!(tt.get(&1), Some(1.0), "and the previous deep entry moves to the other slot");
    }

    #[test]
    fn test_uses_the_whole_memory_budget() {
        let budget = 64 << 20;
        let tt = TranspositionTable::<([u8; 26], [u8; 26], u32)>::with_size_bytes(budget);
        assert!(tt.size_bytes() <= budget);
        assert!(tt.size_bytes() > budget - 1024, "only rounding to whole buckets may be lost");
    }

    #[test]
    fn test_shared_table_from_several_threads() {
        let tt = SharedTable::<u64>::with_size_bytes(1 << 22);
        std::thread::scope(|scope| {
            for t in 0..4u64 {
                let tt = &tt;
                scope.spawn(move || {
                    for key in (t * 1000)..(t * 1000 + 1000) {
                        tt.store(key, 1, key as f64);
                    }
                });
            }
        });
        for key in 0..4000u64 {
            if let Some(v) = tt.get(&key) {
                assert_eq!(v, key as f64);
            }
        }
        assert!(tt.len() > 3500);
        tt.clear();
        assert!(tt.is_empty());
        assert_eq!(tt.stats(), (0, 0));
    }

    #[test]
    fn test_clear() {
        let mut tt = TranspositionTable::<u64>::with_size_bytes(1 << 12);
        tt.store(7, 1, 1.0);
        tt.clear();
        assert!(tt.is_empty());
    }
}
