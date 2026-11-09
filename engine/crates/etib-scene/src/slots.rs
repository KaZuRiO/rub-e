//! Which slots of the static instance buffer are free.
//!
//! The cube culling pass reads a slot's chunk from `chunk_ids` and never a
//! chunk's start, so a chunk's cubes need not sit side by side: they may be
//! spread over several pieces. Any free slots will do, which is why the
//! buffer never needs compacting and an allocation never fails while the
//! free slots add up.

use std::collections::BTreeMap;
use std::ops::Range;

/// The free slots of a buffer of fixed capacity, lowest first.
pub(crate) struct SlotAllocator {
    /// Free pieces, start → length, never touching each other.
    free: BTreeMap<u32, u32>,
    capacity: u32,
    free_count: u32,
}

impl SlotAllocator {
    /// A buffer of `capacity` slots, all free.
    pub fn new(capacity: u32) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self {
            free,
            capacity,
            free_count: capacity,
        }
    }

    /// `count` slots, the lowest first, in one piece or several. `None`, with
    /// nothing taken, if fewer than `count` are free.
    pub fn alloc(&mut self, count: u32) -> Option<Vec<Range<u32>>> {
        if count > self.free_count {
            return None;
        }
        let mut pieces = Vec::new();
        let mut left = count;
        while left > 0 {
            let (start, len) = self.free.pop_first().expect("the free count covers it");
            let taken = len.min(left);
            pieces.push(start..start + taken);
            if taken < len {
                self.free.insert(start + taken, len - taken);
            }
            left -= taken;
        }
        self.free_count -= count;
        Some(pieces)
    }

    /// Gives back slots that [`alloc`](Self::alloc) handed out.
    pub fn free(&mut self, ranges: &[Range<u32>]) {
        for range in ranges.iter().filter(|range| !range.is_empty()) {
            let (mut start, mut end) = (range.start, range.end);
            self.free_count += end - start;
            if let Some((&before, &len)) = self.free.range(..start).next_back()
                && before + len == start
            {
                self.free.remove(&before);
                start = before;
            }
            if let Some(len) = self.free.remove(&end) {
                end += len;
            }
            self.free.insert(start, end - start);
        }
    }

    /// How many slots are free.
    pub fn free_count(&self) -> u32 {
        self.free_count
    }

    /// One past the last slot in use, 0 if every slot is free.
    pub fn end(&self) -> u32 {
        match self.free.last_key_value() {
            Some((&start, &len)) if start + len == self.capacity => start,
            _ => self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_packed_from_the_bottom() {
        let mut s = SlotAllocator::new(10);
        assert_eq!(s.alloc(3), Some(vec![0..3]));
        assert_eq!(s.alloc(2), Some(vec![3..5]));
        assert_eq!(s.end(), 5);
    }

    #[test]
    fn too_much_takes_nothing() {
        let mut s = SlotAllocator::new(2);
        assert_eq!(s.alloc(3), None);
        assert_eq!(s.free_count(), 2);
    }

    #[test]
    fn a_fragmented_buffer_still_fills_up() {
        let mut s = SlotAllocator::new(6);
        let a = s.alloc(2).unwrap();
        let _b = s.alloc(2).unwrap();
        let c = s.alloc(2).unwrap();
        s.free(&a);
        s.free(&c);
        assert_eq!(s.alloc(4), Some(vec![0..2, 4..6]));
        assert_eq!(s.free_count(), 0);
    }

    #[test]
    fn freed_neighbours_merge_and_the_end_comes_down() {
        let mut s = SlotAllocator::new(6);
        let a = s.alloc(3).unwrap();
        let b = s.alloc(3).unwrap();
        s.free(&b);
        s.free(&a);
        assert_eq!(s.end(), 0);
        assert_eq!(s.alloc(6), Some(vec![0..6]));
    }

    #[test]
    fn allocating_nothing_gives_nothing() {
        let mut s = SlotAllocator::new(2);
        assert_eq!(s.alloc(0), Some(vec![]));
    }
}
