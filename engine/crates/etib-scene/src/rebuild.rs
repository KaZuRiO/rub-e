//! Which changed chunks reach the GPU this frame.
//!
//! Block edits are wanted at once; region edits are spread over frames by a
//! budget counted in cubes, nearest chunks first, so that a large change never
//! freezes the image. Counting in cubes rather than milliseconds keeps a run
//! the same every time.

use std::collections::VecDeque;

use crate::culling::ChunkKey;
pub(crate) use crate::world::Priority;

/// Cubes rebuilt per frame for region edits, before
/// [`Scene::set_rebuild_budget`](crate::Scene::set_rebuild_budget) changes it.
pub const DEFAULT_REBUILD_BUDGET: u32 = 50_000;

/// A changed chunk waiting to be written to the GPU.
pub(crate) struct Candidate {
    pub key: ChunkKey,
    pub priority: Priority,
    /// From the chunk's centre to the camera.
    pub distance: f32,
    /// The cubes it takes on the GPU now, and the cubes it will take.
    pub old_len: u32,
    pub new_len: u32,
}

impl Candidate {
    /// Clearing its old slots costs as much as writing new ones.
    fn cost(&self) -> i64 {
        i64::from(self.old_len.max(self.new_len))
    }

    /// Whether it can be written into `room` free slots, its own old ones
    /// given back first.
    fn fits(&self, room: i64) -> bool {
        room + i64::from(self.old_len) >= i64::from(self.new_len)
    }

    fn shrinks(&self) -> bool {
        self.new_len < self.old_len
    }
}

/// The chunks to write this frame, in the order to write them, given `free`
/// slots on the GPU and a `budget` of cubes for the region edits.
///
/// Immediate chunks all go, first, whatever the budget: those that do not
/// grow before those that do, so a block broken far away makes room for one
/// placed nearby. One still without room pulls the nearest shrinking budget
/// chunks in before it until it has some; those spend the budget. Budget
/// chunks then go nearest first while the budget lasts; the first of the frame
/// goes even past it, so a budget of zero still makes progress. One without
/// room pulls the nearest waiting shrinking chunk ahead of it, whatever its
/// distance, until it has some.
pub(crate) fn plan_rebuilds(candidates: &[Candidate], free: u32, budget: u32) -> Vec<ChunkKey> {
    let mut order: Vec<&Candidate> = candidates.iter().collect();
    order.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.key.cmp(&b.key)));

    let mut room = i64::from(free);
    let mut budget = i64::from(budget);
    let mut plan = Vec::new();
    let mut waiting: VecDeque<&Candidate> = order
        .iter()
        .copied()
        .filter(|c| c.priority == Priority::Budget)
        .collect();
    let mut budget_used = false;

    let immediate = || order.iter().filter(|c| c.priority == Priority::Immediate);
    let steady = immediate().filter(|c| c.new_len <= c.old_len);
    let growing = immediate().filter(|c| c.new_len > c.old_len);
    for chunk in steady.chain(growing) {
        while !chunk.fits(room) {
            let Some(at) = waiting.iter().position(|c| c.shrinks()) else {
                break;
            };
            let shrinking = waiting.remove(at).expect("found above");
            room += i64::from(shrinking.old_len) - i64::from(shrinking.new_len);
            budget -= shrinking.cost();
            budget_used = true;
            plan.push(shrinking.key);
        }
        if chunk.fits(room) {
            room += i64::from(chunk.old_len) - i64::from(chunk.new_len);
            plan.push(chunk.key);
        }
    }

    while let Some(chunk) = waiting.pop_front() {
        if !chunk.fits(room) {
            // The nearest shrinking chunk goes ahead of it, then it tries
            // again; with none left, it waits for a later frame.
            if let Some(at) = waiting.iter().position(|c| c.shrinks()) {
                let shrinking = waiting.remove(at).expect("found above");
                waiting.push_front(chunk);
                waiting.push_front(shrinking);
            }
            continue;
        }
        if budget_used && chunk.cost() > budget {
            break;
        }
        room += i64::from(chunk.old_len) - i64::from(chunk.new_len);
        budget -= chunk.cost();
        budget_used = true;
        plan.push(chunk.key);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use Priority::{Budget, Immediate};

    fn chunk(x: i32, priority: Priority, old_len: u32, new_len: u32) -> Candidate {
        Candidate {
            key: [x, 0, 0],
            priority,
            distance: x as f32,
            old_len,
            new_len,
        }
    }

    #[test]
    fn nothing_to_do_plans_nothing() {
        assert!(plan_rebuilds(&[], 10, 100).is_empty());
    }

    #[test]
    fn nearest_chunks_go_first_up_to_the_budget() {
        let c = [
            chunk(3, Budget, 0, 40),
            chunk(1, Budget, 0, 40),
            chunk(2, Budget, 0, 40),
        ];
        assert_eq!(plan_rebuilds(&c, 1000, 100), [[1, 0, 0], [2, 0, 0]]);
    }

    #[test]
    fn immediate_chunks_ignore_the_budget_and_go_first() {
        let c = [chunk(1, Budget, 0, 10), chunk(9, Immediate, 0, 500)];
        assert_eq!(plan_rebuilds(&c, 1000, 100), [[9, 0, 0], [1, 0, 0]]);
    }

    #[test]
    fn a_chunk_over_the_budget_passes_alone() {
        let c = [chunk(1, Budget, 0, 500), chunk(2, Budget, 0, 10)];
        assert_eq!(plan_rebuilds(&c, 1000, 100), [[1, 0, 0]]);
    }

    #[test]
    fn a_budget_of_zero_still_makes_progress() {
        let c = [chunk(1, Budget, 0, 10), chunk(2, Budget, 0, 10)];
        assert_eq!(plan_rebuilds(&c, 1000, 0), [[1, 0, 0]]);
    }

    #[test]
    fn removing_a_chunk_costs_budget_too() {
        let c = [chunk(1, Budget, 80, 0), chunk(2, Budget, 80, 0)];
        assert_eq!(plan_rebuilds(&c, 0, 100), [[1, 0, 0]]);
    }

    #[test]
    fn a_full_buffer_lets_shrinking_chunks_pass_first() {
        let c = [chunk(1, Budget, 0, 50), chunk(9, Budget, 50, 0)];
        assert_eq!(plan_rebuilds(&c, 0, 1000), [[9, 0, 0], [1, 0, 0]]);
    }

    #[test]
    fn an_immediate_chunk_in_a_full_buffer_pulls_shrinking_chunks_past_the_budget() {
        let c = [
            chunk(1, Immediate, 0, 1),
            chunk(8, Budget, 500, 0),
            chunk(9, Budget, 500, 0),
        ];
        assert_eq!(plan_rebuilds(&c, 0, 10), [[8, 0, 0], [1, 0, 0]]);
    }

    #[test]
    fn immediate_chunks_that_shrink_make_room_for_those_that_grow() {
        let c = [chunk(1, Immediate, 0, 1), chunk(9, Immediate, 1, 0)];
        assert_eq!(plan_rebuilds(&c, 0, 10), [[9, 0, 0], [1, 0, 0]]);
    }

    #[test]
    fn a_waiting_chunk_pulls_shrinking_chunks_ahead_whatever_their_distance() {
        let c = [
            chunk(1, Budget, 0, 50),
            chunk(2, Budget, 0, 40),
            chunk(9, Budget, 50, 0),
        ];
        assert_eq!(plan_rebuilds(&c, 40, 100), [[9, 0, 0], [1, 0, 0]]);
    }
}
