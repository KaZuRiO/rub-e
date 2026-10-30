//! GPU timing: per-pass durations read back from timestamp queries.
//!
//! The GPU reports raw ticks. This module turns a frame's worth of them into
//! milliseconds per labelled pass, using the timestamp period the device
//! reports.
//!
//! A [`Timer`] owns a few slots, each a query set plus the buffers its
//! timestamps are copied through. A timed encoder takes a slot, its passes
//! write their begin and end timestamps into it, and `Device::submit` copies
//! them out and starts mapping the result. The map is only ever polled, never
//! waited on, so the slot comes back to the pool a frame or two later, once
//! `Device::poll_timings` has seen it finish.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use log::{debug, warn};

/// Most passes of one encoder that are timed; later ones run untimed.
pub(crate) const MAX_TIMED_PASSES: u32 = 32;

/// How many timed submissions may be in flight at once. A timed encoder asked
/// for while all are busy is an ordinary one.
pub(crate) const TIMING_SLOTS: usize = 4;

/// Timestamps per slot: a begin and an end for each timed pass.
const TIMESTAMPS_PER_SLOT: u32 = 2 * MAX_TIMED_PASSES;

/// Bytes one timestamp takes once resolved.
const TIMESTAMP_BYTES: usize = 8;

const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

/// The GPU time spent in one labelled pass.
#[derive(Clone, Debug, PartialEq)]
pub struct PassTime {
    /// The label the pass was given when it was recorded.
    pub label: String,
    /// How long the pass took on the GPU, in milliseconds.
    pub ms: f32,
}

/// The GPU timings of one timed submission.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuTimings {
    /// The tag passed to `Device::timed_encoder`, so a result can be matched
    /// to the work it measured.
    pub tag: u64,
    /// One entry per pass, in execution order. A label that appears twice
    /// gives two entries.
    pub passes: Vec<PassTime>,
    /// From the earliest start to the latest end across all passes, in
    /// milliseconds. Zero when nothing was recorded.
    pub total_ms: f32,
}

/// Turns the raw timestamps of one submission into [`GpuTimings`].
///
/// `ticks` holds a begin and an end timestamp for each label, in that order,
/// so it has `2 × labels.len()` entries. `period_ns` is the nanoseconds per
/// tick. A pair that runs backwards counts as zero rather than underflowing.
pub(crate) fn to_timings(tag: u64, labels: &[String], ticks: &[u64], period_ns: f32) -> GpuTimings {
    let pairs: Vec<(u64, u64)> = ticks
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .take(labels.len())
        .collect();

    let passes = labels
        .iter()
        .zip(&pairs)
        .map(|(label, &(begin, end))| PassTime {
            label: label.clone(),
            ms: ticks_to_ms(end.saturating_sub(begin), period_ns),
        })
        .collect();

    let earliest = pairs.iter().map(|&(begin, _)| begin).min();
    let latest = pairs.iter().map(|&(_, end)| end).max();
    let total = match (earliest, latest) {
        (Some(begin), Some(end)) => end.saturating_sub(begin),
        _ => 0,
    };

    GpuTimings {
        tag,
        passes,
        total_ms: ticks_to_ms(total, period_ns),
    }
}

/// The GPU objects of one measurement slot.
struct Slot {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl Slot {
    fn new(device: &wgpu::Device, index: usize) -> Self {
        let size = u64::from(TIMESTAMPS_PER_SLOT) * TIMESTAMP_BYTES as u64;
        Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some(&format!("etib timing queries {index}")),
                ty: wgpu::QueryType::Timestamp,
                count: TIMESTAMPS_PER_SLOT,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("etib timing resolve {index}")),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("etib timing readback {index}")),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }
}

/// A submission whose timestamps are being mapped.
struct Pending {
    slot: usize,
    tag: u64,
    labels: Vec<String>,
    state: Arc<AtomicU8>,
}

/// The pool of measurement slots, shared by every clone of a [`Device`].
pub(crate) struct Timer {
    slots: Vec<Arc<Slot>>,
    free: Vec<usize>,
    /// In submission order.
    pending: VecDeque<Pending>,
    period_ns: f32,
}

impl Timer {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            slots: (0..TIMING_SLOTS)
                .map(|i| Arc::new(Slot::new(device, i)))
                .collect(),
            free: (0..TIMING_SLOTS).rev().collect(),
            pending: VecDeque::new(),
            period_ns: queue.get_timestamp_period(),
        }
    }

    /// Takes a free slot for a submission tagged `tag`, or `None` when all are
    /// in flight.
    pub(crate) fn lease(timer: &Arc<Mutex<Self>>, tag: u64) -> Option<Lease> {
        let mut guard = timer.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(index) = guard.free.pop() else {
            debug!("every timing slot is busy: this encoder will not be timed");
            return None;
        };
        let slot = Arc::clone(&guard.slots[index]);
        drop(guard);
        Some(Lease {
            timer: Arc::clone(timer),
            index,
            slot,
            tag,
            labels: Vec::new(),
            overflow_logged: false,
            submitted: false,
        })
    }

    /// The finished measurements, oldest first, with their slots returned to
    /// the pool. Stops at the first one still being mapped.
    pub(crate) fn collect(&mut self) -> Vec<GpuTimings> {
        let mut done = Vec::new();
        while let Some(front) = self.pending.front() {
            let state = front.state.load(Ordering::Acquire);
            if state == MAP_PENDING {
                break;
            }
            let Some(pending) = self.pending.pop_front() else {
                break;
            };
            let ticks = (state == MAP_READY)
                .then(|| read_ticks(&self.slots[pending.slot], pending.labels.len()))
                .flatten();
            match ticks {
                Some(ticks) => done.push(to_timings(
                    pending.tag,
                    &pending.labels,
                    &ticks,
                    self.period_ns,
                )),
                None => warn!("reading back GPU timings for tag {} failed", pending.tag),
            }
            self.free.push(pending.slot);
        }
        done
    }
}

/// The timestamps of a mapped slot, which is unmapped again whatever happens.
fn read_ticks(slot: &Slot, passes: usize) -> Option<Vec<u64>> {
    let view = slot
        .readback
        .slice(..slot_bytes(passes))
        .get_mapped_range()
        .ok();
    let ticks = view.map(|view| {
        view.chunks_exact(TIMESTAMP_BYTES)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap_or_default()))
            .collect()
    });
    slot.readback.unmap();
    ticks
}

/// Bytes holding the timestamps of `passes` passes.
fn slot_bytes(passes: usize) -> u64 {
    passes as u64 * 2 * TIMESTAMP_BYTES as u64
}

/// A slot taken by a timed encoder. Dropping it without submitting (an
/// encoder that was abandoned) returns the slot to the pool.
pub(crate) struct Lease {
    timer: Arc<Mutex<Timer>>,
    index: usize,
    slot: Arc<Slot>,
    tag: u64,
    labels: Vec<String>,
    overflow_logged: bool,
    submitted: bool,
}

impl Lease {
    /// Reserves the next pair of timestamps for a pass called `label`, or
    /// `None` when the slot is full.
    pub(crate) fn next_pass(&mut self, label: &str) -> Option<(u32, u32)> {
        let Some(pass) = u32::try_from(self.labels.len())
            .ok()
            .filter(|&n| n < MAX_TIMED_PASSES)
        else {
            if !self.overflow_logged {
                self.overflow_logged = true;
                debug!(
                    "more than {MAX_TIMED_PASSES} passes in a timed encoder: the rest are not timed"
                );
            }
            return None;
        };
        self.labels.push(label.to_owned());
        Some((2 * pass, 2 * pass + 1))
    }

    pub(crate) fn query_set(&self) -> &wgpu::QuerySet {
        &self.slot.query_set
    }

    /// Records the copy of the written timestamps to the readback buffer.
    /// Returns whether there was anything to copy.
    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if self.labels.is_empty() {
            return false;
        }
        let count = u32::try_from(self.labels.len()).unwrap_or(MAX_TIMED_PASSES) * 2;
        let bytes = slot_bytes(self.labels.len());
        encoder.resolve_query_set(&self.slot.query_set, 0..count, &self.slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.slot.resolve, 0, &self.slot.readback, 0, bytes);
        true
    }

    /// Starts mapping the readback buffer. Call after the work was submitted.
    pub(crate) fn launch(mut self) {
        let state = Arc::new(AtomicU8::new(MAP_PENDING));
        let callback_state = Arc::clone(&state);
        self.slot
            .readback
            .slice(..slot_bytes(self.labels.len()))
            .map_async(wgpu::MapMode::Read, move |result| {
                let outcome = if result.is_ok() {
                    MAP_READY
                } else {
                    MAP_FAILED
                };
                callback_state.store(outcome, Ordering::Release);
            });
        self.submitted = true;
        self.timer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pending
            .push_back(Pending {
                slot: self.index,
                tag: self.tag,
                labels: std::mem::take(&mut self.labels),
                state,
            });
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if !self.submitted {
            self.timer
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .free
                .push(self.index);
        }
    }
}

fn ticks_to_ms(ticks: u64, period_ns: f32) -> f32 {
    // Milliseconds need no more precision than f32 offers, and the f64
    // intermediate keeps the product of a large tick count exact enough.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "milliseconds fit in f32 with more precision than a timer reading"
    )]
    let ms = (ticks as f64 * f64::from(period_ns) / 1e6) as f32;
    ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_become_milliseconds_per_pass() {
        let labels = ["cull".to_owned(), "scene".to_owned()];
        // 1 tick = 1 ms.
        let t = to_timings(7, &labels, &[0, 2, 2, 5], 1_000_000.0);
        assert_eq!(t.tag, 7);
        assert_eq!(
            t.passes,
            vec![
                PassTime {
                    label: "cull".into(),
                    ms: 2.0
                },
                PassTime {
                    label: "scene".into(),
                    ms: 3.0
                },
            ]
        );
        assert_eq!(t.total_ms, 5.0);
    }

    #[test]
    fn backwards_timestamps_count_as_zero() {
        let t = to_timings(0, &["a".to_owned()], &[5, 3], 1_000_000.0);
        assert_eq!(t.passes[0].ms, 0.0);
        assert_eq!(t.total_ms, 0.0);
    }

    #[test]
    fn no_passes_means_no_time() {
        let t = to_timings(0, &[], &[], 1.0);
        assert!(t.passes.is_empty());
        assert_eq!(t.total_ms, 0.0);
    }
}
