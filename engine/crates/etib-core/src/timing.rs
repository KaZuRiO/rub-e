//! GPU timing: per-pass durations read back from timestamp queries.
//!
//! The GPU reports raw ticks. This module turns a frame's worth of them into
//! milliseconds per labelled pass, using the timestamp period the device
//! reports.

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
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the timestamp recorder that calls this lands in a later change"
    )
)]
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
