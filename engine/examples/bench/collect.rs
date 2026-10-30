//! Turns per-frame CPU times and asynchronous GPU timings into summaries.

use std::collections::{HashMap, HashSet};

use anyhow::bail;
use etib::GpuTimings;
use serde::{Deserialize, Serialize};

use crate::scenario::{GRACE_FRAMES, MEASURED_FRAMES, WARMUP_FRAMES};

/// Statistics of a series of times, in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub median_ms: f32,
    pub p95_ms: f32,
    pub mean_ms: f32,
    pub samples: u32,
}

/// Median, nearest-rank 95th percentile and mean; `None` when empty.
pub fn summarize(samples: &[f32]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f32::total_cmp);
    let n = sorted.len();
    let median_ms = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    let rank = (0.95 * n as f64).ceil() as usize;
    let p95_ms = sorted[rank.clamp(1, n) - 1];
    let mean_ms = (sorted.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64) as f32;
    Some(Summary {
        median_ms,
        p95_ms,
        mean_ms,
        samples: n as u32,
    })
}

/// What a finished run measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurements {
    pub cpu_frame: Summary,
    /// `None` when no GPU measurement was accepted.
    pub gpu_total: Option<Summary>,
    /// Per-pass GPU times, in order of first appearance.
    pub passes: Vec<(String, Summary)>,
}

/// Gathers the CPU frame times and the (late, asynchronous) GPU timings of the
/// measured frames.
pub struct Collector {
    surface_size: (u32, u32),
    gpu_timing: bool,
    /// `frame_index` -> bench frame, for every rendered frame.
    frames: HashMap<u64, u32>,
    cpu_ms: Vec<f32>,
    /// Measured bench frames already recorded; a repeat keeps the first dt.
    measured_seen: HashSet<u32>,
    /// Tags whose GPU result was accepted, so none counts twice.
    accepted: HashSet<u64>,
    gpu_total_ms: Vec<f32>,
    passes: Vec<(String, Vec<f32>)>,
}

fn is_measured(bench_frame: u32) -> bool {
    (WARMUP_FRAMES..WARMUP_FRAMES + MEASURED_FRAMES).contains(&bench_frame)
}

impl Collector {
    pub fn new(surface_size: (u32, u32), gpu_timing: bool) -> Self {
        Self {
            surface_size,
            gpu_timing,
            frames: HashMap::new(),
            cpu_ms: Vec::new(),
            measured_seen: HashSet::new(),
            accepted: HashSet::new(),
            gpu_total_ms: Vec::new(),
            passes: Vec::new(),
        }
    }

    /// Records a rendered frame. `dt` is in seconds. Fails if the surface is
    /// no longer the size the run started with.
    pub fn frame_rendered(
        &mut self,
        frame_index: u64,
        bench_frame: u32,
        dt: f32,
        surface_size: (u32, u32),
    ) -> Result<(), anyhow::Error> {
        if surface_size != self.surface_size {
            let (from, to) = (self.surface_size, surface_size);
            bail!(
                "the window changed size during the run ({}x{} → {}x{}); nothing was saved",
                from.0,
                from.1,
                to.0,
                to.1
            );
        }
        self.frames.entry(frame_index).or_insert(bench_frame);
        if is_measured(bench_frame) && self.measured_seen.insert(bench_frame) {
            self.cpu_ms.push(dt * 1000.0);
        }
        Ok(())
    }

    /// Records the GPU timings of one frame. Results for frames that are
    /// unknown or not measured, and repeated results for a frame, are ignored.
    pub fn timings(&mut self, t: &GpuTimings) {
        let Some(&bench_frame) = self.frames.get(&t.tag) else {
            return;
        };
        if !is_measured(bench_frame) || !self.accepted.insert(t.tag) {
            return;
        }
        // Passes sharing a label within a frame add up.
        let mut per_label: Vec<(&str, f32)> = Vec::new();
        for p in &t.passes {
            match per_label.iter_mut().find(|(l, _)| *l == p.label) {
                Some((_, ms)) => *ms += p.ms,
                None => per_label.push((&p.label, p.ms)),
            }
        }
        for (label, ms) in per_label {
            match self.passes.iter_mut().find(|(l, _)| l == label) {
                Some((_, v)) => v.push(ms),
                None => self.passes.push((label.to_owned(), vec![ms])),
            }
        }
        self.gpu_total_ms.push(t.total_ms);
    }

    /// True once every measured frame is rendered and either all their GPU
    /// results arrived, GPU timing is off, or the grace period is over.
    pub fn is_done(&self, bench_frame: u32) -> bool {
        let rendered = self.measured_seen.len() >= MEASURED_FRAMES as usize;
        let gpu_complete = self.gpu_total_ms.len() >= MEASURED_FRAMES as usize;
        let grace_over = bench_frame >= WARMUP_FRAMES + MEASURED_FRAMES + GRACE_FRAMES;
        rendered && (!self.gpu_timing || gpu_complete || grace_over)
    }

    pub fn finish(&self) -> Measurements {
        let empty = Summary {
            median_ms: 0.0,
            p95_ms: 0.0,
            mean_ms: 0.0,
            samples: 0,
        };
        Measurements {
            cpu_frame: summarize(&self.cpu_ms).unwrap_or(empty),
            gpu_total: summarize(&self.gpu_total_ms),
            passes: self
                .passes
                .iter()
                .filter_map(|(l, v)| Some((l.clone(), summarize(v)?)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use etib::PassTime;

    fn pass(label: &str, ms: f32) -> PassTime {
        PassTime {
            label: label.into(),
            ms,
        }
    }

    fn gpu(tag: u64, passes: Vec<PassTime>) -> GpuTimings {
        let total_ms = passes.iter().map(|p| p.ms).sum();
        GpuTimings {
            tag,
            passes,
            total_ms,
        }
    }

    /// Renders every measured frame; frame_index = bench_frame + 1000.
    fn render_measured(c: &mut Collector) {
        for f in WARMUP_FRAMES..WARMUP_FRAMES + MEASURED_FRAMES {
            c.frame_rendered(1000 + f as u64, f, 0.002, (1280, 720))
                .unwrap();
        }
    }

    #[test]
    fn summary_of_known_values() {
        let s = summarize(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(
            (s.median_ms, s.p95_ms, s.mean_ms, s.samples),
            (2.5, 4.0, 2.5, 4)
        );
        assert!(summarize(&[]).is_none());
        let odd = summarize(&[5.0, 1.0, 3.0]).unwrap();
        assert_eq!((odd.median_ms, odd.p95_ms), (3.0, 5.0));
    }

    #[test]
    fn warmup_frames_are_not_measured() {
        let mut c = Collector::new((1280, 720), true);
        c.frame_rendered(1, 0, 0.5, (1280, 720)).unwrap();
        assert_eq!(c.finish().cpu_frame.samples, 0);
        c.frame_rendered(2, WARMUP_FRAMES, 0.004, (1280, 720))
            .unwrap();
        let cpu = c.finish().cpu_frame;
        assert_eq!(cpu.samples, 1);
        assert!((cpu.median_ms - 4.0).abs() < 1e-4);
    }

    #[test]
    fn collector_rejects_a_size_change() {
        let mut c = Collector::new((1280, 720), true);
        assert!(c.frame_rendered(1, 0, 0.01, (1280, 720)).is_ok());
        let err = c
            .frame_rendered(2, 1, 0.01, (640, 360))
            .unwrap_err()
            .to_string();
        assert!(err.contains("changed size"));
    }

    #[test]
    fn collector_sums_repeats_and_skips_absent_labels() {
        let mut c = Collector::new((1280, 720), true);
        let (a, b) = (WARMUP_FRAMES, WARMUP_FRAMES + 1);
        c.frame_rendered(10, a, 0.01, (1280, 720)).unwrap();
        c.frame_rendered(11, b, 0.01, (1280, 720)).unwrap();
        c.timings(&gpu(
            10,
            vec![pass("post", 1.0), pass("post", 2.0), pass("cull", 0.5)],
        ));
        c.timings(&gpu(11, vec![pass("post", 4.0)]));
        let m = c.finish();
        let get = |l: &str| m.passes.iter().find(|(n, _)| n == l).unwrap().1;
        assert_eq!(
            m.passes.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            ["post", "cull"]
        );
        let post = get("post");
        assert_eq!((post.samples, post.mean_ms, post.median_ms), (2, 3.5, 3.5));
        assert_eq!(get("cull").samples, 1);
        let total = m.gpu_total.unwrap();
        assert_eq!(total.samples, 2);
        assert_eq!(total.mean_ms, (3.5 + 4.0) / 2.0);
    }

    #[test]
    fn collector_ignores_unknown_out_of_range_and_duplicate_tags() {
        let mut c = Collector::new((1280, 720), true);
        c.frame_rendered(5, 0, 0.01, (1280, 720)).unwrap(); // warmup
        c.frame_rendered(6, WARMUP_FRAMES, 0.01, (1280, 720))
            .unwrap();
        c.frame_rendered(7, WARMUP_FRAMES + MEASURED_FRAMES, 0.01, (1280, 720))
            .unwrap();
        c.timings(&gpu(5, vec![pass("a", 1.0)]));
        c.timings(&gpu(7, vec![pass("a", 1.0)]));
        c.timings(&gpu(99, vec![pass("a", 1.0)]));
        assert!(c.finish().gpu_total.is_none());
        c.timings(&gpu(6, vec![pass("a", 1.0)]));
        c.timings(&gpu(6, vec![pass("a", 9.0)]));
        let m = c.finish();
        assert_eq!(m.gpu_total.unwrap().samples, 1);
        assert_eq!(m.passes[0].1.samples, 1);
        assert_eq!(m.passes[0].1.mean_ms, 1.0);
    }

    #[test]
    fn collector_finishes_after_grace_without_all_timings() {
        let mut c = Collector::new((1280, 720), true);
        render_measured(&mut c);
        let end = WARMUP_FRAMES + MEASURED_FRAMES;
        assert!(!c.is_done(end));
        assert!(c.is_done(end + GRACE_FRAMES));
        assert!(c.finish().gpu_total.is_none());
    }

    #[test]
    fn a_repeated_measured_frame_counts_once_and_keeps_the_first_dt() {
        let mut c = Collector::new((1280, 720), false);
        let end = WARMUP_FRAMES + MEASURED_FRAMES;
        c.frame_rendered(1, WARMUP_FRAMES, 0.004, (1280, 720))
            .unwrap();
        // Same bench frame again (same and different frame_index), other dt.
        c.frame_rendered(1, WARMUP_FRAMES, 0.9, (1280, 720))
            .unwrap();
        c.frame_rendered(2, WARMUP_FRAMES, 0.9, (1280, 720))
            .unwrap();
        let cpu = c.finish().cpu_frame;
        assert_eq!(cpu.samples, 1);
        assert!((cpu.median_ms - 4.0).abs() < 1e-4);
        // Repeat one frame until the raw call count reaches MEASURED_FRAMES.
        for i in 0..MEASURED_FRAMES as u64 {
            c.frame_rendered(10 + i, WARMUP_FRAMES, 0.004, (1280, 720))
                .unwrap();
        }
        assert_eq!(c.finish().cpu_frame.samples, 1);
        assert!(!c.is_done(end));
        // The size check still applies to repeats.
        assert!(c.frame_rendered(1, WARMUP_FRAMES, 0.004, (1, 1)).is_err());
    }

    #[test]
    fn grace_does_not_finish_before_frames_are_rendered() {
        let mut c = Collector::new((1280, 720), true);
        c.frame_rendered(1, WARMUP_FRAMES, 0.01, (1280, 720))
            .unwrap();
        assert!(!c.is_done(WARMUP_FRAMES + MEASURED_FRAMES + GRACE_FRAMES));
    }

    #[test]
    fn collector_is_done_when_all_gpu_timings_arrived() {
        let mut c = Collector::new((1280, 720), true);
        render_measured(&mut c);
        let end = WARMUP_FRAMES + MEASURED_FRAMES;
        for f in WARMUP_FRAMES..end {
            assert!(!c.is_done(end));
            c.timings(&gpu(1000 + f as u64, vec![pass("a", 1.0)]));
        }
        assert!(c.is_done(end));
    }

    #[test]
    fn without_gpu_timing_it_is_done_as_soon_as_the_frames_are() {
        let mut c = Collector::new((1280, 720), false);
        let end = WARMUP_FRAMES + MEASURED_FRAMES;
        assert!(!c.is_done(end));
        render_measured(&mut c);
        assert!(c.is_done(end));
    }
}
