//! Side-by-side terminal table of two runs.

use std::fmt::Write;

use crate::collect::Summary;
use crate::results::RunResults;

const DASH: &str = "—";

fn commit_label(run: &RunResults) -> String {
    match &run.git.commit {
        Some(c) if run.git.dirty => format!("{c}+dirty"),
        Some(c) => c.clone(),
        None => "no git".to_string(),
    }
}

fn header_line(run: &RunResults) -> String {
    format!(
        "{} ({}, {} cubes generated, {} kept)",
        run.name,
        commit_label(run),
        run.scenario.generated_cubes,
        run.scenario.kept_cubes
    )
}

/// One table row: label, then median and p95 of each side, then the deltas.
fn row(label: &str, a: Option<&Summary>, b: Option<&Summary>) -> [String; 7] {
    let ms = |v: Option<f32>| v.map_or_else(|| DASH.to_string(), |v| format!("{v:.2}"));
    let (delta_ms, delta_pct) = match (a, b) {
        (Some(a), Some(b)) => {
            let d = b.median_ms - a.median_ms;
            let pct = if a.median_ms == 0.0 {
                String::new()
            } else {
                format!("{:+.1}%", d / a.median_ms * 100.0)
            };
            (format!("{d:+.2}"), pct)
        }
        _ => (String::new(), String::new()),
    };
    [
        label.to_string(),
        ms(a.map(|s| s.median_ms)),
        ms(a.map(|s| s.p95_ms)),
        ms(b.map(|s| s.median_ms)),
        ms(b.map(|s| s.p95_ms)),
        delta_ms,
        delta_pct,
    ]
}

/// Renders the comparison of `a` (the base) and `b` as text.
pub fn compare(a: &RunResults, b: &RunResults) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "a: {}", header_line(a));
    let _ = writeln!(out, "b: {}", header_line(b));

    if a.scenario.version != b.scenario.version
        || a.scenario.warmup_frames != b.scenario.warmup_frames
        || a.scenario.measured_frames != b.scenario.measured_frames
    {
        let _ = writeln!(
            out,
            "warning: not the same scenario, the numbers are not comparable"
        );
    }
    if a.adapter != b.adapter || a.backend != b.backend || a.surface_size != b.surface_size {
        let _ = writeln!(out, "warning: not the same machine or resolution");
    }
    for (side, run) in [("a", a), ("b", b)] {
        if !run.gpu_timing {
            let _ = writeln!(
                out,
                "note: run {side} has no GPU timestamps, GPU rows are empty"
            );
        }
    }
    out.push('\n');

    let mut rows = vec![
        ["", "median a", "p95 a", "median b", "p95 b", "Δ ms", "Δ %"].map(String::from),
        row("frame CPU", Some(&a.cpu_frame), Some(&b.cpu_frame)),
        row("GPU total", a.gpu_total.as_ref(), b.gpu_total.as_ref()),
    ];
    fn find<'a>(run: &'a RunResults, label: &str) -> Option<&'a Summary> {
        run.passes.iter().find(|(l, _)| l == label).map(|(_, s)| s)
    }
    let labels = a.passes.iter().map(|(l, _)| l).chain(
        b.passes
            .iter()
            .map(|(l, _)| l)
            .filter(|l| !a.passes.iter().any(|(m, _)| m == *l)),
    );
    for label in labels {
        rows.push(row(label, find(a, label), find(b, label)));
    }

    let widths: Vec<usize> = (0..7)
        .map(|c| rows.iter().map(|r| r[c].chars().count()).max().unwrap_or(0))
        .collect();
    for r in &rows {
        let mut line = String::new();
        for (c, cell) in r.iter().enumerate() {
            let pad = widths[c] - cell.chars().count();
            if c == 0 {
                let _ = write!(line, "{cell}{}", " ".repeat(pad));
            } else {
                let _ = write!(line, "  {}{cell}", " ".repeat(pad));
            }
        }
        let _ = writeln!(out, "{}", line.trim_end());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::tests::sample_run;

    fn run(name: &str, scene_ms: f32) -> RunResults {
        let mut r = sample_run(name);
        r.passes[0].1.median_ms = scene_ms;
        r
    }

    #[test]
    fn compare_shows_the_difference_per_pass() {
        let out = compare(&run("a", 4.0), &run("b", 3.0));
        println!("{out}");
        assert!(
            out.contains("scene") && out.contains("-1.00") && out.contains("-25.0%"),
            "{out}"
        );
        assert!(out.contains("5000") && out.contains("4000"), "{out}");
    }

    #[test]
    fn compare_warns_on_a_different_scenario_but_not_on_kept_cubes() {
        let a = run("a", 4.0);
        let mut b = run("b", 4.0);
        b.scenario.kept_cubes += 7;
        assert!(!compare(&a, &b).contains("not the same"));
        b.scenario.version += 1;
        assert!(compare(&a, &b).contains("not the same scenario"));
    }

    #[test]
    fn compare_warns_on_a_different_resolution() {
        let a = run("a", 4.0);
        let mut b = run("b", 4.0);
        b.surface_size = (1920, 1080);
        assert!(compare(&a, &b).contains("not the same machine or resolution"));
    }

    #[test]
    fn compare_with_one_side_missing_gpu() {
        let a = run("a", 4.0);
        let mut b = run("b", 4.0);
        b.gpu_timing = false;
        b.gpu_total = None;
        b.passes.clear();
        let out = compare(&a, &b);
        println!("{out}");
        let line = out.lines().find(|l| l.contains("GPU total")).unwrap();
        assert!(line.contains('—'));
        assert!(out.contains("timestamp"), "{out}");
        assert!(!out.contains("NaN") && !out.contains("inf"), "{out}");
    }

    #[test]
    fn a_label_only_in_one_run_gets_a_dash() {
        let a = run("a", 4.0);
        let mut b = run("b", 4.0);
        b.passes.push(("cull".into(), b.passes[0].1));
        let out = compare(&a, &b);
        let line = out.lines().find(|l| l.contains("cull")).unwrap();
        assert!(line.contains('—'), "{out}");
    }

    #[test]
    fn zero_base_median_leaves_percent_empty() {
        let out = compare(&run("a", 0.0), &run("b", 1.0));
        assert!(!out.contains("NaN") && !out.contains("inf"), "{out}");
        assert!(out.contains("+1.00"));
    }
}
