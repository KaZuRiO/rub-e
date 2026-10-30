//! Saves and loads a benchmark run as JSON.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::collect::Summary;

/// The scenario a run was measured on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioInfo {
    pub version: u32,
    pub warmup_frames: u32,
    pub measured_frames: u32,
    pub generated_cubes: u32,
    pub kept_cubes: u32,
}

/// The state of the repository when a run was made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GitInfo {
    pub commit: Option<String>,
    pub dirty: bool,
}

/// Everything a run measured, with what is needed to judge a comparison.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunResults {
    pub name: String,
    pub unix_time: u64,
    pub adapter: String,
    pub backend: String,
    pub surface_size: (u32, u32),
    pub git: GitInfo,
    pub scenario: ScenarioInfo,
    pub gpu_timing: bool,
    pub cpu_frame: Summary,
    pub gpu_total: Option<Summary>,
    pub passes: Vec<(String, Summary)>,
}

/// Short commit and dirtiness of this repository; no commit when git fails.
pub fn git_info() -> GitInfo {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|out| out.status.success())
    };
    let commit = git(&["rev-parse", "--short", "HEAD"])
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|commit| !commit.is_empty());
    let dirty = git(&["status", "--porcelain"]).is_some_and(|out| !out.stdout.is_empty());
    GitInfo { commit, dirty }
}

/// Where runs are saved: `bench-results/` at the repository root.
pub fn results_dir() -> PathBuf {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR"));
    examples.parent().unwrap_or(examples).join("bench-results")
}

/// Writes `<dir>/<name>.json`; the flag says whether a file was overwritten.
pub fn save(run: &RunResults, dir: &Path) -> anyhow::Result<(PathBuf, bool)> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(format!("{}.json", run.name));
    let overwrote = path.exists();
    let json = serde_json::to_string_pretty(run)?;
    std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    Ok((path, overwrote))
}

/// Reads `<dir>/<name>.json`.
pub fn load(name: &str, dir: &Path) -> anyhow::Result<RunResults> {
    let path = dir.join(format!("{name}.json"));
    if !path.exists() {
        bail!("no results named '{name}' (expected {})", path.display());
    }
    let json =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn sample_run(name: &str) -> RunResults {
        let s = |m: f32| Summary {
            median_ms: m,
            p95_ms: m * 1.5,
            mean_ms: m,
            samples: 100,
        };
        RunResults {
            name: name.to_string(),
            unix_time: 1_700_000_000,
            adapter: "Test GPU".into(),
            backend: "Vulkan".into(),
            surface_size: (1280, 720),
            git: GitInfo {
                commit: Some("abc1234".into()),
                dirty: false,
            },
            scenario: ScenarioInfo {
                version: 1,
                warmup_frames: 10,
                measured_frames: 100,
                generated_cubes: 5000,
                kept_cubes: 4000,
            },
            gpu_timing: true,
            cpu_frame: s(5.0),
            gpu_total: Some(s(4.5)),
            passes: vec![("scene".into(), s(4.0))],
        }
    }

    fn temp(test: &str) -> PathBuf {
        std::env::temp_dir().join(format!("etib-bench-{}-{test}", std::process::id()))
    }

    #[test]
    fn a_run_survives_a_round_trip() {
        let dir = temp("round-trip");
        let run = sample_run("base");
        let (path, overwrote) = save(&run, &dir).unwrap();
        assert_eq!(path, dir.join("base.json"));
        assert!(!overwrote);
        assert_eq!(load("base", &dir).unwrap(), run);
        let (_, overwrote) = save(&run, &dir).unwrap();
        assert!(overwrote);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_missing_run_names_the_path() {
        let dir = temp("missing");
        let err = load("absent", &dir).unwrap_err().to_string();
        assert!(
            err.contains("absent") && err.contains("absent.json"),
            "{err}"
        );
    }

    #[test]
    fn git_info_does_not_panic() {
        let _ = git_info();
    }
}
