//! Replayable benchmark: a deterministic cube world seen along a fixed camera
//! path, so two runs of the engine can be compared frame for frame.

// The bench app of a later task uses the rest of the scenario.
#[expect(dead_code, reason = "used by the bench app in a later task")]
mod scenario;

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the bench app in a later task")
)]
mod collect;

fn main() {
    println!("{} cubes", scenario::world().len());
}
