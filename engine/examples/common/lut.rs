//! The `--lut` argument, the table it names, and the panel row that picks
//! between them. Shared by both examples, which offer the same choice.
//!
//! A real table arrives as a `.cube` file and [`Lut::from_cube`] reads it —
//! which is what `--lut <file>` does. The one built here is so the stage can be
//! seen working without going and finding such a file first, and because a
//! grade written as arithmetic says plainly what a table does: it is a function
//! from colour to colour, sampled on a grid.

use anyhow::{Context as _, Result};
use etib::ui::egui;
use etib::{Device, Lut, PostProcess, TextureView};
use log::info;

/// A look built into the examples, which `--lut` can name in place of a path.
struct BuiltIn {
    /// What `--lut` takes for it.
    name: &'static str,
    /// What the picker shows.
    label: &'static str,
    /// The look itself: what one display-referred colour becomes.
    look: fn([f32; 3]) -> [f32; 3],
}

/// The built-in looks, in the order the picker lists them.
const BUILT_IN: &[BuiltIn] = &[BuiltIn {
    name: "teal",
    label: "Teal and orange",
    look: teal_and_orange,
}];

/// Which table the post-processing pass is grading through.
///
/// The tables themselves belong to [`Tables`]; the panel only picks one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// No table at all, which is the pass's own default.
    None,
    /// One of the looks built below, by its place in [`BUILT_IN`].
    BuiltIn(usize),
    /// The `.cube` file named on the command line. Only offered when there was
    /// one.
    File,
}

/// The tables an example can grade through, and which of them is uploaded.
///
/// Uploading is not free, so [`Tables::apply`] does it when the choice changes
/// rather than every frame.
///
/// Clonable because the runner may build the application more than once, if the
/// window is torn down and comes back.
#[derive(Clone)]
pub struct Tables {
    /// One table per entry of [`BUILT_IN`], baked at startup.
    built_in: Vec<Lut>,
    /// The `--lut` file, with the path it came from for the panel's label.
    file: Option<(String, Lut)>,
    /// What the command line asked for, which is what the panel starts on.
    initial: Choice,
    /// What is on the GPU. Nothing, until the first [`apply`](Self::apply).
    applied: Choice,
}

impl Tables {
    /// Reads `--lut`, which takes the name of a built-in look — `teal` — or
    /// the path to a `.cube` file.
    ///
    /// Reading happens before the window opens: a table that cannot be parsed
    /// is a mistake in the command line, and saying so straight away beats
    /// opening a window that quietly grades through nothing.
    pub fn from_args() -> Result<Self> {
        let argument = argument();
        let built_in = argument.as_deref().and_then(|name| {
            BUILT_IN
                .iter()
                .position(|look| look.name.eq_ignore_ascii_case(name))
        });

        let (file, initial) = match (argument.as_deref(), built_in) {
            (None, _) => (None, Choice::None),
            (Some(_), Some(index)) => (None, Choice::BuiltIn(index)),
            // Not a name, so it has to be a path.
            (Some(path), None) => {
                let names: Vec<_> = BUILT_IN.iter().map(|look| look.name).collect();
                let text = std::fs::read_to_string(path).with_context(|| {
                    format!(
                        "reading the lookup table at {path} (the built-in looks are {})",
                        names.join(", ")
                    )
                })?;
                let lut = Lut::from_cube(&text)
                    .with_context(|| format!("parsing the lookup table at {path}"))?;
                info!("loaded a {}³ lookup table from {path}", lut.size());
                (Some((path.to_owned(), lut)), Choice::File)
            }
        };

        Ok(Self {
            built_in: BUILT_IN.iter().map(|look| bake(look.look)).collect(),
            file,
            initial,
            applied: Choice::None,
        })
    }

    /// The choice to start the panel on: whatever `--lut` named.
    pub fn initial(&self) -> Choice {
        self.initial
    }

    /// The `--lut` file's path, for the panel's label, if there was one.
    pub fn file_name(&self) -> Option<&str> {
        self.file.as_ref().map(|(path, _)| path.as_str())
    }

    /// Uploads the chosen table, when the choice has moved since it last was.
    ///
    /// `source` is the texture the pass reads, which
    /// [`PostProcess::set_lut`] wants for the same reason
    /// [`PostProcess::rebind`] does.
    pub fn apply(
        &mut self,
        post: &mut PostProcess,
        device: &Device,
        source: &TextureView,
        choice: Choice,
    ) {
        if choice == self.applied {
            return;
        }

        self.applied = choice;
        let lut = match choice {
            Choice::None => None,
            Choice::BuiltIn(index) => self.built_in.get(index),
            Choice::File => self.file.as_ref().map(|(_, lut)| lut),
        };
        post.set_lut(device, source, lut);
    }
}

/// The argument to `--lut`, if it was given.
fn argument() -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == "--lut" {
            return args.next();
        }
    }
    None
}

/// The panel row: which table, and how much of it.
pub fn picker(ui: &mut egui::Ui, choice: &mut Choice, strength: &mut f32, file: Option<&str>) {
    let name = |choice: Choice| match choice {
        Choice::None => "None",
        Choice::BuiltIn(index) => BUILT_IN[index].label,
        // The whole path would push the panel wide; the file name is what
        // identifies it anyway.
        Choice::File => file.map_or("File", file_name),
    };

    ui.horizontal(|ui| {
        ui.label("table");

        let selected = *choice;
        egui::ComboBox::from_id_salt("lut")
            .selected_text(name(selected))
            .show_ui(ui, |ui| {
                let mut choices = vec![Choice::None];
                choices.extend((0..BUILT_IN.len()).map(Choice::BuiltIn));
                if file.is_some() {
                    choices.push(Choice::File);
                }
                for option in choices {
                    if ui
                        .selectable_label(selected == option, name(option))
                        .clicked()
                    {
                        *choice = option;
                        info!("lookup table changed to {}", name(option));
                    }
                }
            });
    })
    .response
    .on_hover_text(
        "A cube of colours a grade has been baked into, applied after the curve and the \
         sliders above. --lut <name> starts on one of the built-in looks, --lut \
         <file.cube> on one exported from a grading tool.",
    );

    ui.add_enabled(
        *choice != Choice::None,
        egui::Slider::new(strength, 0.0..=1.0)
            .text("strength")
            .fixed_decimals(2),
    )
    .on_hover_text("How much of the table to mix in. At 0.0 it is bypassed entirely.");
}

/// The last component of a path, for a label that has to stay narrow.
fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Entries a side. Seventeen is the smallest size in common use, and plenty for
/// a grade this smooth; a table with hard transitions in it wants 32 or 64.
const SIZE: u32 = 17;

const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Bakes a look into a table by evaluating it at every entry of the cube.
///
/// Both what a look is handed and what it returns are gamma-encoded, as every
/// table's are — which is also the space these looks want to work in, since a
/// tint of a tenth means a tenth of the way up what the eye sees rather than a
/// tenth of the light.
fn bake(look: fn([f32; 3]) -> [f32; 3]) -> Lut {
    let step = (SIZE - 1) as f32;

    // Red varies fastest and blue slowest, which is the order both `.cube`
    // files and `Lut::from_rgb` use.
    let mut rgb = Vec::with_capacity((SIZE as usize).pow(3) * 3);
    for b in 0..SIZE {
        for g in 0..SIZE {
            for r in 0..SIZE {
                let input = [r as f32 / step, g as f32 / step, b as f32 / step];
                // A table holds display-referred colours, so its entries live
                // in 0..1 however far a look would otherwise push them.
                rgb.extend(look(input).map(|channel| channel.clamp(0.0, 1.0)));
            }
        }
    }

    Lut::from_rgb(SIZE, &rgb).expect("the table is filled to the size it declares")
}

fn luma(color: [f32; 3]) -> f32 {
    color[0] * LUMA[0] + color[1] * LUMA[1] + color[2] * LUMA[2]
}

/// Teal and orange: the blockbuster grade, where the cool half of the frame is
/// pulled towards teal and the warm half towards orange until it holds little
/// else.
///
/// It works in two passes, because the look is really two things at once. The
/// first sorts colours by hue and steers each towards the nearer of two
/// anchors, which is what carries skin towards orange and sky towards teal. The
/// second sorts them by brightness and tints the two ends, which is what
/// reaches the neutrals — a white highlight has no hue to steer, so without
/// this half it would come out exactly as white as it went in.
fn teal_and_orange(color: [f32; 3]) -> [f32; 3] {
    /// The two anchors, as ordinary colours; what matters is their hue, since
    /// each is rescaled to the brightness of the colour being steered.
    const TEAL: [f32; 3] = [0.06, 0.50, 0.58];
    const ORANGE: [f32; 3] = [0.95, 0.51, 0.16];
    /// How far a fully saturated colour is carried towards its anchor.
    const STRENGTH: f32 = 0.6;
    /// Lifted into the shadows, where there is room below for an addition and
    /// an addition is what reads as a tint.
    const SHADOW_LIFT: [f32; 3] = [-0.01, 0.03, 0.07];
    /// Multiplied into the highlights, where a lift would only clip the red it
    /// was trying to add. Taking green and blue away instead is what a warming
    /// filter does, and it cannot push anything out of range.
    const HIGHLIGHT_FILTER: [f32; 3] = [1.0, 0.96, 0.86];

    let brightness = luma(color);
    let peak = color[0].max(color[1]).max(color[2]);
    let base = color[0].min(color[1]).min(color[2]);
    // How much hue there is to work with at all. A grey has none, and is left
    // entirely to the second pass.
    let chroma = peak - base;

    // Which side of the warm/cool axis the colour sits on, and how clearly:
    // red against blue separates the two well enough, and dividing by the
    // chroma makes it a direction rather than an amount.
    let warmth = (color[0] - color[2]) / chroma.max(1e-4);
    let anchor = if warmth >= 0.0 { ORANGE } else { TEAL };

    // Steering the hue should not also change the exposure, so the anchor is
    // scaled to the brightness of the colour it is being mixed into.
    let scale = brightness / luma(anchor);
    let amount = (STRENGTH * chroma * warmth.abs()).clamp(0.0, 1.0);

    // How far the first pass has already committed this colour one way, so the
    // second can leave it alone rather than fight it: a warm shadow stays warm,
    // and a cool highlight stays cool, which is the whole reason the hue pass
    // comes first.
    let warm_bias = warmth.max(0.0) * chroma;
    let cool_bias = (-warmth).max(0.0) * chroma;
    // Squared so the tints stay out of the midtones and take hold at the ends.
    let shadows = (1.0 - brightness) * (1.0 - brightness) * (1.0 - warm_bias);
    let highlights = brightness * brightness * (1.0 - cool_bias);

    std::array::from_fn(|channel| {
        let steered = color[channel] * (1.0 - amount) + anchor[channel] * scale * amount;
        let filtered = steered * (1.0 + (HIGHLIGHT_FILTER[channel] - 1.0) * highlights);
        filtered + SHADOW_LIFT[channel] * shadows
    })
}
