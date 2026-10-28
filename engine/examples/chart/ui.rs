//! The chart's control panel: the tone curve under test, the grade, and what
//! the window is presenting to.

use etib::ui::egui;
use etib::{ColorSpace, Display, FrameStats, Gpu, Grade, Output, Tonemap};
use log::info;

use crate::lut::{self, Choice};

/// What each zone of the chart is for, top to bottom. The chart itself carries
/// no text — a compute shader has no font — so the reading goes here.
const ZONES: &[(&str, &str)] = &[
    (
        "1  grey wedge",
        "One patch per stop, -6 to +5 around mid-grey (notched). Watch where the \
         patches stop differing: that is where the curve has run out of range.",
    ),
    (
        "2  1, 2, 4, 8x white",
        "One flat white block on a standard-range surface. A staircase on an \
         extended-range one, unless the curve saturates at SDR white.",
    ),
    (
        "3  hues by intensity",
        "Hue across, intensity down. Clipping marches the rows towards white, a \
         luminance curve holds the hue, a filmic curve desaturates smoothly.",
    ),
    (
        "4  continuous ramp",
        "Zone 1 without the steps, so the shape of the roll-off shows: hard \
         edges, smooth shoulders, and any kink in between.",
    ),
];

/// State the panel owns and the renderer reads back each frame.
pub struct UiState {
    /// What the post-processing pass does with the chart.
    pub grade: Grade,
    /// Which look-up table that grade ends with, if any.
    pub lut: Choice,
    /// The `--lut` file's path, shown in the picker. `None` when none was
    /// given, in which case the picker does not offer that choice.
    lut_file: Option<String>,
    /// The grade's colour filter in sRGB, as edited by the picker.
    tint_srgb: [f32; 3],
    /// Where SDR white sits on this surface, and how far above it the display
    /// can go. Followed automatically unless the sliders take over.
    pub output: Output,
    /// Whether [`UiState::output`] tracks the display.
    follow_display: bool,
    /// How the swapchain was configured.
    color_space: ColorSpace,
    /// The last thing the platform said about the display.
    display: Display,
    /// Whether the zone descriptions are expanded.
    show_legend: bool,
    /// Fraction of the window the panel occupies, measured from the last frame
    /// and handed to the chart so the two do not overlap.
    pub inset: f32,
}

impl UiState {
    pub fn new(gpu: &Gpu, lut: Choice, lut_file: Option<String>) -> Self {
        Self {
            grade: Grade::default(),
            lut,
            lut_file,
            tint_srgb: [1.0, 1.0, 1.0],
            output: gpu.surface.output(),
            follow_display: true,
            color_space: gpu.color_space(),
            display: gpu.surface.display(),
            show_legend: true,
            inset: 0.0,
        }
    }

    /// Re-reads what the display can show, on a timer rather than every frame.
    pub fn poll_display(&mut self, gpu: &Gpu) {
        let display = gpu.surface.display();
        if display == self.display {
            return;
        }

        info!("display changed: {display:?}");
        self.display = display;
        if self.follow_display {
            self.output = Output::detect(self.color_space, &display);
        }
    }
}

pub fn build_panel(root: &mut egui::Ui, state: &mut UiState, stats: &FrameStats) {
    let full = root.max_rect();

    // A panel rather than a floating window, so that what it leaves over is a
    // rectangle the chart can be laid out into instead of being covered by.
    egui::Panel::left("controls")
        .resizable(false)
        .exact_size(330.0)
        .show(root, |ui| {
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(
                    "The chart is drawn scene-referred; everything you see is the curve",
                )
                .small()
                .weak(),
            );
            ui.separator();

            build_curve(ui, state);

            ui.separator();
            build_output(ui, state);

            ui.separator();
            ui.checkbox(&mut state.show_legend, "What am I looking at?");
            if state.show_legend {
                for (title, description) in ZONES {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(*title).strong().small());
                    ui.label(egui::RichText::new(*description).small().weak());
                }
            }

            ui.separator();
            egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
                ui.label("fps");
                ui.label(format!("{:.0}", stats.fps));
                ui.end_row();

                ui.label("surface");
                ui.label(format!("{}x{}", stats.surface_size.0, stats.surface_size.1));
                ui.end_row();

                ui.label("backend");
                ui.label(stats.backend);
                ui.end_row();
            });
        });

    // What the panel left over, as a fraction of the window. The chart picks
    // this up on the next frame, which is soon enough for something that only
    // moves when the window is resized.
    let free = root.available_rect_before_wrap();
    state.inset = if full.width() > 0.0 {
        ((free.min.x - full.min.x) / full.width()).clamp(0.0, 1.0)
    } else {
        0.0
    };
}

/// The curve under test, and the grade applied after it.
fn build_curve(ui: &mut egui::Ui, state: &mut UiState) {
    let grade = &mut state.grade;

    ui.horizontal(|ui| {
        ui.label("tone curve");
        for option in Tonemap::ALL {
            if ui
                .selectable_value(&mut grade.tonemap, option, option.name())
                .clicked()
            {
                info!("tone curve changed to {}", option.name());
            }
        }
    });

    ui.add(
        egui::Slider::new(&mut grade.exposure_ev, -4.0..=4.0)
            .text("exposure")
            .suffix(" EV")
            .fixed_decimals(2),
    )
    .on_hover_text(
        "Slides the whole chart along the curve, which is the quickest way to see its shape.",
    );
    ui.add(
        egui::Slider::new(&mut grade.contrast, 0.5..=2.0)
            .text("contrast")
            .fixed_decimals(2),
    );
    ui.add(
        egui::Slider::new(&mut grade.saturation, 0.0..=2.0)
            .text("saturation")
            .fixed_decimals(2),
    );

    ui.horizontal(|ui| {
        ui.label("filter");
        if ui.color_edit_button_rgb(&mut state.tint_srgb).changed() {
            state.grade.tint = [
                srgb_to_linear(state.tint_srgb[0]),
                srgb_to_linear(state.tint_srgb[1]),
                srgb_to_linear(state.tint_srgb[2]),
            ];
        }
        if ui.button("Reset").clicked() {
            state.tint_srgb = [1.0, 1.0, 1.0];
            let lut_strength = state.grade.lut_strength;
            state.grade = Grade::default();
            // The table is a choice made above the grade rather than part of
            // it, so resetting the sliders leaves it where it was.
            state.grade.lut_strength = lut_strength;
            info!("grade reset");
        }
    });

    ui.add_space(4.0);
    lut::picker(
        ui,
        &mut state.lut,
        &mut state.grade.lut_strength,
        state.lut_file.as_deref(),
    );
}

/// What the window is presenting to, and how the image is placed on it.
fn build_output(ui: &mut egui::Ui, state: &mut UiState) {
    let hdr = state.color_space.is_hdr();

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Output").small().weak());
        ui.label(
            egui::RichText::new(if hdr {
                "extended range (scRGB)"
            } else {
                "standard range (sRGB)"
            })
            .small(),
        );
    });

    egui::Grid::new("display").num_columns(2).show(ui, |ui| {
        ui.label("display");
        ui.label(match state.display.hdr_mode {
            Some(true) => "HDR mode".to_owned(),
            Some(false) => "SDR mode".to_owned(),
            None => "unreported".to_owned(),
        });
        ui.end_row();

        ui.label("SDR white");
        ui.label(match state.display.sdr_white_nits {
            Some(nits) => format!("{nits:.0} nits"),
            None => "unreported".to_owned(),
        });
        ui.end_row();

        ui.label("headroom");
        ui.label(format!("{:.2}x", state.output.headroom));
        ui.end_row();
    });

    if !hdr {
        ui.label(
            egui::RichText::new("Zone 2 will be one flat block. Run without --sdr to separate it.")
                .small()
                .weak(),
        );
        return;
    }

    ui.add_space(4.0);
    if ui
        .checkbox(&mut state.follow_display, "Follow the display")
        .changed()
        && state.follow_display
    {
        state.output = Output::detect(state.color_space, &state.display);
    }

    // The sliders are here to override what the display reported, so they must
    // not quietly rewrite it: egui clamps an existing value to a slider's range
    // by default, even a disabled slider's, which silently capped a 16x
    // headroom at whatever the range happened to end at -- and that capped
    // value is the one the pass tone maps against. `Edits` clamps only what is
    // typed or dragged.
    ui.add_enabled_ui(!state.follow_display, |ui| {
        ui.add(
            egui::Slider::new(&mut state.output.sdr_white, 0.5..=8.0)
                .clamping(egui::SliderClamping::Edits)
                .text("white level")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut state.output.headroom, 1.0..=16.0)
                .clamping(egui::SliderClamping::Edits)
                .text("headroom")
                .fixed_decimals(2),
        );
    });
}

/// sRGB electro-optical transfer function. The picker is sRGB; the filter
/// multiplies linear scene values.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
