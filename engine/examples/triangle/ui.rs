//! The control panel painted over the triangle.

use etib::ui::egui;
use etib::{ColorSpace, Display, FrameStats, Gpu, Grade, Output, Tonemap};
use log::info;

use crate::lut::{self, Choice};

/// Colours the "Change color" button cycles through, as sRGB hex plus a name
/// for the label.
const PALETTE: &[(&str, [u8; 3])] = &[
    ("Ember", [0xFF, 0x6B, 0x35]),
    ("Viridian", [0x2E, 0xC4, 0x8C]),
    ("Cobalt", [0x4D, 0x8D, 0xFF]),
    ("Amethyst", [0xB0, 0x6A, 0xF5]),
    ("Citrine", [0xFF, 0xC7, 0x3C]),
    ("Rose", [0xFF, 0x5D, 0x8F]),
];

/// State the panel owns and the renderer reads back each frame.
pub struct UiState {
    palette_index: usize,
    /// Current triangle colour in sRGB, as edited by the picker.
    pub color_srgb: [f32; 3],
    pub rotation_speed: f32,
    pub radius: f32,
    /// How far past display range the fill is driven, for the tone curve to
    /// bring back.
    pub intensity: f32,
    pub paused: bool,
    /// What the post-processing pass does with the result.
    pub grade: Grade,
    /// Which table that grade ends with, if any.
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
    /// How the swapchain was configured, which decides whether any of the
    /// extended-range controls mean anything.
    color_space: ColorSpace,
    /// The last thing the platform said about the display.
    display: Display,
}

impl UiState {
    pub fn new(gpu: &Gpu, lut: Choice, lut_file: Option<String>) -> Self {
        Self {
            palette_index: 0,
            color_srgb: srgb_u8_to_f32(PALETTE[0].1),
            rotation_speed: 0.8,
            radius: 0.62,
            intensity: 3.0,
            paused: false,
            grade: Grade::default(),
            lut,
            lut_file,
            tint_srgb: [1.0, 1.0, 1.0],
            output: gpu.surface.output(),
            follow_display: true,
            color_space: gpu.color_space(),
            display: gpu.surface.display(),
        }
    }

    /// Re-reads what the display can show. Called on a timer rather than every
    /// frame: the numbers only move when a setting or the window does.
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

impl UiState {
    /// The triangle colour in linear space, which is what the shader wants.
    pub fn color_linear(&self) -> [f32; 3] {
        [
            srgb_to_linear(self.color_srgb[0]),
            srgb_to_linear(self.color_srgb[1]),
            srgb_to_linear(self.color_srgb[2]),
        ]
    }
}

pub fn build_panel(ui: &mut egui::Ui, state: &mut UiState, stats: &FrameStats) {
    egui::Window::new("Controls")
        .default_pos([16.0, 16.0])
        .resizable(false)
        .show(ui.ctx(), |ui| {
            ui.label(
                egui::RichText::new("Triangle rasterized in a compute shader")
                    .small()
                    .weak(),
            );
            ui.separator();

            ui.horizontal(|ui| {
                // The required feature: one button, cycling the palette.
                if ui.button("Change color").clicked() {
                    state.palette_index = (state.palette_index + 1) % PALETTE.len();
                    let (name, rgb) = PALETTE[state.palette_index];
                    state.color_srgb = srgb_u8_to_f32(rgb);
                    info!("triangle colour changed to {name} {rgb:?}");
                }

                let (name, _) = PALETTE[state.palette_index];
                ui.label(name);
                ui.color_edit_button_rgb(&mut state.color_srgb);
            });

            ui.add_space(4.0);
            ui.checkbox(&mut state.paused, "Pause rotation");
            ui.add(
                egui::Slider::new(&mut state.rotation_speed, -4.0..=4.0)
                    .text("speed")
                    .fixed_decimals(2),
            );
            ui.add(
                egui::Slider::new(&mut state.radius, 0.1..=1.2)
                    .text("size")
                    .fixed_decimals(2),
            );
            ui.add(
                egui::Slider::new(&mut state.intensity, 0.2..=12.0)
                    .text("intensity")
                    .logarithmic(true)
                    .fixed_decimals(2),
            )
            .on_hover_text(
                "Multiplier on the fill colour. Past 1.0 the triangle is brighter than                  the display can show, and what you see is the tone curve below.",
            );

            ui.separator();
            build_grade(ui, state);

            ui.separator();
            build_output(ui, state);

            ui.separator();
            egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
                ui.label("fps");
                ui.label(format!("{:.0}", stats.fps));
                ui.end_row();

                ui.label("frame");
                ui.label(stats.frame_index.to_string());
                ui.end_row();

                ui.label("surface");
                ui.label(format!("{}x{}", stats.surface_size.0, stats.surface_size.1));
                ui.end_row();

                ui.label("backend");
                ui.label(&stats.backend);
                ui.end_row();
            });
        });
}

/// What the window is presenting to, and how the image is placed on it. All of
/// it is detected; the controls are here to check the detection against what
/// the display actually does.
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

        ui.label("peak");
        ui.label(match state.display.peak_nits {
            Some(nits) => format!("{nits:.0} nits"),
            None => "unreported".to_owned(),
        });
        ui.end_row();
    });

    if !hdr {
        // Nothing below applies: white is 1.0 and there is nothing above it.
        return;
    }

    ui.add_space(4.0);
    if ui
        .checkbox(&mut state.follow_display, "Follow the display")
        .on_hover_text(
            "Take the white level and headroom from the platform, and keep taking them as              the brightness slider moves.",
        )
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
        )
        .on_hover_text(
            "What to write for SDR white. scRGB fixes 1.0 at 80 nits, so matching a              desktop at 280 nits means writing 3.5.",
        );
        ui.add(
            egui::Slider::new(&mut state.output.headroom, 1.0..=16.0)
                .clamping(egui::SliderClamping::Edits)
                .text("headroom")
                .fixed_decimals(2),
        )
        .on_hover_text("How far above SDR white highlights may go, as a multiplier.");
    });
}

/// The post-processing controls: the tone curve, then the grade applied to its
/// result.
fn build_grade(ui: &mut egui::Ui, state: &mut UiState) {
    let grade = &mut state.grade;

    ui.label(egui::RichText::new("Post-processing").small().weak());

    ui.horizontal(|ui| {
        ui.label("tone curve");
        egui::ComboBox::from_id_salt("tonemap")
            .selected_text(grade.tonemap.name())
            .show_ui(ui, |ui| {
                for option in Tonemap::ALL {
                    if ui
                        .selectable_value(&mut grade.tonemap, option, option.name())
                        .clicked()
                    {
                        info!("tone curve changed to {}", option.name());
                    }
                }
            });
    });

    ui.add(
        egui::Slider::new(&mut grade.exposure_ev, -4.0..=4.0)
            .text("exposure")
            .suffix(" EV")
            .fixed_decimals(2),
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
            // The picker is sRGB; the filter multiplies linear scene values.
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

fn srgb_u8_to_f32(rgb: [u8; 3]) -> [f32; 3] {
    [
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    ]
}

/// sRGB electro-optical transfer function. The picker and palette are in sRGB;
/// the compute shader writes into a linear texture, so convert once on the CPU.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
