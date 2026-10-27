//! The menus and the power-up bars.
//!
//! The menus only say what was clicked; the game decides what that does.

use etib::ui::egui::{
    self, Align, Align2, Area, Button, Color32, Frame, Id, Layout, ProgressBar, RichText, Ui,
    Window,
};

use crate::state::{
    DASH_COOLDOWN, DASH_DURATION, LOB_COOLDOWN, Player, SPEED_BOOST_COOLDOWN, SPEED_BOOST_DURATION,
};

/// A button that was clicked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiAction {
    Start,
    Resume,
    Restart,
    Quit,
}

/// START GAME and EXIT, in the middle of the screen.
pub fn main_menu(ctx: &egui::Context) -> Option<UiAction> {
    let mut action = None;
    Area::new(Id::new("Main Menu"))
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(100.0);
                let size = egui::vec2(250.0, 60.0);
                if big_button(ui, size, "START GAME", 28.0) {
                    action = Some(UiAction::Start);
                }
                ui.add_space(15.0);
                if big_button(ui, size, "EXIT", 28.0) {
                    action = Some(UiAction::Quit);
                }
                ui.add_space(20.0);
            });
        });
    action
}

/// PAUSED, with RESUME, RESTART and QUIT.
pub fn pause_menu(ctx: &egui::Context) -> Option<UiAction> {
    let mut action = None;
    Window::new("Pause Menu")
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .collapsible(false)
        .resizable(false)
        .title_bar(false)
        .frame(Frame::NONE.fill(Color32::from_black_alpha(100)))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(
                    RichText::new("PAUSED")
                        .size(40.0)
                        .strong()
                        .color(Color32::from_rgb(255, 180, 50)),
                );
                ui.add_space(30.0);
                let size = egui::vec2(220.0, 50.0);
                for (label, choice) in [
                    ("RESUME", UiAction::Resume),
                    ("RESTART", UiAction::Restart),
                    ("QUIT", UiAction::Quit),
                ] {
                    if big_button(ui, size, label, 24.0) {
                        action = Some(choice);
                    }
                    ui.add_space(if choice == UiAction::Quit { 20.0 } else { 10.0 });
                }
            });
        });
    action
}

/// Each player's boost, dash and lob along the bottom of the screen.
pub fn hud(ui: &mut Ui, left: &Player, right: &Player) {
    egui::Panel::bottom("powerups")
        .frame(
            Frame::NONE
                .fill(Color32::from_black_alpha(150))
                .inner_margin(10.0),
        )
        .show(ui, |ui| {
            ui.columns(2, |cols| {
                cols[0].vertical_centered(|ui| {
                    ui.heading(RichText::new("PLAYER 1").color(Color32::from_rgb(50, 190, 255)));
                    ui.horizontal(|ui| {
                        for bar in [boost(left), dash(left), lob(left)] {
                            ui.add_space(20.0);
                            show(ui, bar);
                        }
                    });
                });

                cols[1].with_layout(Layout::top_down(Align::Max), |ui| {
                    ui.heading(RichText::new("PLAYER 2").color(Color32::from_rgb(255, 115, 25)));
                    ui.horizontal(|ui| {
                        // Laid out from the right edge, so the order mirrors
                        // player 1's.
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            for bar in [dash(right), boost(right), lob(right)] {
                                ui.add_space(20.0);
                                show(ui, bar);
                            }
                        });
                    });
                });
            });
        });
}

fn big_button(ui: &mut Ui, size: egui::Vec2, label: &str, text_size: f32) -> bool {
    ui.add_sized(size, Button::new(RichText::new(label).size(text_size)))
        .clicked()
}

/// How full a bar is, what it says, and its colour.
type Bar = (f32, String, Color32);

fn boost(p: &Player) -> Bar {
    if p.boost_timer > 0.0 {
        (
            p.boost_timer / SPEED_BOOST_DURATION,
            format!("BOOST: ACTIVE {:.1}s", p.boost_timer),
            Color32::GOLD,
        )
    } else if p.cooldown_timer > 0.0 {
        (
            1.0 - (p.cooldown_timer / SPEED_BOOST_COOLDOWN),
            format!("BOOST: COOLDOWN {:.1}s", p.cooldown_timer),
            Color32::GRAY,
        )
    } else {
        (1.0, "BOOST: READY".to_owned(), Color32::GREEN)
    }
}

fn dash(p: &Player) -> Bar {
    if p.dash_timer > 0.0 {
        (
            p.dash_timer / DASH_DURATION,
            "DASHING!".to_owned(),
            Color32::WHITE,
        )
    } else if p.dash_cooldown > 0.0 {
        (
            1.0 - (p.dash_cooldown / DASH_COOLDOWN),
            format!("DASH: {:.1}s", p.dash_cooldown),
            Color32::DARK_GRAY,
        )
    } else {
        (
            1.0,
            "DASH: READY".to_owned(),
            Color32::from_rgb(100, 255, 100),
        )
    }
}

fn lob(p: &Player) -> Bar {
    if p.lob_cooldown > 0.0 {
        (
            1.0 - (p.lob_cooldown / LOB_COOLDOWN),
            format!("LOB: {:.1}s", p.lob_cooldown),
            Color32::DARK_GRAY,
        )
    } else {
        (
            1.0,
            "LOB: READY".to_owned(),
            Color32::from_rgb(180, 100, 255),
        )
    }
}

fn show(ui: &mut Ui, (progress, text, color): Bar) {
    ui.add(
        ProgressBar::new(progress)
            .text(RichText::new(text).color(Color32::BLACK))
            .fill(color)
            .desired_width(180.0),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_follow_the_timers() {
        let mut p = Player::left();
        assert_eq!(boost(&p).1, "BOOST: READY");
        p.boost_timer = SPEED_BOOST_DURATION / 2.0;
        assert_eq!(boost(&p).0, 0.5);
        p.boost_timer = 0.0;
        p.cooldown_timer = SPEED_BOOST_COOLDOWN;
        assert_eq!(boost(&p).0, 0.0);
        assert!(boost(&p).1.starts_with("BOOST: COOLDOWN"));

        p.dash_timer = DASH_DURATION;
        assert_eq!(dash(&p).1, "DASHING!");
        p.lob_cooldown = LOB_COOLDOWN / 4.0;
        assert_eq!(lob(&p).0, 0.75);
    }
}
