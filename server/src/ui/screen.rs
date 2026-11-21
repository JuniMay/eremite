//! The terminal UI's whole screen: the chrome, the view shown in it, and the
//! keys row.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::Line;
use unicode_width::UnicodeWidthStr;

use super::draw::plain;
use super::frame::{
    Choices, Keys, Kind, ModelInfo, Screen, SubView, View, draw_chrome, list_chrome_keys,
    list_model_keys,
};
use super::history::draw_history;
use super::input::{find_delete_question, list_view_keys};
use super::llm::experts::{draw_experts, has_no_experts};
use super::llm::states::draw_states;
use super::log::LogLine;
use super::memory::draw_memory;
use super::overview::draw_overview;

/// Draws the whole screen in `area`: the chrome, the view that `choices`
/// shows and the keys row. 4 Models shows the chosen sub-view of the chosen
/// model's kind. The log of 1 Overview shows `log_lines`. The places that a
/// click chooses go to `choices`, in place of those of the frame before.
pub fn draw_screen(
    buf: &mut Buffer,
    area: Rect,
    screen: &Screen,
    log_lines: &[LogLine],
    choices: &mut Choices,
) {
    choices.targets.clear();
    let keys = list_keys(screen, choices, area.width as usize);
    let view_area = draw_chrome(buf, area, screen, choices, &keys);
    if view_area.height == 0 {
        return;
    }
    match choices.view {
        View::Overview => draw_overview(buf, view_area, screen, log_lines, choices),
        View::Memory => draw_memory(buf, view_area, screen, choices),
        View::History => draw_history(buf, view_area, screen),
        View::Models => {
            let Some(model) = screen.models.get(choices.model) else {
                return;
            };
            match (model.kind, choices.sub_view) {
                (Kind::Llm, SubView::States) => draw_states(buf, view_area, screen, choices),
                (Kind::Llm, SubView::Experts) => draw_experts(buf, view_area, screen, choices),
                (Kind::Decision, _) => draw_decision_model(buf, view_area, model),
                (Kind::Image, _) => draw_image_model(buf, view_area, model),
                (Kind::Video, _) => draw_video_model(buf, view_area, model),
            }
        }
    }
}

/// Draws the one line of a decision model in 4 Models in `area`.
fn draw_decision_model(buf: &mut Buffer, area: Rect, model: &ModelInfo) {
    if area.height == 0 {
        return;
    }
    let line = Line::from(plain(format!(
        " {} is a decision model: it keeps no sessions and has no experts.",
        model.name
    )));
    buf.set_line(area.x, area.y, &line, area.width);
}

/// Draws the one line of an image model in 4 Models in `area`.
fn draw_image_model(buf: &mut Buffer, area: Rect, model: &ModelInfo) {
    if area.height == 0 {
        return;
    }
    let line = Line::from(plain(format!(
        " {} is an image model: it keeps no sessions and has no experts.",
        model.name
    )));
    buf.set_line(area.x, area.y, &line, area.width);
}

/// Draws the one line of a video model in 4 Models in `area`.
fn draw_video_model(buf: &mut Buffer, area: Rect, model: &ModelInfo) {
    if area.height == 0 {
        return;
    }
    let line = Line::from(plain(format!(
        " {} is a video model: it keeps no sessions and has no experts.",
        model.name
    )));
    buf.set_line(area.x, area.y, &line, area.width);
}

/// Returns the keys of the keys row for `choices` in `columns`: the chrome's
/// first keys, the model's keys in 4 Models, the view's own keys, and the
/// chrome's last keys. A view that shows only its one line, such as the
/// Experts of a model without experts or a decision model in 4 Models, has no
/// keys of its own. When the row does not fit, the model's and the view's keys
/// go, from the last one, so that the chrome's keys stay. While d asks to
/// delete a state, its question takes the place of the model's and the view's
/// keys, so that even a narrow row shows it.
fn list_keys(screen: &Screen, choices: &Choices, columns: usize) -> Keys {
    let (first, last) = list_chrome_keys(choices);
    if let Some(question) = find_delete_question(screen, choices) {
        return first.into_iter().chain([question]).chain(last).collect();
    }
    let mut own = Vec::new();
    if choices.view == View::Models {
        own.extend(list_model_keys(screen, choices));
    }
    let has_no_sub_views = screen
        .models
        .get(choices.model)
        .is_some_and(|model| matches!(model.kind, Kind::Decision | Kind::Image | Kind::Video));
    let shows_one_line = choices.view == View::Models
        && (has_no_sub_views
            || choices.sub_view == SubView::Experts
                && screen
                    .statuses
                    .get(choices.model)
                    .is_some_and(has_no_experts));
    if !shows_one_line {
        own.extend(list_view_keys(screen, choices));
    }
    loop {
        let row: Keys = first.iter().chain(&own).chain(&last).cloned().collect();
        if own.is_empty() || measure_keys_row(&row) <= columns {
            return row;
        }
        own.pop();
    }
}

/// Returns the columns of the keys row of `keys`, as the chrome draws it: a
/// space at the start, and after each key a space, what the key does and
/// three spaces.
fn measure_keys_row(keys: &[(&'static str, String)]) -> usize {
    1 + keys
        .iter()
        .map(|(key, what)| key.width() + 1 + what.width() + 3)
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{ModelInfo, Target};
    use crate::ui::samples::Samples;
    use crate::ui::status::ModelStatus;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, read_row, read_screen};
    use crate::ui::theme::DARK;

    /// Each view and sub-view, a text of its first row under the chrome, and
    /// its keys row at 160 and at 100 columns.
    const SHOWN: [(View, SubView, &str, &str, &str); 5] = [
        (
            View::Overview,
            SubView::States,
            "─ System ─",
            " 1–4 views   ↑↓ PgUp PgDn scroll   Home End oldest, newest   f filter   m mouse: \
             clicks   q stop the server",
            " 1–4 views   ↑↓ PgUp PgDn scroll   Home End oldest, newest   m mouse: clicks   q \
             stop the server",
        ),
        (
            View::Memory,
            SubView::States,
            " Mac  128 GiB installed",
            " 1–4 views   m mouse: clicks   q stop the server",
            " 1–4 views   m mouse: clicks   q stop the server",
        ),
        (
            View::History,
            SubView::States,
            "▼ a request arrived",
            " 1–4 views   m mouse: clicks   q stop the server",
            " 1–4 views   m mouse: clicks   q stop the server",
        ),
        (
            View::Models,
            SubView::States,
            " 1 session",
            " 1–4 views   ←→ model   Tab States or Experts   ↑↓ or click session   PgUp PgDn \
             text   d session 5 is in use   m mouse: clicks   q stop the server",
            " 1–4 views   ←→ model   Tab States or Experts   m mouse: clicks   q stop the server",
        ),
        (
            View::Models,
            SubView::Experts,
            " hit rate",
            " 1–4 views   ←→ model   Tab States or Experts   [ ] or click request   ↑↓ or wheel \
             scroll layers   m mouse: clicks   q stop the server",
            " 1–4 views   ←→ model   Tab States or Experts   m mouse: clicks   q stop the server",
        ),
    ];

    /// The statuses of two models in a Mac of 128 GiB: deepseek-v4.1-flash
    /// decodes request 5 in session 5, and qwen3.8-27b, a model without
    /// experts, is idle.
    fn statuses() -> Vec<ModelStatus> {
        let deepseek = r#"{"loaded": true, "phase": "decode",
            "requests": [{"request": 5, "status": "decoding", "session": 5,
                          "prompt_tokens": 400, "generated_tokens": 116}],
            "sessions": [{"session": 5, "tokens": 516, "busy": true}],
            "engine_counters": {"layers": 43, "experts_per_row": 240, "routed_layers": 40,
                "experts_per_layer": 384, "memory": {"total": 137438953472}}}"#;
        let qwen = r#"{"loaded": true, "engine_counters": {"layers": 64, "experts_per_row": 0}}"#;
        [deepseek, qwen]
            .map(|json| serde_json::from_str(json).expect("the worker's form"))
            .to_vec()
    }

    /// Draws the screen at `width` x `height` with `choices`.
    fn draw(width: u16, height: u16, choices: &mut Choices) -> Buffer {
        draw_models(width, height, choices, [Kind::Llm, Kind::Llm])
    }

    /// Draws the screen at `width` x `height` with `choices`, with the two
    /// models of `statuses` of the kinds `kinds`.
    fn draw_models(width: u16, height: u16, choices: &mut Choices, kinds: [Kind; 2]) -> Buffer {
        draw_statuses(width, height, choices, kinds, &statuses())
    }

    /// Draws the screen at `width` x `height` with `choices`, with two models
    /// of the kinds `kinds` and the statuses `statuses`.
    fn draw_statuses(
        width: u16,
        height: u16,
        choices: &mut Choices,
        kinds: [Kind; 2],
        statuses: &[ModelStatus],
    ) -> Buffer {
        let models =
            [("deepseek-v4.1-flash", kinds[0]), ("qwen3.8-27b", kinds[1])].map(|(name, kind)| {
                ModelInfo {
                    name: name.into(),
                    kind,
                }
            });
        let mut samples = Samples::new(WINDOWS);
        samples.add_sample(0.0, statuses);
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 57.0,
            models: &models,
            statuses,
            samples: &samples,
            home: "",
            session_text: None,
        };
        let log_lines = [LogLine {
            time: "19:19:40.184".into(),
            model: Some(0),
            text: "req 5: 245 tokens, 24.5 tokens/s".into(),
        }];
        draw_on_terminal(width, height, |buf, area| {
            draw_screen(buf, area, &screen, &log_lines, choices)
        })
    }

    #[test]
    fn every_view_draws_under_the_chrome_with_its_keys() {
        for (view, sub_view, first_text, wide_keys, narrow_keys) in SHOWN {
            let mut choices = Choices {
                view,
                sub_view,
                ..Choices::default()
            };
            let buf = draw(160, 45, &mut choices);
            assert!(read_row(&buf, 0).starts_with(" eremite  127.0.0.1:52878"));
            let first_row = if view == View::Models { 4 } else { 3 };
            assert!(
                read_row(&buf, first_row).contains(first_text),
                "{view:?} {sub_view:?}:\n{}",
                read_screen(&buf)
            );
            assert_eq!(read_row(&buf, 44).trim_end(), wide_keys);
            // the view's last keys give way to the chrome's keys
            let buf = draw(100, 30, &mut choices);
            assert!(read_row(&buf, first_row).contains(first_text));
            assert_eq!(read_row(&buf, 29).trim_end(), narrow_keys);
            for (width, height) in [(30, 8), (1, 1)] {
                let buf = draw(width, height, &mut choices);
                let title: String = " eremite".chars().take(width as usize).collect();
                assert!(read_row(&buf, 0).starts_with(&title));
            }
        }
    }

    #[test]
    fn the_question_of_the_d_key_shows_in_a_narrow_keys_row() {
        let mut statuses = statuses();
        // deepseek-v4.1-flash's session 5 is idle.
        statuses[0].requests.clear();
        statuses[0].sessions[0].busy = false;
        let mut choices = Choices {
            view: View::Models,
            delete_asked: Some(5),
            ..Choices::default()
        };
        let question =
            " 1–4 views   d again deletes session 5   m mouse: clicks   q stop the server";
        for (width, height) in [(160, 45), (100, 30)] {
            let buf = draw_statuses(width, height, &mut choices, [Kind::Llm; 2], &statuses);
            assert_eq!(read_row(&buf, height - 1).trim_end(), question, "{width}");
        }
        // a row narrower than the question's row still shows the question
        let buf = draw_statuses(60, 20, &mut choices, [Kind::Llm; 2], &statuses);
        let row = read_row(&buf, 19);
        assert!(row.contains("d again deletes session 5"), "{row}");
    }

    #[test]
    fn experts_of_a_model_without_experts_has_no_keys_of_its_own() {
        let mut choices = Choices {
            view: View::Models,
            model: 1,
            sub_view: SubView::Experts,
            ..Choices::default()
        };
        let buf = draw(160, 45, &mut choices);
        assert_eq!(
            read_row(&buf, 4).trim_end(),
            " qwen3.8-27b has no experts to show."
        );
        assert_eq!(
            read_row(&buf, 44).trim_end(),
            " 1–4 views   ←→ model   Tab States or Experts   m mouse: clicks   q stop the server"
        );
    }

    #[test]
    fn a_decision_model_shows_one_line_and_no_keys_of_its_own() {
        for sub_view in [SubView::States, SubView::Experts] {
            let mut choices = Choices {
                view: View::Models,
                model: 1,
                sub_view,
                ..Choices::default()
            };
            let buf = draw_models(160, 45, &mut choices, [Kind::Llm, Kind::Decision]);
            assert_eq!(
                read_row(&buf, 4).trim_end(),
                " qwen3.8-27b is a decision model: it keeps no sessions and has no experts.",
                "{sub_view:?}"
            );
            assert_eq!(
                read_row(&buf, 44).trim_end(),
                " 1–4 views   ←→ model   m mouse: clicks   q stop the server",
                "{sub_view:?}"
            );
        }
    }

    #[test]
    fn a_frame_keeps_only_its_own_click_targets() {
        let mut choices = Choices {
            view: View::Models,
            ..Choices::default()
        };
        draw(160, 45, &mut choices);
        let session = *choices
            .targets
            .iter()
            .find(|place| place.target == Target::Session(5))
            .expect("the session's place");
        let count = choices.targets.len();
        draw(160, 45, &mut choices);
        assert_eq!(choices.targets.len(), count, "the same places again");
        choices.view = View::Memory;
        draw(160, 45, &mut choices);
        assert_eq!(choices.find_target(session.x0, session.y), None);
    }
}
