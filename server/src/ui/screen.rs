//! The terminal UI's whole screen: the chrome, the view shown in it, and the
//! keys row.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use unicode_width::UnicodeWidthStr;

use super::frame::{
    Choices, Keys, Kind, Screen, SubView, View, draw_chrome, list_chrome_keys, list_model_keys,
};
use super::history::draw_history;
use super::input::list_view_keys;
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
            }
        }
    }
}

/// Returns the keys of the keys row for `choices` in `columns`: the chrome's
/// first keys, the model's keys in 4 Models, the view's own keys, and the
/// chrome's last keys. A sub-view that shows only its one line, such as the
/// Experts of a model without experts, has no keys of its own. When the row
/// does not fit, the model's and the view's keys go, from the last one, so
/// that the chrome's keys stay.
fn list_keys(screen: &Screen, choices: &Choices, columns: usize) -> Keys {
    let (first, last) = list_chrome_keys(choices);
    let mut own = Vec::new();
    if choices.view == View::Models {
        own.extend(list_model_keys(screen, choices));
    }
    let shows_one_line = choices.view == View::Models
        && choices.sub_view == SubView::Experts
        && screen
            .statuses
            .get(choices.model)
            .is_some_and(has_no_experts);
    if !shows_one_line {
        own.extend(list_view_keys(choices));
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
             text   m mouse: clicks   q stop the server",
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
        let models = ["deepseek-v4.1-flash", "qwen3.8-27b"].map(|name| ModelInfo {
            name: name.into(),
            kind: Kind::Llm,
        });
        let statuses = statuses();
        let mut samples = Samples::new(WINDOWS);
        samples.add_sample(0.0, &statuses);
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 57.0,
            models: &models,
            statuses: &statuses,
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
