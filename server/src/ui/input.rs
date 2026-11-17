//! The terminal UI's keys and mouse: what each one chooses.

use ratatui::crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};

use super::frame::{Choices, Screen, Target, VIEWS, View};

/// What an input asks of the view's loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asked {
    /// Draw the screen again: the choices changed.
    Redraw,
    /// Turn the mouse's reports on, or off.
    Mouse(bool),
    /// Stop the server.
    Quit,
    /// Nothing changed.
    Nothing,
}

/// Returns whether `key` stops the server: q, or Ctrl-C. Raw mode delivers
/// Ctrl-C as a key.
pub fn is_quit_key(key: &KeyEvent) -> bool {
    key.kind == KeyEventKind::Press
        && (matches!(key.code, KeyCode::Char('q' | 'Q'))
            || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)))
}

/// Handles `event`: a key, a click or a resize. It changes `choices` and
/// returns what the view's loop must do.
pub fn handle_event(event: &Event, choices: &mut Choices, screen: &Screen) -> Asked {
    match event {
        Event::Key(key) if is_quit_key(key) => Asked::Quit,
        Event::Key(key) if key.kind == KeyEventKind::Press => handle_key(key.code, choices, screen),
        Event::Mouse(mouse) if choices.mouse => match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                match choices.find_target(mouse.column, mouse.row) {
                    Some(target) => choose_target(target, choices, screen),
                    None => Asked::Nothing,
                }
            }
            _ => Asked::Nothing,
        },
        Event::Resize(..) => Asked::Redraw,
        _ => Asked::Nothing,
    }
}

/// Chooses what a click on `target` chooses.
fn choose_target(target: Target, choices: &mut Choices, screen: &Screen) -> Asked {
    match target {
        Target::View(view) => choices.view = view,
        Target::Model(index) => choose_model(choices, screen, index),
        Target::SubView(sub_view) => choices.sub_view = sub_view,
        Target::Session(number) => {
            choices.session = Some(number);
            choices.text_scroll = 0;
        }
        Target::Request(id) => choices.request = Some(id),
    }
    Asked::Redraw
}

/// Chooses the model at `index` in Models. It keeps the sub-view when the
/// model's kind has it, and else takes the first sub-view of its kind.
fn choose_model(choices: &mut Choices, screen: &Screen, index: usize) {
    let Some(model) = screen.models.get(index) else {
        return;
    };
    let sub_views = model.kind.list_sub_views();
    if !sub_views.contains(&choices.sub_view)
        && let Some(&first) = sub_views.first()
    {
        choices.sub_view = first;
    }
    if index != choices.model {
        choices.model = index;
        choices.session = None;
        choices.request = None;
        (choices.text_scroll, choices.layer_scroll) = (0, 0);
    }
}

/// Handles the key `code` in the view that `choices` shows.
fn handle_key(code: KeyCode, choices: &mut Choices, screen: &Screen) -> Asked {
    let view = choices.view;
    let sub_view = choices.sub_view;
    let models = screen.models.len();
    match code {
        KeyCode::Char(digit @ '1'..='4') => {
            choices.view = VIEWS[digit as usize - '1' as usize].0;
        }
        KeyCode::Char('m' | 'M') => {
            choices.mouse = !choices.mouse;
            return Asked::Mouse(choices.mouse);
        }
        KeyCode::Left if view == View::Models && models > 0 => {
            choose_model(choices, screen, (choices.model + models - 1) % models);
        }
        KeyCode::Right if view == View::Models && models > 0 => {
            choose_model(choices, screen, (choices.model + 1) % models);
        }
        KeyCode::Tab if view == View::Models => {
            let sub_views = screen
                .models
                .get(choices.model)
                .map_or(&[][..], |model| model.kind.list_sub_views());
            let at = sub_views.iter().position(|&other| other == sub_view);
            if let Some(&next) = at.and_then(|at| sub_views.get((at + 1) % sub_views.len())) {
                choices.sub_view = next;
            }
        }
        _ => return Asked::Nothing,
    }
    Asked::Redraw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{ClickTarget, Kind, ModelInfo, SubView};
    use crate::ui::samples::Samples;
    use crate::ui::status::ModelStatus;
    use crate::ui::testing::WINDOWS;
    use crate::ui::theme::DARK;
    use ratatui::crossterm::event::MouseEvent;

    /// Two models' statuses as the worker builds them: deepseek decodes
    /// requests 5 and 6 in sessions 5 and 2, and keeps three sessions.
    fn statuses() -> Vec<ModelStatus> {
        let deepseek = r#"{"loaded": true,
            "requests": [{"request": 5, "status": "decoding", "session": 5},
                         {"request": 6, "status": "decoding", "session": 2}],
            "sessions": [
                {"session": 5, "tokens": 604, "idle_seconds": 0, "shared_with_next": 2},
                {"session": 2, "tokens": 550, "idle_seconds": 23, "shared_with_next": 7},
                {"session": 3, "tokens": 469, "idle_seconds": 30, "shared_with_next": 0}]}"#;
        vec![
            serde_json::from_str(deepseek).unwrap(),
            ModelStatus::default(),
        ]
    }

    /// Hands `key` to `handle_event` with `choices`.
    fn press(code: KeyCode, choices: &mut Choices) -> Asked {
        let models = ["deepseek-v4.1-flash", "qwen3.8-27b"].map(|name| ModelInfo {
            name: name.into(),
            kind: Kind::Llm,
        });
        let statuses = statuses();
        let samples = Samples::new(WINDOWS);
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
        let event = Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
        handle_event(&event, choices, &screen)
    }

    #[test]
    fn the_keys_choose_views_models_and_sub_views() {
        let mut choices = Choices::default();
        assert_eq!(press(KeyCode::Char('2'), &mut choices), Asked::Redraw);
        assert_eq!(choices.view, View::Memory);
        press(KeyCode::Char('4'), &mut choices);
        assert_eq!(choices.view, View::Models);
        press(KeyCode::Right, &mut choices);
        assert_eq!(choices.model, 1);
        press(KeyCode::Right, &mut choices);
        assert_eq!(choices.model, 0, "the models go round");
        press(KeyCode::Left, &mut choices);
        assert_eq!(choices.model, 1);
        press(KeyCode::Tab, &mut choices);
        assert_eq!(choices.sub_view, SubView::Experts);
        press(KeyCode::Tab, &mut choices);
        assert_eq!(choices.sub_view, SubView::States);
        assert_eq!(press(KeyCode::Char('m'), &mut choices), Asked::Mouse(false));
        assert_eq!(press(KeyCode::Char('x'), &mut choices), Asked::Nothing);
        assert_eq!(press(KeyCode::Char('q'), &mut choices), Asked::Quit);
        let control_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(is_quit_key(&control_c));
        assert!(!is_quit_key(&KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::NONE
        )));
    }

    #[test]
    fn a_click_chooses_its_target() {
        let models = [ModelInfo {
            name: "deepseek-v4.1-flash".into(),
            kind: Kind::Llm,
        }];
        let statuses = statuses();
        let samples = Samples::new(WINDOWS);
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
        let mut choices = Choices::default();
        choices.targets.push(ClickTarget {
            y: 1,
            x0: 15,
            x1: 25,
            target: Target::View(View::Memory),
        });
        let mouse = |kind, column, row| {
            Event::Mouse(MouseEvent {
                kind,
                column,
                row,
                modifiers: KeyModifiers::NONE,
            })
        };
        let click = mouse(MouseEventKind::Down(MouseButton::Left), 18, 1);
        assert_eq!(handle_event(&click, &mut choices, &screen), Asked::Redraw);
        assert_eq!(choices.view, View::Memory);
        let elsewhere = mouse(MouseEventKind::Down(MouseButton::Left), 30, 1);
        assert_eq!(
            handle_event(&elsewhere, &mut choices, &screen),
            Asked::Nothing
        );
        // without the mouse, a click does nothing
        choices.mouse = false;
        assert_eq!(handle_event(&click, &mut choices, &screen), Asked::Nothing);
        assert_eq!(
            handle_event(&Event::Resize(100, 30), &mut choices, &screen),
            Asked::Redraw
        );
    }
}
