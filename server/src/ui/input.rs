//! The terminal UI's keys and mouse: what each one chooses or scrolls.

use ratatui::crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};

use super::frame::{Choices, Keys, Screen, SubView, Target, VIEWS, View};
use super::llm::experts::find_decoding_requests;
use super::llm::states::{build_session_tree, find_chosen_session};

/// The lines that one step of the mouse wheel scrolls.
const WHEEL_LINES: isize = 3;

/// What an input asks of the view's loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asked {
    /// Draw the screen again: the choices changed.
    Redraw,
    /// Turn the mouse's reports on, or off.
    Mouse(bool),
    /// Stop the server.
    Quit,
    /// Delete the state of session `session` of the model at `model_index`
    /// from memory, without saving it to the SSD.
    DeleteSessionState {
        /// The model's index.
        model_index: usize,
        /// The session's number.
        session: u64,
    },
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

/// Handles `event`: a key, a click, a turn of the wheel or a resize. It
/// changes `choices` and returns what the view's loop must do.
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
            MouseEventKind::ScrollUp => scroll(choices, -WHEEL_LINES),
            MouseEventKind::ScrollDown => scroll(choices, WHEEL_LINES),
            _ => Asked::Nothing,
        },
        Event::Resize(..) => Asked::Redraw,
        _ => Asked::Nothing,
    }
}

/// Chooses what a click on `target` chooses. A click also asks no more to
/// delete a state.
fn choose_target(target: Target, choices: &mut Choices, screen: &Screen) -> Asked {
    choices.delete_asked = None;
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

/// Handles the key `code` in the view that `choices` shows. Any key but a
/// second d asks no more to delete a state.
fn handle_key(code: KeyCode, choices: &mut Choices, screen: &Screen) -> Asked {
    let delete_asked = choices.delete_asked.take();
    let view = choices.view;
    let sub_view = choices.sub_view;
    let models = screen.models.len();
    let in_models = |wanted: SubView| view == View::Models && sub_view == wanted;
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
        KeyCode::Char('f' | 'F') if view == View::Overview => {
            choices.log_filter = choices.log_filter.next(models);
            choices.log_back = 0;
        }
        KeyCode::Up if view == View::Overview => return scroll(choices, -1),
        KeyCode::Down if view == View::Overview => return scroll(choices, 1),
        KeyCode::Home if view == View::Overview => choices.log_back = usize::MAX,
        KeyCode::End if view == View::Overview => choices.log_back = 0,
        KeyCode::Up if in_models(SubView::States) => choose_next_session(choices, screen, -1),
        KeyCode::Down if in_models(SubView::States) => choose_next_session(choices, screen, 1),
        KeyCode::Char('d' | 'D') if in_models(SubView::States) => {
            return handle_delete_key(choices, screen, delete_asked);
        }
        KeyCode::Up if in_models(SubView::Experts) => return scroll(choices, -1),
        KeyCode::Down if in_models(SubView::Experts) => return scroll(choices, 1),
        KeyCode::Char(bracket @ ('[' | ']')) if in_models(SubView::Experts) => {
            let (requests, chosen) = find_decoding_requests(screen, choices);
            let at = chosen.and_then(|chosen| requests.iter().position(|r| r.id == chosen.id));
            if let Some(at) = at {
                let next = if bracket == ']' {
                    (at + 1) % requests.len()
                } else {
                    (at + requests.len() - 1) % requests.len()
                };
                choices.request = Some(requests[next].id);
            }
        }
        KeyCode::PageUp => {
            let page = page_of(choices) as isize;
            return scroll(choices, -page);
        }
        KeyCode::PageDown => {
            let page = page_of(choices) as isize;
            return scroll(choices, page);
        }
        // The keys row asks no more.
        _ if delete_asked.is_some() => {}
        _ => return Asked::Nothing,
    }
    Asked::Redraw
}

/// Handles the d key in States. The first press asks to delete the state of
/// the chosen session, and the keys row shows the question. A second press,
/// while `asked` is that session, asks the worker to delete it. A session
/// that a request runs in is neither asked about nor deleted.
fn handle_delete_key(choices: &mut Choices, screen: &Screen, asked: Option<u64>) -> Asked {
    let Some(status) = screen.statuses.get(choices.model) else {
        return Asked::Nothing;
    };
    let Some(index) = find_chosen_session(status, choices) else {
        return Asked::Nothing;
    };
    let session = &status.sessions[index];
    if session.busy {
        return Asked::Redraw;
    }
    if asked == Some(session.number) {
        choices.session = None;
        choices.text_scroll = 0;
        return Asked::DeleteSessionState {
            model_index: choices.model,
            session: session.number,
        };
    }
    // The question names this session, so a status that changes the
    // session used last does not change it.
    choices.session = Some(session.number);
    choices.delete_asked = Some(session.number);
    Asked::Redraw
}

/// Returns the rows of a page of what the view shown scrolls: the log, the
/// text panel or the layers.
fn page_of(choices: &Choices) -> usize {
    match (choices.view, choices.sub_view) {
        (View::Models, SubView::States) => choices.text_page,
        (View::Models, SubView::Experts) => choices.layer_page,
        _ => choices.log_page,
    }
    .max(1)
}

/// Scrolls what the view shown scrolls by `lines`, forward for more than 0.
/// The log scrolls back toward older lines, the text panel and the layers
/// down. Each view keeps its scroll within its lines when it draws.
fn scroll(choices: &mut Choices, lines: isize) -> Asked {
    match (choices.view, choices.sub_view) {
        (View::Overview, _) => choices.log_back = choices.log_back.saturating_add_signed(-lines),
        (View::Models, SubView::States) => {
            choices.text_scroll = choices.text_scroll.saturating_add_signed(lines)
        }
        (View::Models, SubView::Experts) => {
            choices.layer_scroll = choices.layer_scroll.saturating_add_signed(lines)
        }
        _ => return Asked::Nothing,
    }
    Asked::Redraw
}

/// Chooses the session `step` rows above or below the chosen one in the
/// tree of the chosen model's sessions.
fn choose_next_session(choices: &mut Choices, screen: &Screen, step: isize) {
    let Some(status) = screen.statuses.get(choices.model) else {
        return;
    };
    let (nodes, _) = build_session_tree(&status.sessions);
    let mut order: Vec<(usize, usize)> = nodes
        .iter()
        .flat_map(|node| node.ending.iter().map(move |&index| (node.row, index)))
        .collect();
    order.sort_unstable();
    let chosen = find_chosen_session(status, choices);
    let Some(at) = order.iter().position(|&(_, index)| Some(index) == chosen) else {
        return;
    };
    let next = at.saturating_add_signed(step).min(order.len() - 1);
    choices.session = Some(status.sessions[order[next].1].number);
    choices.text_scroll = 0;
}

/// Returns the keys of the view that `choices` shows in `screen`, for the
/// keys row. In States, d says what it does for the chosen session.
pub fn list_view_keys(screen: &Screen, choices: &Choices) -> Keys {
    let keys: &[(&'static str, &str)] = match (choices.view, choices.sub_view) {
        (View::Overview, _) => &[
            ("↑↓ PgUp PgDn", "scroll"),
            ("Home End", "oldest, newest"),
            ("f", "filter"),
        ],
        (View::Models, SubView::States) => return list_states_keys(screen, choices),
        (View::Models, SubView::Experts) => &[
            ("[ ] or click", "request"),
            ("↑↓ or wheel", "scroll layers"),
        ],
        (View::Memory | View::History, _) => &[],
    };
    keys.iter()
        .map(|&(key, what)| (key, what.to_string()))
        .collect()
}

/// Returns the keys of States in `screen` for `choices`. d deletes the
/// chosen session's state, unless a request runs in the session.
fn list_states_keys(screen: &Screen, choices: &Choices) -> Keys {
    let mut keys = vec![
        ("↑↓ or click", "session".to_string()),
        ("PgUp PgDn", "text".to_string()),
    ];
    let Some(status) = screen.statuses.get(choices.model) else {
        return keys;
    };
    let Some(index) = find_chosen_session(status, choices) else {
        return keys;
    };
    let session = &status.sessions[index];
    if session.busy {
        keys.push(("d", format!("session {} is in use", session.number)));
    } else {
        keys.push(("d", "delete state".to_string()));
    }
    keys
}

/// Returns the question of the keys row while d asks to delete the state of
/// the session chosen in States, such as "d again deletes session 3". Returns
/// None when d asks nothing, and for a session that a request runs in.
pub fn find_delete_question(screen: &Screen, choices: &Choices) -> Option<(&'static str, String)> {
    if (choices.view, choices.sub_view) != (View::Models, SubView::States) {
        return None;
    }
    let status = screen.statuses.get(choices.model)?;
    let session = &status.sessions[find_chosen_session(status, choices)?];
    (!session.busy && choices.delete_asked == Some(session.number))
        .then(|| ("d", format!("again deletes session {}", session.number)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{ClickTarget, Kind, ModelInfo, SubView};
    use crate::ui::log::LogFilter;
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
                {"session": 5, "tokens": 604, "idle_seconds": 0, "busy": true,
                 "shared_with_next": 2},
                {"session": 2, "tokens": 550, "idle_seconds": 23, "busy": true,
                 "shared_with_next": 7},
                {"session": 3, "tokens": 469, "idle_seconds": 30, "shared_with_next": 0}]}"#;
        vec![
            serde_json::from_str(deepseek).unwrap(),
            ModelStatus::default(),
        ]
    }

    /// Returns what `use_screen` returns from the screen of the two models of
    /// `statuses`.
    fn with_screen<T>(use_screen: impl FnOnce(&Screen) -> T) -> T {
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
        use_screen(&screen)
    }

    /// Hands `key` to `handle_event` with `choices`.
    fn press(code: KeyCode, choices: &mut Choices) -> Asked {
        let event = Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
        with_screen(|screen| handle_event(&event, choices, screen))
    }

    /// Returns the keys of the view that `choices` shows.
    fn view_keys(choices: &Choices) -> Keys {
        with_screen(|screen| list_view_keys(screen, choices))
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
    fn the_keys_scroll_and_filter_the_log() {
        let mut choices = Choices {
            log_page: 10,
            ..Choices::default()
        };
        press(KeyCode::Up, &mut choices);
        assert_eq!(choices.log_back, 1, "one line older");
        press(KeyCode::PageUp, &mut choices);
        assert_eq!(choices.log_back, 11);
        press(KeyCode::PageDown, &mut choices);
        press(KeyCode::Down, &mut choices);
        assert_eq!(choices.log_back, 0);
        press(KeyCode::Home, &mut choices);
        assert_eq!(choices.log_back, usize::MAX);
        press(KeyCode::End, &mut choices);
        assert_eq!(choices.log_back, 0);
        press(KeyCode::Char('f'), &mut choices);
        assert_eq!(choices.log_filter, LogFilter::Model(0));
        assert_eq!(view_keys(&choices)[2], ("f", "filter".to_string()));
    }

    #[test]
    fn the_keys_choose_sessions_requests_and_scroll_in_models() {
        let mut choices = Choices {
            view: View::Models,
            text_page: 20,
            ..Choices::default()
        };
        // the session used last is chosen first; the tree shows 5, 2, 3
        press(KeyCode::Down, &mut choices);
        assert_eq!(choices.session, Some(2));
        press(KeyCode::Down, &mut choices);
        press(KeyCode::Down, &mut choices);
        assert_eq!(choices.session, Some(3), "no further than the last");
        press(KeyCode::Up, &mut choices);
        assert_eq!(choices.session, Some(2));
        press(KeyCode::PageDown, &mut choices);
        assert_eq!(choices.text_scroll, 20);
        choices.sub_view = SubView::Experts;
        press(KeyCode::Char(']'), &mut choices);
        assert_eq!(choices.request, Some(6));
        press(KeyCode::Char(']'), &mut choices);
        assert_eq!(choices.request, Some(5), "the requests go round");
        press(KeyCode::Down, &mut choices);
        assert_eq!(choices.layer_scroll, 1);
        assert_eq!(view_keys(&choices)[0].0, "[ ] or click");
        // a model of the same kind keeps the sub-view
        press(KeyCode::Right, &mut choices);
        assert_eq!((choices.model, choices.sub_view), (1, SubView::Experts));
        assert_eq!((choices.request, choices.layer_scroll), (None, 0));
    }

    #[test]
    fn the_d_key_asks_then_deletes_an_idle_sessions_state() {
        let mut choices = Choices {
            view: View::Models,
            ..Choices::default()
        };
        let delete_key = |choices: &Choices| view_keys(choices).last().cloned();
        // the session used last, 5, runs request 5
        assert_eq!(
            delete_key(&choices),
            Some(("d", "session 5 is in use".to_string()))
        );
        press(KeyCode::Char('d'), &mut choices);
        assert_eq!(choices.delete_asked, None, "d does nothing for it");
        press(KeyCode::Down, &mut choices);
        press(KeyCode::Down, &mut choices);
        assert_eq!(choices.session, Some(3));
        assert_eq!(
            delete_key(&choices),
            Some(("d", "delete state".to_string()))
        );
        assert_eq!(press(KeyCode::Char('d'), &mut choices), Asked::Redraw);
        assert_eq!(choices.delete_asked, Some(3));
        assert_eq!(
            with_screen(|screen| find_delete_question(screen, &choices)),
            Some(("d", "again deletes session 3".to_string())),
            "the keys row asks"
        );
        // another key asks no more
        assert_eq!(press(KeyCode::Char('x'), &mut choices), Asked::Redraw);
        assert_eq!(choices.delete_asked, None);
        press(KeyCode::Char('d'), &mut choices);
        assert_eq!(
            press(KeyCode::Char('d'), &mut choices),
            Asked::DeleteSessionState {
                model_index: 0,
                session: 3
            }
        );
        assert_eq!((choices.delete_asked, choices.session), (None, None));
        // a click asks no more
        press(KeyCode::Down, &mut choices);
        press(KeyCode::Down, &mut choices);
        press(KeyCode::Char('d'), &mut choices);
        let target = Target::View(View::Models);
        with_screen(|screen| choose_target(target, &mut choices, screen));
        assert_eq!(choices.delete_asked, None);
        // d does nothing outside States
        choices.sub_view = SubView::Experts;
        assert_eq!(press(KeyCode::Char('d'), &mut choices), Asked::Nothing);
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

    #[test]
    fn the_wheel_scrolls_what_the_view_scrolls() {
        let mut choices = Choices::default();
        let wheel = |kind| {
            Event::Mouse(MouseEvent {
                kind,
                column: 30,
                row: 20,
                modifiers: KeyModifiers::NONE,
            })
        };
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
        handle_event(&wheel(MouseEventKind::ScrollUp), &mut choices, &screen);
        assert_eq!(choices.log_back, 3, "the log goes back three lines");
        choices.view = View::Models;
        handle_event(&wheel(MouseEventKind::ScrollDown), &mut choices, &screen);
        assert_eq!(choices.text_scroll, 3);
        choices.view = View::Memory;
        let asked = handle_event(&wheel(MouseEventKind::ScrollDown), &mut choices, &screen);
        assert_eq!(asked, Asked::Nothing, "2 Memory does not scroll");
    }
}
