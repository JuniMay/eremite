//! The terminal UI's view: every model's latest status, the samples of the
//! history window, the chosen session's text and what the person chose. It
//! takes the statuses and the keys, and draws the whole screen. Its thread
//! takes them as they come, and draws the screen after each. One thread takes
//! the statuses from the worker once a sample, and another one reads the
//! keys, so that neither waits for the other.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{self, Event};
use ratatui::layout::Rect;

use super::frame::{Choices, ModelInfo, Screen, SubView, View};
use super::input::{Asked, handle_event};
use super::llm::states::find_chosen_session;
use super::llm::text::{SessionText, split_into_messages};
use super::log::{LogLine, copy_lines};
use super::samples::{Samples, Windows};
use super::screen::draw_screen;
use super::status::ModelStatus;
use super::terminal::{Opened, ViewTerminal, draw_frame, leave, set_mouse_reports};
use super::theme::Theme;
use crate::{engine, worker};

/// A model as the view's thread needs it.
pub struct ViewModel {
    /// Its name and kind.
    pub info: ModelInfo,
    /// Its tokenizer, for the text of its sessions.
    pub tokenizer: engine::Tokenizer,
    /// Returns the role whose message a special token of its family starts.
    pub find_role: fn(&str) -> Option<&'static str>,
}

/// What the view's thread takes, in the order it comes.
enum Wake {
    /// Every model's status, taken this many seconds after the view started.
    Statuses(f64, Vec<ModelStatus>),
    /// A key, a click, a turn of the wheel or a resize.
    Input(Event),
    /// The end of the view.
    Stop,
}

/// What the screen shows, besides the person's choices.
struct Shown {
    /// The colors.
    theme: Theme,
    /// The server's address.
    address: String,
    /// The home folder, which the views show as ~.
    home: String,
    /// When the server started, for its uptime.
    started: Instant,
    /// The models, in their configured order.
    models: Vec<ModelInfo>,
    /// Each model's latest status.
    statuses: Vec<ModelStatus>,
    /// The samples of the history window.
    samples: Samples,
    /// The text of a session, with the index of its model, once it is read.
    session_text: Option<(usize, SessionText)>,
}

impl Shown {
    /// Returns what the views draw from while the model at `model_index` is
    /// chosen.
    fn build_screen(&self, model_index: usize) -> Screen<'_> {
        Screen {
            theme: &self.theme,
            address: &self.address,
            uptime_seconds: self.started.elapsed().as_secs_f64(),
            models: &self.models,
            statuses: &self.statuses,
            samples: &self.samples,
            home: &self.home,
            session_text: self
                .session_text
                .as_ref()
                .filter(|(model, _)| *model == model_index)
                .map(|(_, text)| text),
        }
    }
}

/// The view: what it shows, and what the person chose.
pub struct ViewState {
    /// What the screen shows.
    shown: Shown,
    /// What the person chose.
    pub choices: Choices,
}

impl ViewState {
    /// Returns the view of `models` before their first status, in `theme`.
    /// The header shows `address`, and the uptime since `started`. The
    /// figures average over `windows`. `mouse` says whether the terminal
    /// reports the mouse.
    pub fn new(
        theme: Theme,
        address: String,
        home: String,
        started: Instant,
        models: Vec<ModelInfo>,
        windows: Windows,
        mouse: bool,
    ) -> ViewState {
        ViewState {
            shown: Shown {
                theme,
                address,
                home,
                started,
                models,
                statuses: Vec::new(),
                samples: Samples::new(windows),
                session_text: None,
            },
            choices: Choices {
                mouse,
                ..Choices::default()
            },
        }
    }

    /// Takes `statuses`, every model's status taken `at` seconds after the
    /// view started, and adds their sample.
    pub fn take_statuses(&mut self, at: f64, statuses: Vec<ModelStatus>) {
        self.shown.samples.add_sample(at, &statuses);
        self.shown.statuses = statuses;
    }

    /// Hands `event` to the choices, and returns what the view's thread must
    /// do.
    pub fn handle_event(&mut self, event: &Event) -> Asked {
        let screen = self.shown.build_screen(self.choices.model);
        handle_event(event, &mut self.choices, &screen)
    }

    /// Returns the model's index and the number of the session whose text
    /// States needs: the chosen session, when its text is not read yet or
    /// the session grew since. Returns None while States is not shown.
    pub fn find_wanted_text(&self) -> Option<(usize, u64)> {
        let choices = &self.choices;
        if (choices.view, choices.sub_view) != (View::Models, SubView::States) {
            return None;
        }
        let status = self.shown.statuses.get(choices.model)?;
        let session = &status.sessions[find_chosen_session(status, choices)?];
        let wanted = (choices.model, session.number);
        let read = self
            .shown
            .session_text
            .as_ref()
            .is_some_and(|(model, text)| {
                (*model, text.session) == wanted && text.tokens as u64 == session.tokens
            });
        (!read).then_some(wanted)
    }

    /// Takes `tokens`, the tokens of session `session` of the model at
    /// `model_index`, as the session's text. `token_text` gives a token's
    /// bytes and whether it is a special token, and `find_role` the role
    /// whose message a special token starts.
    pub fn take_session_tokens<'a>(
        &mut self,
        model_index: usize,
        session: u64,
        tokens: &[i32],
        token_text: impl Fn(i32) -> (&'a [u8], bool),
        find_role: fn(&str) -> Option<&'static str>,
    ) {
        let text = SessionText {
            session,
            tokens: tokens.len(),
            messages: split_into_messages(tokens, token_text, find_role),
        };
        self.shown.session_text = Some((model_index, text));
    }

    /// Draws the whole screen in `area`, with `log_lines` in the log.
    pub fn draw(&mut self, buf: &mut Buffer, area: Rect, log_lines: &[LogLine]) {
        let screen = self.shown.build_screen(self.choices.model);
        draw_screen(buf, area, &screen, log_lines, &mut self.choices);
    }
}

/// The view's thread. finish() ends it.
pub struct ViewThread {
    /// Sends the view's thread what it takes.
    wake: mpsc::Sender<Wake>,
    /// The view's thread.
    thread: JoinHandle<()>,
}

impl ViewThread {
    /// Ends the view's thread, and gives the terminal back.
    pub fn finish(self) {
        let _ = self.wake.send(Wake::Stop);
        let _ = self.thread.join();
        leave();
    }
}

/// Starts the view of `models` on the terminal that enter() opened. The
/// header shows `address`, and the uptime since `started`. A thread takes
/// every model's status from the worker once a sample of `windows`, and
/// another one reads the keys when the terminal takes them.
pub fn start_view(
    opened: Opened,
    models: Vec<ViewModel>,
    address: String,
    started: Instant,
    windows: Windows,
) -> ViewThread {
    let home = std::env::var("HOME").unwrap_or_default();
    let infos = models.iter().map(|model| model.info.clone()).collect();
    let state = ViewState::new(
        opened.theme,
        address,
        home,
        started,
        infos,
        windows,
        opened.mouse,
    );
    let (wake, woken) = mpsc::channel();
    let view_started = Instant::now();
    let interval = Duration::from_secs_f64(windows.sample_seconds());
    let sampler_wake = wake.clone();
    let model_count = models.len();
    thread::Builder::new()
        .name("view-samples".into())
        .spawn(move || take_samples(model_count, view_started, interval, sampler_wake))
        .expect("the view's sample thread");
    if opened.keys {
        let input_wake = wake.clone();
        thread::Builder::new()
            .name("view-input".into())
            .spawn(move || read_input(input_wake))
            .expect("the view's input thread");
    }
    let thread = thread::Builder::new()
        .name("view".into())
        .spawn(move || run_view(opened.terminal, state, &models, woken))
        .expect("the view's thread");
    ViewThread { wake, thread }
}

/// Takes every model's status from the worker once each `interval` from
/// `view_started`, and sends it on `wake`. It ends once the view's thread
/// has ended.
fn take_samples(
    model_count: usize,
    view_started: Instant,
    interval: Duration,
    wake: mpsc::Sender<Wake>,
) {
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread().build() else {
        return;
    };
    let mut next = Instant::now();
    loop {
        runtime.block_on(worker::refresh_status());
        let statuses = (0..model_count)
            .map(|model_index| {
                serde_json::from_value(worker::build_model_status_json(model_index))
                    .unwrap_or_default()
            })
            .collect();
        let at = view_started.elapsed().as_secs_f64();
        if wake.send(Wake::Statuses(at, statuses)).is_err() {
            return;
        }
        // A sample that came late moves the next ones, so that none come
        // in a burst.
        next = (next + interval).max(Instant::now());
        thread::sleep(next.saturating_duration_since(Instant::now()));
    }
}

/// Reads the keys, the clicks, the wheel and the resizes, and sends them on
/// `wake`. It ends when the terminal gives no more, or once the view's
/// thread has ended.
fn read_input(wake: mpsc::Sender<Wake>) {
    while let Ok(event) = event::read() {
        if wake.send(Wake::Input(event)).is_err() {
            return;
        }
    }
}

/// Runs the view on `view_terminal` from `state`: it takes what comes on
/// `woken` and draws the screen after each change. It reads the chosen
/// session's text from the worker with the tokenizers of `models`. It ends
/// at Wake::Stop, or once the terminal is given back.
fn run_view(
    mut view_terminal: ViewTerminal,
    mut state: ViewState,
    models: &[ViewModel],
    woken: mpsc::Receiver<Wake>,
) {
    let mut redraw = true;
    loop {
        if redraw
            && !draw_frame(&mut view_terminal, |frame| {
                let area = frame.area();
                state.draw(frame.buffer_mut(), area, &copy_lines())
            })
        {
            return;
        }
        redraw = match woken.recv() {
            Ok(Wake::Statuses(at, statuses)) => {
                state.take_statuses(at, statuses);
                if let Some((model_index, session)) = state.find_wanted_text() {
                    if let Some(tokens) = worker::read_session_tokens(model_index, session) {
                        let model = &models[model_index];
                        let tokenizer = model.tokenizer;
                        let token_text = |id| (tokenizer.token_bytes(id), tokenizer.is_added(id));
                        state.take_session_tokens(
                            model_index,
                            session,
                            &tokens,
                            token_text,
                            model.find_role,
                        );
                    }
                    // The tokens come with the worker's next status. Those
                    // read may be older than the session.
                    if state.find_wanted_text().is_some() {
                        worker::ask_session_tokens(model_index, session);
                    }
                }
                true
            }
            Ok(Wake::Input(event)) => match state.handle_event(&event) {
                Asked::Redraw => true,
                Asked::Mouse(on) => {
                    state.choices.mouse = set_mouse_reports(on);
                    true
                }
                Asked::Quit => {
                    engine::log(if engine::is_stopping() {
                        "q again: the server stops without saving more states"
                    } else {
                        "q: the server stops"
                    });
                    worker::request_stop();
                    false
                }
                Asked::Nothing => false,
            },
            Ok(Wake::Stop) | Err(_) => return,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::Kind;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, read_row, read_screen};
    use crate::ui::theme::DARK;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    /// The tokens of a session's text, by id: each one's text, and whether
    /// it is special.
    const TOKENS: [(&str, bool); 4] = [
        ("<user>", true),
        ("Stake the tomatoes.", false),
        ("<assistant>", true),
        ("Done.", false),
    ];

    /// The role whose message a test token starts.
    fn find_role(token: &str) -> Option<&'static str> {
        match token {
            "<user>" => Some("user"),
            "<assistant>" => Some("assistant"),
            _ => None,
        }
    }

    /// deepseek-v4.1-flash's status while it decodes request 5 in session
    /// 5 of `tokens` tokens, and the status of qwen3.8-27b, which has its
    /// own session 5.
    fn statuses(tokens: usize) -> Vec<ModelStatus> {
        let deepseek = format!(
            r#"{{"loaded": true, "phase": "decode",
            "requests": [{{"request": 5, "status": "decoding", "session": 5}}],
            "sessions": [{{"session": 5, "tokens": {tokens}, "busy": true}}]}}"#
        );
        let qwen = r#"{"loaded": true, "sessions": [{"session": 5, "tokens": 9}]}"#;
        [deepseek.as_str(), qwen]
            .map(|json| serde_json::from_str(json).expect("the worker's form"))
            .to_vec()
    }

    /// The view of the two models, before their first status.
    fn new_view() -> ViewState {
        let models = ["deepseek-v4.1-flash", "qwen3.8-27b"].map(|name| ModelInfo {
            name: name.into(),
            kind: Kind::Llm,
        });
        ViewState::new(
            DARK,
            "127.0.0.1:52878".into(),
            String::new(),
            Instant::now(),
            models.to_vec(),
            WINDOWS,
            true,
        )
    }

    /// Hands the key `code` to `view`.
    fn press(view: &mut ViewState, code: KeyCode) -> Asked {
        view.handle_event(&Event::Key(KeyEvent::new(code, KeyModifiers::NONE)))
    }

    /// Draws `view` at 160 x 45.
    fn draw(view: &mut ViewState) -> Buffer {
        draw_on_terminal(160, 45, |buf, area| view.draw(buf, area, &[]))
    }

    #[test]
    fn the_view_draws_its_statuses_and_takes_the_keys() {
        let mut view = new_view();
        assert!(read_row(&draw(&mut view), 0).starts_with(" eremite  127.0.0.1:52878"));
        view.take_statuses(0.0, statuses(4));
        view.take_statuses(0.5, statuses(4));
        assert_eq!(view.shown.samples.samples.len(), 2);
        let screen = read_screen(&draw(&mut view));
        assert!(screen.contains(" #5 "), "{screen}");
        assert!(
            read_row(&draw(&mut view), 44)
                .trim_end()
                .ends_with("q stop the server")
        );
        assert_eq!(press(&mut view, KeyCode::Char('2')), Asked::Redraw);
        assert_eq!(view.choices.view, View::Memory);
        assert_eq!(press(&mut view, KeyCode::Char('m')), Asked::Mouse(false));
        assert_eq!(press(&mut view, KeyCode::Char('q')), Asked::Quit);
    }

    #[test]
    fn the_view_wants_the_chosen_sessions_text_in_states_only() {
        let mut view = new_view();
        view.take_statuses(0.0, statuses(4));
        assert_eq!(view.find_wanted_text(), None, "1 Overview");
        press(&mut view, KeyCode::Char('4'));
        assert_eq!(view.find_wanted_text(), Some((0, 5)));
        let token_text = |id: i32| {
            let (text, special) = TOKENS[id as usize];
            (text.as_bytes(), special)
        };
        view.take_session_tokens(0, 5, &[0, 1, 2, 3], token_text, find_role);
        assert_eq!(view.find_wanted_text(), None, "the text is read");
        let screen = read_screen(&draw(&mut view));
        assert!(screen.contains("Stake the tomatoes."), "{screen}");
        assert!(screen.contains("assistant"));
        // the session grew
        view.take_statuses(0.5, statuses(6));
        assert_eq!(view.find_wanted_text(), Some((0, 5)));
        // another model's session 5 shows none of that text
        press(&mut view, KeyCode::Right);
        assert_eq!(view.find_wanted_text(), Some((1, 5)));
        let screen = read_screen(&draw(&mut view));
        assert!(!screen.contains("Stake the tomatoes."), "{screen}");
        press(&mut view, KeyCode::Tab);
        assert_eq!(view.find_wanted_text(), None, "Experts");
    }
}
