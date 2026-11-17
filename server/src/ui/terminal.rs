//! The terminal that the terminal UI draws on. enter() sets raw mode for the
//! keys, chooses the colors, shows the alternate screen, asks for the mouse's
//! reports and captures stderr's lines into the log. leave() gives the
//! terminal back as the view found it, on every way out: the normal end, a
//! fatal error of the engine, a panic and the process's exit.

use std::fs::File;
use std::io::{self, IsTerminal, Write};
use std::mem::ManuallyDrop;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, RawFd};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::crossterm::terminal::{
    self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::crossterm::{cursor, execute};
use ratatui::{Frame, Terminal};

use super::capture::{Capture, capture_lines, find_thread_id};
use super::log::{keep_captured_line, set_on_screen, take_unprinted_lines};
use super::theme::{DARK, LIGHT, Rgb, Theme, is_true_color, parse_background_answer};

/// The terminal while the view is on it. leave() takes it. The hooks of a
/// fatal error, of a panic and of the exit reach it here, so it is global.
static ENTERED: Mutex<Option<Entered>> = Mutex::new(None);
/// Held while a frame is written, so that leave() never gives the terminal
/// back in the middle of one.
static DRAWING: Mutex<()> = Mutex::new(());
/// The pthread id of the thread that writes a frame now, if one does. leave()
/// on that thread, from a panic in the frame, does not wait for the frame.
static DRAWING_THREAD: Mutex<Option<libc::pthread_t>> = Mutex::new(None);

/// The question for the terminal's background color: OSC 11.
const BACKGROUND_QUESTION: &[u8] = b"\x1b]11;?\x1b\\";
/// The question for the terminal's attributes: DA1. Every terminal answers
/// it, after the questions before it.
const ATTRIBUTES_QUESTION: &[u8] = b"\x1b[c";
/// The start of the answer to the attributes question.
const ATTRIBUTES_ANSWER_START: &[u8] = b"\x1b[?";
/// The last byte of the answer to the attributes question.
const ATTRIBUTES_ANSWER_END: u8 = b'c';
/// The bytes that one read of the answers takes at most: more than both
/// answers together.
const ANSWER_READ_BYTES: usize = 256;
/// The mouse's reports that the view asks for: its clicks and its wheel
/// (1000), in the SGR form (1006). The view uses no moves.
const MOUSE_REPORTS_ON: &[u8] = b"\x1b[?1000h\x1b[?1006h";
/// The end of the mouse's reports.
const MOUSE_REPORTS_OFF: &[u8] = b"\x1b[?1006l\x1b[?1000l";

/// How the terminal UI chooses its colors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThemeChoice {
    /// By the terminal's background, which the terminal tells when asked.
    /// Without its answer, the light colors.
    #[default]
    Auto,
    /// The light colors.
    Light,
    /// The dark colors.
    Dark,
}

/// The terminal that the view draws on.
pub type ViewTerminal = Terminal<CrosstermBackend<File>>;

/// What enter() gives the view.
pub struct Opened {
    /// The terminal to draw on.
    pub terminal: ViewTerminal,
    /// The colors for the terminal's background.
    pub theme: Theme,
    /// Whether the view reads keys: the terminal is in raw mode.
    pub keys: bool,
    /// Whether the terminal reports the mouse.
    pub mouse: bool,
}

/// What the view changed in the terminal.
struct Entered {
    /// The terminal, for what gives it back.
    terminal: File,
    /// Whether the terminal is in raw mode.
    raw: bool,
    /// Whether the terminal reports the mouse.
    mouse: bool,
    /// The capture of stderr's lines.
    capture: Option<Capture>,
}

/// Locks `mutex`. It also takes the data after a thread panicked with the lock
/// held.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Puts the view on stderr's terminal: raw mode for the keys, the colors by
/// `choice`, the alternate screen cleared and without the cursor, the
/// mouse's reports, and stderr's lines in the log. With ThemeChoice::Auto,
/// it waits at most `answer_wait` for the terminal's background color. From
/// then on, a panic and the process's exit give the terminal back with
/// leave(). Returns None when stderr is not a terminal, or when the
/// terminal cannot show the view. The log then stays on stderr.
pub fn enter(choice: ThemeChoice, answer_wait: Duration) -> Option<Opened> {
    if !io::stderr().is_terminal() {
        return None;
    }
    let mut entered = Entered {
        terminal: File::from(io::stderr().as_fd().try_clone_to_owned().ok()?),
        raw: terminal::enable_raw_mode().is_ok(),
        mouse: false,
        capture: None,
    };
    // The terminal answers on stdin, in raw mode.
    let stdin = io::stdin();
    let background = (choice == ThemeChoice::Auto && entered.raw && stdin.is_terminal())
        .then(|| ask_background(stdin.as_raw_fd(), &mut entered.terminal, answer_wait))
        .flatten();
    let colorterm = std::env::var("COLORTERM").ok();
    let theme = choose_theme(choice, background, colorterm.as_deref());
    if execute!(entered.terminal, EnterAlternateScreen, cursor::Hide).is_err() {
        give_back(entered);
        return None;
    }
    // From here on, the log keeps its lines for the view instead of writing
    // them to stderr, which the capture takes.
    set_on_screen(true);
    entered.mouse = entered.raw && entered.terminal.write_all(MOUSE_REPORTS_ON).is_ok();
    let opened = capture_lines(libc::STDERR_FILENO, keep_captured_line).and_then(
        |(capture, for_drawing)| {
            entered.capture = Some(capture);
            // Anything written before the capture leaves the screen. This
            // clear asks the terminal nothing, unlike Terminal::clear(),
            // which asks for the cursor's place and waits for the answer.
            execute!(entered.terminal, Clear(ClearType::All))?;
            Terminal::new(CrosstermBackend::new(for_drawing))
        },
    );
    let (keys, mouse) = (entered.raw, entered.mouse);
    *lock(&ENTERED) = Some(entered);
    leave_on_every_exit();
    match opened {
        Ok(view_terminal) => Some(Opened {
            terminal: view_terminal,
            theme,
            keys,
            mouse,
        }),
        Err(_) => {
            leave();
            None
        }
    }
}

/// Makes a panic and the process's exit give the terminal back first.
fn leave_on_every_exit() {
    let earlier_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        leave();
        earlier_hook(info);
    }));
    // Safety: leave_at_exit may run at any exit, on any thread.
    unsafe { libc::atexit(leave_at_exit) };
}

/// Gives the terminal back at the process's exit.
extern "C" fn leave_at_exit() {
    leave();
}

/// Gives the terminal back before the engine writes a fatal error's message
/// to stderr and ends the process: the engine's hook for its fatal errors.
pub extern "C" fn leave_before_fatal_error() {
    leave();
}

/// Gives the terminal back as the view found it, waiting for a frame that
/// another thread writes. Then it writes to stderr the log's lines that
/// stderr has not shown. Any thread may call it, any number of times. At the
/// process's exit, the thread-local data that thread::current() and
/// io::stderr() may need can be gone, so it uses neither.
pub fn leave() {
    let this_thread = find_thread_id();
    let drawing_here = *lock(&DRAWING_THREAD) == Some(this_thread);
    let _frame = (!drawing_here).then(|| lock(&DRAWING));
    let Some(entered) = lock(&ENTERED).take() else {
        return;
    };
    give_back(entered);
    set_on_screen(false);
    // Safety: descriptor 2 stays open, and ManuallyDrop never closes it.
    let mut stderr = ManuallyDrop::new(unsafe { File::from_raw_fd(libc::STDERR_FILENO) });
    for line in take_unprinted_lines() {
        let _ = writeln!(stderr, "{line}");
    }
}

/// Writes to the terminal of `entered` what gives it back as the view found
/// it: no mouse reports, the normal screen and the cursor. It ends raw mode,
/// and gives stderr back once its captured lines are in the log.
fn give_back(mut entered: Entered) {
    if entered.mouse {
        let _ = entered.terminal.write_all(MOUSE_REPORTS_OFF);
    }
    let _ = execute!(entered.terminal, LeaveAlternateScreen, cursor::Show);
    if entered.raw {
        let _ = terminal::disable_raw_mode();
    }
    if let Some(capture) = entered.capture {
        capture.release();
    }
}

/// Draws a frame on `view_terminal` with `draw`, while the view is on the
/// screen. Returns false once leave() has given the terminal back, without
/// drawing.
pub fn draw_frame<B: Backend>(
    view_terminal: &mut Terminal<B>,
    draw: impl FnOnce(&mut Frame),
) -> bool {
    let _frame = lock(&DRAWING);
    if lock(&ENTERED).is_none() {
        return false;
    }
    *lock(&DRAWING_THREAD) = Some(find_thread_id());
    let _ = view_terminal.draw(draw);
    *lock(&DRAWING_THREAD) = None;
    true
}

/// Turns the mouse's reports on or off while the view is on the screen and
/// takes keys. Returns whether the terminal reports the mouse now.
pub fn set_mouse_reports(on: bool) -> bool {
    let _frame = lock(&DRAWING);
    let mut entered = lock(&ENTERED);
    let Some(entered) = entered.as_mut().filter(|entered| entered.raw) else {
        return false;
    };
    let reports = if on {
        MOUSE_REPORTS_ON
    } else {
        MOUSE_REPORTS_OFF
    };
    if entered.terminal.write_all(reports).is_ok() {
        entered.mouse = on;
    }
    entered.mouse
}

/// Returns the colors that `choice` gives, for a terminal whose background
/// is `background`, if it told it, and whose COLORTERM variable is
/// `colorterm`. ThemeChoice::Auto without a background gives the light
/// colors.
fn choose_theme(choice: ThemeChoice, background: Option<Rgb>, colorterm: Option<&str>) -> Theme {
    let mut theme = match (choice, background) {
        (ThemeChoice::Dark, _) => DARK,
        (ThemeChoice::Auto, Some(background)) => Theme::for_background(background),
        (ThemeChoice::Light | ThemeChoice::Auto, _) => LIGHT,
    };
    theme.true_color = is_true_color(colorterm);
    theme
}

/// Asks the terminal on `output` for its background color, and reads its
/// answer from `input` for at most `wait`. The question for the terminal's
/// attributes follows. Every terminal answers that one, in order, so its
/// answer ends the wait at once when no color comes. Returns None without a
/// color.
fn ask_background(input: RawFd, output: &mut impl Write, wait: Duration) -> Option<Rgb> {
    output.write_all(BACKGROUND_QUESTION).ok()?;
    output.write_all(ATTRIBUTES_QUESTION).ok()?;
    output.flush().ok()?;
    let deadline = Instant::now() + wait;
    let mut answer = Vec::new();
    while !has_attributes_answer(&answer) {
        let left = deadline.saturating_duration_since(Instant::now());
        let mut ready = libc::pollfd {
            fd: input,
            events: libc::POLLIN,
            revents: 0,
        };
        let milliseconds = left.as_millis().min(i32::MAX as u128) as i32;
        // Safety: one pollfd, owned here.
        if unsafe { libc::poll(&mut ready, 1, milliseconds) } <= 0 {
            break;
        }
        let mut bytes = [0u8; ANSWER_READ_BYTES];
        // Safety: the read fills at most the bytes owned here.
        let read = unsafe { libc::read(input, bytes.as_mut_ptr().cast(), bytes.len()) };
        if read <= 0 {
            break;
        }
        answer.extend_from_slice(&bytes[..read as usize]);
    }
    parse_background_answer(&answer)
}

/// Returns whether `answer` holds the whole answer to the attributes
/// question.
fn has_attributes_answer(answer: &[u8]) -> bool {
    answer
        .windows(ATTRIBUTES_ANSWER_START.len())
        .position(|window| window == ATTRIBUTES_ANSWER_START)
        .is_some_and(|start| answer[start..].contains(&ATTRIBUTES_ANSWER_END))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::os::fd::FromRawFd;
    use std::sync::mpsc;

    use ratatui::backend::TestBackend;

    /// A wait longer than any test: an answer must end it.
    const LONG_WAIT: Duration = Duration::from_secs(3600);

    /// Returns a new pipe's read end and write end.
    fn make_pipe() -> (File, File) {
        let mut ends = [0; 2];
        // Safety: pipe fills both descriptors when it succeeds.
        assert_eq!(unsafe { libc::pipe(ends.as_mut_ptr()) }, 0);
        // Safety: both descriptors are new, and owned by the files alone.
        unsafe { (File::from_raw_fd(ends[0]), File::from_raw_fd(ends[1])) }
    }

    #[test]
    fn the_terminals_answers_give_its_background_or_end_the_wait() {
        let (input, mut terminal) = make_pipe();
        terminal
            .write_all(b"\x1b]11;rgb:1c1c/1c1c/1e1e\x1b\\\x1b[?62;22c")
            .unwrap();
        let mut questions = Vec::new();
        let background = ask_background(input.as_raw_fd(), &mut questions, LONG_WAIT);
        assert_eq!(background, Some((0x1c, 0x1c, 0x1e)));
        assert_eq!(questions, b"\x1b]11;?\x1b\\\x1b[c");
        // a terminal without the color answers the attributes alone
        terminal.write_all(b"\x1b[?1;2c").unwrap();
        assert_eq!(
            ask_background(input.as_raw_fd(), &mut Vec::new(), LONG_WAIT),
            None
        );
        // a terminal that answers nothing
        assert_eq!(
            ask_background(input.as_raw_fd(), &mut Vec::new(), Duration::ZERO),
            None
        );
    }

    #[test]
    fn the_choice_and_the_background_give_the_colors() {
        let dark = choose_theme(
            ThemeChoice::Auto,
            Some((0x1c, 0x1c, 0x1e)),
            Some("truecolor"),
        );
        assert_eq!(
            (dark.accent, dark.ground),
            (DARK.accent, (0x1c, 0x1c, 0x1e))
        );
        assert!(dark.true_color);
        let unknown = choose_theme(ThemeChoice::Auto, None, None);
        assert_eq!(unknown.accent, LIGHT.accent);
        assert!(!unknown.true_color, "256 colors");
        let chosen = choose_theme(ThemeChoice::Dark, Some((255, 255, 255)), None);
        assert_eq!(chosen.ground, DARK.ground);
        let light = choose_theme(ThemeChoice::Light, Some((0, 0, 0)), None);
        assert_eq!(light.ground, LIGHT.ground);
    }

    #[test]
    fn the_terminal_goes_back_as_the_view_found_it() {
        let (mut terminal_out, terminal) = make_pipe();
        // A pipe's end stands for stderr, captured.
        let (mut stderr_out, stderr_end) = make_pipe();
        let stderr = stderr_end.as_raw_fd();
        let (sender, kept) = mpsc::channel();
        let keep_line = move |text: &str| sender.send(text.to_string()).unwrap();
        let (capture, _) = capture_lines(stderr, keep_line).unwrap();
        let entered = Entered {
            terminal,
            raw: false,
            mouse: true,
            capture: Some(capture),
        };
        // Safety: the test's own open descriptor.
        unsafe { libc::write(stderr, b"a last line\n".as_ptr().cast(), 12) };
        give_back(entered);
        assert_eq!(kept.recv().unwrap(), "a last line");
        let mut shown = [0; 30];
        terminal_out.read_exact(&mut shown).unwrap();
        assert_eq!(&shown, b"\x1b[?1006l\x1b[?1000l\x1b[?1049l\x1b[?25h");
        // stderr is the pipe's end again
        // Safety: the test's own open descriptor.
        unsafe { libc::write(stderr, b"after\n".as_ptr().cast(), 6) };
        let mut after = [0; 6];
        stderr_out.read_exact(&mut after).unwrap();
        assert_eq!(&after, b"after\n");
    }

    #[test]
    fn off_the_screen_nothing_is_drawn_or_given_back() {
        leave();
        let mut view_terminal = Terminal::new(TestBackend::new(20, 4)).unwrap();
        assert!(!draw_frame(&mut view_terminal, |_| panic!("no frame")));
        assert!(!set_mouse_reports(true));
    }
}
