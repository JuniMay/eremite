//! The terminal UI's log: the lines of the engine, of the server and of each
//! model, each with the model it is about. The engine's log hook, the
//! worker's hook for model lines and the capture of stderr keep them here.
//! The log view draws the newest lines that pass its filter.

use std::collections::VecDeque;
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::draw::{clip_text, colored, fit_line, label, plain, strong};
use super::frame::{Choices, ModelInfo, Screen};
use crate::engine;

/// The lines that the log keeps. The hooks reach them here, so they are
/// global.
static KEPT: Mutex<Kept> = Mutex::new(Kept {
    lines: VecDeque::new(),
    logged: 0,
    printed: 0,
});
/// Whether the view is on the screen. While it is not, the hooks also write
/// each line to stderr.
static ON_SCREEN: AtomicBool = AtomicBool::new(false);

/// The log lines that the view keeps: the newest ones. The view copies them
/// for each frame. The reason for this count is unknown.
const KEPT_LINES: usize = 2000;
/// The name of the log's lines about no model.
const SERVER: &str = "server";
/// The columns between a line's model and its text.
const MODEL_GAP: usize = 2;

/// A line of the log.
#[derive(Clone, Debug, PartialEq)]
pub struct LogLine {
    /// The time of day when it was logged, such as 19:19:22.784.
    pub time: String,
    /// The index of the model that it is about. None for a line about no
    /// model.
    pub model: Option<usize>,
    /// The line's text.
    pub text: String,
}

/// The lines that the log keeps, and how many of them stderr showed.
struct Kept {
    /// The newest lines.
    lines: VecDeque<LogLine>,
    /// The lines logged since the start.
    logged: u64,
    /// Of those, the first ones that stderr showed.
    printed: u64,
}

/// Which lines the log shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogFilter {
    /// Every line.
    #[default]
    All,
    /// The lines about the model with this index.
    Model(usize),
    /// The lines about no model.
    Server,
}

impl LogFilter {
    /// Returns the filter after this one, with `models` models: all lines,
    /// then each model's, then the server's own, then all again.
    pub fn next(self, models: usize) -> LogFilter {
        match self {
            LogFilter::All if models > 0 => LogFilter::Model(0),
            LogFilter::Model(i) if i + 1 < models => LogFilter::Model(i + 1),
            LogFilter::All | LogFilter::Model(_) => LogFilter::Server,
            LogFilter::Server => LogFilter::All,
        }
    }

    /// Returns whether `line` passes the filter.
    pub fn passes(self, line: &LogLine) -> bool {
        match self {
            LogFilter::All => true,
            LogFilter::Model(i) => line.model == Some(i),
            LogFilter::Server => line.model.is_none(),
        }
    }

    /// Describes what the filter shows, for the log's rule: all, a model's
    /// name, or server.
    pub fn describe(self, models: &[ModelInfo]) -> String {
        match self {
            LogFilter::All => "all".into(),
            LogFilter::Model(i) => models
                .get(i)
                .map_or(String::new(), |model| model.name.clone()),
            LogFilter::Server => SERVER.into(),
        }
    }
}

/// Locks the kept lines. It also takes them after a thread panicked with the
/// lock held.
fn lock_kept() -> MutexGuard<'static, Kept> {
    KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Keeps `line`. The oldest line goes when the log is full.
fn keep(line: LogLine) {
    let mut kept = lock_kept();
    if !ON_SCREEN.load(Ordering::Acquire) {
        eprintln!("{} {}", line.time, line.text);
        kept.printed = kept.logged + 1;
    }
    kept.logged += 1;
    if kept.lines.len() == KEPT_LINES {
        kept.lines.pop_front();
    }
    kept.lines.push_back(line);
}

/// The engine's log hook: it keeps each line of the engine's log.
pub const ENGINE_LOG_HOOK: extern "C" fn(*const c_char) = keep_engine_line;

/// Keeps a line of the engine's log. The line starts with its time of day.
/// The engine passes a NUL-terminated line that stays valid during the call.
extern "C" fn keep_engine_line(line: *const c_char) {
    // Safety: the engine passes a NUL-terminated line, valid during the call.
    let line = unsafe { CStr::from_ptr(line) }.to_string_lossy();
    let (time, text) = line.split_once(' ').unwrap_or(("", &line));
    keep(LogLine {
        time: time.to_string(),
        model: None,
        text: text.to_string(),
    });
}

/// Keeps `line`, a line about the model at `model_index`: the worker's hook
/// for model lines. While the view is not on the screen, the line goes to the
/// engine's log instead.
pub fn keep_model_line(model_index: usize, line: &str) {
    if !ON_SCREEN.load(Ordering::Acquire) {
        engine::log(line);
        return;
    }
    keep(LogLine {
        time: format_time_of_day(SystemTime::now()),
        model: Some(model_index),
        text: line.to_string(),
    });
}

/// Keeps `text`, a line that something wrote to stderr while the view
/// captured it, as a line about no model with the time it was read. It never
/// writes to stderr, which is the pipe that the line came from.
pub fn keep_captured_line(text: &str) {
    let line = LogLine {
        time: format_time_of_day(SystemTime::now()),
        model: None,
        text: text.to_string(),
    };
    let mut kept = lock_kept();
    kept.logged += 1;
    if kept.lines.len() == KEPT_LINES {
        kept.lines.pop_front();
    }
    kept.lines.push_back(line);
}

/// Sets whether the view is on the screen.
pub fn set_on_screen(on: bool) {
    let _kept = lock_kept();
    ON_SCREEN.store(on, Ordering::Release);
}

/// Returns a copy of the kept lines, the oldest first.
pub fn copy_lines() -> Vec<LogLine> {
    lock_kept().lines.iter().cloned().collect()
}

/// Returns the kept lines that stderr has not shown, as the engine's log
/// writes them, and counts them as shown.
pub fn take_unprinted_lines() -> Vec<String> {
    let mut kept = lock_kept();
    let first = kept.logged - kept.lines.len() as u64;
    let from = kept.printed.saturating_sub(first) as usize;
    let lines = kept
        .lines
        .iter()
        .skip(from)
        .map(|line| format!("{} {}", line.time, line.text))
        .collect();
    kept.printed = kept.logged;
    lines
}

/// Formats the local time of day at `now` as the engine's log does, such as
/// 19:19:22.784.
fn format_time_of_day(now: SystemTime) -> String {
    let since = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = since.as_secs() as libc::time_t;
    // Safety: localtime_r fills the zeroed struct, and keeps no pointer to it.
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&seconds, &mut local) };
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        local.tm_hour,
        local.tm_min,
        local.tm_sec,
        since.subsec_millis()
    )
}

/// Draws the log in `area`. Its rule names what the filter of `choices`
/// shows. Under it come the newest lines that pass the filter, scrolled back
/// from the newest by the lines that `choices` says. Each line shows its time,
/// its model or "server", and its text. A text too long for its row is cut,
/// and "…" takes the row's last column. The scroll stays within the lines,
/// and `choices` keeps the rows of a page.
pub fn draw_log(
    buf: &mut Buffer,
    area: Rect,
    screen: &Screen,
    lines: &[LogLine],
    choices: &mut Choices,
) {
    let theme = screen.theme;
    if area.height == 0 {
        return;
    }
    let width = area.width as usize;
    let title = Style::new()
        .fg(theme.color(theme.accent))
        .add_modifier(Modifier::BOLD);
    let mut rule = Line::from(vec![
        colored(theme, theme.rule, "─ "),
        Span::styled("Log ", title),
        colored(theme, theme.rule, "─ "),
        label(theme, "showing "),
        strong(choices.log_filter.describe(screen.models)),
        plain(" "),
    ]);
    let rest = width.saturating_sub(rule.width());
    rule.spans
        .push(colored(theme, theme.rule, "─".repeat(rest)));
    buf.set_line(area.x, area.y, &rule, area.width);
    let rows = (area.height - 1) as usize;
    let shown: Vec<&LogLine> = lines
        .iter()
        .filter(|line| choices.log_filter.passes(line))
        .collect();
    choices.log_page = rows.max(1);
    choices.log_back = choices.log_back.min(shown.len().saturating_sub(rows));
    let end = shown.len() - choices.log_back;
    let name_width = screen
        .models
        .iter()
        .map(|model| model.name.chars().count())
        .chain([SERVER.len()])
        .max()
        .unwrap_or(0)
        + MODEL_GAP;
    let name_style = Style::new()
        .fg(theme.color(theme.ink))
        .add_modifier(Modifier::BOLD);
    for (row, line) in shown[end.saturating_sub(rows)..end].iter().enumerate() {
        let name = match line.model {
            Some(i) => screen.models.get(i).map_or("", |model| model.name.as_str()),
            None => SERVER,
        };
        let padding = " ".repeat(name_width.saturating_sub(name.chars().count()));
        let name = if line.model.is_some() {
            Span::styled(name.to_string(), name_style)
        } else {
            label(theme, name)
        };
        // The text takes the columns left after the time and the name.
        let room = width.saturating_sub(1 + line.time.len() + 1 + name_width);
        let spans = vec![
            plain(" "),
            label(theme, format!("{} ", line.time)),
            name,
            plain(padding),
            label(theme, clip_text(&line.text, room)),
        ];
        let y = area.y + 1 + row as u16;
        buf.set_line(
            area.x,
            y,
            &fit_line(theme, Line::from(spans), width),
            area.width,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::Kind;
    use crate::ui::samples::Samples;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, find_in_row, read_row};
    use crate::ui::theme::DARK;
    use std::ffi::CString;

    /// The two models of the tests.
    fn models() -> Vec<ModelInfo> {
        ["deepseek-v4.1-flash", "qwen3.8-27b"]
            .map(|name| ModelInfo {
                name: name.into(),
                kind: Kind::Llm,
            })
            .to_vec()
    }

    /// `count` lines: request lines of the two models in turn, and every
    /// fifth one about no model.
    fn lines(count: usize) -> Vec<LogLine> {
        (0..count)
            .map(|i| LogLine {
                time: format!("19:19:{:02}.000", i % 60),
                model: (i % 5 != 4).then_some(i % 2),
                text: format!("line {i}"),
            })
            .collect()
    }

    /// Draws the log of `lines` at 160 x `height` with `choices`.
    fn draw(lines: &[LogLine], height: u16, choices: &mut Choices) -> Buffer {
        let models = models();
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 57.0,
            models: &models,
            statuses: &[],
            samples: &Samples::new(WINDOWS),
            home: "",
            session_text: None,
        };
        draw_on_terminal(160, height, |buf, area| {
            draw_log(buf, area, &screen, lines, choices);
        })
    }

    #[test]
    fn the_log_shows_the_newest_lines_with_their_models() {
        let mut choices = Choices::default();
        let buf = draw(&lines(40), 11, &mut choices);
        assert!(read_row(&buf, 0).starts_with("─ Log ─ showing all ───"));
        let last = read_row(&buf, 10);
        assert!(
            last.starts_with(" 19:19:39.000 server               line 39"),
            "{last}"
        );
        let model = read_row(&buf, 8);
        assert!(model.starts_with(" 19:19:37.000 qwen3.8-27b          line 37"));
        // a model's name in bold ink, the server's in the label color
        let model_at = find_in_row(&buf, 8, "qwen").unwrap();
        assert_eq!(buf[(model_at, 8)].fg, DARK.color(DARK.ink));
        assert!(buf[(model_at, 8)].modifier.contains(Modifier::BOLD));
        let server_at = find_in_row(&buf, 10, "server").unwrap();
        assert_eq!(buf[(server_at, 10)].fg, DARK.color(DARK.label));
        assert_eq!(choices.log_page, 10);
        // scrolled back, and no further than the oldest line
        choices.log_back = 5;
        let buf = draw(&lines(40), 11, &mut choices);
        assert!(read_row(&buf, 10).contains("line 34"));
        choices.log_back = usize::MAX;
        let buf = draw(&lines(40), 11, &mut choices);
        assert!(
            read_row(&buf, 1).contains("line 0 "),
            "{}",
            read_row(&buf, 1)
        );
        assert_eq!(choices.log_back, 30);
        // a long line keeps the start of its text
        let long = [LogLine {
            time: "19:19:40.000".into(),
            model: None,
            text: format!("/Users/me/models/flash: registered {}", "x".repeat(200)),
        }];
        let row = read_row(&draw(&long, 2, &mut Choices::default()), 1);
        assert!(
            row.starts_with(
                " 19:19:40.000 server               /Users/me/models/flash: registered x"
            ),
            "{row}"
        );
        assert!(row.ends_with("xx…"), "{row}");
    }

    #[test]
    fn the_filter_shows_each_models_lines_then_the_servers() {
        let models = models();
        let mut filter = LogFilter::All;
        let mut seen = Vec::new();
        for _ in 0..4 {
            seen.push(filter.describe(&models));
            filter = filter.next(models.len());
        }
        assert_eq!(
            seen,
            ["all", "deepseek-v4.1-flash", "qwen3.8-27b", "server"]
        );
        assert_eq!(filter, LogFilter::All);
        let mut choices = Choices {
            log_filter: LogFilter::Model(1),
            ..Choices::default()
        };
        let buf = draw(&lines(40), 6, &mut choices);
        assert!(read_row(&buf, 0).starts_with("─ Log ─ showing qwen3.8-27b ───"));
        for y in 1..6 {
            assert!(
                read_row(&buf, y).contains("qwen3.8-27b"),
                "{}",
                read_row(&buf, y)
            );
        }
        choices.log_filter = LogFilter::Server;
        let buf = draw(&lines(40), 3, &mut choices);
        assert!(read_row(&buf, 2).contains("server               line 39"));
    }

    #[test]
    fn the_hooks_keep_the_lines_with_their_time_and_model() {
        let line = CString::new("19:19:22.784 http 4: arrived from 127.0.0.1").unwrap();
        set_on_screen(true);
        ENGINE_LOG_HOOK(line.as_ptr());
        keep_model_line(1, "req 6: queued for qwen3.8-27b");
        let kept = copy_lines();
        assert!(kept.contains(&LogLine {
            time: "19:19:22.784".into(),
            model: None,
            text: "http 4: arrived from 127.0.0.1".into(),
        }));
        let model_line = kept
            .iter()
            .find(|line| line.text == "req 6: queued for qwen3.8-27b")
            .expect("the model's line");
        assert_eq!(model_line.model, Some(1));
        assert_eq!(model_line.time.len(), "19:19:22.784".len());
        let unprinted = take_unprinted_lines();
        assert!(unprinted.contains(&"19:19:22.784 http 4: arrived from 127.0.0.1".to_string()));
        assert!(take_unprinted_lines().is_empty(), "each line once");
        // a captured line stays unprinted, even off the screen
        set_on_screen(false);
        keep_captured_line("a line that a library wrote");
        set_on_screen(true);
        let unprinted = take_unprinted_lines();
        assert_eq!(unprinted.len(), 1);
        assert!(unprinted[0].ends_with(" a line that a library wrote"));
        assert_eq!(copy_lines().last().map(|line| line.model), Some(None));
        // the seconds and the milliseconds do not depend on the time zone
        let at = UNIX_EPOCH + std::time::Duration::from_millis(3_723_456);
        assert!(format_time_of_day(at).ends_with(":03.456"));
    }
}
