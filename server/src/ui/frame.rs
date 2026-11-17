//! The terminal UI's chrome: the header, the tab rows, the rule, and the keys
//! at the bottom. It also holds what the person chose, and the places that a
//! click chooses.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;

use super::draw::{fit_line, format_duration, label, plain, strong};
use super::status::{ModelStatus, Phase};
use super::theme::{Rgb, Theme};

/// The views of the top tab row. Each one shows what all models share, except
/// Models, which shows one model at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    /// The system, the requests and the log.
    Overview,
    /// The memory map.
    Memory,
    /// The speeds and the requests over the history window.
    History,
    /// One model's own sub-views.
    Models,
}

/// The views in the order of their tabs and keys, with their names.
pub const VIEWS: [(View, &str); 4] = [
    (View::Overview, "Overview"),
    (View::Memory, "Memory"),
    (View::History, "History"),
    (View::Models, "Models"),
];

/// A kind of model. Each kind has its own sub-views in Models, and every model
/// of the kind shows them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A language model: it keeps sessions, and it may route its tokens to
    /// experts.
    Llm,
}

/// A sub-view of Models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubView {
    /// An LLM's sessions as a tree of their shared prefixes, and the text of
    /// the chosen one.
    States,
    /// The experts that a decoding request's latest tokens chose.
    Experts,
}

impl Kind {
    /// Returns the sub-views of a model of this kind, in the order of their
    /// tabs.
    pub fn list_sub_views(self) -> &'static [SubView] {
        match self {
            Kind::Llm => &[SubView::States, SubView::Experts],
        }
    }
}

impl SubView {
    /// Returns the sub-view's name on its tab.
    pub fn name(self) -> &'static str {
        match self {
            SubView::States => "States",
            SubView::Experts => "Experts",
        }
    }
}

/// A model as the views show it.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    /// The model's name, as requests give it.
    pub name: String,
    /// The model's kind.
    pub kind: Kind,
}

/// What every view draws from.
pub struct Screen<'a> {
    /// The colors.
    pub theme: &'a Theme,
    /// The server's address, such as 127.0.0.1:8008.
    pub address: &'a str,
    /// The seconds since the server started.
    pub uptime_seconds: f64,
    /// The served models, in their configured order.
    pub models: &'a [ModelInfo],
    /// Each model's latest status, in the same order.
    pub statuses: &'a [ModelStatus],
}

/// The keys that the keys row names: each key, and what it does.
pub type Keys = Vec<(&'static str, String)>;

/// What a click on a place of the screen chooses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// A view of the top tab row.
    View(View),
}

/// A place that a click acts on: the columns from x0 to before x1 of row y.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClickTarget {
    /// The row.
    pub y: u16,
    /// The first column.
    pub x0: u16,
    /// The column after the last.
    pub x1: u16,
    /// What a click there chooses.
    pub target: Target,
}

/// What the person chose in the terminal UI.
#[derive(Clone, Debug, PartialEq)]
pub struct Choices {
    /// The view shown.
    pub view: View,
    /// Whether the view takes the mouse. Without it, the terminal selects
    /// text.
    pub mouse: bool,
    /// The places that a click acts on, in the frame drawn last.
    pub targets: Vec<ClickTarget>,
}

impl Default for Choices {
    fn default() -> Choices {
        Choices {
            view: View::Overview,
            mouse: true,
            targets: Vec::new(),
        }
    }
}

impl Choices {
    /// Returns what a click at column `x` of row `y` chooses.
    pub fn find_target(&self, x: u16, y: u16) -> Option<Target> {
        self.targets
            .iter()
            .find(|target| target.y == y && (target.x0..target.x1).contains(&x))
            .map(|target| target.target)
    }
}

/// What a model's tab shows that the model does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    /// It runs a request.
    Busy,
    /// It is loaded and runs nothing.
    Idle,
    /// It loads its weights.
    Loading,
    /// It is not loaded, or a yield holds it.
    Unloaded,
    /// Its last load failed.
    Failed,
}

impl Mark {
    /// Returns the mark of a model with `status`.
    pub fn of(status: &ModelStatus) -> Mark {
        if status.load_error.is_some() {
            Mark::Failed
        } else if status.phase == Phase::Loading {
            Mark::Loading
        } else if status.phase == Phase::Held || !status.loaded {
            Mark::Unloaded
        } else if status.count_requests().0 > 0 {
            Mark::Busy
        } else {
            Mark::Idle
        }
    }

    /// Returns the mark's symbol and color. None takes the terminal's own
    /// text color.
    pub fn look(self, theme: &Theme) -> (&'static str, Option<Rgb>) {
        match self {
            Mark::Busy => ("●", Some(theme.accent)),
            Mark::Idle => ("○", None),
            Mark::Loading => ("●", Some(theme.amber)),
            Mark::Unloaded => ("○", Some(theme.label)),
            Mark::Failed => ("●", Some(theme.red)),
        }
    }
}

/// The rows of the chrome above a view of what all models share: the header,
/// the tabs and the rule.
const CHROME_ROWS_ABOVE: u16 = 3;
/// The fewest columns in which the chrome draws more than its title.
const LEAST_CHROME_COLUMNS: u16 = 20;
/// The columns between two tabs.
const TAB_GAP: u16 = 2;

/// Draws the chrome in `area` with the keys of the view shown, and returns the
/// area left for the view. The click targets of the tabs go to `choices`. In
/// an area too small for the chrome, only the title shows, and the area left
/// is empty.
pub fn draw_chrome(
    buf: &mut Buffer,
    area: Rect,
    screen: &Screen,
    choices: &mut Choices,
    keys: &[(&'static str, String)],
) -> Rect {
    let theme = screen.theme;
    let above = CHROME_ROWS_ABOVE;
    let width = area.width as usize;
    let title = Line::from(vec![strong(" eremite  "), label(theme, screen.address)]);
    if area.height < above + 2 || area.width < LEAST_CHROME_COLUMNS {
        buf.set_line(area.x, area.y, &fit_line(theme, title, width), area.width);
        return Rect { height: 0, ..area };
    }
    let uptime = Line::from(label(
        theme,
        format!("up {} ", format_duration(screen.uptime_seconds)),
    ));
    buf.set_line(area.x, area.y, &fit_line(theme, title, width), area.width);
    let uptime_width = uptime.width() as u16 + 1;
    buf.set_line(
        area.right().saturating_sub(uptime_width),
        area.y,
        &uptime,
        uptime_width,
    );
    let tabs = VIEWS
        .iter()
        .enumerate()
        .map(|(i, &(view, name))| {
            let line = Line::from(vec![
                label(theme, format!(" {} ", i + 1)),
                plain(format!("{name} ")),
            ]);
            (line, Target::View(view), view == choices.view)
        })
        .collect();
    draw_tabs(
        buf,
        theme,
        (area.x + 1, area.y + 1),
        area.right(),
        tabs,
        &mut choices.targets,
    );
    buf.set_string(
        area.x,
        area.y + above - 1,
        "─".repeat(width),
        Style::new().fg(theme.color(theme.rule)),
    );
    let mut spans = vec![plain(" ")];
    for (key, what) in keys {
        spans.extend([
            strong(*key),
            plain(" "),
            label(theme, what.clone()),
            plain("   "),
        ]);
    }
    buf.set_line(
        area.x,
        area.bottom() - 1,
        &fit_line(theme, Line::from(spans), width),
        area.width,
    );
    Rect {
        y: area.y + above,
        height: area.height - above - 1,
        ..area
    }
}

/// Draws `items` as tabs from column `start.0` of row `start.1` up to
/// `right`, two columns apart. Each item is its line, the target that a click
/// on it chooses, and whether it is chosen. A chosen tab is white on the
/// accent. Returns the column where a next tab would start.
fn draw_tabs(
    buf: &mut Buffer,
    theme: &Theme,
    start: (u16, u16),
    right: u16,
    items: Vec<(Line<'static>, Target, bool)>,
    targets: &mut Vec<ClickTarget>,
) -> u16 {
    let (mut x, y) = start;
    let chosen_style = Style::new()
        .fg(theme.color(theme.on_accent))
        .bg(theme.color(theme.accent))
        .add_modifier(Modifier::BOLD);
    for (line, target, chosen) in items {
        let width = line.width() as u16;
        if x + width > right {
            break;
        }
        if chosen {
            let text: String = line
                .spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect();
            buf.set_string(x, y, text, chosen_style);
        } else {
            buf.set_line(x, y, &line, width);
        }
        targets.push(ClickTarget {
            y,
            x0: x,
            x1: x + width,
            target,
        });
        x += width + TAB_GAP;
    }
    x
}

/// Returns the keys of the chrome for `choices` at the start and at the end of
/// the keys row, around those of the view shown.
pub fn list_chrome_keys(choices: &Choices) -> (Keys, Keys) {
    let first = vec![("1–4", "views".to_string())];
    let mouse = if choices.mouse {
        "mouse: clicks"
    } else {
        "mouse: selects text"
    };
    let last = vec![
        ("m", mouse.to_string()),
        ("q", "stop the server".to_string()),
    ];
    (first, last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::status::{Activity, Request};
    use crate::ui::testing::{draw_on_terminal, find_in_row, read_row};
    use crate::ui::theme::DARK;

    /// Draws the chrome at `width` x `height` with `choices`.
    fn draw(width: u16, height: u16, choices: &mut Choices) -> (Buffer, Rect) {
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 57.0,
            models: &[],
            statuses: &[],
        };
        let (mut keys, last) = list_chrome_keys(choices);
        keys.extend(last);
        let mut inner = Rect::default();
        let buf = draw_on_terminal(width, height, |buf, area| {
            inner = draw_chrome(buf, area, &screen, choices, &keys);
        });
        (buf, inner)
    }

    #[test]
    fn the_chrome_shows_the_header_the_chosen_view_and_the_keys() {
        let mut choices = Choices::default();
        let (buf, inner) = draw(160, 45, &mut choices);
        assert!(read_row(&buf, 0).starts_with(" eremite  127.0.0.1:52878"));
        assert!(
            read_row(&buf, 0).ends_with("up 57s  "),
            "{}",
            read_row(&buf, 0)
        );
        let at = find_in_row(&buf, 1, " 1 Overview ").expect("the tab");
        assert_eq!(buf[(at, 1)].bg, DARK.color(DARK.accent), "the chosen view");
        assert_eq!(buf[(at + 3, 1)].fg, DARK.color(DARK.on_accent));
        assert_eq!(read_row(&buf, 2), "─".repeat(160));
        assert!(read_row(&buf, 44).starts_with(" 1–4 views   m mouse: clicks   q stop"));
        assert_eq!(inner, Rect::new(0, 3, 160, 41));
        let memory = find_in_row(&buf, 1, " 2 Memory ").unwrap();
        assert_eq!(
            choices.find_target(memory + 4, 1),
            Some(Target::View(View::Memory))
        );
        choices.mouse = false;
        let (buf, _) = draw(100, 30, &mut choices);
        assert!(read_row(&buf, 29).contains("m mouse: selects text"));
    }

    #[test]
    fn a_models_mark_shows_what_it_does() {
        let busy = ModelStatus {
            loaded: true,
            requests: vec![Request {
                status: Activity::Prefill,
                ..Request::default()
            }],
            ..ModelStatus::default()
        };
        let loading = ModelStatus {
            phase: Phase::Loading,
            ..ModelStatus::default()
        };
        let failed = ModelStatus {
            load_error: Some("does not fit".into()),
            ..ModelStatus::default()
        };
        let idle = ModelStatus {
            loaded: true,
            ..ModelStatus::default()
        };
        let held = ModelStatus {
            loaded: true,
            phase: Phase::Held,
            ..ModelStatus::default()
        };
        let marks = [
            &busy,
            &loading,
            &failed,
            &idle,
            &held,
            &ModelStatus::default(),
        ]
        .map(Mark::of);
        assert_eq!(
            marks,
            [
                Mark::Busy,
                Mark::Loading,
                Mark::Failed,
                Mark::Idle,
                Mark::Unloaded,
                Mark::Unloaded
            ]
        );
        assert_eq!(Mark::Busy.look(&DARK), ("●", Some(DARK.accent)));
        assert_eq!(Mark::Loading.look(&DARK), ("●", Some(DARK.amber)));
        assert_eq!(Mark::Failed.look(&DARK), ("●", Some(DARK.red)));
        assert_eq!(Mark::Unloaded.look(&DARK), ("○", Some(DARK.label)));
        assert_eq!(Mark::Idle.look(&DARK), ("○", None));
        assert_eq!(
            Kind::Llm.list_sub_views(),
            [SubView::States, SubView::Experts]
        );
        assert_eq!(SubView::Experts.name(), "Experts");
    }

    #[test]
    fn a_small_terminal_shows_the_title_alone() {
        for (width, height) in [(100, 30), (40, 8), (19, 10), (12, 4), (1, 1)] {
            let mut choices = Choices::default();
            let (buf, inner) = draw(width, height, &mut choices);
            assert!(read_row(&buf, 0).starts_with(" eremite"[..width.min(8) as usize].trim_end()));
            if width < LEAST_CHROME_COLUMNS || height < 5 {
                assert_eq!(inner.height, 0, "{width} x {height}");
            } else {
                assert_eq!(inner.height, height - 4, "{width} x {height}");
            }
        }
    }
}
