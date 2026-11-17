//! The terminal UI's chrome: the header, the tab rows, the rule, and the keys
//! at the bottom. It also holds what the person chose, and the places that a
//! click chooses.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;

use super::draw::{fit_line, format_duration, label, plain, strong};
use super::theme::Theme;

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

/// What every view draws from.
pub struct Screen<'a> {
    /// The colors.
    pub theme: &'a Theme,
    /// The server's address, such as 127.0.0.1:8008.
    pub address: &'a str,
    /// The seconds since the server started.
    pub uptime_seconds: f64,
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
    use crate::ui::testing::{draw_on_terminal, find_in_row, read_row};
    use crate::ui::theme::DARK;

    /// Draws the chrome at `width` x `height` with `choices`.
    fn draw(width: u16, height: u16, choices: &mut Choices) -> (Buffer, Rect) {
        let screen = Screen {
            theme: &DARK,
            address: "127.0.0.1:52878",
            uptime_seconds: 57.0,
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
