//! What the terminal UI's views draw with: styled text, figures, cut lines,
//! section rules, gauges, sparklines and cells filled by eighths.

use ratatui::buffer::Buffer;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::theme::{Rgb, Theme};

/// The block characters that fill a cell from the bottom: [k] fills k
/// eighths of it.
pub const LOWER_EIGHTHS: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
/// The block characters that fill a cell from the left: [k] fills k eighths
/// of it.
pub const LEFT_EIGHTHS: [&str; 9] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
/// The eighths of one cell.
const EIGHTHS: usize = 8;
/// A sparkline's level of a value is 1 and its share of the range times this.
/// It is just under 8, so that the top value takes the full cell and no value
/// goes past it.
const SPARKLINE_LEVELS: f64 = 7.99;

/// The seconds of one minute.
const SECONDS_PER_MINUTE: u64 = 60;
/// The seconds of one hour.
const SECONDS_PER_HOUR: u64 = 3600;

/// Returns `text` in the terminal's own text color.
pub fn plain(text: impl Into<String>) -> Span<'static> {
    Span::raw(text.into())
}

/// Returns `text` in bold.
pub fn strong(text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::new().add_modifier(Modifier::BOLD))
}

/// Returns `text` in the color of labels.
pub fn label(theme: &Theme, text: impl Into<String>) -> Span<'static> {
    colored(theme, theme.label, text)
}

/// Returns `text` in the color `rgb`.
pub fn colored(theme: &Theme, rgb: Rgb, text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::new().fg(theme.color(rgb)))
}

/// Formats `value` as a whole number with a comma after each three digits,
/// such as 1,234,567. A negative value shows as 0.
pub fn format_count(value: f64) -> String {
    let digits = format!("{:.0}", value.max(0.0));
    let mut out = String::new();
    for (i, digit) in digits.chars().enumerate() {
        out.push(digit);
        let left = digits.len() - i - 1;
        if left > 0 && left % 3 == 0 {
            out.push(',');
        }
    }
    out
}

/// Formats `seconds` in whole seconds, minutes or hours, such as 59s, 1m01s
/// or 1h02m.
pub fn format_duration(seconds: f64) -> String {
    let whole = seconds.max(0.0) as u64;
    if whole >= SECONDS_PER_HOUR {
        format!(
            "{}h{:02}m",
            whole / SECONDS_PER_HOUR,
            whole % SECONDS_PER_HOUR / SECONDS_PER_MINUTE
        )
    } else if whole >= SECONDS_PER_MINUTE {
        format!(
            "{}m{:02}s",
            whole / SECONDS_PER_MINUTE,
            whole % SECONDS_PER_MINUTE
        )
    } else {
        format!("{whole}s")
    }
}

/// Formats `part` as a percent of `whole` with one decimal, such as 12.5%. A
/// whole of 0 shows as 0.0%.
pub fn format_percent(part: f64, whole: f64) -> String {
    let share = if whole > 0.0 { part / whole } else { 0.0 };
    format!("{:.1}%", 100.0 * share)
}

/// Returns the columns that `spans` take on the terminal.
pub fn measure_width(spans: &[Span]) -> usize {
    spans.iter().map(|span| span.content.width()).sum()
}

/// Returns `spans` and spaces after them up to `columns` columns, at least one
/// space.
pub fn pad_spans(mut spans: Vec<Span<'static>>, columns: usize) -> Vec<Span<'static>> {
    let spaces = columns.saturating_sub(measure_width(&spans)).max(1);
    spans.push(plain(" ".repeat(spaces)));
    spans
}

/// Returns `text` in at most `columns` columns. A longer text is cut, and
/// "…" takes the last column.
pub fn clip_text(text: &str, columns: usize) -> String {
    if text.width() <= columns {
        return text.into();
    }
    let (mut out, mut used) = (String::new(), 0);
    for character in text.chars() {
        let width = character.width().unwrap_or(0);
        if used + width + 1 > columns {
            break;
        }
        out.push(character);
        used += width;
    }
    if columns > 0 {
        out.push('…');
    }
    out
}

/// Returns `line` in at most `columns` columns. A longer line ends after its
/// last whole figure, which ends before two spaces or " ·". Without one, it
/// ends after its last whole word. Then " …" follows.
pub fn fit_line(theme: &Theme, line: Line<'static>, columns: usize) -> Line<'static> {
    if line.width() <= columns {
        return line;
    }
    let cells: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|span| span.content.chars().map(move |c| (c, span.style)))
        .collect();
    // The room before " …".
    let room = columns.saturating_sub(2);
    let (mut used, mut word_end, mut figure_end, mut stop) = (0, 0, 0, cells.len());
    for (i, &(character, _)) in cells.iter().enumerate() {
        let width = character.width().unwrap_or(0);
        if used + width > room {
            stop = i;
            break;
        }
        if character == ' ' && used > 0 && cells[i - 1].0 != ' ' {
            word_end = i;
            if matches!(cells.get(i + 1), Some((' ' | '·', _))) {
                figure_end = i;
            }
        }
        used += width;
    }
    let cut = if figure_end > 0 {
        figure_end
    } else if word_end > 0 {
        word_end
    } else {
        stop
    };
    let mut spans: Vec<Span<'static>> = Vec::new();
    for &(character, style) in &cells[..cut] {
        match spans.last_mut() {
            Some(last) if last.style == style => last.content.to_mut().push(character),
            _ => spans.push(Span::styled(character.to_string(), style)),
        }
    }
    if columns >= 2 {
        spans.push(label(theme, " …"));
    }
    Line::from(spans)
}

/// Returns a section's rule across `columns` columns: "─ Title ───", with the
/// title bold in the accent.
pub fn build_section_rule(theme: &Theme, columns: usize, title: &str) -> Line<'static> {
    let rule = columns.saturating_sub(title.width() + 3);
    Line::from(vec![
        colored(theme, theme.rule, "─ "),
        Span::styled(
            format!("{title} "),
            Style::new()
                .fg(theme.color(theme.accent))
                .add_modifier(Modifier::BOLD),
        ),
        colored(theme, theme.rule, "─".repeat(rule)),
    ])
}

/// Returns a gauge of `cells` cells. Each part, in order, fills its share of
/// the cells in its style. The cells left show in the color of rules.
pub fn build_gauge(theme: &Theme, cells: usize, parts: &[(f64, Style)]) -> Vec<Span<'static>> {
    let (mut out, mut sum, mut filled) = (Vec::new(), 0.0, 0);
    for &(share, style) in parts {
        sum += share.max(0.0);
        let to = ((sum * cells as f64).round() as usize).min(cells);
        if to > filled {
            out.push(Span::styled("█".repeat(to - filled), style));
            filled = to;
        }
    }
    if filled < cells {
        out.push(colored(theme, theme.rule, "░".repeat(cells - filled)));
    }
    out
}

/// Returns `values` as a sparkline of at most `columns` columns. Each column
/// shows the mean of its values in eighths of a cell. The scale starts at 0,
/// or at the least mean when `from_zero` is false. A column without a value
/// stays blank.
pub fn build_sparkline(values: &[Option<f64>], columns: usize, from_zero: bool) -> String {
    if values.is_empty() || columns == 0 {
        return String::new();
    }
    let per_column = values.len().div_ceil(columns.min(values.len()));
    let means: Vec<Option<f64>> = values
        .chunks(per_column)
        .map(|chunk| {
            let known: Vec<f64> = chunk.iter().flatten().copied().collect();
            (!known.is_empty()).then(|| known.iter().sum::<f64>() / known.len() as f64)
        })
        .collect();
    let top = means.iter().flatten().copied().fold(0.0, f64::max);
    let low = if from_zero {
        0.0
    } else {
        means.iter().flatten().copied().fold(top, f64::min)
    };
    means
        .iter()
        .map(|mean| match mean {
            Some(value) if top > low => {
                let level = (SPARKLINE_LEVELS * (value - low) / (top - low)) as usize;
                LOWER_EIGHTHS[1 + level.min(EIGHTHS - 1)]
            }
            // All the same: at the base from 0, else in the middle.
            Some(_) if from_zero => LOWER_EIGHTHS[1],
            Some(_) => LOWER_EIGHTHS[EIGHTHS / 2],
            None => " ",
        })
        .collect()
}

/// Draws `parts` stacked along `cells` cells of `buf`. The first cell is at
/// `start`, and each next cell is one `step` further. Each part has its length
/// in cells and its color. The parts fill the cells by eighths. A cell where a
/// part ends shows that part over the next part's color.
pub fn draw_stacked_parts(
    buf: &mut Buffer,
    theme: &Theme,
    start: (u16, u16),
    step: (i32, i32),
    cells: u16,
    eighths: &[&str; 9],
    parts: &[(f64, Rgb)],
) {
    let eighth = EIGHTHS as i64;
    // Each part's end, in eighths from the first cell, and its color.
    let mut ends: Vec<(i64, Rgb)> = Vec::new();
    let mut sum = 0.0;
    for &(length, rgb) in parts {
        if length > 0.0 {
            sum += length * EIGHTHS as f64;
            ends.push(((sum.round() as i64).min(cells as i64 * eighth), rgb));
        }
    }
    for k in 0..cells as i64 {
        let Some(first) = ends.iter().position(|&(end, _)| end > k * eighth) else {
            break;
        };
        let (end, rgb) = ends[first];
        let x = start.0 as i64 + step.0 as i64 * k;
        let y = start.1 as i64 + step.1 as i64 * k;
        let cell = &mut buf[(x as u16, y as u16)];
        cell.set_symbol(eighths[(end - k * eighth).min(eighth) as usize])
            .set_fg(theme.color(rgb));
        if end < (k + 1) * eighth
            && let Some(&(_, next)) = ends.get(first + 1)
        {
            cell.set_bg(theme.color(next));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::theme::DARK;
    use ratatui::layout::Rect;

    /// The text of `line`.
    fn text_of(line: &Line) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn figures_show_their_units() {
        assert_eq!(format_count(1234567.0), "1,234,567");
        assert_eq!(format_count(999.0), "999");
        assert_eq!(format_count(-5.0), "0");
        assert_eq!(format_duration(59.0), "59s");
        assert_eq!(format_duration(61.0), "1m01s");
        assert_eq!(format_duration(3725.0), "1h02m");
        assert_eq!(format_percent(1.0, 8.0), "12.5%");
        assert_eq!(format_percent(1.0, 0.0), "0.0%");
    }

    #[test]
    fn a_long_line_ends_after_a_whole_figure() {
        let line = Line::from(vec![
            label(&DARK, "sessions"),
            plain(" 4   1.23 / 8.0 GiB   "),
            label(&DARK, "reused"),
            plain(" 66.7%   new 4"),
        ]);
        let cut = fit_line(&DARK, line.clone(), 30);
        assert!(cut.width() <= 30, "{}", text_of(&cut));
        assert_eq!(text_of(&cut), "sessions 4   1.23 / 8.0 GiB …");
        assert_eq!(text_of(&fit_line(&DARK, line.clone(), 80)), text_of(&line));
        // without a figure, the last whole word
        let words = Line::from(plain("one two three four"));
        assert_eq!(text_of(&fit_line(&DARK, words, 12)), "one two …");
        assert_eq!(clip_text("deepseek-v4.1-flash", 8), "deepsee…");
        assert_eq!(clip_text("qwen", 8), "qwen");
    }

    #[test]
    fn rules_gauges_and_sparklines_take_their_columns() {
        let rule = build_section_rule(&DARK, 20, "Log");
        assert_eq!(text_of(&rule), "─ Log ──────────────");
        assert_eq!(rule.spans[1].style.fg, Some(DARK.color(DARK.accent)));
        let gauge = build_gauge(&DARK, 10, &[(0.25, Style::new()), (0.25, Style::new())]);
        assert_eq!(measure_width(&gauge), 10);
        assert_eq!(text_of(&Line::from(gauge)), "█████░░░░░");
        assert_eq!(
            build_sparkline(&[Some(0.0), Some(1.0), Some(2.0)], 3, true),
            "▁▄█"
        );
        assert_eq!(
            build_sparkline(&[Some(0.0), None, Some(2.0)], 3, true),
            "▁ █"
        );
        // all zero: at the base
        assert_eq!(build_sparkline(&[Some(0.0); 3], 3, true), "▁▁▁");
        // six values in three columns: the mean of each two
        assert_eq!(
            build_sparkline(
                &[Some(0.0), Some(0.0), Some(1.0), Some(1.0), None, Some(2.0)],
                3,
                true
            ),
            "▁▄█"
        );
        let padded = pad_spans(vec![plain("ab")], 5);
        assert_eq!(measure_width(&padded), 5);
    }

    #[test]
    fn stacked_parts_fill_cells_by_eighths() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 4));
        // a column from the bottom: 1.5 cells of one color, then 1 of another
        let (first, second) = (DARK.sources[0], DARK.sources[4]);
        draw_stacked_parts(
            &mut buf,
            &DARK,
            (0, 3),
            (0, -1),
            4,
            &LOWER_EIGHTHS,
            &[(1.5, first), (1.0, second)],
        );
        let symbols: Vec<&str> = (0..4).map(|y| buf[(0, y)].symbol()).collect();
        assert_eq!(symbols, [" ", "▄", "▄", "█"]);
        assert_eq!(buf[(0, 3)].fg, DARK.color(first));
        // the cell where the first part ends shows the second behind it
        assert_eq!(buf[(0, 2)].fg, DARK.color(first));
        assert_eq!(buf[(0, 2)].bg, DARK.color(second));
        assert_eq!(buf[(0, 1)].fg, DARK.color(second));
    }
}
