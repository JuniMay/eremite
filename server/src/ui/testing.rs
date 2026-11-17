//! What the terminal UI's tests share: drawing on a test terminal, and the
//! text of the cells drawn.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use super::samples::Windows;

/// The windows' defaults: 5 s now, and 60 s recent and history.
pub const WINDOWS: Windows = Windows {
    now: 5.0,
    recent: 60.0,
    history: 60.0,
};

/// Draws with `draw` on a test terminal of `width` x `height` cells, and
/// returns its cells.
pub fn draw_on_terminal(width: u16, height: u16, draw: impl FnOnce(&mut Buffer, Rect)) -> Buffer {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("a test terminal");
    terminal
        .draw(|frame| {
            let area = frame.area();
            draw(frame.buffer_mut(), area);
        })
        .expect("a frame");
    terminal.backend().buffer().clone()
}

/// Returns the text of row `y` of `buf`.
pub fn read_row(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width)
        .map(|x| buf[(x, y)].symbol().to_string())
        .collect()
}

/// Returns the text of every row of `buf`, one line each.
pub fn read_screen(buf: &Buffer) -> String {
    (0..buf.area.height)
        .map(|y| read_row(buf, y))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns the column where `text` starts in row `y` of `buf`, or None.
pub fn find_in_row(buf: &Buffer, y: u16, text: &str) -> Option<u16> {
    let wanted: Vec<char> = text.chars().collect();
    let cells: Vec<char> = (0..buf.area.width)
        .map(|x| buf[(x, y)].symbol().chars().next().unwrap_or(' '))
        .collect();
    cells
        .windows(wanted.len())
        .position(|window| window == wanted.as_slice())
        .map(|x| x as u16)
}
