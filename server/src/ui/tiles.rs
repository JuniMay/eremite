//! The layout of the memory map: rectangles that tile an area, each with cells
//! in proportion to its size. It is the squarified layout of Bruls, Huizing
//! and van Wijk. Each row of rectangles goes along the shorter side of the
//! room left. A rectangle joins the row while it makes the row's worst aspect
//! ratio no worse. The module also draws the tiles. A model's tile holds its
//! parts under its title.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use unicode_width::UnicodeWidthStr;

use super::draw::clip_text;
use super::theme::{Rgb, Theme, mix};

/// A cell is about twice as tall as it is wide. The layout counts a cell's
/// height as this many widths, so that a square looks square.
const CELL_HEIGHT_IN_WIDTHS: f64 = 2.0;
/// The bytes of one GiB.
const GIB: f64 = (1u64 << 30) as f64;
/// A tile's color blends halfway to the background, so that its text stays
/// readable.
pub const TILE_TINT: f64 = 0.5;
/// A model's title row blends its text color this far into the background.
pub const TITLE_TINT: f64 = 0.18;

/// Lays out one rectangle for each of `sizes`, in their order, that together
/// tile `area`. Each rectangle's cells follow its size. A size of 0, and one
/// too small for a cell, gets an empty rectangle.
pub fn lay_out_tiles(sizes: &[u64], area: Rect) -> Vec<Rect> {
    let mut out = vec![Rect::new(area.x, area.y, 0, 0); sizes.len()];
    let total: f64 = sizes.iter().map(|&size| size as f64).sum();
    if total <= 0.0 || area.width == 0 || area.height == 0 {
        return out;
    }
    let mut order: Vec<usize> = (0..sizes.len()).filter(|&i| sizes[i] > 0).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));
    // The room left, in units where a cell is 1 wide.
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let (mut width, mut height) = (
        area.width as f64,
        CELL_HEIGHT_IN_WIDTHS * area.height as f64,
    );
    let scale = width * height / total;
    // A row's worst ratio of length to thickness, along a side of the room.
    let worst = |row: &[usize], sum: f64, side: f64| {
        let thickness = sum / side;
        row.iter()
            .map(|&i| {
                let length = sizes[i] as f64 * scale / thickness;
                (thickness / length).max(length / thickness)
            })
            .fold(0.0, f64::max)
    };
    let mut i = 0;
    while i < order.len() {
        let side = width.min(height);
        let (mut j, mut sum, mut best) = (i, 0.0, f64::INFINITY);
        while j < order.len() {
            let more = sum + sizes[order[j]] as f64 * scale;
            let ratio = worst(&order[i..=j], more, side);
            if ratio > best {
                break;
            }
            (best, sum, j) = (ratio, more, j + 1);
        }
        let thickness = sum / side;
        let mut along = 0.0;
        for &k in &order[i..j] {
            let length = sizes[k] as f64 * scale / thickness;
            out[k] = if width >= height {
                round_to_cells(area, (x, y + along), (thickness, length))
            } else {
                round_to_cells(area, (x + along, y), (length, thickness))
            };
            along += length;
        }
        if width >= height {
            x += thickness;
            width -= thickness;
        } else {
            y += thickness;
            height -= thickness;
        }
        i = j;
    }
    out
}

/// Returns the cells of `area` that a rectangle covers. The rectangle starts
/// at `start` and has `size`, in units where a cell is 1 wide. Its edges round
/// to the nearest edge of a cell, so rectangles that share an edge share it in
/// cells.
fn round_to_cells(area: Rect, start: (f64, f64), size: (f64, f64)) -> Rect {
    let edge = |value: f64, most: u16| (value.round().max(0.0) as u16).min(most);
    let (x0, x1) = (
        edge(start.0, area.width),
        edge(start.0 + size.0, area.width),
    );
    let rows = |value: f64| edge(value / CELL_HEIGHT_IN_WIDTHS, area.height);
    let (y0, y1) = (rows(start.1), rows(start.1 + size.1));
    Rect::new(area.x + x0, area.y + y0, x1 - x0, y1 - y0)
}

/// A tile of the map. A model's tile holds its parts under its title.
pub struct Tile {
    /// The tile's name.
    pub name: String,
    /// Its bytes.
    pub bytes: u64,
    /// Its color.
    pub tone: Rgb,
    /// The parts of a model's tile. Empty for another tile.
    pub parts: Vec<Tile>,
    /// The bytes that a model's title names: its memory without its states.
    pub title_bytes: u64,
    /// Whether it is the model chosen in 4 Models.
    pub chosen: bool,
}

impl Tile {
    /// Returns a tile without parts.
    pub fn part(name: &str, bytes: u64, tone: Rgb) -> Tile {
        Tile {
            name: name.into(),
            bytes,
            tone,
            parts: Vec::new(),
            title_bytes: bytes,
            chosen: false,
        }
    }
}

/// Formats `bytes` in GiB: with two decimals under 1 GiB, else with one.
fn format_tile_gib(bytes: u64) -> String {
    let gib = bytes as f64 / GIB;
    if gib < 1.0 {
        format!("{gib:.2} GiB")
    } else {
        format!("{gib:.1} GiB")
    }
}

/// Fills `rect` of `buf` with `background`.
fn fill_rect(buf: &mut Buffer, theme: &Theme, rect: Rect, background: Rgb) {
    let style = Style::new()
        .fg(theme.color(theme.ink))
        .bg(theme.color(background));
    for y in rect.top()..rect.bottom() {
        for x in rect.left()..rect.right() {
            buf[(x, y)].set_symbol(" ").set_style(style);
        }
    }
}

/// Writes a tile's name and size inside `rect`: on one row when they fit,
/// else on two rows, else the size alone. Returns false when not even the
/// size fits.
fn write_tile_name(
    buf: &mut Buffer,
    theme: &Theme,
    rect: Rect,
    tile: &Tile,
    background: Rgb,
) -> bool {
    let style = Style::new()
        .fg(theme.color(theme.ink))
        .bg(theme.color(background));
    let (width, size) = (rect.width as usize, format_tile_gib(tile.bytes));
    let one_row = format!(" {} {size}", tile.name);
    if one_row.width() <= width {
        buf.set_string(rect.x, rect.y, one_row, style);
    } else if rect.height >= 2 && size.width() < width {
        let name = format!(" {}", clip_text(&tile.name, width - 1));
        buf.set_string(rect.x, rect.y, name, style);
        buf.set_string(rect.x, rect.y + 1, format!(" {size}"), style);
    } else if size.width() < width {
        buf.set_string(rect.x, rect.y, format!(" {size}"), style);
    } else {
        return false;
    }
    true
}

/// Draws `tile` in `rect`. A tile with no room for its size goes to
/// `unlabeled`, with the name of the model that it belongs to, or none.
pub fn draw_tile(
    buf: &mut Buffer,
    theme: &Theme,
    rect: Rect,
    tile: &Tile,
    owner: &str,
    unlabeled: &mut Vec<(String, String)>,
) {
    let named = || {
        let what = format!("{} {}", tile.name, format_tile_gib(tile.bytes));
        (owner.to_string(), what)
    };
    if rect.width == 0 || rect.height == 0 {
        unlabeled.push(named());
        return;
    }
    if tile.parts.is_empty() {
        let background = mix(theme.ground, tile.tone, TILE_TINT);
        fill_rect(buf, theme, rect, background);
        if !write_tile_name(buf, theme, rect, tile, background) {
            unlabeled.push(named());
        }
        return;
    }
    // A model: its name and size on its title row, its parts below. A column
    // of the background at its right keeps its parts apart from the next
    // model's. A long name gives way to the size. The parts too small for
    // their size follow on the title row when they fit there.
    let rect = Rect {
        width: if rect.width > 2 {
            rect.width - 1
        } else {
            rect.width
        },
        ..rect
    };
    let below = Rect {
        y: rect.y + 1,
        height: rect.height - 1,
        ..rect
    };
    let sizes: Vec<u64> = tile.parts.iter().map(|part| part.bytes).collect();
    let mut small = Vec::new();
    let placed = lay_out_tiles(&sizes, below);
    for (part, part_rect) in tile.parts.iter().zip(placed) {
        draw_tile(buf, theme, part_rect, part, &tile.name, &mut small);
    }
    let (fg, bg) = if tile.chosen {
        (theme.on_accent, theme.accent)
    } else {
        (theme.ink, mix(theme.ground, theme.ink, TITLE_TINT))
    };
    fill_rect(buf, theme, Rect { height: 1, ..rect }, bg);
    let size = format_tile_gib(tile.title_bytes);
    let room = (rect.width as usize).saturating_sub(size.width() + 2);
    let mut title = if room >= 4 {
        format!(" {} {size}", clip_text(&tile.name, room))
    } else {
        clip_text(&format!(" {size}"), rect.width as usize)
    };
    for (i, (part_owner, what)) in small.into_iter().enumerate() {
        let gap = if i == 0 { "   " } else { " · " };
        if title.width() + gap.width() + what.width() <= rect.width as usize {
            title = format!("{title}{gap}{what}");
        } else {
            unlabeled.push((part_owner, what));
        }
    }
    let style = Style::new()
        .fg(theme.color(fg))
        .bg(theme.color(bg))
        .add_modifier(Modifier::BOLD);
    buf.set_string(rect.x, rect.y, title, style);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts how many of `rects` cover each cell of `area`.
    fn count_covers(rects: &[Rect], area: Rect) -> Vec<u32> {
        let mut counts = vec![0u32; area.width as usize * area.height as usize];
        for rect in rects {
            for y in rect.top()..rect.bottom() {
                for x in rect.left()..rect.right() {
                    assert!(area.contains((x, y).into()), "{rect:?} in {area:?}");
                    let i = (y - area.y) as usize * area.width as usize + (x - area.x) as usize;
                    counts[i] += 1;
                }
            }
        }
        counts
    }

    #[test]
    fn the_rectangles_tile_the_area_once() {
        // A linear congruential generator: Knuth's MMIX constants.
        let mut seed = 0x5eed_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        for _ in 0..500 {
            let area = Rect::new(3, 2, 1 + (next() % 160) as u16, 1 + (next() % 50) as u16);
            let sizes: Vec<u64> = (0..1 + next() % 14)
                .map(|_| if next() % 5 == 0 { 0 } else { next() % 100_000 })
                .collect();
            let rects = lay_out_tiles(&sizes, area);
            assert_eq!(rects.len(), sizes.len());
            let counts = count_covers(&rects, area);
            if sizes.iter().all(|&size| size == 0) {
                assert!(counts.iter().all(|&count| count == 0));
            } else {
                assert!(
                    counts.iter().all(|&count| count == 1),
                    "{sizes:?} in {area:?}: {rects:?}"
                );
            }
            for (size, rect) in sizes.iter().zip(&rects) {
                assert!(*size > 0 || rect.width * rect.height == 0, "{sizes:?}");
            }
        }
    }

    #[test]
    fn the_areas_follow_the_sizes() {
        let area = Rect::new(0, 0, 120, 40);
        let sizes = [600, 250, 100, 50];
        let total: u64 = sizes.iter().sum();
        for (size, rect) in sizes.iter().zip(lay_out_tiles(&sizes, area)) {
            let wanted = *size as f64 / total as f64 * 4800.0;
            let got = (rect.width * rect.height) as f64;
            // each edge moves at most half a cell
            let slack = (rect.width + rect.height) as f64;
            assert!(
                (got - wanted).abs() <= slack,
                "{size}: {got} cells for {wanted:.0}, {rect:?}"
            );
        }
    }

    #[test]
    fn equal_sizes_in_a_square_look_square() {
        // 40 columns by 20 rows look square: four equal sizes take a quarter
        // each
        let rects = lay_out_tiles(&[5, 5, 5, 5], Rect::new(0, 0, 40, 20));
        for rect in rects {
            assert_eq!((rect.width, rect.height), (20, 10), "{rect:?}");
        }
    }

    #[test]
    fn a_models_tile_holds_its_parts_under_its_title() {
        let theme = &crate::ui::theme::DARK;
        let gib = |value: f64| (value * GIB) as u64;
        let model = Tile {
            name: "deepseek-v4.1-flash".into(),
            bytes: gib(51.35),
            tone: theme.ink,
            parts: vec![
                Tile::part("experts", gib(44.1), theme.sources[0]),
                Tile::part("weights", gib(7.2), theme.sources[2]),
                Tile::part("states", gib(0.05), theme.prefill),
            ],
            title_bytes: gib(51.3),
            chosen: true,
        };
        let area = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(area);
        let mut unlabeled = Vec::new();
        draw_tile(&mut buf, theme, area, &model, "", &mut unlabeled);
        let row = |y: u16| -> String { (0..60).map(|x| buf[(x, y)].symbol()).collect() };
        // the title names the memory without the states, on the accent
        assert!(
            row(0).starts_with(" deepseek-v4.1-flash 51.3 GiB"),
            "{}",
            row(0)
        );
        assert_eq!(buf[(1, 0)].bg, theme.color(theme.accent));
        assert!(row(1).starts_with(" experts 44.1 GiB"), "{}", row(1));
        assert_eq!(
            buf[(1, 1)].bg,
            theme.color(mix(theme.ground, theme.sources[0], TILE_TINT))
        );
        // the last column keeps the model apart from the next tile
        assert_eq!(buf[(59, 1)].bg, ratatui::style::Color::Reset);
        // the states have no room for their size: the title takes them
        assert!(unlabeled.is_empty());
        assert!(row(0).contains("states 0.05 GiB"), "{}", row(0));
        assert_eq!(format_tile_gib(gib(0.33)), "0.33 GiB");
        assert_eq!(format_tile_gib(gib(15.2)), "15.2 GiB");
    }
}
