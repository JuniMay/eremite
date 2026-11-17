//! The view 2 Memory: the Mac's installed memory as a map of tiles. Each
//! tile's area follows its bytes. Each model is a tile under its name, with
//! its weights, experts, workspace and states inside it. Beside the models
//! are the server's other memory, other programs, the file cache, the
//! reserve and the free memory.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};

use super::draw::{colored, fit_line, label, plain, strong};
use super::frame::{Choices, Screen};
use super::theme::mix;
use super::tiles::{TILE_TINT, Tile, draw_tile, lay_out_tiles};

/// The bytes of one GiB.
const GIB: f64 = (1u64 << 30) as f64;
/// The rows above the map: the line of figures and an empty row.
const ROWS_ABOVE_MAP: u16 = 2;
/// The rows under the map: the colors, and the tiles too small for a label.
const ROWS_UNDER_MAP: u16 = 2;
/// The fewest columns and rows in which the view draws the map.
const LEAST_MAP_AREA: (u16, u16) = (20, 8);

/// Draws 2 Memory in `area`: the line of the figures that the map does not
/// show, the map, the colors, and the tiles too small for a label. The model
/// chosen in 4 Models, by `choices`, has its name on the accent.
pub fn draw_memory(buf: &mut Buffer, area: Rect, screen: &Screen, choices: &Choices) {
    let theme = screen.theme;
    let Some(first) = screen.statuses.first() else {
        return;
    };
    let memory = &first.engine_counters.memory;
    let width = area.width as usize;
    let gib = |bytes: u64| format!("{:.1} GiB", bytes as f64 / GIB);
    let mut figures = vec![
        plain(" "),
        strong("Mac"),
        plain(format!("  {:.0} GiB installed", memory.total as f64 / GIB)),
        label(theme, "   total limit now "),
        plain(gib(first.model_memory_budget.limit)),
    ];
    if first.memory_gib > 0.0 {
        figures.push(label(theme, format!(" (setting {} GiB)", first.memory_gib)));
    }
    // The chosen model's state room.
    if let (Some(model), Some(status)) = (
        screen.models.get(choices.model),
        screen.statuses.get(choices.model),
    ) {
        figures.extend([
            label(theme, "   "),
            plain(model.name.clone()),
            label(theme, "'s states up to "),
            plain(format!("{:.1} GiB", status.state_cache_limit_gib)),
        ]);
    }
    figures.extend([
        label(theme, "   wired "),
        plain(gib(memory.wired)),
        label(theme, "   swap "),
        plain(gib(memory.swap_used)),
    ]);
    if area.height == 0 {
        return;
    }
    buf.set_line(
        area.x,
        area.y,
        &fit_line(theme, Line::from(figures), width),
        area.width,
    );
    if area.width < LEAST_MAP_AREA.0 || area.height < LEAST_MAP_AREA.1 {
        return;
    }
    // Every part has its own color, so that parts side by side stay apart.
    let kinds = [
        ("weights", theme.sources[2]),
        ("experts", theme.sources[0]),
        ("workspace", theme.sources[3]),
        ("states", theme.prefill),
    ];
    let mut tiles = Vec::new();
    let mut tracked = 0u64;
    for (i, model) in first.model_memory.iter().enumerate() {
        let bytes = [model.weights, model.experts, model.workspace, model.states];
        let held: u64 = bytes.iter().sum();
        if held == 0 {
            continue;
        }
        tracked += held;
        let parts = kinds
            .iter()
            .zip(bytes)
            .filter(|&(_, part_bytes)| part_bytes > 0)
            .map(|(&(name, tone), part_bytes)| Tile::part(name, part_bytes, tone))
            .collect();
        tiles.push(Tile {
            name: model.model.clone(),
            bytes: held,
            tone: theme.ink,
            parts,
            title_bytes: held - model.states,
            chosen: i == choices.model,
        });
    }
    // In use without the file cache: this server, other programs and the
    // system. Of the free memory, the reserve's share shows apart.
    let in_use = memory
        .total
        .saturating_sub(memory.free.saturating_add(memory.file));
    let others = in_use.saturating_sub(memory.footprint.max(tracked));
    let reserve = memory.free.min((first.reserve_gib * GIB) as u64);
    let rest = [
        (
            "other server memory",
            memory.footprint.saturating_sub(tracked),
            theme.brown,
        ),
        ("other programs", others, theme.label),
        ("file cache", memory.file, theme.rule),
        ("reserve", reserve, theme.green),
        ("free", memory.free - reserve, theme.faint),
    ];
    for (name, bytes, tone) in rest {
        tiles.push(Tile::part(name, bytes, tone));
    }
    let map = Rect {
        y: area.y + ROWS_ABOVE_MAP,
        height: area.height - ROWS_ABOVE_MAP - ROWS_UNDER_MAP,
        ..area
    };
    let sizes: Vec<u64> = tiles.iter().map(|tile| tile.bytes).collect();
    let mut unlabeled = Vec::new();
    for (tile, rect) in tiles.iter().zip(lay_out_tiles(&sizes, map)) {
        if tile.bytes > 0 {
            draw_tile(buf, theme, rect, tile, "", &mut unlabeled);
        }
    }
    // What each color is. Free memory, nearly the background, is a hollow
    // square.
    let mut key = vec![plain(" ")];
    let names = kinds
        .into_iter()
        .chain(rest.iter().map(|&(name, _, tone)| (name, tone)));
    for (name, tone) in names {
        key.push(if name == "free" {
            label(theme, "□ ")
        } else {
            colored(theme, mix(theme.ground, tone, TILE_TINT), "■ ")
        });
        key.push(label(theme, format!("{name}   ")));
    }
    let key_row = area.bottom() - ROWS_UNDER_MAP;
    buf.set_line(
        area.x,
        key_row,
        &fit_line(theme, Line::from(key), width),
        area.width,
    );
    // What has no room for its size, after the model that it belongs to.
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for (owner, what) in unlabeled {
        match groups.last_mut() {
            Some((last, list)) if *last == owner => list.push(what),
            _ => groups.push((owner, vec![what])),
        }
    }
    if !groups.is_empty() {
        let parts: Vec<String> = groups
            .into_iter()
            .map(|(owner, list)| match owner.as_str() {
                "" => list.join(", "),
                _ => format!("{owner}: {}", list.join(", ")),
            })
            .collect();
        let note = Span::styled(
            format!(" too small to label: {}", parts.join("; ")),
            Style::new().fg(theme.color(theme.label)),
        );
        buf.set_line(
            area.x,
            key_row + 1,
            &fit_line(theme, Line::from(note), width),
            area.width,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::frame::{Kind, ModelInfo};
    use crate::ui::samples::Samples;
    use crate::ui::status::ModelStatus;
    use crate::ui::testing::{WINDOWS, draw_on_terminal, find_in_row, read_row, read_screen};
    use crate::ui::theme::DARK;
    use crate::ui::tiles::TITLE_TINT;

    /// The bytes of `gib` GiB.
    fn bytes(gib: f64) -> u64 {
        (gib * GIB) as u64
    }

    /// The status of two models in a Mac of 128 GiB, as the worker builds it:
    /// deepseek-v4.1-flash with its experts, and qwen3.8-27b without them.
    fn status() -> ModelStatus {
        let json = format!(
            r#"{{
            "state_cache_limit_gib": 8.0, "reserve_gib": 10.0, "memory_gib": 80.0,
            "model_memory": [
                {{"model": "deepseek-v4.1-flash", "weights": {}, "experts": {},
                  "workspace": {}, "states": {}}},
                {{"model": "qwen3.8-27b", "weights": {}, "experts": 0,
                  "workspace": {}, "states": {}}}],
            "model_memory_budget": {{"limit": {}}},
            "engine_counters": {{"memory": {{"total": {}, "free": {}, "file": {},
                "footprint": {}, "wired": {}, "swap_used": 0}}}}
            }}"#,
            bytes(7.2),
            bytes(44.1),
            bytes(0.06),
            bytes(0.33),
            bytes(15.2),
            bytes(0.4),
            bytes(1.4),
            bytes(68.0),
            bytes(128.0),
            bytes(11.1),
            bytes(7.3),
            bytes(69.2),
            bytes(73.7),
        );
        serde_json::from_str(&json).expect("the worker's form")
    }

    /// Draws 2 Memory at `width` x `height` with qwen3.8-27b chosen.
    fn draw(width: u16, height: u16) -> Buffer {
        let models = ["deepseek-v4.1-flash", "qwen3.8-27b"].map(|name| ModelInfo {
            name: name.into(),
            kind: Kind::Llm,
        });
        let statuses = [status(), status()];
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
        let choices = Choices {
            model: 1,
            ..Choices::default()
        };
        draw_on_terminal(width, height, |buf, area| {
            draw_memory(buf, area, &screen, &choices)
        })
    }

    #[test]
    fn the_map_shows_every_model_with_its_parts_and_the_rest() {
        let buf = draw(160, 41);
        assert_eq!(
            read_row(&buf, 0).trim_end(),
            " Mac  128 GiB installed   total limit now 68.0 GiB (setting 80 GiB)   qwen3.8-27b's \
             states up to 8.0 GiB   wired 73.7 GiB   swap 0.0 GiB"
        );
        let screen = read_screen(&buf);
        // each model under its name and its memory without its states
        for name in [
            "deepseek-v4.1-flash 51.4 GiB",
            "qwen3.8-27b 15.6 GiB",
            "experts 44.1 GiB",
            "weights 15.2 GiB",
            "other programs 40.4 GiB",
            "file cache 7.3 GiB",
            "reserve 10.0 GiB",
        ] {
            assert!(screen.contains(name), "{name}:\n{screen}");
        }
        // the chosen model's title on the accent, the other's on a gray
        let (x, y) = find_tile(&buf, "qwen3.8-27b");
        assert_eq!(buf[(x, y)].bg, DARK.color(DARK.accent));
        assert_eq!(buf[(x, y)].fg, DARK.color(DARK.on_accent));
        let (x, y) = find_tile(&buf, "deepseek-v4.1-flash");
        assert_eq!(
            buf[(x, y)].bg,
            DARK.color(mix(DARK.ground, DARK.ink, TITLE_TINT))
        );
        let (x, y) = find_tile(&buf, "experts 44.1");
        assert_eq!(
            buf[(x, y)].bg,
            DARK.color(mix(DARK.ground, DARK.sources[0], TILE_TINT))
        );
        let key = read_row(&buf, 39);
        assert!(
            key.starts_with(
                " ■ weights   ■ experts   ■ workspace   ■ states   ■ other server memory   \
                 ■ other programs   ■ file cache   ■ reserve   □ free"
            ),
            "{key}"
        );
        let small = read_row(&buf, 40);
        assert!(
            small.starts_with(" too small to label: deepseek-v4.1-flash: states 0.33 GiB"),
            "{small}"
        );
    }

    /// Returns the cell where `text` starts, in the map's rows of `buf`.
    fn find_tile(buf: &Buffer, text: &str) -> (u16, u16) {
        (ROWS_ABOVE_MAP..buf.area.height)
            .find_map(|y| find_in_row(buf, y, text).map(|x| (x, y)))
            .unwrap_or_else(|| panic!("no {text}:\n{}", read_screen(buf)))
    }

    #[test]
    fn the_map_fits_narrow_and_tiny_terminals() {
        let screen = read_screen(&draw(100, 26));
        assert!(screen.contains("qwen3.8-27b 15.6 GiB"), "{screen}");
        assert!(read_screen(&draw(19, 6)).starts_with(" Mac "));
        draw(1, 1);
    }
}
