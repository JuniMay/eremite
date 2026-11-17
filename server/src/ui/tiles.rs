//! The layout of the memory map: rectangles that tile an area, each with cells
//! in proportion to its size. It is the squarified layout of Bruls, Huizing
//! and van Wijk. Each row of rectangles goes along the shorter side of the
//! room left. A rectangle joins the row while it makes the row's worst aspect
//! ratio no worse.

use ratatui::layout::Rect;

/// A cell is about twice as tall as it is wide. The layout counts a cell's
/// height as this many widths, so that a square looks square.
const CELL_HEIGHT_IN_WIDTHS: f64 = 2.0;

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
}
