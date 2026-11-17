//! The terminal UI's colors, for a light and for a dark terminal. A terminal
//! without 24-bit color shows the nearest of its 256 standard colors.

use ratatui::style::Color;

/// A color: its red, green and blue, each from 0 to 255.
pub type Rgb = (u8, u8, u8);

/// The terminal UI's colors for one kind of background.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theme {
    /// The terminal's background. Tints blend toward it.
    pub ground: Rgb,
    /// Names that stand out, such as a session's name in the states tree.
    pub ink: Rgb,
    /// Labels beside figures.
    pub label: Rgb,
    /// Rules, frames and the empty part of a gauge.
    pub rule: Rgb,
    /// Free memory in the memory map.
    pub faint: Rgb,
    /// The states tree's lines of the session idle for the shortest time.
    pub idle_recent: Rgb,
    /// The states tree's lines of the session idle for the longest time.
    pub idle_oldest: Rgb,
    /// The chosen tabs, and what decodes.
    pub accent: Rgb,
    /// Text on the accent.
    pub on_accent: Rgb,
    /// Prompts that prefill.
    pub prefill: Rgb,
    /// Where an expert was when a token needed it, in the engine's order: in
    /// memory, read ahead, late, substituted, and from the SSD.
    pub sources: [Rgb; 5],
    /// Each model's own color, by the model's place in the configured order.
    /// The colors repeat after the last one.
    pub models: [Rgb; 4],
    /// Green: normal memory pressure, and the reserve in the memory map.
    pub green: Rgb,
    /// Amber: a model that loads, a request that waits, and memory pressure
    /// that warns.
    pub amber: Rgb,
    /// Red: a model whose load failed, and critical memory pressure.
    pub red: Rgb,
    /// Brown: the server's other memory in the memory map.
    pub brown: Rgb,
    /// Whether the terminal shows 24-bit color. Without it, each color shows
    /// as the nearest of the terminal's 256 standard colors.
    pub true_color: bool,
}

/// The colors for a light background.
pub const LIGHT: Theme = Theme {
    ground: (0xFF, 0xFF, 0xFF),
    ink: (0x1D, 0x1D, 0x1F),
    label: (0x6E, 0x6E, 0x73),
    rule: (0xC7, 0xC7, 0xCC),
    faint: (0xEC, 0xEC, 0xF1),
    idle_recent: (0x6E, 0x6E, 0x73),
    idle_oldest: (0xBC, 0xBC, 0xC2),
    accent: (0x58, 0x56, 0xD6),
    on_accent: (0xFF, 0xFF, 0xFF),
    prefill: (0x2F, 0xA7, 0xB8),
    sources: [
        (0x2F, 0x6F, 0xED),
        (0x8C, 0xC2, 0xFF),
        (0xF2, 0xA9, 0x3B),
        (0xA5, 0x7E, 0xF2),
        (0xEE, 0x5A, 0x52),
    ],
    models: [
        (0x3B, 0xA5, 0x5C),
        (0xD6, 0x45, 0x7A),
        (0x9C, 0x7A, 0x4F),
        (0x8E, 0x8E, 0x93),
    ],
    green: (0x3B, 0xA5, 0x5C),
    amber: (0xF2, 0xA9, 0x3B),
    red: (0xEE, 0x5A, 0x52),
    brown: (0x9C, 0x7A, 0x4F),
    true_color: true,
};

/// The colors for a dark background.
pub const DARK: Theme = Theme {
    ground: (0x1E, 0x1E, 0x1E),
    ink: (0xF2, 0xF2, 0xF7),
    label: (0x98, 0x98, 0x9F),
    rule: (0x48, 0x48, 0x4A),
    faint: (0x2C, 0x2C, 0x2E),
    idle_recent: (0xAE, 0xAE, 0xB2),
    idle_oldest: (0x58, 0x58, 0x5C),
    accent: (0x7D, 0x7A, 0xFF),
    on_accent: (0xFF, 0xFF, 0xFF),
    prefill: (0x40, 0xC8, 0xE0),
    sources: [
        (0x40, 0x8C, 0xFF),
        (0x9E, 0xCB, 0xFF),
        (0xFF, 0xB3, 0x40),
        (0xBF, 0x8C, 0xFF),
        (0xFF, 0x6B, 0x60),
    ],
    models: [
        (0x30, 0xD1, 0x58),
        (0xFF, 0x37, 0x5F),
        (0xAC, 0x8E, 0x68),
        (0x98, 0x98, 0x9D),
    ],
    green: (0x30, 0xD1, 0x58),
    amber: (0xFF, 0xB3, 0x40),
    red: (0xFF, 0x6B, 0x60),
    brown: (0xAC, 0x8E, 0x68),
    true_color: true,
};

/// The weights of red, green and blue in a color's brightness: the luma of
/// ITU-R BT.601.
const LUMA_WEIGHTS: [f64; 3] = [0.299, 0.587, 0.114];
/// A background at least this bright, on a scale from 0 to 1, takes the light
/// colors.
const LIGHT_BACKGROUND_BRIGHTNESS: f64 = 0.5;

/// The levels of each channel in the 6 x 6 x 6 color cube of the 256
/// standard colors.
const CUBE_LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
/// The index of the cube's first color among the 256 standard colors.
const CUBE_FIRST_INDEX: u8 = 16;
/// The index of the first of the 24 standard grays.
const GRAY_FIRST_INDEX: u8 = 232;
/// The level of each channel in the first standard gray.
const GRAY_FIRST_LEVEL: i32 = 8;
/// The step of the level from one standard gray to the next.
const GRAY_STEP: i32 = 10;
/// The number of standard grays.
const GRAY_COUNT: i32 = 24;

/// What starts a terminal's answer to the OSC 11 query, after its ESC.
const BACKGROUND_ANSWER_START: &str = "]11;rgb:";
/// The most hexadecimal digits of one channel in that answer.
const MOST_CHANNEL_DIGITS: usize = 4;
/// The bits of one hexadecimal digit.
const BITS_PER_DIGIT: usize = 4;

impl Theme {
    /// Returns the light colors for a light `background`, and the dark colors
    /// for a dark one. Their tints blend toward `background`.
    pub fn for_background(background: Rgb) -> Theme {
        let channels = [background.0, background.1, background.2];
        let brightness: f64 = channels
            .iter()
            .zip(LUMA_WEIGHTS)
            .map(|(&channel, weight)| channel as f64 * weight)
            .sum::<f64>()
            / u8::MAX as f64;
        let theme = if brightness >= LIGHT_BACKGROUND_BRIGHTNESS {
            LIGHT
        } else {
            DARK
        };
        Theme {
            ground: background,
            ..theme
        }
    }

    /// Returns the terminal color that shows `rgb`.
    pub fn color(&self, rgb: Rgb) -> Color {
        if self.true_color {
            Color::Rgb(rgb.0, rgb.1, rgb.2)
        } else {
            Color::Indexed(find_nearest_standard_color(rgb))
        }
    }

    /// Returns the color of model `model_index`, by its place in the
    /// configured order.
    pub fn model_color(&self, model_index: usize) -> Rgb {
        self.models[model_index % self.models.len()]
    }
}

/// Returns `from` blended toward `to` by `share`: 0 gives `from`, and 1 gives
/// `to`.
pub fn mix(from: Rgb, to: Rgb, share: f64) -> Rgb {
    let share = share.clamp(0.0, 1.0);
    let blend = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * share).round() as u8;
    (
        blend(from.0, to.0),
        blend(from.1, to.1),
        blend(from.2, to.2),
    )
}

/// Returns the index of the standard color nearest to `rgb`: a color of the
/// cube, or a gray.
fn find_nearest_standard_color(rgb: Rgb) -> u8 {
    let nearest_level = |value: u8| {
        (0..CUBE_LEVELS.len())
            .min_by_key(|&i| (CUBE_LEVELS[i] as i32 - value as i32).abs())
            .unwrap_or(0)
    };
    let distance = |a: Rgb, b: Rgb| {
        let square = |p: u8, q: u8| (p as i32 - q as i32).pow(2);
        square(a.0, b.0) + square(a.1, b.1) + square(a.2, b.2)
    };
    let (red, green, blue) = (
        nearest_level(rgb.0),
        nearest_level(rgb.1),
        nearest_level(rgb.2),
    );
    let cube = (CUBE_LEVELS[red], CUBE_LEVELS[green], CUBE_LEVELS[blue]);
    let mean = (rgb.0 as i32 + rgb.1 as i32 + rgb.2 as i32) / 3;
    let gray = ((mean - GRAY_FIRST_LEVEL).max(0) / GRAY_STEP).min(GRAY_COUNT - 1);
    let level = (GRAY_FIRST_LEVEL + GRAY_STEP * gray) as u8;
    if distance(rgb, (level, level, level)) < distance(rgb, cube) {
        GRAY_FIRST_INDEX + gray as u8
    } else {
        let side = CUBE_LEVELS.len();
        CUBE_FIRST_INDEX + (side * side * red + side * green + blue) as u8
    }
}

/// Returns the background color in a terminal's answer to the OSC 11 query.
/// The answer is "ESC ] 11 ; rgb:R/G/B", ended by BEL or by ESC \. Each
/// channel has 1 to 4 hexadecimal digits. Returns None until a whole answer
/// is there, and for an answer of another form.
pub fn parse_background_answer(answer: &[u8]) -> Option<Rgb> {
    let text = String::from_utf8_lossy(answer);
    let start = text.find(BACKGROUND_ANSWER_START)? + BACKGROUND_ANSWER_START.len();
    let rest = &text[start..];
    let end = rest.find(['\x07', '\x1b'])?;
    let mut channels = rest[..end].split('/');
    let mut read_channel = || -> Option<u8> {
        let digits = channels.next()?;
        if digits.is_empty() || digits.len() > MOST_CHANNEL_DIGITS {
            return None;
        }
        let value = u32::from_str_radix(digits, 16).ok()?;
        let most = (1u32 << (BITS_PER_DIGIT * digits.len())) - 1;
        Some(((value * u8::MAX as u32 + most / 2) / most) as u8)
    };
    let rgb = (read_channel()?, read_channel()?, read_channel()?);
    channels.next().is_none().then_some(rgb)
}

/// Returns whether a terminal shows 24-bit color, from the value of its
/// COLORTERM variable.
pub fn is_true_color(colorterm: Option<&str>) -> bool {
    matches!(colorterm, Some("truecolor" | "24bit"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_terminals_answer_gives_its_background() {
        assert_eq!(
            parse_background_answer(b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\"),
            Some((255, 255, 255))
        );
        assert_eq!(
            parse_background_answer(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07"),
            Some((0x1E, 0x1E, 0x1E))
        );
        assert_eq!(
            parse_background_answer(b"x\x1b]11;rgb:0/8/f\x07"),
            Some((0, 136, 255))
        );
        // not whole yet, or not a color
        assert_eq!(parse_background_answer(b"\x1b]11;rgb:ffff/ff"), None);
        assert_eq!(parse_background_answer(b"\x1b]11;rgb:ffff/ffff\x07"), None);
        assert_eq!(parse_background_answer(b"\x1b]11;rgb:gg/00/00\x07"), None);
        assert_eq!(
            parse_background_answer(b"\x1b]11;rgb:1/2/3/4\x07"),
            None,
            "a fourth channel"
        );
    }

    #[test]
    fn a_light_background_takes_the_light_colors_and_a_dark_one_the_dark() {
        let light = Theme::for_background((0xFA, 0xF9, 0xF5));
        assert_eq!(
            light,
            Theme {
                ground: (0xFA, 0xF9, 0xF5),
                ..LIGHT
            }
        );
        let dark = Theme::for_background((0x10, 0x12, 0x18));
        assert_eq!(
            dark,
            Theme {
                ground: (0x10, 0x12, 0x18),
                ..DARK
            }
        );
        // tints blend toward the background
        assert_eq!(mix((0, 0, 0), (200, 100, 50), 0.5), (100, 50, 25));
        assert_eq!(mix((0, 0, 0), (200, 100, 50), 2.0), (200, 100, 50));
    }

    #[test]
    fn without_24_bit_color_a_color_is_the_nearest_standard_one() {
        assert_eq!(find_nearest_standard_color((0, 0, 0)), 16);
        assert_eq!(find_nearest_standard_color((255, 255, 255)), 231);
        // the SSD's red becomes (255, 95, 95) of the cube
        assert_eq!(
            find_nearest_standard_color((0xEE, 0x5A, 0x52)),
            16 + 36 * 5 + 6 + 1
        );
        // a gray
        assert_eq!(find_nearest_standard_color((0x80, 0x80, 0x80)), 244);
        let indexed = Theme {
            true_color: false,
            ..DARK
        };
        assert_eq!(indexed.color((0x80, 0x80, 0x80)), Color::Indexed(244));
        assert_eq!(DARK.color((1, 2, 3)), Color::Rgb(1, 2, 3));
        assert!(is_true_color(Some("truecolor")) && is_true_color(Some("24bit")));
        assert!(!is_true_color(Some("256")) && !is_true_color(None));
    }

    #[test]
    fn each_model_has_its_own_color_and_the_colors_repeat() {
        assert_eq!(DARK.model_color(0), (0x30, 0xD1, 0x58));
        assert_eq!(DARK.model_color(1), (0xFF, 0x37, 0x5F));
        assert_eq!(DARK.model_color(4), DARK.model_color(0));
    }
}
