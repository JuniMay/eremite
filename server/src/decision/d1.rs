//! LiquidAI's d1 decision models: their prompt and their answers.
//!
//! d1 reads a state, text or JSON, and questions about it. One pass of the
//! engine reads the state and up to the checkpoint's set size of questions
//! after it. Each question ends at its answer row. The answer is a softmax
//! over the logits of a few tokens there: yes or no, an option's code, or a
//! level's digit. This module renders a request as LiquidAI's code renders it
//! (prompt.py in the checkpoint), packs the questions into passes, and turns
//! the passes' logits into each question's probabilities.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::engine::Tokenizer;

/// A zero-width space. The caller's text gets one after the first character
/// of a special token's spelling, so that the text reads the same but is no
/// longer that token.
const BREAK: char = '\u{200B}';
/// What opens the user's turn.
const USER_TURN: &str = "<|im_start|>user\n";
/// What follows the state, and opens each question after the first in a pass.
const QUESTION_HEADER: &str = "QUESTION:\n";
/// What ends a question: the user's turn ends, and the assistant's opens with
/// what LFM3's chat template writes before a reply without reasoning. The
/// answer row is its last token.
const ANSWER_TURN: &str = "<|im_end|>\n<|im_start|>assistant\n<think></think>";
/// The forms of yes and of no whose tokens a yes-or-no question reads.
const YES_FORMS: [&str; 3] = ["yes", "Yes", "YES"];
const NO_FORMS: [&str; 3] = ["no", "No", "NO"];
/// The letters that are option codes, in order.
const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/// The most options that take one letter as their code.
const MOST_LETTER_CODES: usize = 26;
/// The most choice options whose temperature is a small, a middle-sized and a
/// large choice's (calibration.py's buckets); more take an inventory's.
const SMALL_CHOICE: usize = 3;
const MIDDLE_CHOICE: usize = 8;
const LARGE_CHOICE: usize = 26;
/// The most decimal places of a float in Python's shortest form before it
/// takes an exponent, and the least: x = d.ddd * 10^e is written whole for
/// -4 <= e < 16.
const PLAIN_FLOAT_EXPONENTS: std::ops::Range<i32> = -4..16;

/// What a d1 checkpoint's files set: its question layout, its option codes,
/// its special tokens and its temperatures.
pub struct Checkpoint {
    /// The most questions that one pass holds after its state (d1_set_size).
    pub set_size: usize,
    /// How option codes go on past 26 options (d1_codes): 1 gives 00, 01 and
    /// on; 2 gives AA, AB and on after A to Z.
    code_scheme: u32,
    /// The text that opens a sequence (tokenizer_config.json's bos_token).
    start: String,
    /// The special tokens' spellings, the longest first.
    specials: Vec<String>,
    /// The temperature that the logits are divided by (calibration.json's
    /// default), and the ones of some kinds of question (its temperatures).
    default_temperature: f64,
    temperatures: HashMap<String, f64>,
}

/// What a request decides about.
pub enum State {
    /// Text.
    Text(String),
    /// Any JSON value, written as Python's json.dumps writes it, indented by 2.
    Json(Value),
}

/// A question, as d1 asks it.
pub enum Question {
    /// Yes or no (d1's noul).
    YesNo { instructions: String },
    /// One of the options, each with a label and an optional description.
    Choice {
        instructions: String,
        options: Vec<ChoiceOption>,
    },
    /// One of 2 to 10 ordered levels, the lowest first.
    Score {
        instructions: String,
        levels: Vec<String>,
    },
}

/// A choice's option.
pub struct ChoiceOption {
    /// Its label: d1's name for it.
    pub label: String,
    /// What it means. Without one, the prompt shows the label.
    pub description: Option<String>,
}

/// One pass of the engine: a sequence, and the logits to read from it.
pub struct Pass {
    /// The sequence's tokens: the state's, then each question's.
    pub tokens: Vec<i32>,
    /// The tokens of the state at the sequence's start.
    pub state_tokens: usize,
    /// Where each question's tokens end.
    pub question_ends: Vec<i32>,
    /// The rows and token ids of the logits that the pass returns.
    pub rows: Vec<i32>,
    pub ids: Vec<i32>,
}

/// Where a question's answer is read: its pass, each option's logits in that
/// pass, and the temperature.
struct Readout {
    pass: usize,
    options: Vec<Vec<usize>>,
    temperature: f64,
}

/// A request's passes, and how its answers are read from their logits.
pub struct Plan {
    pub passes: Vec<Pass>,
    readouts: Vec<Readout>,
}

impl Checkpoint {
    /// Reads the d1 settings of the checkpoint in `dir`: config.json's
    /// d1_set_size and d1_codes, tokenizer_config.json's bos_token,
    /// calibration.json's temperatures, and the special tokens of
    /// `tokenizer`, which has `vocabulary` tokens.
    pub fn read(dir: &str, tokenizer: Tokenizer, vocabulary: i32) -> Result<Checkpoint, String> {
        let read = |name: &str| -> Result<Value, String> {
            let path = std::path::Path::new(dir).join(name);
            let text =
                std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            serde_json::from_slice(&text).map_err(|e| format!("{}: {e}", path.display()))
        };
        let config = read("config.json")?;
        let set_size = config["d1_set_size"]
            .as_u64()
            .filter(|&size| size >= 1)
            .ok_or("config.json: no d1_set_size")? as usize;
        let code_scheme = config["d1_codes"].as_u64().unwrap_or(1) as u32;
        let start = read("tokenizer_config.json")?["bos_token"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let calibration = read("calibration.json")?;
        let default_temperature = calibration["default"].as_f64().unwrap_or(1.0);
        let temperatures = calibration["temperatures"]
            .as_object()
            .map(|kinds| {
                kinds
                    .iter()
                    .filter_map(|(kind, t)| t.as_f64().map(|t| (kind.clone(), t)))
                    .collect()
            })
            .unwrap_or_default();
        let mut specials: Vec<String> = (0..vocabulary)
            .filter(|&id| tokenizer.is_special(id))
            .filter_map(|id| String::from_utf8(tokenizer.token_bytes(id).to_vec()).ok())
            .filter(|name| name.chars().count() > 1 && !name.contains(BREAK))
            .collect();
        specials.sort_by_key(|name| std::cmp::Reverse(name.chars().count()));
        Ok(Checkpoint {
            set_size,
            code_scheme,
            start,
            specials,
            default_temperature,
            temperatures,
        })
    }

    /// Returns `text` with a zero-width space after the first character of
    /// every special token's spelling in it, again until none is left.
    fn escape_specials(&self, text: &str) -> String {
        let mut text = text.to_string();
        loop {
            let mut out = String::with_capacity(text.len());
            let mut changed = false;
            let mut rest = text.as_str();
            while let Some(c) = rest.chars().next() {
                match self
                    .specials
                    .iter()
                    .find(|name| rest.starts_with(name.as_str()))
                {
                    Some(name) => {
                        out.push(c);
                        out.push(BREAK);
                        out.push_str(&name[c.len_utf8()..]);
                        rest = &rest[name.len()..];
                        changed = true;
                    }
                    None => {
                        out.push(c);
                        rest = &rest[c.len_utf8()..];
                    }
                }
            }
            if !changed {
                return out;
            }
            text = out;
        }
    }

    /// Returns the temperature of a question: its kind's, or the default.
    fn temperature(&self, question: &Question) -> f64 {
        let kind = match question {
            Question::YesNo { .. } => "noul".to_string(),
            Question::Score { levels, .. } => format!("score:{}", levels.len()),
            Question::Choice { options, .. } => match options.len() {
                n if n <= SMALL_CHOICE => "choice:small".into(),
                n if n <= MIDDLE_CHOICE => "choice:mid".into(),
                n if n <= LARGE_CHOICE => "choice:large".into(),
                _ => "choice:inventory".into(),
            },
        };
        self.temperatures
            .get(&kind)
            .copied()
            .unwrap_or(self.default_temperature)
    }
}

/// Returns `x` as Python's repr writes a float: its shortest digits, whole
/// with ".0" for 10^-4 <= |x| < 10^16, else with an exponent of at least two
/// digits.
pub fn format_python_float(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust writes the shortest digits that read back as x.
    let written = format!("{x:e}");
    let (mantissa, exponent) = written.split_once('e').expect("a float in {:e} has an e");
    let exponent: i32 = exponent.parse().expect("an exponent is a number");
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.trim_start_matches('-').replace('.', "");
    if PLAIN_FLOAT_EXPONENTS.contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!(
                "{sign}{digits}{}.0",
                "0".repeat(point as usize - digits.len())
            )
        } else {
            let (whole, fraction) = digits.split_at(point as usize);
            format!("{sign}{whole}.{fraction}")
        }
    } else {
        let mantissa = match digits.split_at(1) {
            (first, "") => first.to_string(),
            (first, rest) => format!("{first}.{rest}"),
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs())
    }
}

/// Appends a JSON string as Python's json writes it without ASCII escapes.
fn write_python_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Appends `value` as Python's json.dumps writes it with indent=2 and
/// ensure_ascii=False, at the given depth. A whole number beyond 64 bits is
/// written as a float, since it is read as one.
fn write_python_json(out: &mut String, value: &Value, depth: usize) {
    let indent = |out: &mut String, depth: usize| out.push_str(&"  ".repeat(depth));
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_u64(), n.as_i64(), n.as_f64()) {
            (Some(u), _, _) => out.push_str(&u.to_string()),
            (_, Some(i), _) => out.push_str(&i.to_string()),
            (_, _, Some(f)) => out.push_str(&format_python_float(f)),
            _ => out.push_str("null"),
        },
        Value::String(s) => write_python_string(out, s),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                indent(out, depth + 1);
                write_python_json(out, item, depth + 1);
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            indent(out, depth);
            out.push(']');
        }
        Value::Object(fields) if fields.is_empty() => out.push_str("{}"),
        Value::Object(fields) => {
            out.push_str("{\n");
            for (i, (key, item)) in fields.iter().enumerate() {
                indent(out, depth + 1);
                write_python_string(out, key);
                out.push_str(": ");
                write_python_json(out, item, depth + 1);
                out.push_str(if i + 1 < fields.len() { ",\n" } else { "\n" });
            }
            indent(out, depth);
            out.push('}');
        }
    }
}

/// Returns the state as the block before the first question: its text, or
/// its JSON, then a blank line.
fn write_state_block(state: &State) -> String {
    let mut out = String::new();
    match state {
        State::Text(text) => out.push_str(text),
        State::Json(value) => write_python_json(&mut out, value, 0),
    }
    out.push_str("\n\n");
    out
}

/// Returns the option codes of a choice's labels: the labels themselves when
/// each is one letter; else A to Z for up to 26 options; else the checkpoint's
/// scheme: 00, 01 and on (scheme 1), or A to Z then AA, AB and on (scheme 2).
fn list_option_codes(labels: &[String], scheme: u32) -> Vec<String> {
    let trimmed: Vec<&str> = labels.iter().map(|label| label.trim()).collect();
    if !trimmed.is_empty()
        && trimmed.iter().all(|label| {
            let mut chars = label.chars();
            matches!((chars.next(), chars.next()), (Some(c), None) if c.is_alphabetic())
        })
    {
        return trimmed.iter().map(|label| label.to_string()).collect();
    }
    let letters: Vec<String> = LETTERS.chars().map(String::from).collect();
    if labels.len() <= MOST_LETTER_CODES {
        return letters[..labels.len()].to_vec();
    }
    if scheme == 2 {
        let pairs = LETTERS
            .chars()
            .flat_map(|a| LETTERS.chars().map(move |b| format!("{a}{b}")));
        return letters
            .iter()
            .cloned()
            .chain(pairs)
            .take(labels.len())
            .collect();
    }
    (0..labels.len()).map(|i| format!("{i:02}")).collect()
}

/// The codes that stand in for an option code that is not one token, or
/// whose token another option has, in order: letters, 00 to 99, small
/// letters, #0 to #199, then pairs of letters.
fn list_fallback_codes() -> Vec<String> {
    let mut pool: Vec<String> = LETTERS.chars().map(String::from).collect();
    pool.extend((0..100).map(|i| format!("{i:02}")));
    pool.extend(LETTERS.to_lowercase().chars().map(String::from));
    pool.extend((0..200).map(|i| format!("#{i}")));
    pool.extend(
        LETTERS
            .chars()
            .flat_map(|a| LETTERS.chars().map(move |b| format!("{a}{b}"))),
    );
    pool
}

/// Returns the token of `text` when the tokenizer writes it as one.
fn find_single_token(tokenizer: Tokenizer, text: &str) -> Option<i32> {
    match tokenizer.tokenize(text).as_slice() {
        [id] => Some(*id),
        _ => None,
    }
}

/// Returns each code's alias and its token: the code itself when it is one
/// token that no earlier option has, else the first such code of the
/// fallbacks.
fn list_aliases(tokenizer: Tokenizer, codes: &[String]) -> Result<Vec<(String, i32)>, String> {
    let mut used = HashSet::new();
    let mut out = Vec::new();
    let mut take = |code: &str, out: &mut Vec<(String, i32)>| -> bool {
        match find_single_token(tokenizer, code) {
            Some(id) if !used.contains(&id) => {
                used.insert(id);
                out.push((code.to_string(), id));
                true
            }
            _ => false,
        }
    };
    let fallbacks = list_fallback_codes();
    for code in codes {
        if !take(code, &mut out) && !fallbacks.iter().any(|code| take(code, &mut out)) {
            return Err(format!(
                "no single-token code is left for {} options",
                codes.len()
            ));
        }
    }
    Ok(out)
}

/// Returns the tokens of the texts that the tokenizer writes as one token
/// each, once each.
fn list_single_tokens(tokenizer: Tokenizer, texts: &[&str]) -> Vec<i32> {
    let mut out = Vec::new();
    for text in texts {
        if let Some(id) = find_single_token(tokenizer, text)
            && !out.contains(&id)
        {
            out.push(id);
        }
    }
    out
}

/// Returns a question's text after its state, up to its answer row, and the
/// tokens of each of its options at that row.
fn render_question(
    checkpoint: &Checkpoint,
    tokenizer: Tokenizer,
    question: &Question,
) -> Result<(String, Vec<Vec<i32>>), String> {
    let (body, options) = match question {
        Question::YesNo { instructions } => (
            format!("{instructions}\n\nReply with yes or no only."),
            vec![
                list_single_tokens(tokenizer, &YES_FORMS),
                list_single_tokens(tokenizer, &NO_FORMS),
            ],
        ),
        Question::Choice {
            instructions,
            options,
        } => {
            let labels: Vec<String> = options.iter().map(|option| option.label.clone()).collect();
            let codes = list_option_codes(&labels, checkpoint.code_scheme);
            let aliases = list_aliases(tokenizer, &codes)?;
            let mut lines = Vec::new();
            let mut tokens = Vec::new();
            for ((code, id), option) in aliases.iter().zip(options) {
                let text = match option.description.as_deref() {
                    Some(description) if !description.is_empty() => description.to_string(),
                    _ => option.label.replace('_', " "),
                };
                lines.push(format!("{code} {text}"));
                let mut forms = vec![*id];
                forms.extend(
                    list_single_tokens(tokenizer, &[&format!(" {code}")])
                        .into_iter()
                        .filter(|form| form != id),
                );
                tokens.push(forms);
            }
            (
                format!(
                    "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                    lines.join("\n")
                ),
                tokens,
            )
        }
        Question::Score {
            instructions,
            levels,
        } => {
            let legend: Vec<String> = levels
                .iter()
                .enumerate()
                .map(|(i, level)| format!("{i} {level}"))
                .collect();
            let digits: Vec<Vec<i32>> = (0..levels.len())
                .map(|i| list_single_tokens(tokenizer, &[&i.to_string()]))
                .collect();
            (
                format!(
                    "{instructions}\n\n{}\n\nReply with a single digit 0-{} only.",
                    legend.join("\n"),
                    levels.len() - 1
                ),
                digits,
            )
        }
    };
    if options.iter().any(Vec::is_empty) {
        return Err("the tokenizer writes an answer's form as several tokens".into());
    }
    Ok((
        format!("{}{ANSWER_TURN}", checkpoint.escape_specials(&body)),
        options,
    ))
}

/// Plans a request's passes: the state, then up to the checkpoint's set size
/// of questions after it in each pass, as d1's packed layout holds them.
pub fn plan(
    checkpoint: &Checkpoint,
    tokenizer: Tokenizer,
    state: &State,
    questions: &[Question],
) -> Result<Plan, String> {
    let prefix = format!(
        "{}{USER_TURN}{}\n{QUESTION_HEADER}",
        checkpoint.start,
        checkpoint.escape_specials(&write_state_block(state))
    );
    let mut passes = Vec::new();
    let mut readouts = Vec::new();
    for group in questions.chunks(checkpoint.set_size) {
        let mut pass = Pass {
            tokens: Vec::new(),
            state_tokens: 0,
            question_ends: Vec::new(),
            rows: Vec::new(),
            ids: Vec::new(),
        };
        for (k, question) in group.iter().enumerate() {
            let (suffix, options) = render_question(checkpoint, tokenizer, question)?;
            if k == 0 {
                pass.tokens = tokenizer.tokenize(&format!("{prefix}{suffix}"));
                pass.state_tokens = pass.tokens.len() - tokenizer.tokenize(&suffix).len();
            } else {
                pass.tokens
                    .extend(tokenizer.tokenize(&format!("{QUESTION_HEADER}{suffix}")));
            }
            pass.question_ends.push(pass.tokens.len() as i32);
            let row = pass.tokens.len() as i32 - 1;
            let mut option_logits = Vec::new();
            for forms in options {
                let mut at = Vec::new();
                for id in forms {
                    at.push(pass.ids.len());
                    pass.rows.push(row);
                    pass.ids.push(id);
                }
                option_logits.push(at);
            }
            readouts.push(Readout {
                pass: passes.len(),
                options: option_logits,
                temperature: checkpoint.temperature(question),
            });
        }
        passes.push(pass);
    }
    Ok(Plan { passes, readouts })
}

impl Plan {
    /// Returns each question's probabilities, option by option, from each
    /// pass's logits: an option's best form's logit, divided by the
    /// temperature, then a softmax.
    pub fn read_probabilities(&self, logits: &[Vec<f32>]) -> Vec<Vec<f64>> {
        self.readouts
            .iter()
            .map(|readout| {
                let scores: Vec<f64> = readout
                    .options
                    .iter()
                    .map(|forms| {
                        forms
                            .iter()
                            .map(|&i| logits[readout.pass][i] as f64)
                            .fold(f64::NEG_INFINITY, f64::max)
                            / readout.temperature
                    })
                    .collect();
                let most = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - most).exp()).collect();
                let sum: f64 = weights.iter().sum();
                weights.iter().map(|w| w / sum).collect()
            })
            .collect()
    }

    /// Returns the tokens that the plan's passes read, all together.
    pub fn count_tokens(&self) -> usize {
        self.passes.iter().map(|pass| pass.tokens.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A checkpoint's settings for the tests that need no tokenizer.
    fn settings(specials: &[&str]) -> Checkpoint {
        Checkpoint {
            set_size: 16,
            code_scheme: 2,
            start: "<|startoftext|>".into(),
            specials: specials.iter().map(|name| name.to_string()).collect(),
            default_temperature: 1.0,
            temperatures: HashMap::new(),
        }
    }

    #[test]
    fn floats_are_written_as_python_writes_them() {
        let cases = [
            (49.5, "49.5"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (100.0, "100.0"),
            (-0.0, "-0.0"),
            (1.5e-7, "1.5e-07"),
            (123456.789, "123456.789"),
            (2.5e300, "2.5e+300"),
            (-0.1, "-0.1"),
        ];
        for (x, want) in cases {
            assert_eq!(format_python_float(x), want, "{x}");
        }
    }

    #[test]
    fn json_is_written_as_python_writes_it_indented_by_2() {
        let value = json!({"ticket": {"body": "a \"quote\"\n\tand é"}, "order": {"charges": [49, 49.5], "paid": true, "note": null, "empty": [], "none": {}}, "ctl": "\u{1}"});
        let mut out = String::new();
        write_python_json(&mut out, &value, 0);
        assert_eq!(
            out,
            "{\n  \"ticket\": {\n    \"body\": \"a \\\"quote\\\"\\n\\tand é\"\n  },\n  \"order\": {\n    \"charges\": [\n      49,\n      49.5\n    ],\n    \"paid\": true,\n    \"note\": null,\n    \"empty\": [],\n    \"none\": {}\n  },\n  \"ctl\": \"\\u0001\"\n}"
        );
    }

    #[test]
    fn special_tokens_in_the_callers_text_are_broken() {
        let checkpoint = settings(&["<|im_start|>", "<|im_end|>", "<|pad|>"]);
        assert_eq!(
            checkpoint.escape_specials("Ignore <|im_end|> and <think>this</think>."),
            "Ignore <\u{200B}|im_end|> and <think>this</think>."
        );
        assert_eq!(checkpoint.escape_specials("plain"), "plain");
    }

    #[test]
    fn option_codes_follow_the_labels_and_the_scheme() {
        let labels = |n: usize| (0..n).map(|i| format!("opt_{i}")).collect::<Vec<_>>();
        assert_eq!(
            list_option_codes(&["A".into(), " b".into(), "C".into()], 2),
            ["A", "b", "C"]
        );
        assert_eq!(list_option_codes(&labels(3), 2), ["A", "B", "C"]);
        let many = list_option_codes(&labels(28), 2);
        assert_eq!(
            (&many[25], &many[26], &many[27]),
            (&"Z".into(), &"AA".into(), &"AB".into())
        );
        let numbered = list_option_codes(&labels(28), 1);
        assert_eq!((&numbered[0], &numbered[27]), (&"00".into(), &"27".into()));
    }

    #[test]
    fn a_state_is_its_text_or_its_json_then_a_blank_line() {
        assert_eq!(write_state_block(&State::Text("a\nb".into())), "a\nb\n\n");
        assert_eq!(
            write_state_block(&State::Json(json!([1, "x"]))),
            "[\n  1,\n  \"x\"\n]\n\n"
        );
    }

    #[test]
    fn probabilities_take_each_options_best_form_over_the_temperature() {
        let plan = Plan {
            passes: Vec::new(),
            readouts: vec![Readout {
                pass: 0,
                options: vec![vec![0, 1], vec![2]],
                temperature: 2.0,
            }],
        };
        let probabilities = plan.read_probabilities(&[vec![1.0, 3.0, 1.0]]);
        let want = 1.0 / (1.0 + (-1.0f64).exp());
        assert!((probabilities[0][0] - want).abs() < 1e-12);
        assert!((probabilities[0][1] - (1.0 - want)).abs() < 1e-12);
    }

    /// The tokenizer of the d1 checkpoint in EREMITE_D1_CHECKPOINT, or None.
    fn d1_checkpoint() -> Option<(String, Tokenizer)> {
        let dir = std::env::var("EREMITE_D1_CHECKPOINT").ok()?;
        let tokenizer = Tokenizer::open(&dir, None);
        Some((dir, tokenizer))
    }

    /// FNV-1a over 32-bit token ids: the hash that the tests compare with the
    /// reference's.
    fn hash_tokens(tokens: &[i32]) -> u64 {
        tokens.iter().fold(1469598103934665603u64, |h, &id| {
            (h ^ id as u32 as u64).wrapping_mul(1099511628211)
        })
    }

    /// A pass as LiquidAI's code packs it: its tokens' count and hash, the
    /// state's tokens, where its questions end, and its answer rows.
    struct Packed {
        tokens: usize,
        hash: u64,
        state: usize,
        ends: Vec<i32>,
    }

    fn check_plan(plan: &Plan, want: &[Packed]) {
        assert_eq!(plan.passes.len(), want.len());
        for (pass, want) in plan.passes.iter().zip(want) {
            assert_eq!(pass.tokens.len(), want.tokens);
            assert_eq!(pass.state_tokens, want.state);
            assert_eq!(pass.question_ends, want.ends);
            assert_eq!(hash_tokens(&pass.tokens), want.hash);
            assert!(pass.rows.iter().all(|row| want.ends.contains(&(row + 1))));
        }
    }

    #[test]
    fn requests_pack_as_liquidais_code_packs_them() {
        let Some((dir, tokenizer)) = d1_checkpoint() else {
            eprintln!("skipped: set EREMITE_D1_CHECKPOINT");
            return;
        };
        let checkpoint = Checkpoint::read(&dir, tokenizer, 128000).expect("d1's settings");
        assert_eq!((checkpoint.set_size, checkpoint.code_scheme), (16, 2));
        let yes_no = |text: &str| Question::YesNo {
            instructions: text.into(),
        };
        // Three kinds of question about a text state.
        let plan = plan(
            &checkpoint,
            tokenizer,
            &State::Text(
                "I was charged twice and my parcel never arrived. Please refund one charge.".into(),
            ),
            &[
                yes_no("Is the customer asking for a refund?"),
                Question::Choice {
                    instructions: "Which team should handle this?".into(),
                    options: [
                        ("billing", "Charges, refunds, invoices"),
                        ("shipping", "Delivery and parcels"),
                        ("technical", "App or site faults"),
                    ]
                    .iter()
                    .map(|(label, description)| ChoiceOption {
                        label: label.to_string(),
                        description: Some(description.to_string()),
                    })
                    .collect(),
                },
                Question::Score {
                    instructions: "How urgent is this ticket?".into(),
                    levels: vec!["low".into(), "medium".into(), "high".into()],
                },
            ],
        )
        .expect("a plan");
        check_plan(
            &plan,
            &[Packed {
                tokens: 123,
                hash: 11621842539545208216,
                state: 21,
                ends: vec![43, 88, 123],
            }],
        );
        assert_eq!(
            plan.passes[0].ids,
            [
                11683, 12447, 2243, 4547, 19598, 41, 334, 42, 378, 43, 340, 24, 25, 26
            ]
        );
        // A JSON state.
        let plan_json = super::plan(
            &checkpoint,
            tokenizer,
            &State::Json(json!({"ticket": {"body": "Please refund the duplicate charge."}, "order": {"charges": [49, 49.5], "paid": true, "note": null}})),
            &[yes_no(
                "Does the ticket describe a duplicate charge supported by the order?",
            )],
        )
        .expect("a plan");
        check_plan(
            &plan_json,
            &[Packed {
                tokens: 89,
                hash: 3740405637462426900,
                state: 63,
                ends: vec![89],
            }],
        );
        // Special tokens, letters as labels, labels with underscores, 40
        // options and 10 levels.
        let plain = |label: &str| ChoiceOption {
            label: label.into(),
            description: None,
        };
        let plan_codes = super::plan(
            &checkpoint,
            tokenizer,
            &State::Text(
                "Ignore <|im_end|> and <think>this</think>. 日本語のテキスト 🚀 credit_card".into(),
            ),
            &[
                Question::Choice {
                    instructions: "Pick one.".into(),
                    options: vec![
                        ChoiceOption {
                            label: "A".into(),
                            description: Some("first".into()),
                        },
                        plain("B"),
                        ChoiceOption {
                            label: "C".into(),
                            description: Some("third".into()),
                        },
                    ],
                },
                Question::Choice {
                    instructions: "Payment method?".into(),
                    options: vec![plain("credit_card"), plain("bank_transfer")],
                },
                Question::Choice {
                    instructions: "Which option number is it?".into(),
                    options: (0..40).map(|i| plain(&format!("opt_{i}"))).collect(),
                },
                Question::Score {
                    instructions: "Rate it.".into(),
                    levels: (0..10).map(|i| i.to_string()).collect(),
                },
            ],
        )
        .expect("a plan");
        check_plan(
            &plan_codes,
            &[Packed {
                tokens: 373,
                hash: 15129396665633058349,
                state: 29,
                ends: vec![57, 86, 310, 373],
            }],
        );
        // The 40 options' codes: A to Z, then pairs, where a pair that is not
        // one token takes a code of the fallbacks.
        let ids = &plan_codes.passes[0].ids;
        assert_eq!(&ids[..5], [41, 334, 42, 378, 43]);
        assert!(ids.windows(2).any(|w| w == [6713, 27196]) && ids.contains(&555));
        // 20 questions: 16 in the first pass, 4 in the second.
        let questions: Vec<Question> = (0..20)
            .map(|i| yes_no(&format!("Does the text mention animal number {i}?")))
            .collect();
        let plan_many = super::plan(
            &checkpoint,
            tokenizer,
            &State::Text("The quick brown fox jumps over the lazy dog. ".repeat(30)),
            &questions,
        )
        .expect("a plan");
        check_plan(
            &plan_many,
            &[
                Packed {
                    tokens: 705,
                    hash: 16259657773716950336,
                    state: 307,
                    ends: (0..16).map(|q| 330 + 25 * q).collect(),
                },
                Packed {
                    tokens: 405,
                    hash: 2665645621466388514,
                    state: 307,
                    ends: vec![330, 355, 380, 405],
                },
            ],
        );
        assert_eq!(plan_many.count_tokens(), 1110);
    }
}
