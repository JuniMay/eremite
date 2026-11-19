//! OpenAI's Decisions API (POST /v1/decisions): a request's input and
//! questions, and the answers. A request names a model, gives its input, a
//! string or a list of messages and tool items, which may hold images, and
//! asks 1 to 400 questions: predicates (yes or no), choices and scores. The
//! answers come back in the questions' order, with the tokens that the model
//! read.

use serde_json::{Map, Value, json};

use super::d1::{ChoiceOption, Question, State, format_python_json};
use crate::chat::Error;

/// The most questions of a request, as OpenAI's API takes them.
const MOST_QUESTIONS: usize = 400;
/// The fewest and the most options of a choice, as OpenAI's API takes them.
const CHOICE_OPTIONS: std::ops::RangeInclusive<usize> = 2..=255;
/// The fewest and the most levels of a score, as OpenAI's API and d1 take
/// them.
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;

/// A decision request: its images and its state, and its questions with their
/// names and their options' values.
pub struct Request {
    /// What the questions are about, besides the images: none for images
    /// alone.
    pub state: Option<State>,
    /// The image files, in the input's order. They head the state.
    pub images: Vec<Vec<u8>>,
    /// The questions, as d1 asks them.
    pub questions: Vec<Question>,
    /// Each question's name, if it has one, and its kind and options' values
    /// as the answers give them back.
    asked: Vec<Asked>,
}

/// A question as the answers name it.
struct Asked {
    name: Option<String>,
    kind: AskedKind,
}

/// A question's kind, with what its answer gives back.
enum AskedKind {
    /// A predicate's answer: the probability that it holds.
    Predicate,
    /// A choice's answer: one of its options' values, a string or a boolean.
    Choice(Vec<Value>),
    /// A score's answer: the probability-weighted level, with each level's
    /// label.
    Score(Vec<String>),
}

/// Returns the text of `value`, a JSON string, or a 400 error about `what`.
fn read_text(value: Option<&Value>, what: &str) -> Result<String, Error> {
    value
        .and_then(Value::as_str)
        .map(String::from)
        .ok_or_else(|| Error::new_bad_request(format!("{what} must be a string")))
}

/// Returns a question's instructions: a string, or text and JSON parts, the
/// parts joined by blank lines with the JSON written as text.
fn read_instructions(value: Option<&Value>, at: &str) -> Result<String, Error> {
    match value {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) if !parts.is_empty() => {
            let mut texts = Vec::new();
            for part in parts {
                match part["type"].as_str() {
                    Some("input_text") => texts.push(read_text(part.get("text"), "a text part")?),
                    Some("input_json") => texts.push(format_python_json(&part["data"])),
                    _ => {
                        return Err(Error::new_bad_request(format!(
                            "{at}.instructions: this model reads text and JSON parts only"
                        )));
                    }
                }
            }
            Ok(texts.join("\n\n"))
        }
        _ => Err(Error::new_bad_request(format!(
            "{at}.instructions must be a string or a list of parts"
        ))),
    }
}

/// Reads one question at `at` of a request.
fn read_question(value: &Value, at: &str) -> Result<(Question, Asked), Error> {
    let name = match value.get("name") {
        None | Some(Value::Null) => None,
        Some(name) => Some(read_text(Some(name), &format!("{at}.name"))?),
    };
    let instructions = read_instructions(value.get("instructions"), at)?;
    let read_list = |key: &str, range: &std::ops::RangeInclusive<usize>| match value.get(key) {
        Some(Value::Array(items)) if range.contains(&items.len()) => Ok(items.clone()),
        _ => Err(Error::new_bad_request(format!(
            "{at}.{key} must list {} to {} items",
            range.start(),
            range.end()
        ))),
    };
    let optional_text = |item: &Value, what: &str| match item.get("description") {
        None | Some(Value::Null) => Ok(None),
        description => read_text(description, what).map(Some),
    };
    match value["type"].as_str() {
        Some("predicate") => Ok((
            Question::YesNo { instructions },
            Asked {
                name,
                kind: AskedKind::Predicate,
            },
        )),
        Some("choice") => {
            let mut options = Vec::new();
            let mut values = Vec::new();
            for item in read_list("choices", &CHOICE_OPTIONS)? {
                let label = match &item["value"] {
                    Value::String(text) => text.clone(),
                    Value::Bool(b) => b.to_string(),
                    _ => {
                        return Err(Error::new_bad_request(format!(
                            "{at}.choices: a value is a string or a boolean"
                        )));
                    }
                };
                if options
                    .iter()
                    .any(|option: &ChoiceOption| option.label == label)
                {
                    return Err(Error::new_bad_request(format!(
                        "{at}.choices: the value {label} is there twice"
                    )));
                }
                options.push(ChoiceOption {
                    label,
                    description: optional_text(&item, &format!("{at}.choices: a description"))?,
                });
                values.push(item["value"].clone());
            }
            Ok((
                Question::Choice {
                    instructions,
                    options,
                },
                Asked {
                    name,
                    kind: AskedKind::Choice(values),
                },
            ))
        }
        Some("score") => {
            let mut levels = Vec::new();
            let mut labels = Vec::new();
            for item in read_list("levels", &SCORE_LEVELS)? {
                let label = read_text(item.get("label"), &format!("{at}.levels: a label"))?;
                // d1 shows a level's description when it has one, as it shows
                // a choice's.
                let description = optional_text(&item, &format!("{at}.levels: a description"))?;
                levels.push(
                    description
                        .filter(|d| !d.is_empty())
                        .unwrap_or(label.clone()),
                );
                labels.push(label);
            }
            Ok((
                Question::Score {
                    instructions,
                    levels,
                },
                Asked {
                    name,
                    kind: AskedKind::Score(labels),
                },
            ))
        }
        _ => Err(Error::new_bad_request(format!(
            "{at}.type is predicate, choice or score"
        ))),
    }
}

/// Returns a message part's value in the state: its text or its JSON.
fn read_part(part: &Value, role: &str) -> Result<Value, Error> {
    match (part["type"].as_str(), role) {
        (Some("input_text" | "output_text"), _) => {
            Ok(Value::String(read_text(part.get("text"), "a text part")?))
        }
        (Some("refusal"), "assistant") => Ok(Value::String(read_text(
            part.get("refusal"),
            "a refusal part",
        )?)),
        (Some("input_json"), "user") => Ok(part["data"].clone()),
        _ => Err(Error::new_bad_request(format!(
            "a {role} message's parts are not of that type"
        ))),
    }
}

/// Returns a message's content in the state: its text, its one part's value,
/// or the list of its parts' values.
fn read_content(content: &Value, role: &str) -> Result<Value, Error> {
    match content {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(parts) if parts.len() == 1 => read_part(&parts[0], role),
        Value::Array(parts) => Ok(Value::Array(
            parts
                .iter()
                .map(|part| read_part(part, role))
                .collect::<Result<_, _>>()?,
        )),
        _ => Err(Error::new_bad_request(
            "a message's content is a string or a list of parts",
        )),
    }
}

/// Returns the files of the input_image parts of the user messages of a
/// request's input, in order, and the input without those parts. An image's
/// file comes from its image_url, a base64 data URL.
fn take_images(input: &Value) -> Result<(Value, Vec<Vec<u8>>), Error> {
    let Value::Array(items) = input else {
        return Ok((input.clone(), Vec::new()));
    };
    let mut images = Vec::new();
    let mut kept = Vec::new();
    for item in items {
        let mut item = item.clone();
        if item["role"] == "user"
            && let Some(parts) = item["content"].as_array()
        {
            let mut rest = Vec::new();
            for part in parts {
                if part["type"] != "input_image" {
                    rest.push(part.clone());
                    continue;
                }
                let url = part["image_url"].as_str().ok_or_else(|| {
                    Error::new_bad_request(
                        "input_image: only an image_url with a base64 data URL is read (the server fetches no URLs and reads no files)",
                    )
                })?;
                images.push(crate::chat::decode_data_url(url)?);
            }
            item["content"] = Value::Array(rest);
        }
        kept.push(item);
    }
    Ok((Value::Array(kept), images))
}

/// Returns the state of a request's input, its images taken out. A string is
/// the state's text. One user message whose content is a string or one text
/// or JSON part is that text or JSON; one whose content held images only gives
/// no state. Any other list becomes a JSON list of its items: a message as its
/// role and content (system as developer, the content a string, or a list of
/// its parts' texts and JSON), a tool call or output as sent.
fn read_state(input: &Value) -> Result<Option<State>, Error> {
    let items = match input {
        Value::String(text) => return Ok(Some(State::Text(text.clone()))),
        Value::Array(items) => items,
        _ => {
            return Err(Error::new_bad_request(
                "input must be a string or a list of items",
            ));
        }
    };
    let role_of = |item: &Value| item["role"].as_str().map(String::from);
    if let [item] = items.as_slice()
        && role_of(item).as_deref() == Some("user")
        && !item["content"]
            .as_array()
            .is_some_and(|parts| parts.len() > 1)
    {
        if item["content"].as_array().is_some_and(Vec::is_empty) {
            return Ok(None);
        }
        return Ok(Some(match read_content(&item["content"], "user")? {
            Value::String(text) => State::Text(text),
            data => State::Json(data),
        }));
    }
    let mut out = Vec::new();
    for item in items {
        match role_of(item).as_deref() {
            Some(role @ ("user" | "developer" | "system" | "assistant")) => {
                let role = if role == "system" { "developer" } else { role };
                out.push(json!({"role": role, "content": read_content(&item["content"], role)?}));
            }
            Some(role) => {
                return Err(Error::new_bad_request(format!("no message role {role}")));
            }
            None => match item["type"].as_str() {
                Some(
                    "function_call"
                    | "function_call_output"
                    | "custom_tool_call"
                    | "custom_tool_call_output",
                ) => out.push(item.clone()),
                _ => {
                    return Err(Error::new_bad_request(
                        "input items are messages and tool calls and outputs",
                    ));
                }
            },
        }
    }
    Ok(Some(State::Json(Value::Array(out))))
}

/// Reads a decision request's JSON body.
pub fn read_request(body: &[u8]) -> Result<Request, Error> {
    let root: Map<String, Value> = serde_json::from_slice(body)
        .map_err(|error| Error::new_bad_request(format!("not a decision request: {error}")))?;
    let (input, images) = take_images(root.get("input").ok_or_else(|| {
        Error::new_bad_request("input is required: a string or a list of items")
    })?)?;
    let state = read_state(&input)?;
    if state.is_none() && images.is_empty() {
        return Err(Error::new_bad_request("input holds no text, JSON or image"));
    }
    let questions = match root.get("questions") {
        Some(Value::Array(items)) if (1..=MOST_QUESTIONS).contains(&items.len()) => items,
        _ => {
            return Err(Error::new_bad_request(format!(
                "questions must list 1 to {MOST_QUESTIONS} questions"
            )));
        }
    };
    let mut request = Request {
        state,
        images,
        questions: Vec::new(),
        asked: Vec::new(),
    };
    for (i, item) in questions.iter().enumerate() {
        let (question, asked) = read_question(item, &format!("questions[{i}]"))?;
        request.questions.push(question);
        request.asked.push(asked);
    }
    Ok(request)
}

impl Request {
    /// Returns the answers' JSON from each question's probabilities, option
    /// by option: a predicate's probability of yes; a choice's likeliest
    /// option, its probability and every option's; a score's
    /// probability-weighted level, its likeliest level's probability and
    /// every level's.
    pub fn build_answers(&self, probabilities: &[Vec<f64>]) -> Vec<Value> {
        self.asked
            .iter()
            .zip(probabilities)
            .map(|(asked, p)| {
                let best = (0..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best });
                match &asked.kind {
                    AskedKind::Predicate => json!({
                        "type": "predicate", "name": asked.name, "probability": p[0],
                    }),
                    AskedKind::Choice(values) => json!({
                        "type": "choice", "name": asked.name, "choice": values[best],
                        "confidence": p[best],
                        "probabilities": values.iter().zip(p).map(|(value, probability)| {
                            json!({"value": value, "probability": probability})
                        }).collect::<Vec<_>>(),
                    }),
                    AskedKind::Score(labels) => json!({
                        "type": "score", "name": asked.name,
                        "score": p.iter().enumerate().map(|(i, q)| i as f64 * q).sum::<f64>(),
                        "confidence": p[best],
                        "probabilities": labels.iter().zip(p).enumerate().map(|(i, (label, probability))| {
                            json!({"value": i, "label": label, "probability": probability})
                        }).collect::<Vec<_>>(),
                    }),
                }
            })
            .collect()
    }
}

/// Returns a decision's response: the model's name, the answers, and the
/// tokens that the model read. A decision writes no tokens.
pub fn build_response(model: &str, answers: Vec<Value>, input_tokens: usize) -> Value {
    json!({
        "model": model,
        "answers": answers,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(body: Value) -> Result<Request, Error> {
        read_request(body.to_string().as_bytes())
    }

    fn state_json(input: Value) -> Value {
        match read(
            json!({"input": input, "questions": [{"type": "predicate", "instructions": "?"}]}),
        )
        .expect("a request")
        .state
        {
            Some(State::Json(value)) => value,
            Some(State::Text(text)) => Value::String(format!("text: {text}")),
            None => Value::Null,
        }
    }

    #[test]
    fn the_input_becomes_the_state() {
        assert_eq!(state_json(json!("plain")), json!("text: plain"));
        assert_eq!(
            state_json(json!([{"role": "user", "content": "one"}])),
            json!("text: one")
        );
        assert_eq!(
            state_json(
                json!([{"role": "user", "content": [{"type": "input_json", "data": {"a": 1}}]}])
            ),
            json!({"a": 1})
        );
        assert_eq!(
            state_json(json!([{"role": "user", "content": [
                {"type": "input_text", "text": "Check this."},
                {"type": "input_json", "data": [1]}]}])),
            json!([{"role": "user", "content": ["Check this.", [1]]}])
        );
        assert_eq!(
            state_json(json!([
                {"role": "system", "content": "Be fair."},
                {"role": "user", "content": "Hi"},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}])),
            json!([
                {"role": "developer", "content": "Be fair."},
                {"role": "user", "content": "Hi"},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}])
        );
    }

    #[test]
    fn bad_requests_are_refused_with_what_to_change() {
        let refused = |body: Value| read(body).err().map(|error| error.message);
        assert!(refused(json!({"questions": []})).is_some_and(|m| m.contains("input")));
        assert!(
            refused(json!({"input": "x", "questions": []})).is_some_and(|m| m.contains("1 to 400"))
        );
        assert!(
            refused(
                json!({"input": "x", "questions": [{"type": "choice", "instructions": "?",
            "choices": [{"value": "a"}]}]})
            )
            .is_some_and(|m| m.contains("2 to 255"))
        );
        assert!(
            refused(
                json!({"input": "x", "questions": [{"type": "choice", "instructions": "?",
            "choices": [{"value": "a"}, {"value": "a"}]}]})
            )
            .is_some_and(|m| m.contains("twice"))
        );
        assert!(
            refused(
                json!({"input": "x", "questions": [{"type": "score", "instructions": "?",
            "levels": [{"label": "only"}]}]})
            )
            .is_some_and(|m| m.contains("2 to 10"))
        );
        assert!(
            refused(json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.com/a.png"}]}],
            "questions": [{"type": "predicate", "instructions": "?"}]}))
            .is_some_and(|m| m.contains("data URL"))
        );
        assert!(
            refused(json!({"input": [{"role": "user", "content": []}],
            "questions": [{"type": "predicate", "instructions": "?"}]}))
            .is_some_and(|m| m.contains("no text, JSON or image"))
        );
    }

    #[test]
    fn images_head_the_state_in_the_inputs_order() {
        // "aGk=" and "eW8=" are "hi" and "yo" in base64.
        let read_images = |input: Value| {
            let request = read(
                json!({"input": input, "questions": [{"type": "predicate", "instructions": "?"}]}),
            )
            .expect("a request");
            let state = match request.state {
                Some(State::Text(text)) => json!(text),
                Some(State::Json(value)) => value,
                None => Value::Null,
            };
            (request.images, state)
        };
        let image = |data: &str| json!({"type": "input_image", "image_url": format!("data:image/png;base64,{data}")});
        assert_eq!(
            read_images(json!([{"role": "user", "content": [image("aGk=")]}])),
            (vec![b"hi".to_vec()], Value::Null),
            "an image alone gives no state"
        );
        assert_eq!(
            read_images(json!([{"role": "user", "content": [
                image("aGk="), {"type": "input_text", "text": "Is it a cat?"}, image("eW8=")]}])),
            (vec![b"hi".to_vec(), b"yo".to_vec()], json!("Is it a cat?")),
            "the images head the state's text, in order"
        );
        assert_eq!(
            read_images(json!([
                {"role": "user", "content": [image("aGk=")]},
                {"role": "assistant", "content": "A cat."}])),
            (
                vec![b"hi".to_vec()],
                json!([{"role": "user", "content": []}, {"role": "assistant", "content": "A cat."}])
            ),
            "a list keeps its messages, without their images"
        );
    }

    #[test]
    fn answers_follow_openais_shapes() {
        let request = read(json!({"input": "x", "questions": [
            {"type": "predicate", "name": "p", "instructions": "?"},
            {"type": "choice", "instructions": "?", "choices": [{"value": "a"}, {"value": true}]},
            {"type": "score", "name": "s", "instructions": "?",
             "levels": [{"label": "low"}, {"label": "mid", "description": "Middle"}, {"label": "high"}]}]}))
        .expect("a request");
        match &request.questions[2] {
            Question::Score { levels, .. } => assert_eq!(levels, &["low", "Middle", "high"]),
            _ => panic!("a score"),
        }
        let answers =
            request.build_answers(&[vec![0.9, 0.1], vec![0.25, 0.75], vec![0.1, 0.7, 0.2]]);
        assert_eq!(
            answers[0],
            json!({"type": "predicate", "name": "p", "probability": 0.9})
        );
        assert_eq!(answers[1]["name"], Value::Null);
        assert_eq!(
            (&answers[1]["choice"], &answers[1]["confidence"]),
            (&json!(true), &json!(0.75))
        );
        assert_eq!(
            answers[1]["probabilities"][0],
            json!({"value": "a", "probability": 0.25})
        );
        assert!((answers[2]["score"].as_f64().unwrap() - 1.1).abs() < 1e-12);
        assert_eq!(answers[2]["confidence"], json!(0.7));
        assert_eq!(
            answers[2]["probabilities"][1],
            json!({"value": 1, "label": "mid", "probability": 0.7})
        );
        let response = build_response("d1-2b", answers, 96);
        assert_eq!(response["usage"]["total_tokens"], json!(96));
        assert_eq!(response["usage"]["output_tokens"], json!(0));
    }
}
