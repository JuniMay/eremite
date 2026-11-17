//! An LLM session's text, by its messages, for the text panel of States. The
//! model's family tells which special tokens start a message of which role.

use unicode_width::UnicodeWidthChar;

/// A message of a session's text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Message {
    /// The message's role, such as user or assistant. Empty for the text
    /// before the first message.
    pub role: String,
    /// The message's text.
    pub text: String,
}

/// A session's text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionText {
    /// The session's number.
    pub session: u64,
    /// The session's tokens, for the panel to tell when the session grew.
    pub tokens: usize,
    /// The session's messages, in order.
    pub messages: Vec<Message>,
}

/// Splits `tokens` into messages. `token_text` gives a token's bytes, and
/// whether it is a special token. `find_role` gives the role whose message a
/// special token starts. Another special token stays out of the text. A
/// message without text is left out.
pub fn split_into_messages<'a>(
    tokens: &[i32],
    token_text: impl Fn(i32) -> (&'a [u8], bool),
    find_role: fn(&str) -> Option<&'static str>,
) -> Vec<Message> {
    let mut messages = Vec::new();
    let (mut role, mut bytes) = (String::new(), Vec::new());
    let mut finish = |role: &str, bytes: &mut Vec<u8>| {
        let text = String::from_utf8_lossy(bytes).trim().to_string();
        if !text.is_empty() {
            messages.push(Message {
                role: role.to_string(),
                text,
            });
        }
        bytes.clear();
    };
    for &token in tokens {
        let (piece, special) = token_text(token);
        if !special {
            bytes.extend_from_slice(piece);
            continue;
        }
        if let Some(next) = find_role(&String::from_utf8_lossy(piece)) {
            finish(&role, &mut bytes);
            role = next.to_string();
        }
    }
    finish(&role, &mut bytes);
    messages
}

/// Wraps `text` in lines of at most `columns` columns. It breaks at the
/// text's newlines, and at spaces where it can.
pub fn wrap_text(text: &str, columns: usize) -> Vec<String> {
    let columns = columns.max(1);
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let (mut line, mut used) = (String::new(), 0);
        for word in paragraph.split_inclusive(' ') {
            let width: usize = word.chars().map(|c| c.width().unwrap_or(0)).sum();
            if used + width > columns && used > 0 {
                lines.push(std::mem::take(&mut line).trim_end().to_string());
                used = 0;
            }
            for character in word.chars() {
                let character_width = character.width().unwrap_or(0);
                if used + character_width > columns {
                    lines.push(std::mem::take(&mut line));
                    used = 0;
                }
                line.push(character);
                used += character_width;
            }
        }
        lines.push(line.trim_end().to_string());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tokens of a test: each token's text, and whether it is special.
    const TOKENS: [(&str, bool); 9] = [
        ("<bos>", true),
        ("<user>", true),
        ("Read the note ", false),
        ("and list the jobs.", false),
        ("<assistant>", true),
        ("<think>", true),
        ("1. Stake the tomatoes.\n", false),
        ("2. Fix the tap.", false),
        ("<end>", true),
    ];

    /// The role whose message a test token starts.
    fn find_role(token: &str) -> Option<&'static str> {
        match token {
            "<user>" => Some("user"),
            "<assistant>" => Some("assistant"),
            _ => None,
        }
    }

    #[test]
    fn a_sessions_tokens_split_into_messages_by_role() {
        let ids: Vec<i32> = (0..TOKENS.len() as i32).collect();
        let token_text = |id: i32| {
            let (text, special) = TOKENS[id as usize];
            (text.as_bytes(), special)
        };
        let messages = split_into_messages(&ids, token_text, find_role);
        assert_eq!(
            messages,
            [
                Message {
                    role: "user".into(),
                    text: "Read the note and list the jobs.".into(),
                },
                Message {
                    role: "assistant".into(),
                    text: "1. Stake the tomatoes.\n2. Fix the tap.".into(),
                },
            ]
        );
        // a character split over two tokens stays whole
        let bytes = "é".as_bytes();
        let halves = |id: i32| (&bytes[id as usize..id as usize + 1], false);
        assert_eq!(split_into_messages(&[0, 1], halves, find_role)[0].text, "é");
    }

    #[test]
    fn text_wraps_at_spaces_and_newlines() {
        assert_eq!(
            wrap_text("Below is a short note.\nRead it.", 12),
            ["Below is a", "short note.", "Read it."]
        );
        // a word longer than the line breaks inside
        assert_eq!(wrap_text("abcdefgh", 3), ["abc", "def", "gh"]);
        assert_eq!(wrap_text("", 10), [""]);
    }
}
