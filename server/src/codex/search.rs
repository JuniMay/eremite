//! POST /v1/alpha/search: the service behind Codex's standalone web tool
//! (`web.run`, Codex 0.167).
//!
//! - HTML pages become numbered lines, and their links are numbered too.

use url::Url;

/// Lines longer than this are wrapped at a space.
const MAX_LINE_CHARS: usize = 200;

// ---- pages ----

/// A page as lines of text, where link n appears as 【n†text】.
#[derive(Debug)]
pub struct Page {
    /// The page's URL.
    pub url: String,
    /// The page's title, or its URL when it has none.
    pub title: String,
    /// The page's text in lines.
    pub lines: Vec<String>,
    /// The URL of each link: link n's URL is `links[n]`.
    pub links: Vec<String>,
}

// ---- HTML to lines ----

/// Elements whose content is not text.
const NON_TEXT_ELEMENTS: [&str; 9] = [
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "head",
];
/// Elements that start a new line.
const BLOCK_ELEMENTS: [&str; 33] = [
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "main",
    "aside",
    "ul",
    "ol",
    "li",
    "table",
    "thead",
    "tbody",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "blockquote",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "form",
    "details",
    "summary",
    "address",
];

/// A page from HTML: text in lines, headings marked with #, list items with
/// -, table cells separated by |, and each link as 【n†text】 with its
/// absolute URL as link n.
pub fn convert_html_to_page(base_url: &Url, html: &str) -> Page {
    let mut out = String::new();
    let mut title = String::new();
    let mut links: Vec<String> = Vec::new();
    // Each open anchor's link number and where its text starts in `out`.
    // An anchor without an http or https link has usize::MAX.
    let mut open_links: Vec<(usize, usize)> = Vec::new();
    // How many <pre> elements are open. Their text keeps its whitespace.
    let mut pre = 0u32;
    let mut position = 0;
    let bytes = html.as_bytes();
    while position < bytes.len() {
        if bytes[position] != b'<' {
            let end = html[position..]
                .find('<')
                .map_or(html.len(), |offset| position + offset);
            let text = decode_entities(&html[position..end]);
            if pre > 0 {
                out.push_str(&text);
            } else {
                append_collapsed(&mut out, &text);
            }
            position = end;
            continue;
        }
        if html[position..].starts_with("<!--") {
            // 3 bytes: the --> that ends a comment.
            position = html[position..]
                .find("-->")
                .map_or(html.len(), |offset| position + offset + 3);
            continue;
        }
        let Some(close) = find_tag_end(html, position) else {
            break;
        };
        let tag = &html[position + 1..close];
        position = close + 1;
        if tag.starts_with('!') || tag.starts_with('?') {
            continue;
        }
        let is_closing_tag = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if name.is_empty() {
            continue;
        }
        if !is_closing_tag && (name == "title") {
            let end = find_closing_tag(html, position, "title");
            title = collapse_whitespace(&decode_entities(&html[position..end]));
            position = skip_closing_tag(html, end);
            continue;
        }
        if !is_closing_tag && NON_TEXT_ELEMENTS.contains(&name.as_str()) && !tag.ends_with('/') {
            let end = find_closing_tag(html, position, &name);
            // a head's title is still read
            if name == "head"
                && let Some(t) = html[position..end].to_ascii_lowercase().find("<title")
            {
                let start = position + t;
                if let Some(gt) = html[start..end].find('>') {
                    let from = start + gt + 1;
                    let to = find_closing_tag(html, from, "title").min(end);
                    title = collapse_whitespace(&decode_entities(&html[from..to]));
                }
            }
            position = skip_closing_tag(html, end);
            continue;
        }
        match (name.as_str(), is_closing_tag) {
            ("br", _) => out.push('\n'),
            ("hr", false) => out.push_str("\n---\n"),
            ("pre", false) => {
                out.push('\n');
                pre += 1;
            }
            ("pre", true) => {
                out.push('\n');
                pre = pre.saturating_sub(1);
            }
            ("a", false) => {
                let href = find_attribute(tag, "href").unwrap_or_default();
                let target = base_url
                    .join(href.trim())
                    .ok()
                    .filter(|link_url| matches!(link_url.scheme(), "http" | "https"));
                match target {
                    Some(link_url) if !href.trim().is_empty() => {
                        let mut link_url = link_url;
                        link_url.set_fragment(None);
                        links.push(link_url.to_string());
                        open_links.push((links.len() - 1, out.len()));
                    }
                    _ => open_links.push((usize::MAX, out.len())),
                }
            }
            ("a", true) => {
                if let Some((link_number, start)) = open_links.pop()
                    && link_number != usize::MAX
                {
                    let text = collapse_whitespace(&out[start..]);
                    out.truncate(start);
                    if text.is_empty() {
                        out.push_str(&format!("【{link_number}】"));
                    } else {
                        out.push_str(&format!("【{link_number}†{text}】"));
                    }
                }
            }
            ("img", false) => {
                if !open_links.is_empty()
                    && let Some(alt) = find_attribute(tag, "alt").filter(|a| !a.trim().is_empty())
                {
                    append_collapsed(&mut out, &format!(" {} ", decode_entities(&alt)));
                }
            }
            ("td" | "th", false) => {
                if !out.ends_with('\n') && !out.is_empty() {
                    out.push_str(" | ");
                }
            }
            (n, false) if BLOCK_ELEMENTS.contains(&n) => {
                out.push('\n');
                if let Some(level) = n.strip_prefix('h').and_then(|l| l.parse::<usize>().ok()) {
                    out.push_str(&"#".repeat(level));
                    out.push(' ');
                } else if n == "li" {
                    out.push_str("- ");
                }
            }
            (n, true) if BLOCK_ELEMENTS.contains(&n) => out.push('\n'),
            _ => {}
        }
    }
    let lines = wrap_lines(
        out.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && *l != "-"),
    );
    Page {
        url: base_url.to_string(),
        title: if title.is_empty() {
            base_url.to_string()
        } else {
            title
        },
        lines,
        links,
    }
}

/// The index of the tag's closing `>`, quotes respected.
fn find_tag_end(html: &str, start: usize) -> Option<usize> {
    let mut quote: Option<u8> = None;
    for (offset, &byte) in html.as_bytes()[start + 1..].iter().enumerate() {
        match (quote, byte) {
            (Some(quote_byte), byte) if byte == quote_byte => quote = None,
            (Some(_), _) => {}
            (None, b'"' | b'\'') => quote = Some(byte),
            (None, b'>') => return Some(start + 1 + offset),
            _ => {}
        }
    }
    None
}

/// Where the element's closing tag starts, or the end of the text.
fn find_closing_tag(html: &str, from: usize, name: &str) -> usize {
    let lower = html[from..].to_ascii_lowercase();
    lower
        .find(&format!("</{name}"))
        .map_or(html.len(), |offset| from + offset)
}

/// Returns the position after the `>` of the closing tag that starts at
/// `at`, or the end of the text.
fn skip_closing_tag(html: &str, at: usize) -> usize {
    html[at..]
        .find('>')
        .map_or(html.len(), |offset| at + offset + 1)
}

/// An attribute's value in a start tag.
fn find_attribute(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let mut search = 0;
    while let Some(offset) = lower[search..].find(name) {
        let at = search + offset;
        search = at + name.len();
        let before = lower[..at].chars().last();
        if !before.is_some_and(char::is_whitespace) {
            continue;
        }
        let rest = tag[at + name.len()..].trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        let rest = rest.trim_start();
        return Some(match rest.chars().next() {
            Some(q @ ('"' | '\'')) => rest[1..].split(q).next().unwrap_or("").to_string(),
            _ => rest
                .split(|c: char| c.is_whitespace() || c == '>')
                .next()
                .unwrap_or("")
                .to_string(),
        });
    }
    None
}

/// Decodes the text's HTML entities, such as &amp; and &#39;. An unknown
/// entity stays as it is.
fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(offset) = rest.find('&') {
        out.push_str(&rest[..offset]);
        rest = &rest[offset..];
        // An entity's ';' is at most 10 bytes after its '&'. The reason for
        // this value is unknown.
        let end = rest[1..]
            .find(';')
            .map(|offset| offset + 1)
            .filter(|&offset| offset <= 10);
        let decoded = end.and_then(|end| {
            let entity = &rest[1..end];
            let c = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" | "#39" => Some('\''),
                "nbsp" => Some(' '),
                "ndash" => Some('–'),
                "mdash" => Some('—'),
                "hellip" => Some('…'),
                "lsquo" => Some('‘'),
                "rsquo" => Some('’'),
                "ldquo" => Some('“'),
                "rdquo" => Some('”'),
                "copy" => Some('©'),
                "reg" => Some('®'),
                "trade" => Some('™'),
                e if e.starts_with("#x") || e.starts_with("#X") => u32::from_str_radix(&e[2..], 16)
                    .ok()
                    .and_then(char::from_u32),
                e if e.starts_with('#') => e[1..].parse::<u32>().ok().and_then(char::from_u32),
                _ => None,
            };
            c.map(|c| (c, end))
        });
        match decoded {
            Some((c, end)) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Appends the text with each run of whitespace as one space.
/// It never doubles a space or starts a line with one.
/// A space comes before the text only where the text starts with whitespace.
fn append_collapsed(out: &mut String, text: &str) {
    let mut needs_space = text.starts_with(char::is_whitespace);
    for word in text.split_whitespace() {
        if needs_space && !out.is_empty() && !out.ends_with([' ', '\n']) {
            out.push(' ');
        }
        out.push_str(word);
        needs_space = true;
    }
    if text.ends_with(char::is_whitespace) && !out.is_empty() && !out.ends_with([' ', '\n']) {
        out.push(' ');
    }
}

/// Returns the text with each run of whitespace as one space, and none at
/// its ends.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Lines no longer than [`MAX_LINE_CHARS`], longer ones wrapped at spaces.
fn wrap_lines<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        let line = line.trim_end();
        if line.chars().count() <= MAX_LINE_CHARS {
            out.push(line.to_string());
            continue;
        }
        let mut current = String::new();
        for word in line.split(' ') {
            if !current.is_empty()
                && current.chars().count() + 1 + word.chars().count() > MAX_LINE_CHARS
            {
                out.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_becomes_lines_with_numbered_links() {
        let base = Url::parse("https://example.com/docs/intro.html").unwrap();
        let html = r#"<!doctype html><html><head><title>Intro &amp; more</title>
            <style>p { color: red }</style><script>var x = "<p>no</p>";</script></head>
            <body><h1>Getting  started</h1><!-- a comment <p>hidden</p> -->
            <p>Read the <a href="setup.html#top">setup guide</a> or the <a href='/faq'>FAQ</a>.</p>
            <ul><li>one</li><li>two &lt;three&gt;</li></ul>
            <table><tr><td>a</td><td>b</td></tr></table>
            <pre>let x = 1;
  indented</pre><a href="javascript:void(0)">nothing</a><a href="https://other.org/"><img alt="Logo"></a></body></html>"#;
        let p = convert_html_to_page(&base, html);
        assert_eq!(p.title, "Intro & more");
        assert_eq!(
            p.links,
            [
                "https://example.com/docs/setup.html",
                "https://example.com/faq",
                "https://other.org/"
            ]
        );
        let text = p.lines.join("\n");
        assert!(text.contains("# Getting started"), "{text}");
        assert!(
            text.contains("Read the 【0†setup guide】 or the 【1†FAQ】."),
            "{text}"
        );
        assert!(text.contains("- one\n- two <three>"), "{text}");
        assert!(text.contains("a | b"), "{text}");
        assert!(text.contains("let x = 1;\nindented"), "{text}");
        assert!(text.contains("【2†Logo】"), "{text}");
        assert!(
            text.contains("nothing") && !text.contains("javascript"),
            "{text}"
        );
        assert!(
            !text.contains("color") && !text.contains("hidden") && !text.contains("var x"),
            "{text}"
        );
    }
}
