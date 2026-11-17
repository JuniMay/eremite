//! POST /v1/alpha/search: the service behind Codex's standalone web tool
//! (`web.run`, Codex 0.167).
//!
//! - The system curl fetches pages.
//!   Pages on loopback, private, link-local and other non-public addresses are
//!   refused, at every redirect.
//! - HTML pages become numbered lines, and their links are numbered too.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use url::{Host, Url};

use crate::http::{KIB, MIB};

/// The system's curl.
const CURL_PATH: &str = "/usr/bin/curl";
/// A page's bytes at most; a longer page is cut there.
const MAX_PAGE_BYTES: usize = 5 * MIB;
/// The redirects that a fetch follows at most.
const MAX_REDIRECTS: usize = 5;
/// Lines longer than this are wrapped at a space.
const MAX_LINE_CHARS: usize = 200;
/// The User-Agent header that curl sends: Safari's on macOS.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";

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

// ---- fetching ----

/// Which addresses pages may be fetched from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AddressPolicy {
    /// Public addresses only.
    Public,
    /// Any address, for tests against a local server.
    Any,
}

/// Whether pages may be fetched from the address: public addresses only,
/// unless the policy allows any.
fn is_address_allowed(ip: IpAddr, policy: AddressPolicy) -> bool {
    policy == AddressPolicy::Any || is_public(ip)
}

/// Whether the address is public. An IPv4 address in IPv6 form counts as the
/// IPv4 address.
fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => is_public_v4(v4),
            None => is_public_v6(v6),
        },
    }
}

/// Whether the IPv4 address is public: outside every special range.
fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [first_octet, second_octet, ..] = ip.octets();
    !(ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        // this network
        || first_octet == 0
        // shared address space
        || (first_octet == 100 && (64..128).contains(&second_octet))
        // IETF protocols
        || (first_octet == 192 && second_octet == 0 && ip.octets()[2] == 0)
        // benchmarking
        || (first_octet == 198 && (18..20).contains(&second_octet))
        // reserved
        || first_octet >= 240)
}

/// Whether the IPv6 address is public: outside every special range.
fn is_public_v6(ip: Ipv6Addr) -> bool {
    let first = ip.segments()[0];
    !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (first & 0xfe00) == 0xfc00 // unique local
        || (first & 0xffc0) == 0xfe80 // link-local
        || first == 0x2001 && ip.segments()[1] == 0x0db8) // documentation
}

/// The address the URL's host is fetched from, checked against the policy:
/// None for a literal address, else the name, port and pinned address.
async fn resolve_and_check_host(
    url: &Url,
    policy: AddressPolicy,
) -> Result<Option<(String, u16, IpAddr)>, String> {
    let port = url.port_or_known_default().ok_or("the URL has no port")?;
    match url.host() {
        Some(Host::Ipv4(ip)) => check_address(IpAddr::V4(ip), policy).map(|_| None),
        Some(Host::Ipv6(ip)) => check_address(IpAddr::V6(ip), policy).map(|_| None),
        Some(Host::Domain(name)) => {
            let addrs: Vec<IpAddr> = tokio::net::lookup_host((name, port))
                .await
                .map_err(|e| format!("cannot resolve {name}: {e}"))?
                .map(|a| a.ip())
                .collect();
            let Some(&first) = addrs.first() else {
                return Err(format!("{name} has no address"));
            };
            for &ip in &addrs {
                check_address(ip, policy)?;
            }
            Ok(Some((name.to_string(), port, first)))
        }
        None => Err("the URL has no host".into()),
    }
}

/// Refuses an address that the policy does not allow.
fn check_address(ip: IpAddr, policy: AddressPolicy) -> Result<(), String> {
    if is_address_allowed(ip, policy) {
        Ok(())
    } else {
        Err(format!(
            "{ip} is not a public address; pages on local and private networks are not fetched"
        ))
    }
}

/// Whether the URL's host is one of the domains or inside one. Any host is
/// when no domain is given.
fn is_in_domains(url: &Url, domains: &[String]) -> bool {
    domains.is_empty()
        || url.host_str().is_some_and(|host| {
            let host = host.to_ascii_lowercase();
            domains.iter().any(|domain| {
                let domain = domain.trim().trim_start_matches("*.").to_ascii_lowercase();
                host == domain || host.ends_with(&format!(".{domain}"))
            })
        })
}

/// A response that curl fetched.
struct FetchedResponse {
    /// The HTTP status.
    status: u16,
    /// The headers, with their names in lower case.
    headers: Vec<(String, String)>,
    /// The body, at most [`MAX_PAGE_BYTES`].
    body: Vec<u8>,
    /// Whether the body was cut at [`MAX_PAGE_BYTES`].
    is_cut: bool,
}

impl FetchedResponse {
    /// Returns the value of a header. The name must be in lower case.
    fn find_header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

/// One GET with curl: no redirect followed, the body read up to
/// [`MAX_PAGE_BYTES`].
/// curl connects to the pinned address when one is given.
async fn get_with_curl(
    url: &Url,
    pinned: Option<&(String, u16, IpAddr)>,
    timeout: Duration,
) -> Result<FetchedResponse, String> {
    let mut command = Command::new(CURL_PATH);
    command
        .args([
            "-q",
            "-sS",
            "--globoff",
            "--proto",
            "=http,https",
            "--compressed",
            "--max-time",
        ])
        .arg(timeout.as_secs_f64().to_string())
        .args(["-A", USER_AGENT, "-D", "-"]);
    if let Some((name, port, ip)) = pinned {
        let ip = match ip {
            IpAddr::V6(v6) => format!("[{v6}]"),
            IpAddr::V4(v4) => v4.to_string(),
        };
        command.arg("--resolve").arg(format!("{name}:{port}:{ip}"));
    }
    let mut child = command
        .arg("--")
        .arg(url.as_str())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    let mut stdout = child.stdout.take().ok_or("no curl output")?;
    let mut data = Vec::new();
    // curl's output is read 64 KB at a time. The reason for this size is
    // unknown.
    let mut buf = vec![0u8; 64 * KIB];
    let mut is_cut = false;
    loop {
        let bytes_read = stdout.read(&mut buf).await.map_err(|e| e.to_string())?;
        if bytes_read == 0 {
            break;
        }
        data.extend_from_slice(&buf[..bytes_read]);
        // The output holds the headers before the body: they get 64 KB more.
        // The reason for this size is unknown.
        if data.len() > MAX_PAGE_BYTES + 64 * KIB {
            is_cut = true;
            let _ = child.start_kill();
            break;
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !is_cut && !status.success() {
        let mut err = String::new();
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_string(&mut err).await;
        }
        return Err(format!("the fetch failed: {}", err.trim()));
    }
    parse_curl_output(data, is_cut)
}

/// curl's `-D -` output: the header blocks (one per response, the last one
/// counts), then the body.
fn parse_curl_output(output: Vec<u8>, is_cut: bool) -> Result<FetchedResponse, String> {
    let mut at = 0;
    let mut last: Option<(u16, Vec<(String, String)>)> = None;
    while output[at..].starts_with(b"HTTP/") {
        let Some(end) = output[at..].windows(4).position(|w| w == b"\r\n\r\n") else {
            return Err("the response has no complete header".into());
        };
        let head = String::from_utf8_lossy(&output[at..at + end]).to_string();
        let mut lines = head.split("\r\n");
        let status = lines
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|s| s.parse().ok())
            .ok_or("the response has no status")?;
        let headers = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(n, v)| (n.trim().to_ascii_lowercase(), v.trim().to_string()))
            .collect();
        last = Some((status, headers));
        // 4 bytes: the \r\n\r\n that ends a header.
        at += end + 4;
        // an interim 1xx answer is followed by the real one
        if !(100..200).contains(&status) {
            break;
        }
    }
    let (status, headers) = last.ok_or("the response has no header")?;
    let mut body = output[at..].to_vec();
    body.truncate(MAX_PAGE_BYTES);
    Ok(FetchedResponse {
        status,
        headers,
        body,
        is_cut,
    })
}

/// Fetches a page, following up to [`MAX_REDIRECTS`] redirects, each checked.
/// Each request may take `timeout`.
pub async fn fetch_page(
    url_text: &str,
    policy: AddressPolicy,
    allowed_domains: &[String],
    timeout: Duration,
) -> Result<Page, String> {
    let mut url = Url::parse(url_text).map_err(|e| format!("not a URL: {e}"))?;
    for _ in 0..=MAX_REDIRECTS {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!(
                "only http and https pages are fetched, not {}",
                url.scheme()
            ));
        }
        if !is_in_domains(&url, allowed_domains) {
            return Err(format!(
                "{} is outside the allowed domains",
                url.host_str().unwrap_or("")
            ));
        }
        let pinned = resolve_and_check_host(&url, policy).await?;
        let response = get_with_curl(&url, pinned.as_ref(), timeout).await?;
        if (300..400).contains(&response.status) {
            let Some(location) = response.find_header("location") else {
                return Err(format!("HTTP {} without a location", response.status));
            };
            url = url
                .join(location)
                .map_err(|e| format!("a bad redirect: {e}"))?;
            continue;
        }
        if response.status >= 400 {
            return Err(format!("the server answered HTTP {}", response.status));
        }
        let content_type = response
            .find_header("content-type")
            .unwrap_or("")
            .to_ascii_lowercase();
        let text = String::from_utf8_lossy(&response.body).to_string();
        // A page without a content type is HTML when it starts with a
        // doctype. 15 is the length of "<!doctype html>".
        let is_html = content_type.contains("html")
            || (content_type.is_empty()
                && text
                    .trim_start()
                    .get(..15)
                    .is_some_and(|s| s.to_ascii_lowercase().starts_with("<!doctype html")));
        let mut page = if is_html {
            convert_html_to_page(&url, &text)
        } else if content_type.is_empty()
            || content_type.starts_with("text/")
            || content_type.contains("json")
            || content_type.contains("xml")
        {
            Page {
                url: url.to_string(),
                title: url.to_string(),
                lines: wrap_lines(text.lines()),
                links: Vec::new(),
            }
        } else {
            return Err(format!("not a text page ({content_type})"));
        };
        if response.is_cut {
            page.lines.push(format!(
                "(The page is longer than {} MB; the rest is not shown.)",
                MAX_PAGE_BYTES / MIB
            ));
        }
        return Ok(page);
    }
    Err(format!("more than {MAX_REDIRECTS} redirects"))
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

    /// What each fetch may take in the tests: the setting's default.
    const TEST_TIMEOUT: Duration = Duration::from_secs(30);

    /// Serves the test pages on a local port and returns the port.
    /// /old redirects to /page, which is HTML with a link to /next.
    /// /next is plain text. Other paths are missing.
    async fn serve_test_pages() -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    use tokio::io::AsyncWriteExt;
                    // Each test request fits in one read of 4 KB.
                    let mut buf = vec![0u8; 4 * KIB];
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    let request = String::from_utf8_lossy(&buf[..n]).to_string();
                    let path = request.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let reply = match path.as_str() {
                        "/old" => "HTTP/1.1 302 Found\r\nLocation: /page\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
                        "/page" => {
                            let body = "<html><title>Page</title><p>Hello <a href=\"/next\">next page</a></p></html>";
                            format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                        }
                        "/next" => {
                            let body = "plain text\nsecond line";
                            format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                        }
                        _ => "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
                    };
                    let _ = socket.write_all(reply.as_bytes()).await;
                });
            }
        });
        port
    }

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

    #[test]
    fn only_public_addresses_are_fetched() {
        for blocked in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fd00::1",
            "::ffff:127.0.0.1",
        ] {
            let ip: IpAddr = blocked.parse().unwrap();
            assert!(!is_public(ip), "{blocked}");
            assert!(is_address_allowed(ip, AddressPolicy::Any));
        }
        for open in ["93.184.215.14", "1.1.1.1", "2606:4700::1111"] {
            assert!(is_public(open.parse().unwrap()), "{open}");
        }
        // numeric hosts in other forms parse to the address they name
        let u = Url::parse("http://0x7f.1/").unwrap();
        assert_eq!(u.host(), Some(Host::Ipv4(Ipv4Addr::LOCALHOST)));
    }

    #[test]
    fn curl_output_parses_past_a_1xx_response() {
        let data = b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 301 Moved\r\nLocation: /b\r\nContent-Type: text/html\r\n\r\nbody".to_vec();
        let r = parse_curl_output(data, false).unwrap();
        assert_eq!(
            (r.status, r.find_header("location"), r.body.as_slice()),
            (301, Some("/b"), b"body".as_slice())
        );
    }

    /// Pages on a local server, through curl: a redirect, a text page, a host
    /// outside the allowed domains and a missing page.
    /// The local server is refused, by its address and by its name, unless
    /// the policy allows any address.
    #[tokio::test]
    async fn pages_are_fetched_through_curl() {
        let port = serve_test_pages().await;
        let url = |path: &str| format!("http://127.0.0.1:{port}{path}");
        let refused = fetch_page(&url("/page"), AddressPolicy::Public, &[], TEST_TIMEOUT).await;
        assert!(refused.unwrap_err().contains("not a public address"));
        let by_name = format!("http://localhost:{port}/page");
        let refused = fetch_page(&by_name, AddressPolicy::Public, &[], TEST_TIMEOUT).await;
        assert!(refused.unwrap_err().contains("not a public address"));
        let page = fetch_page(&url("/old"), AddressPolicy::Any, &[], TEST_TIMEOUT)
            .await
            .unwrap();
        assert_eq!((page.url, page.title), (url("/page"), "Page".to_string()));
        assert_eq!(page.lines, ["Hello 【0†next page】"]);
        assert_eq!(page.links, [url("/next")]);
        let text = fetch_page(&url("/next"), AddressPolicy::Any, &[], TEST_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(text.lines, ["plain text", "second line"]);
        let outside = fetch_page(
            &url("/page"),
            AddressPolicy::Any,
            &["example.com".into()],
            TEST_TIMEOUT,
        )
        .await;
        assert!(outside.unwrap_err().contains("outside the allowed domains"));
        let missing = fetch_page(&url("/missing"), AddressPolicy::Any, &[], TEST_TIMEOUT).await;
        assert!(missing.unwrap_err().contains("HTTP 404"));
    }
}
