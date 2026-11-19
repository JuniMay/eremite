//! What the model tests share: the server they start, with its log, the HTTP
//! calls they make, and test images. A test waits for the server's events, not
//! for times: the server is up once its log shows where it serves, and it is
//! gone once its process has exited.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use serde_json::Value;

/// Returns the repository's folder: the folder above server/.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the repository")
        .to_path_buf()
}

/// Held while a test runs servers: one model process at a time.
static MODEL: Mutex<()> = Mutex::new(());

/// Waits until no other test of this binary runs a model process, and returns
/// the guard that keeps the others waiting.
pub fn one_model() -> MutexGuard<'static, ()> {
    MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---- HTTP/1.1, enough for the server's JSON answers ----

/// Sends one request and reads its whole answer. Returns the status and the
/// body, without its chunking.
pub fn http(
    port: u16,
    key: &str,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let sent_body = body.map(|body| body.to_string()).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        sent_body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(sent_body.as_bytes()))
        .expect("send");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read");
    let split = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("an HTTP answer");
    let headers = String::from_utf8_lossy(&raw[..split]).to_lowercase();
    let status = headers
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let mut rest = raw[split + 4..].to_vec();
    if headers.contains("transfer-encoding: chunked") {
        let (mut out, mut at) = (Vec::new(), 0);
        while let Some(end) = rest[at..].windows(2).position(|window| window == b"\r\n") {
            let size =
                usize::from_str_radix(String::from_utf8_lossy(&rest[at..at + end]).trim(), 16)
                    .unwrap_or(0);
            if size == 0 {
                break;
            }
            let start = at + end + 2;
            out.extend_from_slice(&rest[start..start + size]);
            at = start + size + 2;
        }
        rest = out;
    }
    (status, rest)
}

// ---- the server ----

/// The server's log as it arrives.
#[derive(Default)]
struct Log {
    /// The log's lines so far.
    lines: Vec<String>,
    /// Whether the server's output ended.
    ended: bool,
}

/// A server process that a test started. Dropping it stops it.
pub struct Server {
    /// The server's process.
    pub child: Child,
    /// The port that the server listens on.
    pub port: u16,
    /// The API key that the server takes.
    pub key: String,
    /// The server's log, and the condition that a new line or the log's end
    /// notifies.
    log: Arc<(Mutex<Log>, Condvar)>,
}

/// Returns the configuration of a test server with one model: port `port` and
/// the server-wide settings `server`, then the checkpoint `checkpoint`, with
/// the GGUF `experts` if given, and the model's own settings `model`. `server`
/// and `model` are JSON objects of a configuration file's keys.
pub fn build_config(
    port: u16,
    server: Value,
    checkpoint: &str,
    experts: Option<&str>,
    model: Value,
) -> Value {
    let mut entry = serde_json::json!({ "model": checkpoint });
    if let Some(experts) = experts {
        entry["experts"] = experts.into();
    }
    add_keys(&mut entry, model);
    let mut config = serde_json::json!({ "port": port });
    add_keys(&mut config, server);
    config["models"] = serde_json::json!([entry]);
    config
}

/// Adds the keys of the JSON object `from` to the object `to`, in place of
/// those it holds already.
pub fn add_keys(to: &mut Value, from: Value) {
    if let (Some(to), Value::Object(from)) = (to.as_object_mut(), from) {
        to.extend(from);
    }
}

/// A configuration file for one server's start. It is removed when dropped.
struct ConfigFile(PathBuf);

impl Drop for ConfigFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Server {
    /// Starts `program` with the configuration `config`, whose port is `port`,
    /// and the API key `key`, and returns once its log shows where it serves.
    /// Panics if it exits first. The configuration's file is removed then.
    pub fn start(program: &Path, config: &Value, key: &str, port: u16) -> Server {
        let file = ConfigFile(
            std::env::temp_dir().join(format!("eremite-test-{}-{port}.json", std::process::id())),
        );
        std::fs::write(&file.0, config.to_string()).expect("write the configuration");
        let mut child = Command::new(program)
            .arg("--config")
            .arg(&file.0)
            // Keep test servers out of the user's default search folder.
            .env_remove("HOME")
            .env("EREMITE_API_KEY", key)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the server");
        let log = Arc::new((Mutex::new(Log::default()), Condvar::new()));
        let output = child.stderr.take().expect("the server's log");
        let writer = log.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let Ok(line) = line else { break };
                let (log, arrived) = &*writer;
                log.lock().unwrap().lines.push(line);
                arrived.notify_all();
            }
            let (log, arrived) = &*writer;
            log.lock().unwrap().ended = true;
            arrived.notify_all();
        });
        let server = Server {
            child,
            port,
            key: key.to_string(),
            log,
        };
        let serving = format!("at http://127.0.0.1:{port}/v1");
        server.wait_for_line(0, |line| line.contains(&serving));
        server
    }

    /// Counts the log's lines so far.
    pub fn count_lines(&self) -> usize {
        self.log.0.lock().unwrap().lines.len()
    }

    /// Waits for a log line from line `from` on that `matches` accepts, and
    /// returns it. Panics if the log ends first.
    pub fn wait_for_line(&self, from: usize, matches: impl Fn(&str) -> bool) -> String {
        let (log, arrived) = &*self.log;
        let mut log = log.lock().unwrap();
        let mut next = from;
        loop {
            if let Some(line) = log.lines[next.min(log.lines.len())..]
                .iter()
                .find(|line| matches(line))
            {
                return line.clone();
            }
            next = log.lines.len();
            if log.ended {
                let tail = log.lines[log.lines.len().saturating_sub(5)..].join("\n");
                panic!("the server's log ended before the line came; its last lines:\n{tail}");
            }
            log = arrived.wait(log).unwrap();
        }
    }

    /// Sends GET `path` and returns the status and the body.
    pub fn get(&self, path: &str) -> (u16, Vec<u8>) {
        http(self.port, &self.key, "GET", path, None)
    }

    /// Reads the served model's status: GET /v1/models/{id}'s "status".
    pub fn status(&self) -> Value {
        let models: Value =
            serde_json::from_slice(&self.get("/v1/models").1).unwrap_or(Value::Null);
        let id = models["data"][0]["id"].as_str().unwrap_or("").to_string();
        serde_json::from_slice::<Value>(&self.get(&format!("/v1/models/{id}")).1)
            .map(|object| object["status"].clone())
            .unwrap_or(Value::Null)
    }

    /// Reads the status once `requests` requests have ended and the server is
    /// idle. Each status read waits for the worker to publish, so the reads
    /// follow the worker's turns.
    pub fn settled(&self, requests: i64) -> Value {
        loop {
            let status = self.status();
            if status["phase"] == "idle"
                && status["totals"]["requests"].as_i64().unwrap_or(0) >= requests
            {
                return status;
            }
        }
    }

    /// Stops it as SIGTERM does, and waits for it to exit (it saves its states
    /// first).
    pub fn stop(mut self) {
        self.terminate();
    }

    /// Sends SIGTERM to the server, unless it exited, and waits for it to exit.
    fn terminate(&mut self) {
        if let Ok(Some(_)) = self.child.try_wait() {
            return;
        }
        let _ = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .stderr(Stdio::null())
            .status();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    /// Stops the server. When the test fails, the server's last log lines go to
    /// stderr first.
    fn drop(&mut self) {
        if std::thread::panicking() {
            let log = self
                .log
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let tail = &log.lines[log.lines.len().saturating_sub(20)..];
            eprintln!("the server's last log lines:\n{}", tail.join("\n"));
        }
        self.terminate();
    }
}

// ---- test images ----

/// Returns a BMP file of the test pattern at width x height pixels: 24-bit
/// pixels, red rising from left to right, green from top to bottom, and blue
/// a texture of sharp edges, as the engine's tests make it.
pub fn make_test_bmp(width: usize, height: usize) -> Vec<u8> {
    let row = (width * 3).div_ceil(4) * 4;
    let data = row * height;
    let mut f = vec![0u8; 54 + data];
    let mut put = |at: usize, value: u32, bytes: usize| {
        f[at..at + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
    };
    put(0, u32::from_le_bytes(*b"BM\0\0"), 2);
    put(2, (54 + data) as u32, 4);
    put(10, 54, 4);
    put(14, 40, 4);
    put(18, width as u32, 4);
    put(22, height as u32, 4);
    put(26, 1, 2);
    put(28, 24, 2);
    put(34, data as u32, 4);
    // 2835 pixels per meter: 72 per inch.
    put(38, 2835, 4);
    put(42, 2835, 4);
    for y in 0..height {
        // The rows go from the bottom up, each pixel blue, green, red.
        let line = 54 + (height - 1 - y) * row;
        for x in 0..width {
            f[line + 3 * x] = ((x / 7) * 37 + (y / 5) * 91 + (x * y) % 13 * 11) as u8;
            f[line + 3 * x + 1] = (y * 255 / (height - 1)) as u8;
            f[line + 3 * x + 2] = (x * 255 / (width - 1)) as u8;
        }
    }
    f
}

/// The base64 alphabet of data URLs.
const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Returns `bytes` in base64, padded with = to a multiple of 4 characters.
pub fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() {
                BASE64[(n >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}
