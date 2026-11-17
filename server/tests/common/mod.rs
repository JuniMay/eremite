//! What the model tests share: the server they start, with its log, and the
//! HTTP calls they make. A test waits for the server's events, not for times:
//! the server is up once its log shows where it serves, and it is gone once
//! its process has exited.
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

impl Server {
    /// Starts `program` with `args` and the API key `key`, and returns once its
    /// log shows where it serves. Panics if it exits first.
    pub fn start(program: &Path, args: &[String], key: &str, port: u16) -> Server {
        let mut child = Command::new(program)
            .args(args)
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
