//! The server's HTTP connections and their limits.
//! A request's head is its request line and its headers.
//!
//! - At most 64 connections are open at once.
//!   The server answers another connection with 503.
//!   Then it closes that connection.
//! - A request's head must arrive within the time limit.
//!   A late head closes the connection.
//! - A request's head may be at most 64 KB. A larger head gets 431.
//!
//! The time limit is the http_timeout setting, 60 s unless set.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::{Extension, Router};
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

use crate::{chat, engine};

/// 1 KB is 1024 bytes here.
pub const KIB: usize = 1 << 10;
/// The size in bytes of the buffer that reads a refused connection's data.
/// The reason for this value is unknown.
const REFUSED_READ_BUFFER_BYTES: usize = 4096;

/// The limits on connections and requests.
#[derive(Clone, Copy)]
pub struct Limits {
    /// The most connections open at once.
    /// The server answers another one with 503.
    pub max_connections: usize,
    /// The time limit for a request's head.
    pub timeout: Duration,
    /// The largest request head in bytes. A larger head gets 431.
    /// hyper's buffer for a head grows to this size and at most one read more.
    /// That bounds the buffer's memory.
    pub max_head_bytes: usize,
}

/// The limits that the server runs with, unless its settings change them.
/// The reason for each value is unknown.
pub const DEFAULT_LIMITS: Limits = Limits {
    max_connections: 64,
    timeout: Duration::from_secs(60),
    max_head_bytes: 64 * KIB,
};

/// Serves `app` on `listener` with `limits` until `shutdown` completes.
/// Then it accepts no more connections. It waits for the open ones to end.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    shutdown: impl Future<Output = ()>,
    limits: Limits,
) {
    let connection_permits = Arc::new(Semaphore::new(limits.max_connections));
    let graceful_shutdown = GracefulShutdown::new();
    let mut builder = http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(limits.timeout)
        .max_buf_size(limits.max_head_bytes);
    let app = app.layer(middleware::from_fn(
        move |request: Request, next: Next| async move {
            if count_head_bytes(&request) > limits.max_head_bytes {
                return refuse_large_head(&request, limits.max_head_bytes);
            }
            next.run(request).await
        },
    ));
    tokio::pin!(shutdown);
    loop {
        let (stream, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(connection) => connection,
                // When an accept fails because the process has no file
                // descriptor left, macOS closes the waiting connection
                // (tests/http_descriptors.rs). So the next accept waits for a
                // new connection, and the loop does not spin.
                Err(e) => {
                    engine::log(&format!("accept: {e}"));
                    continue;
                }
            },
            () = &mut shutdown => break,
        };
        // Send each write at once.
        // Then each token's chunk reaches the client without delay.
        let _ = stream.set_nodelay(true);
        let Ok(permit) = connection_permits.clone().try_acquire_owned() else {
            engine::log(&format!("{}: 503: too many connections", peer.ip()));
            tokio::spawn(refuse_connection(stream, limits));
            continue;
        };
        let service = TowerToHyperService::new(app.clone().layer(Extension(ConnectInfo(peer))));
        let connection =
            graceful_shutdown.watch(builder.serve_connection(TokioIo::new(stream), service));
        tokio::spawn(async move {
            let _ = connection.await;
            drop(permit);
        });
    }
    graceful_shutdown.shutdown().await;
}

/// Counts the bytes of `request`'s head as hyper parsed it.
/// hyper's buffer can take a head a little larger than the limit.
/// That happens when the head arrives in a few large reads.
fn count_head_bytes(request: &Request) -> usize {
    // A request without a target counts as "/".
    let target = request
        .uri()
        .path_and_query()
        .map_or("/".len(), |path| path.as_str().len());
    let headers: usize = request
        .headers()
        .iter()
        .map(|(name, value)| name.as_str().len() + ": ".len() + value.len() + "\r\n".len())
        .sum();
    request.method().as_str().len()
        + " ".len()
        + target
        + " ".len()
        + "HTTP/1.1\r\n".len()
        + headers
        + "\r\n".len()
}

/// Refuses `request` with 431 because its head is larger than `max_head_bytes`.
/// Logs the client's address with the error.
fn refuse_large_head(request: &Request, max_head_bytes: usize) -> Response {
    let peer = request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|connect_info| connect_info.0.ip().to_string());
    engine::log(&format!(
        "{}: 431: a request's line and headers above {} KB",
        peer.unwrap_or_default(),
        max_head_bytes / KIB
    ));
    let message = format!(
        "the request's line and headers are larger than {} KB",
        max_head_bytes / KIB
    );
    chat::build_error_response(
        StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
        "invalid_request_error",
        &message,
    )
}

/// Refuses a connection beyond the limit with 503, before reading its request.
/// Then it discards what the client sends, until the client closes.
/// It reads at most `limits.max_head_bytes` bytes, for at most
/// `limits.timeout`.
/// Closing with unread data would reset the connection.
/// A reset can discard the answer before the client reads it.
async fn refuse_connection(mut stream: TcpStream, limits: Limits) {
    let body = chat::format_error_json("server_error", "too many connections");
    let head = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let answered = tokio::time::timeout(limits.timeout, async {
        stream.write_all(head.as_bytes()).await?;
        stream.write_all(body.as_bytes()).await?;
        stream.shutdown().await
    })
    .await;
    if !matches!(answered, Ok(Ok(()))) {
        return;
    }
    let _ = tokio::time::timeout(limits.timeout, async {
        let (mut sink, mut left) = ([0u8; REFUSED_READ_BUFFER_BYTES], limits.max_head_bytes);
        while left > 0 {
            match stream.read(&mut sink).await {
                Ok(0) | Err(_) => break,
                Ok(n) => left = left.saturating_sub(n),
            }
        }
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use std::net::SocketAddr;
    use std::time::Instant;
    use tokio::sync::oneshot;

    /// Small limits for quick tests: 2 connections, 300 ms and a 16 KB head.
    const TEST_LIMITS: Limits = Limits {
        max_connections: 2,
        timeout: Duration::from_millis(300),
        max_head_bytes: 16 * KIB,
    };
    /// How long a slow client waits before it reads: long enough for a reset to
    /// arrive first, if the server closed the connection with the request
    /// unread.
    const LATE_READ: Duration = Duration::from_millis(200);

    /// Starts the server on a free local port.
    /// Returns its address and the sender that stops it.
    async fn start_server(app: Router, limits: Limits) -> (SocketAddr, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        tokio::spawn(serve(
            listener,
            app,
            async move {
                let _ = stopped.await;
            },
            limits,
        ));
        (addr, stop)
    }

    /// Builds the test routes: /health.
    fn build_test_app() -> Router {
        Router::new().route("/health", get(|| async { "ok" }))
    }

    /// Reads the server's answer until the connection closes.
    async fn read_answer(stream: &mut TcpStream) -> String {
        let mut out = Vec::new();
        let _ = stream.read_to_end(&mut out).await;
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Sends GET /health on a connection that stays open, and reads its answer.
    async fn ask_health(stream: &mut TcpStream) -> String {
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            .await
            .unwrap();
        let (mut out, mut buffer) = (Vec::new(), [0u8; KIB]);
        while !out.ends_with(b"ok") {
            let n = stream.read(&mut buffer).await.unwrap();
            assert!(n > 0, "the connection closed before its answer");
            out.extend_from_slice(&buffer[..n]);
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    #[tokio::test]
    async fn connections_beyond_the_limit_get_503() {
        let (addr, _stop) = start_server(
            build_test_app(),
            Limits {
                // An idle connection closes after the time limit. A longer
                // limit keeps the first two connections open during the test.
                timeout: Duration::from_secs(5),
                ..TEST_LIMITS
            },
        )
        .await;
        // Two connections that the server has answered hold its two permits.
        let mut first = TcpStream::connect(addr).await.unwrap();
        let mut second = TcpStream::connect(addr).await.unwrap();
        for connection in [&mut first, &mut second] {
            let answer = ask_health(connection).await;
            assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        }
        let mut third = TcpStream::connect(addr).await.unwrap();
        // The client sends its request before it reads.
        // Closing the connection with that request unread would reset the
        // connection.
        // A reset can discard the answer before the client reads it.
        third
            .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            .await
            .unwrap();
        tokio::time::sleep(LATE_READ).await;
        let mut refused = Vec::new();
        let read = third.read_to_end(&mut refused).await;
        assert!(read.is_ok(), "the connection was reset: {read:?}");
        let refused = String::from_utf8_lossy(&refused);
        assert!(refused.starts_with("HTTP/1.1 503"), "{refused}");
        assert!(refused.contains("too many connections"), "{refused}");
        // The server frees the first connection's permit once it sees the
        // close. Until then, a new connection gets 503. The test tries again.
        drop(first);
        let served = loop {
            let mut fourth = TcpStream::connect(addr).await.unwrap();
            fourth
                .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            let answer = read_answer(&mut fourth).await;
            if !answer.starts_with("HTTP/1.1 503") {
                break answer;
            }
        };
        assert!(served.starts_with("HTTP/1.1 200"), "{served}");
    }

    #[tokio::test]
    async fn a_head_above_the_limit_gets_431() {
        let (addr, _stop) = start_server(build_test_app(), TEST_LIMITS).await;
        // Under, just over and far over the 16 KB head limit.
        for (pad, status) in [
            (15 * KIB, "HTTP/1.1 200"),
            (17 * KIB, "HTTP/1.1 431"),
            (40 * KIB, "HTTP/1.1 431"),
        ] {
            let mut client = TcpStream::connect(addr).await.unwrap();
            let head = format!(
                "GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\nX-Pad: {}\r\n\r\n",
                "a".repeat(pad)
            );
            let _ = client.write_all(head.as_bytes()).await;
            let got = read_answer(&mut client).await;
            assert!(got.starts_with(status), "{pad} bytes of padding: {got}");
        }
    }

    #[tokio::test]
    async fn a_late_head_closes_the_connection() {
        let (addr, _stop) = start_server(build_test_app(), TEST_LIMITS).await;
        let mut client = TcpStream::connect(addr).await.unwrap();
        let started = Instant::now();
        client
            .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n")
            .await
            .unwrap();
        let left = read_answer(&mut client).await;
        assert!(left.is_empty(), "{left}");
        assert!(
            started.elapsed() >= TEST_LIMITS.timeout,
            "closed after {:?}, before the time limit",
            started.elapsed()
        );
    }
}
