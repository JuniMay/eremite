//! The server's HTTP connections and their limits.
//! A request's head is its request line and its headers.
//!
//! - A request's head must arrive within the time limit.
//!   A late head closes the connection.
//!
//! The time limit is the http_timeout setting, 60 s unless set.

use std::future::Future;
use std::time::Duration;

use axum::extract::ConnectInfo;
use axum::{Extension, Router};
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::net::TcpListener;

use crate::engine;

/// The limits on connections and requests.
#[derive(Clone, Copy)]
pub struct Limits {
    /// The time limit for a request's head.
    pub timeout: Duration,
}

/// The limits that the server runs with, unless its settings change them.
/// The reason for each value is unknown.
pub const DEFAULT_LIMITS: Limits = Limits {
    timeout: Duration::from_secs(60),
};

/// Serves `app` on `listener` with `limits` until `shutdown` completes.
/// Then it accepts no more connections. It waits for the open ones to end.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    shutdown: impl Future<Output = ()>,
    limits: Limits,
) {
    let graceful_shutdown = GracefulShutdown::new();
    let mut builder = http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(limits.timeout);
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
        let service = TowerToHyperService::new(app.clone().layer(Extension(ConnectInfo(peer))));
        let connection =
            graceful_shutdown.watch(builder.serve_connection(TokioIo::new(stream), service));
        tokio::spawn(async move {
            let _ = connection.await;
        });
    }
    graceful_shutdown.shutdown().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use std::net::SocketAddr;
    use std::time::Instant;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use tokio::sync::oneshot;

    /// Small limits for quick tests: 300 ms.
    const TEST_LIMITS: Limits = Limits {
        timeout: Duration::from_millis(300),
    };

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
