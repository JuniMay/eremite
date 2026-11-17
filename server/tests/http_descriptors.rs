//! The server while the process has no file descriptor left. A connection that
//! arrives then ends without an answer, and the log shows its failed accept
//! once: the accept loop does not spin. Once descriptors are free again, the
//! server serves the next connection.
use std::ffi::{CStr, c_char};
use std::fs::File;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpStream};
use std::os::fd::{FromRawFd, RawFd};
use std::sync::Mutex;

use axum::Router;
use axum::routing::get;
use eremite_server::{engine, http};

/// The lines that the log hook received.
static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Adds the logged `line` to LINES.
extern "C" fn keep_line(line: *const c_char) {
    let line = unsafe { CStr::from_ptr(line) }
        .to_string_lossy()
        .into_owned();
    LINES.lock().unwrap().push(line);
}

/// Makes a TCP socket that is not connected yet.
fn make_socket() -> RawFd {
    let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    assert!(fd >= 0, "socket: {}", std::io::Error::last_os_error());
    fd
}

/// Connects `fd` to `port` on this host. It needs no new file descriptor.
fn connect(fd: RawFd, port: u16) -> TcpStream {
    let address = libc::sockaddr_in {
        sin_len: size_of::<libc::sockaddr_in>() as u8,
        sin_family: libc::AF_INET as u8,
        sin_port: port.to_be(),
        sin_addr: libc::in_addr {
            s_addr: u32::from(Ipv4Addr::LOCALHOST).to_be(),
        },
        sin_zero: [0; 8],
    };
    let connected = unsafe {
        libc::connect(
            fd,
            (&address as *const libc::sockaddr_in).cast(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        )
    };
    assert_eq!(connected, 0, "connect: {}", std::io::Error::last_os_error());
    unsafe { TcpStream::from_raw_fd(fd) }
}

/// Sends a request for / and returns the whole answer.
fn get_root(mut stream: TcpStream) -> String {
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut answer = String::new();
    stream.read_to_string(&mut answer).unwrap();
    answer
}

/// Counts the log's lines about failed accepts.
fn count_failed_accepts() -> usize {
    let lines = LINES.lock().unwrap();
    lines.iter().filter(|line| line.contains("accept:")).count()
}

#[test]
fn a_connection_without_a_descriptor_left_ends_and_the_server_serves_on() {
    // The thread that starts the engine becomes the engine's. It needs a 16 MiB
    // stack.
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            assert!(engine::init(
                &engine::ProcessSettings::default(),
                Some(keep_line),
                None,
                None
            ))
        })
        .unwrap()
        .join()
        .unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)))
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = Router::new().route("/", get(|| async { "served" }));
    runtime.spawn(http::serve(
        listener,
        app,
        std::future::pending(),
        http::DEFAULT_LIMITS,
    ));
    assert!(
        get_root(connect(make_socket(), port)).ends_with("served"),
        "the server serves while descriptors are free"
    );
    let (starved, later) = (make_socket(), make_socket());
    // New descriptors must be below the soft limit. Files take every one that
    // is free below it. With a low soft limit, few files fill it.
    let limit = libc::rlimit {
        rlim_cur: 256,
        rlim_max: libc::RLIM_INFINITY,
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
    let mut fillers = Vec::new();
    while let Ok(file) = File::open("/dev/null") {
        fillers.push(file);
    }
    let mut answer = String::new();
    let _ = connect(starved, port).read_to_string(&mut answer);
    assert_eq!(
        answer, "",
        "a connection without a descriptor gets no answer"
    );
    drop(fillers);
    assert!(
        get_root(connect(later, port)).ends_with("served"),
        "the server serves again once descriptors are free"
    );
    assert_eq!(
        count_failed_accepts(),
        1,
        "the log shows the one failed accept"
    );
}
