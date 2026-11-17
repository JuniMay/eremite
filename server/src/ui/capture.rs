//! The lines written to a descriptor, such as stderr's, kept in the terminal
//! UI's log while the view is on the screen. Text that anything writes to
//! the terminal then would break the view's frames. A pipe takes the
//! descriptor's place, and a thread hands each line from it to the log.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::thread::JoinHandleExt;
use std::sync::mpsc;
use std::thread;

/// The line that the release writes last into the pipe. A line of text holds
/// no NUL byte, so no other line is the same.
const RELEASE_MARK: &[u8] = b"\0release\n";

/// A descriptor whose lines go to the log until the release.
pub struct Capture {
    /// The descriptor that the pipe took over.
    descriptor: RawFd,
    /// What the descriptor was before, for the release to give back.
    original: OwnedFd,
    /// The pipe's end that the release writes the mark to.
    pipe_end: File,
    /// Receives a message once the reading thread has read the mark. It
    /// closes when that thread ends.
    marked: mpsc::Receiver<()>,
    /// The reading thread's pthread id.
    reader: libc::pthread_t,
}

/// Returns the calling thread's pthread id. It needs no thread-local data,
/// which may be gone at the process's exit.
pub fn find_thread_id() -> libc::pthread_t {
    // Safety: pthread_self only returns the calling thread's id.
    unsafe { libc::pthread_self() }
}

/// Points `descriptor` at a pipe, and starts a thread that hands each line
/// written to it to `keep_line`. Returns the capture, and a copy of what
/// `descriptor` was, for the view to draw on.
pub fn capture_lines(
    descriptor: RawFd,
    keep_line: impl Fn(&str) + Send + 'static,
) -> io::Result<(Capture, File)> {
    // Safety: the caller's descriptor is open. The copies are new
    // descriptors, owned here.
    let original = unsafe { BorrowedFd::borrow_raw(descriptor) }.try_clone_to_owned()?;
    let for_drawing = File::from(original.try_clone()?);
    let for_later_lines = File::from(original.try_clone()?);
    let mut ends = [0; 2];
    // Safety: pipe fills both descriptors when it succeeds.
    if unsafe { libc::pipe(ends.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // Safety: both descriptors are new, and owned here alone.
    let (read_end, write_end) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
    for end in [&read_end, &write_end] {
        // Safety: a descriptor owned here. A program that this one starts
        // gets neither end.
        unsafe { libc::fcntl(end.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) };
    }
    let (sender, marked) = mpsc::channel();
    let reader = thread::Builder::new()
        .name("capture".into())
        .spawn(move || read_lines(File::from(read_end), keep_line, sender, for_later_lines))?;
    // Safety: the descriptor becomes a copy of the pipe's write end.
    if unsafe { libc::dup2(write_end.as_raw_fd(), descriptor) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let capture = Capture {
        descriptor,
        original,
        pipe_end: File::from(write_end),
        marked,
        reader: reader.as_pthread_t(),
    };
    Ok((capture, for_drawing))
}

impl Capture {
    /// Gives the descriptor back what it was, and returns once the reading
    /// thread has kept every line written to the pipe before. Of a line
    /// without its end yet, the part written so far is kept. A program that
    /// this one started may still write to the pipe; the reading thread
    /// passes that on to the original. On the reading thread itself, it
    /// returns at once.
    pub fn release(mut self) {
        // Safety: the descriptor becomes a copy of the original again.
        unsafe { libc::dup2(self.original.as_raw_fd(), self.descriptor) };
        if find_thread_id() == self.reader {
            return;
        }
        // An error means that the reading thread has ended.
        if self.pipe_end.write_all(RELEASE_MARK).is_ok() {
            drop(self.pipe_end);
            let _ = self.marked.recv();
        }
    }
}

/// Reads `pipe` line by line and hands each line's text to `keep_line`, up to
/// the release mark. Then it tells `marked`, and writes the rest of what it
/// reads to `later`.
fn read_lines(pipe: File, keep_line: impl Fn(&str), marked: mpsc::Sender<()>, mut later: File) {
    let mut reader = BufReader::new(pipe);
    let mut line = Vec::new();
    loop {
        line.clear();
        if !matches!(reader.read_until(b'\n', &mut line), Ok(read) if read > 0) {
            return;
        }
        let is_mark = line.ends_with(RELEASE_MARK);
        if is_mark {
            line.truncate(line.len() - RELEASE_MARK.len());
        }
        if let Some(text) = clean_line(&line) {
            keep_line(&text);
        }
        if is_mark {
            break;
        }
    }
    let _ = marked.send(());
    let _ = io::copy(&mut reader, &mut later);
}

/// Returns the text of `line` as a terminal shows it: the part after its last
/// carriage return, with each control character as a space. Returns None for
/// a line without text.
fn clean_line(line: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(line);
    let shown = text
        .trim_end_matches(['\n', '\r'])
        .rsplit('\r')
        .next()
        .unwrap_or("");
    let cleaned: String = shown
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect();
    let cleaned = cleaned.trim_end();
    (!cleaned.trim_start().is_empty()).then(|| cleaned.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// Returns a new pipe's read end and write end.
    fn make_pipe() -> (File, File) {
        let mut ends = [0; 2];
        // Safety: pipe fills both descriptors when it succeeds.
        assert_eq!(unsafe { libc::pipe(ends.as_mut_ptr()) }, 0);
        // Safety: both descriptors are new, and owned by the files alone.
        unsafe { (File::from_raw_fd(ends[0]), File::from_raw_fd(ends[1])) }
    }

    /// Writes `bytes` to `descriptor` in one write.
    fn write_to(descriptor: RawFd, bytes: &[u8]) {
        // Safety: the test's own open descriptor and bytes.
        let written = unsafe { libc::write(descriptor, bytes.as_ptr().cast(), bytes.len()) };
        assert_eq!(written, bytes.len() as isize);
    }

    #[test]
    fn a_descriptors_lines_go_to_the_log_until_the_release() {
        // A pipe stands for the terminal, and a copy of its write end for
        // stderr.
        let (mut terminal, terminal_end) = make_pipe();
        // Safety: a copy of the test's own descriptor.
        let descriptor = unsafe { libc::dup(terminal_end.as_raw_fd()) };
        let (sender, kept) = mpsc::channel();
        let keep_line = move |text: &str| sender.send(text.to_string()).unwrap();
        let (capture, for_drawing) = capture_lines(descriptor, keep_line).unwrap();
        write_to(
            descriptor,
            b"first line\nloading 50%\rloading 100%\n\x1b[1mbold\x1b[0m\n\n",
        );
        assert_eq!(kept.recv().unwrap(), "first line");
        assert_eq!(kept.recv().unwrap(), "loading 100%", "the last version");
        assert_eq!(kept.recv().unwrap(), " [1mbold [0m");
        // a program that this one started keeps a copy of the pipe's end
        // Safety: a copy of the test's own descriptor.
        let program = unsafe { libc::dup(descriptor) };
        write_to(descriptor, b"no end yet");
        capture.release();
        assert_eq!(
            kept.recv().unwrap(),
            "no end yet",
            "kept before the release"
        );
        write_to(descriptor, b"after the release\n");
        write_to(program, b"from the program\n");
        let mut shown = [0; 35];
        terminal.read_exact(&mut shown).unwrap();
        assert_eq!(&shown, b"after the release\nfrom the program\n");
        // Safety: the test's own descriptors, closed once.
        unsafe {
            libc::close(program);
            libc::close(descriptor);
        }
        drop(for_drawing);
        assert!(kept.recv().is_err(), "the reading thread ends");
        assert_eq!(clean_line(b" \r\n"), None);
        // each thread has its own id
        assert_eq!(find_thread_id(), find_thread_id());
        assert_ne!(
            find_thread_id(),
            thread::spawn(find_thread_id).join().unwrap()
        );
    }
}
