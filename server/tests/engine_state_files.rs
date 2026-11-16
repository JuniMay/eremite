//! A state's rows written into files and read back into a new state, in two
//! parts and with the rolling rows' snapshot, continue as the original does.
//! The identity of the saved states stays the same. Set EREMITE_CHECKPOINT to
//! a DeepSeek V4.1 checkpoint's directory, and EREMITE_EXPERTS to GGUF experts
//! if wanted; without EREMITE_CHECKPOINT, the test is skipped.
use std::os::fd::AsRawFd;
use std::path::PathBuf;

use eremite_server::engine::{self, Logits, Model, Sampler, StateHandle, Stream};

/// The tokens that each state decodes after the prompt.
const DECODED_TOKENS: usize = 8;

/// A file for one test. It is removed when the test ends.
struct TestFile(PathBuf);

impl TestFile {
    /// Creates an empty file named after `name` and the process, open for
    /// reading and writing.
    fn create(name: &str) -> (TestFile, std::fs::File) {
        let path =
            std::env::temp_dir().join(format!("eremite-state-files-{name}-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        (TestFile(path), file)
    }
}

impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs the stream's posted call to its end.
fn run(stream: &Stream) {
    assert_eq!(engine::run_streams(&[stream]), Some(0), "the call finishes");
}

/// Decodes DECODED_TOKENS tokens greedily in `state`, which continues at
/// position `pos` with its first token sampled from `first`. Returns the
/// tokens.
fn decode(
    model: &Model,
    stream: &Stream,
    state: StateHandle,
    pos: usize,
    first: Logits,
) -> Vec<i32> {
    let mut greedy = Sampler::new(0.0, 1.0, 0, None).expect("a greedy sampler");
    let mut tokens = vec![model.sample(first, &mut greedy)];
    for step in 0..DECODED_TOKENS - 1 {
        stream.decode_rows(&[state], &tokens[step..=step], &[(pos + step) as i32]);
        run(stream);
        tokens.push(model.sample(stream.logits().unwrap(), &mut greedy));
    }
    tokens
}

#[test]
fn a_state_read_from_files_continues_as_the_original() {
    let Ok(checkpoint) = std::env::var("EREMITE_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_CHECKPOINT");
        return;
    };
    // The thread that starts the engine becomes the engine's. It needs a 16 MiB
    // stack.
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(move || {
            assert!(engine::init(
                &engine::ProcessSettings::default(),
                None,
                None,
                None
            ));
            let mut options = engine::Options {
                model: checkpoint,
                ctx: 4096,
                ..Default::default()
            };
            options.deepseek_mut().unwrap().experts = std::env::var("EREMITE_EXPERTS").ok();
            let model = Model::open(&options).expect("the checkpoint opens");
            let identity = model.state_identity();
            assert_ne!(identity, [0; 32]);
            model.load().expect("the model loads");
            assert_eq!(
                model.state_identity(),
                identity,
                "the identity is known before the load"
            );
            let tokenizer = model.tokenizer();
            let mut prompt = tokenizer.tokenize("<｜begin▁of▁sentence｜>");
            prompt.extend(tokenizer.tokenize(
                "The capital of France is Paris. The capital of Germany is Berlin. \
                 The capital of Spain is Madrid. The capital of Italy is",
            ));
            let (n, half) = (prompt.len(), prompt.len() / 2);
            let stream = Stream::new(&model);
            let original = model.state_new().unwrap();
            stream.prefill(original, &prompt, 0, false, None);
            run(&stream);
            let logits = stream.logits().unwrap();
            let kept = model.snapshot_take(original, n, Some(logits));

            // Two parts, then the rolling rows at the end.
            let (_first_path, first) = TestFile::create("first");
            let (_second_path, second) = TestFile::create("second");
            let (_rolling_path, rolling) = TestFile::create("rolling");
            let size = model.state_write(original, 0, half, -1).unwrap();
            assert_eq!(
                model.state_write(original, 0, half, first.as_raw_fd()),
                Some(size)
            );
            assert!(
                model
                    .state_write(original, half, n, second.as_raw_fd())
                    .is_some()
            );
            assert!(
                model
                    .snapshot_write(None, original, n, rolling.as_raw_fd())
                    .is_some()
            );
            let want = decode(&model, &stream, original, n, logits);

            let read = model.state_new().unwrap();
            for file in [&first, &second, &rolling] {
                std::io::Seek::rewind(&mut &*file).unwrap();
            }
            assert!(model.state_read(read, n, &[first.as_raw_fd(), second.as_raw_fd()]));
            let snapshot = model
                .snapshot_read(rolling.as_raw_fd())
                .expect("the snapshot reads");
            model.snapshot_restore(read, snapshot);
            model.set_history(read, &prompt, n);
            let first_logits = model.snapshot_logits(kept).unwrap();
            assert_eq!(decode(&model, &stream, read, n, first_logits), want);

            // A part that ends early does not read.
            first.set_len(size / 2).unwrap();
            std::io::Seek::rewind(&mut &first).unwrap();
            let damaged = model.state_new().unwrap();
            assert!(!model.state_read(damaged, half, &[first.as_raw_fd()]));
            for state in [original, read, damaged] {
                model.state_free(state);
            }
            for snapshot in [kept, snapshot] {
                model.snapshot_free(snapshot);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
