//! A state's rows carry over to other states. A copy with a snapshot of the
//! rolling rows continues as the original does. A copy of fewer positions
//! continues the same after a prefill that replays, and after a prefill from a
//! snapshot that the first prefill captured. Set EREMITE_CHECKPOINT to a
//! DeepSeek V4.1 checkpoint's directory, and EREMITE_EXPERTS to GGUF experts
//! if wanted; without EREMITE_CHECKPOINT, the test is skipped.
use eremite_server::engine::{self, Logits, Model, Sampler, StateHandle, Stream};

/// The tokens that each state decodes after the prompt.
const DECODED_TOKENS: usize = 8;

/// Runs the stream's posted call to its end.
fn run(stream: &Stream) {
    assert_eq!(engine::run_streams(&[stream]), Some(0), "the call finishes");
}

/// Returns a copy of one row of logits.
fn copy_row(model: &Model, logits: Logits) -> Vec<f32> {
    // Safety: a call's or a snapshot's logits hold a row of vocab_size floats.
    unsafe { logits.copy_row(model.vocab_size()) }
}

/// Decodes DECODED_TOKENS tokens greedily in `state`, which continues at
/// position `pos` with `first` sampled from its logits. Returns the tokens
/// and the logits of the first decoding call.
fn decode(
    model: &Model,
    stream: &Stream,
    state: StateHandle,
    pos: usize,
    first: Logits,
) -> (Vec<i32>, Vec<f32>) {
    let mut greedy = Sampler::new(0.0, 1.0, 0, None).expect("a greedy sampler");
    let mut token = model.sample(first, &mut greedy);
    let mut tokens = vec![token];
    let mut first_call_logits = Vec::new();
    for step in 0..DECODED_TOKENS - 1 {
        stream.decode_rows(&[state], &[token], &[(pos + step) as i32]);
        run(stream);
        let logits = stream.logits().expect("the decoding call's logits");
        if step == 0 {
            first_call_logits = copy_row(model, logits);
        }
        token = model.sample(logits, &mut greedy);
        tokens.push(token);
    }
    (tokens, first_call_logits)
}

#[test]
fn a_state_s_rows_carry_over_to_other_states() {
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
            model.load().expect("the model loads");
            let tokenizer = model.tokenizer();
            let mut prompt = tokenizer.tokenize("<｜begin▁of▁sentence｜>");
            prompt.extend(tokenizer.tokenize(
                "The capital of France is Paris. The capital of Germany is Berlin. \
                 The capital of Spain is Madrid. The capital of Italy is",
            ));
            let (n, half) = (prompt.len(), prompt.len() / 2);
            let stream = Stream::new(&model);

            // The original: the whole prompt, with a snapshot captured at half
            // of it and one taken at its end.
            let original = model.state_new().unwrap();
            stream.prefill(original, &prompt, 0, false, Some(half), None);
            run(&stream);
            let info = stream.prefill_info();
            assert_eq!(info.tokens as usize, n, "the prefill ran every row");
            assert_eq!(
                info.prefix_tokens as usize, half,
                "the captured snapshot's position"
            );
            let at_half = stream
                .captured_snapshot()
                .expect("a snapshot at half the prompt");
            let logits = stream.logits().expect("the prefill's logits");
            let at_end = model.snapshot_take(original, n, Some(logits));
            let kept = model
                .snapshot_logits(at_end)
                .expect("the snapshot keeps the logits");
            assert_eq!(copy_row(&model, kept), copy_row(&model, logits));
            let (want, want_first_call) = decode(&model, &stream, original, n, logits);
            let text: Vec<u8> = want
                .iter()
                .flat_map(|&t| tokenizer.token_bytes(t).iter().copied())
                .collect();
            assert!(
                String::from_utf8_lossy(&text).contains("Rome"),
                "{:?}",
                String::from_utf8_lossy(&text)
            );

            // A copy of the whole prompt with the snapshot at its end.
            let copy = model.state_new().unwrap();
            model.state_copy(copy, original, n);
            model.snapshot_restore(copy, at_end);
            model.set_history(copy, &prompt, n);
            let (tokens, first_call) = decode(&model, &stream, copy, n, kept);
            assert_eq!(tokens, want, "the copy continues as the original");
            assert_eq!(first_call, want_first_call, "with the same logits");

            // A copy of half the prompt, then a prefill that replays.
            let replayed = model.state_new().unwrap();
            model.state_copy(replayed, original, half);
            stream.prefill(replayed, &prompt, half, true, None, None);
            run(&stream);
            assert!(stream.prefill_info().tokens as usize >= n - half);
            let logits = stream.logits().unwrap();
            assert_eq!(
                decode(&model, &stream, replayed, n, logits).0,
                want,
                "after a replay"
            );

            // A copy of half the prompt with the captured snapshot, then a
            // prefill of the rest.
            let resumed = model.state_new().unwrap();
            model.state_copy(resumed, original, half);
            model.snapshot_restore(resumed, at_half);
            stream.prefill(resumed, &prompt, half, false, None, None);
            run(&stream);
            let info = stream.prefill_info();
            assert_eq!(info.tokens as usize, n - half, "only the rest");
            assert_eq!(info.prefix_tokens, 0, "no snapshot captured");
            let logits = stream.logits().unwrap();
            assert_eq!(
                decode(&model, &stream, resumed, n, logits).0,
                want,
                "from the snapshot"
            );

            assert!(
                model.replay_rows(half).is_some(),
                "DeepSeek resumes by replays"
            );
            assert!(model.replay_start(half) <= half);
            assert!(model.rewind_margin() < n);
            assert!(model.state_bytes(None, 2 * n) >= model.state_bytes(None, n));
            assert!(model.state_bytes(Some(original), n) > 0);
            assert!(model.snapshot_bytes(true) > model.snapshot_bytes(false));
            for snapshot in [at_half, at_end] {
                model.snapshot_free(snapshot);
            }
            for state in [original, copy, replayed, resumed] {
                model.state_free(state);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
