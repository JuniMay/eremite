//! A loaded model continues a prompt: the prompt is prefilled in a state,
//! then each decoding call gives one token, for one state alone and for two
//! states in one call. Set EREMITE_CHECKPOINT to a DeepSeek V4.1 checkpoint's
//! directory, and EREMITE_EXPERTS to GGUF experts if wanted; without
//! EREMITE_CHECKPOINT, the test is skipped.
use eremite_server::engine::{self, Model, Sampler, StateHandle, Stream};

/// The tokens that each test decodes after the prompt.
const DECODED_TOKENS: usize = 8;

/// Runs the stream's posted call to its end.
fn run(stream: &Stream) {
    assert_eq!(engine::run_streams(&[stream]), Some(0), "the call finishes");
}

/// Prefills `prompt` in `states`, then decodes DECODED_TOKENS tokens for all
/// of them in one call per token, greedily. Returns each state's text.
fn continue_prompt(model: &Model, states: &[StateHandle], prompt: &[i32]) -> Vec<String> {
    let stream = Stream::new(model);
    let mut greedy = Sampler::new(0.0, 1.0, 0, None).expect("a greedy sampler");
    let vocab = model.vocab_size();
    let mut tokens = Vec::new();
    for &state in states {
        stream.prefill(state, prompt, 0, false, None);
        run(&stream);
        let logits = stream.logits().expect("the prefill's logits");
        tokens.push(model.sample(logits, &mut greedy));
    }
    let mut texts = vec![Vec::new(); states.len()];
    for step in 0..DECODED_TOKENS {
        for (text, &token) in texts.iter_mut().zip(&tokens) {
            text.extend_from_slice(model.tokenizer().token_bytes(token));
        }
        let pos = vec![(prompt.len() + step) as i32; states.len()];
        stream.decode_rows(states, &tokens, &pos);
        run(&stream);
        let logits = stream.logits().expect("the decoding call's logits");
        for (row, token) in tokens.iter_mut().enumerate() {
            // Safety: the call's logits hold one row for each state.
            *token = model.sample(unsafe { logits.row(row, vocab) }, &mut greedy);
        }
    }
    texts
        .into_iter()
        .map(|text| String::from_utf8_lossy(&text).into_owned())
        .collect()
}

#[test]
fn a_loaded_model_continues_a_prompt() {
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
                model: checkpoint.clone(),
                ctx: 4096,
                ..Default::default()
            };
            options.deepseek_mut().unwrap().experts = std::env::var("EREMITE_EXPERTS").ok();
            let model = Model::open(&options).expect("the checkpoint opens");
            assert!(!model.is_loaded());
            model.load().expect("the model loads");
            assert!(model.is_loaded());
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(format!("{checkpoint}/config.json")).unwrap(),
            )
            .unwrap();
            let end = config["eos_token_id"].as_i64().unwrap() as i32;
            assert!(
                model.is_stop_token(end),
                "config.json's eos_token_id ends a reply"
            );
            let tokenizer = model.tokenizer();
            let mut prompt = tokenizer.tokenize("<｜begin▁of▁sentence｜>");
            prompt.extend(tokenizer.tokenize("The capital of France is"));
            assert!(!model.is_stop_token(prompt[1]));

            let alone = model.state_new().expect("a state");
            let text = continue_prompt(&model, &[alone], &prompt).remove(0);
            assert!(text.contains("Paris"), "one state: {text:?}");
            let pair = [model.state_new().unwrap(), model.state_new().unwrap()];
            for (row, text) in continue_prompt(&model, &pair, &prompt).iter().enumerate() {
                assert!(text.contains("Paris"), "row {row} of two: {text:?}");
            }
            for state in [alone, pair[0], pair[1]] {
                model.state_free(state);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
