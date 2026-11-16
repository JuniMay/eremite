//! Verify calls keep the tokens that plain decoding gives. With greedy
//! sampling, the tokens that the draft head's drafts and the context-copy
//! proposals let through are the tokens of one decoding call per token. Set
//! EREMITE_CHECKPOINT to a DeepSeek V4.1 checkpoint's directory, and
//! EREMITE_EXPERTS to GGUF experts if wanted; without EREMITE_CHECKPOINT, the
//! test is skipped.
use eremite_server::engine::{self, Lookup, Model, Sampler, StateHandle, Stream};

/// The tokens that each way of decoding gives after the prompt.
const DECODED_TOKENS: usize = 24;
/// The draft head's drafts per pass.
const DRAFTS: i32 = 2;
/// The context-copy proposals per pass at most.
const PROPOSALS: i32 = 4;
/// The context's capacity in tokens.
const CTX: usize = 4096;

/// Runs the stream's posted call to its end.
fn run(stream: &Stream) {
    assert_eq!(engine::run_streams(&[stream]), Some(0), "the call finishes");
}

/// Prefills `prompt` in a new state. Returns the state and the first token,
/// sampled greedily.
fn prefill(model: &Model, stream: &Stream, prompt: &[i32]) -> (StateHandle, i32) {
    let state = model.state_new().unwrap();
    stream.prefill(state, prompt, 0, false, None, None);
    run(stream);
    let mut greedy = Sampler::new(0.0, 1.0, 0, None).unwrap();
    (state, model.sample(stream.logits().unwrap(), &mut greedy))
}

/// Returns DECODED_TOKENS tokens after `prompt`, one decoding call each.
fn decode_plainly(model: &Model, stream: &Stream, prompt: &[i32]) -> Vec<i32> {
    let (state, first) = prefill(model, stream, prompt);
    let mut greedy = Sampler::new(0.0, 1.0, 0, None).unwrap();
    let mut tokens = vec![first];
    while tokens.len() < DECODED_TOKENS {
        let pos = (prompt.len() + tokens.len() - 1) as i32;
        stream.decode_rows(&[state], &tokens[tokens.len() - 1..], &[pos]);
        run(stream);
        tokens.push(model.sample(stream.logits().unwrap(), &mut greedy));
    }
    model.state_free(state);
    tokens
}

/// Returns DECODED_TOKENS tokens after `prompt`, from verify calls. With a
/// lookup, each call verifies the lookup's proposals; without one, the draft
/// head's drafts. Also returns how many drafts the calls kept.
fn decode_by_verifying(
    model: &Model,
    stream: &Stream,
    prompt: &[i32],
    mut lookup: Option<Lookup>,
) -> (Vec<i32>, usize) {
    let (state, first) = prefill(model, stream, prompt);
    let mut greedy = Sampler::new(0.0, 1.0, 0, None).unwrap();
    if let Some(lookup) = lookup.as_mut() {
        assert!(lookup.append(prompt));
    }
    let (mut tokens, mut next, mut kept_drafts) = (Vec::new(), first, 0);
    while tokens.len() < DECODED_TOKENS {
        let pos = prompt.len() + tokens.len();
        let mut rows = vec![next];
        match lookup.as_mut() {
            Some(lookup) => {
                assert!(lookup.append(&[next]));
                let mut proposals = vec![0; PROPOSALS as usize];
                let count = lookup.propose(PROPOSALS as usize, &mut proposals);
                rows.extend_from_slice(&proposals[..count]);
            }
            None => rows.resize(model.drafts_per_pass() + 1, 0),
        }
        stream.verify(state, &mut rows, lookup.is_some(), pos, &mut greedy);
        run(stream);
        let (after, accepted) = stream.next_token();
        if let Some(lookup) = lookup.as_mut() {
            lookup.record_verification(rows.len() - 1, accepted);
            assert!(lookup.append(&rows[1..=accepted]));
        }
        tokens.extend_from_slice(&rows[..=accepted]);
        kept_drafts += accepted;
        next = after;
    }
    tokens.truncate(DECODED_TOKENS);
    model.state_free(state);
    (tokens, kept_drafts)
}

#[test]
fn verify_calls_keep_the_tokens_of_plain_decoding() {
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
                ctx: CTX as i32,
                draft: DRAFTS,
                lookup: PROPOSALS,
                ..Default::default()
            };
            options.deepseek_mut().unwrap().experts = std::env::var("EREMITE_EXPERTS").ok();
            let model = Model::open(&options).expect("the checkpoint opens");
            model.load().expect("the model loads");
            assert_eq!(model.drafts_per_pass(), DRAFTS as usize);
            assert!(model.max_verify() > PROPOSALS as usize);
            let tokenizer = model.tokenizer();
            let mut prompt = tokenizer.tokenize("<｜begin▁of▁sentence｜>");
            prompt.extend(tokenizer.tokenize(
                "The quick brown fox jumps over the lazy dog. The quick brown fox jumps \
                 over the lazy dog. The quick brown fox jumps over the lazy dog. The quick",
            ));
            let stream = Stream::new(&model);
            let want = decode_plainly(&model, &stream, &prompt);
            let (drafted, _) = decode_by_verifying(&model, &stream, &prompt, None);
            assert_eq!(drafted, want, "with the draft head's drafts");
            let lookup = Lookup::new(CTX).unwrap();
            let (proposed, kept) = decode_by_verifying(&model, &stream, &prompt, Some(lookup));
            assert_eq!(proposed, want, "with context-copy proposals");
            assert!(kept > 0, "the repeated sentence's proposals are kept");
        })
        .unwrap()
        .join()
        .unwrap();
}
