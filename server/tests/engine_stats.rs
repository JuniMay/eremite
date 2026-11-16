//! A model's counters and the process's memory: decoding counts its rows,
//! the memory limit can be set, idle memory can be given back, and a wake
//! makes run_streams return before its call finishes. Set EREMITE_CHECKPOINT
//! to a DeepSeek V4.1 checkpoint's directory, and EREMITE_EXPERTS to GGUF
//! experts if wanted; without EREMITE_CHECKPOINT, the test is skipped.
use eremite_server::engine::{self, Model, Sampler, Stream};

/// The tokens decoded after the prompt.
const DECODED_TOKENS: usize = 4;

#[test]
fn counters_and_memory_follow_the_model() {
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
            engine::refresh_stats();
            let before = model.read_stats();
            assert!(
                before.layers > 0 && before.experts_per_row > 0,
                "{before:?}"
            );
            assert!(before.model_memory_used > 0);
            let memory = engine::read_memory();
            assert!(memory.model_bytes > 0 && memory.limit >= memory.model_bytes);

            model.set_request_running(true);
            let tokenizer = model.tokenizer();
            let prompt = tokenizer.tokenize("<｜begin▁of▁sentence｜>One, two, three,");
            let state = model.state_new().unwrap();
            let stream = Stream::new(&model);
            stream.prefill(state, &prompt, 0, false, None, None);
            assert_eq!(engine::run_streams(&[&stream]), Some(0));
            let mut greedy = Sampler::new(0.0, 1.0, 0, None).unwrap();
            let mut token = model.sample(stream.logits().unwrap(), &mut greedy);
            for step in 0..DECODED_TOKENS {
                stream.decode_rows(&[state], &[token], &[(prompt.len() + step) as i32]);
                if step == 0 {
                    // A wake before the run ends the run after one look.
                    engine::wake();
                    assert_eq!(
                        engine::run_streams(&[&stream]),
                        None,
                        "the wake ends the run"
                    );
                }
                assert_eq!(engine::run_streams(&[&stream]), Some(0));
                token = model.sample(stream.logits().unwrap(), &mut greedy);
            }
            engine::refresh_stats();
            let after = model.read_stats();
            assert_eq!(
                after.decode_rows - before.decode_rows,
                DECODED_TOKENS as u64
            );
            assert!(after.prefill_rows - before.prefill_rows >= prompt.len() as u64);
            assert!(after.decode_seconds > before.decode_seconds);
            model.state_free(state);
            model.set_request_running(false);

            engine::drain();
            assert!(model.run_idle_work() >= 0);
            let limit = engine::read_memory().limit;
            engine::set_memory_limit(limit - (1 << 30));
            assert_eq!(engine::read_memory().limit, limit - (1 << 30));
            engine::set_memory_limit(limit);
            assert!(
                engine::reclaim(1 << 20) <= memory.model_bytes,
                "it frees what it holds at most"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
