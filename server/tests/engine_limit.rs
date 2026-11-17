//! A model's own memory limit: the model loads and runs within it, its expert
//! cache included, and its statistics report the limit, what the model holds
//! and what a full prompt pass needs. Set EREMITE_CHECKPOINT to a DeepSeek V4.1
//! checkpoint's directory, and EREMITE_EXPERTS to GGUF experts if wanted;
//! without EREMITE_CHECKPOINT, the test is skipped.
use eremite_server::engine::{self, Model, Sampler, Stream};

/// The model's own limit in GiB: room for DeepSeek V4.1 Flash's weights and a
/// full prompt pass with GGUF experts, its workspace and its smallest expert
/// cache (about 24 GiB), and for a larger expert cache and its states. Without
/// the limit, its expert cache grows into the free memory above the reserve.
const LIMIT_GIB: f64 = 32.0;
/// The bytes in one GiB.
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// The tokens decoded after the prompt.
const DECODED_TOKENS: usize = 16;

#[test]
fn a_model_runs_within_its_own_memory_limit() {
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
                memory_gb: LIMIT_GIB,
                ..Default::default()
            };
            options.deepseek_mut().unwrap().experts = std::env::var("EREMITE_EXPERTS").ok();
            let model = Model::open(&options).expect("the checkpoint opens");
            model.load().expect("the model loads within its limit");
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
                assert_eq!(engine::run_streams(&[&stream]), Some(0));
                token = model.sample(stream.logits().unwrap(), &mut greedy);
            }
            engine::refresh_stats();
            let stats = model.read_stats();
            let limit = (LIMIT_GIB * GIB) as u64;
            assert_eq!(stats.own_memory_limit, limit);
            assert!(
                stats.required_bytes > 0 && stats.required_bytes <= limit,
                "a full prompt pass needs {:.2} GiB",
                stats.required_bytes as f64 / GIB
            );
            assert!(
                stats.own_memory_counted <= limit,
                "the model holds {:.2} GiB",
                stats.own_memory_counted as f64 / GIB
            );
            assert!(stats.own_state_bytes > 0, "the state counts in the model");
            println!(
                "within {LIMIT_GIB} GiB: a full prompt pass needs {:.2} GiB; the model holds {:.2} GiB, \
                 {:.2} GiB of experts in {} slots",
                stats.required_bytes as f64 / GIB,
                stats.own_memory_counted as f64 / GIB,
                stats.cache_bytes as f64 / GIB,
                stats.slots
            );
            model.state_free(state);
            model.set_request_running(false);
        })
        .unwrap()
        .join()
        .unwrap();
}
