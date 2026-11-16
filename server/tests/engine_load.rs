//! A checkpoint's weights load and unload. The loaded model knows its end
//! token. Set EREMITE_CHECKPOINT to a DeepSeek V4.1 checkpoint's directory,
//! and EREMITE_EXPERTS to GGUF experts if wanted; without EREMITE_CHECKPOINT,
//! the test is skipped.
use eremite_server::engine::{self, Model};

#[test]
fn a_model_loads_and_unloads() {
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
            let text = model.tokenizer().tokenize("Paris");
            assert!(!model.is_stop_token(text[0]));
            // An idle model: its weights may go, and no GPU work or read may
            // still use the memory that the unload gives back.
            model.set_request_running(false);
            engine::drain();
            assert!(model.unload() > 0, "unloading gives bytes back");
            assert!(!model.is_loaded());
            model.load().expect("the model loads again");
            assert!(model.is_loaded());
        })
        .unwrap()
        .join()
        .unwrap();
}
