//! The worker's engine thread. It starts the engine and registers the
//! configured models without loading their weights. A stop request ends it.
//! Set EREMITE_CHECKPOINT to a DeepSeek V4.1 checkpoint's directory, and
//! EREMITE_EXPERTS to GGUF experts if wanted. Without EREMITE_CHECKPOINT, the
//! worker starts without models.
use eremite_server::{config, engine, worker};

#[test]
fn the_worker_registers_its_models_and_stops() {
    let models: Vec<worker::ModelSettings> = std::env::var("EREMITE_CHECKPOINT")
        .ok()
        .map(|checkpoint| {
            let mut options = engine::Options {
                model: checkpoint,
                ctx: 4096,
                ..Default::default()
            };
            options.deepseek_mut().unwrap().experts = std::env::var("EREMITE_EXPERTS").ok();
            worker::ModelSettings {
                id: "deepseek".into(),
                options,
                idle_unload: 0.0,
            }
        })
        .into_iter()
        .collect();
    assert!(
        worker::start(
            models.clone(),
            config::Budgets::default(),
            None,
            None,
            None,
            worker::ViewOptions::default()
        ),
        "the worker starts"
    );
    let listed = worker::list_models();
    assert_eq!(listed.len(), models.len());
    for ((id, tokenizer, settings), configured) in listed.iter().zip(&models) {
        assert_eq!((id, settings), (&configured.id, configured));
        assert!(
            !tokenizer.tokenize("Hello").is_empty(),
            "the model's tokenizer works"
        );
    }
    worker::request_stop();
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(worker::wait_for_stop());
    worker::stop();
    assert!(engine::is_stopping());
}
