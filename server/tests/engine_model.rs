//! A checkpoint opened without its weights: its limits and its tokenizer.
//! Set EREMITE_CHECKPOINT to a DeepSeek V4.1 checkpoint's directory; without
//! it, the test is skipped.
use eremite_server::engine;

#[test]
fn an_opened_checkpoint_gives_its_limits_and_its_tokenizer() {
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
                ctx: 0,
                ..Default::default()
            };
            assert!(
                engine::Model::open(&options).is_none(),
                "the engine refuses a context of 0 tokens"
            );
            options.ctx = 4096;
            let model = engine::Model::open(&options).expect("the checkpoint opens");
            assert!(model.max_rows() >= 1, "{} rows", model.max_rows());
            assert_eq!(
                model.prefill_rows(),
                None,
                "one call prefills a whole prompt"
            );
            let tokenizer = model.tokenizer();
            let text = "Hello, world! 17 × 23 = 391.";
            let tokens = tokenizer.tokenize(text);
            assert!(!tokens.is_empty());
            assert!(
                tokens.iter().all(|&t| (t as usize) < model.vocab_size()),
                "every token is in the vocabulary of {}",
                model.vocab_size()
            );
            let bytes: Vec<u8> = tokens
                .iter()
                .flat_map(|&t| tokenizer.token_bytes(t).iter().copied())
                .collect();
            assert_eq!(bytes, text.as_bytes(), "the tokens spell the text");
            let image = tokenizer
                .find_added_token("<｜deepseek_image｜>")
                .expect("DeepSeek's image token is an added token");
            assert!(tokenizer.is_added(image));
            assert!(!tokenizer.is_added(tokens[0]));
            assert_eq!(tokenizer.tokenize("<｜deepseek_image｜>"), [image]);
            let prompt = tokenizer.tokenize_prompt(b"a\xffb");
            assert!(prompt.contains(&image), "0xFF is an image's place");
            assert_eq!(tokenizer.find_added_token("Hello"), None);
            assert!(
                tokenizer.token_bytes(-1).is_empty(),
                "no bytes for an unknown id"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
