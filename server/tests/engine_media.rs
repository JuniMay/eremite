//! An image in a prompt: its file decodes, its span takes the image's place
//! in the prompt, and the prefill reads the encoded image. The model then
//! names the image's color. Set EREMITE_CHECKPOINT to a DeepSeek V4.1
//! checkpoint's directory, and EREMITE_EXPERTS to GGUF experts if wanted;
//! without EREMITE_CHECKPOINT, the test is skipped.
use eremite_server::engine::{self, ExpandError, Model, Sampler, Stream};

/// A 64 by 64 PNG image of one red color.
const RED_PNG: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x40\x00\x00\x00\x40\x08\x02\x00\x00\x00\x25\x0b\xe6\x89\x00\x00\x00\x4f\x49\x44\x41\x54\x78\xda\xed\xcf\x41\x09\x00\x00\x08\x04\xb0\x8b\x72\xfd\xd3\x18\xc9\x08\xbe\x85\xc1\x0a\x2c\xd3\xbe\x16\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x81\xcb\x02\x3a\x2d\x01\x00\xc3\x21\xbe\xb0\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

/// The tokens that the model decodes after the prompt.
const DECODED_TOKENS: usize = 6;

/// The context's capacity in tokens.
const CTX: usize = 4096;

/// Runs the stream's posted call to its end.
fn run(stream: &Stream) {
    assert_eq!(engine::run_streams(&[stream]), Some(0), "the call finishes");
}

#[test]
fn the_model_reads_an_image_in_its_prompt() {
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
                ..Default::default()
            };
            options.deepseek_mut().unwrap().experts = std::env::var("EREMITE_EXPERTS").ok();
            let model = Model::open(&options).expect("the checkpoint opens");
            model.load().expect("the model loads");
            let error = model.open_media(&[b"not an image".to_vec()]).err();
            assert!(
                error.is_some_and(|e| !e.is_empty()),
                "a file that is no image"
            );
            let tokenizer = model.tokenizer();
            let prompt = tokenizer.tokenize(
                "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>What color is this \
                 image? Answer in one word.<｜Assistant｜></think>",
            );
            let mut media = model
                .open_media(&[RED_PNG.to_vec()])
                .expect("the image decodes");
            let text_only = tokenizer.tokenize("<｜begin▁of▁sentence｜>No image here.");
            assert_eq!(
                media.expand(&text_only, CTX),
                Err(ExpandError::PlacesAndMediaDiffer)
            );
            let expanded = media.expand(&prompt, CTX).expect("the span fits");
            assert!(
                expanded.len() > prompt.len(),
                "the span takes more than one token"
            );
            assert_eq!(
                media.expand(&prompt, prompt.len()),
                Err(ExpandError::TooLong(expanded.len()))
            );
            assert_eq!(media.encode(0), Ok(1), "one image encoded");
            let state = model.state_new().unwrap();
            let stream = Stream::new(&model);
            stream.prefill(state, &expanded, 0, false, None, Some(&media));
            run(&stream);
            drop(media);
            let mut greedy = Sampler::new(0.0, 1.0, 0, None).unwrap();
            let mut tokens = vec![model.sample(stream.logits().unwrap(), &mut greedy)];
            for step in 0..DECODED_TOKENS - 1 {
                let pos = (expanded.len() + step) as i32;
                stream.decode_rows(&[state], &tokens[step..=step], &[pos]);
                run(&stream);
                tokens.push(model.sample(stream.logits().unwrap(), &mut greedy));
            }
            let text: Vec<u8> = tokens
                .iter()
                .flat_map(|&t| tokenizer.token_bytes(t).iter().copied())
                .collect();
            let text = String::from_utf8_lossy(&text).to_lowercase();
            assert!(text.contains("red"), "{text:?}");
            model.state_free(state);
        })
        .unwrap()
        .join()
        .unwrap();
}
