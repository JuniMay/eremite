//! A loaded decision model runs a pass on a stream: a state and a question
//! after it. The pass returns the logits of a few tokens at the question's
//! last row, as close to LiquidAI's FP32 run as LiquidAI's BF16 run, and the
//! same bits when it runs again. Set EREMITE_D1_CHECKPOINT to d1-2B's
//! directory; without it, the test is skipped.
use eremite_server::engine::{self, Model, Stream};

/// A JSON state and one yes-or-no question, as d1's prompt holds them.
const TOKENS: [i32; 89] = [
    124894, 124899, 5922, 207, 2718, 229, 496, 99489, 6380, 827, 360, 496, 6929, 6380, 496, 16308,
    52966, 278, 48505, 7285, 1917, 229, 10016, 229, 496, 3166, 6380, 827, 360, 496, 16414, 273,
    6380, 19843, 733, 229, 3255, 920, 733, 229, 3255, 22, 29, 207, 360, 40439, 360, 496, 37507,
    6380, 2846, 920, 360, 496, 14700, 6380, 3495, 207, 229, 1130, 64024, 63509, 833, 29088, 278,
    29936, 7986, 267, 48505, 7285, 7450, 467, 278, 2064, 64929, 57922, 415, 18171, 432, 794, 1049,
    22, 124900, 207, 124899, 63514, 207, 124901, 124902,
];
/// The state's tokens at the start of TOKENS.
const STATE: usize = 63;
/// The tokens of yes, Yes, no, No and NO, read at the question's last row,
/// their logits in LiquidAI's own code in FP32, and the largest difference of
/// its BF16 run from them.
const IDS: [i32; 5] = [11683, 12447, 2243, 4547, 19598];
const REFERENCE: [f32; 5] = [8.003815, 10.504749, 5.083753, 7.219336, 1.851632];
const BOUND: f32 = 0.0619;

/// Runs the pass of TOKENS on `stream` to its end, and returns its logits.
fn decide(stream: &Stream) -> Vec<f32> {
    let rows = [TOKENS.len() as i32 - 1; IDS.len()];
    stream.decide(&TOKENS, STATE, &[TOKENS.len() as i32], &rows, &IDS);
    assert_eq!(engine::run_streams(&[stream]), Some(0), "the pass finishes");
    stream
        .copy_decision_logits(IDS.len())
        .expect("the pass's logits")
}

#[test]
fn a_decision_model_reads_its_logits_from_a_pass() {
    let Ok(checkpoint) = std::env::var("EREMITE_D1_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_D1_CHECKPOINT");
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
            let options = engine::Options {
                model: checkpoint,
                ctx: 4096,
                ..Default::default()
            };
            let model = Model::open(&options).expect("the checkpoint opens");
            assert!(model.is_decision_model());
            model.load().expect("the model loads");
            let stream = Stream::new(&model);
            let logits = decide(&stream);
            let difference = logits
                .iter()
                .zip(REFERENCE)
                .map(|(logit, reference)| (logit - reference).abs())
                .fold(0.0, f32::max);
            println!("the logits differ from FP32 by {difference:.4}");
            assert!(difference <= BOUND, "{logits:?}");
            assert_eq!(decide(&stream), logits, "the same pass gives the same bits");
        })
        .unwrap()
        .join()
        .unwrap();
}
