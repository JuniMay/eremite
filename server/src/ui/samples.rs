//! What the terminal UI keeps of the history window: one sample of every
//! model's status for each column of the history charts. The samples live in
//! the view's memory only. The figures "now" and "recent" are means over the
//! samples of their windows.

use std::collections::{HashMap, VecDeque};

use super::status::{Activity, ModelStatus};

/// The columns of the history charts. Each column shows one sample. 120
/// columns fit beside the charts' scale and labels on a terminal of 160
/// columns.
pub const CHART_COLUMNS: usize = 120;

/// The bytes of one GB, for the reads from the SSD.
const GB: f64 = 1e9;
/// The milliseconds of one second.
const MILLISECONDS_PER_SECOND: f64 = 1e3;

/// The windows of the figures, in seconds. Each one is a setting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Windows {
    /// The seconds that the "now" figures average.
    pub now: f64,
    /// The seconds that the "recent" figures average.
    pub recent: f64,
    /// The seconds that the history charts span.
    pub history: f64,
}

impl Windows {
    /// Returns the seconds between two samples: the history window spread
    /// over the charts' columns.
    pub fn sample_seconds(&self) -> f64 {
        self.history / CHART_COLUMNS as f64
    }

    /// Returns how many samples a window of `seconds` holds, at least one.
    pub fn count_samples(&self, seconds: f64) -> usize {
        ((seconds / self.sample_seconds()).round() as usize).max(1)
    }
}

/// A model's figures in one sample, since the sample before.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelSample {
    /// The tokens per second that the model decoded. None while it decoded
    /// nothing.
    pub decoding: Option<f64>,
    /// The prompt tokens per second that it prefilled. None while it
    /// prefilled nothing.
    pub prefill: Option<f64>,
    /// The share of the decoded rows' experts that were ready when needed.
    pub hit: Option<f64>,
    /// The milliseconds that each decoded row waited for expert reads.
    pub read_wait_ms: Option<f64>,
    /// What each of the model's running requests did, by its number.
    pub requests: Vec<(u64, Activity)>,
    /// The model's requests that began to run since the sample before.
    pub arrivals: Vec<u64>,
}

/// One sample of every model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    /// When the sample was taken: seconds since the view started.
    pub at: f64,
    /// Each model's figures, in the configured order.
    pub models: Vec<ModelSample>,
    /// The share of all models' decoding time in which the GPU was busy.
    pub gpu_busy: Option<f64>,
    /// The GB per second that the server read from the SSD while a model
    /// worked.
    pub ssd_read: Option<f64>,
}

impl Sample {
    /// Returns the sum of `figure` over the models that have it, or None when
    /// none has it.
    pub fn sum_models(&self, figure: impl Fn(&ModelSample) -> Option<f64>) -> Option<f64> {
        let known: Vec<f64> = self.models.iter().filter_map(figure).collect();
        (!known.is_empty()).then(|| known.iter().sum())
    }
}

/// A model's counters at a sample. The next sample takes the differences.
#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    /// The tokens that the model's requests generated.
    generated_tokens: u64,
    /// The prompt rows that the model computed.
    prompt_rows: f64,
    /// The seconds of its decoding calls.
    decode_seconds: f64,
    /// The seconds in which the GPU was busy with them.
    decode_gpu_seconds: f64,
    /// The rows that its decoding computed.
    decode_rows: u64,
    /// The seconds that the decoded rows waited for expert reads.
    read_wait_seconds: f64,
    /// The decoded rows' experts by where each one was.
    sources: [u64; 5],
    /// The bytes of experts that the model read.
    expert_read_bytes: u64,
    /// Whether a request of the model prefilled or decoded.
    worked: bool,
}

impl Counters {
    /// Returns the counters of `status`.
    fn of(status: &ModelStatus) -> Counters {
        Counters {
            generated_tokens: status.count_generated_tokens(),
            prompt_rows: status.count_prompt_rows(),
            decode_seconds: status.model_decode.seconds,
            decode_gpu_seconds: status.model_decode.main_queue_gpu_seconds,
            decode_rows: status.model_decode.computed_rows,
            read_wait_seconds: status.model_decode.read_wait_seconds,
            sources: status.engine_counters.decode_expert_sources,
            expert_read_bytes: status.engine_counters.expert_read_bytes,
            worked: status
                .requests
                .iter()
                .any(|request| request.status != Activity::Waiting),
        }
    }
}

/// The samples of the history window, and what the next sample needs.
#[derive(Clone, Debug)]
pub struct Samples {
    /// The windows of the figures.
    pub windows: Windows,
    /// The samples, the oldest first.
    pub samples: VecDeque<Sample>,
    /// Each model's counters at the latest sample.
    counters: Vec<Counters>,
    /// Each decoding request's generated tokens at the samples of the now
    /// window, by the request's number: when, and how many.
    tracks: HashMap<u64, VecDeque<(f64, u64)>>,
}

impl Samples {
    /// Returns no samples yet, for figures over `windows`.
    pub fn new(windows: Windows) -> Samples {
        Samples {
            windows,
            samples: VecDeque::new(),
            counters: Vec::new(),
            tracks: HashMap::new(),
        }
    }

    /// Adds a sample of `statuses`, every model's status, taken `at` seconds
    /// after the view started. It drops the samples that no window needs.
    pub fn add_sample(&mut self, at: f64, statuses: &[ModelStatus]) {
        let counters: Vec<Counters> = statuses.iter().map(Counters::of).collect();
        let before = self.samples.back();
        let seconds = before.map_or(0.0, |sample| at - sample.at);
        let compared = seconds > 0.0 && self.counters.len() == counters.len();
        let mut models = Vec::new();
        for (i, (status, now)) in statuses.iter().zip(&counters).enumerate() {
            let requests: Vec<(u64, Activity)> = status
                .requests
                .iter()
                .filter(|request| request.status != Activity::Waiting)
                .map(|request| (request.id, request.status))
                .collect();
            let ran_before = |id: u64| {
                before
                    .and_then(|sample| sample.models.get(i))
                    .is_some_and(|model| model.requests.iter().any(|&(other, _)| other == id))
            };
            let mut sample = ModelSample {
                arrivals: if compared {
                    requests
                        .iter()
                        .map(|&(id, _)| id)
                        .filter(|&id| !ran_before(id))
                        .collect()
                } else {
                    Vec::new()
                },
                requests,
                ..ModelSample::default()
            };
            if compared {
                let then = self.counters[i];
                sample.decoding = (now.generated_tokens > then.generated_tokens
                    || has_activity(status, Activity::Decoding))
                .then(|| {
                    now.generated_tokens.saturating_sub(then.generated_tokens) as f64 / seconds
                });
                sample.prefill = (now.prompt_rows > then.prompt_rows
                    || has_activity(status, Activity::Prefill))
                .then(|| (now.prompt_rows - then.prompt_rows).max(0.0) / seconds);
                let sources: [u64; 5] =
                    std::array::from_fn(|k| now.sources[k].saturating_sub(then.sources[k]));
                sample.hit = find_hit_share(sources);
                let rows = now.decode_rows.saturating_sub(then.decode_rows);
                sample.read_wait_ms = (rows > 0).then(|| {
                    (now.read_wait_seconds - then.read_wait_seconds).max(0.0) / rows as f64
                        * MILLISECONDS_PER_SECOND
                });
            }
            models.push(sample);
        }
        let (mut gpu_busy, mut ssd_read) = (None, None);
        if compared {
            let then = &self.counters;
            let decode_seconds: f64 = (0..counters.len())
                .map(|i| counters[i].decode_seconds - then[i].decode_seconds)
                .sum();
            let gpu_seconds: f64 = (0..counters.len())
                .map(|i| counters[i].decode_gpu_seconds - then[i].decode_gpu_seconds)
                .sum();
            gpu_busy =
                (decode_seconds > 0.0).then(|| (gpu_seconds / decode_seconds).clamp(0.0, 1.0));
            let worked = (0..counters.len()).any(|i| counters[i].worked || then[i].worked);
            let read: u64 = (0..counters.len())
                .map(|i| {
                    counters[i]
                        .expert_read_bytes
                        .saturating_sub(then[i].expert_read_bytes)
                })
                .sum();
            ssd_read = worked.then(|| read as f64 / GB / seconds);
        }
        self.track_requests(at, statuses);
        self.counters = counters;
        self.samples.push_back(Sample {
            at,
            models,
            gpu_busy,
            ssd_read,
        });
        let kept = CHART_COLUMNS
            .max(self.windows.count_samples(self.windows.recent))
            .max(self.windows.count_samples(self.windows.now));
        while self.samples.len() > kept {
            self.samples.pop_front();
        }
    }

    /// Records each decoding request's generated tokens at `at`, over the now
    /// window. It forgets the requests that no longer decode.
    fn track_requests(&mut self, at: f64, statuses: &[ModelStatus]) {
        let decoding: Vec<(u64, u64)> = statuses
            .iter()
            .flat_map(|status| &status.requests)
            .filter(|request| request.status == Activity::Decoding)
            .map(|request| (request.id, request.generated_tokens))
            .collect();
        self.tracks
            .retain(|id, _| decoding.iter().any(|&(other, _)| other == *id));
        let window = self.windows.now;
        for (id, generated) in decoding {
            let track = self.tracks.entry(id).or_default();
            track.push_back((at, generated));
            // Keep the oldest point inside the window, so that a whole window
            // lies between the ends.
            while track.len() > 2 && at - track[1].0 >= window {
                track.pop_front();
            }
        }
    }

    /// Returns the tokens per second that request `id` decoded over the now
    /// window, or None before two samples of it.
    pub fn find_request_speed(&self, id: u64) -> Option<f64> {
        let track = self.tracks.get(&id)?;
        let (&(first_at, first), &(last_at, last)) = (track.front()?, track.back()?);
        (last_at > first_at).then(|| last.saturating_sub(first) as f64 / (last_at - first_at))
    }

    /// Returns the mean of `figure` over the samples of the latest `seconds`
    /// that have it, or None when none has it.
    pub fn find_mean(&self, seconds: f64, figure: impl Fn(&Sample) -> Option<f64>) -> Option<f64> {
        let known: Vec<f64> = self
            .samples
            .iter()
            .rev()
            .take(self.windows.count_samples(seconds))
            .filter_map(figure)
            .collect();
        (!known.is_empty()).then(|| known.iter().sum::<f64>() / known.len() as f64)
    }
}

/// Returns whether a request of `status` does `activity`.
fn has_activity(status: &ModelStatus, activity: Activity) -> bool {
    status
        .requests
        .iter()
        .any(|request| request.status == activity)
}

/// Returns the share of `sources`, the decoded rows' experts by where each
/// one was, that were ready when needed: in memory, read ahead, or read ahead
/// and late. Returns None without experts.
pub fn find_hit_share(sources: [u64; 5]) -> Option<f64> {
    let all: u64 = sources.iter().sum();
    (all > 0).then(|| (sources[0] + sources[1] + sources[2]) as f64 / all as f64)
}

/// Returns the tokens per second that the models of `statuses` decoded since
/// the start: their tokens over their decoding seconds.
pub fn find_decoding_since_start(statuses: &[&ModelStatus]) -> Option<f64> {
    let seconds: f64 = statuses
        .iter()
        .map(|status| status.model_decode.seconds)
        .sum();
    let tokens: u64 = statuses
        .iter()
        .map(|status| status.count_generated_tokens())
        .sum();
    (seconds > 0.0).then(|| tokens as f64 / seconds)
}

/// Returns the prompt tokens per second that the models of `statuses`
/// prefilled since the start: the ended prefills' tokens over their seconds.
pub fn find_prefill_since_start(statuses: &[&ModelStatus]) -> Option<f64> {
    let seconds: f64 = statuses
        .iter()
        .map(|status| status.totals.prefill_seconds)
        .sum();
    let tokens: u64 = statuses
        .iter()
        .map(|status| status.totals.prefilled_tokens)
        .sum();
    (seconds > 0.0).then(|| tokens as f64 / seconds)
}

/// Returns the share of all models' decoding time since the start in which
/// the GPU was busy.
pub fn find_gpu_busy_since_start(statuses: &[ModelStatus]) -> Option<f64> {
    let seconds: f64 = statuses
        .iter()
        .map(|status| status.model_decode.seconds)
        .sum();
    let busy: f64 = statuses
        .iter()
        .map(|status| status.model_decode.main_queue_gpu_seconds)
        .sum();
    (seconds > 0.0).then(|| (busy / seconds).clamp(0.0, 1.0))
}

/// Returns the GB per second that all models read from the SSD since the
/// start, over the seconds in which they decoded or prefilled.
pub fn find_ssd_read_since_start(statuses: &[ModelStatus]) -> Option<f64> {
    let seconds: f64 = statuses
        .iter()
        .map(|status| status.model_decode.seconds + status.totals.prefill_seconds)
        .sum();
    let gb: f64 = statuses
        .iter()
        .map(|status| {
            status.model_decode.requested_expert_gb
                + status.engine_counters.prefill_read_bytes as f64 / GB
        })
        .sum();
    (seconds > 0.0).then(|| gb / seconds)
}

/// Returns the milliseconds that each row that the model of `status` decoded
/// since the start waited for expert reads.
pub fn find_read_wait_since_start(status: &ModelStatus) -> Option<f64> {
    let rows = status.model_decode.computed_rows;
    (rows > 0)
        .then(|| status.model_decode.read_wait_seconds / rows as f64 * MILLISECONDS_PER_SECOND)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::status::Request;

    /// The default windows: 5 s now, and 60 s recent and history.
    const WINDOWS: Windows = Windows {
        now: 5.0,
        recent: 60.0,
        history: 60.0,
    };

    /// A model's status with `requests`, `generated` tokens of ended
    /// requests, and counters that grow with `step`.
    fn status(requests: Vec<Request>, generated: u64, step: u64) -> ModelStatus {
        let mut status = ModelStatus {
            requests,
            ..ModelStatus::default()
        };
        status.totals.generated_tokens = generated;
        status.model_decode.seconds = 0.5 * step as f64;
        status.model_decode.main_queue_gpu_seconds = 0.4 * step as f64;
        status.model_decode.computed_rows = 10 * step;
        status.model_decode.read_wait_seconds = 0.01 * step as f64;
        status.engine_counters.decode_expert_sources = [90 * step, 0, 0, 0, 10 * step];
        status.engine_counters.expert_read_bytes = 1_000_000_000 * step;
        status
    }

    /// A request numbered `id` that does `status` with `generated` tokens.
    fn request(id: u64, status: Activity, generated: u64) -> Request {
        Request {
            id,
            status,
            generated_tokens: generated,
            ..Request::default()
        }
    }

    #[test]
    fn a_sample_takes_the_differences_of_the_counters() {
        let mut samples = Samples::new(WINDOWS);
        assert_eq!(WINDOWS.sample_seconds(), 0.5);
        assert_eq!(WINDOWS.count_samples(5.0), 10);
        samples.add_sample(
            0.0,
            &[status(vec![request(5, Activity::Decoding, 10)], 0, 0)],
        );
        let first = samples.samples.back().unwrap();
        assert_eq!(first.models[0].decoding, None, "no differences yet");
        assert!(first.models[0].arrivals.is_empty());
        samples.add_sample(
            0.5,
            &[status(
                vec![
                    request(5, Activity::Decoding, 24),
                    request(6, Activity::Prefill, 0),
                ],
                0,
                1,
            )],
        );
        let last = samples.samples.back().unwrap();
        let model = &last.models[0];
        // 14 tokens in 0.5 s; 90 of 100 experts in memory; 0.01 s for 10 rows
        assert_eq!(model.decoding, Some(28.0));
        assert_eq!(model.hit, Some(0.9));
        assert!((model.read_wait_ms.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(model.arrivals, [6], "request 6 began to run");
        assert!((last.gpu_busy.unwrap() - 0.8).abs() < 1e-9);
        assert!((last.ssd_read.unwrap() - 2.0).abs() < 1e-9, "1 GB in 0.5 s");
        assert_eq!(
            samples.find_mean(5.0, |sample| sample.gpu_busy),
            last.gpu_busy
        );
        assert_eq!(last.sum_models(|model| model.decoding), Some(28.0));
        assert_eq!(find_hit_share([0; 5]), None);
        let first = &samples.samples[0];
        assert_eq!(first.sum_models(|model| model.decoding), None);
    }

    #[test]
    fn the_speeds_since_the_start_count_each_models_tokens_and_seconds() {
        let mut first = status(vec![request(5, Activity::Decoding, 100)], 500, 20);
        first.totals.prefilled_tokens = 1000;
        first.totals.prefill_seconds = 10.0;
        let mut second = status(Vec::new(), 300, 0);
        second.model_decode.seconds = 20.0;
        second.totals.prefilled_tokens = 9000;
        second.totals.prefill_seconds = 20.0;
        // 600 tokens in 10 s, then also the second model's 300 in 20 s
        assert_eq!(find_decoding_since_start(&[&first]), Some(60.0));
        assert_eq!(find_decoding_since_start(&[&first, &second]), Some(30.0));
        assert_eq!(
            find_prefill_since_start(&[&first, &second]),
            Some(10_000.0 / 30.0)
        );
        assert_eq!(find_prefill_since_start(&[]), None);
    }

    #[test]
    fn the_gpu_and_the_ssd_since_the_start_count_all_models() {
        let mut first = status(Vec::new(), 0, 20);
        first.totals.prefill_seconds = 10.0;
        first.engine_counters.prefill_read_bytes = 5_000_000_000;
        first.model_decode.requested_expert_gb = 15.0;
        let mut second = status(Vec::new(), 0, 0);
        second.model_decode.seconds = 20.0;
        second.totals.prefill_seconds = 20.0;
        let both = [first, second];
        // 8 GPU seconds in 30 s of decoding calls
        assert_eq!(find_gpu_busy_since_start(&both), Some(8.0 / 30.0));
        // 15 GB decoded and 5 GB prefilled, in 20 s and then 40 s
        assert_eq!(find_ssd_read_since_start(&both), Some(20.0 / 60.0));
        assert_eq!(find_gpu_busy_since_start(&[]), None);
    }

    #[test]
    fn a_decoding_requests_speed_spans_the_now_window() {
        let mut samples = Samples::new(WINDOWS);
        for k in 0..200u64 {
            let requests = vec![request(5, Activity::Decoding, 10 * k)];
            samples.add_sample(0.5 * k as f64, &[status(requests, 0, k)]);
        }
        // 10 tokens a sample, two samples a second
        let speed = samples.find_request_speed(5).unwrap();
        assert!((speed - 20.0).abs() < 1e-9, "{speed}");
        let track = &samples.tracks[&5];
        assert!(track.back().unwrap().0 - track.front().unwrap().0 <= 5.5);
        // a request that no longer decodes is forgotten
        samples.add_sample(100.0, &[status(Vec::new(), 2000, 200)]);
        assert_eq!(samples.find_request_speed(5), None);
    }

    #[test]
    fn the_read_wait_since_the_start_is_per_decoded_row() {
        let first = status(Vec::new(), 0, 20);
        // 0.2 s for 200 rows
        let wait = find_read_wait_since_start(&first).unwrap();
        assert!((wait - 1.0).abs() < 1e-9, "{wait}");
        assert_eq!(find_read_wait_since_start(&ModelStatus::default()), None);
    }

    #[test]
    fn the_samples_keep_the_longest_window() {
        let mut samples = Samples::new(WINDOWS);
        for k in 0..200u64 {
            let requests = vec![request(5, Activity::Decoding, 10 * k)];
            samples.add_sample(0.5 * k as f64, &[status(requests, 0, k)]);
        }
        assert_eq!(samples.samples.len(), CHART_COLUMNS);
        let longer = Windows {
            recent: 120.0,
            ..WINDOWS
        };
        let mut samples = Samples::new(longer);
        for k in 0..300u64 {
            samples.add_sample(0.5 * k as f64, &[status(Vec::new(), 0, k)]);
        }
        assert_eq!(samples.samples.len(), 240, "the recent window's 120 s");
    }
}
