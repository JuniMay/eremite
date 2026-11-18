//! Decision models: models that read a state and questions about it, and
//! answer each question with probabilities in one pass. api.rs reads OpenAI's
//! decision requests and writes their answers; d1.rs holds the prompt and the
//! answers' probabilities of LiquidAI's d1 models.

pub mod api;
pub mod d1;
