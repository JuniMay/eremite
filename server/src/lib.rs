//! eremite-server's modules: the engine's C interface, the decoding calls, the
//! yield and resume control, the errors that end requests, and the HTTP
//! connections.

pub mod calls;
pub mod chat;
pub mod control;
pub mod engine;
pub mod http;
