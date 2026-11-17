//! eremite-server's modules: the engine's C interface, the configuration, the
//! state cache and its saved states, the decoding calls, the yield and resume
//! control, the worker on the engine thread, the chat formats, Codex's web
//! tool, and the HTTP connections.

pub mod calls;
pub mod chat;
pub mod codex;
pub mod config;
pub mod control;
pub mod disk;
pub mod engine;
pub mod http;
pub mod state_cache;
pub mod worker;
