//! eremite-server's modules: the engine's C interface, the configuration, the
//! state cache and its saved states, the decoding calls, the yield and resume
//! control, the worker on the engine thread, the chat formats, the decision
//! models' prompts, Codex's catalog and web tool, the HTTP connections, and
//! the terminal UI.

pub mod calls;
pub mod chat;
pub mod codex;
pub mod config;
pub mod control;
pub mod decision;
pub mod disk;
pub mod engine;
pub mod http;
pub mod image;
pub mod state_cache;
pub mod ui;
pub mod video;
pub mod worker;
