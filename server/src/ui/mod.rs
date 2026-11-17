//! The terminal UI (--tui): the server's status, its requests, its memory and
//! its log, drawn in the terminal with ratatui.

pub mod draw;
pub mod frame;
pub mod history;
pub mod llm;
pub mod log;
pub mod memory;
pub mod overview;
pub mod samples;
pub mod status;
#[cfg(test)]
mod testing;
pub mod theme;
pub mod tiles;
