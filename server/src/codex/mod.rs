//! What the server gives Codex beyond the Responses API. GET /v1/codex/models
//! returns Codex's model catalog. Codex reads the catalog from the URL in its
//! `model_catalog_url` setting. POST /v1/alpha/search is the service behind
//! Codex's web tool. search.rs holds it.
//!
//! Codex has defaults for a model without catalog metadata. It assumes a
//! context window of 272,000 tokens. It caps `model_context_window` there. It
//! offers no `apply_patch` tool to such a model. The catalog gives each
//! served text model its served context and Codex's reasoning levels. The
//! server maps each level to that model family's thinking effort. The catalog
//! also gives Codex's own instructions and the `apply_patch` tool. Every
//! catalog entry must carry instructions. Codex rejects a catalog in which an
//! entry has none.

pub mod search;

use serde_json::{Value, json};

use crate::chat::deepseek::{EFFORT_HIGH, EFFORT_LOW, EFFORT_MAX};
use crate::chat::inkling2;

/// Codex's instructions for a model without its own: `prompt.md` of Codex
/// 0.167.2. assets/codex/README.md says where the file comes from and how to
/// update it.
pub const INSTRUCTIONS: &str = include_str!("../../assets/codex/prompt.md");

/// A served model family's reasoning effort settings for Codex's catalog.
pub enum CatalogReasoning {
    /// DeepSeek V4.1 reasoning effort (1 to 100, or None when thinking is off).
    DeepSeek(Option<u32>),
    /// Inkling-2-Mini reasoning effort in 0.0..=0.99.
    Inkling2(f64),
}

/// A model in the catalog: a text model that answers the Responses API.
pub struct CatalogModel<'a> {
    /// The model's id. Codex calls it the slug.
    pub model_id: &'a str,
    /// The name that Codex shows for the model.
    pub display_name: &'a str,
    /// The number of tokens in the model's context.
    pub context_window: usize,
    /// The model family's default reasoning effort and supported levels.
    pub reasoning: CatalogReasoning,
}

/// Chooses Codex's reasoning level for a DeepSeek effort.
/// No effort gives "none". An effort of 50 or less gives "low".
/// An effort of 100 or more gives "max". Any other effort gives "high".
/// The server reads these levels back as the efforts 50, 75 and 100.
fn choose_reasoning_level(effort: Option<u32>) -> &'static str {
    match effort {
        None => "none",
        Some(effort) if effort <= EFFORT_LOW => "low",
        Some(effort) if effort >= EFFORT_MAX => "max",
        Some(_) => "high",
    }
}

/// Chooses Codex's reasoning level for an Inkling-2-Mini effort in 0.0..=0.99.
fn choose_inkling2_reasoning_level(effort: f64) -> &'static str {
    if effort <= inkling2::EFFORT_NONE {
        "none"
    } else if effort <= inkling2::EFFORT_MINIMAL {
        "minimal"
    } else if effort <= inkling2::EFFORT_LOW {
        "low"
    } else if effort <= inkling2::EFFORT_MEDIUM {
        "medium"
    } else if effort <= inkling2::EFFORT_HIGH {
        "high"
    } else {
        "xhigh"
    }
}

/// Builds the catalog entry of `model` in the form of Codex's `ModelInfo`.
pub fn build_catalog_entry(model: &CatalogModel) -> Value {
    let (default_level, supported_levels) = match model.reasoning {
        CatalogReasoning::DeepSeek(effort) => (
            choose_reasoning_level(effort),
            json!([
                {"effort": "none", "description": "Thinking off"},
                {"effort": "low", "description": format!("DeepSeek reasoning effort {EFFORT_LOW}")},
                {"effort": "high", "description": format!("DeepSeek reasoning effort {EFFORT_HIGH}")},
                {"effort": "max", "description": format!("DeepSeek reasoning effort {EFFORT_MAX}")},
            ]),
        ),
        CatalogReasoning::Inkling2(effort) => (
            choose_inkling2_reasoning_level(effort),
            json!([
                {"effort": "none", "description": "Thinking off"},
                {"effort": "minimal", "description": format!("Inkling-2-Mini reasoning effort {}", inkling2::EFFORT_MINIMAL)},
                {"effort": "low", "description": format!("Inkling-2-Mini reasoning effort {}", inkling2::EFFORT_LOW)},
                {"effort": "medium", "description": format!("Inkling-2-Mini reasoning effort {}", inkling2::EFFORT_MEDIUM)},
                {"effort": "high", "description": format!("Inkling-2-Mini reasoning effort {}", inkling2::EFFORT_HIGH)},
                {"effort": "xhigh", "description": format!("Inkling-2-Mini reasoning effort {}", inkling2::EFFORT_MAX)},
            ]),
        ),
    };
    json!({
        "slug": model.model_id,
        "display_name": model.display_name,
        "description": "Served by eremite",
        "default_reasoning_level": default_level,
        "supported_reasoning_levels": supported_levels,
        // The fields below keep the values of Codex's fallback model.
        // The model gives no reason to change them.
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "availability_nux": null,
        "upgrade": null,
        "model_messages": {"instructions_template": INSTRUCTIONS},
        "include_skills_usage_instructions": false,
        "include_plugin_usage_instructions": false,
        "include_apps_usage_instructions": false,
        // The server returns reasoning as text, never as summaries.
        "supports_reasoning_summary_parameter": true,
        "default_reasoning_summary": "none",
        "support_verbosity": false,
        "default_verbosity": null,
        // deepseek-recipe turns the custom apply_patch tool into a function
        // that the model knows. It returns the model's calls of that function
        // as custom tool calls.
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        // The reason for this value is unknown.
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_image_detail_original": false,
        "context_window": model.context_window,
        "max_context_window": model.context_window,
        // The reason for this value is unknown.
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_parallel_tool_calls": true,
        "supports_search_tool": false,
    })
}

/// Builds the catalog of `models` as `{"models": [...]}`.
pub fn build_catalog<'a>(models: impl IntoIterator<Item = CatalogModel<'a>>) -> Value {
    let models: Vec<Value> = models
        .into_iter()
        .map(|model| build_catalog_entry(&model))
        .collect();
    json!({ "models": models })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_carry_the_served_context_the_instructions_and_apply_patch() {
        let catalog = build_catalog([CatalogModel {
            model_id: "deepseek-v4.1-flash",
            display_name: "DeepSeek V4.1 Flash (eremite)",
            context_window: 1_048_576,
            reasoning: CatalogReasoning::DeepSeek(Some(75)),
        }]);
        let entry = &catalog["models"][0];
        assert_eq!(entry["slug"], "deepseek-v4.1-flash");
        assert_eq!(
            (
                entry["context_window"].as_u64(),
                entry["max_context_window"].as_u64()
            ),
            (Some(1_048_576), Some(1_048_576))
        );
        assert_eq!(entry["apply_patch_tool_type"], "freeform");
        assert_eq!(entry["default_reasoning_level"], "high");
        let template = entry["model_messages"]["instructions_template"]
            .as_str()
            .unwrap();
        // A failure shows the template's first 80 bytes.
        // The reason for this value is unknown.
        assert!(
            template.starts_with("You are a coding agent running in the Codex CLI"),
            "{}",
            &template[..80]
        );
        let levels: Vec<&str> = entry["supported_reasoning_levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|level| level["effort"].as_str().unwrap())
            .collect();
        assert_eq!(levels, ["none", "low", "high", "max"]);
    }

    #[test]
    fn efforts_become_the_levels_the_server_reads_back() {
        assert_eq!(choose_reasoning_level(None), "none");
        assert_eq!(choose_reasoning_level(Some(50)), "low");
        assert_eq!(choose_reasoning_level(Some(30)), "low");
        assert_eq!(choose_reasoning_level(Some(75)), "high");
        assert_eq!(choose_reasoning_level(Some(62)), "high");
        assert_eq!(choose_reasoning_level(Some(100)), "max");
    }

    #[test]
    fn inkling2_catalog_entry_carries_its_reasoning_levels_and_apply_patch() {
        let catalog = build_catalog([CatalogModel {
            model_id: inkling2::DEFAULT_MODEL_ID,
            display_name: inkling2::DISPLAY_NAME,
            context_window: 1_048_576,
            reasoning: CatalogReasoning::Inkling2(inkling2::EFFORT_HIGH),
        }]);
        let entry = &catalog["models"][0];
        assert_eq!(entry["slug"], "inkling-2-mini");
        assert_eq!(entry["default_reasoning_level"], "high");
        assert_eq!(entry["apply_patch_tool_type"], "freeform");
        let levels: Vec<&str> = entry["supported_reasoning_levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|level| level["effort"].as_str().unwrap())
            .collect();
        assert_eq!(
            levels,
            ["none", "minimal", "low", "medium", "high", "xhigh"]
        );
        assert_eq!(choose_inkling2_reasoning_level(0.0), "none");
        assert_eq!(choose_inkling2_reasoning_level(0.1), "minimal");
        assert_eq!(choose_inkling2_reasoning_level(0.2), "low");
        assert_eq!(choose_inkling2_reasoning_level(0.7), "medium");
        assert_eq!(choose_inkling2_reasoning_level(0.9), "high");
        assert_eq!(choose_inkling2_reasoning_level(0.99), "xhigh");
    }
}
