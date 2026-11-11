//! Shared `/v1/models` entry assembly (spec models-list-context-metadata):
//! entry collection with dedup, config-driven capacity attachment, and the
//! virtual-key `model_access` visibility filter — one pipeline reused by the
//! list and retrieve endpoints.

use crate::config::{Config, ModelGroup, ProviderModel};
use crate::providers::Model;
use crate::virtual_keys::models::AuthenticatedKey;

/// Build the aggregated `/v1/models` entry list: model-group names, per-group
/// configured models, provider manual models, the NVIDIA NIM built-in
/// fallback, then Copilot catalog models (passed in pre-fetched by the caller
/// because the catalog read is async) — all in the existing emission order
/// with the existing first-seen dedup. Capacity metadata is attached from
/// config, then the virtual-key `model_access` filter is applied exactly as
/// the list endpoint has always done.
pub(super) fn build_models_entries(
    authenticated_key: Option<&AuthenticatedKey>,
    config: &Config,
    copilot_models: &[Model],
) -> Vec<Model> {
    let mut all_models: Vec<Model> = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    // List model group names first so clients can target groups directly
    for group in &config.model_groups {
        if seen_ids.insert(group.name.clone()) {
            all_models.push(Model {
                id: group.name.clone(),
                object: "model".to_string(),
                owned_by: "gateway".to_string(),
                created: None,
                context_window: None,
                max_completion_tokens: None,
                supports_vision: false,
            });
        }
    }

    // Also list individual model names for backward compatibility
    for group in &config.model_groups {
        for pm in &group.models {
            if seen_ids.insert(pm.model.clone()) {
                all_models.push(Model {
                    id: pm.model.clone(),
                    object: "model".to_string(),
                    owned_by: pm.provider.clone(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                    supports_vision: false,
                });
            }
        }
    }

    // Include manually specified models from provider configs first, then add
    // the NVIDIA NIM built-in fallback. The existing dedup preserves manual
    // entries as explicit overrides.
    for provider in &config.providers {
        for model_id in &provider.manual_models {
            if seen_ids.insert(model_id.clone()) {
                all_models.push(Model {
                    id: model_id.clone(),
                    object: "model".to_string(),
                    owned_by: provider.name.clone(),
                    created: None,
                    context_window: None,
                    max_completion_tokens: None,
                    supports_vision: false,
                });
            }
        }

        if provider.provider_type == "nvidia_nim" {
            for mut model in crate::providers::nvidia_nim::fallback_models() {
                model.owned_by = provider.name.clone();
                if seen_ids.insert(model.id.clone()) {
                    all_models.push(model);
                }
            }
        }
    }

    // GitHub Copilot catalog models (github-copilot-oauth-provider,
    // Req 5.2 / task 9 wiring): entitlement-filtered enabled models,
    // pre-fetched by the caller (empty until the first successful
    // fetch; never blocks, never errors).
    for model in copilot_models {
        if seen_ids.insert(model.id.clone()) {
            all_models.push(model.clone());
        }
    }

    attach_capacity(&mut all_models, config);

    // When the caller authenticated with a virtual key that has a model_access
    // restriction, filter the list to only models/groups in the allowed set.
    // A permitted model group name also grants visibility to all individual
    // models within that group.
    if let Some(authenticated) = authenticated_key {
        if let Some(ref allowed) = authenticated.model_access {
            // Build the expanded set: explicit allowed entries + all individual
            // model IDs belonging to allowed model groups.
            let mut visible: std::collections::HashSet<&str> =
                allowed.iter().map(|s| s.as_str()).collect();
            for group in &config.model_groups {
                if allowed.iter().any(|a| a == &group.name) {
                    for pm in &group.models {
                        visible.insert(&pm.model);
                    }
                }
            }
            all_models.retain(|m| visible.contains(m.id.as_str()));
        }
    }

    all_models
}

/// Attach capacity metadata (Req 1-2) to emitted `/v1/models` entries.
///
/// Pure resolution over config; fills only fields that are `None` so
/// catalog/manual-sourced entries (NIM, Bedrock, Copilot) keep the values
/// they already carry, and unknown windows stay omitted rather than
/// advertised as 0 (Req 1.2).
///
/// Deterministic resolution rules:
/// - Per-model entries take their `ProviderModel.context_window`; when the
///   same model id is declared in multiple model groups with differing
///   values, the FIRST declaration in config order wins (Req 1.3).
/// - Group entries aggregate MAX over their candidates' positive windows
///   (Req 2.1/2.3). A group name always resolves to the group aggregate:
///   group entries are emitted before per-model entries, so an id present
///   in both maps is by construction a group entry.
fn attach_capacity(entries: &mut [Model], config: &Config) {
    // Group name → aggregated capacity (first group wins on duplicate
    // names, matching the first-seen emission dedup).
    let mut group_caps: std::collections::HashMap<&str, (Option<u32>, Option<u32>)> =
        std::collections::HashMap::new();
    for group in &config.model_groups {
        group_caps
            .entry(group.name.as_str())
            .or_insert_with(|| group_capacity(group));
    }

    // Model id → declared capacity; first declaration in config order wins.
    let mut model_caps: std::collections::HashMap<&str, (Option<u32>, Option<u32>)> =
        std::collections::HashMap::new();
    for group in &config.model_groups {
        for pm in &group.models {
            model_caps
                .entry(pm.model.as_str())
                .or_insert_with(|| (declared_window(pm), declared_max_completion(pm)));
        }
    }

    for entry in entries.iter_mut() {
        let (window, max_completion) = if let Some(&caps) = group_caps.get(entry.id.as_str()) {
            caps
        } else {
            model_caps
                .get(entry.id.as_str())
                .copied()
                .unwrap_or((None, None))
        };
        if entry.context_window.is_none() {
            entry.context_window = window;
        }
        if entry.max_completion_tokens.is_none() {
            entry.max_completion_tokens = max_completion;
        }
    }
}

/// Aggregated capacity for one model group: MAX over the candidates'
/// positive `context_window` values (Req 2.1) and, aggregated the same way,
/// MAX over the candidates' `max_completion_tokens` (Req 2.3 — no config
/// field exists today, so this yields `None` until one is added).
fn group_capacity(group: &ModelGroup) -> (Option<u32>, Option<u32>) {
    let window = group.models.iter().filter_map(declared_window).max();
    let max_completion = group
        .models
        .iter()
        .filter_map(declared_max_completion)
        .max();
    (window, max_completion)
}

/// A candidate's declared context window; `0`/absent means unknown, so it
/// resolves to `None` and is never advertised as a false 0 capacity
/// (Req 1.2).
fn declared_window(pm: &ProviderModel) -> Option<u32> {
    (pm.context_window > 0).then_some(pm.context_window)
}

/// Forward-compat hook: `ProviderModel` carries no `max_completion_tokens`
/// config field today, so candidates never declare one. Group aggregation
/// and per-model resolution stay wired through this hook for when a field
/// is added.
fn declared_max_completion(_pm: &ProviderModel) -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_keys::models::KeyStatus;

    /// Minimal valid Config via YAML (every optional section defaults), with
    /// caller-supplied `model_groups` and `manual_models` provider entry.
    fn config_from_yaml(groups_yaml: &str) -> Config {
        let yaml = format!(
            r#"
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 30
  max_request_size_mb: 10
providers:
  - name: "test"
    type: "openai"
    base_url: "http://localhost:1234"
    manual_models: []
{groups_yaml}
"#
        );
        serde_yaml::from_str(&yaml).unwrap()
    }

    fn find<'a>(models: &'a [Model], id: &str) -> &'a Model {
        models
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("entry {id} not present"))
    }

    // Req 1.1: a per-model entry resolves its declared context_window.
    #[test]
    fn per_model_entry_carries_declared_context_window() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "default"
    models:
      - provider: "test"
        model: "gpt-4o"
        context_window: 128000
"#,
        );
        let entries = build_models_entries(None, &config, &[]);
        assert_eq!(find(&entries, "gpt-4o").context_window, Some(128_000));
    }

    // Req 1.2: `0` means unknown → omitted, never advertised as 0.
    #[test]
    fn zero_context_window_resolves_to_none() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "default"
    models:
      - provider: "test"
        model: "mystery"
        context_window: 0
"#,
        );
        let entries = build_models_entries(None, &config, &[]);
        assert_eq!(find(&entries, "mystery").context_window, None);
    }

    // Req 2.1: group entry aggregates MAX over positive candidate windows;
    // unknown (0/absent) candidates do not drag the max down.
    #[test]
    fn group_entry_aggregates_max_over_candidate_windows() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "mixed"
    models:
      - provider: "test"
        model: "small"
        context_window: 8000
      - provider: "test"
        model: "big"
        context_window: 200000
      - provider: "test"
        model: "unknown"
        context_window: 0
"#,
        );
        let entries = build_models_entries(None, &config, &[]);
        assert_eq!(find(&entries, "mixed").context_window, Some(200_000));
    }

    // Req 2.2: a group whose candidates are all unknown omits capacity.
    #[test]
    fn group_with_all_unknown_windows_omits_capacity() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "opaque"
    models:
      - provider: "test"
        model: "a"
        context_window: 0
      - provider: "test"
        model: "b"
"#,
        );
        let entries = build_models_entries(None, &config, &[]);
        assert_eq!(find(&entries, "opaque").context_window, None);
    }

    // Req 1.3: the same model id declared in two groups resolves
    // deterministically to the FIRST declaration in config order.
    #[test]
    fn duplicate_model_declaration_first_wins() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "first"
    models:
      - provider: "test"
        model: "shared"
        context_window: 100000
  - name: "second"
    models:
      - provider: "test"
        model: "shared"
        context_window: 200000
"#,
        );
        let entries = build_models_entries(None, &config, &[]);
        assert_eq!(find(&entries, "shared").context_window, Some(100_000));
    }

    // Req 1.4: entries that already carry capacity (NIM/Bedrock/Copilot
    // catalog entries) keep their values; attach_capacity fills only None.
    #[test]
    fn catalog_capacity_not_overwritten() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "default"
    models:
      - provider: "test"
        model: "catalog-model"
        context_window: 128000
"#,
        );
        let mut entries = vec![Model {
            id: "catalog-model".to_string(),
            object: "model".to_string(),
            owned_by: "nim".to_string(),
            created: None,
            context_window: Some(42_000),
            max_completion_tokens: Some(8_192),
            supports_vision: false,
        }];
        attach_capacity(&mut entries, &config);
        assert_eq!(entries[0].context_window, Some(42_000));
        assert_eq!(entries[0].max_completion_tokens, Some(8_192));
    }

    // Same guarantee at the pipeline level: a Copilot catalog entry passed
    // in with its own capacity is emitted unchanged.
    #[test]
    fn copilot_catalog_entries_keep_own_capacity() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "default"
    models:
      - provider: "test"
        model: "gpt-4o"
        context_window: 128000
"#,
        );
        let copilot = vec![Model {
            id: "claude-sonnet-copilot".to_string(),
            object: "model".to_string(),
            owned_by: "github-copilot".to_string(),
            created: None,
            context_window: Some(200_000),
            max_completion_tokens: Some(16_384),
            supports_vision: true,
        }];
        let entries = build_models_entries(None, &config, &copilot);
        let entry = find(&entries, "claude-sonnet-copilot");
        assert_eq!(entry.context_window, Some(200_000));
        assert_eq!(entry.max_completion_tokens, Some(16_384));
    }

    // Emission order: group names first, then per-group models, then
    // provider manual models; dedup keeps the first occurrence.
    #[test]
    fn entry_order_groups_then_models_then_manual() {
        let mut config = config_from_yaml(
            r#"
model_groups:
  - name: "alpha"
    models:
      - provider: "test"
        model: "m1"
  - name: "beta"
    models:
      - provider: "test"
        model: "m2"
"#,
        );
        config.providers[0].manual_models = vec![
            "manual-x".to_string(),
            // Already emitted via group "alpha" — dedup must skip it.
            "m1".to_string(),
        ];
        let entries = build_models_entries(None, &config, &[]);
        let ids: Vec<&str> = entries.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "beta", "m1", "m2", "manual-x"]);
    }

    fn restricted_key(model_access: Vec<&str>) -> AuthenticatedKey {
        AuthenticatedKey {
            id: "vk-test".to_string(),
            name: None,
            status: KeyStatus::Active,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            current_spend_usd: 0.0,
            current_tokens_used: 0,
            window_start: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: Some(model_access.into_iter().map(String::from).collect()),
            expires_at: None,
            loop_detection: None,
        }
    }

    // Virtual-key model_access restriction: an allowed group name expands
    // to its member models; everything else is hidden.
    #[test]
    fn vk_model_access_filter_restricts_list_and_expands_groups() {
        let config = config_from_yaml(
            r#"
model_groups:
  - name: "allowed"
    models:
      - provider: "test"
        model: "a1"
      - provider: "test"
        model: "a2"
  - name: "hidden"
    models:
      - provider: "test"
        model: "h1"
"#,
        );
        let key = restricted_key(vec!["allowed"]);
        let entries = build_models_entries(Some(&key), &config, &[]);
        let ids: Vec<&str> = entries.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["allowed", "a1", "a2"]);

    // A bare model id in the allowed set grants only that model.
    let key = restricted_key(vec!["h1"]);
    let entries = build_models_entries(Some(&key), &config, &[]);
    let ids: Vec<&str> = entries.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, vec!["h1"]);
}

// Group/model id collision: a ProviderModel in one group named the same as
// another group must not override the group's aggregate window. The dedup
// emits the group entry first (groups before per-model entries), so the
// single visible entry for the colliding id carries the GROUP aggregate
// window, and the retrieve selection (find on the same built list) picks
// that same entry.
#[test]
fn group_model_id_collision_keeps_group_aggregate_window() {
    let config = config_from_yaml(
        r#"
model_groups:
- name: "shared"
  models:
  - provider: "test"
    model: "collide"
    context_window: 100
- name: "other"
  models:
  - provider: "test"
    model: "shared"
    context_window: 999
"#,
    );
    let entries = build_models_entries(None, &config, &[]);

    // Dedup collapses the collision to exactly one entry per id; "shared"
    // is the group entry (emitted first).
    let shared: Vec<&str> = entries
        .iter()
        .filter(|m| m.id == "shared")
        .map(|m| m.id.as_str())
        .collect();
    assert_eq!(shared, vec!["shared"]);
    let shared_entry = find(&entries, "shared");

    // The emitted entry for "shared" carries the group aggregate MAX (100),
    // not the colliding ProviderModel's declared 999.
    assert_eq!(shared_entry.context_window, Some(100));

    // "other" keeps its own group aggregate from its candidate ("shared":
    // 999) — the collision does not leak into other groups' aggregates.
    assert_eq!(find(&entries, "other").context_window, Some(999));

    // Retrieve selection (same pipeline + find, as the endpoint does) picks
    // the same group-aggregate entry.
    let selected = build_models_entries(None, &config, &[])
        .into_iter()
        .find(|m| m.id == "shared");
    assert!(selected.is_some());
    assert_eq!(selected.map(|m| m.context_window), Some(Some(100)));
}
}
