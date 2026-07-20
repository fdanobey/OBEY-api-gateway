//! RTK command-output compression engine.

use super::{
    CompressibleMessage, CompressiblePayload, CompressionContext, CompressionEngine, EngineResult,
};
use crate::compression::config::{RtkConfig, RtkGroupingStrategy};
use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::Instant;

const EDGE_LINES: usize = 5;
const REDACTED: &str = "[REDACTED]";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CommandCategory {
    Git,
    Test,
    Build,
    Lint,
    Package,
    Docker,
    Search,
    FileRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputProfile {
    Git,
    Summary,
    Docker,
    Search,
    FileRead,
}

#[derive(Debug, Clone, Copy)]
struct CommandHandler {
    name: &'static str,
    aliases: &'static [&'static str],
    category: CommandCategory,
    profile: OutputProfile,
    detection_tokens: &'static [&'static str],
}

const GIT_TOKENS: &[&str] = &[
    "diff --git",
    "on branch",
    "commit ",
    "fatal:",
    "nothing to commit",
];
const TEST_TOKENS: &[&str] = &[
    "test result:",
    "tests passed",
    "tests failed",
    " passed",
    " failed",
];
const BUILD_TOKENS: &[&str] = &[
    "compiling ",
    "build succeeded",
    "build failed",
    "finished release",
    "finished dev",
];
const LINT_TOKENS: &[&str] = &["warning:", "lint", "problems (", "diagnostic", " --> "];
const PACKAGE_TOKENS: &[&str] = &[
    "packages:",
    "added ",
    "lockfile",
    "resolved ",
    "dependencies",
];
const DOCKER_TOKENS: &[&str] = &[
    "pulling fs layer",
    "digest: sha256:",
    "container ",
    "exporting layers",
    "[+] building",
];
const SEARCH_TOKENS: &[&str] = &[
    "matches found",
    "binary file",
    "no matches",
    "search results",
];
const READ_TOKENS: &[&str] = &["<content>", "(end of file", "showing lines", "total lines"];

macro_rules! handler {
    ($name:literal, $category:ident, $profile:ident, [$($alias:literal),+ $(,)?], $tokens:ident) => {
        CommandHandler {
            name: $name,
            aliases: &[$($alias),+],
            category: CommandCategory::$category,
            profile: OutputProfile::$profile,
            detection_tokens: $tokens,
        }
    };
}

// Each entry describes a distinct command invocation rather than a cosmetic alias.
// Filtering remains category-driven, while aliases and output tokens make detection
// work for shell tools that expose either the command or only its output.
const COMMAND_HANDLERS: &[CommandHandler] = &[
    handler!("git-status", Git, Git, ["git status"], GIT_TOKENS),
    handler!("git-diff", Git, Git, ["git diff"], GIT_TOKENS),
    handler!("git-log", Git, Git, ["git log"], GIT_TOKENS),
    handler!("git-show", Git, Git, ["git show"], GIT_TOKENS),
    handler!("git-branch", Git, Git, ["git branch"], GIT_TOKENS),
    handler!("git-checkout", Git, Git, ["git checkout"], GIT_TOKENS),
    handler!("git-switch", Git, Git, ["git switch"], GIT_TOKENS),
    handler!("git-merge", Git, Git, ["git merge"], GIT_TOKENS),
    handler!("git-rebase", Git, Git, ["git rebase"], GIT_TOKENS),
    handler!("git-cherry-pick", Git, Git, ["git cherry-pick"], GIT_TOKENS),
    handler!("git-fetch", Git, Git, ["git fetch"], GIT_TOKENS),
    handler!("git-pull", Git, Git, ["git pull"], GIT_TOKENS),
    handler!("git-push", Git, Git, ["git push"], GIT_TOKENS),
    handler!("git-clone", Git, Git, ["git clone"], GIT_TOKENS),
    handler!("git-add", Git, Git, ["git add"], GIT_TOKENS),
    handler!("git-commit", Git, Git, ["git commit"], GIT_TOKENS),
    handler!("git-restore", Git, Git, ["git restore"], GIT_TOKENS),
    handler!("git-reset", Git, Git, ["git reset"], GIT_TOKENS),
    handler!("git-stash", Git, Git, ["git stash"], GIT_TOKENS),
    handler!("git-tag", Git, Git, ["git tag"], GIT_TOKENS),
    handler!("git-blame", Git, Git, ["git blame"], GIT_TOKENS),
    handler!("git-grep", Git, Git, ["git grep"], GIT_TOKENS),
    handler!("git-rev-list", Git, Git, ["git rev-list"], GIT_TOKENS),
    handler!("git-ls-files", Git, Git, ["git ls-files"], GIT_TOKENS),
    handler!("git-submodule", Git, Git, ["git submodule"], GIT_TOKENS),
    handler!("git-worktree", Git, Git, ["git worktree"], GIT_TOKENS),
    handler!("git-remote", Git, Git, ["git remote"], GIT_TOKENS),
    handler!("git-clean", Git, Git, ["git clean"], GIT_TOKENS),
    handler!("git-bisect", Git, Git, ["git bisect"], GIT_TOKENS),
    handler!("git-describe", Git, Git, ["git describe"], GIT_TOKENS),
    handler!("cargo-test", Test, Summary, ["cargo test"], TEST_TOKENS),
    handler!("pytest", Test, Summary, ["pytest", "py.test"], TEST_TOKENS),
    handler!(
        "python-pytest",
        Test,
        Summary,
        ["python -m pytest", "python3 -m pytest"],
        TEST_TOKENS
    ),
    handler!("npm-test", Test, Summary, ["npm test"], TEST_TOKENS),
    handler!("npm-run-test", Test, Summary, ["npm run test"], TEST_TOKENS),
    handler!("pnpm-test", Test, Summary, ["pnpm test"], TEST_TOKENS),
    handler!("yarn-test", Test, Summary, ["yarn test"], TEST_TOKENS),
    handler!("bun-test", Test, Summary, ["bun test"], TEST_TOKENS),
    handler!("go-test", Test, Summary, ["go test"], TEST_TOKENS),
    handler!("dotnet-test", Test, Summary, ["dotnet test"], TEST_TOKENS),
    handler!(
        "maven-test",
        Test,
        Summary,
        ["mvn test", "mvnw test"],
        TEST_TOKENS
    ),
    handler!(
        "gradle-test",
        Test,
        Summary,
        ["gradle test", "gradlew test", "./gradlew test"],
        TEST_TOKENS
    ),
    handler!("junit", Test, Summary, ["junit"], TEST_TOKENS),
    handler!("jest", Test, Summary, ["jest"], TEST_TOKENS),
    handler!("vitest", Test, Summary, ["vitest"], TEST_TOKENS),
    handler!("mocha", Test, Summary, ["mocha"], TEST_TOKENS),
    handler!("ava", Test, Summary, ["ava"], TEST_TOKENS),
    handler!("rspec", Test, Summary, ["rspec"], TEST_TOKENS),
    handler!("rake-test", Test, Summary, ["rake test"], TEST_TOKENS),
    handler!("phpunit", Test, Summary, ["phpunit"], TEST_TOKENS),
    handler!("mix-test", Test, Summary, ["mix test"], TEST_TOKENS),
    handler!("swift-test", Test, Summary, ["swift test"], TEST_TOKENS),
    handler!(
        "xcode-test",
        Test,
        Summary,
        ["xcodebuild test"],
        TEST_TOKENS
    ),
    handler!("ctest", Test, Summary, ["ctest"], TEST_TOKENS),
    handler!("bazel-test", Test, Summary, ["bazel test"], TEST_TOKENS),
    handler!("cargo-build", Build, Summary, ["cargo build"], BUILD_TOKENS),
    handler!("cargo-check", Build, Summary, ["cargo check"], BUILD_TOKENS),
    handler!("npm-build", Build, Summary, ["npm run build"], BUILD_TOKENS),
    handler!(
        "pnpm-build",
        Build,
        Summary,
        ["pnpm build", "pnpm run build"],
        BUILD_TOKENS
    ),
    handler!("yarn-build", Build, Summary, ["yarn build"], BUILD_TOKENS),
    handler!(
        "bun-build",
        Build,
        Summary,
        ["bun run build", "bun build"],
        BUILD_TOKENS
    ),
    handler!("go-build", Build, Summary, ["go build"], BUILD_TOKENS),
    handler!(
        "dotnet-build",
        Build,
        Summary,
        ["dotnet build"],
        BUILD_TOKENS
    ),
    handler!(
        "maven-package",
        Build,
        Summary,
        ["mvn package", "mvnw package"],
        BUILD_TOKENS
    ),
    handler!(
        "gradle-build",
        Build,
        Summary,
        ["gradle build", "gradlew build", "./gradlew build"],
        BUILD_TOKENS
    ),
    handler!("make", Build, Summary, ["make", "gmake"], BUILD_TOKENS),
    handler!(
        "cmake-build",
        Build,
        Summary,
        ["cmake --build"],
        BUILD_TOKENS
    ),
    handler!("ninja", Build, Summary, ["ninja"], BUILD_TOKENS),
    handler!(
        "meson-compile",
        Build,
        Summary,
        ["meson compile"],
        BUILD_TOKENS
    ),
    handler!("bazel-build", Build, Summary, ["bazel build"], BUILD_TOKENS),
    handler!("buck2-build", Build, Summary, ["buck2 build"], BUILD_TOKENS),
    handler!("swift-build", Build, Summary, ["swift build"], BUILD_TOKENS),
    handler!("xcode-build", Build, Summary, ["xcodebuild"], BUILD_TOKENS),
    handler!(
        "typescript",
        Build,
        Summary,
        ["tsc", "npx tsc"],
        BUILD_TOKENS
    ),
    handler!("vite-build", Build, Summary, ["vite build"], BUILD_TOKENS),
    handler!("next-build", Build, Summary, ["next build"], BUILD_TOKENS),
    handler!("webpack", Build, Summary, ["webpack"], BUILD_TOKENS),
    handler!("rollup", Build, Summary, ["rollup"], BUILD_TOKENS),
    handler!("esbuild", Build, Summary, ["esbuild"], BUILD_TOKENS),
    handler!("cargo-clippy", Lint, Summary, ["cargo clippy"], LINT_TOKENS),
    handler!(
        "cargo-fmt",
        Lint,
        Summary,
        ["cargo fmt", "rustfmt"],
        LINT_TOKENS
    ),
    handler!(
        "eslint",
        Lint,
        Summary,
        ["eslint", "npx eslint"],
        LINT_TOKENS
    ),
    handler!(
        "biome",
        Lint,
        Summary,
        ["biome check", "biome lint"],
        LINT_TOKENS
    ),
    handler!(
        "prettier-check",
        Lint,
        Summary,
        ["prettier --check"],
        LINT_TOKENS
    ),
    handler!("stylelint", Lint, Summary, ["stylelint"], LINT_TOKENS),
    handler!("ruff", Lint, Summary, ["ruff check"], LINT_TOKENS),
    handler!("flake8", Lint, Summary, ["flake8"], LINT_TOKENS),
    handler!("pylint", Lint, Summary, ["pylint"], LINT_TOKENS),
    handler!("mypy", Lint, Summary, ["mypy"], LINT_TOKENS),
    handler!("pyright", Lint, Summary, ["pyright"], LINT_TOKENS),
    handler!("black-check", Lint, Summary, ["black --check"], LINT_TOKENS),
    handler!(
        "golangci-lint",
        Lint,
        Summary,
        ["golangci-lint"],
        LINT_TOKENS
    ),
    handler!("go-vet", Lint, Summary, ["go vet"], LINT_TOKENS),
    handler!("staticcheck", Lint, Summary, ["staticcheck"], LINT_TOKENS),
    handler!("shellcheck", Lint, Summary, ["shellcheck"], LINT_TOKENS),
    handler!("hadolint", Lint, Summary, ["hadolint"], LINT_TOKENS),
    handler!("markdownlint", Lint, Summary, ["markdownlint"], LINT_TOKENS),
    handler!("yamllint", Lint, Summary, ["yamllint"], LINT_TOKENS),
    handler!("actionlint", Lint, Summary, ["actionlint"], LINT_TOKENS),
    handler!("rubocop", Lint, Summary, ["rubocop"], LINT_TOKENS),
    handler!("phpstan", Lint, Summary, ["phpstan"], LINT_TOKENS),
    handler!("checkstyle", Lint, Summary, ["checkstyle"], LINT_TOKENS),
    handler!("detekt", Lint, Summary, ["detekt"], LINT_TOKENS),
    handler!(
        "npm-install",
        Package,
        Summary,
        ["npm install", "npm i"],
        PACKAGE_TOKENS
    ),
    handler!("npm-ci", Package, Summary, ["npm ci"], PACKAGE_TOKENS),
    handler!(
        "npm-update",
        Package,
        Summary,
        ["npm update"],
        PACKAGE_TOKENS
    ),
    handler!("npm-audit", Package, Summary, ["npm audit"], PACKAGE_TOKENS),
    handler!(
        "pnpm-install",
        Package,
        Summary,
        ["pnpm install", "pnpm i"],
        PACKAGE_TOKENS
    ),
    handler!(
        "pnpm-update",
        Package,
        Summary,
        ["pnpm update"],
        PACKAGE_TOKENS
    ),
    handler!(
        "yarn-install",
        Package,
        Summary,
        ["yarn install"],
        PACKAGE_TOKENS
    ),
    handler!(
        "bun-install",
        Package,
        Summary,
        ["bun install"],
        PACKAGE_TOKENS
    ),
    handler!("cargo-add", Package, Summary, ["cargo add"], PACKAGE_TOKENS),
    handler!(
        "cargo-update",
        Package,
        Summary,
        ["cargo update"],
        PACKAGE_TOKENS
    ),
    handler!(
        "cargo-fetch",
        Package,
        Summary,
        ["cargo fetch"],
        PACKAGE_TOKENS
    ),
    handler!(
        "pip-install",
        Package,
        Summary,
        ["pip install", "pip3 install"],
        PACKAGE_TOKENS
    ),
    handler!(
        "poetry-install",
        Package,
        Summary,
        ["poetry install"],
        PACKAGE_TOKENS
    ),
    handler!("uv-sync", Package, Summary, ["uv sync"], PACKAGE_TOKENS),
    handler!(
        "bundle-install",
        Package,
        Summary,
        ["bundle install"],
        PACKAGE_TOKENS
    ),
    handler!(
        "composer-install",
        Package,
        Summary,
        ["composer install"],
        PACKAGE_TOKENS
    ),
    handler!(
        "go-mod-download",
        Package,
        Summary,
        ["go mod download"],
        PACKAGE_TOKENS
    ),
    handler!(
        "go-mod-tidy",
        Package,
        Summary,
        ["go mod tidy"],
        PACKAGE_TOKENS
    ),
    handler!(
        "dotnet-restore",
        Package,
        Summary,
        ["dotnet restore"],
        PACKAGE_TOKENS
    ),
    handler!(
        "maven-dependency",
        Package,
        Summary,
        ["mvn dependency", "mvnw dependency"],
        PACKAGE_TOKENS
    ),
    handler!(
        "gradle-dependencies",
        Package,
        Summary,
        ["gradle dependencies", "gradlew dependencies"],
        PACKAGE_TOKENS
    ),
    handler!(
        "apt-install",
        Package,
        Summary,
        ["apt install", "apt-get install"],
        PACKAGE_TOKENS
    ),
    handler!("apk-add", Package, Summary, ["apk add"], PACKAGE_TOKENS),
    handler!(
        "brew-install",
        Package,
        Summary,
        ["brew install"],
        PACKAGE_TOKENS
    ),
    handler!(
        "choco-install",
        Package,
        Summary,
        ["choco install"],
        PACKAGE_TOKENS
    ),
    handler!(
        "winget-install",
        Package,
        Summary,
        ["winget install"],
        PACKAGE_TOKENS
    ),
    handler!(
        "docker-build",
        Docker,
        Docker,
        ["docker build"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-buildx",
        Docker,
        Docker,
        ["docker buildx build"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-compose-build",
        Docker,
        Docker,
        ["docker compose build", "docker-compose build"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-compose-up",
        Docker,
        Docker,
        ["docker compose up", "docker-compose up"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-compose-down",
        Docker,
        Docker,
        ["docker compose down", "docker-compose down"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-compose-logs",
        Docker,
        Docker,
        ["docker compose logs", "docker-compose logs"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-compose-ps",
        Docker,
        Docker,
        ["docker compose ps", "docker-compose ps"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-compose-pull",
        Docker,
        Docker,
        ["docker compose pull", "docker-compose pull"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-pull",
        Docker,
        Docker,
        ["docker pull"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-push",
        Docker,
        Docker,
        ["docker push"],
        DOCKER_TOKENS
    ),
    handler!("docker-run", Docker, Docker, ["docker run"], DOCKER_TOKENS),
    handler!("docker-ps", Docker, Docker, ["docker ps"], DOCKER_TOKENS),
    handler!(
        "docker-logs",
        Docker,
        Docker,
        ["docker logs"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-images",
        Docker,
        Docker,
        ["docker images"],
        DOCKER_TOKENS
    ),
    handler!(
        "docker-inspect",
        Docker,
        Docker,
        ["docker inspect"],
        DOCKER_TOKENS
    ),
    handler!(
        "podman-build",
        Docker,
        Docker,
        ["podman build"],
        DOCKER_TOKENS
    ),
    handler!("podman-run", Docker, Docker, ["podman run"], DOCKER_TOKENS),
    handler!(
        "podman-pull",
        Docker,
        Docker,
        ["podman pull"],
        DOCKER_TOKENS
    ),
    handler!(
        "nerdctl-build",
        Docker,
        Docker,
        ["nerdctl build"],
        DOCKER_TOKENS
    ),
    handler!(
        "nerdctl-run",
        Docker,
        Docker,
        ["nerdctl run"],
        DOCKER_TOKENS
    ),
    handler!(
        "kubectl-logs",
        Docker,
        Docker,
        ["kubectl logs"],
        DOCKER_TOKENS
    ),
    handler!(
        "kubectl-get",
        Docker,
        Docker,
        ["kubectl get"],
        DOCKER_TOKENS
    ),
    handler!(
        "kubectl-describe",
        Docker,
        Docker,
        ["kubectl describe"],
        DOCKER_TOKENS
    ),
    handler!(
        "helm-install",
        Docker,
        Docker,
        ["helm install"],
        DOCKER_TOKENS
    ),
    handler!(
        "helm-upgrade",
        Docker,
        Docker,
        ["helm upgrade"],
        DOCKER_TOKENS
    ),
    handler!("grep", Search, Search, ["grep"], SEARCH_TOKENS),
    handler!("ripgrep", Search, Search, ["rg", "ripgrep"], SEARCH_TOKENS),
    handler!("silver-searcher", Search, Search, ["ag"], SEARCH_TOKENS),
    handler!("ack", Search, Search, ["ack"], SEARCH_TOKENS),
    handler!("findstr", Search, Search, ["findstr"], SEARCH_TOKENS),
    handler!(
        "select-string",
        Search,
        Search,
        ["select-string"],
        SEARCH_TOKENS
    ),
    handler!("fd", Search, Search, ["fd"], SEARCH_TOKENS),
    handler!("find", Search, Search, ["find"], SEARCH_TOKENS),
    handler!("locate", Search, Search, ["locate"], SEARCH_TOKENS),
    handler!("where", Search, Search, ["where"], SEARCH_TOKENS),
    handler!("which", Search, Search, ["which"], SEARCH_TOKENS),
    handler!(
        "get-child-item",
        Search,
        Search,
        ["get-childitem", "get-child-item"],
        SEARCH_TOKENS
    ),
    handler!("cat", FileRead, FileRead, ["cat"], READ_TOKENS),
    handler!("type", FileRead, FileRead, ["type"], READ_TOKENS),
    handler!(
        "get-content",
        FileRead,
        FileRead,
        ["get-content"],
        READ_TOKENS
    ),
    handler!("bat", FileRead, FileRead, ["bat"], READ_TOKENS),
    handler!("less", FileRead, FileRead, ["less"], READ_TOKENS),
    handler!("more", FileRead, FileRead, ["more"], READ_TOKENS),
    handler!("head", FileRead, FileRead, ["head"], READ_TOKENS),
    handler!("tail", FileRead, FileRead, ["tail"], READ_TOKENS),
    handler!("sed-print", FileRead, FileRead, ["sed -n"], READ_TOKENS),
    handler!("awk", FileRead, FileRead, ["awk"], READ_TOKENS),
    handler!(
        "read-file",
        FileRead,
        FileRead,
        ["read_file", "read-file"],
        READ_TOKENS
    ),
    handler!("read-tool", FileRead, FileRead, ["read"], READ_TOKENS),
    handler!("view-file", FileRead, FileRead, ["view"], READ_TOKENS),
    handler!("nl", FileRead, FileRead, ["nl"], READ_TOKENS),
];

/// Compresses recognized command output while retaining diagnostic context.
#[derive(Debug, Clone, Copy)]
pub struct RtkEngine {
    grouping_strategy: RtkGroupingStrategy,
}

impl Default for RtkEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RtkEngine {
    pub const fn new() -> Self {
        Self {
            grouping_strategy: RtkGroupingStrategy::Balanced,
        }
    }

    pub const fn with_grouping_strategy(grouping_strategy: RtkGroupingStrategy) -> Self {
        Self { grouping_strategy }
    }

    pub const fn from_config(config: &RtkConfig) -> Self {
        Self::with_grouping_strategy(config.grouping_strategy)
    }

    pub const fn grouping_strategy(&self) -> RtkGroupingStrategy {
        self.grouping_strategy
    }
}

#[async_trait]
impl CompressionEngine for RtkEngine {
    fn name(&self) -> &str {
        "rtk"
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        let started = Instant::now();
        let original = payload.clone();
        let tokens_before = count_payload_tokens(&original, context);
        let mut changed = false;

        for message in &mut payload.messages {
            if message.cache_protected || !is_tool_result(message) {
                continue;
            }

            let hints = command_hints(message);
            message.content.for_each_text_leaf_mut(|text| {
                if text.trim().is_empty() || is_structured_json(text) {
                    return;
                }
                let Some(detected) = detect_command(&hints, text) else {
                    return;
                };
                let filtered = filter_output(text, detected, self.grouping_strategy);
                if filtered != *text && filtered.len() <= text.len() {
                    *text = filtered;
                    changed = true;
                }
            });
        }

        if changed {
            payload.refresh_metadata();
            refresh_message_token_counts(payload, context);
        }

        let tokens_after = count_payload_tokens(payload, context);
        let applied = changed && tokens_after <= tokens_before;
        if !applied {
            *payload = original;
        }

        EngineResult {
            engine_name: self.name().to_owned(),
            tokens_before,
            tokens_after: if applied { tokens_after } else { tokens_before },
            duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            applied,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DetectedCommand {
    category: CommandCategory,
    profile: OutputProfile,
}

fn detect_command(hints: &[String], output: &str) -> Option<DetectedCommand> {
    for hint in hints {
        let hint = hint.to_ascii_lowercase();
        for handler in COMMAND_HANDLERS {
            if handler.name.is_empty() {
                continue;
            }
            if handler
                .aliases
                .iter()
                .any(|alias| command_alias_matches(&hint, alias))
            {
                return Some(DetectedCommand {
                    category: handler.category,
                    profile: handler.profile,
                });
            }
        }
    }

    let lower = output.to_ascii_lowercase();
    let mut best: Option<(&CommandHandler, usize)> = None;
    for handler in COMMAND_HANDLERS {
        let score = handler
            .detection_tokens
            .iter()
            .filter(|token| lower.contains(**token))
            .count();
        if score > best.map_or(0, |(_, best_score)| best_score) {
            best = Some((handler, score));
        }
    }
    if let Some((handler, score)) = best {
        if score > 0 {
            return Some(DetectedCommand {
                category: handler.category,
                profile: handler.profile,
            });
        }
    }

    if search_match_regex().find_iter(output).count() >= 2 {
        return Some(DetectedCommand {
            category: CommandCategory::Search,
            profile: OutputProfile::Search,
        });
    }
    if numbered_line_regex().find_iter(output).count() >= 12 {
        return Some(DetectedCommand {
            category: CommandCategory::FileRead,
            profile: OutputProfile::FileRead,
        });
    }
    None
}

fn command_alias_matches(hint: &str, alias: &str) -> bool {
    let alias = alias.to_ascii_lowercase();
    if hint == alias {
        return true;
    }
    hint.match_indices(&alias).any(|(start, matched)| {
        let end = start + matched.len();
        let before_ok = start == 0
            || hint[..start]
                .chars()
                .next_back()
                .is_none_or(|character| !is_command_character(character));
        let after_ok = end == hint.len()
            || hint[end..]
                .chars()
                .next()
                .is_none_or(|character| !is_command_character(character));
        before_ok && after_ok
    })
}

fn is_command_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
}

fn command_hints(message: &CompressibleMessage) -> Vec<String> {
    let mut hints = message.relationships.tool_names.clone();
    for (key, value) in &message.extra {
        collect_command_hints(key, value, &mut hints);
    }
    hints
}

fn collect_command_hints(key: &str, value: &Value, hints: &mut Vec<String>) {
    let relevant = matches!(
        key.to_ascii_lowercase().as_str(),
        "command" | "cmd" | "script" | "name" | "tool_name" | "arguments" | "input"
    );
    match value {
        Value::String(text) if relevant => hints.push(text.clone()),
        Value::Object(object) => {
            for (nested_key, nested_value) in object {
                collect_command_hints(nested_key, nested_value, hints);
            }
        }
        Value::Array(values) => {
            for nested in values {
                collect_command_hints(key, nested, hints);
            }
        }
        _ => {}
    }
}

fn is_tool_result(message: &CompressibleMessage) -> bool {
    matches!(message.role.as_str(), "tool" | "function")
        || !message.relationships.tool_result_for_ids.is_empty()
}

fn is_structured_json(text: &str) -> bool {
    serde_json::from_str::<Value>(text.trim())
        .is_ok_and(|value| matches!(value, Value::Array(_) | Value::Object(_)))
}

fn filter_output(text: &str, detected: DetectedCommand, strategy: RtkGroupingStrategy) -> String {
    let sanitized = sanitize_output(text);
    let lines = remove_noise(&sanitized, detected.category);
    if lines.is_empty() {
        return String::new();
    }

    let specialized = match detected.profile {
        OutputProfile::Search => group_search_results(&lines, strategy),
        OutputProfile::FileRead => trim_file_read(&lines, strategy),
        OutputProfile::Git | OutputProfile::Summary | OutputProfile::Docker => lines.join("\n"),
    };
    let specialized_lines: Vec<String> = specialized.lines().map(str::to_owned).collect();
    let compacted = if matches!(
        detected.profile,
        OutputProfile::Search | OutputProfile::FileRead
    ) {
        specialized_lines
    } else {
        compact_repetitions(&specialized_lines, strategy, detected.category)
    };
    let limited = limit_large_output(&compacted, strategy, detected.category);
    let candidate = limited.join("\n");

    if candidate.len() <= text.len() {
        candidate
    } else if sanitized.len() <= text.len() {
        sanitized
    } else {
        text.to_owned()
    }
}

fn sanitize_output(text: &str) -> String {
    let without_ansi = ansi_regex().replace_all(text, "");
    let without_osc = osc_regex().replace_all(&without_ansi, "");
    let mut cleaned = String::with_capacity(without_osc.len());
    for character in without_osc.chars() {
        match character {
            '\r' => cleaned.push('\n'),
            '\n' | '\t' => cleaned.push(character),
            character if character.is_control() => {}
            character => cleaned.push(character),
        }
    }
    redact_secrets(&cleaned)
}

fn redact_secrets(text: &str) -> String {
    let text = openai_key_regex().replace_all(text, REDACTED);
    let text = aws_key_regex().replace_all(&text, REDACTED);
    let text = bearer_regex().replace_all(&text, "Bearer [REDACTED]");
    url_password_regex()
        .replace_all(&text, "${scheme}${user}[REDACTED]@")
        .into_owned()
}

fn remove_noise(text: &str, category: CommandCategory) -> Vec<String> {
    let mut lines = Vec::new();
    let mut removed = 0usize;
    let mut blank = false;

    for raw_line in text.lines() {
        let line = raw_line.trim_end();
        if line.trim().is_empty() {
            if !blank && !lines.is_empty() {
                lines.push(String::new());
            }
            blank = true;
            continue;
        }
        blank = false;
        if is_noise_line(line, category) && !is_important_line(line) {
            removed += 1;
            continue;
        }
        lines.push(strip_spinner_prefix(line).to_owned());
    }

    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    if removed >= 3 && lines.len() >= EDGE_LINES * 2 {
        let marker = format!("[... {removed} progress/decorative lines removed ...]");
        lines.insert(EDGE_LINES.min(lines.len()), marker);
    }
    lines
}

fn is_noise_line(line: &str, category: CommandCategory) -> bool {
    let trimmed = line.trim();
    if timestamp_only_regex().is_match(trimmed)
        || decorative_regex().is_match(trimmed)
        || spinner_only_regex().is_match(trimmed)
        || progress_only_regex().is_match(trimmed)
    {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    match category {
        CommandCategory::Docker => {
            (lower.contains("waiting") || lower.contains("downloading"))
                && percent_regex().is_match(&lower)
        }
        CommandCategory::Package => {
            (lower.starts_with("resolved ")
                || lower.starts_with("downloaded ")
                || lower.starts_with("progress:"))
                && !lower.contains("error")
        }
        CommandCategory::Build | CommandCategory::Test | CommandCategory::Lint => {
            lower == "running..." || lower.starts_with("⠋") || lower.starts_with("⠙")
        }
        _ => false,
    }
}

fn strip_spinner_prefix(line: &str) -> &str {
    line.trim_start_matches(|character: char| {
        matches!(
            character,
            '⠋' | '⠙' | '⠹' | '⠸' | '⠼' | '⠴' | '⠦' | '⠧' | '⠇' | '⠏'
        ) || character == '\u{2800}'
    })
    .trim_start()
}

fn is_important_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let diagnostic = lower.contains("error")
        || lower.contains("warning")
        || lower.contains("failed")
        || lower.contains("failure")
        || lower.contains("fatal")
        || lower.contains("panic")
        || lower.contains("exception");
    diagnostic && !looks_like_repeated_diagnostic_item(&lower)
        || lower.contains("test result:")
        || lower.contains("build succeeded")
        || lower.contains("build failed")
        || lower.contains("exit code")
        || lower.contains("exit status")
        || lower.contains("process exited")
        || lower.starts_with("summary:")
}

fn looks_like_repeated_diagnostic_item(lower: &str) -> bool {
    lower.contains(".rs:")
        || lower.contains(".ts:")
        || lower.contains(".tsx:")
        || lower.contains(".js:")
        || lower.contains(".jsx:")
        || lower.contains(".py:")
        || lower.contains(".go:")
        || lower.contains(".java:")
}

fn group_search_results(lines: &[String], strategy: RtkGroupingStrategy) -> String {
    #[derive(Debug)]
    struct MatchLine {
        line_number: String,
        column: Option<String>,
        body: String,
    }

    let mut order = Vec::<String>::new();
    let mut groups = HashMap::<String, Vec<MatchLine>>::new();
    let mut passthrough = Vec::<String>::new();

    for line in lines {
        let Some(captures) = search_match_regex().captures(line) else {
            passthrough.push(line.clone());
            continue;
        };
        let path = captures[1].to_owned();
        if !groups.contains_key(&path) {
            order.push(path.clone());
        }
        groups.entry(path).or_default().push(MatchLine {
            line_number: captures[2].to_owned(),
            column: captures.get(3).map(|capture| capture.as_str().to_owned()),
            body: captures[4].trim_start().to_owned(),
        });
    }

    let grouped_count: usize = groups.values().map(Vec::len).sum();
    if grouped_count < 3 || groups.values().all(|matches| matches.len() == 1) {
        return lines.join("\n");
    }

    let per_file_limit = match strategy {
        RtkGroupingStrategy::Aggressive => 10,
        RtkGroupingStrategy::Balanced => 16,
        RtkGroupingStrategy::Conservative => 30,
    };
    let mut output = Vec::new();
    for path in order {
        let matches = &groups[&path];
        output.push(format!("{path}:"));
        let keep = selected_indices(matches.len(), per_file_limit, &HashSet::new());
        let mut previous = None;
        for &index in &keep {
            if previous.is_some_and(|previous_index| index > previous_index + 1) {
                output.push(format!(
                    "  [... {} matches omitted ...]",
                    index - previous.unwrap_or(0) - 1
                ));
            }
            let item = &matches[index];
            let location = item.column.as_ref().map_or_else(
                || item.line_number.clone(),
                |column| format!("{}:{column}", item.line_number),
            );
            output.push(format!("  {location}: {}", item.body));
            previous = Some(index);
        }
        if previous.is_some_and(|previous_index| previous_index + 1 < matches.len()) {
            output.push(format!(
                "  [... {} matches omitted ...]",
                matches.len() - previous.unwrap_or(0) - 1
            ));
        }
        if !keep.contains(&(matches.len() - 1)) {
            let item = &matches[matches.len() - 1];
            let location = item.column.as_ref().map_or_else(
                || item.line_number.clone(),
                |column| format!("{}:{column}", item.line_number),
            );
            output.push(format!("  {location}: {}", item.body));
        }
    }
    output.extend(passthrough);
    let candidate = output.join("\n");
    if candidate.len() < lines.join("\n").len() {
        candidate
    } else {
        lines.join("\n")
    }
}

fn trim_file_read(lines: &[String], strategy: RtkGroupingStrategy) -> String {
    let limit = match strategy {
        RtkGroupingStrategy::Aggressive => 36,
        RtkGroupingStrategy::Balanced => 64,
        RtkGroupingStrategy::Conservative => 120,
    };
    if lines.len() <= limit {
        return lines.join("\n");
    }

    let mut important = HashSet::new();
    for (index, line) in lines.iter().enumerate() {
        if is_useful_source_line(line) || is_important_line(line) {
            important.insert(index);
        }
    }
    let selected = selected_indices(lines.len(), limit, &important);
    let mut selected = selected;
    for index in important {
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    selected.sort_unstable();
    selected.dedup();
    render_selected(lines, &selected, "lines")
}

fn is_useful_source_line(line: &str) -> bool {
    let body = numbered_line_capture_regex()
        .captures(line)
        .map_or(line, |captures| {
            captures.get(1).map_or(line, |capture| capture.as_str())
        });
    let trimmed = body.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    trimmed.starts_with('#')
        || lower.starts_with("fn ")
        || lower.starts_with("pub fn ")
        || lower.starts_with("async fn ")
        || lower.starts_with("class ")
        || lower.starts_with("struct ")
        || lower.starts_with("enum ")
        || lower.starts_with("impl ")
        || lower.starts_with("trait ")
        || lower.starts_with("def ")
        || lower.starts_with("function ")
        || lower.starts_with("import ")
        || lower.starts_with("use ")
        || lower.contains("todo")
}

fn compact_repetitions(
    lines: &[String],
    strategy: RtkGroupingStrategy,
    category: CommandCategory,
) -> Vec<String> {
    if lines.len() <= EDGE_LINES * 2 + 1 {
        return lines.to_vec();
    }

    #[derive(Debug)]
    struct Item {
        text: String,
        collapsed: usize,
    }

    let mut items = Vec::<Item>::new();
    let mut seen = HashMap::<String, usize>::new();
    for (index, line) in lines.iter().enumerate() {
        let edge = index < EDGE_LINES || index >= lines.len().saturating_sub(EDGE_LINES);
        let forced = edge
            || is_important_line(line)
            || (category == CommandCategory::Git && path_regex().is_match(line));
        let signature = repetition_signature(line, strategy);
        if !forced && !signature.is_empty() {
            if let Some(item_index) = seen.get(&signature).copied() {
                items[item_index].collapsed += 1;
                continue;
            }
        }

        let item_index = items.len();
        items.push(Item {
            text: line.clone(),
            collapsed: 0,
        });
        if !edge && !signature.is_empty() {
            seen.entry(signature).or_insert(item_index);
        }
    }

    let mut output = Vec::new();
    for item in items {
        output.push(item.text);
        if item.collapsed > 0 {
            output.push(format!(
                "[... {} identical/similar items collapsed ...]",
                item.collapsed
            ));
        }
    }
    output
}

fn repetition_signature(line: &str, strategy: RtkGroupingStrategy) -> String {
    let normalized = line.trim().to_ascii_lowercase();
    match strategy {
        RtkGroupingStrategy::Conservative => normalized,
        RtkGroupingStrategy::Balanced => {
            let normalized = numeric_variation_regex().replace_all(&normalized, "#");
            identifier_variation_regex()
                .replace_all(&normalized, "_$VAR")
                .into_owned()
        }
        RtkGroupingStrategy::Aggressive => {
            let normalized = numeric_variation_regex().replace_all(&normalized, "#");
            let normalized = identifier_variation_regex().replace_all(&normalized, "_$VAR");
            whitespace_regex()
                .replace_all(normalized.trim(), " ")
                .into_owned()
        }
    }
}

fn limit_large_output(
    lines: &[String],
    strategy: RtkGroupingStrategy,
    category: CommandCategory,
) -> Vec<String> {
    let base_limit: usize = match strategy {
        RtkGroupingStrategy::Aggressive => 60,
        RtkGroupingStrategy::Balanced => 120,
        RtkGroupingStrategy::Conservative => 240,
    };
    let limit = match category {
        CommandCategory::Git => base_limit.saturating_mul(2),
        CommandCategory::Search | CommandCategory::FileRead => base_limit,
        _ => base_limit,
    };
    if lines.len() <= limit {
        return lines.to_vec();
    }

    let important = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| is_important_line(line).then_some(index))
        .collect::<HashSet<_>>();
    let selected = selected_indices(lines.len(), limit, &important);
    render_selected(lines, &selected, "lines")
        .lines()
        .map(str::to_owned)
        .collect()
}

fn selected_indices(total: usize, limit: usize, important: &HashSet<usize>) -> Vec<usize> {
    let mut selected = important.clone();
    for index in 0..EDGE_LINES.min(total) {
        selected.insert(index);
    }
    for index in total.saturating_sub(EDGE_LINES)..total {
        selected.insert(index);
    }

    if selected.len() < limit {
        let remaining = limit - selected.len();
        let stride =
            ((total.saturating_sub(EDGE_LINES * 2)).max(1) + remaining - 1) / remaining.max(1);
        let mut index = EDGE_LINES;
        while selected.len() < limit && index < total.saturating_sub(EDGE_LINES) {
            selected.insert(index);
            index = index.saturating_add(stride.max(1));
        }
    }

    let mut selected = selected.into_iter().collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

fn render_selected(lines: &[String], selected: &[usize], unit: &str) -> String {
    let mut output = Vec::new();
    let mut previous = None;
    for &index in selected {
        if previous.is_some_and(|previous_index| index > previous_index + 1) {
            output.push(format!(
                "[... {} {unit} omitted ...]",
                index - previous.unwrap_or(0) - 1
            ));
        }
        output.push(lines[index].clone());
        previous = Some(index);
    }
    output.join("\n")
}

fn count_payload_tokens(payload: &CompressiblePayload, context: &CompressionContext) -> u32 {
    context
        .token_counter
        .count_request(&payload.clone().into_openai_request())
}

fn refresh_message_token_counts(payload: &mut CompressiblePayload, context: &CompressionContext) {
    for message in &mut payload.messages {
        let content_tokens = match message.content.as_value() {
            Value::Null => 0,
            Value::String(text) => context.token_counter.count_text(&context.model, text),
            structured => context
                .token_counter
                .count_text(&context.model, &structured.to_string()),
        };
        let extra_tokens = if message.extra.is_empty() {
            0
        } else {
            context.token_counter.count_text(
                &context.model,
                &Value::Object(message.extra.clone()).to_string(),
            )
        };
        message.token_count = 4u32
            .saturating_add(
                context
                    .token_counter
                    .count_text(&context.model, &message.role),
            )
            .saturating_add(content_tokens)
            .saturating_add(extra_tokens);
    }
}

fn regex_cell(cell: &'static OnceLock<Regex>, pattern: &'static str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("RTK uses a static, valid regex"))
}

fn ansi_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"\x1B\[[0-?]*[ -/]*[@-~]")
}

fn osc_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"\x1B\][^\x07]*(?:\x07|\x1B\\)")
}

fn timestamp_only_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(
        &CELL,
        r"(?i)^\[?(?:\d{4}-\d{2}-\d{2}[t ]?)?(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d+)?)?(?:z| ?[+-]\d{2}:?\d{2})?\]?$",
    )
}

fn decorative_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"^[\s=*_~#─━═.\-]{4,}$")
}

fn spinner_only_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"^[\s|/\\\-⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏.]+$")
}

fn progress_only_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(
        &CELL,
        r"(?i)^(?:progress:\s*)?(?:\[[#=>.\-\s]+\]\s*)?\d{1,3}%\s*(?:\([\d.]+\s*(?:kb|mb|gb)(?:/s)?\))?$",
    )
}

fn percent_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"\b\d{1,3}%\b")
}

fn openai_key_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"\bsk-[A-Za-z0-9_-]{12,}\b")
}

fn aws_key_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"\bAKIA[0-9A-Z]{16}\b")
}

fn bearer_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
}

fn url_password_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(
        &CELL,
        r"(?i)(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<user>[^\s/@:]+:)[^\s/@]+@",
    )
}

fn search_match_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"(?m)^(.+?):(\d+)(?::(\d+))?:(.*)$")
}

fn numbered_line_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"(?m)^\s*\d+[:|]\s?(.*)$")
}

fn numbered_line_capture_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"^\s*\d+[:|]\s?(.*)$")
}

fn numeric_variation_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(
        &CELL,
        r"(?i)(?:0x)?[0-9a-f]{7,}|\b\d+(?:\.\d+)?(?:ms|s|kb|mb|gb|%)?\b",
    )
}

fn identifier_variation_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"_[A-Za-z0-9-]*\d+[A-Za-z0-9-]*")
}

fn whitespace_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(&CELL, r"\s+")
}

fn path_regex() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex_cell(
        &CELL,
        r"(?:[A-Za-z]:\\|/|\./|\.\./)?(?:[\w.@+ -]+[/\\])+[\w.@+ -]+(?:\.\w+)?",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::openai::OpenAIRequest;
    use serde_json::json;

    fn payload(command: &str, output: impl Into<String>) -> CompressiblePayload {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "tool",
                "content": output.into(),
                "tool_call_id": "call-1",
                "command": command,
                "name": "terminal",
                "vendor": {"trace": "kept"}
            }],
            "tools": [{"type": "function", "function": {"name": "terminal"}}],
            "metadata": {"order": [3, 2, 1]}
        }))
        .unwrap();
        CompressiblePayload::from(request)
    }

    fn context() -> CompressionContext {
        CompressionContext::new("gpt-4o", "test")
    }

    async fn compress(
        payload: &mut CompressiblePayload,
        strategy: RtkGroupingStrategy,
    ) -> EngineResult {
        RtkEngine::with_grouping_strategy(strategy)
            .compress(payload, &context())
            .await
    }

    fn text(payload: &CompressiblePayload) -> &str {
        payload.messages[0].content.as_text().unwrap()
    }

    #[test]
    fn registry_has_over_one_hundred_real_unique_handlers() {
        assert!(COMMAND_HANDLERS.len() >= 100);
        let names = COMMAND_HANDLERS
            .iter()
            .map(|handler| handler.name)
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), COMMAND_HANDLERS.len());
        assert!(COMMAND_HANDLERS.iter().all(|handler| {
            !handler.aliases.is_empty()
                && !handler.detection_tokens.is_empty()
                && handler.aliases.iter().all(|alias| !alias.trim().is_empty())
        }));
        let categories = COMMAND_HANDLERS
            .iter()
            .map(|handler| handler.category)
            .collect::<HashSet<_>>();
        assert_eq!(categories.len(), 8);
    }

    #[tokio::test]
    async fn filters_git_ansi_noise_duplicates_and_preserves_paths_and_failure() {
        let middle = (0..30)
            .map(|_| " M crates/core/src/lib.rs")
            .collect::<Vec<_>>()
            .join("\n");
        let output = format!(
            "\u{1b}[32mOn branch main\u{1b}[0m\n12:34:56\n=====\n{middle}\nwarning: conflict in crates/core/src/lib.rs\nprocess exited with code 1"
        );
        let mut payload = payload("git status --short", output.clone());
        let result = compress(&mut payload, RtkGroupingStrategy::Balanced).await;
        let filtered = text(&payload);

        assert!(result.applied);
        assert!(filtered.len() < output.len());
        assert!(!filtered.contains("\u{1b}["));
        assert!(!filtered.contains("12:34:56"));
        assert!(filtered.contains("crates/core/src/lib.rs"));
        assert!(filtered.contains("warning: conflict"));
        assert!(filtered.contains("code 1"));
    }

    #[tokio::test]
    async fn handles_test_build_lint_package_and_docker_profiles() {
        let cases = [
            (
                "cargo test",
                "test case_{} ... ok",
                "test result: FAILED. 99 passed; 1 failed",
            ),
            (
                "cargo build",
                "Compiling crate_{} v1.0.0",
                "error: could not compile `broken`",
            ),
            (
                "eslint .",
                "src/file{}.ts: warning: unused value",
                "10 problems (1 error, 9 warnings)",
            ),
            (
                "npm install",
                "resolved package-{} 100%",
                "npm error exit code 1",
            ),
            (
                "docker build .",
                "#4 {}% downloading",
                "ERROR: failed to solve: exit code: 1",
            ),
        ];

        for (command, repeated, summary) in cases {
            let output = (0..80)
                .map(|index| repeated.replace("{}", &index.to_string()))
                .chain(std::iter::once(summary.to_owned()))
                .collect::<Vec<_>>()
                .join("\n");
            let mut payload = payload(command, output.clone());
            let result = compress(&mut payload, RtkGroupingStrategy::Aggressive).await;
            assert!(result.applied, "{command}");
            assert!(text(&payload).len() < output.len(), "{command}");
            assert!(text(&payload).contains(summary), "{command}");
        }
    }

    #[tokio::test]
    async fn groups_search_results_by_file_with_explicit_omission_counts() {
        let mut lines = Vec::new();
        for index in 1..=40 {
            lines.push(format!("src/alpha.rs:{index}:let match_{index} = true;"));
            lines.push(format!("src/beta.rs:{index}:fn match_{index}() {{}}"));
        }
        let original = lines.join("\n");
        let mut payload = payload("rg match src", original.clone());
        let result = compress(&mut payload, RtkGroupingStrategy::Balanced).await;
        let output = text(&payload);

        assert!(result.applied);
        assert!(output.contains("src/alpha.rs:"));
        assert!(output.contains("src/beta.rs:"));
        assert!(output.contains("matches omitted"));
        assert!(output.contains("1: let match_1"));
        assert!(output.contains("40: let match_40"));
        assert!(output.len() < original.len());
    }

    #[tokio::test]
    async fn trims_file_reads_around_useful_sections_and_edges() {
        let lines = (1..=220)
            .map(|line| match line {
                1 => "1: first line".to_owned(),
                80 => "80: pub fn important_function() {".to_owned(),
                81 => "81:     warning!(\"preserve diagnostic\");".to_owned(),
                220 => "220: last line".to_owned(),
                _ => format!("{line}: ordinary implementation detail number {line}"),
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut payload = payload("cat src/large.rs", lines.clone());
        let result = compress(&mut payload, RtkGroupingStrategy::Balanced).await;
        let output = text(&payload);

        assert!(result.applied);
        assert!(output.contains("1: first line"));
        assert!(output.contains("80: pub fn important_function"));
        assert!(output.contains("warning!"));
        assert!(output.contains("220: last line"));
        assert!(output.contains("lines omitted"));
        assert!(output.len() < lines.len());
    }

    #[tokio::test]
    async fn redacts_all_supported_secret_shapes_before_returning_output() {
        let output = [
            "build started",
            "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456",
            "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz.1234567890",
            "registry=https://alice:super-secret-password@example.test/image",
            "build finished with exit code 0",
        ]
        .join("\n");
        let mut payload = payload("npm run build", output);
        let result = compress(&mut payload, RtkGroupingStrategy::Balanced).await;
        let output = text(&payload);

        assert!(result.applied);
        assert!(!output.contains("sk-abcdefghijklmnopqrstuvwxyz"));
        assert!(!output.contains("AKIAABCDEFGHIJKLMNOP"));
        assert!(!output.contains("abcdefghijklmnopqrstuvwxyz.1234567890"));
        assert!(!output.contains("super-secret-password"));
        assert!(output.matches(REDACTED).count() >= 4);
    }

    #[tokio::test]
    async fn grouping_strategies_are_configurable_and_monotonic() {
        let output = (0..180)
            .map(|index| format!("test generated_case_{index} ... ok in {index}ms"))
            .chain(std::iter::once(
                "test result: ok. 180 passed; 0 failed".to_owned(),
            ))
            .collect::<Vec<_>>()
            .join("\n");
        let mut conservative = payload("cargo test", output.clone());
        let mut balanced = payload("cargo test", output.clone());
        let mut aggressive = payload("cargo test", output);

        compress(&mut conservative, RtkGroupingStrategy::Conservative).await;
        compress(&mut balanced, RtkGroupingStrategy::Balanced).await;
        compress(&mut aggressive, RtkGroupingStrategy::Aggressive).await;

        assert!(text(&aggressive).len() <= text(&balanced).len());
        assert!(text(&balanced).len() <= text(&conservative).len());
        assert_eq!(
            RtkEngine::from_config(&RtkConfig {
                grouping_strategy: RtkGroupingStrategy::Aggressive,
            })
            .grouping_strategy(),
            RtkGroupingStrategy::Aggressive
        );
    }

    #[tokio::test]
    async fn skips_cache_protected_and_structured_json_content_exactly() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "content": "warning: sk-abcdefghijklmnopqrstuvwxyz123456",
                    "command": "cargo build",
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "role": "tool",
                    "content": "{\"token\":\"sk-abcdefghijklmnopqrstuvwxyz123456\",\"lines\":[1,2,3]}",
                    "command": "cargo build"
                }
            ]
        }))
        .unwrap();
        let mut payload = CompressiblePayload::from(request);
        let original = payload.clone();
        let result = compress(&mut payload, RtkGroupingStrategy::Aggressive).await;

        assert!(!result.applied);
        assert_eq!(payload, original);
    }

    #[tokio::test]
    async fn preserves_message_order_tool_fields_and_top_level_fields() {
        let long_output = (0..100)
            .map(|index| format!("Compiling dependency_{index} v1.0.0"))
            .chain(std::iter::once("Finished release target(s)".to_owned()))
            .collect::<Vec<_>>()
            .join("\n");
        let mut payload = payload("cargo build --release", long_output);
        let original_tools = payload.tool_definitions.clone();
        let original_extra = payload.extra.clone();
        let original_message_extra = payload.messages[0].extra.clone();
        let original_index = payload.messages[0].original_index;

        let result = compress(&mut payload, RtkGroupingStrategy::Balanced).await;

        assert!(result.applied);
        assert_eq!(payload.messages.len(), 1);
        assert_eq!(payload.messages[0].original_index, original_index);
        assert_eq!(payload.messages[0].extra, original_message_extra);
        assert_eq!(payload.tool_definitions, original_tools);
        assert_eq!(payload.extra, original_extra);
        assert!(result.tokens_after <= result.tokens_before);
    }

    #[tokio::test]
    async fn leaves_unrecognized_and_small_non_reducing_output_unchanged() {
        let mut unknown = payload("echo hello", "hello");
        let unknown_original = unknown.clone();
        let result = compress(&mut unknown, RtkGroupingStrategy::Balanced).await;
        assert!(!result.applied);
        assert_eq!(unknown, unknown_original);
        assert_eq!(result.tokens_before, result.tokens_after);

        let mut small = payload("git status", "On branch main\nnothing to commit");
        let small_original = small.clone();
        let result = compress(&mut small, RtkGroupingStrategy::Balanced).await;
        assert!(!result.applied);
        assert_eq!(small, small_original);
        assert_eq!(result.tokens_before, result.tokens_after);
    }

    #[tokio::test]
    async fn detects_output_without_command_metadata_and_preserves_first_last_five() {
        let output = (0..80)
            .map(|index| format!("Compiling crate_{index} v1.0.0"))
            .chain(std::iter::once(
                "Build failed with exit code 101".to_owned(),
            ))
            .collect::<Vec<_>>()
            .join("\n");
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": output}]
        }))
        .unwrap();
        let mut payload = CompressiblePayload::from(request);
        let result = compress(&mut payload, RtkGroupingStrategy::Aggressive).await;
        let output = text(&payload);

        assert!(result.applied);
        for index in 0..5 {
            assert!(output.contains(&format!("crate_{index}")));
        }
        for index in 76..80 {
            assert!(output.contains(&format!("crate_{index}")));
        }
        assert!(output.contains("exit code 101"));
    }
}
