//! Supply-chain regression tests
//! (spec: black-october-security-hardening, task 13, Req 6).
//!
//! `Cargo.lock` must stay committed at the repository root so the CI
//! cargo-deny gate and `cargo audit` local runs always check the exact
//! resolved dependency set. This is a binary crate, not a library: the
//! lockfile is part of the shipped supply chain and must be tracked. These
//! tests fail (never soft-skip) when the lockfile is missing, empty, or
//! matched by a .gitignore pattern; the tracked-by-git assertion runs in
//! CI, where a clean checkout makes the staging state unambiguous.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn git_repo_present() -> bool {
    repo_root().join(".git").exists()
}

fn run_git(args: &[&str]) -> std::process::Output {
    std::process::Command::new("git")
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn git")
}

#[test]
fn cargo_lock_exists_at_repo_root_and_is_non_empty() {
    let lock_path = repo_root().join("Cargo.lock");
    assert!(
        lock_path.is_file(),
        "Cargo.lock must exist at the repo root ({}); it pins the dependency \
         set checked by the CI cargo-deny gate and local `cargo audit` runs",
        lock_path.display()
    );

    let content = std::fs::read(&lock_path)
        .expect("Cargo.lock must be readable (check checkout permissions)");
    assert!(
        !content.is_empty(),
        "Cargo.lock must be non-empty — an empty lockfile means the resolved \
         dependency graph is no longer pinned"
    );

    let text = String::from_utf8_lossy(&content);
    assert!(
        text.contains("[[package]]"),
        "Cargo.lock must be a valid v3/v4 lockfile containing [[package]] entries"
    );
    assert!(
        text.contains("crates.io-index"),
        "Cargo.lock must pin registry sources (crates.io-index references)"
    );
}

#[test]
fn cargo_lock_is_not_gitignored() {
    // Deterministic text scan: no active (non-comment) .gitignore pattern may
    // mention Cargo.lock. Works even without a .git directory.
    let ignore_path = repo_root().join(".gitignore");
    if let Ok(text) = std::fs::read_to_string(&ignore_path) {
        for line in text.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') && trimmed.contains("Cargo.lock") {
                panic!(
                    ".gitignore line {:?} hides Cargo.lock; delete the entry so the \
                     lockfile can be committed and checked by the CI supply-chain gate",
                    line
                );
            }
        }
    }

    // Authoritative matcher when a git repo is present (a source tarball has
    // no .git and cannot be gitignored).
    if git_repo_present() {
        let ignored = run_git(&["check-ignore", "--", "Cargo.lock"])
            .status
            .success();
        assert!(
            !ignored,
            "git check-ignore matches Cargo.lock: a .gitignore pattern hides the \
             lockfile. Remove it so the lockfile can be tracked."
        );
    }
}

#[test]
fn cargo_lock_is_tracked_by_git() {
    // `ls-files` inspects the index: committed and staged (`git add`ed)
    // lockfiles both pass; an untracked one fails. The strict check runs in
    // CI only (GITHUB_ACTIONS/CI are preset there), because a local working
    // copy may legitimately hold an unstaged lockfile awaiting
    // `git add Cargo.lock` — the orchestrator stages it before every commit.
    // In CI a clean checkout makes the state deterministic: cargo may
    // regenerate an uncommitted lockfile, but it can never appear in the
    // index, so a forgotten commit fails here rather than silently pinning
    // whatever cargo happened to resolve.
    let in_ci = std::env::var_os("CI").is_some() || std::env::var_os("GITHUB_ACTIONS").is_some();
    if !in_ci {
        return;
    }

    let output = run_git(&["ls-files", "--error-unmatch", "--", "Cargo.lock"]);
    assert!(
        output.status.success(),
        "Cargo.lock is neither tracked nor staged by git. The CI supply-chain \
         gate must check the committed lockfile, not one regenerated at build \
         time. Run `git add Cargo.lock` before committing.\ngit output: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn deny_toml_exists_for_ci_supply_chain_gate() {
    let deny_path = repo_root().join("deny.toml");
    assert!(
        deny_path.is_file(),
        "deny.toml must exist at the repo root; the CI supply-chain gate job \
         runs cargo-deny against it"
    );
}
