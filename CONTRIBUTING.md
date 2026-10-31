# Contributing to OBEY API Router

Thanks for taking the time to contribute. The following is a set of guidelines
for contributing to this repository. These are mostly guidelines, not rules. Use
your best judgment, and feel free to propose changes to this document in a pull
request.

## I just have a question

Please don't file an issue to ask a question. Use [Discussions](https://github.com/fdanobey/OBEY-api-gateway/discussions)
instead. You'll get faster results there, and the answer stays searchable for
everyone else.

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code. Please report unacceptable behavior by contacting the
maintainer privately via [GitHub](https://github.com/fdanobey).

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/).
Before creating a bug report, check the [docs](CONTEXT.md) and perform a
[cursory search](https://github.com/search?q=is%3Aissue+user%3Afdanobey&type=issues)
to see if the problem has already been reported. If you find a **Closed** issue
that seems like the same thing you're experiencing, open a new issue and include
a link to the original issue in the body of your new one.

When you are creating a bug report, please include as many details as possible
and fill out [the required template](.github/ISSUE_TEMPLATE/bug_report.yml).
The information it asks for helps maintainers resolve issues faster.

A good bug report contains:

- A clear and descriptive title
- The exact steps that reproduce the problem, including the request payload
  (redact API keys) and the config sections involved
- What you expected to happen and what actually happened
- The gateway version and OS you're running
- Relevant log output — request outcome rows from the admin Errors tab are
  usually enough; never paste provider API keys or Authorization headers

### Suggesting Enhancements

Enhancement suggestions are also tracked as [GitHub issues](https://github.com/fdanobey/OBEY-api-gateway/issues).
Fill out [the feature template](.github/ISSUE_TEMPLATE/feature_request.yml)
before submitting. Note that every YAML-configurable feature also needs
matching controls in the embedded admin panel, so features that require admin-UI
work take longer to land.

### Your First Code Contribution

Unsure where to start? Look for issues labeled
[`good first issue`](https://github.com/fdanobey/OBEY-api-gateway/labels/good%20first%20issue)
or [`help wanted`](https://github.com/fdanobey/OBEY-api-gateway/labels/help%20wanted).
The issues marked `good first issue` are small, self-contained, and don't
require deep knowledge of the routing internals.

### Pull Requests

- Fork the repo and create your branch from `development`.
- Open your pull request against the `development` branch — **not** `master`.
  `master` only receives changes through the release process (see
  [Releases](#releases) below), so PRs targeting it will be retargeted or
  closed.
- Follow the [Rust styleguide](#rust-styleguide) below.
- CI must pass: `cargo check -p ai-gateway --all-targets` with zero warnings,
  and the full test suite green.
- Fill out [the pull request template](.github/PULL_REQUEST_TEMPLATE.md).
- Feel free to open a draft PR early if you want feedback on the approach.

## Releases

`development` is the integration branch: everything lands there first, and CI
runs on every pull request into it and every push to it. `master` tracks
releases only. When the maintainer is ready to ship, they run the
**Release** workflow (Actions tab → Release → Run workflow → branch `master` →
pick patch/minor/major). That button press promotes `development` into
`master`: it bumps the crate version, merges the bump into `development`,
fast-forwards `master`, tags, and publishes the Windows build, installer, and
release notes automatically. Contributors never need to touch this process.

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* When only changing documentation, include `[ci skip]` in the commit title
* Consider starting the commit message with an applicable emoji:
  * :tada: `:tada:` when adding a feature
  * :wrench: `:wrench:` when fixing a bug
  * :books: `:books:` when writing docs
  * :art: `:art:` when improving the format/structure of code
  * :fire: `:fire:` when removing code or files

### Rust Styleguide

All Rust code is formatted and linted with the standard toolchain.

* Format with `cargo fmt --all` before committing.
* `cargo check -p ai-gateway --all-targets` must produce **zero warnings** —
  fix warnings rather than silencing them, and never add blanket `#[allow]`
  attributes; remove dead code or gate test-only helpers with `#[cfg(test)]`.
* Run the test suite before submitting: `cargo test -p ai-gateway`. If tests
  OOM on your machine, cap parallelism with `-- --test-threads=2`.
* The main source lives in `crates/ai-gateway/src/`; the routing logic is in
  `router/router.rs`, config structs (the YAML schema) in `config/mod.rs`.
* Match the existing patterns around you. When in doubt about how something
  works, read `AGENTS.md` and `CONTEXT.md` first — they document the
  non-obvious decisions.
* Behavior covered by guard tests (buffered dispatch, failover, rate-limit
  accounting) must not regress. If your change touches those paths, run the
  tests named in the relevant ADRs under `docs/adr/`.

## Additional Notes

### Issue and Pull Request Labels

This project separates issues into a few labels based on the part of the
gateway they touch (`area:routing`, `area:admin-ui`, `area:providers`, ...) and
severity (`priority:low` ... `priority:high`). See the
[label list](https://github.com/fdanobey/OBEY-api-gateway/labels) for the full
set. Maintainers apply labels during triage, so you don't need to worry about
them when filing.
