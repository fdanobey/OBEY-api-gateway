# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| latest `master` | yes |
| tagged releases | latest release only |
| anything older | no |

## Reporting a Vulnerability

Do not open a public issue for a security problem. Use
[private vulnerability reporting](https://github.com/fdanobey/OBEY-api-gateway/security/advisories/new)
instead, or contact the repo owner privately via GitHub
([@fdanobey](https://github.com/fdanobey)). Expect a response within a few days.

Please include what you found, how you found it, and any proof of concept you
can share. This gateway handles provider API keys, admin panel access, and
request logs, so anything that exposes credentials, lets a caller bypass
guardrails, or crosses admin/user boundaries is in scope.

## Scope

- The `ai-gateway` crate and its embedded admin/dashboard panels
- Configuration handling, hot-reload, and anything that parses untrusted input
- Anything requiring authentication or authorization

Out of scope: vulnerabilities in dependencies (report those upstream) and
issues that require already-compromised provider credentials.

## What happens next

1. You get an acknowledgment and a tracking advisory.
2. The maintainers investigate and fix on a private branch.
3. A patched release goes out, then the advisory is published with credit.
