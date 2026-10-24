//! Outbound-fetch guard (SSRF defense) — black-october-security-hardening
//! Component 3.
//!
//! Central validation for every URL the gateway fetches on behalf of callers
//! (admin model discovery, connection tests) or operators (provider base
//! URLs, guardrail endpoints). The pure validator [`validate_outbound_url`]
//! never performs DNS itself — resolution is injected as a closure so unit
//! tests stay offline, and the redirect wrapper re-runs the same rules on
//! every hop via [`guarded_redirect_policy`].
//!
//! Rejections intentionally carry the **host string only** — never
//! userinfo/credentials, never the query string — so a guard failure can
//! never leak a secret through logs or error surfaces (Req 2.5).
//!
//! ## DNS rebinding (TOCTOU) posture
//!
//! [`guarded_redirect_policy`] re-resolves and re-validates every redirect
//! hop: each policy callback runs [`validate_outbound_url`] — including
//! fresh DNS via the injected resolver — on the hop's URL, so no hop ever
//! trusts the previous hop's check.
//!
//! For the initial request there is an accepted residual TOCTOU between
//! `validate_outbound_url` (resolve + classify) and reqwest's own
//! connect-time resolution: a hostile DNS server could answer validation
//! with a public address and the later connect with a private one. This
//! is an explicit design decision (no custom connector): the exposed
//! surface is admin-tier callers only (authenticated panel users, never
//! anonymous `/v1` traffic), named-provider base URLs are pinned by
//! operator config rather than caller-controlled DNS, and these fetches
//! are rare admin/config operations. Redirect hops — the classic
//! rebinding vector — are closed by the per-hop re-resolution above.
//!
//! This module is deliberately NOT wired into the `/v1` hot path; later
//! spec tasks attach it at the specific fetch call sites.

use std::collections::HashSet;
use std::net::{IpAddr, ToSocketAddrs};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Injectable DNS-resolution function: host string in, resolved IP list out.
/// Tests pass a closure; production uses [`std_resolve_host`].
pub type HostResolver = Arc<dyn Fn(&str) -> Vec<IpAddr> + Send + Sync>;

/// Trust tier for an outbound fetch. Only the caller tier exists today;
/// later spec tasks add tiers (e.g. operator-configured base URLs) as new
/// variants without changing the validator's signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchTier {
    /// Caller-supplied URL (request body/query — admin model discovery,
    /// connection tests). Strictest rules: https-only, no private hosts.
    Caller,
}

impl FetchTier {
    /// Per-tier settings applied by [`validate_outbound_url`].
    fn settings(self, config: &OutboundFetchConfig) -> TierSettings {
        match self {
            FetchTier::Caller => TierSettings {
                allow_private_hosts: config.caller.allow_private_hosts,
                allow_http: config.caller.allow_http,
            },
        }
    }
}

struct TierSettings {
    allow_private_hosts: bool,
    allow_http: bool,
}

/// `security.outbound_fetch.caller` — rules for caller-supplied URLs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CallerFetchConfig {
    /// Permit resolving to loopback/link-local/CGNAT/private/metadata
    /// addresses. Default `false` — caller input is untrusted.
    pub allow_private_hosts: bool,
    /// Permit plain `http:` scheme. Default `false` — https only.
    pub allow_http: bool,
    /// Exact hostnames (case-insensitive) whose private-IP resolution is
    /// permitted even when `allow_private_hosts` is `false` — e.g. a LAN
    /// LLM box at `ollama`. Matching is full-string equality only, never
    /// wildcard or suffix matching: an entry `ollama` must not match
    /// `evil-ollama.attacker.com` (suffix matching would let any attacker
    /// domain ending in a listed host bypass the guard).
    #[serde(default)]
    pub allowed_private_hosts: Vec<String>,
}

impl Default for CallerFetchConfig {
    fn default() -> Self {
        Self {
            allow_private_hosts: false,
            allow_http: false,
            allowed_private_hosts: Vec::new(),
        }
    }
}

/// `security.outbound_fetch` — outbound-fetch (SSRF) guard configuration.
/// Defaults mirror [`CallerFetchConfig::default`] (strictest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutboundFetchConfig {
    /// Caller-supplied URL tier (strictest). See [`CallerFetchConfig`].
    pub caller: CallerFetchConfig,
}

impl Default for OutboundFetchConfig {
    fn default() -> Self {
        Self {
            caller: CallerFetchConfig::default(),
        }
    }
}

/// Guard rejection. Carries the host string ONLY — never userinfo,
/// credentials, or the query string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OutboundFetchError {
    /// Scheme is neither https nor (permitted) http.
    #[error("outbound fetch to host {host:?} rejected: scheme {scheme:?} not allowed")]
    SchemeNotAllowed {
        /// Destination host (no userinfo, no credentials).
        host: String,
        /// Offending scheme.
        scheme: String,
    },
    /// Host (or one of its resolved addresses) is a forbidden IP class.
    #[error("outbound fetch to host {host:?} rejected: {reason}")]
    ForbiddenAddress {
        /// Destination host (no userinfo, no credentials).
        host: String,
        /// Which forbidden class the address fell into.
        reason: &'static str,
    },
    /// Hostname resolved to zero addresses.
    #[error("outbound fetch to host {host:?} rejected: host did not resolve")]
    UnresolvableHost {
        /// Destination host (no userinfo, no credentials).
        host: String,
    },
}

/// Classify a single IP against the forbidden set: loopback, link-local
/// (includes the `169.254.169.254` cloud-metadata endpoint), CGNAT,
/// unique-local, multicast, broadcast, unspecified, and (v4) RFC1918
/// private ranges. Returns the reason string when forbidden.
pub fn forbidden_ip_reason(ip: IpAddr) -> Option<&'static str> {
    match ip {
        IpAddr::V4(v4) => forbidden_ipv4_reason(v4),
        IpAddr::V6(v6) => forbidden_ipv6_reason(v6),
    }
}

fn forbidden_ipv4_reason(ip: std::net::Ipv4Addr) -> Option<&'static str> {
    if ip.is_loopback() {
        Some("loopback address")
    } else if ip.is_link_local() {
        // 169.254.0.0/16 — explicitly covers 169.254.169.254 (asserted in
        // tests) used by every major cloud's metadata service.
        Some("link-local address (cloud metadata range)")
    } else if {
        let o = ip.octets();
        o[0] == 100 && (o[1] & 0xC0) == 64
    } {
        // 100.64.0.0/10 — carrier-grade NAT.
        Some("CGNAT address")
    } else if ip.is_private() {
        Some("private address")
    } else if ip.is_multicast() {
        Some("multicast address")
    } else if ip.is_broadcast() {
        Some("broadcast address")
    } else if ip.is_unspecified() {
        Some("unspecified address")
    } else {
        None
    }
}

fn forbidden_ipv6_reason(ip: std::net::Ipv6Addr) -> Option<&'static str> {
    let segments = ip.segments();
    if ip.is_loopback() {
        Some("loopback address")
    } else if (segments[0] & 0xffc0) == 0xfe80 {
        // fe80::/10 link-local.
        Some("link-local address")
    } else if (segments[0] & 0xfe00) == 0xfc00 {
        // fc00::/7 unique-local.
        Some("unique-local address")
    } else if ip.is_multicast() {
        Some("multicast address")
    } else if ip.is_unspecified() {
        Some("unspecified address")
    } else if segments[0] == 0x0064 && segments[1] == 0xff9b && segments[2] == 0 && segments[3] == 0
    {
        // 64:ff9b::/96 NAT64 (RFC 6052 well-known prefix): the embedded
        // IPv4 address sits in the low 32 bits and is routable as IPv6
        // to the v4 destination, so re-run the v4 rules on it. Only
        // this /96 prefix is NAT64 — e.g. 64:ff9b:1::… is a normal
        // global address.
        let o = [
            (segments[6] >> 8) as u8,
            segments[6] as u8,
            (segments[7] >> 8) as u8,
            segments[7] as u8,
        ];
        forbidden_ipv4_reason(std::net::Ipv4Addr::from(o))
    } else if let Some(mapped) = ip.to_ipv4_mapped() {
        // ::ffff:a.b.c.d — re-run the v4 rules so mapped private/loopback
        // addresses cannot slip past the v6 checks.
        forbidden_ipv4_reason(mapped)
    } else if let Some(v4) = ip.to_ipv4() {
        // Deprecated IPv4-compatible form ::a.b.c.d (covers ::127.0.0.1
        // style) — treat like the mapped form: classify the embedded v4.
        forbidden_ipv4_reason(v4)
    } else {
        None
    }
}

/// Validate an outbound URL before fetch (caller tier rules).
///
/// * scheme must be `https`, or `http` when `caller.allow_http`;
/// * the host is resolved via the injected `resolver` (no DNS here —
/// tests pass closures; production passes [`std_resolve_host`]);
/// * rejection when ANY resolved IP (or a literal-IP host) is loopback,
/// link-local (incl. `169.254.169.254`), CGNAT, unique-local, multicast,
/// broadcast, unspecified, or (v4) RFC1918 private — unless
/// `caller.allow_private_hosts`, or the exact hostname is listed in
/// `caller.allowed_private_hosts`.
///
/// On rejection the error carries the host string only.
pub fn validate_outbound_url(
    url: &reqwest::Url,
    config: &OutboundFetchConfig,
    resolver: &dyn Fn(&str) -> Vec<IpAddr>,
) -> Result<(), OutboundFetchError> {
    let settings = FetchTier::Caller.settings(config);
    let host = url.host_str().unwrap_or_default().to_owned();

    let scheme = url.scheme();
    if scheme != "https" && !(scheme == "http" && settings.allow_http) {
        return Err(OutboundFetchError::SchemeNotAllowed {
            host,
            scheme: scheme.to_owned(),
        });
    }

    if settings.allow_private_hosts || host_in_private_allowlist(&config.caller, &host) {
        return Ok(());
    }

    if settings.allow_private_hosts {
        return Ok(());
    }

    // Literal-IP hosts: `host_str` keeps brackets on IPv6 forms, so trim
    // them before parsing; domain hostnames fall through to resolution.
    if let Ok(ip) = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
    {
        return match forbidden_ip_reason(ip) {
            Some(reason) => Err(OutboundFetchError::ForbiddenAddress { host, reason }),
            None => Ok(()),
        };
    }

    // Domain hostname: every resolved address must pass.
    let resolved = resolver(&host);
    if resolved.is_empty() {
        return Err(OutboundFetchError::UnresolvableHost { host });
    }
    for ip in resolved {
        if let Some(reason) = forbidden_ip_reason(ip) {
            return Err(OutboundFetchError::ForbiddenAddress { host, reason });
        }
    }
    Ok(())
}

/// Exact case-insensitive hostname match against the private-host
/// allowlist. Deliberately full-string equality (no wildcards, no suffix
/// matching): a `ollama` entry must exempt only `ollama` / `OLLAMA`, never
/// `evil-ollama.attacker.com` — suffix matching would hand any attacker
/// domain sharing a listed suffix a private-IP bypass.
fn host_in_private_allowlist(caller: &CallerFetchConfig, host: &str) -> bool {
    caller
        .allowed_private_hosts
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(host))
}

/// Standard resolver used in production (and by the redirect wrapper):
/// blocking `ToSocketAddrs`, deduplicated. These fetches are rare
/// admin/config operations, never the `/v1` hot path.
pub fn std_resolve_host(host: &str) -> Vec<IpAddr> {
    (host, 0u16)
        .to_socket_addrs()
        .map(|iter| iter.map(|addr| addr.ip()).collect::<HashSet<_>>())
        .map(|set| set.into_iter().collect())
        .unwrap_or_default()
}

/// Build a `reqwest` redirect policy that re-validates every hop URL with
/// the same rules as [`validate_outbound_url`]. On violation the attempt is
/// stopped with an error whose display includes the host only (never
/// userinfo or credentials).
pub fn guarded_redirect_policy(
    config: OutboundFetchConfig,
    resolver: HostResolver,
) -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(move |attempt| {
        match validate_outbound_url(attempt.url(), &config, &*resolver) {
            Ok(()) if attempt.previous().len() < 10 => attempt.follow(),
            Ok(()) => attempt.error("too many redirects"),
            Err(error) => {
                let (host, reason) = error.host_and_reason();
                tracing::warn!(
                target: "security::outbound_fetch",
                host = %host,
                reason = %reason,
                                    "outbound fetch guard rejected redirect hop"
                                );
                attempt.error(error)
            }
        }
    })
}

/// Structured rejection warning message: host + reason only — never
/// credentials, never userinfo, never the full URL. Call sites feed this
/// into `tracing::warn!` (the redirect policy above shows the convention).
pub fn outbound_fetch_rejection_warning(host: &str, reason: &str) -> String {
    format!("outbound fetch to host {host:?} rejected: {reason}")
}

impl OutboundFetchError {
    /// Host-only destination and rejection reason, suitable for admin responses.
    pub fn host_and_reason(&self) -> (&str, String) {
        match self {
            Self::SchemeNotAllowed { host, scheme } => {
                (host, format!("scheme {scheme:?} not allowed"))
            }
            Self::ForbiddenAddress { host, reason } => (host, (*reason).to_owned()),
            Self::UnresolvableHost { host } => (host, "host did not resolve".to_owned()),
        }
    }

    /// Warning line for structured logging — host only, never credentials.
    /// Test-only for now: production logging goes through
    /// [`OutboundFetchError::host_and_reason`] (admin wiring); promoted
    /// out of `#[cfg(test)]` when a caller lands.
    #[cfg(test)]
    pub fn warning(&self) -> String {
        let (host, reason) = self.host_and_reason();
        outbound_fetch_rejection_warning(host, &reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(allow_private_hosts: bool, allow_http: bool) -> OutboundFetchConfig {
        OutboundFetchConfig {
            caller: CallerFetchConfig {
                allow_private_hosts,
                allow_http,
                allowed_private_hosts: Vec::new(),
            },
        }
    }

    fn strict_config() -> OutboundFetchConfig {
        config(false, false)
    }

    fn noop_resolver(_: &str) -> Vec<IpAddr> {
        Vec::new()
    }

    fn public_resolver(_: &str) -> Vec<IpAddr> {
        vec![[93, 184, 216, 34].into()]
    }

    fn loopback_resolver(_: &str) -> Vec<IpAddr> {
        vec![[127, 0, 0, 1].into()]
    }

    fn parse(url: &str) -> reqwest::Url {
        reqwest::Url::parse(url).expect("test URL must parse")
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().expect("test IP must parse")
    }

    #[test]
    fn literal_ip_reject_matrix() {
        let forbidden = [
            ("127.0.0.1", "loopback address"),
            ("127.255.0.13", "loopback address"),
            (
                "169.254.169.254",
                "link-local address (cloud metadata range)",
            ),
            ("169.254.0.1", "link-local address (cloud metadata range)"),
            ("100.64.0.1", "CGNAT address"),
            ("100.127.255.254", "CGNAT address"),
            ("10.0.0.5", "private address"),
            ("192.168.1.110", "private address"),
            ("172.16.9.9", "private address"),
            ("224.0.0.1", "multicast address"),
            ("255.255.255.255", "broadcast address"),
            ("0.0.0.0", "unspecified address"),
            ("[::1]", "loopback address"),
            ("[fe80::1]", "link-local address"),
            ("[fc00::1]", "unique-local address"),
            ("[fd12:3456::a]", "unique-local address"),
            ("[ff02::1]", "multicast address"),
            ("[::]", "unspecified address"),
            // The url crate normalizes IPv4-mapped literals to hex form
            // (::ffff:7f00:1); classification still unmaps to the v4 rules.
            ("[::ffff:7f00:1]", "loopback address"),
            (
                "[::ffff:a9fe:a9fe]",
                "link-local address (cloud metadata range)",
            ),
        ];

        for (text, expected_reason) in forbidden {
            let url = parse(&format!("https://{text}/x"));
            let err = validate_outbound_url(&url, &strict_config(), &noop_resolver)
                .expect_err("literal forbidden IP must be rejected");
            match &err {
                OutboundFetchError::ForbiddenAddress { host, reason } => {
                    // `host_str` keeps brackets on IPv6 literals; that is the
                    // host string callers see and errors must carry.
                    assert_eq!(host, text, "host must be preserved for {text}");
                    assert_eq!(*reason, expected_reason, "reason mismatch for {text}");
                }
                other => panic!("expected ForbiddenAddress for {text}, got {other:?}"),
            }
            // allow_private_hosts=true lifts the block (scheme is https).
            assert!(
                validate_outbound_url(&url, &config(true, false), &noop_resolver).is_ok(),
                "allow_private_hosts must permit {text}"
            );
        }
    }

    #[test]
    fn cloud_metadata_endpoint_explicitly_forbidden() {
        let reason =
            forbidden_ip_reason(ip("169.254.169.254")).expect("metadata IP must be forbidden");
        assert!(reason.contains("link-local"));
    }

    #[test]
    fn allowed_private_host_listed_permits_private_resolution() {
        let mut cfg = strict_config();
        cfg.caller.allowed_private_hosts = vec!["ollama".to_string()];

        // Listed host + private resolution → allowed.
        let listed = parse("https://ollama/v1/models");
        assert!(
            validate_outbound_url(&listed, &cfg, &loopback_resolver).is_ok(),
            "listed host must bypass private-IP rejection"
        );
        // Case-insensitive exact match.
        let upper = parse("https://OLLAMA/v1/models");
        assert!(
            validate_outbound_url(&upper, &cfg, &loopback_resolver).is_ok(),
            "allowlist match must be case-insensitive"
        );

        // Same host unlisted → rejected.
        let unlisted = parse("https://ollama.example/");
        assert!(
            validate_outbound_url(&unlisted, &strict_config(), &loopback_resolver).is_err(),
            "unlisted host must keep private-IP rejection"
        );

        // Suffix spoof: `evil-ollama.attacker.com` must NOT match `ollama`.
        let spoof = parse("https://evil-ollama.attacker.com/");
        let err = validate_outbound_url(&spoof, &cfg, &loopback_resolver)
            .expect_err("suffix-spoofing host must not match allowlist");
        assert!(matches!(err, OutboundFetchError::ForbiddenAddress { .. }));
    }

    #[test]
    fn forbidden_ipv6_nat64_and_ipv4_compatible_forms() {
        // IPv4-mapped loopback.
        assert_eq!(
            forbidden_ip_reason(ip("::ffff:127.0.0.1")),
            Some("loopback address")
        );
        // IPv4-compatible loopback (::127.0.0.1 deprecated form).
        assert_eq!(
            forbidden_ip_reason(ip("::127.0.0.1")),
            Some("loopback address")
        );
        // NAT64 well-known prefix 64:ff9b::/96 embedding a private v4.
        assert_eq!(
            forbidden_ip_reason(ip("64:ff9b::192.168.1.1")),
            Some("private address")
        );
        // Outside the /96 (64:ff9b:1::/…) the address is global — allowed.
        assert_eq!(forbidden_ip_reason(ip("64:ff9b:1::192.168.1.1")), None);
        // NAT64 embedding a public v4 stays allowed.
        assert_eq!(forbidden_ip_reason(ip("64:ff9b::93.184.216.34")), None);
    }

    #[test]
    fn public_literal_ips_pass() {
        for text in ["1.1.1.1", "93.184.216.34", "[2606:4700:4700::1111]"] {
            let url = parse(&format!("https://{text}/"));
            assert!(
                validate_outbound_url(&url, &strict_config(), &noop_resolver).is_ok(),
                "public literal {text} must pass"
            );
        }
    }

    #[test]
    fn scheme_checks() {
        let ftp = parse("ftp://example.com/x");
        let err = validate_outbound_url(&ftp, &strict_config(), &public_resolver)
            .expect_err("ftp must be rejected");
        assert!(matches!(err, OutboundFetchError::SchemeNotAllowed { .. }));

        let http = parse("http://example.com/x");
        assert!(
            validate_outbound_url(&http, &strict_config(), &public_resolver).is_err(),
            "http rejected by default"
        );
        assert!(
            validate_outbound_url(&http, &config(false, true), &public_resolver).is_ok(),
            "allow_http=true permits http"
        );
        // Scheme is checked before resolution: a resolver that would fail
        // still reports the scheme error first.
        assert!(matches!(
            validate_outbound_url(&ftp, &strict_config(), &noop_resolver),
            Err(OutboundFetchError::SchemeNotAllowed { .. })
        ));
    }

    #[test]
    fn hostname_resolving_to_private_is_rejected() {
        let url = parse("https://rebind.example/");
        let mixed = |_host: &str| vec![ip("93.184.216.34"), ip("10.1.2.3")];

        // Any resolved IP private → reject (DNS rebinding defense).
        let err = validate_outbound_url(&url, &strict_config(), &mixed)
            .expect_err("mixed resolution containing private IP must reject");
        match &err {
            OutboundFetchError::ForbiddenAddress { host, reason } => {
                assert_eq!(host, "rebind.example");
                assert_eq!(*reason, "private address");
            }
            other => panic!("expected ForbiddenAddress, got {other:?}"),
        }

        // All-public resolution passes.
        assert!(
            validate_outbound_url(&url, &strict_config(), &public_resolver).is_ok(),
            "public-only resolution must pass"
        );

        // Hostname resolving straight to loopback (DNS rebinding shape).
        let err = validate_outbound_url(&url, &strict_config(), &loopback_resolver)
            .expect_err("loopback resolution must reject");
        assert!(matches!(err, OutboundFetchError::ForbiddenAddress { .. }));

        // allow_private_hosts=true overrides.
        assert!(
            validate_outbound_url(&url, &config(true, false), &mixed).is_ok(),
            "allow_private_hosts must bypass resolved-IP checks"
        );
    }

    #[test]
    fn unresolvable_hostname_rejected() {
        let url = parse("https://does-not-resolve.invalid/");
        let err = validate_outbound_url(&url, &strict_config(), &noop_resolver)
            .expect_err("empty resolution must reject");
        assert!(matches!(err, OutboundFetchError::UnresolvableHost { .. }));
    }

    #[test]
    fn warning_and_errors_never_carry_credentials() {
        let sneaky = parse("https://user:sk-secret@127.0.0.1:8080/v1/models");
        let err = validate_outbound_url(&sneaky, &strict_config(), &noop_resolver)
            .expect_err("userinfo cannot bypass guard");
        let rendered = err.to_string();
        assert!(rendered.contains("127.0.0.1"));
        assert!(
            !rendered.contains("sk-secret"),
            "error leaked credentials: {rendered}"
        );
        assert!(
            !rendered.contains("user:"),
            "error leaked userinfo: {rendered}"
        );

        let warning = err.warning();
        assert!(warning.contains("127.0.0.1"));
        assert!(
            !warning.contains("sk-secret"),
            "warning leaked credentials: {warning}"
        );
        assert_eq!(
            outbound_fetch_rejection_warning("host.example", "loopback address"),
            "outbound fetch to host \"host.example\" rejected: loopback address"
        );
    }

    #[test]
    fn std_resolver_returns_loopback_for_localhost() {
        let localhost = std_resolve_host("localhost");
        assert!(!localhost.is_empty(), "localhost must resolve in test envs");
        assert!(
            localhost.iter().all(|ip| ip.is_loopback()),
            "localhost must only resolve to loopback, got {localhost:?}"
        );
        // No assertion on `.invalid` hostnames: corporate resolvers with
        // wildcard-DNS answer them, flaking the empty-result expectation.
    }

    proptest::proptest! {
        /// Every possible IPv4 address classifies identically to a plain
        /// range-membership reference implementation.
        #[test]
        fn ipv4_classification_matches_reference(raw in proptest::prelude::any::<u32>()) {
            let ip = std::net::Ipv4Addr::from(raw);
            let o = ip.octets();
            let expected: Option<&str> = if o[0] == 127 {
                Some("loopback address")
            } else if o[0] == 169 && o[1] == 254 {
                Some("link-local address (cloud metadata range)")
            } else if o[0] == 100 && (o[1] & 0xC0) == 64 {
                Some("CGNAT address")
            } else if o[0] == 10
                || (o[0] == 172 && (o[1] & 0xF0) == 16)
                || (o[0] == 192 && o[1] == 168)
            {
                Some("private address")
            } else if (o[0] & 0xF0) == 224 {
                Some("multicast address")
            } else if raw == u32::MAX {
                Some("broadcast address")
            } else if raw == 0 {
                Some("unspecified address")
            } else {
                None
            };
            proptest::prelude::prop_assert_eq!(
                forbidden_ip_reason(IpAddr::V4(ip)),
                expected
            );
        }
    }

    #[tokio::test]
    async fn redirect_policy_revalidates_and_stops_on_loopback_hop() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/start"))
            .respond_with(
                ResponseTemplate::new(302).insert_header("Location", "http://127.0.0.1:1/"),
            )
            .expect(1)
            .mount(&server)
            .await;

        // allow_http=true so BOTH the mock first hop (loopback-bound, http)
        // and the loopback redirect target clear the scheme check — the only
        // remaining rejection reason is the IP class, which the assertions
        // pin to loopback specifically.
        let client = reqwest::Client::builder()
            .redirect(guarded_redirect_policy(
                config(false, true),
                Arc::new(|h: &str| std_resolve_host(h)),
            ))
            .build()
            .unwrap();

        let result = client.get(format!("{}/start", server.uri())).send().await;
        let err = result.expect_err("redirect to loopback must fail");
        let rendered = error_chain_string(&err);
        assert!(
            rendered.contains("127.0.0.1"),
            "error must name the loopback host: {rendered}"
        );
        assert!(
            rendered.contains("loopback"),
            "rejection must be the loopback IP reason, not a scheme rejection: {rendered}"
        );
        assert!(
            !rendered.contains("scheme"),
            "scheme was allowed; must not appear: {rendered}"
        );
        assert!(
            !rendered.contains("sk-"),
            "no credentials in redirect error"
        );
    }

    /// reqwest wraps policy errors in `error following redirect for url …`;
    /// render the full source chain so guard reasons are visible to tests.
    #[cfg(test)]
    fn error_chain_string(err: &reqwest::Error) -> String {
        let mut rendered = err.to_string();
        let mut source = std::error::Error::source(err);
        while let Some(error) = source {
            rendered.push_str(" | ");
            rendered.push_str(&error.to_string());
            source = std::error::Error::source(error);
        }
        rendered
    }

    #[tokio::test]
    async fn redirect_policy_follows_hop_when_rules_allow() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/hop"))
            .respond_with(ResponseTemplate::new(302).insert_header("Location", "/final"))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/final"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .expect(1)
            .mount(&server)
            .await;

        // The mock server binds loopback, so per-hop re-validation only
        // allows the (same-host) redirect when private hosts are permitted —
        // proving the policy evaluates hop URLs rather than skipping them.
        let client = reqwest::Client::builder()
            .redirect(guarded_redirect_policy(
                config(true, true),
                Arc::new(|h: &str| std_resolve_host(h)),
            ))
            .build()
            .unwrap();

        let response = client
            .get(format!("{}/hop", server.uri()))
            .send()
            .await
            .expect("permitted redirect chain must succeed");
        assert_eq!(response.text().await.unwrap(), "ok");
    }

    #[tokio::test]
    async fn redirect_policy_stops_after_ten_hops() {
        // Self-redirecting chain; the mock breaks the loop at hop 12 so a
        // policy that (incorrectly) allowed >10 hops would finish clean.
        let server = MockServer::start().await;
        for hop in 0..12u32 {
            let next = hop + 1;
            let response = if next >= 12 {
                ResponseTemplate::new(200).set_body_string("done")
            } else {
                ResponseTemplate::new(302).insert_header("Location", format!("/chain/{next}"))
            };
            Mock::given(method("GET"))
                .and(path(format!("/chain/{hop}")))
                .respond_with(response)
                .mount(&server)
                .await;
        }

        // Loopback-bound http mock: allow private + http so hop URLs pass
        // validation and only the hop limit can fire.
        let client = reqwest::Client::builder()
            .redirect(guarded_redirect_policy(
                config(true, true),
                Arc::new(|h: &str| std_resolve_host(h)),
            ))
            .build()
            .unwrap();

        let result = client.get(format!("{}/chain/0", server.uri())).send().await;
        let err = result.expect_err("chain longer than 10 hops must be stopped");
        assert!(
            error_chain_string(&err).contains("too many redirects"),
            "expected hop-limit error, got: {err}"
        );
    }
}
