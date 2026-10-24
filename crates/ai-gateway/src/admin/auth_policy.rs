//! Pure admin security-mode policy resolver (spec:
//! black-october-security-hardening, Component 1).
//!
//! Deterministic function of [`Config`] only — no IO, no async, no global
//! state. Callers (config validation, middleware, startup warnings) feed the
//! resolved [`AdminSecurityMode`] forward; this module never reaches for the
//! environment or network itself.

use crate::config::{AdminSecurityMode, Config};

/// Resolve the admin security posture for a loaded config (Req 1).
///
/// Steps are applied in order; first match wins:
///
/// 1. `admin.auth.enabled == true` → [`AdminSecurityMode::Authenticated`]
/// (existing Basic-auth behavior, unchanged).
/// 2. The `allow_unauthenticated_non_loopback` acknowledgment (Req 1.4)
/// combined with an explicit `open-opt-in` marker OR any marker-less
/// config (no `security_mode`, auth section present or not) →
/// [`AdminSecurityMode::OpenOptIn`]. Explicit `open-opt-in` without the
/// acknowledgment falls through to step 5 (fail closed).
/// 3. Marker-less legacy semantics split on `auth` section YAML presence:
/// loopback binds stay credential-free (Req 1.3) →
/// [`AdminSecurityMode::LoopbackOpen`]; on a non-loopback bind, an ABSENT
/// auth section preserves [`AdminSecurityMode::OpenWarnedLegacy`] (loudly
/// warned), while an EXPLICIT auth section with `enabled` unset/false
/// fails closed → [`AdminSecurityMode::Authenticated`] (Req 1.2 — the
/// operator wrote an auth section and left it disabled).
/// 4. An explicit `open-warned-legacy` marker follows the legacy split:
/// loopback → [`AdminSecurityMode::LoopbackOpen`], non-loopback →
/// [`AdminSecurityMode::OpenWarnedLegacy`] (operator-selected mode).
/// 5. Anything else (explicit marker on a non-loopback bind without the
/// opt-in acknowledgment) → [`AdminSecurityMode::Authenticated`]
/// (fail closed, Req 1.2).
pub fn resolve_admin_security_mode(config: &Config) -> AdminSecurityMode {
    let admin = &config.admin;

    // Step 1: explicit Basic auth always wins.
    if admin.auth.enabled {
        return AdminSecurityMode::Authenticated;
    }

    // Step 2: the unauthenticated-non-loopback acknowledgment (Req 1.4)
    // combined with an explicit `open-opt-in` marker or any marker-less
    // config (auth section present or not — an explicit section carrying
    // the flag is itself an operator acknowledgment).
    if admin.auth.allow_unauthenticated_non_loopback {
        let explicit_opt_in = matches!(admin.security_mode, Some(AdminSecurityMode::OpenOptIn));
        let marker_less = admin.security_mode.is_none();
        if explicit_opt_in || marker_less {
            return AdminSecurityMode::OpenOptIn;
        }
    }

    let loopback = is_loopback_bind(&config.server.host);

    // Steps 3-4: legacy semantics — no `security_mode` marker (or an
    // explicit `open-warned-legacy` marker, treated identically).
    let legacy = matches!(
        admin.security_mode,
        None | Some(AdminSecurityMode::OpenWarnedLegacy)
    );
    if legacy {
        if loopback {
            // Loopback stays credential-free (Req 1.3).
            return AdminSecurityMode::LoopbackOpen;
        }
        // An explicit `open-warned-legacy` marker, or a marker-less config
        // with NO auth section in YAML, preserves the legacy open behavior
        // (loudly warned). A marker-less config WITH an explicit auth
        // section (enabled unset/false) fails closed (Req 1.2).
        let legacy_open = matches!(
            admin.security_mode,
            Some(AdminSecurityMode::OpenWarnedLegacy)
        ) || !admin.auth_section_present;
        return if legacy_open {
            AdminSecurityMode::OpenWarnedLegacy
        } else {
            AdminSecurityMode::Authenticated
        };
    }

    // Explicit markers only reach here when they did not match an earlier
    // step. `Authenticated` is the requested enforcement; `LoopbackOpen` on
    // a loopback bind is honored; on a non-loopback bind it fails closed
    // (step 5).
    match admin.security_mode {
        Some(AdminSecurityMode::Authenticated) => AdminSecurityMode::Authenticated,
        Some(AdminSecurityMode::LoopbackOpen) if loopback => AdminSecurityMode::LoopbackOpen,
        // `OpenOptIn` without the acknowledgment, `LoopbackOpen` on a
        // non-loopback bind — fail closed.
        _ => AdminSecurityMode::Authenticated,
    }
}

/// Returns `true` when the given bind host addresses this machine only.
///
/// Covers the IPv4 loopback block `127.0.0.0/8` (parsed, not prefix-matched),
/// the IPv6 loopback `::1`, and `localhost` (case-insensitive). Empty hosts,
/// wildcards (`0.0.0.0`, `::`), bracketed IPv6 forms, and other hostnames or
/// addresses are NOT loopback.
pub fn is_loopback_bind(host: &str) -> bool {
    let trimmed = host.trim();
    if trimmed.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match trimmed.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(addr)) => addr.is_loopback(),
        Ok(std::net::IpAddr::V6(addr)) => addr.is_loopback(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdminAuthConfig;

    fn test_config(host: &str) -> Config {
        let yaml = format!(
            "server:\n  host: {}\n  port: 0\nproviders: []\nmodel_groups: []\n",
            host
        );
        serde_yaml::from_str(&yaml).expect("test config must parse")
    }

    fn auth_config(enabled: bool) -> AdminAuthConfig {
        AdminAuthConfig {
            enabled,
            username_env: None,
            password_env: None,
            allow_unauthenticated_non_loopback: false,
        }
    }

    #[test]
    fn auth_enabled_resolves_authenticated() {
        let mut config = test_config("192.168.1.110");
        config.admin.auth = auth_config(true);
        config.admin.security_mode = Some(AdminSecurityMode::OpenOptIn);
        config.admin.auth_section_present = false;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::Authenticated
        );
    }

    #[test]
    fn legacy_absent_auth_section_resolves_loopback_open_on_loopback_bind() {
        let mut config = test_config("127.0.0.1");
        config.admin.security_mode = None;
        config.admin.auth_section_present = false;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::LoopbackOpen
        );
    }

    #[test]
    fn legacy_absent_auth_section_resolves_open_warned_legacy_on_non_loopback_bind() {
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = None;
        config.admin.auth_section_present = false;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::OpenWarnedLegacy
        );
    }

    /// R1/Req 1.2 pin: an explicit `auth: {enabled: false}` section on a LAN
    /// bind fails closed to Authenticated.
    #[test]
    fn explicit_auth_section_enabled_false_fails_closed_on_non_loopback_bind() {
        let mut config = test_config("192.168.1.110");
        config.admin.auth = auth_config(false);
        config.admin.security_mode = None;
        config.admin.auth_section_present = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::Authenticated
        );
    }

    #[test]
    fn legacy_present_auth_section_resolves_loopback_open_on_loopback_bind() {
        let mut config = test_config("localhost");
        config.admin.security_mode = None;
        config.admin.auth_section_present = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::LoopbackOpen
        );
    }

    #[test]
    fn legacy_present_auth_section_fails_closed_on_non_loopback_bind() {
        // R1/Req 1.2 (flipped from OpenWarnedLegacy): an explicit `admin.auth`
        // section with `enabled` unset/false on a non-loopback bind fails
        // closed — the operator wrote an auth section and left it disabled.
        let mut config = test_config("0.0.0.0");
        config.admin.security_mode = None;
        config.admin.auth_section_present = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::Authenticated
        );
    }

    #[test]
    fn legacy_absent_auth_with_opt_in_flag_resolves_open_opt_in() {
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = None;
        config.admin.auth_section_present = false;
        config.admin.auth.allow_unauthenticated_non_loopback = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::OpenOptIn
        );
    }

    #[test]
    fn legacy_present_auth_with_opt_in_flag_resolves_open_opt_in() {
        // C4 reconciliation (Req 1.4): an explicit auth section carrying the
        // `allow_unauthenticated_non_loopback` acknowledgment with no
        // `security_mode` resolves OpenOptIn — the flag is an explicit
        // operator acknowledgment.
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = None;
        config.admin.auth_section_present = true;
        config.admin.auth.allow_unauthenticated_non_loopback = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::OpenOptIn
        );
    }

    #[test]
    fn explicit_open_opt_in_with_flag_resolves_open_opt_in() {
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = Some(AdminSecurityMode::OpenOptIn);
        config.admin.auth_section_present = true;
        config.admin.auth.allow_unauthenticated_non_loopback = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::OpenOptIn
        );
    }

    #[test]
    fn explicit_open_opt_in_without_flag_fails_closed() {
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = Some(AdminSecurityMode::OpenOptIn);
        config.admin.auth_section_present = true;
        config.admin.auth.allow_unauthenticated_non_loopback = false;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::Authenticated
        );
    }

    #[test]
    fn loopback_marker_on_loopback_bind_resolves_loopback_open() {
        let mut config = test_config("127.0.0.1");
        config.admin.security_mode = Some(AdminSecurityMode::LoopbackOpen);
        config.admin.auth_section_present = true;
        config.admin.auth.allow_unauthenticated_non_loopback = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::LoopbackOpen
        );
    }

    #[test]
    fn loopback_marker_on_non_loopback_bind_fails_closed() {
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = Some(AdminSecurityMode::LoopbackOpen);
        config.admin.auth_section_present = true;
        config.admin.auth.allow_unauthenticated_non_loopback = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::Authenticated
        );
    }

    #[test]
    fn authenticated_marker_resolves_authenticated() {
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = Some(AdminSecurityMode::Authenticated);
        config.admin.auth_section_present = true;
        config.admin.auth.allow_unauthenticated_non_loopback = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::Authenticated
        );
    }

    #[test]
    fn open_warned_legacy_marker_on_loopback_bind_resolves_loopback_open() {
        let mut config = test_config("127.0.0.1");
        config.admin.security_mode = Some(AdminSecurityMode::OpenWarnedLegacy);
        config.admin.auth_section_present = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::LoopbackOpen
        );
    }

    #[test]
    fn open_warned_legacy_marker_on_non_loopback_bind_stays_open_warned_legacy() {
        let mut config = test_config("192.168.1.110");
        config.admin.security_mode = Some(AdminSecurityMode::OpenWarnedLegacy);
        config.admin.auth_section_present = true;
        assert_eq!(
            resolve_admin_security_mode(&config),
            AdminSecurityMode::OpenWarnedLegacy
        );
    }

    #[test]
    fn loopback_detection_covers_expected_hosts() {
        assert!(is_loopback_bind("127.0.0.1"));
        assert!(is_loopback_bind("127.255.0.3"));
        assert!(is_loopback_bind("::1"));
        assert!(is_loopback_bind("localhost"));
        assert!(is_loopback_bind("LOCALHOST"));
        assert!(!is_loopback_bind("0.0.0.0"));
        assert!(!is_loopback_bind("::"));
        assert!(!is_loopback_bind("192.168.1.110"));
        assert!(!is_loopback_bind(""));
        assert!(!is_loopback_bind("[::1]"));
        assert!(!is_loopback_bind("127.0.0.1.example.com"));
        assert!(!is_loopback_bind("example.com"));
    }
}
