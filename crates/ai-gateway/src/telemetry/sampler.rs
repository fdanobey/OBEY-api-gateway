use super::propagation::TraceId;

/// Deterministic ratio sampler over the trace id (opentelemetry-
/// trace-export spec, D6). `Clone` so the per-request emission config
/// can freeze the entry-time sampler (Req 9.3).
#[derive(Clone)]
pub struct Sampler {
    threshold: u64,
}

impl Sampler {
    pub fn new(sampling_ratio: f64) -> Sampler {
        let clamped = sampling_ratio.clamp(0.0, 1.0);
        let threshold = if clamped >= 1.0 {
            u64::MAX
        } else {
            (clamped * 2f64.powi(64)) as u64
        };
        Sampler { threshold }
    }

    pub fn sample(&self, trace_id: &TraceId, inbound_sampled: Option<bool>) -> bool {
        if let Some(parent) = inbound_sampled {
            return parent;
        }
        let value = hash_trace_id(trace_id);
        value < self.threshold
    }
}

fn hash_trace_id(trace_id: &TraceId) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash_slice(trace_id, &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace_id(hex: &str) -> TraceId {
        let bytes = (0..16)
            .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).expect("hex trace id"))
            .collect::<Vec<u8>>();
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes);
        id
    }

    #[test]
    fn ratio_one_samples_everything_without_inbound() {
        let sampler = Sampler::new(1.0);
        for i in 0..100u8 {
            let id = trace_id(&format!("{:032x}", i as u128 + 1));
            assert!(sampler.sample(&id, None), "id {} must be sampled", i);
        }
    }

    #[test]
    fn ratio_zero_samples_nothing_without_inbound() {
        let sampler = Sampler::new(0.0);
        for i in 0..100u8 {
            let id = trace_id(&format!("{:032x}", i as u128 + 1));
            assert!(!sampler.sample(&id, None), "id {} must be unsampled", i);
        }
    }

    #[test]
    fn determinism_same_id_same_decision() {
        let sampler = Sampler::new(0.37);
        let id = trace_id("4bf92f3577b34da6a3ce929d0e0e4736");
        let first = sampler.sample(&id, None);
        for _ in 0..100 {
            assert_eq!(sampler.sample(&id, None), first);
        }
    }

    #[test]
    fn inbound_sampled_true_wins_over_ratio() {
        let sampler = Sampler::new(0.0);
        let id = trace_id("4bf92f3577b34da6a3ce929d0e0e4736");
        assert!(!sampler.sample(&id, None));
        assert!(sampler.sample(&id, Some(true)));
    }

    #[test]
    fn inbound_sampled_false_wins_over_ratio() {
        let sampler = Sampler::new(1.0);
        let id = trace_id("4bf92f3577b34da6a3ce929d0e0e4736");
        assert!(sampler.sample(&id, None));
        assert!(!sampler.sample(&id, Some(false)));
    }

    #[test]
    fn thresholds_are_deterministic_across_sampler_instances() {
        let id = trace_id("deadbeefdeadbeefdeadbeefdeadbeef");
        let a = Sampler::new(0.5).sample(&id, None);
        let b = Sampler::new(0.5).sample(&id, None);
        assert_eq!(a, b);
    }

    #[test]
    fn ratio_outside_range_is_clamped() {
        let id = trace_id("1234567890abcdef1234567890abcdef");
        assert!(!Sampler::new(-1.0).sample(&id, None));
        assert!(Sampler::new(2.0).sample(&id, None));
    }

    #[test]
    fn distribution_half_ratio_lands_near_half() {
        let sampler = Sampler::new(0.5);
        let mut sampled = 0u32;
        for i in 0..10_000u32 {
            let id = trace_id(&format!("{:032x}", i as u128 + 1));
            if sampler.sample(&id, None) {
                sampled += 1;
            }
        }
        let delta = (sampled as i64 - 5_000).abs();
        assert!(
            delta <= 500,
            "expected sampled within [4500, 5500], got {}",
            sampled
        );
    }
}
