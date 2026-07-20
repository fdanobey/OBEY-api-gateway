use ai_gateway::{
    compression::{
        config::{CompressionConfig, CustomPipelineConfig, EffectiveCompressionConfig},
        pipeline::{CompressionPipeline, CompressionRequestMetadata},
        protection::ProtectionScanner,
        token_counter::TokenCounter,
        CompressiblePayload, CompressionContext, CompressionEngine, CompressionLevel, EngineResult,
    },
    models::openai::{Message, OpenAIRequest},
};
use async_trait::async_trait;
use proptest::prelude::*;
use serde_json::{Map, Value};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

const CUSTOM_PIPELINE: &str = "short_circuit_property";
const CHAIN_LENGTH: usize = 3;

struct CountingEngine {
    name: String,
    messages_to_remove: usize,
    calls: Arc<AtomicUsize>,
    call_order: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl CompressionEngine for CountingEngine {
    fn name(&self) -> &str {
        &self.name
    }

    async fn compress(
        &self,
        payload: &mut CompressiblePayload,
        context: &CompressionContext,
    ) -> EngineResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.call_order.lock().unwrap().push(self.name.clone());

        let tokens_before = count_tokens(&context.token_counter, payload);
        payload.messages.truncate(
            payload
                .messages
                .len()
                .saturating_sub(self.messages_to_remove),
        );
        let tokens_after = count_tokens(&context.token_counter, payload);

        EngineResult {
            engine_name: self.name.clone(),
            tokens_before,
            tokens_after,
            duration_ms: 0,
            applied: tokens_after < tokens_before,
        }
    }
}

fn count_tokens(counter: &TokenCounter, payload: &CompressiblePayload) -> u32 {
    counter.count_request(&payload.clone().into_openai_request())
}

fn payload(message_count: usize) -> CompressiblePayload {
    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages: (0..message_count)
            .map(|index| Message {
                role: "assistant".to_owned(),
                content: Value::String(format!(
                    "compression property segment {index} with stable token content"
                )),
                extra: Map::new(),
            })
            .collect(),
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
    .into()
}

fn expected_counts(
    counter: &TokenCounter,
    original: &CompressiblePayload,
    reductions: [usize; CHAIN_LENGTH],
) -> Vec<u32> {
    let mut candidate = original.clone();
    let mut counts = vec![count_tokens(counter, &candidate)];

    for reduction in reductions {
        candidate
            .messages
            .truncate(candidate.messages.len().saturating_sub(reduction));
        counts.push(count_tokens(counter, &candidate));
    }

    counts
}

fn target_budget(counts: &[u32], stop_case: usize, target_seed: u32) -> u32 {
    if stop_case < CHAIN_LENGTH {
        let tokens_before = counts[stop_case];
        let tokens_after = counts[stop_case + 1];
        tokens_after + target_seed % (tokens_before - tokens_after)
    } else {
        counts[CHAIN_LENGTH].saturating_sub(1)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn target_budget_short_circuits_the_remaining_chain(
        initial_messages in 4usize..64,
        first_seed in any::<usize>(),
        second_seed in any::<usize>(),
        third_seed in any::<usize>(),
        target_seed in any::<u32>(),
        stop_case in 0usize..=CHAIN_LENGTH,
        use_custom_chain in any::<bool>(),
    ) {
        let first_reduction = 1 + first_seed % (initial_messages - 3);
        let second_reduction =
            1 + second_seed % (initial_messages - first_reduction - 2);
        let third_reduction =
            1 + third_seed % (initial_messages - first_reduction - second_reduction - 1);
        let reductions = [first_reduction, second_reduction, third_reduction];
        let chain = if use_custom_chain {
            ["rtk", "standard", "lite"]
        } else {
            ["lite", "standard", "aggressive"]
        };
        let original = payload(initial_messages);
        let token_counter = Arc::new(TokenCounter::new());
        let counts = expected_counts(&token_counter, &original, reductions);
        let target = target_budget(&counts, stop_case, target_seed);
        let expected_calls = if stop_case < CHAIN_LENGTH {
            stop_case + 1
        } else {
            CHAIN_LENGTH
        };
        let call_order = Arc::new(Mutex::new(Vec::new()));
        let call_counts = (0..CHAIN_LENGTH)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect::<Vec<_>>();
        let engines = chain
            .iter()
            .zip(reductions)
            .zip(&call_counts)
            .map(|((&name, messages_to_remove), calls)| {
                let engine: Arc<dyn CompressionEngine> = Arc::new(CountingEngine {
                    name: name.to_owned(),
                    messages_to_remove,
                    calls: Arc::clone(calls),
                    call_order: Arc::clone(&call_order),
                });
                (name.to_owned(), engine)
            })
            .collect::<HashMap<_, _>>();
        let mut config = CompressionConfig::default();
        let metadata = if use_custom_chain {
            config.custom_pipelines.insert(
                CUSTOM_PIPELINE.to_owned(),
                CustomPipelineConfig {
                    engines: chain.iter().map(|name| (*name).to_owned()).collect(),
                },
            );
            CompressionRequestMetadata {
                custom_pipeline: Some(CUSTOM_PIPELINE.to_owned()),
                ..CompressionRequestMetadata::default()
            }
        } else {
            CompressionRequestMetadata::default()
        };
        let pipeline = CompressionPipeline::with_engines(
            Arc::new(tokio::sync::RwLock::new(config)),
            Arc::clone(&token_counter),
            Arc::new(ProtectionScanner::default()),
            engines,
        );
        let context = CompressionContext {
            model: "gpt-4o".to_owned(),
            target_token_budget: Some(target),
            ..CompressionContext::default()
        };
        let effective = EffectiveCompressionConfig {
            enabled: true,
            level: CompressionLevel::Aggressive,
            auto_threshold_tokens: 0,
            caveman_output: false,
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = runtime.block_on(pipeline.compress_explicit(
            original,
            context,
            effective,
            metadata,
        ));
        let observed_order = call_order.lock().unwrap().clone();
        let expected_order = chain[..expected_calls]
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();

        prop_assert_eq!(observed_order, expected_order.clone());
        prop_assert_eq!(&result.engines_applied, &expected_order);
        prop_assert_eq!(result.engine_results.len(), expected_calls);
        prop_assert_eq!(result.final_tokens, counts[expected_calls]);
        prop_assert!(result.final_tokens <= result.original_tokens);
        prop_assert!(result
            .engine_results
            .iter()
            .all(|engine| engine.tokens_after <= engine.tokens_before));
        prop_assert!(result
            .engine_results
            .windows(2)
            .all(|pair| pair[0].tokens_after == pair[1].tokens_before));

        for (index, calls) in call_counts.iter().enumerate() {
            let expected = usize::from(index < expected_calls);
            prop_assert_eq!(calls.load(Ordering::SeqCst), expected);
        }

        if stop_case + 1 < CHAIN_LENGTH {
            prop_assert!(result.final_tokens <= target);
            prop_assert!(result.engines_applied.len() < CHAIN_LENGTH);
            prop_assert!(result.engine_results.len() < CHAIN_LENGTH);
        } else if stop_case < CHAIN_LENGTH {
            prop_assert!(result.final_tokens <= target);
            prop_assert_eq!(result.engines_applied.len(), CHAIN_LENGTH);
            prop_assert_eq!(result.engine_results.len(), CHAIN_LENGTH);
        } else {
            prop_assert!(result.final_tokens > target);
            prop_assert_eq!(result.engines_applied.len(), CHAIN_LENGTH);
            prop_assert_eq!(result.engine_results.len(), CHAIN_LENGTH);
        }
    }
}
