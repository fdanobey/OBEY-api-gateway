//! Responses API → Chat Completions request translator.
//!
//! Converts a [`ResponsesRequest`] into an [`OpenAIRequest`] for downstream
//! provider dispatch. Implements the translation spec from `design.md` §2.
//!
//! # Direction
//!
//! This is the inverse of [`crate::codex::translate_request`]:
//! Responses API (front door) → Chat Completions (backend).

use serde_json::{json, Map, Value};

use crate::models::openai::{Message, OpenAIRequest};
use crate::responses::{
    EasyInputContent, EasyInputMessage, FunctionCall, FunctionCallOutput,
    FunctionCallOutputContent, InputContentPart, InputImage, InputItem,
    Message as ResponsesMessage, OutputContentPart, OutputFunctionCall, OutputItem,
    OutputMessage, ResponsesInput, ResponsesRequest,
    ResponsesTranslationError, TextFormat, ToolChoice, ToolDefinition, TypedInputItem,
};

/// Stored conversation history for `previous_response_id` replay.
#[derive(Debug, Clone, Default)]
pub struct StoredConversation {
    pub input_items: Vec<InputItem>,
    pub output_items: Vec<OutputItem>,
}

/// Translation context provided by the caller.
pub struct TranslationContext<'a> {
    pub resolved_model: &'a str,
    pub model_supports_reasoning: bool,
}

/// Translate a Responses API request into a Chat Completions request.
///
/// # Errors
///
/// Returns [`ResponsesTranslationError::UnsupportedField`] for features that
/// cannot be translated:
/// - `input_audio` content parts
/// - `input_file` content parts
/// - Hosted tool types (web_search, file_search, etc.)
/// - `conversation` param
/// - `background` param
/// - `context_management` param
/// - `item_reference` input items
pub fn translate(
    req: &ResponsesRequest,
    history: Option<StoredConversation>,
    ctx: &TranslationContext<'_>,
) -> Result<OpenAIRequest, ResponsesTranslationError> {
    reject_unsupported_fields(req)?;

    let mut messages = Vec::new();

    if let Some(instructions) = &req.instructions {
        messages.push(Message {
            role: "system".to_string(),
            content: Value::String(instructions.clone()),
            extra: Map::new(),
        });
    }

    if let Some(stored) = history {
        replay_history(&stored, &mut messages)?;
    }

    translate_input(&req.input, &mut messages)?;

    let mut extra = Map::new();

    if let Some(max_tokens) = req.max_output_tokens {
        extra.insert("max_tokens".to_string(), json!(max_tokens));
    }

    if let Some(text) = &req.text {
        if let Some(format) = &text.format {
            extra.insert("response_format".to_string(), translate_text_format(format));
        }
    }

    if ctx.model_supports_reasoning {
        if let Some(reasoning) = &req.reasoning {
            if let Some(effort) = &reasoning.effort {
                extra.insert("reasoning_effort".to_string(), json!(effort));
            }
        }
    }

    if !req.tools.is_empty() {
        extra.insert("tools".to_string(), translate_tools(&req.tools));
    }

    if let Some(tc) = &req.tool_choice {
        extra.insert("tool_choice".to_string(), translate_tool_choice(tc));
    }

    if let Some(parallel) = req.parallel_tool_calls {
        extra.insert("parallel_tool_calls".to_string(), json!(parallel));
    }

    if let Some(temp) = req.temperature {
        extra.insert("temperature".to_string(), json!(temp));
    }

    if let Some(top_p) = req.top_p {
        extra.insert("top_p".to_string(), json!(top_p));
    }

    if req.stream {
        extra.insert("stream".to_string(), json!(true));
        extra.insert(
            "stream_options".to_string(),
            json!({"include_usage": true}),
        );
    }

    for (k, v) in &req.extra {
        extra.insert(k.clone(), v.clone());
    }

    Ok(OpenAIRequest {
        model: ctx.resolved_model.to_string(),
        messages,
        stream: req.stream,
        temperature: req.temperature,
        max_tokens: req.max_output_tokens,
        extra,
    })
}

fn reject_unsupported_fields(req: &ResponsesRequest) -> Result<(), ResponsesTranslationError> {
    if req.extra.contains_key("conversation") {
        return Err(ResponsesTranslationError::UnsupportedField {
            field: "conversation",
        });
    }
    if req.extra.contains_key("background") {
        return Err(ResponsesTranslationError::UnsupportedField {
            field: "background",
        });
    }
    if req.extra.contains_key("context_management") {
        return Err(ResponsesTranslationError::UnsupportedField {
            field: "context_management",
        });
    }

    for tool in &req.tools {
        reject_hosted_tool(&tool.r#type)?;
    }

    match &req.input {
        ResponsesInput::Text(_) => {}
        ResponsesInput::Items(items) => {
            for item in items {
                reject_input_item(item)?;
            }
        }
    }

    Ok(())
}

const HOSTED_TOOL_TYPES: &[&str] = &[
    "web_search",
    "file_search",
    "computer_20241022",
    "computer_20250124",
    "code_interpreter",
    "image_generation",
    "mcp",
    "custom",
    "apply_patch",
    "tool_search",
];

fn reject_hosted_tool(tool_type: &str) -> Result<(), ResponsesTranslationError> {
    if HOSTED_TOOL_TYPES.contains(&tool_type) {
        return Err(ResponsesTranslationError::UnsupportedField {
            field: Box::leak(format!("tool type '{}'", tool_type).into_boxed_str()),
        });
    }
    Ok(())
}

fn reject_input_item(item: &InputItem) -> Result<(), ResponsesTranslationError> {
    match item {
        InputItem::Easy(easy) => {
            reject_easy_input_content(&easy.content)?;
        }
        InputItem::Typed(typed) => {
            reject_typed_input_item(typed)?;
        }
    }
    Ok(())
}

fn reject_easy_input_content(content: &EasyInputContent) -> Result<(), ResponsesTranslationError> {
    match content {
        EasyInputContent::Text(_) => {}
        EasyInputContent::Parts(parts) => {
            for part in parts {
                reject_content_part(part)?;
            }
        }
    }
    Ok(())
}

fn reject_content_part(part: &InputContentPart) -> Result<(), ResponsesTranslationError> {
    match part {
        InputContentPart::Text(_) => {}
        InputContentPart::Image(img) => {
            if img.file_id.is_some() {
                return Err(ResponsesTranslationError::UnsupportedField {
                    field: "input_image.file_id",
                });
            }
        }
        InputContentPart::Other(v) => {
            let part_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if part_type == "input_audio" {
                return Err(ResponsesTranslationError::UnsupportedField {
                    field: "input_audio",
                });
            }
            if part_type == "input_file" {
                return Err(ResponsesTranslationError::UnsupportedField {
                    field: "input_file",
                });
            }
        }
    }
    Ok(())
}

fn reject_typed_input_item(item: &TypedInputItem) -> Result<(), ResponsesTranslationError> {
    match item {
        TypedInputItem::Message(msg) => {
            for part in &msg.content {
                reject_content_part(&part)?;
            }
        }
        TypedInputItem::FunctionCall(_) => {}
        TypedInputItem::FunctionCallOutput(_) => {}
        TypedInputItem::Reasoning(_) => {}
        TypedInputItem::ItemReference(_) => {
            return Err(ResponsesTranslationError::UnsupportedField {
                field: "item_reference",
            });
        }
    }
    Ok(())
}

fn translate_input(input: &ResponsesInput, messages: &mut Vec<Message>) -> Result<(), ResponsesTranslationError> {
    match input {
        ResponsesInput::Text(text) => {
            messages.push(Message {
                role: "user".to_string(),
                content: Value::String(text.clone()),
                extra: Map::new(),
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                translate_input_item(item, messages)?;
            }
        }
    }
    Ok(())
}

fn translate_input_item(item: &InputItem, messages: &mut Vec<Message>) -> Result<(), ResponsesTranslationError> {
    match item {
        InputItem::Easy(easy) => {
            translate_easy_message(easy, messages);
        }
        InputItem::Typed(typed) => {
            translate_typed_input_item(typed, messages)?;
        }
    }
    Ok(())
}

fn translate_easy_message(easy: &EasyInputMessage, messages: &mut Vec<Message>) {
    let content = match &easy.content {
        EasyInputContent::Text(text) => Value::String(text.clone()),
        EasyInputContent::Parts(parts) => {
            let converted: Vec<Value> = parts
                .iter()
                .filter_map(|p| convert_content_part_to_chat(p))
                .collect();
            Value::Array(converted)
        }
    };

    messages.push(Message {
        role: easy.role.clone(),
        content,
        extra: easy.extra.clone(),
    });
}

fn translate_typed_input_item(item: &TypedInputItem, messages: &mut Vec<Message>) -> Result<(), ResponsesTranslationError> {
    match item {
        TypedInputItem::Message(msg) => {
            translate_message(msg, messages);
        }
        TypedInputItem::FunctionCall(fc) => {
            translate_function_call(fc, messages);
        }
        TypedInputItem::FunctionCallOutput(fco) => {
            translate_function_call_output(fco, messages);
        }
        TypedInputItem::Reasoning(_) => {}
        TypedInputItem::ItemReference(_) => {
            return Err(ResponsesTranslationError::UnsupportedField {
                field: "item_reference",
            });
        }
    }
    Ok(())
}

fn translate_message(msg: &ResponsesMessage, messages: &mut Vec<Message>) {
    let content = convert_message_content(&msg.content);

    messages.push(Message {
        role: msg.role.clone(),
        content,
        extra: msg.extra.clone(),
    });
}

fn convert_message_content(parts: &[InputContentPart]) -> Value {
    let converted: Vec<Value> = parts
        .iter()
        .filter_map(|p| convert_content_part_to_chat(p))
        .collect();
    Value::Array(converted)
}

fn convert_content_part_to_chat(part: &InputContentPart) -> Option<Value> {
    match part {
        InputContentPart::Text(text) => Some(json!({
            "type": "text",
            "text": text.text
        })),
        InputContentPart::Image(img) => convert_image_to_chat(img),
        InputContentPart::Other(v) => Some(v.clone()),
    }
}

fn convert_image_to_chat(img: &InputImage) -> Option<Value> {
    let url = img.image_url.clone()?;
    let mut image_obj = Map::new();
    image_obj.insert("url".to_string(), json!(url));
    if let Some(detail) = &img.detail {
        image_obj.insert("detail".to_string(), json!(detail));
    }
    Some(json!({
        "type": "image_url",
        "image_url": image_obj
    }))
}

fn translate_function_call(fc: &FunctionCall, messages: &mut Vec<Message>) {
    let mut extra = fc.extra.clone();
    extra.insert(
        "tool_calls".to_string(),
        json!([{
            "id": fc.call_id,
            "type": "function",
            "function": {
                "name": fc.name,
                "arguments": fc.arguments
            }
        }]),
    );

    messages.push(Message {
        role: "assistant".to_string(),
        content: Value::Null,
        extra,
    });
}

fn translate_function_call_output(fco: &FunctionCallOutput, messages: &mut Vec<Message>) {
    let content = match &fco.output {
        FunctionCallOutputContent::Text(text) => Value::String(text.clone()),
        FunctionCallOutputContent::Parts(parts) => {
            let converted: Vec<Value> = parts
                .iter()
                .filter_map(|p| convert_output_part_to_chat(p))
                .collect();
            if converted.len() == 1 {
                if let Some(Value::String(s)) = converted.first() {
                    Value::String(s.clone())
                } else {
                    Value::Array(converted)
                }
            } else {
                Value::Array(converted)
            }
        }
    };

    let mut extra = fco.extra.clone();
    extra.insert("tool_call_id".to_string(), json!(fco.call_id));

    messages.push(Message {
        role: "tool".to_string(),
        content,
        extra,
    });
}

fn convert_output_part_to_chat(part: &OutputContentPart) -> Option<Value> {
    match part {
        OutputContentPart::OutputText { text, .. } => Some(Value::String(text.clone())),
        OutputContentPart::Refusal { refusal } => Some(Value::String(refusal.clone())),
    }
}

fn translate_text_format(format: &TextFormat) -> Value {
    match format {
        TextFormat::Text => json!({"type": "text"}),
        TextFormat::JsonObject => json!({"type": "json_object"}),
        TextFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
        } => {
            let mut js = Map::new();
            if let Some(n) = name {
                js.insert("name".to_string(), json!(n));
            }
            if let Some(d) = description {
                js.insert("description".to_string(), json!(d));
            }
            if let Some(s) = schema {
                js.insert("schema".to_string(), s.clone());
            }
            if let Some(strict) = strict {
                js.insert("strict".to_string(), json!(strict));
            }

            let mut rf = Map::new();
            rf.insert("type".to_string(), json!("json_schema"));
            rf.insert("json_schema".to_string(), Value::Object(js));
            Value::Object(rf)
        }
    }
}

fn translate_tools(tools: &[ToolDefinition]) -> Value {
    let converted: Vec<Value> = tools
        .iter()
        .map(|t| translate_tool(t))
        .collect();
    Value::Array(converted)
}

fn translate_tool(tool: &ToolDefinition) -> Value {
    let mut obj = Map::new();
    obj.insert("type".to_string(), json!("function"));

    let mut func = Map::new();
    if let Some(name) = &tool.name {
        func.insert("name".to_string(), json!(name));
    }
    if let Some(desc) = &tool.description {
        func.insert("description".to_string(), json!(desc));
    }
    if let Some(params) = &tool.parameters {
        func.insert("parameters".to_string(), params.clone());
    }
    if let Some(strict) = tool.strict {
        func.insert("strict".to_string(), json!(strict));
    }

    obj.insert("function".to_string(), Value::Object(func));

    for (k, v) in &tool.extra {
        if k != "function" {
            obj.insert(k.clone(), v.clone());
        }
    }

    Value::Object(obj)
}

fn translate_tool_choice(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Mode(mode) => Value::String(mode.clone()),
        ToolChoice::Function { r#type, name } => {
            json!({
                "type": r#type,
                "function": {"name": name}
            })
        }
    }
}

fn replay_history(stored: &StoredConversation, messages: &mut Vec<Message>) -> Result<(), ResponsesTranslationError> {
    for item in &stored.input_items {
        replay_input_item(item, messages)?;
    }

    for item in &stored.output_items {
        replay_output_item(item, messages)?;
    }

    Ok(())
}

fn replay_input_item(item: &InputItem, messages: &mut Vec<Message>) -> Result<(), ResponsesTranslationError> {
    match item {
        InputItem::Easy(easy) => {
            translate_easy_message(easy, messages);
        }
        InputItem::Typed(typed) => {
            replay_typed_input_item(typed, messages)?;
        }
    }
    Ok(())
}

fn replay_typed_input_item(item: &TypedInputItem, messages: &mut Vec<Message>) -> Result<(), ResponsesTranslationError> {
    match item {
        TypedInputItem::Message(msg) => {
            translate_message(msg, messages);
        }
        TypedInputItem::FunctionCall(fc) => {
            translate_function_call(fc, messages);
        }
        TypedInputItem::FunctionCallOutput(fco) => {
            translate_function_call_output(fco, messages);
        }
        TypedInputItem::Reasoning(_) => {}
        TypedInputItem::ItemReference(_) => {
            return Err(ResponsesTranslationError::UnsupportedField {
                field: "item_reference",
            });
        }
    }
    Ok(())
}

fn replay_output_item(item: &OutputItem, messages: &mut Vec<Message>) -> Result<(), ResponsesTranslationError> {
    match item {
        OutputItem::Message(msg) => {
            replay_output_message(msg, messages);
        }
        OutputItem::FunctionCall(fc) => {
            replay_output_function_call(fc, messages);
        }
        OutputItem::Reasoning(_) => {}
    }
    Ok(())
}

fn replay_output_message(msg: &OutputMessage, messages: &mut Vec<Message>) {
    let content = convert_output_message_content(&msg.content);

    messages.push(Message {
        role: msg.role.clone(),
        content,
        extra: msg.extra.clone(),
    });
}

fn convert_output_message_content(parts: &[OutputContentPart]) -> Value {
    let converted: Vec<Value> = parts
        .iter()
        .filter_map(|p| match p {
            OutputContentPart::OutputText { text, .. } => Some(json!({
                "type": "text",
                "text": text
            })),
            OutputContentPart::Refusal { refusal } => Some(json!({
                "type": "refusal",
                "refusal": refusal
            })),
        })
        .collect();
    Value::Array(converted)
}

fn replay_output_function_call(fc: &OutputFunctionCall, messages: &mut Vec<Message>) {
    let mut extra = fc.extra.clone();
    extra.insert(
        "tool_calls".to_string(),
        json!([{
            "id": fc.call_id,
            "type": "function",
            "function": {
                "name": fc.name,
                "arguments": fc.arguments
            }
        }]),
    );

    messages.push(Message {
        role: "assistant".to_string(),
        content: Value::Null,
        extra,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::ReasoningItem;

    fn ctx(model_supports_reasoning: bool) -> TranslationContext<'static> {
        TranslationContext {
            resolved_model: "gpt-4o",
            model_supports_reasoning,
        }
    }

    fn make_request(input: ResponsesInput) -> ResponsesRequest {
        ResponsesRequest {
            model: "gpt-4o".to_string(),
            input,
            instructions: None,
            previous_response_id: None,
            store: false,
            metadata: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            truncation: None,
            parallel_tool_calls: None,
            reasoning: None,
            text: None,
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            stream_options: None,
            extra: Map::new(),
        }
    }

    #[test]
    fn string_input_becomes_user_message() {
        let req = make_request(ResponsesInput::Text("Hello world".to_string()));
        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[0].content, json!("Hello world"));
    }

    #[test]
    fn array_input_with_mixed_roles() {
        let req = make_request(ResponsesInput::Items(vec![
            InputItem::Easy(EasyInputMessage {
                content: EasyInputContent::Text("Hello".to_string()),
                role: "user".to_string(),
                phase: None,
                extra: Map::new(),
            }),
            InputItem::Easy(EasyInputMessage {
                content: EasyInputContent::Text("Hi there".to_string()),
                role: "assistant".to_string(),
                phase: None,
                extra: Map::new(),
            }),
        ]));
        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[1].role, "assistant");
    }

    #[test]
    fn instructions_becomes_system_message() {
        let mut req = make_request(ResponsesInput::Text("Hello".to_string()));
        req.instructions = Some("You are helpful.".to_string());

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "system");
        assert_eq!(result.messages[0].content, json!("You are helpful."));
        assert_eq!(result.messages[1].role, "user");
    }

    #[test]
    fn function_call_becomes_assistant_tool_calls() {
        let req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::FunctionCall(FunctionCall {
                id: None,
                call_id: "call_123".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Paris"}"#.to_string(),
                status: None,
                extra: Map::new(),
            }),
        )]));

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "assistant");
        let tool_calls = result.messages[0].extra.get("tool_calls").unwrap().as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], json!("call_123"));
        assert_eq!(tool_calls[0]["function"]["name"], json!("get_weather"));
    }

    #[test]
    fn function_call_output_becomes_tool_message() {
        let req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::FunctionCallOutput(FunctionCallOutput {
                call_id: "call_123".to_string(),
                output: FunctionCallOutputContent::Text("Sunny, 22°C".to_string()),
                id: None,
                status: None,
                extra: Map::new(),
            }),
        )]));

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "tool");
        assert_eq!(result.messages[0].content, json!("Sunny, 22°C"));
        assert_eq!(result.messages[0].extra.get("tool_call_id").unwrap(), &json!("call_123"));
    }

    #[test]
    fn max_output_tokens_becomes_max_tokens() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.max_output_tokens = Some(500);

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.max_tokens, Some(500));
    }

    #[test]
    fn text_format_json_object_becomes_response_format() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.text = Some(crate::responses::TextConfig {
            verbosity: None,
            format: Some(TextFormat::JsonObject),
            extra: Map::new(),
        });

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(
            result.extra.get("response_format"),
            Some(&json!({"type": "json_object"}))
        );
    }

    #[test]
    fn text_format_json_schema_passthrough() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.text = Some(crate::responses::TextConfig {
            verbosity: None,
            format: Some(TextFormat::JsonSchema {
                name: Some("MySchema".to_string()),
                description: None,
                schema: Some(json!({"type": "object"})),
                strict: Some(true),
            }),
            extra: Map::new(),
        });

        let result = translate(&req, None, &ctx(false)).unwrap();

        let rf = result.extra.get("response_format").unwrap();
        assert_eq!(rf["type"], json!("json_schema"));
        assert_eq!(rf["json_schema"]["name"], json!("MySchema"));
        assert_eq!(rf["json_schema"]["schema"], json!({"type": "object"}));
        assert_eq!(rf["json_schema"]["strict"], json!(true));
    }

    #[test]
    fn reasoning_effort_becomes_reasoning_effort_when_supported() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.reasoning = Some(crate::responses::ReasoningConfig {
            effort: Some("high".to_string()),
            summary: None,
            extra: Map::new(),
        });

        let result = translate(&req, None, &ctx(true)).unwrap();

        assert_eq!(result.extra.get("reasoning_effort"), Some(&json!("high")));
    }

    #[test]
    fn reasoning_effort_omitted_when_not_supported() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.reasoning = Some(crate::responses::ReasoningConfig {
            effort: Some("high".to_string()),
            summary: None,
            extra: Map::new(),
        });

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert!(result.extra.get("reasoning_effort").is_none());
    }

    #[test]
    fn tools_flatten_function_tools() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.tools = vec![ToolDefinition {
            r#type: "function".to_string(),
            name: Some("get_weather".to_string()),
            description: Some("Get weather".to_string()),
            parameters: Some(json!({"type": "object"})),
            strict: Some(true),
            extra: Map::new(),
        }];

        let result = translate(&req, None, &ctx(false)).unwrap();

        let tools = result.extra.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], json!("function"));
        assert_eq!(tools[0]["function"]["name"], json!("get_weather"));
        assert_eq!(tools[0]["function"]["parameters"], json!({"type": "object"}));
    }

    #[test]
    fn tool_choice_passthrough() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.tool_choice = Some(ToolChoice::Mode("auto".to_string()));

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.extra.get("tool_choice"), Some(&json!("auto")));
    }

    #[test]
    fn stream_true_injects_stream_options() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.stream = true;

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.stream, true);
        assert_eq!(
            result.extra.get("stream_options"),
            Some(&json!({"include_usage": true}))
        );
    }

    #[test]
    fn reject_input_audio_content() {
        let req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::Message(crate::responses::Message {
                content: vec![InputContentPart::Other(json!({
                    "type": "input_audio",
                    "input_audio": {"data": "AAAA", "format": "wav"}
                }))],
                role: "user".to_string(),
                status: None,
                extra: Map::new(),
            }),
        )]));

        let result = translate(&req, None, &ctx(false));
        assert!(matches!(
            result,
            Err(ResponsesTranslationError::UnsupportedField { field: "input_audio" })
        ));
    }

    #[test]
    fn reject_input_file_content() {
        let req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::Message(crate::responses::Message {
                content: vec![InputContentPart::Other(json!({
                    "type": "input_file",
                    "input_file": {"file_id": "file_123"}
                }))],
                role: "user".to_string(),
                status: None,
                extra: Map::new(),
            }),
        )]));

        let result = translate(&req, None, &ctx(false));
        assert!(matches!(
            result,
            Err(ResponsesTranslationError::UnsupportedField { field: "input_file" })
        ));
    }

    #[test]
    fn reject_hosted_tool_web_search() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.tools = vec![ToolDefinition {
            r#type: "web_search".to_string(),
            name: None,
            description: None,
            parameters: None,
            strict: None,
            extra: Map::new(),
        }];

        let result = translate(&req, None, &ctx(false));
        assert!(matches!(
            result,
            Err(ResponsesTranslationError::UnsupportedField { .. })
        ));
    }

    #[test]
    fn reject_background_param() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.extra.insert("background".to_string(), json!(true));

        let result = translate(&req, None, &ctx(false));
        assert!(matches!(
            result,
            Err(ResponsesTranslationError::UnsupportedField { field: "background" })
        ));
    }

    #[test]
    fn reject_conversation_param() {
        let mut req = make_request(ResponsesInput::Text("Test".to_string()));
        req.extra.insert("conversation".to_string(), json!({"id": "conv_123"}));

        let result = translate(&req, None, &ctx(false));
        assert!(matches!(
            result,
            Err(ResponsesTranslationError::UnsupportedField { field: "conversation" })
        ));
    }

    #[test]
    fn reject_item_reference() {
        let req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::ItemReference(crate::responses::ItemReference {
                id: "msg_123".to_string(),
                extra: Map::new(),
            }),
        )]));

        let result = translate(&req, None, &ctx(false));
        assert!(matches!(
            result,
            Err(ResponsesTranslationError::UnsupportedField { field: "item_reference" })
        ));
    }

    #[test]
    fn history_replay_includes_stored_messages() {
        let req = make_request(ResponsesInput::Text("Follow up".to_string()));

        let history = StoredConversation {
            input_items: vec![InputItem::Easy(EasyInputMessage {
                content: EasyInputContent::Text("First message".to_string()),
                role: "user".to_string(),
                phase: None,
                extra: Map::new(),
            })],
            output_items: vec![OutputItem::Message(OutputMessage {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                status: "completed".to_string(),
                content: vec![OutputContentPart::OutputText {
                    text: "First response".to_string(),
                    annotations: None,
                }],
                extra: Map::new(),
            })],
        };

        let result = translate(&req, Some(history), &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 3);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[1].role, "assistant");
        assert_eq!(result.messages[2].role, "user");
        assert_eq!(result.messages[2].content, json!("Follow up"));
    }

    #[test]
    fn image_content_with_url_becomes_image_url() {
        let req = make_request(ResponsesInput::Items(vec![InputItem::Typed(
            TypedInputItem::Message(crate::responses::Message {
                content: vec![InputContentPart::Image(InputImage {
                    r#type: "input_image".to_string(),
                    image_url: Some("https://example.com/image.png".to_string()),
                    detail: Some("high".to_string()),
                    file_id: None,
                })],
                role: "user".to_string(),
                status: None,
                extra: Map::new(),
            }),
        )]));

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 1);
        let content = result.messages[0].content.as_array().unwrap();
        assert_eq!(content[0]["type"], json!("image_url"));
        assert_eq!(content[0]["image_url"]["url"], json!("https://example.com/image.png"));
        assert_eq!(content[0]["image_url"]["detail"], json!("high"));
    }

    #[test]
    fn reasoning_items_skipped_in_input() {
        let req = make_request(ResponsesInput::Items(vec![
            InputItem::Typed(TypedInputItem::Reasoning(ReasoningItem {
                id: Some("r_1".to_string()),
                summary: vec![],
                extra: Map::new(),
            })),
            InputItem::Easy(EasyInputMessage {
                content: EasyInputContent::Text("Actual message".to_string()),
                role: "user".to_string(),
                phase: None,
                extra: Map::new(),
            }),
        ]));

        let result = translate(&req, None, &ctx(false)).unwrap();

        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
    }
}
