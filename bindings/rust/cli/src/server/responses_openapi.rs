use serde_json::{json, Map, Value};

pub fn patch_responses_openapi_spec(spec: &[u8]) -> Vec<u8> {
    let mut doc: Value = match serde_json::from_slice(spec) {
        Ok(v) => v,
        Err(_) => return spec.to_vec(),
    };

    let responses_path = if doc.pointer("/paths/~1v1~1responses").is_some() {
        "/v1/responses"
    } else if doc.pointer("/paths/~1responses").is_some() {
        "/responses"
    } else {
        return spec.to_vec();
    };

    let Some(root) = doc.as_object_mut() else {
        return spec.to_vec();
    };
    let components = root
        .entry("components".to_string())
        .or_insert_with(|| json!({}));
    if !components.is_object() {
        *components = json!({});
    }
    let components_obj = components.as_object_mut().expect("object");
    let schemas = components_obj
        .entry("schemas".to_string())
        .or_insert_with(|| json!({}));
    if !schemas.is_object() {
        *schemas = json!({});
    }
    let schemas_obj = schemas.as_object_mut().expect("object");

    install_request_schemas(schemas_obj);
    let stream_refs = install_stream_event_schemas(schemas_obj);

    let post_prefix = format!("/paths/{}/post", responses_path.replace('/', "~1"));
    if let Some(request_schema) = doc.pointer_mut(&format!(
        "{post_prefix}/requestBody/content/application~1json/schema"
    )) {
        *request_schema = json!({ "$ref": "#/components/schemas/CreateResponseBody" });
    }

    if let Some(content_node) = doc.pointer_mut(&format!("{post_prefix}/responses/200/content")) {
        if !content_node.is_object() {
            *content_node = json!({});
        }
        let content = content_node.as_object_mut().expect("content object");
        content
            .entry("application/json".to_string())
            .or_insert_with(
                || json!({ "schema": { "$ref": "#/components/schemas/ResponseResource" } }),
            );
        content.insert(
            "text/event-stream".to_string(),
            json!({
                "schema": {
                    "oneOf": stream_refs
                }
            }),
        );
    }

    serde_json::to_vec_pretty(&doc).unwrap_or_else(|_| spec.to_vec())
}

fn install_request_schemas(schemas: &mut Map<String, Value>) {
    let nullable = |schema: Value| -> Value {
        json!({
            "anyOf": [
                schema,
                { "type": "null" }
            ]
        })
    };

    schemas.insert(
        "ResponseInclude".to_string(),
        json!({
            "type": "string",
            "enum": [
                "message.output_text.logprobs",
                "reasoning.encrypted_content"
            ]
        }),
    );

    schemas.insert(
        "ToolDefinition".to_string(),
        json!({
            "type": "object",
            "required": ["type", "name"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["function"] },
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "pattern": "^[a-zA-Z0-9_-]+$"
                },
                "parameters": { "type": "object" },
                "strict": { "type": "boolean" }
            }
        }),
    );

    schemas.insert(
        "ToolChoiceFunction".to_string(),
        json!({
            "type": "object",
            "required": ["type", "name"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["function"] },
                "name": { "type": "string", "minLength": 1 }
            }
        }),
    );

    schemas.insert(
        "AllowedToolRef".to_string(),
        json!({
            "type": "object",
            "required": ["type", "name"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["function"] },
                "name": { "type": "string", "minLength": 1 }
            }
        }),
    );

    schemas.insert(
        "ToolChoiceAllowedTools".to_string(),
        json!({
            "type": "object",
            "required": ["type", "tools"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["allowed_tools"] },
                "tools": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 128,
                    "items": { "$ref": "#/components/schemas/AllowedToolRef" }
                },
                "mode": { "type": "string", "enum": ["none", "auto", "required"] }
            }
        }),
    );

    schemas.insert(
        "ToolChoice".to_string(),
        json!({
            "oneOf": [
                { "type": "string", "enum": ["none", "auto", "required"] },
                { "$ref": "#/components/schemas/ToolChoiceFunction" },
                { "$ref": "#/components/schemas/ToolChoiceAllowedTools" }
            ]
        }),
    );

    schemas.insert(
        "ReasoningConfig".to_string(),
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "effort": {
                    "anyOf": [
                        { "type": "string", "enum": ["none", "low", "medium", "high", "xhigh"] },
                        { "type": "null" }
                    ]
                },
                "summary": {
                    "anyOf": [
                        { "type": "string", "enum": ["auto", "concise", "detailed"] },
                        { "type": "null" }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "TextFormatText".to_string(),
        json!({
            "type": "object",
            "required": ["type"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["text"] }
            }
        }),
    );

    schemas.insert(
        "TextFormatJsonSchema".to_string(),
        json!({
            "type": "object",
            "required": ["type", "name", "schema"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["json_schema"] },
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "pattern": "^[a-zA-Z0-9_-]+$"
                },
                "schema": { "type": "object" },
                "strict": { "type": "boolean" },
                "description": { "type": "string" }
            }
        }),
    );

    schemas.insert(
        "TextFormat".to_string(),
        json!({
            "oneOf": [
                { "$ref": "#/components/schemas/TextFormatText" },
                { "$ref": "#/components/schemas/TextFormatJsonSchema" }
            ]
        }),
    );

    schemas.insert(
        "TextConfig".to_string(),
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "format": {
                    "anyOf": [
                        { "$ref": "#/components/schemas/TextFormat" },
                        { "type": "null" }
                    ]
                },
                "verbosity": {
                    "anyOf": [
                        { "type": "string", "enum": ["low", "medium", "high"] },
                        { "type": "null" }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "InputTextContentPart".to_string(),
        json!({
            "type": "object",
            "required": ["type", "text"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["input_text"] },
                "text": { "type": "string" }
            }
        }),
    );

    schemas.insert(
        "InputImageContentPart".to_string(),
        json!({
            "type": "object",
            "required": ["type", "image_url"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["input_image"] },
                "image_url": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                },
                "detail": { "type": "string", "enum": ["auto", "low", "high"] }
            }
        }),
    );

    schemas.insert(
        "InputFileContentPart".to_string(),
        json!({
            "type": "object",
            "required": ["type"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["input_file"] },
                "filename": { "type": "string" },
                "file_data": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                },
                "file_url": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "InputVideoContentPart".to_string(),
        json!({
            "type": "object",
            "required": ["type", "video_url"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["input_video"] },
                "video_url": { "type": "string" }
            }
        }),
    );

    schemas.insert(
        "UrlCitationAnnotation".to_string(),
        json!({
            "type": "object",
            "required": ["type", "start_index", "end_index", "url", "title"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["url_citation"] },
                "start_index": { "type": "integer" },
                "end_index": { "type": "integer" },
                "url": { "type": "string" },
                "title": { "type": "string" }
            }
        }),
    );

    schemas.insert(
        "OutputTextContentPart".to_string(),
        json!({
            "type": "object",
            "required": ["type", "text"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["output_text"] },
                "text": { "type": "string" },
                "annotations": {
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/UrlCitationAnnotation" }
                }
            }
        }),
    );

    schemas.insert(
        "RefusalContentPart".to_string(),
        json!({
            "type": "object",
            "required": ["type", "refusal"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["refusal"] },
                "refusal": { "type": "string" }
            }
        }),
    );

    schemas.insert(
        "SummaryTextPart".to_string(),
        json!({
            "type": "object",
            "required": ["type", "text"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["summary_text"] },
                "text": { "type": "string" }
            }
        }),
    );

    schemas.insert(
        "UserMessageInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["type", "role", "content"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["message"] },
                "role": { "type": "string", "enum": ["user"] },
                "content": {
                    "anyOf": [
                        { "type": "string" },
                        {
                            "type": "array",
                            "items": {
                                "oneOf": [
                                    { "$ref": "#/components/schemas/InputTextContentPart" },
                                    { "$ref": "#/components/schemas/InputImageContentPart" },
                                    { "$ref": "#/components/schemas/InputFileContentPart" }
                                ]
                            }
                        }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "SystemMessageInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["type", "role", "content"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["message"] },
                "role": { "type": "string", "enum": ["system"] },
                "content": {
                    "anyOf": [
                        { "type": "string" },
                        {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/InputTextContentPart" }
                        }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "DeveloperMessageInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["type", "role", "content"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["message"] },
                "role": { "type": "string", "enum": ["developer"] },
                "content": {
                    "anyOf": [
                        { "type": "string" },
                        {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/InputTextContentPart" }
                        }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "AssistantMessageInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["type", "role", "content"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["message"] },
                "role": { "type": "string", "enum": ["assistant"] },
                "content": {
                    "anyOf": [
                        { "type": "string" },
                        {
                            "type": "array",
                            "items": {
                                "oneOf": [
                                    { "$ref": "#/components/schemas/OutputTextContentPart" },
                                    { "$ref": "#/components/schemas/RefusalContentPart" }
                                ]
                            }
                        }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "FunctionCallInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["type", "call_id", "name", "arguments"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["function_call"] },
                "call_id": { "type": "string", "minLength": 1 },
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "pattern": "^[a-zA-Z0-9_-]+$"
                },
                "arguments": { "type": "string" },
                "status": { "type": "string", "enum": ["in_progress", "completed"] }
            }
        }),
    );

    schemas.insert(
        "FunctionCallOutputPart".to_string(),
        json!({
            "oneOf": [
                { "$ref": "#/components/schemas/InputTextContentPart" },
                { "$ref": "#/components/schemas/InputImageContentPart" },
                { "$ref": "#/components/schemas/InputFileContentPart" },
                { "$ref": "#/components/schemas/InputVideoContentPart" }
            ]
        }),
    );

    schemas.insert(
        "FunctionCallOutputInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["type", "call_id", "output"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["function_call_output"] },
                "call_id": { "type": "string", "minLength": 1 },
                "status": { "type": "string", "enum": ["in_progress", "completed", "incomplete"] },
                "output": {
                    "anyOf": [
                        { "type": "string" },
                        {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/FunctionCallOutputPart" }
                        }
                    ]
                }
            }
        }),
    );

    schemas.insert(
        "ReasoningInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["type"],
            "additionalProperties": false,
            "properties": {
                "type": { "type": "string", "enum": ["reasoning"] },
                "summary": {
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/SummaryTextPart" }
                }
            }
        }),
    );

    schemas.insert(
        "ItemReferenceInputItem".to_string(),
        json!({
            "type": "object",
            "required": ["id"],
            "additionalProperties": false,
            "properties": {
                "type": {
                    "anyOf": [
                        { "type": "string", "enum": ["item_reference"] },
                        { "type": "null" }
                    ]
                },
                "id": { "type": "string", "minLength": 1 }
            }
        }),
    );

    schemas.insert(
        "InputItemParam".to_string(),
        json!({
            "oneOf": [
                { "$ref": "#/components/schemas/UserMessageInputItem" },
                { "$ref": "#/components/schemas/SystemMessageInputItem" },
                { "$ref": "#/components/schemas/DeveloperMessageInputItem" },
                { "$ref": "#/components/schemas/AssistantMessageInputItem" },
                { "$ref": "#/components/schemas/FunctionCallInputItem" },
                { "$ref": "#/components/schemas/FunctionCallOutputInputItem" },
                { "$ref": "#/components/schemas/ReasoningInputItem" },
                { "$ref": "#/components/schemas/ItemReferenceInputItem" }
            ]
        }),
    );

    schemas.insert(
        "CreateResponseBody".to_string(),
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "background": nullable(json!({ "type": "boolean" })),
                "frequency_penalty": nullable(json!({ "type": "number" })),
                "include": nullable(json!({
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/ResponseInclude" }
                })),
                "input": nullable(json!({
                    "oneOf": [
                        { "type": "string", "maxLength": 10485760 },
                        {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/InputItemParam" }
                        }
                    ]
                })),
                "instructions": nullable(json!({ "type": "string" })),
                "max_output_tokens": nullable(json!({ "type": "integer", "minimum": 1 })),
                "max_tool_calls": nullable(json!({ "type": "integer", "minimum": 1 })),
                "metadata": nullable(json!({
                    "type": "object",
                    "maxProperties": 16,
                    "additionalProperties": { "type": "string", "maxLength": 512 }
                })),
                "model": nullable(json!({ "type": "string" })),
                "parallel_tool_calls": nullable(json!({ "type": "boolean" })),
                "presence_penalty": nullable(json!({ "type": "number" })),
                "previous_response_id": nullable(json!({ "type": "string" })),
                "prompt_cache_key": nullable(json!({ "type": "string", "maxLength": 64 })),
                "reasoning": nullable(json!({ "$ref": "#/components/schemas/ReasoningConfig" })),
                "safety_identifier": nullable(json!({ "type": "string", "maxLength": 64 })),
                "service_tier": nullable(json!({ "type": "string", "enum": ["auto", "default", "flex", "priority"] })),
                "store": nullable(json!({ "type": "boolean" })),
                "stream": nullable(json!({ "type": "boolean" })),
                "stream_options": nullable(json!({
                    "type": "object",
                    "properties": {
                        "include_obfuscation": { "type": "boolean" }
                    },
                    "additionalProperties": true
                })),
                "temperature": nullable(json!({ "type": "number" })),
                "text": nullable(json!({ "$ref": "#/components/schemas/TextConfig" })),
                "tool_choice": nullable(json!({ "$ref": "#/components/schemas/ToolChoice" })),
                "tools": nullable(json!({
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/ToolDefinition" }
                })),
                "top_logprobs": nullable(json!({ "type": "integer", "minimum": 0, "maximum": 20 })),
                "top_p": nullable(json!({ "type": "number" })),
                "truncation": nullable(json!({ "type": "string", "enum": ["auto", "disabled"] }))
            }
        }),
    );
}

fn install_stream_event_schemas(schemas: &mut Map<String, Value>) -> Vec<Value> {
    let stream_events = [
        ("ResponseQueuedStreamingEvent", "response.queued"),
        ("ResponseCreatedStreamingEvent", "response.created"),
        ("ResponseInProgressStreamingEvent", "response.in_progress"),
        (
            "ResponseOutputItemAddedStreamingEvent",
            "response.output_item.added",
        ),
        (
            "ResponseContentPartAddedStreamingEvent",
            "response.content_part.added",
        ),
        (
            "ResponseReasoningSummaryPartAddedStreamingEvent",
            "response.reasoning_summary_part.added",
        ),
        (
            "ResponseOutputTextDeltaStreamingEvent",
            "response.output_text.delta",
        ),
        (
            "ResponseFunctionCallArgumentsDeltaStreamingEvent",
            "response.function_call_arguments.delta",
        ),
        (
            "ResponseReasoningSummaryTextDeltaStreamingEvent",
            "response.reasoning_summary_text.delta",
        ),
        (
            "ResponseRefusalDeltaStreamingEvent",
            "response.refusal.delta",
        ),
        (
            "ResponseReasoningDeltaStreamingEvent",
            "response.reasoning.delta",
        ),
        (
            "ResponseOutputTextDoneStreamingEvent",
            "response.output_text.done",
        ),
        (
            "ResponseFunctionCallArgumentsDoneStreamingEvent",
            "response.function_call_arguments.done",
        ),
        (
            "ResponseReasoningSummaryTextDoneStreamingEvent",
            "response.reasoning_summary_text.done",
        ),
        ("ResponseRefusalDoneStreamingEvent", "response.refusal.done"),
        (
            "ResponseReasoningDoneStreamingEvent",
            "response.reasoning.done",
        ),
        (
            "ResponseContentPartDoneStreamingEvent",
            "response.content_part.done",
        ),
        (
            "ResponseReasoningSummaryPartDoneStreamingEvent",
            "response.reasoning_summary_part.done",
        ),
        (
            "ResponseOutputItemDoneStreamingEvent",
            "response.output_item.done",
        ),
        (
            "ResponseOutputTextAnnotationAddedStreamingEvent",
            "response.output_text.annotation.added",
        ),
        ("ResponseCompletedStreamingEvent", "response.completed"),
        ("ResponseIncompleteStreamingEvent", "response.incomplete"),
        ("ErrorStreamingEvent", "response.failed"),
    ];

    let mut refs = Vec::with_capacity(stream_events.len());
    for (schema_name, event_type) in stream_events {
        schemas.insert(
            schema_name.to_string(),
            json!({
                "type": "object",
                "required": ["type"],
                "additionalProperties": true,
                "properties": {
                    "type": { "type": "string", "enum": [event_type] },
                    "response": { "$ref": "#/components/schemas/ResponseResource" }
                }
            }),
        );
        refs.push(json!({ "$ref": format!("#/components/schemas/{schema_name}") }));
    }

    refs
}
