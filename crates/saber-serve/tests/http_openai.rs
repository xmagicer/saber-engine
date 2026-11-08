//! HTTP-level tests: a real axum server on an ephemeral port, driven with `reqwest`,
//! backed by [`MockEngine`] and [`ByteCodec`] so nothing here needs a GPU or a
//! checkpoint.
//!
//! The headline assertion is the one production parity depends on: **the number of
//! SSE chunks equals the number of engine steps**, and a BetterBench-style client
//! classifies the stream the way it classifies prod.

use std::sync::Arc;
use std::time::Duration;

use saber_serve::detok::ByteCodec;
use saber_serve::engine::{FinishReason, MockEngine, MockScript, MockStep, ModelInfo};
use saber_serve::state::{MinimalChatRenderer, ReasoningParserKind, ServerState, ToolParserKind};
use serde_json::{json, Value};

// ── BetterBench's classifier, ported verbatim ───────────────────────────────────
//
// `~/betterbench/betterbench/client.py`:
//
//   CHUNK_TOKEN_TOL = 0.02; CHUNK_TOKEN_SLACK = 2
//   def classify_chunking(completion_tokens, n_chunks):
//       if not completion_tokens or n_chunks <= 1: return "unknown", None
//       tpu = completion_tokens / n_chunks
//       if tpu < 0.90: return "unknown", tpu
//       if (completion_tokens - n_chunks) <= max(CHUNK_TOKEN_SLACK,
//                                                CHUNK_TOKEN_TOL * n_chunks):
//           return "per_token", tpu
//       return "batched", tpu
//
// and `n_chunks` counts only the chunks whose delta carried a non-empty `content` or
// `reasoning` / `reasoning_content` — the role-only opening chunk and the usage chunk
// do not count.
const CHUNK_TOKEN_TOL: f64 = 0.02;
const CHUNK_TOKEN_SLACK: f64 = 2.0;

fn classify_chunking(
    completion_tokens: Option<u64>,
    n_chunks: usize,
) -> (&'static str, Option<f64>) {
    match completion_tokens {
        None | Some(0) => ("unknown", None),
        Some(comp) if n_chunks <= 1 => {
            let _ = comp;
            ("unknown", None)
        }
        Some(comp) => {
            let tpu = comp as f64 / n_chunks as f64;
            if tpu < 0.90 {
                return ("unknown", Some(tpu));
            }
            let slack = CHUNK_TOKEN_SLACK.max(CHUNK_TOKEN_TOL * n_chunks as f64);
            if (comp as f64 - n_chunks as f64) <= slack {
                ("per_token", Some(tpu))
            } else {
                ("batched", Some(tpu))
            }
        }
    }
}

/// What a BetterBench-style reader sees in one stream.
#[derive(Debug, Default)]
struct StreamView {
    /// Every `data:` payload except `[DONE]`.
    events: Vec<Value>,
    /// Chunks that carried content or reasoning — BetterBench's `n_chunks`.
    counted_chunks: usize,
    content: String,
    reasoning: String,
    tool_call_names: Vec<String>,
    tool_call_arguments: Vec<String>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    saw_done: bool,
}

fn read_sse(body: &str) -> StreamView {
    let mut view = StreamView::default();
    let mut tool_names: Vec<String> = Vec::new();
    let mut tool_args: Vec<String> = Vec::new();
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            view.saw_done = true;
            continue;
        }
        let event: Value = serde_json::from_str(payload).expect("chunk is json");
        if let Some(usage) = event.get("usage") {
            if !usage.is_null() {
                view.usage = Some(usage.clone());
            }
        }
        if let Some(choice) = event["choices"].as_array().and_then(|c| c.first()) {
            if let Some(reason) = choice["finish_reason"].as_str() {
                view.finish_reason = Some(reason.to_owned());
            }
            let delta = &choice["delta"];
            let content = delta["content"].as_str().filter(|s| !s.is_empty());
            let reasoning = delta["reasoning"].as_str().filter(|s| !s.is_empty());
            if content.is_some() || reasoning.is_some() {
                view.counted_chunks += 1;
            }
            view.content.push_str(content.unwrap_or_default());
            view.reasoning.push_str(reasoning.unwrap_or_default());
            if let Some(calls) = delta["tool_calls"].as_array() {
                for call in calls {
                    let index = call["index"].as_u64().unwrap_or(0) as usize;
                    while tool_names.len() <= index {
                        tool_names.push(String::new());
                        tool_args.push(String::new());
                    }
                    if let Some(name) = call["function"]["name"].as_str() {
                        tool_names[index].push_str(name);
                    }
                    if let Some(args) = call["function"]["arguments"].as_str() {
                        tool_args[index].push_str(args);
                    }
                }
            }
        }
        view.events.push(event);
    }
    view.tool_call_names = tool_names;
    view.tool_call_arguments = tool_args;
    view
}

// ── harness ─────────────────────────────────────────────────────────────────────

fn model_info() -> ModelInfo {
    ModelInfo {
        served_model_name: "Qwen3.8".to_owned(),
        aliases: vec!["Qwen3.6".to_owned()],
        max_model_len: 4096,
        root: "/models/Qwen3.8-27B-NVFP4".to_owned(),
        // 255 is this vocabulary's end-of-turn marker: counted in usage, never text.
        eos_token_ids: vec![255],
    }
}

fn bytes_of(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// A server whose engine replays `scripts`, with both parsers wired up.
async fn spawn_with(scripts: Vec<MockScript>, step_delay: Duration) -> (String, Arc<MockEngine>) {
    let engine = Arc::new(MockEngine::new(model_info()).with_step_delay(step_delay));
    for script in scripts {
        engine.push_script(script);
    }
    let state = ServerState::builder(
        engine.clone(),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    )
    .reasoning_parser(ReasoningParserKind::Qwen3)
    .tool_parser(ToolParserKind::Qwen3Coder)
    .enable_auto_tool_choice(true)
    .system_fingerprint("saber-test")
    .build();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        saber_serve::server::serve(state, listener)
            .await
            .expect("serve");
    });
    (format!("http://{addr}"), engine)
}

async fn spawn(scripts: Vec<MockScript>) -> (String, Arc<MockEngine>) {
    spawn_with(scripts, Duration::ZERO).await
}

fn chat_body(extra: Value) -> Value {
    let mut body = json!({
        "model": "Qwen3.8",
        "messages": [{"role": "user", "content": "Say hi"}],
    });
    for (k, v) in extra.as_object().expect("extra is an object") {
        body[k] = v.clone();
    }
    body
}

// ── tests ───────────────────────────────────────────────────────────────────────

/// The parity invariant: one SSE chunk per engine step, never one per token.
#[tokio::test]
async fn streaming_emits_exactly_one_chunk_per_engine_step() {
    let text = "Hi! How can I help you today? I can answer questions or write code.";
    let tokens = bytes_of(text);
    let width = 5; // a speculative step commits several tokens
    let expected_steps = tokens.len().div_ceil(width);
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        tokens.clone(),
        width,
        FinishReason::Stop,
    )])
    .await;

    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stream": true,
            "stream_options": {"include_usage": true},
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 200
        })))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");

    let view = read_sse(&body);
    // events = 1 role chunk + one per step + 1 usage chunk
    assert_eq!(
        view.events.len(),
        expected_steps + 2,
        "one chunk per step, plus the role and usage chunks:\n{body}"
    );
    assert_eq!(
        view.counted_chunks, expected_steps,
        "content chunks == steps"
    );
    assert_eq!(view.content, text);
    assert_eq!(view.finish_reason.as_deref(), Some("stop"));
    assert!(view.saw_done, "stream must end with data: [DONE]");

    let usage = view.usage.expect("usage chunk when include_usage is set");
    assert_eq!(usage["completion_tokens"], json!(tokens.len()));
    assert!(usage["prompt_tokens"].as_u64().expect("prompt tokens") > 0);
    assert_eq!(
        usage["total_tokens"],
        json!(usage["prompt_tokens"].as_u64().unwrap() + tokens.len() as u64)
    );
    assert!(
        usage.get("prompt_tokens_details").is_none(),
        "prod's streaming usage omits prompt_tokens_details"
    );
}

/// The same stream, through BetterBench's own classifier: prod emits one update per
/// step and is classified `batched`; a per-token server is classified `per_token` and
/// its `update p50` would not be ms/step. The control is the second half.
#[tokio::test]
async fn betterbench_classifies_the_stream_the_way_it_classifies_prod() {
    let text = "Hi! How can I help you today? I can answer questions or write code.";
    let tokens = bytes_of(text);

    // (a) speculative shape: several tokens per step, as prod serves.
    let width = 3;
    let steps = tokens.len().div_ceil(width);
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        tokens.clone(),
        width,
        FinishReason::Stop,
    )])
    .await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stream": true,
            "stream_options": {"include_usage": true},
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 200
        })))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");
    let view = read_sse(&body);
    let completion_tokens = view.usage.as_ref().expect("usage")["completion_tokens"].as_u64();
    let (chunking, tpu) = classify_chunking(completion_tokens, view.counted_chunks);
    assert_eq!(view.counted_chunks, steps);
    assert_eq!(
        chunking, "batched",
        "prod's one-update-per-step stream classifies as batched; got {chunking} \
         (tokens_per_update {tpu:?})"
    );
    let tpu = tpu.expect("tokens per update");
    assert!(
        (tpu - tokens.len() as f64 / steps as f64).abs() < 1e-9,
        "tokens_per_update must be tokens/steps, got {tpu}"
    );

    // (b) control: an engine that commits one token per step. Same HTTP code path —
    // the classification changes only because the *engine* step changed, which is
    // exactly the property that makes `update p50` comparable with prod.
    let (base, _engine) = spawn(vec![MockScript::per_token(
        tokens.clone(),
        FinishReason::Stop,
    )])
    .await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stream": true,
            "stream_options": {"include_usage": true},
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 200
        })))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");
    let view = read_sse(&body);
    assert_eq!(view.counted_chunks, tokens.len());
    let (chunking, _) = classify_chunking(
        view.usage.as_ref().expect("usage")["completion_tokens"].as_u64(),
        view.counted_chunks,
    );
    assert_eq!(chunking, "per_token");
}

/// Prod's captured behaviour: an 8-token budget spent entirely in reasoning returns
/// `content: null` and a populated `reasoning`, and `usage.prompt_tokens_details` is
/// present and null.
#[tokio::test]
async fn reasoning_only_generation_returns_null_content() {
    let reasoning = "The user is asking a simple arithmetic question";
    let tokens = bytes_of(reasoning);
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        tokens.clone(),
        4,
        FinishReason::Length,
    )])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({"max_tokens": 1000})))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");

    let message = &response["choices"][0]["message"];
    assert_eq!(message["content"], Value::Null, "{response:#}");
    assert_eq!(message["reasoning"], json!(reasoning));
    assert!(
        message.get("reasoning_content").is_none(),
        "The reference server 0.26+ names the field `reasoning`"
    );
    assert_eq!(response["choices"][0]["finish_reason"], json!("length"));
    assert_eq!(response["usage"]["completion_tokens"], json!(tokens.len()));
    assert!(
        response["usage"]
            .as_object()
            .expect("usage object")
            .contains_key("prompt_tokens_details"),
        "prod's non-streaming usage carries prompt_tokens_details: null"
    );
    assert_eq!(response["usage"]["prompt_tokens_details"], Value::Null);
}

/// `</think>` mid-stream splits reasoning from content, and the tag may be split
/// across steps.
#[tokio::test]
async fn reasoning_then_content_splits_across_a_tag_boundary() {
    let output = "thinking hard</think>The answer is 4.";
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of(output),
        3, // splits "</think>" across steps
        FinishReason::Stop,
    )])
    .await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({"stream": true, "max_tokens": 200})))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");
    let view = read_sse(&body);
    assert_eq!(view.reasoning, "thinking hard");
    assert_eq!(view.content, "The answer is 4.");
    assert!(
        !view.content.contains("think"),
        "the tag must not leak into content: {}",
        view.content
    );
}

const TOOL_OUTPUT: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

#[tokio::test]
async fn forced_tool_choices_deliver_parsed_calls_in_both_response_modes() {
    for choice in [
        json!("required"),
        json!({"type":"function","function":{"name":"get_weather"}}),
    ] {
        for streaming in [false, true] {
            let (base, _engine) = spawn(vec![MockScript::in_steps_of(
                bytes_of(TOOL_OUTPUT),
                4,
                FinishReason::Stop,
            )])
            .await;
            let response = reqwest::Client::new().post(format!("{base}/v1/chat/completions"))
                .json(&chat_body(json!({"tools":weather_tools(),"tool_choice":choice,
                    "stream":streaming,"max_tokens":500,"chat_template_kwargs":{"enable_thinking":false}})))
                .send().await.expect("request");
            assert_eq!(response.status(), 200);
            if streaming {
                let text = response.text().await.expect("SSE");
                let view = read_sse(&text);
                assert!(text.contains("get_weather"), "{text}");
                assert!(text.contains("tool_calls"), "{text}");
                assert!(text.contains("[DONE]"));
                assert!(view.content.is_empty());
            } else {
                let response: Value = response.json().await.expect("JSON");
                let call = &response["choices"][0]["message"]["tool_calls"][0];
                assert_eq!(call["function"]["name"], json!("get_weather"));
                let arguments: Value =
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
                assert_eq!(arguments, json!({"city":"Paris","days":3}));
                assert_eq!(response["choices"][0]["finish_reason"], json!("tool_calls"));
            }
        }
    }
}

fn weather_tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                }
            }
        }
    }])
}

#[tokio::test]
async fn tool_calls_are_parsed_and_typed_non_streaming() {
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of(TOOL_OUTPUT),
        7,
        FinishReason::Stop,
    )])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "tools": weather_tools(),
            "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 500
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");

    let choice = &response["choices"][0];
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["type"], json!("function"));
    assert_eq!(call["function"]["name"], json!("get_weather"));
    // `days` is typed `integer` in the schema, so it is a number, not a string.
    assert_eq!(
        call["function"]["arguments"],
        json!("{\"city\": \"Paris\", \"days\": 3}")
    );
    assert!(call["id"]
        .as_str()
        .expect("id")
        .starts_with("chatcmpl-tool-"));
    assert_eq!(choice["message"]["content"], Value::Null);
    assert_eq!(choice["finish_reason"], json!("tool_calls"));
}

#[tokio::test]
async fn tool_calls_stream_as_deltas_and_still_one_chunk_per_step() {
    let tokens = bytes_of(TOOL_OUTPUT);
    let width = 4;
    let steps = tokens.len().div_ceil(width);
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        tokens,
        width,
        FinishReason::Stop,
    )])
    .await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stream": true,
            "tools": weather_tools(),
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 500
        })))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");
    let view = read_sse(&body);
    assert_eq!(
        view.events.len(),
        steps + 1,
        "one chunk per step plus the role chunk (no usage chunk was requested)"
    );
    assert_eq!(view.tool_call_names, vec!["get_weather".to_owned()]);
    assert_eq!(
        view.tool_call_arguments,
        vec!["{\"city\": \"Paris\", \"days\": 3}".to_owned()]
    );
    assert_eq!(view.finish_reason.as_deref(), Some("tool_calls"));
    assert!(
        view.content.is_empty(),
        "tool markup must not leak as content"
    );
}

/// Without `--enable-auto-tool-choice` the markup stays in `content`, as the reference server does.
/// A server started with a parser profile other than prod's (V9): the builder's three knobs are
/// exactly what `saber serve`'s flags resolve to.
async fn spawn_profile(script: MockScript, reasoning: bool, tool: bool, auto: bool) -> String {
    let engine = Arc::new(MockEngine::new(model_info()));
    engine.push_script(script);
    let mut builder = ServerState::builder(
        engine.clone(),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    )
    .enable_auto_tool_choice(auto)
    .default_enable_thinking(false);
    if reasoning {
        builder = builder.reasoning_parser(ReasoningParserKind::Qwen3);
    }
    if tool {
        builder = builder.tool_parser(ToolParserKind::Qwen3Coder);
    }
    let state = builder.build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        saber_serve::server::serve(state, listener)
            .await
            .expect("serve")
    });
    format!("http://{addr}")
}

/// `--tool-call-parser qwen3_coder` WITHOUT `--enable-auto-tool-choice` is the reference server's inert parser: no
/// tool parser is constructed, so a request that sends `tools` and leaves `tool_choice` to the
/// protocol's default ("auto") is refused with the reference server's own 400, a NAMED `tool_choice` likewise, and
/// only `tool_choice: "none"` is served -- with the markup left in `content` (V9; this test said
/// the opposite before, serving the "auto" request, which prod refuses).
#[tokio::test]
async fn auto_tool_choice_disabled_refuses_like_the_reference_and_serves_none_with_markup_in_content(
) {
    let script = || MockScript::in_steps_of(bytes_of(TOOL_OUTPUT), 9, FinishReason::Stop);
    let client = reqwest::Client::new();

    let base = spawn_profile(script(), true, true, false).await;
    let refused = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(
            json!({"tools": weather_tools(), "max_tokens": 500}),
        ))
        .send()
        .await
        .expect("request");
    assert_eq!(refused.status(), 400);
    let body: Value = refused.json().await.expect("json");
    assert_eq!(
        body["error"]["message"],
        json!("\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set")
    );
    assert_eq!(body["error"]["type"], json!("BadRequestError"));
    assert_eq!(body["error"]["param"], Value::Null);

    let named = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "tools": weather_tools(),
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        })))
        .send()
        .await
        .expect("request");
    assert_eq!(named.status(), 400);
    let body: Value = named.json().await.expect("json");
    assert_eq!(
        body["error"]["message"],
        json!("tool_choice=function \"get_weather\" requires --tool-call-parser to be set")
    );

    let base = spawn_profile(script(), true, true, false).await;
    let served: Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "tools": weather_tools(), "tool_choice": "none", "max_tokens": 500,
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(
        served["choices"][0]["message"]["content"],
        json!(TOOL_OUTPUT)
    );
    assert!(served["choices"][0]["message"].get("tool_calls").is_none());
}

/// No parser at all (`--parser-defaults compat`, or `--no-enable-auto-tool-choice
/// --reasoning-parser ""`): the reply is the raw text, think block and all, in `content`, with no
/// `reasoning` field -- the reference server's `parser_cls is None` branch (V9).
#[tokio::test]
async fn no_parser_leaves_the_think_block_in_content() {
    let text = "Let me think.</think>\n\nThe answer is 4.";
    let base = spawn_profile(
        MockScript::in_steps_of(bytes_of(text), 7, FinishReason::Stop),
        false,
        false,
        false,
    )
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({"max_tokens": 500})))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(response["choices"][0]["message"]["content"], json!(text));
    assert!(response["choices"][0]["message"]
        .get("reasoning")
        .is_none_or(Value::is_null));
    // The CONTROL: prod's profile splits the same reply (it must differ from the case the way the
    // case differs from prod's own default).
    let base = spawn_profile(
        MockScript::in_steps_of(bytes_of(text), 7, FinishReason::Stop),
        true,
        true,
        true,
    )
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "max_tokens": 500, "chat_template_kwargs": {"enable_thinking": true},
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(
        response["choices"][0]["message"]["reasoning"],
        json!("Let me think.")
    );
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!("\n\nThe answer is 4.")
    );
}

/// A multi-byte character split across two engine steps must never surface as
/// `U+FFFD`, over HTTP, in both streaming and non-streaming.
#[tokio::test]
async fn multibyte_characters_split_across_steps_are_never_mangled() {
    let text = "héllo 世界 🌤️ café";
    let tokens = bytes_of(text);
    for width in [1usize, 2, 3, 5] {
        let (base, _engine) = spawn(vec![
            MockScript::in_steps_of(tokens.clone(), width, FinishReason::Stop),
            MockScript::in_steps_of(tokens.clone(), width, FinishReason::Stop),
        ])
        .await;
        let client = reqwest::Client::new();
        let response: Value = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&chat_body(json!({
                "chat_template_kwargs": {"enable_thinking": false},
                "max_tokens": 500
            })))
            .send()
            .await
            .expect("request")
            .json()
            .await
            .expect("json");
        assert_eq!(
            response["choices"][0]["message"]["content"],
            json!(text),
            "non-streaming at width {width}"
        );

        let body = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&chat_body(json!({
                "stream": true,
                "chat_template_kwargs": {"enable_thinking": false},
                "max_tokens": 500
            })))
            .send()
            .await
            .expect("request")
            .text()
            .await
            .expect("body");
        let view = read_sse(&body);
        assert_eq!(view.content, text, "streaming at width {width}");
        assert!(
            !view.content.contains('\u{fffd}'),
            "replacement character at width {width}"
        );
    }
}

#[tokio::test]
async fn stop_strings_truncate_and_report_stop() {
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of("one two THREE four"),
        4,
        FinishReason::Stop,
    )])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stop": ["THREE"],
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 500
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!("one two ")
    );
    assert_eq!(response["choices"][0]["finish_reason"], json!("stop"));
    assert_eq!(response["choices"][0]["stop_reason"], json!("THREE"));
}

/// The stop-string search starts mid-text; if it started mid-*character* it would
/// silently find nothing.
#[tokio::test]
async fn stop_strings_are_found_after_a_multibyte_character() {
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of("café—STOPtail"),
        3,
        FinishReason::Stop,
    )])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stop": ["STOP"],
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 500
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(response["choices"][0]["message"]["content"], json!("café—"));
    assert_eq!(response["choices"][0]["finish_reason"], json!("stop"));
}

#[tokio::test]
async fn max_tokens_clamps_an_overrunning_engine_and_reports_length() {
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of("abcdefghijklmnop"),
        4,
        FinishReason::Stop,
    )])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "max_tokens": 6,
            "chat_template_kwargs": {"enable_thinking": false}
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(response["usage"]["completion_tokens"], json!(6));
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!("abcdef")
    );
    assert_eq!(response["choices"][0]["finish_reason"], json!("length"));
}

#[tokio::test]
async fn eos_tokens_count_towards_usage_but_never_reach_the_text() {
    let mut tokens = bytes_of("done");
    tokens.push(255); // the model's end-of-turn token
    let (base, _engine) = spawn(vec![MockScript::new(vec![
        MockStep::new(tokens.clone()).finish(FinishReason::Stop)
    ])])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 50
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(response["choices"][0]["message"]["content"], json!("done"));
    assert_eq!(response["usage"]["completion_tokens"], json!(5));
}

#[tokio::test]
async fn n_greater_than_one_returns_n_choices() {
    let (base, engine) = spawn(vec![
        MockScript::in_steps_of(bytes_of("first"), 2, FinishReason::Stop),
        MockScript::in_steps_of(bytes_of("second"), 2, FinishReason::Stop),
        MockScript::in_steps_of(bytes_of("third"), 2, FinishReason::Stop),
    ])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "n": 3,
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 50
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let choices = response["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 3);
    assert_eq!(choices[0]["message"]["content"], json!("first"));
    assert_eq!(choices[2]["index"], json!(2));
    assert_eq!(
        response["usage"]["completion_tokens"],
        json!(5 + 6 + 5),
        "usage sums every choice, as the reference server does"
    );
    // ONE submission for three choices: the engine owns the fan-out, so the prompt is
    // prefilled once and branched. Submitting three requests here would prefill three
    // times and leave the engine unable to tell that they were the same prompt.
    assert_eq!(engine.submitted().len(), 1);
    assert_eq!(engine.submitted()[0].params.n, 3);
}

/// The same on `/v1/completions` (ws-f38): ONE submission and the choices demultiplexed by
/// `choice_index`. This endpoint submitted the request `n` times, each carrying `n`, and ran each
/// submission's whole group through one runner — every choice was the group's branches
/// interleaved, ended by whichever finished first. On the card that read as groups that never
/// matched the single request and ran to `length` past its EOS.
#[tokio::test]
async fn completions_n_greater_than_one_is_one_group_demultiplexed() {
    let (base, engine) = spawn(vec![
        MockScript::in_steps_of(bytes_of("first"), 2, FinishReason::Stop),
        MockScript::in_steps_of(bytes_of("second"), 2, FinishReason::Stop),
        MockScript::in_steps_of(bytes_of("third"), 2, FinishReason::Stop),
    ])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&json!({
            "model": "Qwen3.8",
            "prompt": "hi",
            "n": 3,
            "max_tokens": 50,
            "temperature": 0.7,
            "return_token_ids": true
        }))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let choices = response["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 3, "{response}");
    assert_eq!(choices[0]["text"], json!("first"));
    assert_eq!(choices[1]["text"], json!("second"));
    assert_eq!(choices[2]["text"], json!("third"));
    assert_eq!(
        choices[1]["token_ids"],
        json!(bytes_of("second")),
        "each choice's ids are its own branch's, once"
    );
    assert_eq!(response["usage"]["completion_tokens"], json!(5 + 6 + 5));
    assert_eq!(engine.submitted().len(), 1, "one submission for the group");
    assert_eq!(engine.submitted()[0].params.n, 3);
}

/// The sampling parameters a client sends must reach the engine unchanged — a
/// server that silently drops `top_k` would still pass every text assertion above.
#[tokio::test]
async fn sampling_parameters_reach_the_engine() {
    let (base, engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of("ok"),
        2,
        FinishReason::Stop,
    )])
    .await;
    let _: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.05,
            "seed": 1234,
            "max_completion_tokens": 64,
            "stop": ["STOP"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.25,
            "repetition_penalty": 1.05,
            "logprobs": true,
            "top_logprobs": 3,
            "prompt_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": false},
            "some_unknown_field_the_reference_would_ignore": 42
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");

    let submitted = engine.submitted();
    let params = &submitted[0].params;
    assert!((params.temperature - 0.7).abs() < 1e-6);
    assert!((params.top_p - 0.8).abs() < 1e-6);
    assert_eq!(params.top_k, 20);
    assert!((params.min_p - 0.05).abs() < 1e-6);
    assert_eq!(params.seed, Some(1234));
    assert_eq!(params.max_tokens, Some(64));
    assert_eq!(params.stop, vec!["STOP".to_owned()]);
    assert_eq!(params.logprobs, Some(3));
    assert_eq!(params.prompt_logprobs, Some(2));
    assert!((params.presence_penalty - 0.5).abs() < 1e-6);
    assert!((params.repetition_penalty - 1.05).abs() < 1e-6);
}

#[tokio::test]
async fn invalid_parameters_are_rejected_with_the_reference_messages() {
    let (base, _engine) = spawn(vec![]).await;
    let client = reqwest::Client::new();
    // `error.param` is `exc.parameter` and nothing else
    // (`upstream/entrypoints/serve/utils/error_response.py`: `param = exc.parameter` on a
    // `ValidationError`), and `_verify_args` passes `parameter=` on FIVE of its
    // checks only — `temperature`, `top_p`, `max_tokens`, `logprobs`, `prompt_logprobs`.
    // Everywhere else prod returns `param: null` AND leaves the
    // ` (parameter=…, value=…)` suffix off the message, because
    // `ValidationError.__str__` appends it only when a parameter was given. `top_k`,
    // `min_p`, `n` and the three penalties are in that second group; this table used to
    // claim otherwise on all four (V1, 2026-09-21).
    for (body, status, needle, param) in [
        (
            json!({"temperature": -1}),
            400,
            "temperature must be non-negative, got -1.0. (parameter=temperature, value=-1.0)",
            json!("temperature"),
        ),
        (
            json!({"temperature": 3}),
            400,
            "temperature must be in [0, 2], got 3.0. (parameter=temperature, value=3.0)",
            json!("temperature"),
        ),
        (
            json!({"top_p": 3}),
            400,
            "top_p must be in (0, 1], got 3.0. (parameter=top_p, value=3.0)",
            json!("top_p"),
        ),
        (
            json!({"top_k": -5}),
            400,
            "top_k must be 0 (disable), or at least 1, got -5.",
            json!(null),
        ),
        (
            json!({"min_p": 2}),
            400,
            "min_p must be in [0, 1], got 2.0.",
            json!(null),
        ),
        (
            json!({"presence_penalty": 3}),
            400,
            "presence_penalty must be in [-2, 2], got 3.0.",
            json!(null),
        ),
        (
            json!({"repetition_penalty": 0}),
            400,
            "repetition_penalty must be greater than zero, got 0.0.",
            json!(null),
        ),
        (
            json!({"n": 0}),
            400,
            "n must be at least 1, got 0.",
            json!(null),
        ),
        (
            json!({"n": 2, "temperature": 0}),
            400,
            "n must be 1 when using greedy sampling, got 2.",
            json!(null),
        ),
        (
            json!({"logprobs": true, "top_logprobs": 99}),
            400,
            "Requested sample logprobs of 99, which is greater than max allowed: 20",
            json!("logprobs"),
        ),
    ] {
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&chat_body(body.clone()))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status().as_u16(), status, "for {body}");
        let error: Value = response.json().await.expect("json");
        assert!(
            error["error"]["message"]
                .as_str()
                .expect("message")
                .contains(needle),
            "for {body}: {error}"
        );
        assert_eq!(error["error"]["param"], param, "for {body}");
        assert_eq!(error["error"]["type"], json!("BadRequestError"));
        assert_eq!(error["error"]["code"], json!(400));
    }

    // `top_k: -1` is not a rejection: the reference server "quietly accepts -1 as disabled, but prefers
    // 0", and only `< -1` is a 400. Saber refused it before this.
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({"top_k": -1, "max_tokens": 1})))
        .send()
        .await
        .expect("request");
    assert_ne!(
        response.status().as_u16(),
        400,
        "top_k: -1 is the disabled form, not a bad request"
    );

    // An unknown model is a 404 with prod's exact message.
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({"model": "nope"})))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status().as_u16(), 404);
    let error: Value = response.json().await.expect("json");
    assert_eq!(
        error["error"]["message"],
        json!("The model `nope` does not exist.")
    );
    assert_eq!(error["error"]["type"], json!("NotFoundError"));
}

#[tokio::test]
async fn logprobs_and_prompt_logprobs_are_reported() {
    use saber_serve::engine::{SampledLogprobs, TokenLogprob};
    let engine = Arc::new(MockEngine::new(model_info()));
    let mut script = MockScript::new(vec![MockStep::new([b'h' as u32, b'i' as u32])
        .finish(FinishReason::Stop)
        .with_logprobs(vec![
            SampledLogprobs {
                sampled: TokenLogprob {
                    token_id: b'h' as u32,
                    logprob: -0.5,
                    rank: Some(1),
                },
                top: vec![TokenLogprob {
                    token_id: b'H' as u32,
                    logprob: -1.5,
                    rank: Some(2),
                }],
            },
            SampledLogprobs {
                sampled: TokenLogprob {
                    token_id: b'i' as u32,
                    logprob: -0.25,
                    rank: Some(1),
                },
                top: vec![],
            },
        ])]);
    script.prompt_logprobs = Some(vec![
        None,
        Some(vec![TokenLogprob {
            token_id: b'x' as u32,
            logprob: -2.0,
            rank: Some(3),
        }]),
    ]);
    engine.push_script(script);
    let state = ServerState::builder(
        engine.clone(),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    )
    .build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        saber_serve::server::serve(state, listener)
            .await
            .expect("serve")
    });

    let response: Value = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&chat_body(json!({
            "logprobs": true,
            "top_logprobs": 2,
            "prompt_logprobs": 1,
            "max_tokens": 50
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");

    let content = response["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("logprobs content");
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["token"], json!("h"));
    assert_eq!(content[0]["bytes"], json!([104]));
    assert!((content[0]["logprob"].as_f64().expect("logprob") + 0.5).abs() < 1e-6);
    // Prod's chat shape (D5): the CHOSEN token first, then the candidates, `top_logprobs` long.
    assert_eq!(content[0]["top_logprobs"][0]["token"], json!("h"));
    assert_eq!(content[0]["top_logprobs"][1]["token"], json!("H"));
    assert_eq!(content[0]["top_logprobs"].as_array().map(Vec::len), Some(2));

    let prompt_logprobs = response["prompt_logprobs"]
        .as_array()
        .expect("prompt_logprobs");
    assert_eq!(prompt_logprobs.len(), 2);
    assert_eq!(
        prompt_logprobs[0],
        Value::Null,
        "position 0 has no distribution"
    );
    assert_eq!(prompt_logprobs[1]["120"]["rank"], json!(3));
    assert_eq!(prompt_logprobs[1]["120"]["decoded_token"], json!("x"));
}

/// An engine that ends a stream without a `finished` event is a contract violation:
/// the client is told `abort` rather than left with a null `finish_reason`, and the
/// metric records it.
#[tokio::test]
async fn a_stream_that_ends_without_finishing_reports_abort() {
    let (base, _engine) = spawn(vec![MockScript::new(vec![
        MockStep::new(bytes_of("half")),
        MockStep::new(bytes_of(" done")),
    ])])
    .await;
    let client = reqwest::Client::new();
    let body = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stream": true,
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 500
        })))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");
    let view = read_sse(&body);
    assert_eq!(view.content, "half done");
    assert_eq!(view.finish_reason.as_deref(), Some("abort"));
    assert!(view.saw_done);

    let metrics = client
        .get(format!("{base}/metrics"))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("text");
    assert!(metrics.contains(
        "saber:request_success_total{engine=\"0\",finished_reason=\"abort\",model_name=\"Qwen3.8\"} 1.0"
    ));
}

#[tokio::test]
async fn models_health_and_metrics() {
    let (base, engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of("hello"),
        2,
        FinishReason::Stop,
    )])
    .await;
    let client = reqwest::Client::new();

    let models: Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(models["object"], json!("list"));
    assert_eq!(models["data"][0]["id"], json!("Qwen3.8"));
    assert_eq!(models["data"][1]["id"], json!("Qwen3.6"));
    assert_eq!(models["data"][0]["max_model_len"], json!(4096));
    assert_eq!(
        models["data"][0]["root"],
        json!("/models/Qwen3.8-27B-NVFP4")
    );

    assert_eq!(
        client
            .get(format!("{base}/health"))
            .send()
            .await
            .expect("request")
            .status()
            .as_u16(),
        200
    );

    let _: Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 50
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    engine.set_waiting(2);

    let metrics = client
        .get(format!("{base}/metrics"))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("text");
    for needle in [
        "saber:num_requests_running{engine=\"0\",model_name=\"Qwen3.8\"}",
        "saber:num_requests_waiting{engine=\"0\",model_name=\"Qwen3.8\"} 2.0",
        "saber:prompt_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"}",
        "saber:generation_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"} 5.0",
        "saber:spec_decode_num_drafts_total{engine=\"0\",model_name=\"Qwen3.8\"} 3.0",
        "saber:spec_decode_num_draft_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"} 6.0",
        "saber:spec_decode_num_accepted_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"} 5.0",
    ] {
        assert!(metrics.contains(needle), "missing {needle}\n{metrics}");
    }
}

#[tokio::test]
async fn completions_endpoint_returns_text_and_usage() {
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of(" world"),
        3,
        FinishReason::Stop,
    )])
    .await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&json!({"model": "Qwen3.8", "prompt": "Hello", "max_tokens": 16}))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(response["object"], json!("text_completion"));
    assert_eq!(response["choices"][0]["text"], json!(" world"));
    assert_eq!(response["usage"]["prompt_tokens"], json!(5));
    assert_eq!(response["usage"]["completion_tokens"], json!(6));
    assert_eq!(response["usage"]["total_tokens"], json!(11));
}

/// Step pacing is visible to the client: with a 20 ms scripted step delay the gaps
/// between updates are step gaps, which is what `update p50` measures.
#[tokio::test]
async fn updates_arrive_one_per_step_in_wall_clock_order() {
    let (base, _engine) = spawn_with(
        vec![MockScript::in_steps_of(
            bytes_of("abcdefgh"),
            2,
            FinishReason::Stop,
        )],
        Duration::from_millis(20),
    )
    .await;
    let started = std::time::Instant::now();
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "stream": true,
            "chat_template_kwargs": {"enable_thinking": false},
            "max_tokens": 50
        })))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");
    let elapsed = started.elapsed();
    let view = read_sse(&body);
    assert_eq!(view.counted_chunks, 4, "8 bytes at width 2");
    assert!(
        elapsed >= Duration::from_millis(60),
        "four 20 ms steps cannot complete in {elapsed:?}"
    );
}

/// A client that hangs up mid-stream is discovered by the **body's drop**, not by the next chunk
/// failing to send.
///
/// The difference is a whole engine step of a dead request's life — and with it a KV block table
/// and a GDN state slot set, which at 432 MiB per set is not bookkeeping. The two routes are
/// counted separately (`saber:stream_abort_total{path=…}`) precisely so a regression from the
/// first to the second is visible rather than merely slower.
#[tokio::test]
async fn a_dropped_stream_aborts_the_request_without_waiting_for_the_next_chunk() {
    // Steps slow enough that the next chunk is far away when the client leaves.
    let (base, engine) = spawn_with(
        vec![MockScript::in_steps_of(
            bytes_of(&"lorem ipsum ".repeat(200)),
            2,
            FinishReason::Stop,
        )],
        Duration::from_millis(50),
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({"stream": true, "max_tokens": 2000})))
        .send()
        .await
        .expect("request");
    assert!(response.status().is_success());

    // Read one chunk, so the stream is established, then hang up.
    use futures_util::StreamExt as _;
    let mut body = Box::pin(response.bytes_stream());
    let first = tokio::time::timeout(Duration::from_secs(5), body.next())
        .await
        .expect("a first chunk")
        .expect("stream item")
        .expect("bytes");
    assert!(!first.is_empty());
    assert!(engine.aborted().is_empty(), "nothing aborted yet");
    drop(body);

    // Well inside one 50 ms step.
    let mut waited = Duration::ZERO;
    while engine.aborted().is_empty() && waited < Duration::from_millis(40) {
        tokio::time::sleep(Duration::from_millis(2)).await;
        waited += Duration::from_millis(2);
    }
    assert_eq!(
        engine.aborted().len(),
        1,
        "the abort should have travelled on the body's drop, within {waited:?}"
    );

    let metrics = reqwest::get(format!("{base}/metrics"))
        .await
        .expect("metrics")
        .text()
        .await
        .expect("body");
    assert!(
        metrics.contains(
            "saber:stream_abort_total{engine=\"0\",model_name=\"Qwen3.8\",path=\"disconnect\"} 1.0"
        ),
        "{metrics}"
    );
    assert!(
        metrics.contains(
            "saber:stream_abort_total{engine=\"0\",model_name=\"Qwen3.8\",path=\"write_failure\"} 0.0"
        ),
        "{metrics}"
    );
}

/// The ordinary teardown is not a disconnect: a stream that ran to `[DONE]` aborts nothing.
#[tokio::test]
async fn a_completed_stream_is_not_counted_as_a_disconnect() {
    let (base, engine) = spawn(vec![MockScript::in_steps_of(
        bytes_of("all done"),
        2,
        FinishReason::Stop,
    )])
    .await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({"stream": true})))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .expect("body");
    let view = read_sse(&body);
    assert!(view.saw_done);
    assert_eq!(view.finish_reason.as_deref(), Some("stop"));

    // Give the body's drop a moment to be wrong.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        engine.aborted().is_empty(),
        "a finished stream must not abort its request: {:?}",
        engine.aborted()
    );
    let metrics = reqwest::get(format!("{base}/metrics"))
        .await
        .expect("metrics")
        .text()
        .await
        .expect("body");
    assert!(metrics.contains("path=\"disconnect\"} 0.0"), "{metrics}");
}

/// AN EOS TOKEN ENDS THE REQUEST, and does not merely vanish from the text.
///
/// The stop rule had been half a rule: `stop_token_ids` was used to FILTER the token out of the
/// reply and nothing ever set `finished` from it, so a model that said "I am done" at token 3 kept
/// being asked for more until `max_tokens`, and everything it produced after the end of the
/// conversation — a fresh turn, then whatever follows one — was detokenized into the answer with
/// the delimiters invisibly removed. ws-p5's multi-turn gate saw it as replies like `ODA6666666…`
/// running the full 256-token budget where prod was 299/300 healthy.
///
/// The negative control is the same script with `ignore_eos`-shaped intent — a budget SHORTER than
/// the EOS — which must still finish for `length`, so this test cannot pass by finishing everything
/// early.
#[tokio::test]
async fn an_eos_token_finishes_the_request_instead_of_being_deleted_from_it() {
    // `model_info()` declares 255 as the EOS. The script says three visible tokens, then EOS, then
    // more text the client must never see.
    let mut tokens = bytes_of("abc");
    tokens.push(255);
    tokens.extend(bytes_of("XXXXXXXXXXXX"));
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        tokens,
        1,
        FinishReason::Length,
    )])
    .await;

    let body: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": false}
        })))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");

    let choice = &body["choices"][0];
    let text = choice["message"]["content"].as_str().unwrap_or_default();
    assert!(
        !text.contains('X'),
        "the reply carried text from after the model's own EOS: {text:?}"
    );
    assert_eq!(text, "abc", "the reply is what came before the EOS");
    assert_eq!(
        choice["finish_reason"].as_str(),
        Some("stop"),
        "an EOS finishes for `stop`, not for `length`"
    );
    // And the EOS is not billed as a visible completion token beyond what was asked for.
    let completion = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(
        completion <= 4,
        "completion_tokens {completion} ran past the EOS"
    );
}

/// The control: a budget that ends the reply BEFORE any EOS still finishes for `length`.
#[tokio::test]
async fn a_budget_shorter_than_the_eos_still_finishes_for_length() {
    let mut tokens = bytes_of("abcdefgh");
    tokens.push(255);
    let (base, _engine) = spawn(vec![MockScript::in_steps_of(
        tokens,
        1,
        FinishReason::Length,
    )])
    .await;

    let body: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "max_tokens": 3,
            "chat_template_kwargs": {"enable_thinking": false}
        })))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");

    let choice = &body["choices"][0];
    assert_eq!(
        choice["finish_reason"].as_str(),
        Some("length"),
        "a reply cut off by the budget finishes for `length`"
    );
    assert_eq!(choice["message"]["content"].as_str(), Some("abc"));
}

// ── the defaults, over HTTP ─────────────────────────────────────────────────────

async fn spawn_with_defaults(
    defaults: saber_serve::sampling::DefaultSampling,
) -> (String, Arc<MockEngine>) {
    let engine = Arc::new(MockEngine::new(model_info()));
    engine.push_script(MockScript::in_steps_of(
        bytes_of("ok"),
        1,
        FinishReason::Stop,
    ));
    let state = ServerState::builder(
        engine.clone(),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    )
    .default_sampling(defaults)
    .system_fingerprint("saber-test")
    .build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        saber_serve::server::serve(state, listener)
            .await
            .expect("serve");
    });
    (format!("http://{addr}"), engine)
}

/// **The defaults a request that names none of them actually gets**, asked over HTTP rather than
/// of a struct: `ws-v1` found prod's are `0.7 / 0.95 / 20` (the unit's
/// `--override-generation-config`, confirmed against prod's own `get_diff_sampling_param()`), not
/// OpenAI's `1.0 / 1.0 / off`. Nothing on the wire shows it — the reply carries no sampling
/// fields — so what is asserted here is what reached the ENGINE.
///
/// The control is the bug: the same server built with the reference server's stock defaults resolves `1.0 / 1.0 /
/// 0` for the identical request, and every gate in this tree would stay green either way, because
/// they all send `temperature: 0`.
#[tokio::test]
async fn an_unparameterised_request_resolves_prods_defaults() {
    use saber_serve::sampling::DefaultSampling;

    let (base, engine) = spawn_with_defaults(DefaultSampling::PROD).await;
    let _: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "Qwen3.8",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 4
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    let params = engine.submitted()[0].params.clone();
    assert!(
        (params.temperature - 0.7).abs() < 1e-6,
        "temperature {} is not prod's 0.7",
        params.temperature
    );
    assert!((params.top_p - 0.95).abs() < 1e-6, "top_p {}", params.top_p);
    assert_eq!(params.top_k, 20, "top_k");
    assert!((params.min_p - 0.0).abs() < 1e-6, "min_p {}", params.min_p);
    assert!(
        (params.repetition_penalty - 1.0).abs() < 1e-6,
        "repetition_penalty {}",
        params.repetition_penalty
    );

    let (base, engine) = spawn_with_defaults(DefaultSampling::STOCK).await;
    let _: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "Qwen3.8",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 4
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    let stock = engine.submitted()[0].params.clone();
    assert!((stock.temperature - 1.0).abs() < 1e-6);
    assert_eq!(stock.top_k, 0);
    assert_ne!(
        (params.temperature, params.top_p, params.top_k),
        (stock.temperature, stock.top_p, stock.top_k),
        "the control must differ from the case, or this test cannot fail"
    );

    // And a request that DOES name them keeps its own, which is the other half of the rule.
    let (base, engine) = spawn_with_defaults(DefaultSampling::PROD).await;
    let _: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "Qwen3.8",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 4, "temperature": 0.0, "top_p": 1.0, "top_k": -1
        }))
        .send()
        .await
        .expect("send")
        .json()
        .await
        .expect("json");
    let asked = engine.submitted()[0].params.clone();
    assert!((asked.temperature - 0.0).abs() < 1e-6);
    assert!((asked.top_p - 1.0).abs() < 1e-6);
    assert_eq!(
        asked.top_k, -1,
        "the reference server quietly accepts -1 as disabled"
    );
}

/// THE ENGINE'S NAMED ERROR ON THE WIRE (ws-k18): a choice the engine ended with a reason (a parked
/// session whose restore was given up on) is an OpenAI error the client can read -- a 500 with the
/// reason as `error.message` for a non-streaming request, and one `data: {"error": ...}` event
/// before `[DONE]` on a stream. The control is the same script ending with a plain `abort`: a 200
/// and no error event.
#[tokio::test]
async fn an_engine_error_reaches_the_client_as_an_openai_error() {
    let reason = "the session could not be restored after 3 attempt(s)";
    let failing = || {
        MockScript::new(vec![
            MockStep::new(bytes_of("Hi")),
            MockStep::new(Vec::<u32>::new()).failed(reason),
        ])
    };
    let aborting = || {
        MockScript::new(vec![
            MockStep::new(bytes_of("Hi")),
            MockStep::new(Vec::<u32>::new()).finish(FinishReason::Abort),
        ])
    };
    let post = |base: String, stream: bool| async move {
        reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&chat_body(json!({"stream": stream, "max_tokens": 16})))
            .send()
            .await
            .expect("request")
    };

    let (base, _e) = spawn(vec![failing()]).await;
    let r = post(base, false).await;
    assert_eq!(r.status().as_u16(), 500);
    let body: Value = r.json().await.expect("json");
    assert_eq!(body["error"]["message"], reason);
    assert_eq!(body["error"]["code"], 500);

    let (base, _e) = spawn(vec![failing()]).await;
    let text = post(base, true).await.text().await.expect("body");
    let errors: Vec<Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter(|v| v.get("error").is_some())
        .collect();
    assert_eq!(errors.len(), 1, "{text}");
    assert_eq!(errors[0]["error"]["message"], reason);
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");

    // THE CONTROL: a plain abort is a reply, not an error.
    let (base, _e) = spawn(vec![aborting()]).await;
    assert_eq!(post(base, false).await.status().as_u16(), 200);
    let (base, _e) = spawn(vec![aborting()]).await;
    let text = post(base, true).await.text().await.expect("body");
    assert!(!text.contains("\"error\""), "{text}");
}

/// `prompt_logprobs` and `--max-logprobs` at validation (V7), with prod's text: the pydantic
/// `check_logprobs` half (stream, negative) and `_validate_logprobs` (the cap, `-1` resolved to the
/// vocabulary on both sides of the test), on both endpoints.
#[tokio::test]
async fn prompt_logprobs_are_validated_like_prod() {
    let (base, _engine) = spawn(vec![]).await;
    let client = reqwest::Client::new();
    for (path, body, needle, param) in [
        (
            "/v1/completions",
            json!({"model": "Qwen3.8", "prompt": "hi", "stream": true, "prompt_logprobs": 1}),
            "`prompt_logprobs` are not available when `stream=True`. (parameter=prompt_logprobs)",
            json!("prompt_logprobs"),
        ),
        (
            "/v1/chat/completions",
            chat_body(json!({"stream": true, "prompt_logprobs": -1})),
            "`prompt_logprobs` are not available when `stream=True`. (parameter=prompt_logprobs)",
            json!("prompt_logprobs"),
        ),
        (
            "/v1/completions",
            json!({"model": "Qwen3.8", "prompt": "hi", "prompt_logprobs": -2}),
            "`prompt_logprobs` must be a positive value or -1. (parameter=prompt_logprobs, value=-2)",
            json!("prompt_logprobs"),
        ),
        (
            "/v1/completions",
            json!({"model": "Qwen3.8", "prompt": "hi", "prompt_logprobs": 21}),
            "Requested prompt logprobs of 21, which is greater than max allowed: 20 \
             (parameter=prompt_logprobs, value=21)",
            json!("prompt_logprobs"),
        ),
        (
            "/v1/completions",
            json!({"model": "Qwen3.8", "prompt": "hi", "prompt_logprobs": -1}),
            "Requested prompt logprobs of 248320, which is greater than max allowed: 20 \
             (parameter=prompt_logprobs, value=248320)",
            json!("prompt_logprobs"),
        ),
    ] {
        let response = client
            .post(format!("{base}{path}"))
            .json(&body)
            .send()
            .await
            .expect("request");
        assert_eq!(response.status().as_u16(), 400, "for {body}");
        let error: Value = response.json().await.expect("json");
        assert_eq!(error["error"]["message"], json!(needle), "for {body}");
        assert_eq!(error["error"]["param"], param, "for {body}");
    }
}

/// `--max-logprobs` moves the cap, and `-1` lifts it to the vocabulary (V7).
#[tokio::test]
async fn max_logprobs_moves_the_cap() {
    for (max, ask, ok) in [
        (21, 21, true),
        (21, 22, false),
        (-1, 248_320, true),
        (0, 1, false),
    ] {
        let engine = Arc::new(MockEngine::new(ModelInfo::default()));
        engine.push_script(MockScript::per_token([b'x' as u32], FinishReason::Length));
        let state =
            ServerState::builder(engine, Arc::new(ByteCodec), Arc::new(MinimalChatRenderer))
                .max_logprobs(max)
                .build();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move { saber_serve::server::serve(state, listener).await });
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/completions"))
            .json(&json!({"model": "Qwen3.8", "prompt": "hi", "max_tokens": 1,
                          "prompt_logprobs": ask}))
            .send()
            .await
            .expect("request");
        assert_eq!(
            response.status().is_success(),
            ok,
            "--max-logprobs {max}, prompt_logprobs {ask}"
        );
    }
}

// ── D3: image parts ─────────────────────────────────────────────────────────────

/// A 1x1 RGB PNG: `smart_resize` lifts it to 256x256 = 256 patches = 64 image tokens.
const PNG_1X1: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMQUDAAAACkAGE0Zn1yAAAAAElFTkSuQmCC";

/// `ByteCodec`, except that the normalised image part `{"type":"image"}` (which
/// `MinimalChatRenderer` prints as JSON) encodes as ONE `<|image_pad|>` id, the way the real
/// template + tokenizer emit it.
#[derive(Debug)]
struct ImageAwareCodec;

impl saber_serve::detok::TextCodec for ImageAwareCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>, saber_serve::protocol::ApiError> {
        let mut out = Vec::new();
        for (i, piece) in text.split("{\"type\":\"image\"}").enumerate() {
            if i > 0 {
                out.push(saber_vision_input::IMAGE_PAD_ID);
            }
            out.extend(piece.bytes().map(u32::from));
        }
        Ok(out)
    }

    fn decode(&self, ids: &[u32], skip: bool) -> String {
        ByteCodec.decode(ids, skip)
    }
}

async fn spawn_vision(max_patches: u32) -> (String, Arc<MockEngine>) {
    let engine = Arc::new(MockEngine::new(model_info()));
    engine.push_script(MockScript::new(vec![
        MockStep::new(bytes_of("ok")).finish(FinishReason::Stop)
    ]));
    let state = ServerState::builder(
        engine.clone(),
        Arc::new(ImageAwareCodec),
        Arc::new(MinimalChatRenderer),
    )
    .vision(max_patches, 999)
    .build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        saber_serve::server::serve(state, listener)
            .await
            .expect("serve")
    });
    (format!("http://{addr}"), engine)
}

fn image_body(url: &str) -> Value {
    json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "what?"}]}]})
}

/// A text-only server refuses an image part BY NAME (400, `param: image`) and counts it; a
/// vision server hands the engine the processed batch and counts images and patches.
#[tokio::test]
async fn image_parts_are_refused_by_name_or_handed_to_the_engine() {
    let client = reqwest::Client::new();
    let (base, _) = spawn_vision(0).await;
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&image_body(PNG_1X1))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["param"], "image");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("TEXT ONLY"));
    let m = client
        .get(format!("{base}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("saber:image_refused_total{engine=\"0\",model_name=\"Qwen3.8\",reason=\"disabled\"} 1.0"), "{m}");

    let (base, engine) = spawn_vision(65_536).await;
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&image_body(PNG_1X1))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let got = engine.submitted_images();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].1.grids, vec![[1, 16, 16]]);
    assert_eq!(got[0].1.pixel_values.len(), 256 * 1536);
    // the prompt carries the image's 64-token run where the one placeholder was
    let pads = got[0]
        .0
        .iter()
        .filter(|&&t| t == saber_vision_input::IMAGE_PAD_ID)
        .count();
    assert_eq!(pads, 64);
    let m = client
        .get(format!("{base}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        m.contains("saber:image_inputs_total{engine=\"0\",model_name=\"Qwen3.8\"} 1.0"),
        "{m}"
    );
    assert!(
        m.contains("saber:image_patches_total{engine=\"0\",model_name=\"Qwen3.8\"} 256.0"),
        "{m}"
    );

    // a bad data URL is prod's 501 with prod's text
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&image_body("data:image/png,abc"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 501);
    let body: Value = r.json().await.unwrap();
    assert_eq!(
        body["error"]["message"],
        "Only base64 data URLs are supported for now."
    );
    assert_eq!(body["error"]["type"], "NotImplementedError");
}

// ── D5: the chat logprob and id shapes, read off prod's own serving code ─────────────────────

/// One step whose chosen token `z` (-5.0) is OUTSIDE the candidates `a` (-0.1), `b` (-0.2), then
/// one whose chosen `a` is the first candidate, then the stop token — the cases prod's one dict
/// reads differently.
fn chosen_outside_top_k() -> MockScript {
    use saber_serve::engine::{SampledLogprobs, TokenLogprob};
    let lp = |id: u8, logprob: f32| TokenLogprob {
        token_id: u32::from(id),
        logprob,
        rank: None,
    };
    MockScript::new(vec![MockStep::new([u32::from(b'z'), u32::from(b'a'), 255])
        .finish(FinishReason::Stop)
        .with_logprobs(vec![
            SampledLogprobs {
                sampled: lp(b'z', -5.0),
                top: vec![lp(b'a', -0.1), lp(b'b', -0.2)],
            },
            SampledLogprobs {
                sampled: lp(b'a', -0.1),
                // The sampled id repeated among the candidates collapses into its first slot.
                top: vec![lp(b'a', -0.1), lp(b'b', -0.2)],
            },
            SampledLogprobs {
                sampled: lp(255, -0.01),
                top: vec![],
            },
        ])])
}

/// `/v1/chat/completions`' `top_logprobs` is the first `top_logprobs` entries of the reference server's dict
/// `{sampled, top-k...}` (`OpenAIServingChat._get_top_logprobs`: `i < top_logprobs`): the chosen
/// token is ALWAYS listed, and one outside the top-k pushes the k-th candidate out. D4 found
/// Saber listing the candidates other than the chosen one. The control is the pre-D5 shape
/// (candidates only), which this asserts against by name.
#[tokio::test]
async fn chat_top_logprobs_list_the_chosen_token_first_as_prod_does() {
    let (base, _engine) = spawn(vec![chosen_outside_top_k()]).await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "logprobs": true,
            "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": false}
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let content = response["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("logprobs content");
    let tops = |i: usize| -> Vec<String> {
        content[i]["top_logprobs"]
            .as_array()
            .expect("top_logprobs")
            .iter()
            .map(|e| e["token"].as_str().expect("token").to_owned())
            .collect()
    };
    assert_eq!(tops(0), vec!["z", "a"], "chosen first, the k-th pushed out");
    assert_ne!(
        tops(0),
        vec!["a", "b"],
        "the pre-D5 shape (candidates only)"
    );
    assert_eq!(tops(1), vec!["a", "b"], "a chosen candidate is listed once");
    assert!(
        (content[0]["top_logprobs"][0]["logprob"]
            .as_f64()
            .expect("lp")
            + 5.0)
            .abs()
            < 1e-6
    );

    // top_logprobs 0 with logprobs on: prod's `top_logprobs and i < ...` lists nothing.
    let (base, _engine) = spawn(vec![chosen_outside_top_k()]).await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(json!({
            "logprobs": true,
            "top_logprobs": 0,
            "chat_template_kwargs": {"enable_thinking": false}
        })))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    assert_eq!(
        response["choices"][0]["logprobs"]["content"][0]["top_logprobs"],
        json!([])
    );
}

/// `/v1/completions` reads the same dict to `i <= logprobs` (`num_output_top_logprobs >= i`):
/// the chosen token plus the whole top-k — k + 1 keys when the chosen one is outside it.
#[tokio::test]
async fn completion_top_logprobs_carry_the_chosen_token_and_the_whole_top_k() {
    let (base, _engine) = spawn(vec![chosen_outside_top_k()]).await;
    let response: Value = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&json!({"model": "Qwen3.8", "prompt": "hi", "logprobs": 2, "max_tokens": 16}))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let top = &response["choices"][0]["logprobs"]["top_logprobs"];
    let keys = |i: usize| -> Vec<String> {
        let mut k: Vec<String> = top[i].as_object().expect("map").keys().cloned().collect();
        k.sort();
        k
    };
    assert_eq!(keys(0), vec!["a", "b", "z"], "{response}");
    assert_eq!(keys(1), vec!["a", "b"]);
}

/// The reference server's `return_token_ids` on chat, as prod's `serving.py` builds it: the choice's ids with
/// the STOP token included (the detokenizer skips it for the text and appends it back), the
/// prompt's ids on the response; `null` on both when not asked (the body is a `model_dump()`).
/// Streamed: the first chunk sets `prompt_token_ids`, each content / finish chunk sets its own
/// `token_ids` (null when not asked), and the opening chunk carries none.
#[tokio::test]
async fn chat_return_token_ids_as_prod_does() {
    for asked in [true, false] {
        let (base, _engine) = spawn(vec![chosen_outside_top_k()]).await;
        let response: Value = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&chat_body(json!({
                "return_token_ids": asked,
                "chat_template_kwargs": {"enable_thinking": false}
            })))
            .send()
            .await
            .expect("request")
            .json()
            .await
            .expect("json");
        let choice = &response["choices"][0];
        assert_eq!(choice["message"]["content"], json!("za"), "{response}");
        if asked {
            assert_eq!(
                choice["token_ids"],
                json!([122, 97, 255]),
                "the stop id included"
            );
            let prompt = response["prompt_token_ids"].as_array().expect("prompt ids");
            assert_eq!(
                prompt.len() as u64,
                response["usage"]["prompt_tokens"].as_u64().expect("usage")
            );
        } else {
            assert_eq!(choice.get("token_ids"), Some(&Value::Null), "{response}");
            assert_eq!(response.get("prompt_token_ids"), Some(&Value::Null));
        }

        let (base, _engine) = spawn(vec![chosen_outside_top_k()]).await;
        let body = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&chat_body(json!({
                "stream": true,
                "return_token_ids": asked,
                "chat_template_kwargs": {"enable_thinking": false}
            })))
            .send()
            .await
            .expect("request")
            .text()
            .await
            .expect("body");
        let chunks: Vec<Value> = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).expect("chunk"))
            .collect();
        let first = &chunks[0];
        assert!(first["choices"][0].get("token_ids").is_none(), "{first}");
        let mut streamed = Vec::new();
        for c in &chunks[1..] {
            assert!(
                c.get("prompt_token_ids").is_none(),
                "only the first chunk: {c}"
            );
            let ids = c["choices"][0]
                .get("token_ids")
                .expect("set on every step chunk");
            if asked {
                streamed.extend(ids.as_array().expect("ids").iter().cloned());
            } else {
                assert_eq!(ids, &Value::Null);
            }
        }
        if asked {
            assert!(first["prompt_token_ids"].is_array(), "{first}");
            assert_eq!(streamed, vec![json!(122), json!(97), json!(255)]);
        } else {
            assert_eq!(first.get("prompt_token_ids"), Some(&Value::Null), "{first}");
        }
    }
}
