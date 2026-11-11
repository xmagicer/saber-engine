//! The HTTP surface: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
//! `/health`, `/metrics`.
//!
//! **The invariant this module exists to hold: one SSE chunk per engine step.**
//! BetterBench classifies a stream from `completion_tokens / n_chunks` and reports
//! `update p50`, which is only ms/step because prod emits one update per step (with
//! DFlash2 that update carries 1..=8 tokens). Every [`StepEvent`] therefore becomes
//! exactly one `chat.completion.chunk`, carrying all of that step's decoded text —
//! never one chunk per token.

use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde_json::{Map, Value};

use crate::detok::IncrementalDetokenizer;
use crate::engine::{
    EngineRequest, FinishReason, PromptPositionLogprobs, SampledLogprobs, StepEvent, Submission,
    TokenId,
};
use crate::ids;
use crate::parse::compose::{tool_choice_refusal, RequestShape, ServeParser};
use crate::parse::{DeltaMessage, ToolCall};
use crate::protocol::{
    ApiError, ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatLogprobs, ChunkChoice, CompletionChoice, CompletionLogprobs, CompletionRequest,
    CompletionResponse, CompletionStreamChoice, CompletionStreamChunk, Delta,
    DeltaFunctionCallWire, DeltaToolCallWire, FunctionCallWire, LogprobEntry, ModelCard, ModelList,
    ResponseMessage, StreamUsage, ToolCallWire, Usage,
};
use crate::sampling::SamplingParams;
use crate::state::ServerState;

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("serve: {0}")]
    Io(#[from] std::io::Error),
}

/// The whole `/v1` surface.
pub fn router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(state)
}

/// Serve until the process ends. `listener` may be bound to port 0, which is how the
/// HTTP tests get an ephemeral port.
pub async fn serve(
    state: Arc<ServerState>,
    listener: tokio::net::TcpListener,
) -> Result<(), ServeError> {
    axum::serve(listener, router(state)).await?;
    Ok(())
}

/// The same server, returning when `shutdown` resolves.
///
/// A server that only stops when its process is killed is a server that cannot be stopped at all
/// once a kill arrives while a thread is inside a blocking device call: on 2026-09-19 a
/// `saber serve --engine real` in a gate script took a `SIGINT`, did not exit, and held the shared
/// GPU lock — and 18.7 GiB of a card — for 2 h 55 m while the script waited on it. So the binary
/// hands a signal future in here, axum stops accepting and lets in-flight responses finish, and the
/// caller then tears the engine down against a deadline of its own.
pub async fn serve_with_shutdown<F>(
    state: Arc<ServerState>,
    listener: tokio::net::TcpListener,
    shutdown: F,
) -> Result<(), ServeError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn health() -> Response {
    axum::http::StatusCode::OK.into_response()
}

async fn metrics(State(state): State<Arc<ServerState>>) -> Response {
    let body = state.metrics.render(state.engine.stats());
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

async fn models(State(state): State<Arc<ServerState>>) -> Json<ModelList> {
    let model = state.engine.model();
    let data = state
        .model_ids()
        .into_iter()
        .map(|id| ModelCard {
            id,
            object: "model",
            created: state.created,
            owned_by: "saber",
            root: model.root.clone(),
            parent: None,
            max_model_len: model.max_model_len,
        })
        .collect();
    Json(ModelList {
        object: "list",
        data,
    })
}

// ── request → sampling params ───────────────────────────────────────────────────

fn f64_to_f32(value: Option<f64>, default: f32) -> f32 {
    value.map_or(default, |v| v as f32)
}

/// A stop list, with the server's own `stop_token_ids` merged in the way
/// `to_sampling_params` merges them: the request's first, the server's appended, dupes
/// dropped, order preserved.
fn merge_stop_token_ids(request: &[TokenId], server: &[TokenId]) -> Vec<TokenId> {
    let mut out: Vec<TokenId> = Vec::with_capacity(request.len() + server.len());
    for id in request.iter().chain(server.iter()) {
        if !out.contains(id) {
            out.push(*id);
        }
    }
    out
}

fn sampling_from_chat(
    req: &ChatCompletionRequest,
    state: &ServerState,
) -> Result<SamplingParams, ApiError> {
    check_prompt_logprobs(req.stream, req.prompt_logprobs)?;
    // `n` and the `top_k` range are checked by `SamplingParams::validate` in the reference server's own
    // order, and neither carries a `parameter=` on the wire. What is checked HERE is only
    // what the cast below would otherwise lose.
    let n = req.n.unwrap_or(1).max(0) as usize;
    // `ChatCompletionRequest.to_sampling_params` fills an omitted temperature, top_p,
    // top_k, min_p or repetition_penalty from `default_sampling_params` — the merged
    // `generation_config.json` — NOT from OpenAI's defaults. Prod's are 0.7 / 0.95 / 20.
    let defaults = state.default_sampling;
    let top_k = req.top_k.unwrap_or(i64::from(defaults.top_k));
    if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&top_k) {
        return Err(ApiError::bad_request(
            format!("top_k must be 0 (disable), or at least 1, got {top_k}."),
            None,
        ));
    }
    // the reference server: `top_logprobs` is meaningless unless `logprobs` is true.
    let logprobs = match (req.logprobs.unwrap_or(false), req.top_logprobs) {
        (true, Some(k)) => Some(k),
        (true, None) => Some(0),
        (false, Some(_)) => {
            return Err(ApiError::bad_request(
                "when using `top_logprobs`, `logprobs` must be set to true.",
                Some("top_logprobs"),
            ))
        }
        (false, None) => None,
    };
    let max_tokens = match (req.max_completion_tokens, req.max_tokens) {
        (Some(v), _) | (None, Some(v)) => {
            if v < 0 {
                return Err(ApiError::invalid_param(
                    format!("max_tokens must be at least 1, got {v}."),
                    "max_tokens",
                    v,
                ));
            }
            Some(v as usize)
        }
        (None, None) => None,
    };

    let params = SamplingParams {
        n,
        temperature: f64_to_f32(req.temperature, defaults.temperature),
        top_p: f64_to_f32(req.top_p, defaults.top_p),
        top_k: top_k as i32,
        min_p: f64_to_f32(req.min_p, defaults.min_p),
        max_tokens,
        stop: req
            .stop
            .clone()
            .map(crate::protocol::StringOrVec::into_vec)
            .unwrap_or_default(),
        stop_token_ids: merge_stop_token_ids(
            &req.stop_token_ids.clone().unwrap_or_default(),
            &state.default_stop_token_ids,
        ),
        seed: req.seed.map(|s| s as u64),
        logprobs: logprobs.map(|v| v as i32),
        prompt_logprobs: req.prompt_logprobs.map(|v| v as i32),
        presence_penalty: f64_to_f32(req.presence_penalty, 0.0),
        frequency_penalty: f64_to_f32(req.frequency_penalty, 0.0),
        repetition_penalty: f64_to_f32(req.repetition_penalty, defaults.repetition_penalty),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        include_stop_str_in_output: req.include_stop_str_in_output.unwrap_or(false),
    };
    params.validate(state.limits)?;
    Ok(params)
}

fn sampling_from_completion(
    req: &CompletionRequest,
    state: &ServerState,
) -> Result<SamplingParams, ApiError> {
    check_prompt_logprobs(req.stream, req.prompt_logprobs)?;
    // `n` and the `top_k` range are checked by `SamplingParams::validate` in the reference server's own
    // order, and neither carries a `parameter=` on the wire. What is checked HERE is only
    // what the cast below would otherwise lose.
    let n = req.n.unwrap_or(1).max(0) as usize;
    let defaults = state.default_sampling;
    let params = SamplingParams {
        n,
        temperature: f64_to_f32(req.temperature, defaults.temperature),
        top_p: f64_to_f32(req.top_p, defaults.top_p),
        top_k: req.top_k.unwrap_or(i64::from(defaults.top_k)) as i32,
        min_p: f64_to_f32(req.min_p, defaults.min_p),
        max_tokens: req.max_tokens.map(|v| v.max(0) as usize),
        stop: req
            .stop
            .clone()
            .map(crate::protocol::StringOrVec::into_vec)
            .unwrap_or_default(),
        stop_token_ids: merge_stop_token_ids(
            &req.stop_token_ids.clone().unwrap_or_default(),
            &state.default_stop_token_ids,
        ),
        seed: req.seed.map(|s| s as u64),
        logprobs: req.logprobs.map(|v| v as i32),
        prompt_logprobs: req.prompt_logprobs.map(|v| v as i32),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        repetition_penalty: f64_to_f32(req.repetition_penalty, defaults.repetition_penalty),
        ..SamplingParams::default()
    };
    params.validate(state.limits)?;
    Ok(params)
}

/// What one request's parsers are built from: the server's flag profile applied to the request's
/// `tools`, `tool_choice` and `enable_thinking` (V9). Owned, because the streaming generator
/// outlives the handler that read the request.
#[derive(Debug, Clone)]
struct ParseSpec {
    profile: crate::parse::compose::ParserProfile,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    enable_thinking: bool,
    drop_tokens: Vec<String>,
}

impl ParseSpec {
    fn new(state: &ServerState, req: &ChatCompletionRequest, kwargs: &Map<String, Value>) -> Self {
        Self {
            profile: state.parser_profile(),
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
            enable_thinking: kwargs
                .get("enable_thinking")
                .and_then(Value::as_bool)
                .unwrap_or(state.default_enable_thinking),
            drop_tokens: state.drop_tokens.clone(),
        }
    }

    fn shape(&self) -> RequestShape<'_> {
        RequestShape {
            tools: self.tools.as_deref(),
            tool_choice: self.tool_choice.as_ref(),
            enable_thinking: self.enable_thinking,
            drop_tokens: &self.drop_tokens,
        }
    }

    /// One choice's parser; `None` when the server runs no parser (the reply is the raw text).
    fn parser(&self) -> Option<ServeParser> {
        ServeParser::new(&self.profile, &self.shape())
    }
}

// ── per-choice step machinery ───────────────────────────────────────────────────

/// What one engine step produced for one choice.
#[derive(Debug, Default)]
struct StepDelta {
    text: String,
    logprobs: Vec<LogprobEntry>,
    finished: Option<FinishReason>,
    /// The ids this step EMITTED, prod's law (D5): after the budget, up to and including a stop
    /// token (the reference server's detokenizer skips the stop id for the text and appends it back to the ids).
    ids: Vec<TokenId>,
}

/// Which endpoint's `top_logprobs` shape a choice reports (D5).
///
/// Both are the reference server's one dict per position — the SAMPLED id inserted first, then the top-k in
/// order, a repeat of the sampled id collapsing into its first slot — read two ways:
/// `/v1/chat/completions` takes the FIRST `top_logprobs` entries of it
/// (`OpenAIServingChat._get_top_logprobs`: `i < top_logprobs`), so the chosen token is always
/// listed and a chosen token outside the top-k pushes the k-th out; `/v1/completions` takes
/// entries `0..=logprobs` (`num_output_top_logprobs >= i`), the chosen token plus the whole top-k.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogprobShape {
    Chat,
    Completion,
}

/// Drives one choice: detokenization, the budget, stop strings and logprobs.
#[derive(Debug)]
struct ChoiceRunner {
    detok: IncrementalDetokenizer,
    text: String,
    completion_tokens: u64,
    finished: Option<FinishReason>,
    stop_reason: Option<String>,
    prompt_logprobs: Option<Vec<Option<Map<String, Value>>>>,
    max_tokens: usize,
    stop: Vec<String>,
    stop_token_ids: HashSet<TokenId>,
    /// Of those, the ones the REQUEST asked for rather than the model's own end-of-turn.
    ///
    /// The reference server reports `stop_reason` for a stop the caller specified and leaves it null when the model
    /// simply finished, and a client that keys on it can tell "it stopped because I said so" from
    /// "it had nothing more to say". Same set, two meanings, so they are two fields.
    user_stop_token_ids: HashSet<TokenId>,
    include_stop_str: bool,
    want_logprobs: bool,
    top_logprobs: usize,
    /// Every id this choice emitted, in order, by prod's law: the stop token included (the reference server's
    /// detokenizer appends the skipped stop id back to `token_ids`). Kept unconditionally: it is
    /// one `Vec<u32>` per choice and a flag that changes what is RECORDED would make the two
    /// paths differ.
    token_ids: Vec<TokenId>,
    shape: LogprobShape,
    /// The last (at most four) sampled ids this choice REPORTED logprobs for: the context prod
    /// re-decodes a split multi-byte token with.
    lp_context: Vec<TokenId>,
    /// Steps where the engine delivered more tokens than the request's budget. A
    /// clamp, not a route, but it is counted so it can never be silent.
    budget_overruns: u64,
    /// The engine's reason for ending this choice, when it gave one (`StepEvent::error`, ws-k18).
    error: Option<String>,
    codec: Arc<dyn crate::detok::TextCodec>,
}

impl ChoiceRunner {
    fn new(
        state: &ServerState,
        params: &SamplingParams,
        prompt_tokens: usize,
        skip_special_tokens: bool,
        shape: LogprobShape,
    ) -> Self {
        let mut stop_token_ids: HashSet<TokenId> =
            state.engine.model().eos_token_ids.iter().copied().collect();
        stop_token_ids.extend(params.stop_token_ids.iter().copied());
        let user_stop_token_ids: HashSet<TokenId> = params.stop_token_ids.iter().copied().collect();
        Self {
            // Special tokens stay in the text whenever a parser runs: every engine adapter's
            // `adjust_request` sets `skip_special_tokens=False` and the parser drops what is not
            // grammar. With no parser the request's own flag (default `true`) is the
            // detokenizer's (V9).
            detok: IncrementalDetokenizer::new(Arc::clone(&state.codec), skip_special_tokens),
            text: String::new(),
            completion_tokens: 0,
            finished: None,
            stop_reason: None,
            prompt_logprobs: None,
            max_tokens: params.effective_max_tokens(prompt_tokens, state.limits),
            stop: params.stop.clone(),
            stop_token_ids,
            user_stop_token_ids,
            include_stop_str: params.include_stop_str_in_output,
            want_logprobs: params.logprobs.is_some(),
            top_logprobs: params.logprobs.unwrap_or(0).max(0) as usize,
            token_ids: Vec::new(),
            shape,
            lp_context: Vec::new(),
            budget_overruns: 0,
            error: None,
            codec: Arc::clone(&state.codec),
        }
    }

    fn on_step(&mut self, event: &StepEvent) -> StepDelta {
        // ONCE FINISHED, NOTHING MORE IS ACCUMULATED. Setting `finished` used to be a message to
        // the caller and nothing else, so a driver that kept feeding steps -- which the
        // non-streaming path does, since it collects the whole stream before assembling -- went on
        // appending text after the reply had ended. The stop rule has to hold at the point the text
        // is built, not only at the point the client is told.
        if self.finished.is_some() {
            return StepDelta::default();
        }
        if event.error.is_some() {
            self.error.clone_from(&event.error);
        }
        if let Some(prompt_logprobs) = &event.prompt_logprobs {
            self.prompt_logprobs =
                Some(encode_prompt_logprobs(prompt_logprobs, self.codec.as_ref()));
        }
        let mut tokens = event.new_tokens.clone();
        let mut finished = event.finished;

        // Budget: the engine is authoritative, but a request's max_tokens is a
        // promise to the client, so an overrun is clamped rather than forwarded.
        let budget = self
            .max_tokens
            .saturating_sub(self.completion_tokens as usize);
        if tokens.len() > budget {
            tokens.truncate(budget);
            self.budget_overruns += 1;
            finished = Some(FinishReason::Length);
        }
        let logprobs = self.take_logprobs(event, tokens.len());
        self.completion_tokens += tokens.len() as u64;
        if finished.is_none() && self.completion_tokens as usize >= self.max_tokens {
            finished = Some(FinishReason::Length);
        }

        // A STOP TOKEN ENDS THE REQUEST. It had only ever been filtered out of the text, which is
        // half a stop rule: the token vanished from the reply and generation carried on to
        // `max_tokens`, so every completion ran its whole budget and everything after the model's
        // own `<|im_end|>` — a fresh `<|im_start|>`, a new turn, and then whatever the model does
        // past the end of a conversation — was detokenized into the reply with the delimiters
        // silently removed. ws-p5's multi-turn gate saw that as replies like `ODA6666666…` running
        // the full 256 tokens against prod's 299/300 healthy. A request that is over is over.
        let stop_at = tokens
            .iter()
            .position(|id| self.stop_token_ids.contains(id));
        let stop_id = stop_at.and_then(|at| tokens.get(at).copied());
        // What the step EMITTED: up to and including the stop token, whatever the text shows.
        let emitted: Vec<TokenId> = match stop_at {
            Some(at) => tokens[..=at].to_vec(),
            None => tokens.clone(),
        };
        self.token_ids.extend_from_slice(&emitted);
        if let Some(at) = stop_at {
            // `include_stop_str_in_output` is about the stop STRING; a stop token is a control
            // token and prod never shows it, so the text stops before it either way. What the
            // client is owed is the reason.
            tokens.truncate(at + usize::from(self.include_stop_str));
            if finished.is_none() {
                finished = Some(FinishReason::Stop);
            }
            // Null for the model's own end-of-turn, the id for a stop the caller asked for.
            if let Some(id) = tokens.get(at).copied().or(stop_id) {
                if self.user_stop_token_ids.contains(&id) && self.stop_reason.is_none() {
                    self.stop_reason = Some(id.to_string());
                }
            }
        }
        let visible: Vec<TokenId> = tokens
            .iter()
            .copied()
            .filter(|id| !self.stop_token_ids.contains(id))
            .collect();
        let mut text = self.detok.push(&visible);
        if finished.is_some() {
            text.push_str(&self.detok.finish());
        }
        let text = self.apply_stop_strings(text, &mut finished);
        self.finished = finished;
        StepDelta {
            text,
            logprobs,
            finished,
            ids: emitted,
        }
    }

    fn take_logprobs(&mut self, event: &StepEvent, kept: usize) -> Vec<LogprobEntry> {
        if !self.want_logprobs {
            return Vec::new();
        }
        let Some(logprobs) = &event.logprobs else {
            return Vec::new();
        };
        logprobs
            .iter()
            .take(kept)
            .map(|lp| {
                let entry = self.encode_logprob(lp);
                // The reference server's context for the NEXT position: this position's sampled id (the first key
                // of its dict), the last four kept (`_get_sampled_context_ids`, max_context 4).
                self.lp_context.push(lp.sampled.token_id);
                if self.lp_context.len() > 4 {
                    self.lp_context.remove(0);
                }
                entry
            })
            .collect()
    }

    /// One token's text as prod reports it in a logprob entry (D5): `tokenizer.decode([id])`
    /// (`convert_ids_list_to_tokens`), NOT the BPE piece — prod's `" user"` where the piece is
    /// `"Ġuser"` — and a text ending in U+FFFD re-decoded with up to four preceding SAMPLED ids of
    /// this choice as context (`LogprobsProcessor._verify_tokens`, the prompt path's own rule).
    fn decoded_token(&self, id: TokenId) -> String {
        let decode = |ids: &[TokenId]| self.codec.decode(ids, false);
        let text = decode(&[id]);
        if text.ends_with('\u{FFFD}') {
            crate::prompt_logprobs::correct_decoded_token(&decode, id, &self.lp_context)
        } else {
            text
        }
    }

    fn encode_logprob(&self, lp: &SampledLogprobs) -> LogprobEntry {
        let piece = self.decoded_token(lp.sampled.token_id);
        // The reference server's dict: the sampled id first, then the candidates (a repeat of it collapses into
        // its first slot), read by the endpoint's own rule (`LogprobShape`).
        let take = match self.shape {
            LogprobShape::Chat => self.top_logprobs,
            LogprobShape::Completion => self.top_logprobs + 1,
        };
        let top = std::iter::once(&lp.sampled)
            .chain(
                lp.top
                    .iter()
                    .filter(|entry| entry.token_id != lp.sampled.token_id),
            )
            .take(take)
            .map(|entry| {
                let piece = self.decoded_token(entry.token_id);
                LogprobEntry {
                    bytes: piece.as_bytes().to_vec(),
                    token: piece,
                    logprob: entry.logprob,
                    top_logprobs: None,
                }
            })
            .collect();
        LogprobEntry {
            bytes: piece.as_bytes().to_vec(),
            token: piece,
            logprob: lp.sampled.logprob,
            top_logprobs: Some(top),
        }
    }

    /// Stop strings are the serving plane's job: it is the side that detokenizes.
    /// The search starts far enough back that a stop string split across two steps is
    /// still found.
    fn apply_stop_strings(&mut self, delta: String, finished: &mut Option<FinishReason>) -> String {
        if self.stop.is_empty() || delta.is_empty() {
            self.text.push_str(&delta);
            return delta;
        }
        let longest = self.stop.iter().map(String::len).max().unwrap_or(0);
        let combined = format!("{}{}", self.text, delta);
        // Start far enough back to catch a stop string split across two steps, and
        // walk to a character boundary: slicing mid-character would silently find
        // nothing rather than fail.
        let mut search_from = self.text.len().saturating_sub(longest.saturating_sub(1));
        while search_from > 0 && !combined.is_char_boundary(search_from) {
            search_from -= 1;
        }
        let mut hit: Option<(usize, &str)> = None;
        for stop in &self.stop {
            if let Some(pos) = combined
                .get(search_from..)
                .and_then(|tail| tail.find(stop.as_str()))
            {
                let pos = search_from + pos;
                if hit.is_none_or(|(best, _)| pos < best) {
                    hit = Some((pos, stop.as_str()));
                }
            }
        }
        match hit {
            None => {
                self.text = combined;
                delta
            }
            Some((pos, stop)) => {
                let end = if self.include_stop_str {
                    pos + stop.len()
                } else {
                    pos
                };
                let truncated = combined[..end].to_owned();
                let emitted = truncated[self.text.len().min(truncated.len())..].to_owned();
                self.stop_reason = Some(stop.to_owned());
                self.text = truncated;
                *finished = Some(FinishReason::Stop);
                emitted
            }
        }
    }
}

/// `prompt_logprobs` on the wire, in prod's shape: one entry per prompt position, `null` for the
/// first, and per position an object keyed by token id whose FIRST key is the prompt token itself
/// (`crate::prompt_logprobs::position` builds that order; `serde_json`'s `preserve_order` keeps it).
///
/// Two of prod's output-side rules are applied here because they are about TEXT, which the
/// engine does not have: `clamp_prompt_logprobs` (`-inf` is sent as `-9999.0`) and
/// `LogprobsProcessor._verify_tokens` — a piece ending in U+FFFD is re-decoded with up to four
/// preceding PROMPT tokens of context (the first key of each earlier non-null position).
fn encode_prompt_logprobs(
    positions: &[PromptPositionLogprobs],
    codec: &dyn crate::detok::TextCodec,
) -> Vec<Option<Map<String, Value>>> {
    let decode = |ids: &[crate::engine::TokenId]| codec.decode(ids, false);
    // `_get_sampled_context_ids(self.prompt_logprobs)`: the first key of each earlier position,
    // `None` positions skipped — so position 0's token is never context.
    let mut context: Vec<crate::engine::TokenId> = Vec::new();
    positions
        .iter()
        .map(|position| {
            let encoded = position.as_ref().map(|entries| {
                let mut map = Map::new();
                let ctx = &context[context.len().saturating_sub(4)..];
                for entry in entries {
                    let mut value = Map::new();
                    let lp = if entry.logprob == f32::NEG_INFINITY {
                        crate::prompt_logprobs::PROMPT_LOGPROB_FLOOR
                    } else {
                        entry.logprob
                    };
                    value.insert("logprob".to_owned(), Value::from(f64::from(lp)));
                    value.insert(
                        "rank".to_owned(),
                        entry.rank.map_or(Value::Null, Value::from),
                    );
                    // `convert_ids_list_to_tokens` is `tokenizer.decode([id])` per id — the
                    // TEXT, not the BPE piece `id_to_piece` gives the sample-logprob path.
                    let mut piece = codec.decode(&[entry.token_id], false);
                    if piece.ends_with('\u{FFFD}') {
                        piece = crate::prompt_logprobs::correct_decoded_token(
                            &decode,
                            entry.token_id,
                            ctx,
                        );
                    }
                    value.insert("decoded_token".to_owned(), Value::String(piece));
                    map.insert(entry.token_id.to_string(), Value::Object(value));
                }
                map
            });
            if let Some(first) = position.as_ref().and_then(|e| e.first()) {
                context.push(first.token_id);
            }
            encoded
        })
        .collect()
}

/// `check_logprobs`, the `mode="before"` validator on both request types: its `prompt_logprobs`
/// half, with prod's text. Runs before the sampling parameters are built, as pydantic does.
fn check_prompt_logprobs(stream: bool, prompt_logprobs: Option<i64>) -> Result<(), ApiError> {
    let Some(plp) = prompt_logprobs else {
        return Ok(());
    };
    if stream && (plp > 0 || plp == -1) {
        return Err(ApiError::bad_request(
            "`prompt_logprobs` are not available when `stream=True`. (parameter=prompt_logprobs)",
            Some("prompt_logprobs"),
        ));
    }
    if plp < 0 && plp != -1 {
        return Err(ApiError::invalid_param(
            "`prompt_logprobs` must be a positive value or -1.",
            "prompt_logprobs",
            plp,
        ));
    }
    Ok(())
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn tool_calls_to_wire(calls: &[ToolCall]) -> Vec<ToolCallWire> {
    calls
        .iter()
        .map(|call| ToolCallWire {
            id: call.id.clone(),
            kind: "function".to_owned(),
            function: FunctionCallWire {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        })
        .collect()
}

fn delta_to_wire(delta: &DeltaMessage) -> Delta {
    Delta {
        role: None,
        content: delta.content.clone(),
        reasoning: delta.reasoning.clone(),
        tool_calls: if delta.tool_calls.is_empty() {
            None
        } else {
            Some(
                delta
                    .tool_calls
                    .iter()
                    .map(|tc| DeltaToolCallWire {
                        index: tc.index.max(0) as u32,
                        id: tc.id.clone(),
                        kind: tc.kind.map(str::to_owned),
                        function: DeltaFunctionCallWire {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    })
                    .collect(),
            )
        },
    }
}

// ── /v1/chat/completions ────────────────────────────────────────────────────────

/// The template kwargs of a chat request: `chat_template_kwargs` + the top-level
/// `reasoning_effort` + the server's thinking default (FN66: the warm runner renders through it).
pub(crate) fn chat_kwargs(state: &ServerState, req: &ChatCompletionRequest) -> Map<String, Value> {
    let mut kwargs = req.chat_template_kwargs.clone().unwrap_or_default();
    if let Some(e) = &req.reasoning_effort {
        kwargs
            .entry("reasoning_effort")
            .or_insert_with(|| Value::String(e.clone()));
    }
    // A server whose default is thinking OFF (FN14's `--fn-thinking off`) tells the TEMPLATE too,
    // not only the parser: a request that does not say renders with `enable_thinking=false`. The
    // default-on server (every 27B boot) leaves the kwargs as sent.
    if !state.default_enable_thinking {
        kwargs
            .entry("enable_thinking")
            .or_insert(Value::Bool(false));
    }
    kwargs
}

/// The chat request path's template + tokenizer half (the tool-call `arguments` law, the
/// conversation form, the render, the encode) of a text conversation. FN66: the warm runner
/// renders its prompts through this very function.
///
/// P27-8: NO PANIC LEAVES IT. A panic anywhere in that half (the arguments law, the conversation
/// rebuild, the template engine, the encoder) is the request's 400 naming it, never an unwind
/// into the handler (axum drops the connection on one: a system-only request did, on the 27B).
pub(crate) fn render_chat_ids(
    state: &ServerState,
    messages: &[Value],
    req: &ChatCompletionRequest,
    kwargs: &Map<String, Value>,
) -> Result<Vec<TokenId>, ApiError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        render_chat_ids_unguarded(state, messages, req, kwargs)
    }))
    .unwrap_or_else(|p| {
        Err(ApiError::bad_request(
            format!(
                "chat template: render failed: {}",
                saber_model::template::panic_text(&*p)
            ),
            None,
        ))
    })
}

fn render_chat_ids_unguarded(
    state: &ServerState,
    messages: &[Value],
    req: &ChatCompletionRequest,
    kwargs: &Map<String, Value>,
) -> Result<Vec<TokenId>, ApiError> {
    // FN25/FN27: the tool-call `arguments` law. `Verbatim` renders the messages as sent; the 27B's
    // plane (`ObjectsStrict`) renders prod 0.27.1's post-processed conversation, the Flash-Next
    // plane (`Objects`) the reference 0.29's; a request with no assistant `tool_calls` list is the same
    // JSON under every law.
    let objects = match state.tool_arguments {
        crate::state::ToolArguments::Verbatim => None,
        crate::state::ToolArguments::Objects => {
            crate::reqcheck::tool_arguments_to_objects(messages)
        }
        // FN27: the 27B plane, prod 0.27.1's step (a malformed `arguments` was already its 400).
        crate::state::ToolArguments::ObjectsStrict => {
            crate::reqcheck::tool_arguments_to_objects_0271(messages)
        }
    };
    let messages = objects.as_ref().map_or(messages, Vec::as_slice);
    // FN28: the conversation the template renders -- as sent, or the one prod 0.27.1 rebuilds
    // (`_parse_chat_message_content`; the 27B's plane).
    let rebuilt = match state.conversation_form {
        crate::state::ConversationForm::AsSent => None,
        crate::state::ConversationForm::Prod0271 => {
            Some(crate::conversation::conversation_0271(messages))
        }
    };
    let messages = rebuilt.as_ref().map_or(messages, Vec::as_slice);
    let prompt = state.renderer.render(
        messages,
        req.tools.as_deref(),
        req.add_generation_prompt.unwrap_or(true),
        kwargs,
    )?;
    state.codec.encode(&prompt)
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    // THE SHAPE FIRST, AND BEFORE THE MODEL. Prod validates the whole body with pydantic before
    // the handler runs, so a request that is both malformed and names a model that does not exist
    // gets the 400 and not the 404 (ws-v5). `_check_model` is inside the handler; this is not.
    if let Err(e) = crate::reqcheck::validate_chat_shape_for(&req, state.tool_arguments) {
        state.metrics.record_request_refusal(&e.body.kind);
        return Err(e);
    }
    let model = state.resolve_model(req.model.as_deref())?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(
            "messages must contain at least one message.",
            Some("messages"),
        ));
    }
    let kwargs = chat_kwargs(&state, &req);
    // THE FLAGS' REFUSAL (V9): a `tool_choice` the server's parser flags cannot serve is prod's
    // 400, before the template -- `OnlineRenderer.render_chat_request` runs it after the model
    // check and before rendering. Under prod's profile it never fires.
    let spec = ParseSpec::new(&state, &req, &kwargs);
    if let Some(message) = tool_choice_refusal(&spec.profile, &spec.shape()) {
        let e = ApiError::bad_request(message, None);
        state.metrics.record_request_refusal(&e.body.kind);
        return Err(e);
    }
    let skip_special_tokens = !spec.profile.any() && req.skip_special_tokens.unwrap_or(true);
    // D3: image parts. The reference server parses the conversation's parts (counting images as they are added)
    // and loads every image BEFORE it renders the template; so does this. A request with no
    // image part is rendered from its messages exactly as sent.
    let refuse = |r: crate::images::Refusal| {
        state.metrics.record_image_refusal(r.label);
        r.error
    };
    let extracted = crate::images::extract(&req.messages, &state.vision).map_err(refuse)?;
    let images = match &extracted {
        None => None,
        Some(e) => {
            let (urls, policy) = (e.urls.clone(), state.vision);
            let loaded =
                tokio::task::spawn_blocking(move || crate::images::load_counted(&urls, &policy))
                    .await
                    .map_err(|e| ApiError::internal(format!("image loader: {e}")))?;
            let l = loaded.map_err(refuse)?;
            // FN126: the images the arena clamp downscaled (answered at the arena's ceiling)
            state.metrics.record_downscaled(u64::from(l.downscaled));
            Some((l.batch, l.tokens))
        }
    };
    let messages = extracted.as_ref().map_or(&req.messages, |e| &e.messages);
    let mut prompt_ids = render_chat_ids(&state, messages, &req, &kwargs)?;
    if let Some((_, tokens)) = &images {
        // The reference server's prompt update: each `<|image_pad|>` becomes the image's token run.
        prompt_ids = saber_vision_input::expand_image_pads(&prompt_ids, tokens)
            .map_err(|e| refuse(e.into()))?;
    }
    let params = sampling_from_chat(&req, &state)?;
    params.check_context(prompt_ids.len(), state.limits)?;

    // One submission, whatever `n` is. The request is a **sequence group**: the engine
    // prefills the prompt once and branches it, and each `StepEvent` says which choice it
    // belongs to. Submitting `n` requests here would prefill `n` times and give the engine
    // no way to know they were the same prompt.
    let engine_request = EngineRequest {
        prompt_token_ids: prompt_ids.clone(),
        params: params.clone(),
    };
    let submission = match images {
        None => state.engine.submit(engine_request)?,
        Some((batch, _)) => {
            let (count, patches) = (batch.grids.len() as u64, batch.patches());
            let sub = state.engine.submit_with_images(engine_request, batch)?;
            state.metrics.record_images(count, patches);
            sub
        }
    };
    state.metrics.record_prompt_tokens(prompt_ids.len() as u64);

    let id = ids::chat_completion_id();
    let created = now_secs();
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    // D5: the reference server's `return_token_ids` on chat -- the prompt's ids travel with the flag.
    let ids_back = req
        .return_token_ids
        .unwrap_or(false)
        .then(|| prompt_ids.clone());

    if req.stream {
        Ok(chat_stream(
            state,
            submission,
            params,
            prompt_ids.len(),
            spec,
            skip_special_tokens,
            id,
            created,
            model,
            include_usage,
            ids_back,
        ))
    } else {
        chat_blocking(
            state,
            submission,
            params,
            prompt_ids.len(),
            spec,
            skip_special_tokens,
            id,
            created,
            model,
            ids_back,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn chat_blocking(
    state: Arc<ServerState>,
    submission: Submission,
    params: SamplingParams,
    prompt_tokens: usize,
    spec: ParseSpec,
    skip_special_tokens: bool,
    id: String,
    created: u64,
    model: String,
    ids_back: Option<Vec<TokenId>>,
) -> Result<Response, ApiError> {
    let n = params.n.max(1);
    // One stream, `n` choices interleaved on it. Each choice is run to its own `finished`
    // and only then rendered, which is why the runners are collected first rather than
    // rendered in arrival order.
    let mut runners: Vec<ChoiceRunner> = (0..n)
        .map(|_| {
            ChoiceRunner::new(
                &state,
                &params,
                prompt_tokens,
                skip_special_tokens,
                LogprobShape::Chat,
            )
        })
        .collect();
    let mut logprob_entries: Vec<Vec<LogprobEntry>> = vec![Vec::new(); n];
    let mut prompt_logprobs = None;

    let (_guard, done) = blocking_guard(&state, submission.id);
    let mut events = submission.events;
    let mut live = n;
    while live > 0 {
        let Some(event) = events.next().await else {
            break;
        };
        let index = event.choice_index as usize;
        if index >= n {
            tracing::error!(
                choice = event.choice_index,
                n,
                "engine event for a choice this request does not have"
            );
            continue;
        }
        let step = runners[index].on_step(&event);
        logprob_entries[index].extend(step.logprobs);
        if step.finished.is_some() {
            live -= 1;
        }
    }

    done.store(true, Ordering::Release); // finished: the guard's drop is ordinary teardown

    // THE ENGINE ENDED A CHOICE WITH A REASON (ws-k18): the request is an error, not a reply with
    // an `abort` finish the client would have to guess at. prod answers an engine exception on a
    // non-streaming request with a 500 and the exception's message; so does this.
    if let Some(error) = runners.iter().find_map(|r| r.error.clone()) {
        state.metrics.record_finish(FinishReason::Abort);
        // FN126: a tower-arena refusal is counted as an image refusal (`arena`)
        state.metrics.record_engine_error(&error);
        return Err(ApiError::engine_ended(error));
    }

    let mut choices = Vec::with_capacity(n);
    let mut completion_tokens = 0u64;
    for (index, runner) in runners.iter().enumerate() {
        let finish = runner.finished.unwrap_or(FinishReason::Abort);
        completion_tokens += runner.completion_tokens;
        if prompt_logprobs.is_none() {
            prompt_logprobs = runner.prompt_logprobs.clone();
        }

        let (reasoning, content, tool_calls) = match spec.parser() {
            Some(mut parser) => parser.parse(&runner.text),
            None => (None, Some(runner.text.clone()), Vec::new()),
        };
        let finish_reason = if tool_calls.is_empty() {
            finish
        } else {
            FinishReason::ToolCalls
        };
        state.metrics.record_finish(finish_reason);

        choices.push(ChatChoice {
            index: index as u32,
            message: ResponseMessage {
                content,
                reasoning,
                tool_calls: (!tool_calls.is_empty()).then(|| tool_calls_to_wire(&tool_calls)),
                ..Default::default()
            },
            logprobs: (!logprob_entries[index].is_empty()).then(|| ChatLogprobs {
                content: logprob_entries[index].clone(),
            }),
            finish_reason: Some(finish_reason.as_str().to_owned()),
            stop_reason: runner.stop_reason.clone().map(Value::String),
            token_ids: ids_back.as_ref().map(|_| runner.token_ids.clone()),
        });
    }
    state.metrics.record_generation_tokens(completion_tokens);

    Ok(Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices,
        service_tier: None,
        system_fingerprint: state.system_fingerprint.clone(),
        usage: Usage::new(prompt_tokens as u64, completion_tokens),
        prompt_logprobs,
        prompt_token_ids: ids_back,
    })
    .into_response())
}

#[allow(clippy::too_many_arguments)]
fn chat_stream(
    state: Arc<ServerState>,
    submission: Submission,
    params: SamplingParams,
    prompt_tokens: usize,
    spec: ParseSpec,
    skip_special_tokens: bool,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    ids_back: Option<Vec<TokenId>>,
) -> Response {
    // Prod's per-chunk `token_ids`: SET on every content and finish chunk (null when not asked).
    let want_ids = ids_back.is_some();
    let chunk_ids = move |ids: &[TokenId]| Some(want_ids.then(|| ids.to_vec()));
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let n = params.n.max(1);
    let request_ids = vec![submission.id];
    // The two disconnect routes share one claim, so the request is aborted once and counted once.
    let once = AbortOnce::new();
    let done = Arc::new(AtomicBool::new(false));
    let guard = DisconnectGuard {
        state: Arc::clone(&state),
        request_ids: request_ids.clone(),
        done: Arc::clone(&done),
        once: Arc::clone(&once),
    };
    let generator_once = Arc::clone(&once);

    tokio::spawn(async move {
        let mut runners: Vec<ChoiceRunner> = (0..n)
            .map(|_| {
                ChoiceRunner::new(
                    &state,
                    &params,
                    prompt_tokens,
                    skip_special_tokens,
                    LogprobShape::Chat,
                )
            })
            .collect();
        let mut parsers: Vec<Option<ServeParser>> = (0..n).map(|_| spec.parser()).collect();
        let mut saw_tool_call = vec![false; n];

        // The opening chunk carries the role and an empty content, as prod does. Its
        // empty content is why a BetterBench-style client does not count it.
        let opening = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: (0..n)
                .map(|index| ChunkChoice {
                    index: index as u32,
                    delta: Delta {
                        role: Some("assistant".to_owned()),
                        content: Some(String::new()),
                        reasoning: None,
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                    stop_reason: None,
                    token_ids: None,
                })
                .collect(),
            usage: None,
            system_fingerprint: None,
            // "return prompt_token_ids at the first chunk ever" -- set there, and only there.
            prompt_token_ids: Some(ids_back),
        };
        if send_chunk(&tx, &opening).await.is_err() {
            abort_on_write_failure(&state, &request_ids, &generator_once);
            return;
        }

        // One stream carrying every choice. `choice_index` is what routes a delta to
        // `choices[i]`; an out-of-range one is the engine breaking its contract, and is
        // dropped loudly rather than written into someone else's choice.
        let mut events = submission.events;

        while let Some(event) = events.next().await {
            let index = event.choice_index as usize;
            if index >= n {
                tracing::error!(
                    choice = event.choice_index,
                    n,
                    "engine event for a choice this request does not have"
                );
                continue;
            }
            let step = runners[index].on_step(&event);
            let delta = match parsers[index].as_mut() {
                Some(parser) => parser
                    .parse_delta(&step.text, step.finished.is_some())
                    .map(|d| {
                        saw_tool_call[index] |= !d.tool_calls.is_empty();
                        delta_to_wire(&d)
                    })
                    .unwrap_or_default(),
                None => Delta {
                    content: (!step.text.is_empty()).then_some(step.text.clone()),
                    ..Default::default()
                },
            };
            let finish_reason = step.finished.map(|reason| {
                let reason = if saw_tool_call[index] {
                    FinishReason::ToolCalls
                } else {
                    reason
                };
                state.metrics.record_finish(reason);
                reason.as_str().to_owned()
            });
            // One chunk per StepEvent — the invariant in this module's docs.
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: index as u32,
                    delta,
                    logprobs: (!step.logprobs.is_empty()).then_some(ChatLogprobs {
                        content: step.logprobs,
                    }),
                    finish_reason,
                    stop_reason: runners[index].stop_reason.clone().map(Value::String),
                    token_ids: chunk_ids(&step.ids),
                }],
                usage: None,
                system_fingerprint: None,
                prompt_token_ids: None,
            };
            if send_chunk(&tx, &chunk).await.is_err() {
                abort_all(&state, &request_ids);
                return;
            }
            // AND THE ENGINE'S REASON, when it ended the choice with one (ws-k18): prod's stream
            // carries an engine exception as one `data: {"error": ...}` event before `[DONE]`,
            // and a client reading only `finish_reason: abort` could not tell a failed restore
            // from its own disconnect.
            if let Some(error) = event.error.as_ref() {
                state.metrics.record_engine_error(error);
                let body = crate::protocol::ApiErrorEnvelope {
                    error: ApiError::engine_ended(error.clone()).body,
                };
                if let Ok(json) = serde_json::to_string(&body) {
                    if tx.send(Event::default().data(json)).await.is_err() {
                        abort_all(&state, &request_ids);
                        return;
                    }
                }
            }
        }

        // The engine contract says exactly one event carries `finished`. If a stream
        // ends without one, say so on the wire and in `request_success_total` rather
        // than leaving the client with a null `finish_reason`.
        for (index, runner) in runners.iter().enumerate() {
            if runner.finished.is_some() {
                continue;
            }
            tracing::warn!(
                choice = index,
                "engine stream ended without a finished event; reporting abort"
            );
            state.metrics.record_finish(FinishReason::Abort);
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: index as u32,
                    delta: Delta::default(),
                    logprobs: None,
                    finish_reason: Some(FinishReason::Abort.as_str().to_owned()),
                    stop_reason: None,
                    token_ids: chunk_ids(&[]),
                }],
                usage: None,
                system_fingerprint: None,
                prompt_token_ids: None,
            };
            if send_chunk(&tx, &chunk).await.is_err() {
                abort_on_write_failure(&state, &request_ids, &generator_once);
                return;
            }
        }

        let completion_tokens: u64 = runners.iter().map(|r| r.completion_tokens).sum();
        state.metrics.record_generation_tokens(completion_tokens);
        if include_usage {
            let usage_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: Vec::new(),
                usage: Some(StreamUsage::from(Usage::new(
                    prompt_tokens as u64,
                    completion_tokens,
                ))),
                system_fingerprint: Some(state.system_fingerprint.clone()),
                prompt_token_ids: None,
            };
            if send_chunk(&tx, &usage_chunk).await.is_err() {
                abort_on_write_failure(&state, &request_ids, &generator_once);
                return;
            }
        }
        let _ = tx.send(Event::default().data("[DONE]")).await;
        // Everything this request had to say has been said: the body's drop from here on is the
        // ordinary teardown, not a disconnect.
        done.store(true, Ordering::Release);
    });

    let stream = GuardedStream {
        inner: tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok::<Event, Infallible>),
        _guard: guard,
    };
    Sse::new(stream).into_response()
}

/// The fallback route: a chunk could not be pushed, so the client is gone. Counted only if the
/// body's drop did not already claim it.
fn abort_on_write_failure(
    state: &Arc<ServerState>,
    ids: &[crate::engine::RequestId],
    once: &AbortOnce,
) {
    if once.claim() {
        abort_all(state, ids);
        state.metrics.record_abort_on_write_failure();
    }
}

async fn send_chunk<T: serde::Serialize>(
    tx: &tokio::sync::mpsc::Sender<Event>,
    chunk: &T,
) -> Result<(), ()> {
    let json = serde_json::to_string(chunk).map_err(|_| ())?;
    tx.send(Event::default().data(json)).await.map_err(|_| ())
}

fn abort_all(state: &ServerState, ids: &[crate::engine::RequestId]) {
    for id in ids {
        state.engine.abort(*id);
    }
}

/// Which of the two ways a streaming request can end early got there first.
///
/// A client that hangs up is discovered twice over: the response body is dropped (immediately,
/// when hyper closes the connection) and the next chunk fails to send (later, when the engine has
/// produced one). Both must abort the request, and exactly one may count it — otherwise the
/// which-path counters say two things happened when one did.
#[derive(Debug)]
struct AbortOnce {
    fired: AtomicBool,
}

impl AbortOnce {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            fired: AtomicBool::new(false),
        })
    }

    /// `true` for the first caller only.
    fn claim(&self) -> bool {
        !self.fired.swap(true, Ordering::AcqRel)
    }
}

/// Aborts the engine's requests when the response body is dropped.
///
/// The body is dropped in exactly two situations: the stream ended normally (`[DONE]` was sent and
/// the generator set `done`), or **the client went away**. The second is the one this exists for:
/// without it, a dead client holds its KV blocks and its GDN state slot set until the engine
/// happens to produce another chunk and the push fails — a whole step, and at a 22 ms step with a
/// slot set worth 432 MiB that is a real cost, paid per abandoned request.
struct DisconnectGuard {
    state: Arc<ServerState>,
    request_ids: Vec<crate::engine::RequestId>,
    done: Arc<AtomicBool>,
    once: Arc<AbortOnce>,
}

impl Drop for DisconnectGuard {
    fn drop(&mut self) {
        if self.done.load(Ordering::Acquire) {
            return; // the stream finished; this is the ordinary teardown
        }
        if self.once.claim() {
            tracing::debug!(
                requests = self.request_ids.len(),
                "client disconnected mid-stream; aborting"
            );
            abort_all(&self.state, &self.request_ids);
            self.state.metrics.record_abort_on_disconnect();
        }
    }
}

/// Non-streaming requests have no response body to drop, so the guard rides the handler future:
/// hyper drops that future when the client disconnects mid-generation, and this aborts the
/// engine request (otherwise it generates to its limit with nobody listening).
fn blocking_guard(
    state: &Arc<ServerState>,
    id: crate::engine::RequestId,
) -> (DisconnectGuard, Arc<AtomicBool>) {
    let done = Arc::new(AtomicBool::new(false));
    let guard = DisconnectGuard {
        state: Arc::clone(state),
        request_ids: vec![id],
        done: Arc::clone(&done),
        once: AbortOnce::new(),
    };
    (guard, done)
}

/// A stream that owns a [`DisconnectGuard`], so dropping the body runs it.
struct GuardedStream<S> {
    inner: S,
    _guard: DisconnectGuard,
}

impl<S: futures_util::Stream + Unpin> futures_util::Stream for GuardedStream<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

// ── /v1/completions ─────────────────────────────────────────────────────────────

async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let model = state.resolve_model(req.model.as_deref())?;
    let (prompt_text, prompt_ids) = match &req.prompt {
        Value::String(text) => (text.clone(), state.codec.encode(text)?),
        Value::Array(items) if items.iter().all(Value::is_number) => {
            let ids: Vec<TokenId> = items
                .iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as TokenId)
                .collect();
            (state.codec.decode(&ids, false), ids)
        }
        Value::Null => return Err(ApiError::bad_request("prompt is required.", Some("prompt"))),
        _ => {
            return Err(ApiError::bad_request(
                "only a single prompt (string or token-id array) is supported.",
                Some("prompt"),
            ))
        }
    };
    let params = sampling_from_completion(&req, &state)?;
    params.check_context(prompt_ids.len(), state.limits)?;
    if req.stream {
        // FN25: streamed completions (the reference server's `completion_stream_generator`). FN17's conc-8 client
        // (`fn17_api.py`: prompt ids, `stream_options.include_usage`, `return_token_ids`) got this
        // path's old 400 ("streaming is not implemented") on every boot.
        let submission = state.engine.submit(EngineRequest {
            prompt_token_ids: prompt_ids.clone(),
            params: params.clone(),
        })?;
        state.metrics.record_prompt_tokens(prompt_ids.len() as u64);
        let opts = CompletionStreamOpts {
            echo: req.echo.unwrap_or(false),
            want_ids: req.return_token_ids.unwrap_or(false),
            include_usage: req.stream_options.as_ref().is_some_and(|o| o.include_usage),
            logprobs: req.logprobs.is_some(),
        };
        return Ok(completion_stream(
            state,
            submission,
            params,
            prompt_ids,
            prompt_text,
            opts,
            ids::completion_id(),
            now_secs(),
            model,
        ));
    }

    // ONE submission, `n` choices demultiplexed by `choice_index` — the scheduler's sequence group
    // (V4). This loop used to submit the request `n` times, each carrying `n`, and to run each
    // submission's WHOLE group through one runner: every choice was `n` branches interleaved,
    // stopped at whichever branch finished first (ws-f38 found it on the card: a group's choices
    // never matched the single request, and ran to `length` past the single request's EOS).
    let n = params.n.max(1);
    let submission = state.engine.submit(EngineRequest {
        prompt_token_ids: prompt_ids.clone(),
        params: params.clone(),
    })?;
    // `/v1/completions` runs no parser; its detokenizer keeps the special tokens, as before V9.
    let mut runners: Vec<ChoiceRunner> = (0..n)
        .map(|_| {
            ChoiceRunner::new(
                &state,
                &params,
                prompt_ids.len(),
                false,
                LogprobShape::Completion,
            )
        })
        .collect();
    let mut all_logprobs: Vec<CompletionLogprobs> =
        (0..n).map(|_| CompletionLogprobs::default()).collect();
    let (_guard, done) = blocking_guard(&state, submission.id);
    let mut events = submission.events;
    let mut live = n;
    while live > 0 {
        let Some(event) = events.next().await else {
            break;
        };
        let index = event.choice_index as usize;
        if index >= n {
            tracing::error!(
                choice = event.choice_index,
                n,
                "engine event for a choice this request does not have"
            );
            continue;
        }
        let runner = &mut runners[index];
        let logprobs = &mut all_logprobs[index];
        let step = runner.on_step(&event);
        for entry in step.logprobs {
            logprobs.text_offset.push(runner.text.len());
            logprobs.token_logprobs.push(entry.logprob);
            logprobs.tokens.push(entry.token.clone());
            let mut top = Map::new();
            for candidate in entry.top_logprobs.into_iter().flatten() {
                top.insert(candidate.token, Value::from(f64::from(candidate.logprob)));
            }
            logprobs.top_logprobs.push(top);
        }
        if step.finished.is_some() {
            live -= 1;
        }
    }

    done.store(true, Ordering::Release);

    let mut choices = Vec::with_capacity(n);
    let mut completion_tokens = 0u64;
    for (index, (runner, logprobs)) in runners.into_iter().zip(all_logprobs).enumerate() {
        let finish = runner.finished.unwrap_or(FinishReason::Abort);
        state.metrics.record_finish(finish);
        completion_tokens += runner.completion_tokens;
        let text = if req.echo.unwrap_or(false) {
            format!("{prompt_text}{}", runner.text)
        } else {
            runner.text.clone()
        };
        choices.push(CompletionChoice {
            index: index as u32,
            text,
            logprobs: (!logprobs.tokens.is_empty()).then_some(logprobs),
            finish_reason: Some(finish.as_str().to_owned()),
            stop_reason: runner.stop_reason.clone().map(Value::String),
            prompt_logprobs: runner.prompt_logprobs.clone(),
            token_ids: req
                .return_token_ids
                .unwrap_or(false)
                .then(|| runner.token_ids.clone()),
        });
    }
    state.metrics.record_prompt_tokens(prompt_ids.len() as u64);
    state.metrics.record_generation_tokens(completion_tokens);

    Ok(Json(CompletionResponse {
        id: ids::completion_id(),
        object: "text_completion",
        created: now_secs(),
        model,
        choices,
        system_fingerprint: state.system_fingerprint.clone(),
        usage: Usage::new(prompt_ids.len() as u64, completion_tokens),
    })
    .into_response())
}

/// What a streamed completion was asked for, beyond its sampling (FN25).
#[derive(Debug, Clone, Copy)]
struct CompletionStreamOpts {
    echo: bool,
    want_ids: bool,
    include_usage: bool,
    logprobs: bool,
}

/// `/v1/completions` with `stream: true` (FN25): the reference server's `completion_stream_generator`, one chunk
/// per engine step per choice, on the machinery the chat stream uses (the disconnect guard, the
/// abort-once claim, the engine's error event, the missing-finish abort chunk).
///
/// Per choice, as the reference server: the first chunk of an `echo` request carries the prompt's text (none when
/// `return_token_ids`, whose ids say it) and `prompt_token_ids`; otherwise `return_token_ids` puts
/// `prompt_token_ids` on the choice's first chunk; a chunk with no text and no ids before any token
/// has been produced is not sent; `token_ids` are the step's ids under `return_token_ids`; the
/// logprob `text_offset`s continue from the text already streamed (one per token, advancing by the
/// previous token's length in code points). The terminal content chunk carries the fingerprint
/// when no usage chunk follows; `stream_options.include_usage` adds the usage chunk
/// (`choices: []`). As on the blocking path, an `echo` streams no prompt logprobs.
#[allow(clippy::too_many_arguments)]
fn completion_stream(
    state: Arc<ServerState>,
    submission: Submission,
    params: SamplingParams,
    prompt_ids: Vec<TokenId>,
    prompt_text: String,
    opts: CompletionStreamOpts,
    id: String,
    created: u64,
    model: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let n = params.n.max(1);
    let request_ids = vec![submission.id];
    let once = AbortOnce::new();
    let done = Arc::new(AtomicBool::new(false));
    let guard = DisconnectGuard {
        state: Arc::clone(&state),
        request_ids: request_ids.clone(),
        done: Arc::clone(&done),
        once: Arc::clone(&once),
    };
    let generator_once = Arc::clone(&once);

    tokio::spawn(async move {
        let prompt_tokens = prompt_ids.len();
        let mut runners: Vec<ChoiceRunner> = (0..n)
            .map(|_| {
                ChoiceRunner::new(
                    &state,
                    &params,
                    prompt_tokens,
                    false,
                    LogprobShape::Completion,
                )
            })
            .collect();
        let mut has_echoed = vec![false; n];
        let mut text_chars = vec![0usize; n];
        let mut produced = vec![0usize; n];
        let chunk_of =
            |choice: CompletionStreamChoice, fingerprint: Option<String>| CompletionStreamChunk {
                id: id.clone(),
                object: "text_completion",
                created,
                model: model.clone(),
                choices: vec![choice],
                usage: None,
                system_fingerprint: fingerprint,
            };
        let mut events = submission.events;
        while let Some(event) = events.next().await {
            let index = event.choice_index as usize;
            if index >= n {
                tracing::error!(
                    choice = event.choice_index,
                    n,
                    "engine event for a choice this request does not have"
                );
                continue;
            }
            let step = runners[index].on_step(&event);
            let mut prompt_back = None;
            let text = if opts.echo && !has_echoed[index] {
                has_echoed[index] = true;
                prompt_back = Some(prompt_ids.clone());
                let head = if opts.want_ids {
                    ""
                } else {
                    prompt_text.as_str()
                };
                format!("{head}{}", step.text)
            } else {
                if !has_echoed[index] && opts.want_ids {
                    prompt_back = Some(prompt_ids.clone());
                    has_echoed[index] = true;
                }
                if step.text.is_empty() && step.ids.is_empty() && produced[index] == 0 {
                    continue; // the reference server: a chunked prefill's empty output is not a chunk
                }
                step.text.clone()
            };
            let logprobs = opts.logprobs.then(|| {
                let mut lp = CompletionLogprobs::default();
                let mut offset = text_chars[index];
                for (k, entry) in step.logprobs.iter().enumerate() {
                    if k > 0 {
                        offset += step.logprobs[k - 1].token.chars().count();
                    }
                    lp.text_offset.push(offset);
                    lp.token_logprobs.push(entry.logprob);
                    lp.tokens.push(entry.token.clone());
                    let mut top = Map::new();
                    for candidate in entry.top_logprobs.iter().flatten() {
                        top.insert(
                            candidate.token.clone(),
                            Value::from(f64::from(candidate.logprob)),
                        );
                    }
                    lp.top_logprobs.push(top);
                }
                lp
            });
            text_chars[index] += step.text.chars().count();
            produced[index] += step.ids.len();
            let finish_reason = step.finished.map(|reason| {
                state.metrics.record_finish(reason);
                reason.as_str().to_owned()
            });
            let fingerprint = (!opts.include_usage && finish_reason.is_some())
                .then(|| state.system_fingerprint.clone());
            let chunk = chunk_of(
                CompletionStreamChoice {
                    index: index as u32,
                    text,
                    logprobs,
                    finish_reason,
                    stop_reason: runners[index].stop_reason.clone().map(Value::String),
                    prompt_token_ids: prompt_back,
                    token_ids: opts.want_ids.then(|| step.ids.clone()),
                },
                fingerprint,
            );
            if send_chunk(&tx, &chunk).await.is_err() {
                abort_all(&state, &request_ids);
                return;
            }
            if let Some(error) = event.error.as_ref() {
                state.metrics.record_engine_error(error);
                let body = crate::protocol::ApiErrorEnvelope {
                    error: ApiError::engine_ended(error.clone()).body,
                };
                if let Ok(json) = serde_json::to_string(&body) {
                    if tx.send(Event::default().data(json)).await.is_err() {
                        abort_all(&state, &request_ids);
                        return;
                    }
                }
            }
        }

        for (index, runner) in runners.iter().enumerate() {
            if runner.finished.is_some() {
                continue;
            }
            tracing::warn!(
                choice = index,
                "engine stream ended without a finished event; reporting abort"
            );
            state.metrics.record_finish(FinishReason::Abort);
            let chunk = chunk_of(
                CompletionStreamChoice {
                    index: index as u32,
                    text: String::new(),
                    logprobs: None,
                    finish_reason: Some(FinishReason::Abort.as_str().to_owned()),
                    stop_reason: None,
                    prompt_token_ids: None,
                    token_ids: opts.want_ids.then(Vec::new),
                },
                (!opts.include_usage).then(|| state.system_fingerprint.clone()),
            );
            if send_chunk(&tx, &chunk).await.is_err() {
                abort_on_write_failure(&state, &request_ids, &generator_once);
                return;
            }
        }

        let completion_tokens: u64 = runners.iter().map(|r| r.completion_tokens).sum();
        state.metrics.record_generation_tokens(completion_tokens);
        if opts.include_usage {
            let usage_chunk = CompletionStreamChunk {
                id: id.clone(),
                object: "text_completion",
                created,
                model: model.clone(),
                choices: Vec::new(),
                usage: Some(StreamUsage::from(Usage::new(
                    prompt_tokens as u64,
                    completion_tokens,
                ))),
                system_fingerprint: Some(state.system_fingerprint.clone()),
            };
            if send_chunk(&tx, &usage_chunk).await.is_err() {
                abort_on_write_failure(&state, &request_ids, &generator_once);
                return;
            }
        }
        let _ = tx.send(Event::default().data("[DONE]")).await;
        done.store(true, Ordering::Release);
    });

    let stream = GuardedStream {
        inner: tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok::<Event, Infallible>),
        _guard: guard,
    };
    Sse::new(stream).into_response()
}
