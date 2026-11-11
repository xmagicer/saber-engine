//! What a SERVER does with the two parsers, per flag profile (workstream V9).
//!
//! The parsers in this module's parent are the classes; which of them a request meets is decided
//! one level up, by the server's flags. The reference server 0.27.1 composes them in
//! `upstream.parser.ParserManager.get_parser(tool_parser_name, reasoning_parser_name,
//! enable_auto_tools)`: the tool parser exists only with `--enable-auto-tool-choice` (a named
//! `--tool-call-parser` without it is never even looked up), the reasoning parser only with
//! `--reasoning-parser`, and a `DelegatingParser` runs whichever of the two there is. Before any of
//! that, `OnlineRenderer.render_chat_request` refuses a `tool_choice` the flags cannot serve.
//!
//! [`ServeParser`] is that composition, and the serve path calls nothing else. The four profiles:
//!
//! | profile | reasoning | tool parsing | non-streaming | streaming |
//! |---|---|---|---|---|
//! | prod (both) | `qwen3` | on | `extract_reasoning`, then `extract_tool_calls` on its content | the reasoning adapter until `</think>`, then the tool adapter on what follows |
//! | tool only | — | on | `extract_tool_calls` from CONTENT | the tool adapter from the first token |
//! | reasoning only | `qwen3` | off | `extract_reasoning` | the reasoning adapter, then the raw text |
//! | none | — | off | the raw text | the raw text |
//!
//! Before V9 the serve path ran the engine's SINGLE pass (`Qwen3Parser::parse` / `parse_delta`,
//! views A/B of `serve_corpus.py`) for prod's profile. Prod does not: it composes the two adapters,
//! and over this corpus that differs on 432 no-`tools` replies carrying tool markup (prod strips the
//! markup from `content`; the single pass left it in) and on 12 streamed replies (text after a tool
//! call, and where a malformed call's name and arguments are cut). The composition is what is
//! served now, for every profile.
//!
//! Gated case for case against prod's own composition by `tests/parser_config_corpus.rs` over the
//! corpus `the internal reference harness` records inside the prod image.

use serde_json::Value;

use super::{DeltaMessage, ParserOptions, Qwen3Parser, ToolCall, ToolChoice};
use crate::state::{ReasoningParserKind, ToolParserKind};

/// The server's parser flags, resolved (see `crate::state::ParserFlags`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParserProfile {
    pub reasoning: Option<ReasoningParserKind>,
    /// The tool parser the server CONSTRUCTS: `None` unless `--enable-auto-tool-choice`, as in
    /// the reference server, whatever `--tool-call-parser` says.
    pub tool: Option<ToolParserKind>,
}

impl ParserProfile {
    /// Prod's unit: `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`.
    pub const PROD: Self = Self {
        reasoning: Some(ReasoningParserKind::Qwen3),
        tool: Some(ToolParserKind::Qwen3Coder),
    };

    /// No parser at all — the reference server's own defaults.
    pub const NONE: Self = Self {
        reasoning: None,
        tool: None,
    };

    /// Whether tool calls are parsed at all (the reference server's `tool_parser_cls is not None`).
    pub fn tool_parsing(&self) -> bool {
        self.tool.is_some()
    }

    /// Whether any parser runs — the reference server's `parser_cls is not None`. Without one the detokenizer
    /// keeps the request's `skip_special_tokens` (default `true`); with one, the engine adapters'
    /// `adjust_request` sets it `false`.
    pub fn any(&self) -> bool {
        self.reasoning.is_some() || self.tool.is_some()
    }
}

/// The part of a request the parsers read.
#[derive(Debug, Clone, Copy)]
pub struct RequestShape<'a> {
    /// The request's `tools`, if it sent any.
    pub tools: Option<&'a [Value]>,
    /// The request's `tool_choice`, as sent (absent = `None`).
    pub tool_choice: Option<&'a Value>,
    /// `chat_template_kwargs.enable_thinking`, or the server's default when the request is silent.
    pub enable_thinking: bool,
    pub drop_tokens: &'a [String],
}

impl RequestShape<'_> {
    fn has_tools(&self) -> bool {
        self.tools.is_some_and(|t| !t.is_empty())
    }

    /// The protocol's own default: `tool_choice` absent with tools present is `"auto"`.
    fn effective_choice(&self) -> Option<&Value> {
        match self.tool_choice {
            Some(Value::Null) | None => None,
            Some(v) => Some(v),
        }
    }

    fn choice_is_none(&self) -> bool {
        self.effective_choice().and_then(Value::as_str) == Some("none")
    }
}

/// `OnlineRenderer.render_chat_request`'s refusal of a `tool_choice` the flags cannot serve, with
/// The reference server's text (a 400 `BadRequestError`, no `param`).
///
/// Only when NO tool parser was constructed. `tool_choice` absent with `tools` present is the
/// protocol's `"auto"`, so a server without `--enable-auto-tool-choice` refuses every request that
/// sends tools and does not say `"none"`.
pub fn tool_choice_refusal(profile: &ParserProfile, shape: &RequestShape<'_>) -> Option<String> {
    if profile.tool_parsing() {
        return None;
    }
    let choice = match shape.effective_choice() {
        None if shape.has_tools() => Value::String("auto".to_owned()),
        None => return None,
        Some(v) => v.clone(),
    };
    match &choice {
        Value::String(s) if s == "none" => None,
        // `profile.tool` is `None` exactly when `--enable-auto-tool-choice` is off (the flags'
        // resolution guarantees a parser whenever it is on), so "auto" is always this refusal.
        Value::String(s) if s == "auto" => Some(
            "\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to \
             be set"
                .to_owned(),
        ),
        Value::Object(o) => {
            let name = o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            Some(format!(
                "tool_choice=function \"{name}\" requires --tool-call-parser to be set"
            ))
        }
        Value::String(s) => Some(format!(
            "tool_choice=\"{s}\" requires --tool-call-parser to be set"
        )),
        other => Some(format!(
            "tool_choice={other} requires --tool-call-parser to be set"
        )),
    }
}

/// One choice's parser, as the server composes it for a profile and a request — the reference server's
/// `DelegatingParser` over the reasoning adapter and the tool adapter, either of which may be
/// absent.
#[derive(Debug)]
pub struct ServeParser {
    /// `--reasoning-parser qwen3`: the engine with `skip_tool_parsing`, in the request's initial
    /// state (the reasoning adapter).
    reasoning: Option<Qwen3Parser>,
    /// The tool adapter: the engine started in CONTENT state, fed what the reasoning phase left.
    tool: Option<Qwen3Parser>,
    /// `StreamState.reasoning_ended`. True from the start when there is no reasoning parser or the
    /// generation prompt already closed the think block (`enable_thinking: false`), which is what
    /// `parse_delta`'s one prompt check decides.
    ended: bool,
    /// `StreamState.tool_call_text_started`.
    tool_started: bool,
    /// Tool calls never reach the client: `tool_choice: "none"`, or a request without `tools` (the
    /// adapter still parses the markup out of `content`; the calls are dropped).
    strip_calls: bool,
}

impl ServeParser {
    /// `None` when the profile has no parser: the reply is the raw text.
    pub fn new(profile: &ParserProfile, shape: &RequestShape<'_>) -> Option<Self> {
        if !profile.any() {
            return None;
        }
        let tools: Vec<Value> = shape.tools.map(<[Value]>::to_vec).unwrap_or_default();
        let choice = if shape.choice_is_none() {
            ToolChoice::None
        } else {
            ToolChoice::Auto
        };
        let reasoning = profile.reasoning.map(|_| {
            Qwen3Parser::new(ParserOptions {
                enable_thinking: shape.enable_thinking,
                tools: Vec::new(),
                tool_choice: ToolChoice::Auto,
                drop_tokens: shape.drop_tokens.to_vec(),
                skip_tool_parsing: true,
            })
        });
        let tool = profile.tool.map(|_| {
            Qwen3Parser::new(ParserOptions {
                enable_thinking: false,
                tools,
                tool_choice: choice,
                drop_tokens: shape.drop_tokens.to_vec(),
                skip_tool_parsing: false,
            })
        });
        Some(Self {
            ended: reasoning.is_none() || !shape.enable_thinking,
            reasoning,
            tool,
            tool_started: false,
            strip_calls: shape.choice_is_none() || !shape.has_tools(),
        })
    }

    /// The non-streaming reply: `(reasoning, content, tool_calls)` after the reference server's
    /// `DelegatingParser.parse` AND the message branch of `chat_completion_full_generator`.
    pub fn parse(&mut self, text: &str) -> (Option<String>, Option<String>, Vec<ToolCall>) {
        let (reasoning, content) = match self.reasoning.as_mut() {
            Some(r) => r.extract_reasoning(text),
            None => (None, Some(text.to_owned())),
        };
        let Some(tool) = self.tool.as_mut() else {
            return (reasoning, content, Vec::new());
        };
        let info = tool.extract_tool_calls(content.as_deref().unwrap_or_default());
        if !info.tools_called {
            return (reasoning, info.content, Vec::new());
        }
        let content = info.content.filter(|c| !c.trim().is_empty());
        let calls = if self.strip_calls {
            Vec::new()
        } else {
            info.tool_calls
        };
        (reasoning, content, calls)
    }

    /// One engine step's text, streaming (the reference server's `DelegatingParser.parse_delta`); `finished` on the
    /// last step.
    pub fn parse_delta(&mut self, delta_text: &str, finished: bool) -> Option<DeltaMessage> {
        let mut delta: Option<DeltaMessage> = None;
        let mut transitioned = false;
        // What the tool phase is fed if it starts in this step: the whole text so far when there
        // was no reasoning phase, the reasoning phase's content (plus its flush) when it just ended.
        let mut tool_text = delta_text.to_owned();

        if !self.ended {
            if let Some(r) = self.reasoning.as_mut() {
                delta = r.extract_reasoning_streaming(delta_text);
                if r.reasoning_ended() {
                    self.ended = true;
                    transitioned = true;
                    let flush = r.finish_streaming();
                    tool_text = delta
                        .as_ref()
                        .and_then(|d| d.content.clone())
                        .unwrap_or_default()
                        + &flush.and_then(|f| f.content).unwrap_or_default();
                    if self.tool.is_some() {
                        if let Some(d) = delta.as_mut() {
                            d.content = None;
                        }
                    }
                }
            }
        }

        if self.ended {
            if let Some(tool) = self.tool.as_mut() {
                let fed = if self.tool_started {
                    delta_text.to_owned()
                } else {
                    self.tool_started = true;
                    tool_text
                };
                let reasoning_from_batch = delta.as_ref().and_then(|d| d.reasoning.clone());
                let mut tool_delta = tool.extract_tool_calls_streaming(&fed);
                if self.strip_calls {
                    if let Some(d) = tool_delta.as_mut() {
                        d.tool_calls.clear();
                    }
                }
                if let Some(r) = reasoning_from_batch {
                    match tool_delta.as_mut() {
                        None => {
                            tool_delta = Some(DeltaMessage {
                                reasoning: Some(r),
                                ..DeltaMessage::default()
                            });
                        }
                        Some(d) if d.reasoning.is_none() => d.reasoning = Some(r),
                        Some(_) => {}
                    }
                }
                delta = tool_delta;
            } else if !transitioned {
                // No phase active: the raw text, as content.
                delta = Some(DeltaMessage {
                    content: Some(delta_text.to_owned()),
                    ..DeltaMessage::default()
                });
            }
        }

        if finished {
            if !self.ended {
                if let Some(r) = self.reasoning.as_mut() {
                    delta = merge(delta, r.finish_streaming());
                }
            }
            if let Some(tool) = self.tool.as_mut() {
                // `finalize_generation`'s `_append_unstreamed_tool_args`: the step's last tool
                // call gets `arguments or ""` (the engine adapters stream every argument, so the
                // remainder is empty) — which is how a name-only delta reaches the wire as `""`.
                if let Some(last) = delta.as_mut().and_then(|d| d.tool_calls.last_mut()) {
                    last.arguments = Some(last.arguments.take().unwrap_or_default());
                }
                // The flush is merged as it comes: prod strips a no-`tools` request's calls from
                // the steps' deltas but not from `_flush_engine_parsers`' (measured, V9 corpus).
                delta = merge(delta, tool.finish_streaming());
            }
        }
        delta
    }
}

/// `_flush_engine_parsers`' merge of a flush into the step's own delta.
fn merge(delta: Option<DeltaMessage>, flush: Option<DeltaMessage>) -> Option<DeltaMessage> {
    let Some(flush) = flush else {
        return delta;
    };
    let Some(mut delta) = delta else {
        return Some(flush);
    };
    if let Some(c) = flush.content {
        delta.content = Some(delta.content.unwrap_or_default() + &c);
    }
    if let Some(r) = flush.reasoning {
        delta.reasoning = Some(delta.reasoning.unwrap_or_default() + &r);
    }
    delta.tool_calls.extend(flush.tool_calls);
    Some(delta)
}
