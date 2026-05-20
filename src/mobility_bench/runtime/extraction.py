"""Extraction helpers for model responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)
COMMAND_RE = re.compile(r'"command"\s*:\s*"(?P<command>(?:[^"\\]|\\.)*)"', re.DOTALL)
CMD_RE = re.compile(r'"cmd"\s*:\s*"(?P<command>(?:[^"\\]|\\.)*)"', re.DOTALL)
TOOL_RE = re.compile(r'"tool"\s*:\s*"(?P<tool>[^"]+)"', re.DOTALL)
STRING_FIELD_RE = re.compile(
    r'"(?P<key>path|file|content|old|new|command|cmd|message)"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
BOOL_FIELD_RE = re.compile(r'"(?P<key>replace_all)"\s*:\s*(?P<value>true|false)', re.DOTALL)
TOP_LEVEL_ARG_KEYS = {"path", "file", "content", "old", "new", "replace_all", "command", "cmd", "message"}
RECOVERABLE_TOOLS = {"read", "write", "edit", "bash", "submit"}
SOLUTION_FILE_RE = re.compile(
    r"<solution_file\s+path=[\"'](?P<path>[^\"']+)[\"']\s*>\s*(?P<content>.*?)\s*</solution_file>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedToolCall:
    tool: str
    args: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class SolutionFile:
    path: str
    content: str


def _parse_tool_call_body(body: str) -> dict[str, Any]:
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        command_match = COMMAND_RE.search(body) or CMD_RE.search(body)
        if command_match:
            return {
                "tool": "bash",
                "args": {"command": json.loads(f'"{command_match.group("command")}"')},
                "_repair": f"recovered bash command from malformed JSON: {exc}",
            }
        recovered = _recover_malformed_tool_call(body, exc)
        if recovered is not None:
            return recovered
        return {
            "tool": "__invalid__",
            "args": {"error": f"invalid tool-call JSON: {exc}", "body": body},
            "_repair": "unrecoverable malformed JSON",
        }
    if not isinstance(raw, dict):
        return {
            "tool": "__invalid__",
            "args": {"error": "tool call JSON must be an object", "body": body},
        }
    return raw


def _recover_malformed_tool_call(body: str, exc: json.JSONDecodeError) -> dict[str, Any] | None:
    tool_match = TOOL_RE.search(body)
    if tool_match is None:
        return None
    tool = tool_match.group("tool").strip().lstrip(":").strip()
    if tool not in RECOVERABLE_TOOLS:
        return None

    args: dict[str, Any] = {}
    for match in STRING_FIELD_RE.finditer(body):
        args[match.group("key")] = json.loads(f'"{match.group("value")}"')
    for match in BOOL_FIELD_RE.finditer(body):
        args[match.group("key")] = match.group("value") == "true"

    required = {
        "read": {"path"},
        "write": {"path", "content"},
        "edit": {"path", "old", "new"},
        "bash": {"command"},
        "submit": set(),
    }
    if not required[tool].issubset(args):
        return None
    return {
        "tool": tool,
        "args": args,
        "_repair": f"recovered {tool} call from malformed JSON: {exc}",
    }


def extract_tool_calls(text: str) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for match in TOOL_CALL_RE.finditer(text):
        raw = _parse_tool_call_body(match.group("body"))
        tool = raw.get("tool")
        args = raw.get("args")
        if args is None:
            args = {key: value for key, value in raw.items() if key in TOP_LEVEL_ARG_KEYS}
        if not isinstance(tool, str):
            tool = "__invalid__"
            args = {"error": "tool call is missing string field 'tool'", "body": match.group("body")}
        if not isinstance(args, dict):
            tool = "__invalid__"
            args = {"error": "tool call field 'args' must be an object", "body": match.group("body")}
        calls.append(ParsedToolCall(tool=tool, args=args, raw=raw))
    return calls


def extract_solution_files(text: str) -> list[SolutionFile]:
    return [
        SolutionFile(path=match.group("path"), content=match.group("content"))
        for match in SOLUTION_FILE_RE.finditer(text)
    ]
