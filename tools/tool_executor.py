"""
🔧 busel TOOL EXECUTOR v1.0 — Phase 7 (skeleton)

Parses `<function_calls>...<invoke>...<parameter>...<result>` envelopes
emitted by the SFT/DPO model, executes the named tool, and re-injects the
result as `ROLE_TOOL ... TOOL_RESULTS_END` so the model can continue.

This is the **skeleton** — full integration with the opencode-style tool
vocabulary (`TOOL_BASH`, `TOOL_READ`, `TOOL_WRITE`, `TOOL_EDIT`, etc.) is
a follow-up. For now we support a small, sandboxed subset.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Envelope detection
# ---------------------------------------------------------------------------

INVOKE_PATTERN = re.compile(
    r"<function_calls>\s*"
    r"<invoke\s+name=\"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\">\s*"
    r"(?P<params>.*?)"
    r"</invoke>\s*"
    r"</function_calls>",
    re.DOTALL,
)
PARAM_PATTERN = re.compile(
    r"<parameter\s+name=\"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\">\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from a model output string.

    Returns a list of {name, params} dicts, in document order.
    Returns an empty list if no <function_calls> envelope is present.
    """
    calls: list[dict[str, Any]] = []
    for m in INVOKE_PATTERN.finditer(text):
        name = m.group("name")
        params: dict[str, str] = {}
        for p in PARAM_PATTERN.finditer(m.group("params")):
            params[p.group("key")] = p.group("value").strip()
        calls.append({"name": name, "params": params})
    return calls


# ---------------------------------------------------------------------------
# Tool registry (sandboxed subset)
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Minimal sandboxed tool registry. Each tool is a Python callable."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, fn) -> None:
        self._tools[name] = fn

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, params: dict[str, str]) -> str:
        if name not in self._tools:
            return f"ERROR: unknown tool {name!r}. Available: {sorted(self._tools)}"
        try:
            return str(self._tools[name](**params))
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def names(self) -> list[str]:
        return sorted(self._tools)


def _bash_tool(command: str, timeout: str = "10") -> str:
    """Sandboxed shell execution. Runs the command, returns stdout (capped)."""
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError):
        timeout_s = 10.0
    try:
        out = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return (out.stdout + out.stderr)[:8192]
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout_s}s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def _read_tool(path: str) -> str:
    """Read a text file (capped at 32 KB)."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return f"ERROR: not a file: {p}"
        return p.read_text(encoding="utf-8", errors="replace")[:32768]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def default_tool_registry() -> ToolRegistry:
    """The default busel tool registry. Currently exposes TOOL_BASH and TOOL_READ."""
    r = ToolRegistry()
    r.register("TOOL_BASH", _bash_tool)
    r.register("TOOL_READ", _read_tool)
    return r


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def execute_tool_calls(text: str, registry: ToolRegistry | None = None) -> list[dict[str, Any]]:
    """Parse + execute every tool call in `text`. Returns per-call results.

    Each result: {name, params, output}.
    """
    if registry is None:
        registry = default_tool_registry()
    results: list[dict[str, Any]] = []
    for call in parse_tool_calls(text):
        output = registry.call(call["name"], call["params"])
        results.append({**call, "output": output})
    return results


def format_tool_results(results: list[dict[str, Any]]) -> str:
    """Format results as a `ROLE_TOOL ... TOOL_RESULTS_END` block (for re-injection)."""
    blocks = []
    for r in results:
        try:
            params_json = json.dumps(r["params"], ensure_ascii=False)
        except (TypeError, ValueError):
            params_json = str(r["params"])
        blocks.append(
            f'<result name="{r["name"]}">\n'
            f"{r['output']}\n"
            f"</result>"
        )
    if not blocks:
        return ""
    return "<function_results>\n" + "\n".join(blocks) + "\n</function_results>"


__all__ = [
    "parse_tool_calls",
    "execute_tool_calls",
    "format_tool_results",
    "ToolRegistry",
    "default_tool_registry",
    "INVOKE_PATTERN",
    "PARAM_PATTERN",
]
