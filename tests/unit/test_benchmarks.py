from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any

import pytest
import mcp.types as types
from mcp_server_appwrite.docs_search import DocsSearch, _clamp_limit
from mcp_server_appwrite.operator import Operator, ResultStore
from mcp_server_appwrite.tool_manager import ToolManager


def test_public_tools_have_safety_annotations():
    """Verify that all public tools exposed by Operator contain MCP ToolAnnotations."""
    manager = ToolManager()
    executor = ThreadPoolExecutor(max_workers=1)
    docs_search = DocsSearch()
    operator = Operator(manager, executor, docs_search=docs_search)
    tools = operator.get_public_tools()

    assert len(tools) >= 3
    for tool in tools:
        assert tool.annotations is not None, f"Tool {tool.name} missing annotations"
        assert isinstance(tool.annotations.readOnlyHint, bool)
        assert isinstance(tool.annotations.destructiveHint, bool)


def test_docs_search_tool_annotation():
    """Verify docs_search tool specific safety annotations."""
    docs_search = DocsSearch()
    tool = docs_search.get_tool()
    assert tool.annotations is not None
    assert tool.annotations.title == "Appwrite Documentation Search"
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.idempotentHint is True


def test_benchmark_clamp_limit():
    """Benchmark micro-performance of limit clamping under high throughput."""
    iterations = 100_000
    start = time.perf_counter()
    for i in range(iterations):
        _ = _clamp_limit((i % 50) + 1, 10)
    elapsed = time.perf_counter() - start

    # 100k iterations must complete in under 0.25 seconds
    assert elapsed < 0.25, f"Clamping benchmark took too long: {elapsed:.4f}s"


def test_benchmark_result_store_ops():
    """Benchmark ResultStore resource storage and retrieval latency."""
    store = ResultStore(max_size=50)

    start = time.perf_counter()
    for i in range(500):
        res = store.save(f"tool_{i}", [types.TextContent(type="text", text=f"sample_{i}")], f"text_{i}")
        retrieved = store.get(res.result_id)
        assert retrieved is not None

    elapsed = time.perf_counter() - start
    # 500 save + get ops must take < 0.1s
    assert elapsed < 0.1, f"ResultStore benchmark took too long: {elapsed:.4f}s"
