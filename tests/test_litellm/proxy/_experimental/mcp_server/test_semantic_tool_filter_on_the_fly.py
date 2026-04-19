
import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from mcp.types import Tool as MCPTool
from litellm.proxy._experimental.mcp_server.semantic_tool_filter import SemanticMCPToolFilter
from litellm.types.utils import Embedding, EmbeddingResponse

@pytest.mark.asyncio
async def test_semantic_filter_on_the_fly_tools():
    """
    Test that tools provided on the fly are added to the router and filtered.
    """
    mock_router = Mock()
    def mock_embedding_sync(*args, **kwargs):
        return EmbeddingResponse(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
    async def mock_embedding_async(*args, **kwargs):
        return mock_embedding_sync()
    mock_router.embedding = mock_embedding_sync
    mock_router.aembedding = mock_embedding_async

    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=2,
        similarity_threshold=0.1, # Low threshold to ensure matches
        enabled=True,
    )

    # Initialize with empty list or some tools
    await filter_instance._build_router([])
    assert filter_instance.tool_router is not None

    # Tools provided on the fly
    tools = [
        {"name": "tool_1", "description": "Description 1"},
        {"name": "tool_2", "description": "Description 2"},
        {"name": "tool_3", "description": "Description 3"},
    ]

    # Filter tools - this should trigger _add_new_tools_to_router
    filtered = await filter_instance.filter_tools(
        query="test query",
        available_tools=tools,
    )

    # Assertions
    assert filter_instance.tool_router is not None
    assert len(filter_instance._tool_map) == 3
    assert len(filtered) <= 2 # top_k is 2
    
    # Check if they are actually in the router routes
    assert len(filter_instance.tool_router.routes) == 3

    # Add more tools on the fly
    more_tools = [
        {"name": "tool_1", "description": "Description 1"}, # Duplicate
        {"name": "tool_4", "description": "Description 4"}, # New
    ]
    
    # Combined tools for the next call
    combined_available = tools + [{"name": "tool_4", "description": "Description 4"}]

    filtered2 = await filter_instance.filter_tools(
        query="another query",
        available_tools=combined_available,
    )

    assert len(filter_instance._tool_map) == 4
    assert len(filter_instance.tool_router.routes) == 4
    assert len(filtered2) <= 2

@pytest.mark.asyncio
async def test_semantic_filter_limit_after_available_filter():
    """
    Test that the limit is applied AFTER filtering by available tools.
    
    If we have 10 tools in the router, but only 2 are available in the request,
    and top_k=5, we should get those 2 tools (if they match), 
    even if they are not in the top 5 of ALL tools in the router.
    """
    mock_router = Mock()
    def mock_embedding_sync(*args, **kwargs):
        return EmbeddingResponse(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
    async def mock_embedding_async(*args, **kwargs):
        return mock_embedding_sync()
    mock_router.embedding = mock_embedding_sync
    mock_router.aembedding = mock_embedding_async
    
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=2,
        similarity_threshold=0.1,
        enabled=True,
    )

    # 10 tools in total known to the router
    all_known_tools = [
        {"name": f"tool_{i}", "description": f"Description {i}"} for i in range(10)
    ]
    await filter_instance._build_router(all_known_tools)
    
    # Available tools in this request are only tool_8 and tool_9
    available_tools = [
        {"name": "tool_8", "description": "Description 8"},
        {"name": "tool_9", "description": "Description 9"},
    ]
    
    # Mock the router call to return all tools, but tool_8 and tool_9 are at the end
    class MockMatch:
        def __init__(self, name):
            self.name = name
            self.score = 0.5

    mock_matches = [MockMatch(f"tool_{i}") for i in range(10)]
    
    with patch("semantic_router.routers.SemanticRouter.__call__", return_value=mock_matches) as mock_call:
        filtered = await filter_instance.filter_tools(
            query="test",
            available_tools=available_tools,
            top_k=5
        )
        
        # matched_tool_names will be [tool_0, tool_1, ..., tool_9]
        # available_tools are [tool_8, tool_9]
        # After filtering by available, we should have [tool_8, tool_9]
        # Then limit 5 is applied.
        
        assert len(filtered) == 2
        assert filtered[0]["name"] == "tool_8"
        assert filtered[1]["name"] == "tool_9"

    # Now test the case where limit is smaller than available
    with patch("semantic_router.routers.SemanticRouter.__call__", return_value=mock_matches) as mock_call:
        filtered = await filter_instance.filter_tools(
            query="test",
            available_tools=available_tools,
            top_k=1
        )
        assert len(filtered) == 1
        assert filtered[0]["name"] == "tool_8"

@pytest.mark.asyncio
async def test_semantic_filter_chat_format_tools():
    """
    Test that tools in OpenAI chat format (nested 'function' key) are handled correctly.
    """
    mock_router = Mock()
    def mock_embedding_sync(*args, **kwargs):
        return EmbeddingResponse(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
    async def mock_embedding_async(*args, **kwargs):
        return mock_embedding_sync()
    mock_router.embedding = mock_embedding_sync
    mock_router.aembedding = mock_embedding_async

    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=2,
        similarity_threshold=0.1,
        enabled=True,
    )

    await filter_instance._build_router([])

    # Tools in chat format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "question",
                "description": "Use this tool to ask questions",
                "parameters": {"type": "object"}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {"type": "object"}
            }
        }
    ]

    filtered = await filter_instance.filter_tools(
        query="I want to run a script",
        available_tools=tools,
    )

    assert len(filtered) > 0
    # The tool names should be preserved in the map and correctly extracted
    assert "question" in filter_instance._tool_map
    assert "bash" in filter_instance._tool_map
    
    # Check if results maintain chat format
    for tool in filtered:
        assert "function" in tool
        assert "name" in tool["function"]
