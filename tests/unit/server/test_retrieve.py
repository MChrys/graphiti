"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import os

# Add server directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'server'))

import pytest
from datetime import datetime, timezone
from time import time
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from graph_service.routers.retrieve import (
    router, search, search_advanced, get_entity_edge, get_episodes, get_memory,
    _convert_date_filter_to_core, _convert_property_filter_to_core,
    _convert_search_filters_to_core, _build_search_filters_from_query,
    _convert_search_config_from_query, compose_query_from_messages
)
from graph_service.dto import (
    SearchQuery, SearchResults, GetMemoryRequest, GetMemoryResponse,
    DateFilter, PropertyFilter, SearchFilters, Message, ComparisonOperator,
    NodeSearchMethod, EdgeSearchMethod, Reranker, FactResult
)
from graph_service.zep_graphiti import ZepGraphitiDep


class TestDateFilterConversion:
    """Test date filter conversion functions"""

    def test_convert_date_filter_to_core(self):
        """Test converting API DateFilter to core DateFilter"""
        api_filter = DateFilter(
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            comparison_operator=ComparisonOperator.greater_than_equal
        )

        with patch('graph_service.routers.retrieve.CoreDateFilter') as mock_core_date, \
             patch('graph_service.routers.retrieve.CoreComparisonOperator') as mock_core_comp:

            mock_core_date.return_value = MagicMock()
            mock_core_comp.return_value = MagicMock()

            result = _convert_date_filter_to_core(api_filter)

            mock_core_comp.assert_called_once_with(ComparisonOperator.greater_than_equal.value)
            mock_core_date.assert_called_once_with(
                date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                comparison_operator=mock_core_comp.return_value
            )

    def test_convert_property_filter_to_core(self):
        """Test converting API PropertyFilter to core PropertyFilter"""
        api_filter = PropertyFilter(
            property_name="test_property",
            property_value="test_value",
            comparison_operator=ComparisonOperator.equals
        )

        with patch('graph_service.routers.retrieve.CorePropertyFilter') as mock_core_prop, \
             patch('graph_service.routers.retrieve.CoreComparisonOperator') as mock_core_comp:

            mock_core_prop.return_value = MagicMock()
            mock_core_comp.return_value = MagicMock()

            result = _convert_property_filter_to_core(api_filter)

            mock_core_comp.assert_called_once_with(ComparisonOperator.equals.value)
            mock_core_prop.assert_called_once_with(
                property_name="test_property",
                property_value="test_value",
                comparison_operator=mock_core_comp.return_value
            )


class TestSearchFiltersConversion:
    """Test search filters conversion functions"""

    def test_convert_search_filters_none(self):
        """Test converting None filters returns None"""
        result = _convert_search_filters_to_core(None)
        assert result is None

    def test_convert_search_filters_empty(self):
        """Test converting empty SearchFilters"""
        filters = SearchFilters()
        with patch('graph_service.routers.retrieve.CoreSearchFilters') as mock_core:
            mock_instance = MagicMock()
            mock_core.return_value = mock_instance

            result = _convert_search_filters_to_core(filters)

            mock_core.assert_called_once_with(
                node_labels=None,
                edge_types=None,
                created_at=None,
                valid_at=None,
                invalid_at=None,
                expired_at=None,
                edge_uuids=None,
                property_filters=None,
            )
            assert result == mock_instance

    def test_convert_search_filters_with_data(self):
        """Test converting SearchFilters with data"""
        date_filter = DateFilter(
            date=datetime.now(timezone.utc),
            comparison_operator=ComparisonOperator.greater_than
        )
        prop_filter = PropertyFilter(
            property_name="test",
            property_value="value",
            comparison_operator=ComparisonOperator.equals
        )

        filters = SearchFilters(
            node_labels=["Person", "Organization"],
            edge_types=["WORKS_FOR"],
            created_at=[[date_filter]],
            property_filters=[prop_filter]
        )

        with patch('graph_service.routers.retrieve.CoreSearchFilters') as mock_core, \
             patch('graph_service.routers.retrieve._convert_date_filter_to_core') as mock_date_conv, \
             patch('graph_service.routers.retrieve._convert_property_filter_to_core') as mock_prop_conv:

            mock_instance = MagicMock()
            mock_core.return_value = mock_instance
            mock_date_conv.return_value = MagicMock()
            mock_prop_conv.return_value = MagicMock()

            result = _convert_search_filters_to_core(filters)

            mock_date_conv.assert_called_once_with(date_filter)
            mock_prop_conv.assert_called_once_with(prop_filter)
            mock_core.assert_called_once()

    def test_build_search_filters_from_query_none(self):
        """Test building search filters with None query"""
        query = SearchQuery(query="test")

        with patch('graph_service.routers.retrieve._convert_search_filters_to_core') as mock_convert:
            mock_convert.return_value = None

            with patch('graph_service.routers.retrieve.CoreSearchFilters') as mock_core:
                mock_instance = MagicMock()
                mock_core.return_value = mock_instance

                result = _build_search_filters_from_query(query)

                mock_convert.assert_called_once_with(None)
                mock_core.assert_called_once()


class TestSearchConfigConversion:
    """Test search config conversion functions"""

    def test_convert_search_config_basic_search(self):
        """Test search config conversion for basic search"""
        query = SearchQuery(
            query="test query",
            max_facts=10
        )

        with patch('graph_service.routers.retrieve.EDGE_HYBRID_SEARCH_RRF') as mock_config:
            mock_config.return_value = MagicMock()

            result = _convert_search_config_from_query(query)

            mock_config.return_value.__setattr__.assert_called_with('limit', 10)

    def test_convert_search_config_advanced_search(self):
        """Test search config conversion for advanced search"""
        query = SearchQuery(
            query="test query",
            max_facts=15,
            node_search_methods=[NodeSearchMethod.cosine_similarity],
            edge_search_methods=[EdgeSearchMethod.bm25],
            reranker=Reranker.mmr,
            min_score=0.7,
            mmr_lambda=0.3,
            bfs_max_depth=5
        )

        with patch('graph_service.routers.retrieve.SearchConfig') as mock_config_class, \
             patch('graph_service.routers.retrieve.EdgeSearchConfig') as mock_edge_config, \
             patch('graph_service.routers.retrieve.NodeSearchConfig') as mock_node_config, \
             patch('graph_service.routers.retrieve.CoreEdgeSearchMethod') as mock_edge_method, \
             patch('graph_service.routers.retrieve.CoreNodeSearchMethod') as mock_node_method, \
             patch('graph_service.routers.retrieve.CoreEdgeReranker') as mock_reranker:

            mock_config = MagicMock()
            mock_edge_config_instance = MagicMock()
            mock_node_config_instance = MagicMock()

            mock_config_class.return_value = mock_config
            mock_edge_config.return_value = mock_edge_config_instance
            mock_node_config.return_value = mock_node_config_instance

            result = _convert_search_config_from_query(query)

            # Verify advanced search configuration was built
            mock_edge_method.assert_called()
            mock_node_method.assert_called()
            mock_reranker.assert_called()
            mock_edge_config.assert_called()
            mock_config_class.assert_called()


class TestSearchEndpoint:
    """Test search endpoint functionality"""

    @pytest.mark.asyncio
    async def test_search_basic_mode(self):
        """Test search in basic mode"""
        query = SearchQuery(
            query="test query",
            max_facts=10,
            group_ids=["group-123"]
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_edge = MagicMock()
        mock_edge.to_dict.return_value = {"uuid": "test-edge", "name": "test"}
        mock_graphiti.search.return_value = [mock_edge]

        with patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:
            mock_get_fact.return_value = FactResult(
                uuid="test-edge",
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await search(query, mock_graphiti)

            assert isinstance(response, SearchResults)
            assert len(response.facts) == 1
            assert response.total_results == 1
            assert response.ranking_method == "rrf"
            assert "hybrid_search_rrf" in response.search_config_used["method"]

    @pytest.mark.asyncio
    async def test_search_advanced_mode(self):
        """Test search in advanced mode"""
        query = SearchQuery(
            query="test query",
            max_facts=10,
            node_search_methods=[NodeSearchMethod.cosine_similarity],
            reranker=Reranker.mmr
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_search_results = MagicMock()
        mock_search_results.edges = [MagicMock()]
        mock_search_results.nodes = [MagicMock()]
        mock_search_results.edge_reranker_scores = [0.9]
        mock_graphiti.search.return_value = mock_search_results

        with patch('graph_service.routers.retrieve._convert_search_config_from_query') as mock_config, \
             patch('graph_service.routers.retrieve._build_search_filters_from_query') as mock_filters, \
             patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:

            mock_config.return_value = MagicMock()
            mock_filters.return_value = None
            mock_get_fact.return_value = FactResult(
                uuid="test",
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await search(query, mock_graphiti)

            assert isinstance(response, SearchResults)
            assert len(response.facts) == 1
            assert response.fact_scores == [0.9]
            assert response.search_time_ms is not None

    @pytest.mark.asyncio
    async def test_search_with_include_options(self):
        """Test search with include options"""
        query = SearchQuery(
            query="test query",
            max_facts=10,
            include_nodes=True,
            include_edges=True,
            include_episodes=True,
            include_communities=True,
            node_search_methods=[NodeSearchMethod.cosine_similarity]  # Force advanced mode
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_search_results = MagicMock()
        mock_search_results.edges = [MagicMock()]
        mock_search_results.nodes = [MagicMock(to_dict=lambda: {"uuid": "node-1"})]
        mock_search_results.episodes = [MagicMock(to_dict=lambda: {"uuid": "ep-1"})]
        mock_search_results.communities = [MagicMock(to_dict=lambda: {"uuid": "comm-1"})]
        mock_search_results.edge_reranker_scores = [0.8]
        mock_search_results.node_reranker_scores = [0.7]
        mock_search_results.episode_reranker_scores = [0.6]
        mock_search_results.community_reranker_scores = [0.5]
        mock_graphiti.search.return_value = mock_search_results

        with patch('graph_service.routers.retrieve._convert_search_config_from_query'), \
             patch('graph_service.routers.retrieve._build_search_filters_from_query'), \
             patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:

            mock_get_fact.return_value = FactResult(
                uuid="test",
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await search(query, mock_graphiti)

            assert response.nodes is not None
            assert response.edges is not None
            assert response.episodes is not None
            assert response.communities is not None
            assert response.node_scores == [0.7]
            assert response.edge_scores == [0.8]
            assert response.episode_scores == [0.6]
            assert response.community_scores == [0.5]


class TestSearchAdvancedEndpoint:
    """Test search advanced endpoint functionality"""

    @pytest.mark.asyncio
    async def test_search_advanced_always_uses_advanced_mode(self):
        """Test search advanced always uses advanced mode"""
        query = SearchQuery(
            query="test query",
            max_facts=10
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_search_results = MagicMock()
        mock_search_results.edges = [MagicMock()]
        mock_graphiti.search.return_value = mock_search_results

        with patch('graph_service.routers.retrieve._convert_search_config_from_query') as mock_config, \
             patch('graph_service.routers.retrieve._build_search_filters_from_query') as mock_filters, \
             patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:

            mock_config.return_value = MagicMock()
            mock_filters.return_value = None
            mock_get_fact.return_value = FactResult(
                uuid="test",
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await search_advanced(query, mock_graphiti)

            # Should always use advanced search
            mock_config.assert_called_once()
            mock_filters.assert_called_once()
            assert isinstance(response, SearchResults)


class TestGetEntityEdgeEndpoint:
    """Test get entity edge endpoint"""

    @pytest.mark.asyncio
    async def test_get_entity_edge_success(self):
        """Test successful entity edge retrieval"""
        edge_uuid = "edge-123"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_edge = MagicMock()
        mock_graphiti.get_entity_edge.return_value = mock_edge

        with patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:
            mock_get_fact.return_value = FactResult(
                uuid=edge_uuid,
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await get_entity_edge(edge_uuid, mock_graphiti)

            mock_graphiti.get_entity_edge.assert_called_once_with(edge_uuid)
            mock_get_fact.assert_called_once_with(mock_edge)

    @pytest.mark.asyncio
    async def test_get_entity_edge_not_found(self):
        """Test entity edge retrieval when not found"""
        edge_uuid = "nonexistent-edge"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.get_entity_edge.side_effect = Exception("Edge not found")

        with pytest.raises(Exception, match="Edge not found"):
            await get_entity_edge(edge_uuid, mock_graphiti)


class TestGetEpisodesEndpoint:
    """Test get episodes endpoint"""

    @pytest.mark.asyncio
    async def test_get_episodes_success(self):
        """Test successful episodes retrieval"""
        group_id = "group-123"
        last_n = 5

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_episodes = [MagicMock(), MagicMock()]
        mock_graphiti.retrieve_episodes.return_value = mock_episodes

        response = await get_episodes(group_id, last_n, mock_graphiti)

        mock_graphiti.retrieve_episodes.assert_called_once()
        call_args = mock_graphiti.retrieve_episodes.call_args
        assert call_args.args[0] == [group_id]  # group_ids
        assert call_args.args[1] == last_n      # last_n
        assert "reference_time" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_get_episodes_with_different_params(self):
        """Test episodes retrieval with different parameters"""
        group_id = "group-456"
        last_n = 10

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_episodes = []
        mock_graphiti.retrieve_episodes.return_value = mock_episodes

        response = await get_episodes(group_id, last_n, mock_graphiti)

        assert response == mock_episodes
        mock_graphiti.retrieve_episodes.assert_called_once()


class TestGetMemoryEndpoint:
    """Test get memory endpoint functionality"""

    @pytest.mark.asyncio
    async def test_get_memory_basic_mode(self):
        """Test get memory in basic mode"""
        request = GetMemoryRequest(
            group_id="group-123",
            messages=[
                Message(content="Hello", role_type="user"),
                Message(content="Hi there", role_type="assistant")
            ],
            max_facts=5
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_edge = MagicMock()
        mock_graphiti.search.return_value = [mock_edge]

        with patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:
            mock_get_fact.return_value = FactResult(
                uuid="test",
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await get_memory(request, mock_graphiti)

            assert isinstance(response, GetMemoryResponse)
            assert len(response.facts) == 1
            assert response.total_results == 1
            assert response.ranking_method == "rrf"

    @pytest.mark.asyncio
    async def test_get_memory_advanced_mode(self):
        """Test get memory in advanced mode"""
        request = GetMemoryRequest(
            group_id="group-123",
            messages=[Message(content="Hello", role_type="user")],
            max_facts=10,
            reranker=Reranker.mmr,
            min_score=0.7
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_search_results = MagicMock()
        mock_search_results.edges = [MagicMock()]
        mock_search_results.edge_reranker_scores = [0.9]
        mock_graphiti.search.return_value = mock_search_results

        with patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:
            mock_get_fact.return_value = FactResult(
                uuid="test",
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await get_memory(request, mock_graphiti)

            assert isinstance(response, GetMemoryResponse)
            assert response.fact_scores == [0.9]
            assert response.search_config_used is not None
            assert "mmr" in response.ranking_method

    @pytest.mark.asyncio
    async def test_get_memory_with_include_options(self):
        """Test get memory with include options"""
        request = GetMemoryRequest(
            group_id="group-123",
            messages=[Message(content="Hello", role_type="user")],
            include_nodes=True,
            include_episodes=True,
            node_search_methods=[NodeSearchMethod.cosine_similarity]  # Force advanced mode
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_search_results = MagicMock()
        mock_search_results.edges = [MagicMock()]
        mock_search_results.nodes = [MagicMock(to_dict=lambda: {"uuid": "node-1"})]
        mock_search_results.episodes = [MagicMock(to_dict=lambda: {"uuid": "ep-1"})]
        mock_search_results.edge_reranker_scores = [0.8]
        mock_search_results.node_reranker_scores = [0.7]
        mock_search_results.episode_reranker_scores = [0.6]
        mock_graphiti.search.return_value = mock_search_results

        with patch('graph_service.routers.retrieve.get_fact_result_from_edge') as mock_get_fact:
            mock_get_fact.return_value = FactResult(
                uuid="test",
                name="test",
                fact="test fact",
                valid_at=None,
                invalid_at=None,
                created_at=datetime.now(timezone.utc),
                expired_at=None
            )

            response = await get_memory(request, mock_graphiti)

            assert response.nodes is not None
            assert response.episodes is not None
            assert response.node_scores == [0.7]
            assert response.episode_scores == [0.6]


class TestQueryComposition:
    """Test query composition from messages"""

    def test_compose_query_from_messages_basic(self):
        """Test basic query composition from messages"""
        messages = [
            Message(content="Hello world", role_type="user", role="John"),
            Message(content="Hi there!", role_type="assistant", role="Bot")
        ]

        result = compose_query_from_messages(messages)

        expected = "John(user): Hello world\nBot(assistant): Hi there!\n"
        assert result == expected

    def test_compose_query_from_messages_none_role(self):
        """Test query composition with None role"""
        messages = [
            Message(content="Hello", role_type="user", role=None),
            Message(content="Hi", role_type="assistant")
        ]

        result = compose_query_from_messages(messages)

        expected = "user: Hello\nassistant: Hi\n"
        assert result == expected

    def test_compose_query_from_empty_messages(self):
        """Test query composition with empty messages"""
        messages = []

        result = compose_query_from_messages(messages)

        assert result == ""

    def test_compose_query_from_single_message(self):
        """Test query composition with single message"""
        messages = [
            Message(content="Test message", role_type="system")
        ]

        result = compose_query_from_messages(messages)

        expected = "system: Test message\n"
        assert result == expected

    def test_compose_query_with_special_characters(self):
        """Test query composition with special characters"""
        messages = [
            Message(content="Hello\nWorld\tTest", role_type="user", role="User")
        ]

        result = compose_query_from_messages(messages)

        expected = "User(user): Hello\nWorld\tTest\n"
        assert result == expected


class TestRetrieveRouter:
    """Test retrieve router configuration"""

    def test_router_tags(self):
        """Test that router has correct tags"""
        assert "search" in router.tags

    def test_router_has_no_lifespan(self):
        """Test that router doesn't have lifespan configured (unlike ingest router)"""
        assert router.lifespan is None

    def test_search_endpoints_configuration(self):
        """Test search endpoints configuration"""
        search_routes = [route for route in router.routes if "search" in route.path]
        assert len(search_routes) >= 2  # Should have /search and /search-advanced

        for route in search_routes:
            assert "POST" in route.methods

    def test_get_endpoints_configuration(self):
        """Test get endpoints configuration"""
        get_routes = [route for route in router.routes if "GET" in route.methods]
        assert len(get_routes) >= 2  # Should have /entity-edge and /episodes

        for route in get_routes:
            assert "GET" in route.methods

    def test_get_memory_endpoint_configuration(self):
        """Test get memory endpoint configuration"""
        for route in router.routes:
            if route.path == "/get-memory":
                assert "POST" in route.methods
                break
        else:
            pytest.fail("Get memory endpoint not found")


class TestIntegrationWithFastAPI:
    """Test integration with FastAPI TestClient"""

    def test_router_can_be_included_in_app(self):
        """Test that retrieve router can be included in FastAPI app"""
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router)

        # Should not raise any exceptions
        assert test_app is not None

    def test_endpoints_are_accessible_in_test_client(self):
        """Test that endpoints are accessible through TestClient"""
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router)

        client = TestClient(test_app)

        # Test that endpoints exist (may return 422 due to missing dependencies)
        response = client.post("/search", json={})
        assert response.status_code in [422, 500]

        response = client.post("/get-memory", json={})
        assert response.status_code in [422, 500]


class TestErrorHandling:
    """Test error handling in retrieve endpoints"""

    @pytest.mark.asyncio
    async def test_search_handles_graphiti_exceptions(self):
        """Test search endpoint handles graphiti exceptions"""
        query = SearchQuery(query="test", node_search_methods=[NodeSearchMethod.cosine_similarity])

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.search.side_effect = Exception("Search failed")

        with pytest.raises(Exception, match="Search failed"):
            await search(query, mock_graphiti)

    @pytest.mark.asyncio
    async def test_search_advanced_handles_graphiti_exceptions(self):
        """Test search advanced endpoint handles graphiti exceptions"""
        query = SearchQuery(query="test")

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.search.side_effect = Exception("Advanced search failed")

        with pytest.raises(Exception, match="Advanced search failed"):
            await search_advanced(query, mock_graphiti)

    @pytest.mark.asyncio
    async def test_get_memory_handles_graphiti_exceptions(self):
        """Test get memory endpoint handles graphiti exceptions"""
        request = GetMemoryRequest(
            group_id="test",
            messages=[Message(content="test", role_type="user")]
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.search.side_effect = Exception("Memory search failed")

        with pytest.raises(Exception, match="Memory search failed"):
            await get_memory(request, mock_graphiti)


class TestDateFilterEdgeCases:
    """Test date filter edge cases"""

    def test_date_filter_conversion_with_none_date(self):
        """Test date filter conversion with None date"""
        api_filter = DateFilter(
            date=None,
            comparison_operator=ComparisonOperator.is_null
        )

        with patch('graph_service.routers.retrieve.CoreDateFilter') as mock_core_date:
            mock_core_date.return_value = MagicMock()

            result = _convert_date_filter_to_core(api_filter)

            mock_core_date.assert_called_once_with(
                date=None,
                comparison_operator=mock_core_date.return_value
            )

    def test_property_filter_conversion_with_none_value(self):
        """Test property filter conversion with None value"""
        api_filter = PropertyFilter(
            property_name="test_prop",
            property_value=None,
            comparison_operator=ComparisonOperator.is_null
        )

        with patch('graph_service.routers.retrieve.CorePropertyFilter') as mock_core_prop:
            mock_core_prop.return_value = MagicMock()

            result = _convert_property_filter_to_core(api_filter)

            mock_core_prop.assert_called_once_with(
                property_name="test_prop",
                property_value=None,
                comparison_operator=mock_core_prop.return_value
            )