"""
Integration tests for date range filtering functionality.

These tests focus on testing the search endpoint logic with mock dependencies
to verify the integration between the API layer and the core search functionality.

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

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
import pytest

from graph_service.dto.retrieve import (
    SearchQuery,
    SearchResults,
    FactResult,
    Reranker,
    NodeSearchMethod,
    EdgeSearchMethod
)
from graph_service.routers.retrieve import router
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_filters import SearchFilters as CoreSearchFilters


class MockGraphiti:
    """Mock Graphiti instance for testing."""

    def __init__(self):
        self.search = AsyncMock()

    def create_mock_edge(self, uuid: str, name: str, fact: str, created_at: datetime):
        """Create a mock edge for testing."""
        edge = Mock(spec=EntityEdge)
        edge.uuid = uuid
        edge.name = name
        edge.fact = fact
        edge.created_at = created_at
        edge.valid_at = created_at + timedelta(days=1)
        edge.invalid_at = created_at + timedelta(days=30)
        edge.expired_at = created_at + timedelta(days=60)

        edge.to_dict.return_value = {
            "uuid": uuid,
            "name": name,
            "fact": fact,
            "created_at": created_at.isoformat(),
            "valid_at": edge.valid_at.isoformat(),
            "invalid_at": edge.invalid_at.isoformat(),
            "expired_at": edge.expired_at.isoformat()
        }

        return edge


class TestSearchEndpointIntegration:
    """Integration tests for search endpoint with date filtering."""

    @pytest.fixture
    def mock_graphiti(self):
        """Create mock Graphiti instance."""
        return MockGraphiti()

    @pytest.fixture
    def sample_edges(self):
        """Create sample edges for testing."""
        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_graphiti = MockGraphiti()

        return [
            mock_graphiti.create_mock_edge(
                "edge-1", "Edge 1", "January edge", base_time
            ),
            mock_graphiti.create_mock_edge(
                "edge-2", "Edge 2", "February edge", base_time + timedelta(days=20)
            ),
            mock_graphiti.create_mock_edge(
                "edge-3", "Edge 3", "March edge", base_time + timedelta(days=50)
            )
        ]

    @pytest.fixture
    def sample_search_results(self, sample_edges):
        """Create sample search results."""
        from types import SimpleNamespace

        return SimpleNamespace(
            edges=sample_edges,
            nodes=[],
            episodes=[],
            communities=[],
            edge_reranker_scores=[0.9, 0.8, 0.7],
            node_reranker_scores=None,
            episode_reranker_scores=None,
            community_reranker_scores=None
        )

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_endpoint_with_date_filters_integration(
        self, mock_dep, sample_search_results
    ):
        """Test search endpoint integration with date filters."""
        # Setup mock dependency
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Create search query with date filters
        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            max_facts=10
        )

        # Import and test the search endpoint function directly
        from graph_service.routers.retrieve import search

        # Call the endpoint function
        result = await search(query, mock_dep())

        # Verify the response structure
        assert isinstance(result, SearchResults)
        assert len(result.facts) == 3
        assert result.fact_scores is not None
        assert len(result.fact_scores) == 3
        assert result.total_results == 3
        assert result.search_time_ms is not None
        assert result.search_config_used is not None
        assert result.ranking_method is not None
        assert result.score_normalization == "none"
        assert result.max_score_possible is not None
        assert result.min_score_threshold is not None

        # Verify search was called with correct parameters
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        # Check query parameter
        assert call_args.args[0] == "test query"

        # Check search_filter parameter
        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert isinstance(search_filter, CoreSearchFilters)
        assert search_filter.created_at is not None

        # Check config parameter
        config = call_args.kwargs.get('config')
        assert config is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_endpoint_backward_compatibility(
        self, mock_dep, sample_search_results
    ):
        """Test that search endpoint maintains backward compatibility."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Create basic search query (no date filters)
        query = SearchQuery(
            query="basic test query",
            group_ids=["test-group"],
            max_facts=5
        )

        from graph_service.routers.retrieve import search

        result = await search(query, mock_dep())

        # Should still return SearchResults
        assert isinstance(result, SearchResults)
        assert len(result.facts) == 3

        # Should use basic search (no advanced config)
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        # Basic search should not include config or search_filter
        assert 'config' not in call_args.kwargs
        assert 'search_filter' not in call_args.kwargs

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_endpoint_with_ranking_options(
        self, mock_dep, sample_search_results
    ):
        """Test search endpoint with ranking options combined with date filters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query with ranking",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 3, 31, tzinfo=timezone.utc),
            reranker=Reranker.mmr,
            mmr_lambda=0.6,
            min_score=0.4,
            reranker_min_score=0.2,
            bfs_max_depth=4,
            max_facts=15
        )

        from graph_service.routers.retrieve import search

        result = await search(query, mock_dep())

        # Verify ranking metadata in response
        assert result.ranking_method == "mmr"
        assert result.search_config_used is not None
        assert result.search_config_used["edge_reranker"] == "mmr"
        assert result.search_config_used["mmr_lambda"] == 0.6
        assert result.min_score_threshold == 0.4

        # Verify search was called with config
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        config = call_args.kwargs.get('config')
        assert config is not None
        assert config.limit == 15

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.created_at is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_advanced_endpoint_always_advanced(
        self, mock_dep, sample_search_results
    ):
        """Test that search-advanced endpoint always uses advanced search."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Even basic query should use advanced search
        query = SearchQuery(
            query="simple query",
            group_ids=["test-group"],
            max_facts=10
        )

        from graph_service.routers.retrieve import search_advanced

        result = await search_advanced(query, mock_dep())

        # Should return basic SearchResults (no metadata for advanced endpoint)
        assert isinstance(result, SearchResults)
        assert len(result.facts) == 3
        assert result.fact_scores is not None

        # Should always use advanced search
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        config = call_args.kwargs.get('config')
        search_filter = call_args.kwargs.get('search_filter')

        # Advanced search should always have config and search_filter
        assert config is not None
        # search_filter might be None for simple queries, but config should always be present

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_get_memory_with_date_filters(
        self, mock_dep, sample_search_results
    ):
        """Test get-memory endpoint with date filters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        from graph_service.dto.retrieve import GetMemoryRequest, Message
        from graph_service.routers.retrieve import get_memory

        request = GetMemoryRequest(
            group_id="test-group",
            messages=[
                Message(role="user", role_type="user", content="What do you know?")
            ],
            max_facts=10,
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 2, 29, tzinfo=timezone.utc),
            valid_at_start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            node_labels=["TestEntity"]
        )

        result = await get_memory(request, mock_dep())

        # Should return GetMemoryResponse
        from graph_service.dto.retrieve import GetMemoryResponse
        assert isinstance(result, GetMemoryResponse)
        assert len(result.facts) == 3

        # Verify search was called with date filters
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.created_at is not None
        assert search_filter.valid_at is not None
        assert search_filter.node_labels == ["TestEntity"]

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_complex_filter_combination(
        self, mock_dep, sample_search_results
    ):
        """Test complex combination of all filter types."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        from graph_service.dto.retrieve import SearchFilters, PropertyFilter, ComparisonOperator

        advanced_filters = SearchFilters(
            property_filters=[
                PropertyFilter(
                    property_name="priority",
                    property_value="high",
                    comparison_operator=ComparisonOperator.equals
                )
            ]
        )

        query = SearchQuery(
            query="complex query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            valid_at_start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            valid_at_end=datetime(2024, 2, 15, tzinfo=timezone.utc),
            invalid_at_start=datetime(2024, 3, 1, tzinfo=timezone.utc),
            invalid_at_end=datetime(2024, 3, 31, tzinfo=timezone.utc),
            expired_at_start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            expired_at_end=datetime(2024, 4, 30, tzinfo=timezone.utc),
            node_labels=["Entity", "Document"],
            edge_types=["HAS_RELATIONSHIP", "MENTIONS"],
            filters=advanced_filters,
            reranker=Reranker.cross_encoder,
            min_score=0.7,
            max_facts=20
        )

        from graph_service.routers.retrieve import search

        result = await search(query, mock_dep())

        # Should return valid results
        assert isinstance(result, SearchResults)
        assert len(result.facts) == 3

        # Verify all filters were applied
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None

        # Verify all date filters
        assert search_filter.created_at is not None
        assert search_filter.valid_at is not None
        assert search_filter.invalid_at is not None
        assert search_filter.expired_at is not None

        # Verify entity filters
        assert search_filter.node_labels == ["Entity", "Document"]
        assert search_filter.edge_types == ["HAS_RELATIONSHIP", "MENTIONS"]

        # Verify property filters
        assert search_filter.property_filters is not None
        assert len(search_filter.property_filters) == 1
        assert search_filter.property_filters[0].property_name == "priority"
        assert search_filter.property_filters[0].property_value == "high"

        # Verify ranking config
        config = call_args.kwargs.get('config')
        assert config is not None
        assert config.edge_config.reranker.value == "cross_encoder"
        assert config.edge_config.sim_min_score == 0.7
        assert config.limit == 20

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_result_metadata_completeness(
        self, mock_dep, sample_search_results
    ):
        """Test that search response includes all required metadata."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="metadata test",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            reranker=Reranker.mmr,
            mmr_lambda=0.8,
            min_score=0.6,
            max_facts=12
        )

        from graph_service.routers.retrieve import search

        result = await search(query, mock_dep())

        # Verify all metadata fields are present
        assert result.fact_scores is not None
        assert len(result.fact_scores) == 3
        assert all(isinstance(score, (int, float)) for score in result.fact_scores)

        assert result.total_results == 3
        assert isinstance(result.search_time_ms, int)
        assert result.search_time_ms > 0

        assert result.search_config_used is not None
        config_metadata = result.search_config_used
        assert "edge_search_methods" in config_metadata
        assert "edge_reranker" in config_metadata
        assert "limit" in config_metadata
        assert config_metadata["edge_reranker"] == "mmr"
        assert config_metadata["mmr_lambda"] == 0.8
        assert config_metadata["limit"] == 12

        assert result.ranking_method == "mmr"
        assert result.score_normalization == "none"
        assert result.max_score_possible is not None  # MMR should have max_score_possible
        assert result.min_score_threshold == 0.6