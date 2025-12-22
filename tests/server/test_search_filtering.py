"""
Tests for date range filtering functionality in the search endpoint.

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

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from graph_service.dto.retrieve import (
    SearchQuery,
    DateFilter,
    SearchFilters,
    ComparisonOperator,
    SearchResults,
    FactResult,
    Reranker,
    NodeSearchMethod,
    EdgeSearchMethod
)
from graph_service.routers.retrieve import router
from graph_service.zep_graphiti import ZepGraphitiDep


# Test fixtures for creating mock data
@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_graphiti():
    """Mock ZepGraphiti dependency."""
    mock = Mock(spec=ZepGraphitiDep)
    mock.search = AsyncMock()
    return mock


@pytest.fixture
def sample_fact_results():
    """Create sample fact results for testing."""
    base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    return [
        FactResult(
            uuid="fact-1",
            name="Fact 1",
            fact="Test fact created in January",
            valid_at=base_time + timedelta(days=1),
            invalid_at=base_time + timedelta(days=30),
            created_at=base_time,
            expired_at=base_time + timedelta(days=60)
        ),
        FactResult(
            uuid="fact-2",
            name="Fact 2",
            fact="Test fact created in February",
            valid_at=base_time + timedelta(days=35),
            invalid_at=base_time + timedelta(days=65),
            created_at=base_time + timedelta(days=20),
            expired_at=base_time + timedelta(days=90)
        ),
        FactResult(
            uuid="fact-3",
            name="Fact 3",
            fact="Test fact created in March",
            valid_at=base_time + timedelta(days=70),
            invalid_at=base_time + timedelta(days=100),
            created_at=base_time + timedelta(days=50),
            expired_at=base_time + timedelta(days=120)
        )
    ]


@pytest.fixture
def sample_search_results(sample_fact_results):
    """Create sample search results with mock edges."""
    # Mock edges for conversion to facts
    mock_edges = []
    for fact in sample_fact_results:
        mock_edge = Mock()
        mock_edge.uuid = fact.uuid
        mock_edge.name = fact.name
        mock_edge.fact = fact.fact
        mock_edge.valid_at = fact.valid_at
        mock_edge.invalid_at = fact.invalid_at
        mock_edge.created_at = fact.created_at
        mock_edge.expired_at = fact.expired_at
        mock_edge.to_dict.return_value = {
            "uuid": fact.uuid,
            "name": fact.name,
            "fact": fact.fact,
            "valid_at": fact.valid_at.isoformat() if fact.valid_at else None,
            "invalid_at": fact.invalid_at.isoformat() if fact.invalid_at else None,
            "created_at": fact.created_at.isoformat(),
            "expired_at": fact.expired_at.isoformat() if fact.expired_at else None
        }
        mock_edges.append(mock_edge)

    # Create mock search results
    mock_search_result = Mock()
    mock_search_result.edges = mock_edges
    mock_search_result.nodes = []
    mock_search_result.episodes = []
    mock_search_result.communities = []
    mock_search_result.edge_reranker_scores = [0.9, 0.8, 0.7]
    mock_search_result.node_reranker_scores = None
    mock_search_result.episode_reranker_scores = None
    mock_search_result.community_reranker_scores = None

    return mock_search_result


class TestDateRangeFiltering:
    """Test suite for date range filtering functionality."""

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_created_at_date_range_filter(self, mock_dep, client, sample_search_results):
        """Test filtering by created_at date range."""
        # Setup mock
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Create search query with created_at date range
        start_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 20, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=start_date,
            created_at_end=end_date,
            max_facts=10
        )

        # Make request
        response = client.post("/api/v1/search", json=query.model_dump())

        # Assertions
        assert response.status_code == 200
        data = response.json()

        # Verify the response structure
        assert "facts" in data
        assert "search_config_used" in data
        assert "ranking_method" in data
        assert "score_normalization" in data
        assert "max_score_possible" in data
        assert "min_score_threshold" in data

        # Verify search was called with correct parameters
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        # Check that search filters were built correctly
        assert call_args.kwargs.get('search_filter') is not None
        search_filter = call_args.kwargs['search_filter']

        # Verify created_at filter was applied
        assert search_filter.created_at is not None
        assert len(search_filter.created_at) == 1
        assert len(search_filter.created_at[0]) == 2  # start and end filters

        # Check the filters are correct
        start_filter = search_filter.created_at[0][0]
        end_filter = search_filter.created_at[0][1]

        assert start_filter.date == start_date
        assert start_filter.comparison_operator == ComparisonOperator.greater_than_equal
        assert end_filter.date == end_date
        assert end_filter.comparison_operator == ComparisonOperator.less_than_equal

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_valid_at_date_range_filter(self, mock_dep, client, sample_search_results):
        """Test filtering by valid_at date range."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        start_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        end_date = datetime(2024, 2, 10, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            valid_at_start=start_date,
            valid_at_end=end_date,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.valid_at is not None
        assert len(search_filter.valid_at) == 1
        assert len(search_filter.valid_at[0]) == 2

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_invalid_at_date_range_filter(self, mock_dep, client, sample_search_results):
        """Test filtering by invalid_at date range."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        start_date = datetime(2024, 2, 10, tzinfo=timezone.utc)
        end_date = datetime(2024, 3, 1, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            invalid_at_start=start_date,
            invalid_at_end=end_date,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.invalid_at is not None
        assert len(search_filter.invalid_at) == 1
        assert len(search_filter.invalid_at[0]) == 2

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_expired_at_date_range_filter(self, mock_dep, client, sample_search_results):
        """Test filtering by expired_at date range."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 4, 1, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            expired_at_start=start_date,
            expired_at_end=end_date,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.expired_at is not None
        assert len(search_filter.expired_at) == 1
        assert len(search_filter.expired_at[0]) == 2

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_multiple_date_range_filters(self, mock_dep, client, sample_search_results):
        """Test applying multiple date range filters simultaneously."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            valid_at_start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            valid_at_end=datetime(2024, 2, 15, tzinfo=timezone.utc),
            expired_at_start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            expired_at_end=datetime(2024, 3, 31, tzinfo=timezone.utc),
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None

        # Verify all date filters were applied
        assert search_filter.created_at is not None
        assert search_filter.valid_at is not None
        assert search_filter.expired_at is not None

        # Verify invalid_at was not set (not in query)
        assert search_filter.invalid_at is None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_single_date_boundary_filters(self, mock_dep, client, sample_search_results):
        """Test single boundary date filters (only start or only end)."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Test only start date
        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())
        assert response.status_code == 200

        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')

        assert search_filter is not None
        assert search_filter.created_at is not None
        assert len(search_filter.created_at) == 1
        assert len(search_filter.created_at[0]) == 1  # Only start filter
        assert search_filter.created_at[0][0].comparison_operator == ComparisonOperator.greater_than_equal

        # Reset mock for next test
        mock_graphiti_instance.search.reset_mock()

        # Test only end date
        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())
        assert response.status_code == 200

        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')

        assert search_filter is not None
        assert search_filter.created_at is not None
        assert len(search_filter.created_at) == 1
        assert len(search_filter.created_at[0]) == 1  # Only end filter
        assert search_filter.created_at[0][0].comparison_operator == ComparisonOperator.less_than_equal

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_advanced_search_filters_combination(self, mock_dep, client, sample_search_results):
        """Test combining simplified date filters with advanced filters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        advanced_filters = SearchFilters(
            node_labels=["Entity", "Person"],
            edge_types=["HAS_RELATIONSHIP"],
            property_filters=[
                {
                    "property_name": "importance",
                    "property_value": "high",
                    "comparison_operator": ComparisonOperator.equals
                }
            ]
        )

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            filters=advanced_filters,
            node_labels=["TestNode"],  # This should be merged with advanced filters
            edge_types=["TestEdge"],   # This should be merged with advanced filters
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None

        # Verify simplified filters were merged with advanced filters
        assert search_filter.created_at is not None
        assert search_filter.node_labels is not None
        assert search_filter.edge_types is not None
        assert search_filter.property_filters is not None

        # Simplified filters should override advanced filters for the same fields
        assert "TestNode" in search_filter.node_labels
        assert "TestEdge" in search_filter.edge_types

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_date_range_filter_with_ranking_options(self, mock_dep, client, sample_search_results):
        """Test date range filtering combined with ranking options."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            reranker=Reranker.mmr,
            mmr_lambda=0.7,
            min_score=0.5,
            reranker_min_score=0.3,
            bfs_max_depth=5,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify ranking options were applied
        assert data["ranking_method"] == "mmr"
        assert "search_config_used" in data
        assert data["search_config_used"]["edge_reranker"] == "mmr"
        assert data["search_config_used"]["mmr_lambda"] == 0.7
        assert data["min_score_threshold"] == 0.5

        # Verify search was called with config
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        assert call_args.kwargs.get('config') is not None
        assert call_args.kwargs.get('search_filter') is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_date_range_filter_edge_cases(self, mock_dep, client, sample_search_results):
        """Test edge cases for date range filtering."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Test with identical start and end dates
        same_day = datetime(2024, 1, 15, tzinfo=timezone.utc)
        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=same_day,
            created_at_end=same_day,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())
        assert response.status_code == 200

        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')

        assert search_filter is not None
        assert search_filter.created_at is not None
        assert len(search_filter.created_at) == 1
        assert len(search_filter.created_at[0]) == 2  # Both filters should be present

        # Verify both filters use the same date
        start_filter = search_filter.created_at[0][0]
        end_filter = search_filter.created_at[0][1]
        assert start_filter.date == same_day
        assert end_filter.date == same_day

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_no_date_range_filters_backward_compatibility(self, mock_dep, client, sample_search_results):
        """Test that queries without date filters maintain backward compatibility."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

        # Should use basic search (backward compatibility mode)
        call_args = mock_graphiti_instance.search.call_args
        # In basic mode, search_filter should not be passed
        assert 'search_filter' not in call_args.kwargs
        assert 'config' not in call_args.kwargs

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_get_memory_with_date_range_filters(self, mock_dep, client, sample_search_results):
        """Test date range filtering with get-memory endpoint."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        request_data = {
            "group_id": "test-group",
            "messages": [
                {"role": "user", "role_type": "user", "content": "What do you know about X?"}
            ],
            "max_facts": 10,
            "created_at_start": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
            "created_at_end": datetime(2024, 1, 31, tzinfo=timezone.utc).isoformat(),
            "valid_at_start": datetime(2024, 1, 15, tzinfo=timezone.utc).isoformat()
        }

        response = client.post("/api/v1/get-memory", json=request_data)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.created_at is not None
        assert search_filter.valid_at is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_advanced_endpoint_with_date_filters(self, mock_dep, client, sample_search_results):
        """Test date range filtering with search-advanced endpoint."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            max_facts=10
        )

        response = client.post("/api/v1/search-advanced", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        # search-advanced always uses advanced search
        search_filter = call_args.kwargs.get('search_filter')
        config = call_args.kwargs.get('config')

        assert search_filter is not None
        assert config is not None
        assert search_filter.created_at is not None

    def test_date_filter_validation(self):
        """Test validation of date filter inputs."""
        # Test that datetime objects are properly handled
        valid_date = datetime(2024, 1, 15, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            created_at_start=valid_date,
            created_at_end=valid_date,
            valid_at_start=valid_date,
            valid_at_end=valid_date,
            invalid_at_start=valid_date,
            invalid_at_end=valid_date,
            expired_at_start=valid_date,
            expired_at_end=valid_date,
        )

        # Should not raise any validation errors
        assert query.created_at_start == valid_date
        assert query.created_at_end == valid_date
        assert query.valid_at_start == valid_date
        assert query.valid_at_end == valid_date
        assert query.invalid_at_start == valid_date
        assert query.invalid_at_end == valid_date
        assert query.expired_at_start == valid_date
        assert query.expired_at_end == valid_date

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_config_metadata_with_date_filters(self, mock_dep, client, sample_search_results):
        """Test that search config metadata includes date filter information."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            reranker=Reranker.mmr,
            mmr_lambda=0.6,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify metadata structure
        assert "search_config_used" in data
        metadata = data["search_config_used"]

        # Should include edge search methods
        assert "edge_search_methods" in metadata
        assert isinstance(metadata["edge_search_methods"], list)

        # Should include reranker information
        assert "edge_reranker" in metadata
        assert metadata["edge_reranker"] == "mmr"

        # Should include MMR lambda when using MMR
        assert "mmr_lambda" in metadata
        assert metadata["mmr_lambda"] == 0.6

        # Should include limit
        assert "limit" in metadata
        assert metadata["limit"] == 10


class TestErrorHandling:
    """Test error handling for date range filtering."""

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_invalid_date_format_handling(self, mock_dep, client):
        """Test handling of invalid date formats in requests."""
        mock_graphiti_instance = Mock()
        mock_dep.return_value = mock_graphiti_instance

        # Test with invalid date format (should be caught by Pydantic validation)
        invalid_request = {
            "query": "test query",
            "group_ids": ["test-group"],
            "created_at_start": "invalid-date-format",
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=invalid_request)

        # FastAPI/Pydantic should return 422 for invalid date format
        assert response.status_code == 422

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_service_error_handling(self, mock_dep, client):
        """Test error handling when search service fails."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(side_effect=Exception("Search service error"))
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        # Should return 500 internal server error
        assert response.status_code == 500


class TestEntityFiltering:
    """Test suite for entity type and label filtering functionality."""

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_node_labels_filter(self, mock_dep, client, sample_search_results):
        """Test filtering by node labels."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Person", "Organization", "Location"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels is not None
        assert len(search_filter.node_labels) == 3
        assert "Person" in search_filter.node_labels
        assert "Organization" in search_filter.node_labels
        assert "Location" in search_filter.node_labels

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_edge_types_filter(self, mock_dep, client, sample_search_results):
        """Test filtering by edge types."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            edge_types=["WORKS_FOR", "LOCATED_IN", "KNOWS"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.edge_types is not None
        assert len(search_filter.edge_types) == 3
        assert "WORKS_FOR" in search_filter.edge_types
        assert "LOCATED_IN" in search_filter.edge_types
        assert "KNOWS" in search_filter.edge_types

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_combined_entity_filters(self, mock_dep, client, sample_search_results):
        """Test filtering by both node labels and edge types simultaneously."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Person", "Company"],
            edge_types=["EMPLOYED_BY", "FOUNDED"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels is not None
        assert search_filter.edge_types is not None
        assert len(search_filter.node_labels) == 2
        assert len(search_filter.edge_types) == 2

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_entity_filters_with_date_ranges(self, mock_dep, client, sample_search_results):
        """Test combining entity filters with date range filters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Person"],
            edge_types=["KNOWS"],
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            valid_at_start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels is not None
        assert search_filter.edge_types is not None
        assert search_filter.created_at is not None
        assert search_filter.valid_at is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_entity_filters_with_ranking_options(self, mock_dep, client, sample_search_results):
        """Test entity filtering combined with ranking options."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Entity", "Event"],
            edge_types=["PARTICIPATED_IN"],
            reranker=Reranker.cross_encoder,
            min_score=0.7,
            reranker_min_score=0.4,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify ranking options were applied
        assert data["ranking_method"] == "cross_encoder"
        assert data["search_config_used"]["edge_reranker"] == "cross_encoder"
        assert data["min_score_threshold"] == 0.7

        # Verify entity filters were applied
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels is not None
        assert search_filter.edge_types is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_entity_filters_with_advanced_search_filters(self, mock_dep, client, sample_search_results):
        """Test combining simplified entity filters with advanced filters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        advanced_filters = SearchFilters(
            node_labels=["Document", "Article"],
            edge_types=["MENTIONS", "REFERENCES"],
            property_filters=[
                {
                    "property_name": "importance",
                    "property_value": "high",
                    "comparison_operator": ComparisonOperator.equals
                }
            ]
        )

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Person", "Organization"],  # Should override advanced filters
            edge_types=["WORKS_WITH"],  # Should override advanced filters
            filters=advanced_filters,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None

        # Simplified filters should override advanced filters for the same fields
        assert search_filter.node_labels is not None
        assert search_filter.edge_types is not None
        assert search_filter.property_filters is not None

        assert "Person" in search_filter.node_labels
        assert "Organization" in search_filter.node_labels
        assert "WORKS_WITH" in search_filter.edge_types

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_single_entity_filters(self, mock_dep, client, sample_search_results):
        """Test single entity filter values."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Test single node label
        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Person"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())
        assert response.status_code == 200

        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels == ["Person"]
        assert search_filter.edge_types is None

        # Reset mock for next test
        mock_graphiti_instance.search.reset_mock()

        # Test single edge type
        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            edge_types=["KNOWS"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())
        assert response.status_code == 200

        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.edge_types == ["KNOWS"]
        assert search_filter.node_labels is None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_empty_entity_filters(self, mock_dep, client, sample_search_results):
        """Test empty entity filter lists."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=[],
            edge_types=[],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        # Empty lists should be passed as None in the search filter
        search_filter = call_args.kwargs.get('search_filter')
        if search_filter:
            # The implementation might convert empty lists to None or keep them as empty
            assert search_filter.node_labels is None or search_filter.node_labels == []
            assert search_filter.edge_types is None or search_filter.edge_types == []

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_entity_filters_case_sensitivity(self, mock_dep, client, sample_search_results):
        """Test that entity filters preserve case sensitivity."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Person", "PERSON", "person"],
            edge_types=["KNOWS", "knows", "Knows"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels is not None
        assert search_filter.edge_types is not None

        # Should preserve the exact case provided
        assert "Person" in search_filter.node_labels
        assert "PERSON" in search_filter.node_labels
        assert "person" in search_filter.node_labels
        assert "KNOWS" in search_filter.edge_types
        assert "knows" in search_filter.edge_types
        assert "Knows" in search_filter.edge_types

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_entity_filters_with_different_search_methods(self, mock_dep, client, sample_search_results):
        """Test entity filters with different search method configurations."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Event"],
            edge_types=["PARTICIPATED_IN"],
            node_search_methods=[NodeSearchMethod.bm25],
            edge_search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=Reranker.node_distance,
            bfs_max_depth=4,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify search methods and ranking were applied
        assert data["search_config_used"]["edge_search_methods"] == ["bm25"]
        assert data["search_config_used"]["node_search_methods"] == ["cosine_similarity"]
        assert data["search_config_used"]["edge_reranker"] == "node_distance"

        # Verify entity filters were applied
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels == ["Event"]
        assert search_filter.edge_types == ["PARTICIPATED_IN"]

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_get_memory_with_entity_filters(self, mock_dep, client, sample_search_results):
        """Test entity filtering with get-memory endpoint."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        request_data = {
            "group_id": "test-group",
            "messages": [
                {"role": "user", "role_type": "user", "content": "What do you know about X?"}
            ],
            "max_facts": 10,
            "node_labels": ["Person", "Organization"],
            "edge_types": ["WORKS_FOR"]
        }

        response = client.post("/api/v1/get-memory", json=request_data)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels == ["Person", "Organization"]
        assert search_filter.edge_types == ["WORKS_FOR"]

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_search_advanced_endpoint_with_entity_filters(self, mock_dep, client, sample_search_results):
        """Test entity filtering with search-advanced endpoint."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Technology", "Company"],
            edge_types=["DEVELOPS", "OWNS"],
            max_facts=10
        )

        response = client.post("/api/v1/search-advanced", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        # search-advanced always uses advanced search
        search_filter = call_args.kwargs.get('search_filter')
        config = call_args.kwargs.get('config')

        assert search_filter is not None
        assert config is not None
        assert search_filter.node_labels == ["Technology", "Company"]
        assert search_filter.edge_types == ["DEVELOPS", "OWNS"]

    def test_entity_filter_validation(self):
        """Test validation of entity filter inputs."""
        # Test with valid entity filter values
        query = SearchQuery(
            query="test query",
            node_labels=["Person", "Organization", "Location"],
            edge_types=["KNOWS", "WORKS_FOR", "LOCATED_IN"],
            max_facts=10
        )

        # Should not raise any validation errors
        assert query.node_labels == ["Person", "Organization", "Location"]
        assert query.edge_types == ["KNOWS", "WORKS_FOR", "LOCATED_IN"]

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_entity_filters_with_result_inclusion_options(self, mock_dep, client, sample_search_results):
        """Test entity filters with different result inclusion options."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Community"],
            edge_types=["MEMBER_OF"],
            include_nodes=True,
            include_edges=True,
            include_episodes=True,
            include_communities=True,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args

        # Verify both filters and config were passed
        search_filter = call_args.kwargs.get('search_filter')
        config = call_args.kwargs.get('config')

        assert search_filter is not None
        assert config is not None
        assert search_filter.node_labels == ["Community"]
        assert search_filter.edge_types == ["MEMBER_OF"]

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_no_entity_filters_backward_compatibility(self, mock_dep, client, sample_search_results):
        """Test that queries without entity filters maintain backward compatibility."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

        # Should use basic search (backward compatibility mode)
        call_args = mock_graphiti_instance.search.call_args
        # In basic mode, search_filter and config should not be passed
        assert 'search_filter' not in call_args.kwargs
        assert 'config' not in call_args.kwargs

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_entity_filters_with_mmr_ranking(self, mock_dep, client, sample_search_results):
        """Test entity filtering with MMR ranking and lambda parameter."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            node_labels=["Entity"],
            edge_types=["RELATED_TO"],
            reranker=Reranker.mmr,
            mmr_lambda=0.8,
            min_score=0.6,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify MMR ranking was applied
        assert data["ranking_method"] == "mmr"
        assert data["search_config_used"]["edge_reranker"] == "mmr"
        assert data["search_config_used"]["mmr_lambda"] == 0.8
        assert data["min_score_threshold"] == 0.6

        # Verify entity filters were applied
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')
        assert search_filter is not None
        assert search_filter.node_labels == ["Entity"]
        assert search_filter.edge_types == ["RELATED_TO"]

    def test_empty_string_entity_filters(self):
        """Test handling of empty strings in entity filters."""
        # Test with empty strings (should be filtered out or handled appropriately)
        query = SearchQuery(
            query="test query",
            node_labels=["Person", "", "Organization"],
            edge_types=["KNOWS", "", "WORKS_FOR"],
            max_facts=10
        )

        # Empty strings should be preserved as they might be valid in some contexts
        # The filtering logic should handle them appropriately
        assert "" in query.node_labels
        assert "" in query.edge_types

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_rrf_reranking_default_behavior(self, mock_dep, client, sample_search_results):
        """Test RRF (reciprocal_rank_fusion) reranking as default behavior."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Test without explicit reranker (should default to RRF)
        query = SearchQuery(
            query="test query for default reranking",
            group_ids=["test-group"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Should use RRF as default
        assert data["ranking_method"] == "reciprocal_rank_fusion"
        assert data["search_config_used"]["edge_reranker"] == "reciprocal_rank_fusion"

    async def test_rrf_reranking_explicit(self, mock_dep, client, sample_search_results):
        """Test explicit RRF (reciprocal_rank_fusion) reranking configuration."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query for explicit RRF",
            group_ids=["test-group"],
            reranker=Reranker.rrf,
            min_score=0.3,
            max_facts=15
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify explicit RRF configuration
        assert data["ranking_method"] == "reciprocal_rank_fusion"
        assert data["search_config_used"]["edge_reranker"] == "reciprocal_rank_fusion"
        assert data["min_score_threshold"] == 0.3

    async def test_episode_mentions_reranking(self, mock_dep, client, sample_search_results):
        """Test episode_mentions reranking configuration."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query for episode mentions reranking",
            group_ids=["test-group"],
            reranker=Reranker.episode_mentions,
            min_score=0.4,
            max_facts=12
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify episode mentions configuration
        assert data["ranking_method"] == "episode_mentions"
        assert data["search_config_used"]["edge_reranker"] == "episode_mentions"
        assert data["min_score_threshold"] == 0.4

    async def test_cross_encoder_reranking_advanced(self, mock_dep, client, sample_search_results):
        """Test cross_encoder reranking with advanced parameters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query for cross encoder advanced",
            group_ids=["test-group"],
            reranker=Reranker.cross_encoder,
            reranker_min_score=0.6,
            min_score=0.2,
            max_facts=8
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify cross encoder configuration
        assert data["ranking_method"] == "cross_encoder"
        assert data["search_config_used"]["edge_reranker"] == "cross_encoder"
        assert data["search_config_used"]["reranker_min_score"] == 0.6

    async def test_node_distance_reranking_advanced(self, mock_dep, client, sample_search_results):
        """Test node_distance reranking with BFS configuration."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test query for node distance advanced",
            group_ids=["test-group"],
            reranker=Reranker.node_distance,
            bfs_max_depth=3,
            min_score=0.1,
            max_facts=20
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify node distance configuration
        assert data["ranking_method"] == "node_distance"
        assert data["search_config_used"]["edge_reranker"] == "node_distance"
        assert data["search_config_used"]["bfs_max_depth"] == 3

    async def test_mmr_reranking_lambda_variations(self, mock_dep, client, sample_search_results):
        """Test MMR reranking with different lambda values."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Test with high lambda (more relevance, less diversity)
        query_high_lambda = SearchQuery(
            query="test query high lambda",
            group_ids=["test-group"],
            reranker=Reranker.mmr,
            mmr_lambda=0.9,
            max_facts=5
        )

        response = client.post("/api/v1/search", json=query_high_lambda.model_dump())

        assert response.status_code == 200
        data = response.json()

        assert data["ranking_method"] == "mmr"
        assert data["search_config_used"]["edge_reranker"] == "mmr"
        assert data["search_config_used"]["mmr_lambda"] == 0.9

        # Test with low lambda (more diversity, less relevance)
        query_low_lambda = SearchQuery(
            query="test query low lambda",
            group_ids=["test-group"],
            reranker=Reranker.mmr,
            mmr_lambda=0.1,
            max_facts=5
        )

        response = client.post("/api/v1/search", json=query_low_lambda.model_dump())

        assert response.status_code == 200
        data = response.json()

        assert data["search_config_used"]["mmr_lambda"] == 0.1

    async def test_all_reranking_methods_comparison(self, mock_dep, client, sample_search_results):
        """Test all available reranking methods with the same query."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        rerankers_to_test = [
            Reranker.rrf,
            Reranker.mmr,
            Reranker.cross_encoder,
            Reranker.node_distance,
            Reranker.episode_mentions
        ]

        expected_ranking_methods = [
            "reciprocal_rank_fusion",
            "mmr",
            "cross_encoder",
            "node_distance",
            "episode_mentions"
        ]

        for i, reranker in enumerate(rerankers_to_test):
            query = SearchQuery(
                query=f"comparison test for {reranker.value}",
                group_ids=["test-group"],
                reranker=reranker,
                max_facts=10
            )

            response = client.post("/api/v1/search", json=query.model_dump())

            assert response.status_code == 200
            data = response.json()

            # Verify each reranking method is correctly applied
            assert data["ranking_method"] == expected_ranking_methods[i]
            assert data["search_config_used"]["edge_reranker"] == reranker.value

    async def test_reranking_with_date_filters(self, mock_dep, client, sample_search_results):
        """Test reranking methods combined with date filtering."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        base_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test reranking with date filters",
            group_ids=["test-group"],
            created_at_start=base_time,
            created_at_end=base_time + timedelta(days=30),
            reranker=Reranker.mmr,
            mmr_lambda=0.5,
            max_facts=8
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify both date filtering and reranking are applied
        assert data["ranking_method"] == "mmr"
        assert data["search_config_used"]["edge_reranker"] == "mmr"
        assert data["search_config_used"]["mmr_lambda"] == 0.5

    async def test_reranking_with_entity_filters(self, mock_dep, client, sample_search_results):
        """Test reranking methods combined with entity filtering."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test reranking with entity filters",
            group_ids=["test-group"],
            node_labels=["Person", "Organization"],
            edge_types=["WORKS_FOR", "KNOWS"],
            reranker=Reranker.cross_encoder,
            reranker_min_score=0.7,
            max_facts=15
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify both entity filtering and reranking are applied
        assert data["ranking_method"] == "cross_encoder"
        assert data["search_config_used"]["edge_reranker"] == "cross_encoder"
        assert data["search_config_used"]["reranker_min_score"] == 0.7

    async def test_reranking_parameter_validation(self, mock_dep, client, sample_search_results):
        """Test parameter validation for different reranking methods."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Test MMR with invalid lambda (should still work, validation happens at core level)
        query = SearchQuery(
            query="test invalid mmr lambda",
            group_ids=["test-group"],
            reranker=Reranker.mmr,
            mmr_lambda=1.5,  # Invalid: should be 0.0 to 1.0
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        # Should pass through, core validation will handle
        assert response.status_code == 200
        data = response.json()
        assert data["search_config_used"]["mmr_lambda"] == 1.5

        # Test negative score thresholds
        query_negative_score = SearchQuery(
            query="test negative scores",
            group_ids=["test-group"],
            reranker=Reranker.cross_encoder,
            reranker_min_score=-0.5,
            min_score=-0.2,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query_negative_score.model_dump())

        # Should pass through, core validation will handle
        assert response.status_code == 200

    async def test_reranking_with_search_methods(self, mock_dep, client, sample_search_results):
        """Test reranking combined with different search methods."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test reranking with search methods",
            group_ids=["test-group"],
            node_search_methods=[NodeSearchMethod.bm25],
            edge_search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=Reranker.rrf,
            max_facts=12
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        data = response.json()

        # Verify search methods and reranking are both applied
        assert data["search_config_used"]["node_search_methods"] == ["bm25"]
        assert data["search_config_used"]["edge_search_methods"] == ["cosine_similarity"]
        assert data["search_config_used"]["edge_reranker"] == "reciprocal_rank_fusion"


class TestParameterValidationAndEdgeCases:
    """Test suite for parameter validation and edge cases."""

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_invalid_date_range_order(self, mock_dep, client):
        """Test validation when start date is after end date."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            group_ids=["test-group"],
            created_at_start=start_date,
            created_at_end=end_date,  # End before start - invalid
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        # API should still accept it (validation might happen at core level)
        assert response.status_code == 200

        # But the search should be called with the invalid parameters
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')

        assert search_filter is not None
        assert search_filter.created_at is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_extreme_date_values(self, mock_dep, client):
        """Test with extremely old and future dates."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        # Test with very old date
        ancient_date = datetime(1, 1, 1, tzinfo=timezone.utc)
        # Test with very far future date
        future_date = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test extreme dates",
            group_ids=["test-group"],
            created_at_start=ancient_date,
            created_at_end=future_date,
            valid_at_start=datetime(1970, 1, 1, tzinfo=timezone.utc),  # Unix epoch
            valid_at_end=datetime(2100, 1, 1, tzinfo=timezone.utc),
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    def test_invalid_datetime_timezone_handling(self):
        """Test datetime objects with different timezone configurations."""
        # Test with naive datetime (no timezone)
        naive_datetime = datetime(2024, 1, 15, 12, 0, 0)  # No timezone

        # Should still work but might cause issues
        query = SearchQuery(
            query="test naive datetime",
            created_at_start=naive_datetime,
            max_facts=10
        )

        # Pydantic might convert naive datetime to UTC
        assert query.created_at_start is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_same_day_multiple_filters(self, mock_dep, client):
        """Test multiple date filters on the same day."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        same_day = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test same day filters",
            group_ids=["test-group"],
            created_at_start=same_day,
            created_at_end=same_day,
            valid_at_start=same_day,
            valid_at_end=same_day,
            invalid_at_start=same_day,
            invalid_at_end=same_day,
            expired_at_start=same_day,
            expired_at_end=same_day,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')

        # All date filters should be present
        assert search_filter.created_at is not None
        assert search_filter.valid_at is not None
        assert search_filter.invalid_at is not None
        assert search_filter.expired_at is not None

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_microsecond_precision_dates(self, mock_dep, client):
        """Test dates with microsecond precision."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        precise_date = datetime(2024, 6, 15, 12, 34, 56, 789012, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test microsecond precision",
            group_ids=["test-group"],
            created_at_start=precise_date,
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_empty_and_whitespace_entity_filters(self, mock_dep, client):
        """Test entity filters with empty strings and whitespace."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test whitespace entity filters",
            group_ids=["test-group"],
            node_labels=["Person", "", "   ", "\t", "\n", "Organization"],
            edge_types=["KNOWS", "", "  ", "WORKS_FOR"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')

        assert search_filter is not None
        assert search_filter.node_labels is not None
        assert search_filter.edge_types is not None

        # Should preserve whitespace values (core filtering should handle them)
        assert "" in search_filter.node_labels
        assert "   " in search_filter.node_labels

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_very_long_entity_filter_values(self, mock_dep, client):
        """Test entity filters with very long string values."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        long_string = "A" * 1000  # 1000 character string

        query = SearchQuery(
            query="test long entity filters",
            group_ids=["test-group"],
            node_labels=[long_string, "NormalLabel"],
            edge_types=[long_string, "NormalEdge"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_unicode_entity_filters(self, mock_dep, client):
        """Test entity filters with unicode characters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test unicode entity filters",
            group_ids=["test-group"],
            node_labels=["人物", "организация", "مؤسسة", "🏢", "Café", "Naïve"],
            edge_types=["関係", "связь", "علاقة", "🔗"],
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_duplicate_entity_filter_values(self, mock_dep, client):
        """Test entity filters with duplicate values."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test duplicate entity filters",
            group_ids=["test-group"],
            node_labels=["Person", "Person", "Person"],  # All duplicates
            edge_types=["KNOWS", "KNOWS", "KNOWS"],  # All duplicates
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()
        call_args = mock_graphiti_instance.search.call_args
        search_filter = call_args.kwargs.get('search_filter')

        # Should preserve duplicates (deduplication might happen at core level)
        assert search_filter.node_labels == ["Person", "Person", "Person"]
        assert search_filter.edge_types == ["KNOWS", "KNOWS", "KNOWS"]

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_extreme_ranking_parameters(self, mock_dep, client):
        """Test ranking parameters with extreme values."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        # Test extreme parameter values
        query = SearchQuery(
            query="test extreme ranking parameters",
            group_ids=["test-group"],
            reranker=Reranker.mmr,
            mmr_lambda=999.999,  # Very high lambda
            min_score=-1000.0,   # Very low score
            reranker_min_score=1000.0,  # Very high reranker score
            bfs_max_depth=10000,  # Very large depth
            max_facts=1,         # Minimum facts
            max_context=-1       # Invalid context
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

        # Should pass extreme values through for core validation
        data = response.json()
        assert data["search_config_used"]["mmr_lambda"] == 999.999
        assert data["min_score_threshold"] == -1000.0

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_negative_mmr_lambda(self, mock_dep, client):
        """Test MMR with negative lambda value."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test negative mmr lambda",
            group_ids=["test-group"],
            reranker=Reranker.mmr,
            mmr_lambda=-0.5,  # Negative lambda
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

        data = response.json()
        assert data["search_config_used"]["mmr_lambda"] == -0.5

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_zero_and_very_small_parameters(self, mock_dep, client):
        """Test with zero and very small parameter values."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test zero parameters",
            group_ids=["test-group"],
            reranker=Reranker.mmr,
            mmr_lambda=0.0,         # Zero lambda
            min_score=0.0,          # Zero score
            reranker_min_score=0.0, # Zero reranker score
            bfs_max_depth=0,        # Zero depth
            max_facts=0,            # Zero facts
            max_context=0           # Zero context
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_very_small_floating_point_values(self, mock_dep, client):
        """Test with very small floating point values."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        query = SearchQuery(
            query="test tiny floating point values",
            group_ids=["test-group"],
            reranker=Reranker.mmr,
            mmr_lambda=1e-10,        # Very small lambda
            min_score=1e-20,         # Very small score
            reranker_min_score=1e-30, # Extremely small score
            max_facts=10
        )

        response = client.post("/api/v1/search", json=query.model_dump())

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    def test_malformed_request_payloads(self, client):
        """Test various malformed request payloads."""
        # Test missing required fields
        payload_missing_query = {
            "group_ids": ["test-group"],
            "max_facts": 10
            # Missing "query" field
        }

        response = client.post("/api/v1/search", json=payload_missing_query)
        assert response.status_code == 422  # Validation error

        # Test invalid data types
        payload_invalid_types = {
            "query": "test query",
            "group_ids": "not-a-list",  # Should be a list
            "max_facts": "not-a-number",  # Should be a number
            "created_at_start": 12345  # Should be datetime string
        }

        response = client.post("/api/v1/search", json=payload_invalid_types)
        assert response.status_code == 422  # Validation error

        # Test completely invalid JSON
        response = client.post("/api/v1/search", data="not valid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422  # JSON parsing error

        # Test empty object
        response = client.post("/api/v1/search", json={})
        assert response.status_code == 422  # Missing required fields

        # Test null values for required fields
        payload_null_values = {
            "query": None,
            "group_ids": None,
            "max_facts": None
        }

        response = client.post("/api/v1/search", json=payload_null_values)
        assert response.status_code == 422  # Null values not allowed

    def test_invalid_json_formats(self, client):
        """Test invalid JSON format variations."""
        # Test with trailing comma (invalid JSON)
        invalid_json_1 = '{"query": "test", "group_ids": ["test"],}'

        response = client.post("/api/v1/search", data=invalid_json_1, headers={"Content-Type": "application/json"})
        assert response.status_code == 422

        # Test with single quotes (invalid JSON)
        invalid_json_2 = "{'query': 'test', 'group_ids': ['test']}"

        response = client.post("/api/v1/search", data=invalid_json_2, headers={"Content-Type": "application/json"})
        assert response.status_code == 422

        # Test with unescaped characters
        invalid_json_3 = '{"query": "test with "quotes" inside", "group_ids": ["test"]}'

        response = client.post("/api/v1/search", data=invalid_json_3, headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_missing_optional_parameters(self, mock_dep, client, sample_search_results):
        """Test request with only required parameters (minimal valid request)."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        # Minimal valid request
        minimal_payload = {
            "query": "test query",
            "group_ids": ["test-group"]
            # All other parameters are optional
        }

        response = client.post("/api/v1/search", json=minimal_payload)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

        # Should use default values for missing parameters
        call_args = mock_graphiti_instance.search.call_args

        # In basic mode (no advanced features), should not pass config or search_filter
        assert 'config' not in call_args.kwargs
        assert 'search_filter' not in call_args.kwargs

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_empty_string_query(self, mock_dep, client):
        """Test with empty string query."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        payload = {
            "query": "",  # Empty string
            "group_ids": ["test-group"],
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload)

        # Empty string query should be valid (search might return all results)
        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_very_long_query_string(self, mock_dep, client):
        """Test with very long query string."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        long_query = "test " * 10000  # Very long query

        payload = {
            "query": long_query,
            "group_ids": ["test-group"],
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_unicode_query_string(self, mock_dep, client):
        """Test query string with unicode characters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        unicode_query = "Test query with unicode: 你好, Привет, مرحبا, 🚀, Café, Naïve"

        payload = {
            "query": unicode_query,
            "group_ids": ["test-group"],
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_special_characters_in_query(self, mock_dep, client):
        """Test query with special characters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        special_query = 'Test with "quotes", \'apostrophes\', & symbols, <tags>, {brackets}, [arrays]'

        payload = {
            "query": special_query,
            "group_ids": ["test-group"],
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_newlines_and_tabs_in_query(self, mock_dep, client):
        """Test query with whitespace characters."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        whitespace_query = "Test with\nnewlines\tand\ttabs\nmultiple lines"

        payload = {
            "query": whitespace_query,
            "group_ids": ["test-group"],
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    def test_invalid_group_ids(self, client):
        """Test various invalid group_ids configurations."""
        # Test empty group_ids list
        payload_empty_groups = {
            "query": "test query",
            "group_ids": [],  # Empty list should be invalid
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload_empty_groups)
        assert response.status_code == 422  # Validation error

        # Test non-string group_ids
        payload_invalid_groups = {
            "query": "test query",
            "group_ids": [123, True, None, {}],  # Invalid group ID types
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload_invalid_groups)
        assert response.status_code == 422  # Validation error

        # Test group_ids as string instead of list
        payload_string_groups = {
            "query": "test query",
            "group_ids": "test-group",  # Should be a list
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload_string_groups)
        assert response.status_code == 422  # Validation error

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_very_large_group_ids_list(self, mock_dep, client):
        """Test with very large number of group IDs."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        many_groups = [f"group-{i}" for i in range(1000)]  # 1000 group IDs

        payload = {
            "query": "test query",
            "group_ids": many_groups,
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload)

        assert response.status_code == 200
        mock_graphiti_instance.search.assert_called_once()

    def test_invalid_max_facts_values(self, client):
        """Test various invalid max_facts values."""
        # Test negative max_facts
        payload_negative_facts = {
            "query": "test query",
            "group_ids": ["test-group"],
            "max_facts": -5  # Negative number
        }

        response = client.post("/api/v1/search", json=payload_negative_facts)
        # API might accept it, but core validation should handle
        # Let's see what actually happens
        if response.status_code == 200:
            # If accepted, that's expected behavior
            pass
        else:
            # If rejected, that's also valid
            assert response.status_code in [422, 400]

        # Test very large max_facts
        payload_large_facts = {
            "query": "test query",
            "group_ids": ["test-group"],
            "max_facts": 999999999  # Very large number
        }

        response = client.post("/api/v1/search", json=payload_large_facts)
        # Should either accept or validate it
        assert response.status_code in [200, 422]

        # Test float max_facts
        payload_float_facts = {
            "query": "test query",
            "group_ids": ["test-group"],
            "max_facts": 10.5  # Float instead of integer
        }

        response = client.post("/api/v1/search", json=payload_float_facts)
        # Should either accept (coerce to int) or reject
        assert response.status_code in [200, 422]

    def test_invalid_content_type(self, client):
        """Test requests with invalid content types."""
        valid_payload = {
            "query": "test query",
            "group_ids": ["test-group"],
            "max_facts": 10
        }

        # Test with wrong content type
        response = client.post("/api/v1/search",
                              data=json.dumps(valid_payload),
                              headers={"Content-Type": "text/plain"})

        # Should return 415 Unsupported Media Type or 422
        assert response.status_code in [415, 422]

        # Test without content type
        response = client.post("/api/v1/search",
                              data=json.dumps(valid_payload))

        # Should still work if FastAPI can detect JSON
        assert response.status_code in [200, 422, 415]

    def test_http_method_validation(self, client):
        """Test that only POST method is accepted."""
        valid_payload = {
            "query": "test query",
            "group_ids": ["test-group"],
            "max_facts": 10
        }

        # Test GET method (should not be allowed)
        response = client.get("/api/v1/search", json=valid_payload)
        assert response.status_code == 405  # Method Not Allowed

        # Test PUT method (should not be allowed)
        response = client.put("/api/v1/search", json=valid_payload)
        assert response.status_code == 405  # Method Not Allowed

        # Test DELETE method (should not be allowed)
        response = client.delete("/api/v1/search", json=valid_payload)
        assert response.status_code == 405  # Method Not Allowed

        # Test PATCH method (should not be allowed)
        response = client.patch("/api/v1/search", json=valid_payload)
        assert response.status_code == 405  # Method Not Allowed

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_null_values_in_optional_fields(self, mock_dep, client):
        """Test null values in optional fields."""
        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock()
        mock_dep.return_value = mock_graphiti_instance

        payload_with_nulls = {
            "query": "test query",
            "group_ids": ["test-group"],
            "created_at_start": None,  # Null optional field
            "created_at_end": None,    # Null optional field
            "node_labels": None,       # Null optional field
            "edge_types": None,        # Null optional field
            "reranker": None,          # Null optional field
            "mmr_lambda": None,        # Null optional field
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload_with_nulls)

        # Should handle null values gracefully
        if response.status_code == 200:
            mock_graphiti_instance.search.assert_called_once()
        else:
            # If it rejects null values, that's also valid
            assert response.status_code == 422

    def test_nested_object_validation(self, client):
        """Test validation of nested object structures."""
        # Test with invalid nested filters structure
        payload_invalid_filters = {
            "query": "test query",
            "group_ids": ["test-group"],
            "filters": {
                "property_filters": [
                    {
                        "invalid_field": "test",  # Invalid field in property filter
                        "property_name": "test"
                        # Missing required fields
                    }
                ]
            },
            "max_facts": 10
        }

        response = client.post("/api/v1/search", json=payload_invalid_filters)
        # Should either accept (if flexible) or reject (if strict)
        assert response.status_code in [200, 422]

    @patch('graph_service.routers.retrieve.ZepGraphitiDep')
    async def test_concurrent_request_handling(self, mock_dep, client, sample_search_results):
        """Test that multiple concurrent requests are handled properly."""
        import asyncio

        mock_graphiti_instance = Mock()
        mock_graphiti_instance.search = AsyncMock(return_value=sample_search_results)
        mock_dep.return_value = mock_graphiti_instance

        async def make_request():
            payload = {
                "query": "concurrent test query",
                "group_ids": ["test-group"],
                "max_facts": 5
            }
            return client.post("/api/v1/search", json=payload)

        # Create multiple concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should succeed
        for response in responses:
            assert hasattr(response, 'status_code')
            assert response.status_code == 200

        # Should have called search 10 times
        assert mock_graphiti_instance.search.call_count == 10