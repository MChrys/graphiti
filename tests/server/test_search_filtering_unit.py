"""
Unit tests for date range filtering functionality in search endpoint.

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
from unittest.mock import Mock, AsyncMock
import pytest

from graph_service.dto.retrieve import (
    SearchQuery,
    DateFilter,
    SearchFilters,
    ComparisonOperator,
    Reranker,
    NodeSearchMethod,
    EdgeSearchMethod
)
from graph_service.routers.retrieve import (
    _convert_date_filter_to_core,
    _convert_property_filter_to_core,
    _convert_search_filters_to_core,
    _build_search_filters_from_query,
    _convert_search_config_from_query
)


class TestDateFilterConversion:
    """Test date filter conversion functions."""

    def test_convert_date_filter_to_core(self):
        """Test conversion of API DateFilter to core DateFilter."""
        test_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        api_filter = DateFilter(
            date=test_date,
            comparison_operator=ComparisonOperator.greater_than_equal
        )

        core_filter = _convert_date_filter_to_core(api_filter)

        from graphiti_core.search.search_filters import DateFilter as CoreDateFilter
        from graphiti_core.search.search_filters import ComparisonOperator as CoreComparisonOperator

        assert isinstance(core_filter, CoreDateFilter)
        assert core_filter.date == test_date
        assert core_filter.comparison_operator == CoreComparisonOperator.greater_than_equal

    def test_convert_date_filter_all_operators(self):
        """Test conversion with all comparison operators."""
        test_date = datetime(2024, 1, 15, tzinfo=timezone.utc)

        for operator in ComparisonOperator:
            api_filter = DateFilter(
                date=test_date,
                comparison_operator=operator
            )

            core_filter = _convert_date_filter_to_core(api_filter)

            from graphiti_core.search.search_filters import ComparisonOperator as CoreComparisonOperator
            assert core_filter.comparison_operator == CoreComparisonOperator(operator.value)


class TestSearchFilterBuilding:
    """Test search filter building from SearchQuery."""

    def test_build_search_filters_from_created_at_range(self):
        """Test building search filters from created_at date range."""
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            created_at_start=start_date,
            created_at_end=end_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at is not None
        assert len(search_filters.created_at) == 1
        assert len(search_filters.created_at[0]) == 2

        # Check start filter
        start_filter = search_filters.created_at[0][0]
        assert start_filter.date == start_date
        assert start_filter.comparison_operator.value == '>='

        # Check end filter
        end_filter = search_filters.created_at[0][1]
        assert end_filter.date == end_date
        assert end_filter.comparison_operator.value == '<='

    def test_build_search_filters_from_valid_at_range(self):
        """Test building search filters from valid_at date range."""
        start_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 2, 29, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            valid_at_start=start_date,
            valid_at_end=end_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.valid_at is not None
        assert len(search_filters.valid_at) == 1
        assert len(search_filters.valid_at[0]) == 2

    def test_build_search_filters_from_invalid_at_range(self):
        """Test building search filters from invalid_at date range."""
        start_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 3, 31, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            invalid_at_start=start_date,
            invalid_at_end=end_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.invalid_at is not None
        assert len(search_filters.invalid_at) == 1
        assert len(search_filters.invalid_at[0]) == 2

    def test_build_search_filters_from_expired_at_range(self):
        """Test building search filters from expired_at date range."""
        start_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 4, 30, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            expired_at_start=start_date,
            expired_at_end=end_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.expired_at is not None
        assert len(search_filters.expired_at) == 1
        assert len(search_filters.expired_at[0]) == 2

    def test_build_search_filters_multiple_date_ranges(self):
        """Test building search filters with multiple date ranges."""
        query = SearchQuery(
            query="test query",
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            valid_at_start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            valid_at_end=datetime(2024, 2, 29, tzinfo=timezone.utc),
            invalid_at_start=datetime(2024, 3, 1, tzinfo=timezone.utc),
            invalid_at_end=datetime(2024, 3, 31, tzinfo=timezone.utc),
            expired_at_start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            expired_at_end=datetime(2024, 4, 30, tzinfo=timezone.utc)
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at is not None
        assert search_filters.valid_at is not None
        assert search_filters.invalid_at is not None
        assert search_filters.expired_at is not None

        # Each should have one OR clause with two AND conditions
        assert len(search_filters.created_at) == 1
        assert len(search_filters.created_at[0]) == 2
        assert len(search_filters.valid_at) == 1
        assert len(search_filters.valid_at[0]) == 2
        assert len(search_filters.invalid_at) == 1
        assert len(search_filters.invalid_at[0]) == 2
        assert len(search_filters.expired_at) == 1
        assert len(search_filters.expired_at[0]) == 2

    def test_build_search_filters_single_boundary(self):
        """Test building search filters with only start or only end date."""
        # Test only start date
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        query = SearchQuery(
            query="test query",
            created_at_start=start_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at is not None
        assert len(search_filters.created_at) == 1
        assert len(search_filters.created_at[0]) == 1
        assert search_filters.created_at[0][0].date == start_date
        assert search_filters.created_at[0][0].comparison_operator.value == '>='

        # Test only end date
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        query = SearchQuery(
            query="test query",
            created_at_end=end_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at is not None
        assert len(search_filters.created_at) == 1
        assert len(search_filters.created_at[0]) == 1
        assert search_filters.created_at[0][0].date == end_date
        assert search_filters.created_at[0][0].comparison_operator.value == '<='

    def test_build_search_filters_with_entity_filters(self):
        """Test building search filters with entity type filters."""
        query = SearchQuery(
            query="test query",
            node_labels=["Entity", "Person"],
            edge_types=["HAS_RELATIONSHIP", "KNOWS"]
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.node_labels == ["Entity", "Person"]
        assert search_filters.edge_types == ["HAS_RELATIONSHIP", "KNOWS"]

    def test_build_search_filters_with_advanced_filters(self):
        """Test building search filters with advanced filter objects."""
        advanced_filters = SearchFilters(
            node_labels=["AdvancedEntity"],
            edge_types=["AdvancedEdge"],
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
            filters=advanced_filters
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        # Should preserve the advanced filters
        assert search_filters.node_labels == ["AdvancedEntity"]
        assert search_filters.edge_types == ["AdvancedEdge"]
        assert len(search_filters.property_filters) == 1

    def test_build_search_filters_merges_simple_and_advanced(self):
        """Test that simplified filters override advanced filters."""
        advanced_filters = SearchFilters(
            node_labels=["AdvancedNode"],
            edge_types=["AdvancedEdge"]
        )

        query = SearchQuery(
            query="test query",
            filters=advanced_filters,
            node_labels=["SimpleNode"],  # Should override
            edge_types=["SimpleEdge"],   # Should override
            created_at_start=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        # Simplified filters should override
        assert search_filters.node_labels == ["SimpleNode"]
        assert search_filters.edge_types == ["SimpleEdge"]
        # But date filters should be added
        assert search_filters.created_at is not None

    def test_build_search_filters_empty_query(self):
        """Test building search filters from empty query."""
        query = SearchQuery(query="test query")

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.node_labels is None
        assert search_filters.edge_types is None
        assert search_filters.created_at is None
        assert search_filters.valid_at is None
        assert search_filters.invalid_at is None
        assert search_filters.expired_at is None


class TestSearchConfigConversion:
    """Test search configuration conversion."""

    def test_convert_search_config_basic_query(self):
        """Test converting basic search query without advanced options."""
        query = SearchQuery(
            query="test query",
            max_facts=10
        )

        config = _convert_search_config_from_query(query)

        # Should return default config for basic queries
        assert config is not None
        assert hasattr(config, 'limit')
        assert config.limit == 10

    def test_convert_search_config_with_ranking_options(self):
        """Test converting search query with ranking options."""
        query = SearchQuery(
            query="test query",
            max_facts=15,
            reranker=Reranker.mmr,
            mmr_lambda=0.7,
            min_score=0.5,
            reranker_min_score=0.3,
            bfs_max_depth=4
        )

        config = _convert_search_config_from_query(query)

        assert config is not None
        assert config.limit == 15

        # Check edge config
        assert hasattr(config, 'edge_config')
        assert config.edge_config.reranker.value == 'mmr'
        assert config.edge_config.mmr_lambda == 0.7
        assert config.edge_config.sim_min_score == 0.5
        assert config.edge_config.reranker_min_score == 0.3
        assert config.edge_config.bfs_max_depth == 4

    def test_convert_search_config_with_node_search(self):
        """Test converting search query with node search enabled."""
        query = SearchQuery(
            query="test query",
            max_facts=10,
            include_nodes=True,
            node_search_methods=[NodeSearchMethod.cosine_similarity, NodeSearchMethod.bm25]
        )

        config = _convert_search_config_from_query(query)

        assert config is not None
        assert config.node_config is not None
        assert len(config.node_config.search_methods) == 2

    def test_convert_search_config_advanced_triggers(self):
        """Test that various parameters trigger advanced search config."""
        # Test that each advanced parameter triggers advanced search
        advanced_params = [
            {"reranker": Reranker.mmr},
            {"min_score": 0.5},
            {"mmr_lambda": 0.7},
            {"reranker_min_score": 0.3},
            {"bfs_max_depth": 5},
            {"node_search_methods": [NodeSearchMethod.cosine_similarity]},
            {"edge_search_methods": [EdgeSearchMethod.bm25]},
            {"include_nodes": False},  # Different from default
            {"include_episodes": True},
            {"include_communities": True},
        ]

        for param in advanced_params:
            query = SearchQuery(query="test", max_facts=10, **param)
            config = _convert_search_config_from_query(query)

            assert config is not None
            # Advanced configs should have edge_config
            assert hasattr(config, 'edge_config')


class TestDateFilterEdgeCases:
    """Test edge cases for date filtering."""

    def test_same_day_start_end(self):
        """Test filtering with same start and end date."""
        same_day = datetime(2024, 1, 15, tzinfo=timezone.utc)
        query = SearchQuery(
            query="test query",
            created_at_start=same_day,
            created_at_end=same_day
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at is not None
        assert len(search_filters.created_at) == 1
        assert len(search_filters.created_at[0]) == 2

        # Both filters should use the same date
        assert search_filters.created_at[0][0].date == same_day
        assert search_filters.created_at[0][1].date == same_day

    def test_date_filter_timezone_handling(self):
        """Test that timezone information is preserved."""
        # Test with UTC timezone
        utc_date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        query = SearchQuery(
            query="test query",
            created_at_start=utc_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at[0][0].date == utc_date
        assert search_filters.created_at[0][0].date.tzinfo == timezone.utc

    def test_date_boundary_precision(self):
        """Test date filtering with high precision timestamps."""
        precise_time = datetime(2024, 1, 15, 12, 34, 56, 789123, tzinfo=timezone.utc)
        query = SearchQuery(
            query="test query",
            created_at_start=precise_time,
            created_at_end=precise_time + timedelta(microseconds=1)
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at is not None

        # Check that microsecond precision is preserved
        start_filter = search_filters.created_at[0][0]
        end_filter = search_filters.created_at[0][1]

        assert start_filter.date.microsecond == 789123
        assert end_filter.date.microsecond == 789124

    def test_extreme_date_ranges(self):
        """Test filtering with extreme date ranges."""
        very_old = datetime(1970, 1, 1, tzinfo=timezone.utc)
        future_date = datetime(2100, 12, 31, tzinfo=timezone.utc)

        query = SearchQuery(
            query="test query",
            created_at_start=very_old,
            created_at_end=future_date
        )

        search_filters = _build_search_filters_from_query(query)

        assert search_filters is not None
        assert search_filters.created_at is not None
        assert search_filters.created_at[0][0].date == very_old
        assert search_filters.created_at[0][1].date == future_date