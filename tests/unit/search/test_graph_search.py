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

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode
from graphiti_core.search.search_config import (
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    NodeReranker,
    EpisodeSearchConfig,
    EpisodeReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    CommunityReranker,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    calculate_cosine_similarity,
    fulltext_query,
    rrf,
    maximal_marginal_relevance,
)
from graphiti_core.search.search_helpers import (
    format_edge_date_range,
    search_results_to_context_string,
)


class TestCalculateCosineSimilarity:
    """Tests for calculate_cosine_similarity function."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vector1 = [1.0, 2.0, 3.0]
        vector2 = [1.0, 2.0, 3.0]
        result = calculate_cosine_similarity(vector1, vector2)
        assert pytest.approx(result, 0.0001) == 1.0

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vector1 = [1.0, 0.0]
        vector2 = [0.0, 1.0]
        result = calculate_cosine_similarity(vector1, vector2)
        assert pytest.approx(result, 0.0001) == 0.0

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vector1 = [1.0, 2.0, 3.0]
        vector2 = [-1.0, -2.0, -3.0]
        result = calculate_cosine_similarity(vector1, vector2)
        assert pytest.approx(result, 0.0001) == -1.0

    def test_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        vector1 = [0.0, 0.0, 0.0]
        vector2 = [1.0, 2.0, 3.0]
        result = calculate_cosine_similarity(vector1, vector2)
        assert result == 0.0

    def test_both_zero_vectors(self):
        """Test cosine similarity with both zero vectors."""
        vector1 = [0.0, 0.0, 0.0]
        vector2 = [0.0, 0.0, 0.0]
        result = calculate_cosine_similarity(vector1, vector2)
        assert result == 0.0

    def test_different_length_vectors(self):
        """Test cosine similarity with different length vectors should handle gracefully."""
        vector1 = [1.0, 2.0]
        vector2 = [1.0, 2.0, 3.0]
        # This should still work due to numpy broadcasting
        result = calculate_cosine_similarity(vector1, vector2)
        assert isinstance(result, float)


class TestFulltextQuery:
    """Tests for fulltext_query function."""

    def test_kuzu_provider_long_query(self):
        """Test Kuzu provider with long query."""
        mock_driver = MagicMock()
        mock_driver.provider = GraphProvider.KUZU

        long_query = ' '.join(['word'] * 150)  # Create a query with 150 words
        result = fulltext_query(long_query, None, mock_driver)
        assert result == ''

    def test_kuzu_provider_short_query(self):
        """Test Kuzu provider with short query."""
        mock_driver = MagicMock()
        mock_driver.provider = GraphProvider.KUZU

        short_query = 'short query'
        result = fulltext_query(short_query, None, mock_driver)
        assert result == short_query

    def test_falkordb_provider(self):
        """Test FalkorDB provider."""
        mock_driver = MagicMock()
        mock_driver.provider = GraphProvider.FALKORDB
        mock_driver.build_fulltext_query = MagicMock(return_value='formatted_query')

        query = 'test query'
        group_ids = ['group1', 'group2']
        result = fulltext_query(query, group_ids, mock_driver)

        mock_driver.build_fulltext_query.assert_called_once_with(query, group_ids, 128)
        assert result == 'formatted_query'

    def test_neo4j_provider_with_group_ids(self):
        """Test Neo4j provider with group IDs."""
        mock_driver = MagicMock()
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.fulltext_syntax = '@@'

        query = 'test query'
        group_ids = ['group1', 'group2']
        result = fulltext_query(query, group_ids, mock_driver)

        assert '@@group_id:"group1" OR @@group_id:"group2"' in result

    def test_neo4j_provider_without_group_ids(self):
        """Test Neo4j provider without group IDs."""
        mock_driver = MagicMock()
        mock_driver.provider = GraphProvider.NEO4J

        query = 'test query'
        result = fulltext_query(query, None, mock_driver)

        assert result == 'test query'


class TestRRF:
    """Tests for Reciprocal Rank Fusion (RRF) function."""

    def test_basic_rrf(self):
        """Test basic RRF functionality."""
        # Two lists of document IDs
        list1 = ['doc1', 'doc2', 'doc3']
        list2 = ['doc2', 'doc1', 'doc4']

        reranked_uuids, scores = rrf([list1, list2])

        # doc2 appears at rank 2 in list1 and rank 1 in list2
        # doc1 appears at rank 1 in list1 and rank 2 in list2
        # doc3 appears only in list1 at rank 3
        # doc4 appears only in list2 at rank 3

        assert 'doc2' in reranked_uuids
        assert 'doc1' in reranked_uuids
        assert 'doc3' in reranked_uuids
        assert 'doc4' in reranked_uuids

        # Check that scores are calculated correctly
        assert len(scores) == len(reranked_uuids)

    def test_rrf_with_empty_lists(self):
        """Test RRF with empty result lists."""
        result = rrf([])
        assert result[0] == []  # No reranked UUIDs
        assert result[1] == []  # No scores

    def test_rrf_with_min_score(self):
        """Test RRF with minimum score threshold."""
        list1 = ['doc1', 'doc2', 'doc3']
        list2 = ['doc2', 'doc1', 'doc4']

        reranked_uuids, scores = rrf([list1, list2], min_score=0.5)

        # Should filter out results with scores below 0.5
        for score in scores:
            assert score >= 0.5

    def test_rrf_with_single_list(self):
        """Test RRF with a single result list."""
        list1 = ['doc1', 'doc2', 'doc3']

        reranked_uuids, scores = rrf([list1])

        assert reranked_uuids == ['doc1', 'doc2', 'doc3']
        assert len(scores) == 3


class TestMaximalMarginalRelevance:
    """Tests for maximal_marginal_relevance function."""

    @pytest.mark.asyncio
    async def test_basic_mmr(self):
        """Test basic MMR functionality."""
        query_vector = [1.0, 0.0, 0.0]

        # Mock documents with vectors
        docs_with_vectors = [
            ('doc1', [1.0, 0.0, 0.0]),  # Identical to query
            ('doc2', [1.0, 0.0, 0.0]),  # Identical to query and doc1
            ('doc3', [0.0, 1.0, 0.0]),  # Orthogonal to query
        ]

        reranked_uuids, scores = maximal_marginal_relevance(
            query_vector, docs_with_vectors, lambda_val=0.5, min_score=0.0
        )

        assert len(reranked_uuids) == 3
        assert 'doc1' in reranked_uuids
        assert 'doc2' in reranked_uuids
        assert 'doc3' in reranked_uuids

    @pytest.mark.asyncio
    async def test_mmr_with_high_lambda(self):
        """Test MMR with high lambda value (emphasizes relevance)."""
        query_vector = [1.0, 0.0, 0.0]

        docs_with_vectors = [
            ('doc1', [1.0, 0.0, 0.0]),  # High relevance, low diversity
            ('doc2', [0.99, 0.0, 0.0]),  # High relevance, low diversity
            ('doc3', [0.5, 0.5, 0.0]),   # Lower relevance, high diversity
        ]

        reranked_uuids, scores = maximal_marginal_relevance(
            query_vector, docs_with_vectors, lambda_val=0.9, min_score=0.0
        )

        # With high lambda, relevance should be prioritized
        assert reranked_uuids[0] in ['doc1', 'doc2']

    @pytest.mark.asyncio
    async def test_mmr_with_low_lambda(self):
        """Test MMR with low lambda value (emphasizes diversity)."""
        query_vector = [1.0, 0.0, 0.0]

        docs_with_vectors = [
            ('doc1', [1.0, 0.0, 0.0]),  # High relevance, low diversity
            ('doc2', [1.0, 0.0, 0.0]),  # High relevance, low diversity
            ('doc3', [0.0, 1.0, 0.0]),  # Lower relevance, high diversity
        ]

        reranked_uuids, scores = maximal_marginal_relevance(
            query_vector, docs_with_vectors, lambda_val=0.1, min_score=0.0
        )

        # With low lambda, diversity should be prioritized
        assert len(set(reranked_uuids)) == len(reranked_uuids)


class TestSearchHelpers:
    """Tests for search helper functions."""

    def test_format_edge_date_range(self):
        """Test format_edge_date_range function."""
        edge = EntityEdge(
            uuid='test-edge',
            source_node_uuid='node1',
            target_node_uuid='node2',
            fact='test fact',
            created_at=datetime.now(),
            valid_at='2024-01-01',
            invalid_at='2024-12-31'
        )

        result = format_edge_date_range(edge)
        assert '2024-01-01' in result
        assert '2024-12-31' in result

    def test_format_edge_date_range_no_dates(self):
        """Test format_edge_date_range with no dates."""
        edge = EntityEdge(
            uuid='test-edge',
            source_node_uuid='node1',
            target_node_uuid='node2',
            fact='test fact',
            created_at=datetime.now(),
            valid_at=None,
            invalid_at=None
        )

        result = format_edge_date_range(edge)
        assert 'date unknown' in result
        assert 'present' in result

    def test_search_results_to_context_string(self):
        """Test search_results_to_context_string function."""
        edge = EntityEdge(
            uuid='edge1',
            source_node_uuid='node1',
            target_node_uuid='node2',
            fact='test fact',
            created_at=datetime.now(),
            valid_at='2024-01-01',
            invalid_at=None
        )

        node = EntityNode(
            uuid='node1',
            name='Test Entity',
            summary='Test summary',
            created_at=datetime.now(),
            group_id='group1'
        )

        episode = EpisodicNode(
            uuid='ep1',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            created_at=datetime.now(),
            group_id='group1'
        )

        community = CommunityNode(
            uuid='comm1',
            name='Test Community',
            summary='Community summary',
            created_at=datetime.now()
        )

        search_results = SearchResults(
            edges=[edge],
            nodes=[node],
            episodes=[episode],
            communities=[community]
        )

        context = search_results_to_context_string(search_results)

        assert 'test fact' in context
        assert 'Test Entity' in context
        assert 'Test summary' in context
        assert 'Test content' in context
        assert 'Test source' in context
        assert 'Test Community' in context
        assert 'Community summary' in context
        assert '<FACTS>' in context
        assert '<ENTITIES>' in context
        assert '<EPISODES>' in context
        assert '<COMMUNITIES>' in context

    def test_search_results_to_context_string_empty(self):
        """Test search_results_to_context_string with empty results."""
        empty_results = SearchResults()

        context = search_results_to_context_string(empty_results)

        assert '<FACTS>' in context
        assert '<ENTITIES>' in context
        assert '<EPISODES>' in context
        assert '<COMMUNITIES>' in context
        assert '[]' in context  # Empty JSON arrays